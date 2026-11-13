//! The sim worker as a message-driven server core.
//!
//! `run` consumes `ClientMsg`s (tagged with a client id — the local
//! transport uses a single id, the network server multiplexes several)
//! and emits `ServerEvent`s. It owns the network, script Env and
//! simulator; nothing blocks on a client.
//!
//! Activity frames are rate-limited: publishes while running wait out a
//! minimum wall-clock interval so remote transports stay inside a few
//! MB/s for million-neuron nets. Data is never lost — the publish folds
//! the whole interval (counts window, drained v samples, trace deltas).

use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use net_format::Network;
use proto::{
    Activity, Backend, ClientMsg, Config, Geo, GeoDelta, Grid, Input, NetEdit, NeuronRow,
    ServerEvent, ServerMsg, SynRow,
};

use crate::script::{Env, NetHandle, ProgressFn};
use crate::{BackendKind, Sim, SimConfig};

const GEO_EDGES_PER_NEURON: usize = 4;
const COUNTS_WINDOW: usize = 4096;

/// Spike-trace decay constant for the plane view (sim time, ms).
/// Deliberately slower than the readout trace's 2 ms — the plane is a
/// visual summary, so texels should glow for a few hundred steps.
/// Re-exported — the constant lives in `proto` (it's wire semantics:
/// clients decay with the same τ).
pub use proto::PLANE_TAU_MS;

/// Minimum wall-clock interval between Activity publishes while running —
/// the bandwidth rate-limit for remote clients.
pub const MIN_ACTIVITY_MS: u64 = 33;

/// Console lines truncate at this length (a `neurons_of` on a big grid
/// would otherwise flood the wire).
const MAX_CONSOLE: usize = 400;

// ---- Stage 23: worker-owned network undo ----------------------------------
//
// One undoable step is a full (net, rng, net_gen) snapshot taken just before
// the change. Undo swaps the live state for the snapshot; the displaced live
// state becomes the redo entry. A snapshot is the only inverse that is
// byte-exact for every edit kind — including rng-driven builders
// (`create_connections`, `set_exc_inh`, `InsertSubnetJson`) whose results a
// typed inverse can't reconstruct — and restoring the rng means random edits
// after an undo roll the same dice as the first time.
//
// The stacks are SHARED across clients: the net is shared, so "revert the
// last change *I* made" would silently revert whatever landed on top of it
// from someone else. One ordered history is the honest semantics — Ctrl+Z
// pops the last net change whoever made it — and every client sees the same
// menu label through `Status.undo`/`Status.redo`.

/// An undoable step: the state BEFORE the change plus the menu label.
struct UndoEntry {
    label: String,
    snap: UndoSnap,
}

/// Everything a restore puts back: the net itself, the builder rng (so a
/// redone `create_connections` doesn't re-roll), and the net generation —
/// restores that cross a generation bump `net_gen` again so clients drop
/// id-keyed state, same-generation restores keep it.
struct UndoSnap {
    net: Network,
    rng: net_format::Rng,
    gen: u64,
}

/// History bounds — steps OR approximate bytes, whichever hits first.
/// A step is a deep net clone, so the byte bound is what matters on big
/// hosted nets; the step bound keeps small nets snappy.
const UNDO_MAX_STEPS: usize = 64;
const UNDO_MAX_BYTES: usize = 256 << 20;

/// Rough heap estimate of a `Network` clone — vector capacities plus
/// string payloads. Good enough for a memory bound; the point is that a
/// 1M-neuron history can't grow unboundedly.
fn est_net_bytes(n: &Network) -> usize {
    use std::mem::size_of as sz;
    let mut b = sz::<Network>()
        + n.neurons.capacity() * sz::<net_format::Neuron>()
        + n.synapses.capacity() * sz::<net_format::Synapse>()
        + n.types.capacity() * sz::<net_format::NeuronType>()
        + n.groups.capacity() * sz::<net_format::Group>()
        + n.inputs.capacity() * sz::<net_format::Input>()
        + n.samples.capacity() * sz::<net_format::Samples>()
        + n.readouts.capacity() * sz::<net_format::Readout>()
        + n.scripts.capacity() * sz::<(String, String)>();
    for x in &n.neurons {
        b += x.name.capacity();
        if let Some(p) = &x.params {
            b += p.len() * 4;
        }
    }
    for t in &n.types {
        b += t.name.capacity() + t.model.capacity() + t.params.capacity() * 4;
    }
    for g in &n.groups {
        b += g.name.capacity() + g.members.capacity() * 4;
    }
    for i in &n.inputs {
        b += i.name.capacity() + i.targets.capacity() * 8;
    }
    for s in &n.samples {
        b += s.name.capacity() + (s.times.capacity() + s.values.capacity()) * 4;
    }
    for r in &n.readouts {
        b += r.name.capacity() + r.units.capacity() * 4 + r.weights.capacity() * 4;
    }
    for (a, c) in &n.scripts {
        b += a.capacity() + c.capacity();
    }
    b
}

fn stack_bytes(stack: &std::collections::VecDeque<UndoEntry>) -> usize {
    stack.iter().map(|e| est_net_bytes(&e.snap.net)).sum()
}

/// Push one step, evicting the oldest entries past the bounds. A step
/// bigger than the byte cap is still kept — one net clone is the price
/// of one undo point — but nothing older survives with it.
fn push_step(stack: &mut std::collections::VecDeque<UndoEntry>, e: UndoEntry) {
    push_step_bounded(stack, e, UNDO_MAX_STEPS, UNDO_MAX_BYTES);
}

fn push_step_bounded(
    stack: &mut std::collections::VecDeque<UndoEntry>,
    e: UndoEntry,
    max_steps: usize,
    max_bytes: usize,
) {
    stack.push_back(e);
    while stack.len() > max_steps || (stack.len() > 1 && stack_bytes(stack) > max_bytes) {
        stack.pop_front();
    }
}

fn undo_snap(env: &Env) -> UndoSnap {
    UndoSnap {
        net: env.net.borrow().clone(),
        rng: env.rng.borrow().clone(),
        gen: env.net_gen.get(),
    }
}

/// Swap the live net for a snapshot — an edit like any other: the sim is
/// invalidated and `net_version` raised, which the republish block at the
/// top of the loop turns into a full `Geo` + `GridTraces` broadcast.
/// The FIRST invalidate runs against the OLD net so `DriveState::capture`
/// keys the dropped sim's input drive by the names it actually had —
/// after the swap the same indexes could be different names.
fn undo_restore(env: &Env, s: UndoSnap) {
    env.invalidate_sim();
    *env.net.borrow_mut() = s.net;
    *env.rng.borrow_mut() = s.rng;
    if env.net_gen.get() != s.gen {
        env.net_gen.set(env.net_gen.get() + 1);
    }
    bump(env);
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// The Edit-menu label for a `NetEdit` — "move 12 neurons", "delete 3
/// neurons". Kept server-side: the label names what the worker did.
fn edit_label(e: &NetEdit) -> String {
    use NetEdit::*;
    match e {
        AddNeuron { .. } => "add neuron".into(),
        DeleteNeurons { ids } => format!("delete {} neuron{}", ids.len(), plural(ids.len())),
        Connect { from, to, .. } => format!("connect {from}->{to}"),
        Disconnect { from, to } => format!("disconnect {from}->{to}"),
        MoveNeurons { items } => {
            format!("move {} neuron{}", items.len(), plural(items.len()))
        }
        SetNeuronName { .. } => "rename neuron".into(),
        SetNeuronPos { .. } => "move neuron".into(),
        SetNeuronKind { id, .. } => format!("kind of neuron {id}"),
        InsertSubnetJson { .. } => "paste subnet".into(),
        CreateGrid2d { group, .. } => format!("grid \"{group}\""),
        CreateGrid3d { group, .. } => format!("grid \"{group}\""),
        AddInput { name, .. } => format!("input \"{name}\""),
        AddSamples { name } => format!("samples \"{name}\""),
        AddReadout { name, .. } => format!("readout \"{name}\""),
        MoveInputs { items } => format!("move {} input{}", items.len(), plural(items.len())),
        AddToGroup { name, ids } => {
            format!("add {} to \"{name}\"", ids.len())
        }
        AddInputTarget { name, target, .. } => format!("input \"{name}\" target {target}"),
        CreateConnections { group, .. } => format!("wire \"{group}\""),
        SetSynapseWeights { group, .. } => format!("weights \"{group}\""),
        SetSynTauDist { src, dst, .. } => format!("tau \"{src}\"->\"{dst}\""),
        SetExcInh { group, .. } => format!("exc/inh \"{group}\""),
        ScriptAdd { name, .. } => format!("script \"{name}\""),
        UpsertType { name, .. } => format!("type \"{name}\""),
        SetNeuronType { ids, .. } => {
            format!("type of {} neuron{}", ids.len(), plural(ids.len()))
        }
        SetNeuronParams { ids, .. } => {
            format!("params of {} neuron{}", ids.len(), plural(ids.len()))
        }
        SetSynapse { from, to, .. } => format!("synapse {from}->{to}"),
    }
}

/// One-line label for a script eval — the eval is ONE undo step however
/// many edits it makes.
fn eval_label(code: &str) -> String {
    let one: String = code.split_whitespace().collect::<Vec<_>>().join(" ");
    let short: String = one.chars().take(48).collect();
    format!(
        "eval {short}{}",
        if one.chars().count() > 48 { "…" } else { "" }
    )
}

/// Shared emit handle — the print callback, the mid-run progress hook and
/// the command loop all produce events through the same sink. Borrows are
/// never nested (no emit happens inside another emit).
pub type Emit = Rc<std::cell::RefCell<dyn FnMut(ServerEvent)>>;

/// Everything `run` needs that isn't in `ClientMsg`.
pub struct ServeOpts {
    pub backend: BackendKind,
    pub seed: u64,
    pub cfg: SimConfig,
    /// Sandbox for every file path scripts and load commands touch;
    /// `None` = local mode (paths used verbatim).
    pub data_dir: Option<PathBuf>,
    /// `UploadDataset` refuses payloads over this many bytes (the
    /// server's `--max-upload-mb`; local mode uses a large default).
    pub max_upload_bytes: usize,
    /// Per-run storage cap on `data_dir` (multi-user mode's per-user
    /// quota; 0 = unlimited — local mode). Checked on every protocol
    /// write: `WriteFile`, `UploadDataset`, `UploadStart`/`UploadChunk`.
    pub quota_bytes: u64,
    /// Shared stop flag — UI Stop button / server shutdown. Cleared by
    /// the next `Eval`, so a queued eval can override a pending stop.
    pub abort: std::sync::Arc<AtomicBool>,
    /// Hard shutdown — set once (by `UserRun::stop`), never cleared.
    /// Once set the worker terminates whatever it is doing (eval hooks,
    /// run chunks, builders poll it alongside `abort`) and exits its
    /// command loop without draining further messages.
    pub shutdown: std::sync::Arc<AtomicBool>,
    /// Sim-size ceiling enforced before every build (server flags in
    /// hosted mode; `SimLimits::default()` = unlimited for local).
    pub sim_limits: crate::SimLimits,
}

/// Recursive byte count of `dir` (0 on missing/inaccessible) — used
/// for quota checks and the per-user usage bar.
pub fn dir_size(dir: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if let Ok(md) = std::fs::symlink_metadata(&p) {
                if md.is_dir() {
                    total = total.saturating_add(dir_size(&p));
                } else {
                    total = total.saturating_add(md.len());
                }
            }
        }
    }
    total
}

fn emit_b(emit: &Emit, m: impl Into<ServerMsg>) {
    (emit.borrow_mut())(ServerEvent::broadcast(m.into()));
}
fn emit_to(emit: &Emit, client: u64, m: ServerMsg) {
    (emit.borrow_mut())(ServerEvent::to(client, m));
}
fn emit_ev(emit: &Emit, ev: ServerEvent) {
    (emit.borrow_mut())(ev);
}

/// Convenience: console lines from plain strings.
fn con(s: impl AsRef<str>) -> ServerMsg {
    ServerMsg::Console(trunc(s.as_ref()))
}

fn trunc(s: &str) -> String {
    if s.len() > MAX_CONSOLE {
        // slice only on a char boundary — MAX_CONSOLE can land inside a
        // multi-byte char, and a direct s[..MAX_CONSOLE] would panic.
        let mut end = MAX_CONSOLE;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{} … ({} chars truncated)", &s[..end], s.len() - end)
    } else {
        s.to_string()
    }
}

/// Run the worker loop until the channel closes or `ClientMsg::Bye`
/// arrives. `emit` receives every `ServerEvent`; it must return quickly
/// (it's on the sim's pacing).
pub fn run(net: Network, opts: ServeOpts, rx: mpsc::Receiver<(u64, ClientMsg)>, emit: Emit) {
    let ServeOpts {
        backend,
        seed,
        cfg,
        data_dir,
        max_upload_bytes,
        quota_bytes,
        abort,
        shutdown,
        sim_limits,
    } = opts;
    // `SetQuota` (admin push) adjusts this live — hence `mut`.
    let mut quota_bytes = quota_bytes;
    // Client ids whose Hello carried a wrong `PROTO_VERSION`. Broadcast
    // events get these stamped onto `except` (below) so a v2 frame can
    // never reach a bundle that would mis-decode it; their later
    // ClientMsgs are dropped in the dispatch loop.
    let refused_clients: Rc<std::cell::RefCell<HashSet<u64>>> = Default::default();
    let emit: Emit = {
        let inner = emit;
        let rf = refused_clients.clone();
        Rc::new(std::cell::RefCell::new(move |mut ev: ServerEvent| {
            if ev.to.is_none() {
                ev.except = rf.borrow().iter().copied().collect();
            }
            (inner.borrow_mut())(ev);
        }))
    };
    let mut env = Env::with_print(net, backend, seed, {
        let e = emit.clone();
        Rc::new(move |s: &str| emit_b(&e, con(s)))
    });
    *env.data_dir.borrow_mut() = data_dir.clone();
    *env.sim_limits.borrow_mut() = sim_limits;
    *env.quota_bytes.borrow_mut() = quota_bytes;
    // Confine `import` too — Rhai's default module resolver reads from
    // the process cwd/absolute paths unchecked.
    if let Some(d) = &data_dir {
        env.engine
            .set_module_resolver(crate::script::ConfinedResolver::new(d.clone()));
    }
    // The worker's whole job is publishing snapshots; raster, particles and
    // the activity counts all need the rolling spike bitmap, so force it on
    // regardless of the caller's cfg.
    *env.cfg.borrow_mut() = SimConfig {
        record_spikes: true,
        spike_record_steps: cfg.spike_record_steps.max(COUNTS_WINDOW),
        seed,
        ..cfg
    };

    let pub_rc = Rc::new(std::cell::RefCell::new(Pub::default()));

    // Progress hook: publish between run chunks inside script `sim.run`s;
    // returning false aborts the rest of the run (Stop pressed).
    {
        let (abp, pr, ep) = (abort.clone(), pub_rc.clone(), emit.clone());
        let shut = shutdown.clone();
        let (neth, nver) = (env.net.clone(), env.net_version.clone());
        let cb: ProgressFn = Rc::new(std::cell::RefCell::new(move |sim: &mut Sim| -> bool {
            let mut p = pr.borrow_mut();
            publish_sim(&neth, nver.get(), sim, &mut p, &ep);
            !(abp.load(Ordering::Relaxed) || shut.load(Ordering::Relaxed))
        }));
        env.set_progress(200, cb);
    }

    // Stop also lands inside slow native builders (create_connections,
    // grid builders) — they poll this every few thousand rows.
    {
        let (ab, shut) = (abort.clone(), shutdown.clone());
        env.set_cancel(Rc::new(move || {
            ab.load(Ordering::Relaxed) || shut.load(Ordering::Relaxed)
        }));
    }

    // Rhai op-level abort: the chunked `sim.run` hook only fires between
    // chunks, so a pure-script loop (`while true {}`) would never see the
    // abort flag. on_progress runs inside the eval loop — a `Some` return
    // terminates the script with ErrorTerminated.
    {
        let (ab, shut) = (abort.clone(), shutdown.clone());
        env.engine.on_progress(move |ops| {
            if ops % 64 == 0 && (ab.load(Ordering::Relaxed) || shut.load(Ordering::Relaxed)) {
                Some(rhai::Dynamic::UNIT)
            } else {
                None
            }
        });
    }

    let mut running = false;
    let mut speed = 1.0f32;
    let mut published_geo_version = u64::MAX;
    let mut chunk = 200usize;
    let mut quit = false;
    // Open chunked uploads: path → (bytes so far, declared size). Only
    // `UploadStart` opens one; `UploadChunk` without an entry is an
    // error, and the running count enforces the declared size — so the
    // per-file cap can't be dodged by lying in `size`.
    let mut open_uploads: std::collections::HashMap<String, (u64, u64)> =
        std::collections::HashMap::new();
    // Last-emitted Status — changes only.
    let mut last_status: Option<ServerMsg> = None;
    let mut status_busy = false;
    // Stage 23: the shared network history (see the file-top notes).
    // `undo`'s top is the most recent change; `redo` is cleared by every
    // new step and refilled by undos.
    let mut undo: std::collections::VecDeque<UndoEntry> = Default::default();
    let mut redo: std::collections::VecDeque<UndoEntry> = Default::default();

    macro_rules! status {
        () => {{
            let msg = ServerMsg::Status {
                running,
                busy: status_busy,
                sim_built: pub_rc.borrow().sim_built,
                backend: match env.backend.get() {
                    BackendKind::Cpu => Backend::Cpu,
                    BackendKind::Cuda => Backend::Cuda,
                },
                cfg: cfg_wire(&env.cfg.borrow()),
                notice: None,
                undo: undo.back().map(|e| e.label.clone()),
                redo: redo.back().map(|e| e.label.clone()),
            };
            if last_status.as_ref() != Some(&msg) {
                last_status = Some(msg.clone());
                emit_ev(&emit, ServerEvent::broadcast(msg));
            }
        }};
    }

    while !quit {
        // Hard shutdown outranks queued work: nothing clears the flag,
        // so a queued Eval can't clear `abort` and stall `stop()` for
        // minutes. (Bye also becomes unnecessary — the loop just ends.)
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        // ---- geometry republish on version bump ----
        let gv = env.net_version.get();
        if gv != published_geo_version {
            let new_gen = env.net_gen.get();
            if new_gen != pub_rc.borrow().net_gen {
                // Net replaced: sim stats (t, spikes) describe the old
                // net — publish a clean slate until the next run.
                {
                    let mut p = pub_rc.borrow_mut();
                    p.reset_dyn(new_gen);
                    p.seq += 1;
                    let seq = p.seq;
                    emit_ev(
                        &emit,
                        ServerEvent::broadcast(ServerMsg::Activity(Box::new(Activity {
                            seq,
                            ..Default::default()
                        }))),
                    );
                }
            }
            let geo = geo_snap(&env);
            // Fresh planes sized to the new geometry — the sim was
            // invalidated so every texel is zero. Resizing the accumulators
            // here keeps deltas consistent.
            let (seq, grids) = {
                let mut p = pub_rc.borrow_mut();
                (p.seq, p.trace_snapshot(&geo))
            };
            emit_ev(&emit, ServerEvent::broadcast(ServerMsg::Geo(Box::new(geo))));
            emit_ev(
                &emit,
                ServerEvent::broadcast(ServerMsg::GridTraces {
                    base_seq: seq,
                    grids,
                }),
            );
            published_geo_version = gv;
        }
        status!();

        // ---- commands ----
        let cmd = if running {
            rx.try_recv().ok()
        } else {
            rx.recv_timeout(Duration::from_millis(30)).ok()
        };
        let mut pending = Vec::new();
        if let Some(c) = cmd {
            pending.push(c);
        }
        while let Ok(c) = rx.try_recv() {
            pending.push(c);
        }
        for (client, c) in pending {
            // Once shutdown is set the worker drops everything queued
            // behind Bye — an Eval here would clear `abort` and stall a
            // `stop()` that is trying to join this thread.
            if shutdown.load(Ordering::Relaxed) && !matches!(c, ClientMsg::Bye) {
                continue;
            }
            // A client whose Hello carried the wrong PROTO_VERSION is
            // refused: it already got an Error explaining the reload, and
            // must never be sent v2 frames it would mis-decode (stale
            // deployed web bundles have hit exactly this). Its later
            // messages are dropped until it reconnects and re-hellos.
            if refused_clients.borrow().contains(&client) {
                // A refused client can still hang up — Bye must reach
                // the loop or a stale bundle's disconnect would wedge
                // the worker in `join`.
                if matches!(c, ClientMsg::Bye) {
                    quit = true;
                }
                continue;
            }
            // One bad message must not kill the worker: a panic anywhere
            // in dispatch (edit apply, file op, snapshot) is reported and
            // the loop continues with the next command.
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match c {
                    ClientMsg::Hello {
                        proto: client_proto,
                    } if client_proto != proto::PROTO_VERSION => {
                        emit_to(
                            &emit,
                            client,
                            ServerMsg::Error(format!(
                                "protocol mismatch: client speaks {client_proto}, server {} — reload the page",
                                proto::PROTO_VERSION
                            )),
                        );
                        refused_clients.borrow_mut().insert(client);
                    }
                    ClientMsg::Hello { .. } | ClientMsg::Subscribe => {
                        let geo = geo_snap(&env);
                        let (seq, grids) = {
                            let mut p = pub_rc.borrow_mut();
                            (p.seq, p.trace_snapshot(&geo))
                        };
                        emit_to(
                            &emit,
                            client,
                            ServerMsg::Hello {
                                proto: proto::PROTO_VERSION,
                                net_version: env.net_version.get(),
                                net_gen: env.net_gen.get(),
                                models: model_descs(),
                            },
                        );
                        emit_to(&emit, client, ServerMsg::Geo(Box::new(geo)));
                        emit_to(
                            &emit,
                            client,
                            ServerMsg::GridTraces {
                                base_seq: seq,
                                grids,
                            },
                        );
                        // force a status re-send to this client
                        let msg = ServerMsg::Status {
                            running,
                            busy: status_busy,
                            sim_built: pub_rc.borrow().sim_built,
                            backend: match env.backend.get() {
                                BackendKind::Cpu => Backend::Cpu,
                                BackendKind::Cuda => Backend::Cuda,
                            },
                            cfg: cfg_wire(&env.cfg.borrow()),
                            notice: None,
                            undo: undo.back().map(|e| e.label.clone()),
                            redo: redo.back().map(|e| e.label.clone()),
                        };
                        emit_to(&emit, client, msg);
                        published_geo_version = env.net_version.get();
                    }
                    ClientMsg::Bye => quit = true,
                    ClientMsg::Abort => {
                        abort.store(true, Ordering::Relaxed);
                        running = false;
                    }
                    ClientMsg::Eval { code } => {
                        // Stage 23: ONE undo step per eval — snapshot before,
                        // record after if the net (or its generation) moved.
                        let pre = undo_snap(&env);
                        let pre_ver = env.net_version.get();
                        let pre_gen = env.net_gen.get();
                        abort.store(false, Ordering::Relaxed);
                        status_busy = true;
                        status!();
                        emit_b(&emit, con(format!("> {code}")));
                        // catch_unwind so a panic inside a script fn (or a CUDA
                        // call) can't kill the worker thread silently.
                        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            env.eval_str(&code)
                        }));
                        match res {
                            Ok(Ok(d)) if !d.is_unit() => emit_b(&emit, con(format!("= {d}"))),
                            Ok(Ok(_)) => {}
                            Ok(Err(e)) => emit_b(&emit, con(format!("! {e}"))),
                            Err(_) => emit_b(&emit, con("! script panicked (worker survived)")),
                        }
                        if env.net_version.get() != pre_ver || env.net_gen.get() != pre_gen {
                            push_step(
                                &mut undo,
                                UndoEntry {
                                    label: eval_label(&code),
                                    snap: pre,
                                },
                            );
                            redo.clear();
                        }
                        status_busy = false;
                        status!();
                        // sim.* calls inside the eval (set_input, set_noise…)
                        // leave state the UI shows — push one snapshot, but
                        // only when a sim exists (don't build one for a
                        // read-only `net.*` eval). The rate limiter may have
                        // skipped the run's tail — force the window closed.
                        if env.existing_sim().is_some() {
                            let mut p = pub_rc.borrow_mut();
                            publish_env(&env, &mut p, &emit);
                        }
                    }
                    ClientMsg::LoadNet { path } => {
                        let pre = undo_snap(&env);
                        match resolve(&env, &path)
                            .and_then(|p| net_format::io::load(&p).map_err(|e| e.to_string()))
                        {
                            Ok(n) => {
                                // M3: the file's sim settings become live.
                                crate::meta_to_cfg(&n.meta, &mut env.cfg.borrow_mut());
                                env.invalidate_sim();
                                *env.net.borrow_mut() = n;
                                env.net_gen.set(env.net_gen.get() + 1);
                                bump(&env);
                                push_step(
                                    &mut undo,
                                    UndoEntry {
                                        label: format!("load {path}"),
                                        snap: pre,
                                    },
                                );
                                redo.clear();
                                // Q7: typed confirmation — clients only
                                // adopt net_path/recent on this frame.
                                emit_to(
                                    &emit,
                                    client,
                                    ServerMsg::NetLoaded {
                                        path: path.clone(),
                                        version: env.net_version.get(),
                                    },
                                );
                                emit_b(&emit, con(format!("loaded {path}")));
                            }
                            Err(e) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!("load {path}: {e}")),
                            ),
                        }
                    }
                    // Q6: typed save — the ack carries the version that
                    // hit disk so the client marks clean on truth only.
                    ClientMsg::SaveNet { path } => {
                        let done = resolve(&env, &path).and_then(|p| {
                            let mut n = env.net.borrow_mut();
                            crate::cfg_to_meta(&env.cfg.borrow(), &mut n.meta);
                            let bytes = net_format::io::to_bytes(&n).map_err(|e| e.to_string())?;
                            crate::script::quota_fits(
                                &env.data_dir.borrow(),
                                &p,
                                bytes.len() as u64,
                                *env.quota_bytes.borrow(),
                            )?;
                            std::fs::write(&p, &bytes).map_err(|e| format!("{p}: {e}"))
                        });
                        match done {
                            Ok(()) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Saved {
                                    path: path.clone(),
                                    version: env.net_version.get(),
                                },
                            ),
                            Err(e) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!("save {path}: {e}")),
                            ),
                        }
                    }
                    ClientMsg::NewNet => {
                        let pre = undo_snap(&env);
                        env.invalidate_sim();
                        *env.net.borrow_mut() = Network::new();
                        env.net_gen.set(env.net_gen.get() + 1);
                        bump(&env);
                        push_step(
                            &mut undo,
                            UndoEntry {
                                label: "new net".into(),
                                snap: pre,
                            },
                        );
                        redo.clear();
                    }
                    ClientMsg::AutoRun(b) => {
                        // Only a fresh start clears abort — stopping must not
                        // erase a flag an in-flight eval is about to see.
                        if b {
                            abort.store(false, Ordering::Relaxed);
                        }
                        running = b;
                    }
                    ClientMsg::SetSpeed(s) => speed = s.max(0.01),
                    ClientMsg::RunMs(ms) => {
                        abort.store(false, Ordering::Relaxed);
                        if let Ok(h) = env.sim_handle() {
                            let dt = env.cfg.borrow().dt_ms;
                            let steps = (ms / dt).round().max(0.0) as usize;
                            run_published(&env, &h, steps, &pub_rc, &abort, &emit);
                        }
                    }
                    ClientMsg::Reset => {
                        if let Ok(h) = env.sim_handle() {
                            h.borrow_mut().reset_state();
                        }
                        pub_rc.borrow_mut().prev_step = 0;
                        let mut p = pub_rc.borrow_mut();
                        publish_env(&env, &mut p, &emit);
                    }
                    ClientMsg::Watch(ids) => {
                        // Apply now too — `p.watch` is only pushed into
                        // the sim at publish time, which is AFTER the
                        // steps ran; a Watch→RunMs pair would record
                        // nothing (state slots are captured per-step).
                        if let Some(h) = env.existing_sim() {
                            h.borrow_mut().set_v_watch(&ids);
                        }
                        pub_rc.borrow_mut().watch = ids;
                    }
                    ClientMsg::Edit { base, edit } => {
                        let cur = env.net_version.get();
                        if base != cur {
                            emit_to(
                            &emit,
                            client,
                            ServerMsg::Error(format!(
                                "edit against stale net version {base} (worker is at {cur}) — resyncing"
                            )),
                        );
                            let geo = geo_snap(&env);
                            let (seq, grids) = {
                                let mut p = pub_rc.borrow_mut();
                                (p.seq, p.trace_snapshot(&geo))
                            };
                            emit_to(&emit, client, ServerMsg::Geo(Box::new(geo)));
                            emit_to(
                                &emit,
                                client,
                                ServerMsg::GridTraces {
                                    base_seq: seq,
                                    grids,
                                },
                            );
                        } else {
                            // The edits that replaced `format!`-built `Eval`s
                            // keep that path's semantics: the `> net.…` echo,
                            // busy status while a slow builder runs, a cleared
                            // abort flag so Stop still interrupts it, and the
                            // panic caught like a script's.
                            let equiv = script_equiv(&edit);
                            if let Some(line) = &equiv {
                                abort.store(false, Ordering::Relaxed);
                                status_busy = true;
                                status!();
                                emit_b(&emit, con(format!("> {line}")));
                            }
                            // Stage 23: the pre-edit state for the history
                            // stack — taken while the version check passed,
                            // before apply mutates anything.
                            let pre = undo_snap(&env);
                            // Scope the net borrow: the Delta arm needs fresh
                            // reads (non_grid, geo_snap) — a match scrutinee
                            // RefMut would live across the arms.
                            let applied = {
                                let mut n = env.net.borrow_mut();
                                let dt = env.cfg.borrow().dt_ms;
                                let cancel = env.cancel.borrow();
                                if equiv.is_some() {
                                    // catch_unwind so a panic inside a builder
                                    // can't kill the worker thread silently —
                                    // the Eval arm wrapped `eval_str` the same
                                    // way.
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        apply_edit(
                                            &mut n,
                                            dt,
                                            &mut env.rng.borrow_mut(),
                                            &edit,
                                            &**cancel,
                                        )
                                    }))
                                } else {
                                    Ok(apply_edit(
                                        &mut n,
                                        dt,
                                        &mut env.rng.borrow_mut(),
                                        &edit,
                                        &**cancel,
                                    ))
                                }
                            };
                            if equiv.is_some() {
                                status_busy = false;
                                status!();
                            }
                            match applied {
                                Err(_) => emit_b(&emit, con("! script panicked (worker survived)")),
                                Ok(Err(e)) => emit_to(
                                    &emit,
                                    client,
                                    ServerMsg::Error(format!("edit refused: {e}")),
                                ),
                                Ok(Ok(applied)) => {
                                    // A stopped builder's partial synapses
                                    // stay in the net — the delta below
                                    // carries them to every client, like the
                                    // Eval's post-bump Geo republish did.
                                    if matches!(applied, Applied::Stopped(_)) {
                                        // What the Eval's `! {e}` printed for
                                        // the same stop — rhai's Display adds
                                        // the call position, always (1,5) for
                                        // `net.create_connections(…)`.
                                        emit_b(
                                            &emit,
                                            con("! Runtime error: stopped by user \
                                                 (line 1, position 5)"),
                                        );
                                    }
                                    push_step(
                                        &mut undo,
                                        UndoEntry {
                                            label: edit_label(&edit),
                                            snap: pre,
                                        },
                                    );
                                    redo.clear();
                                    bump(&env);
                                    let gen = env.net_gen.get();
                                    let ver = env.net_version.get();
                                    match applied {
                                        Applied::Full => {
                                            published_geo_version = ver;
                                            let geo = geo_snap(&env);
                                            let (seq, grids) = {
                                                let mut p = pub_rc.borrow_mut();
                                                (p.seq, p.trace_snapshot(&geo))
                                            };
                                            emit_ev(
                                                &emit,
                                                ServerEvent::broadcast(ServerMsg::Geo(Box::new(
                                                    geo,
                                                ))),
                                            );
                                            emit_ev(
                                                &emit,
                                                ServerEvent::broadcast(ServerMsg::GridTraces {
                                                    base_seq: seq,
                                                    grids,
                                                }),
                                            );
                                        }
                                        Applied::Delta(mut d) | Applied::Stopped(mut d) => {
                                            published_geo_version = ver;
                                            d.version = ver;
                                            d.gen = gen;
                                            d.dt_ms = env.cfg.borrow().dt_ms;
                                            d.non_grid = non_grid(&env.net.borrow());
                                            emit_ev(
                                                &emit,
                                                ServerEvent::broadcast(ServerMsg::GeoDelta(
                                                    Box::new(d),
                                                )),
                                            );
                                        }
                                        Applied::Noop => {}
                                    }
                                }
                            }
                        }
                    }
                    // Stage 23: shared network history. An undo/redo is an
                    // edit like any other — `undo_restore` bumps the version
                    // and the loop-top republish block broadcasts the fresh
                    // `Geo` to every client on the run.
                    ClientMsg::Undo => match undo.pop_back() {
                        None => emit_to(&emit, client, con("nothing to undo")),
                        Some(e) => {
                            push_step(
                                &mut redo,
                                UndoEntry {
                                    label: e.label.clone(),
                                    snap: undo_snap(&env),
                                },
                            );
                            undo_restore(&env, e.snap);
                            emit_b(&emit, con(format!("undo: {}", e.label)));
                        }
                    },
                    ClientMsg::Redo => match redo.pop_back() {
                        None => emit_to(&emit, client, con("nothing to redo")),
                        Some(e) => {
                            push_step(
                                &mut undo,
                                UndoEntry {
                                    label: e.label.clone(),
                                    snap: undo_snap(&env),
                                },
                            );
                            undo_restore(&env, e.snap);
                            emit_b(&emit, con(format!("redo: {}", e.label)));
                        }
                    },
                    ClientMsg::CopySubnet { ids } => {
                        let set: HashSet<u32> = ids.into_iter().collect();
                        let sub = env.net.borrow().subnet(&set);
                        match net_format::io::to_json(&sub) {
                            Ok(txt) => emit_to(&emit, client, ServerMsg::Clipboard(txt)),
                            Err(e) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!("copy failed: {e}")),
                            ),
                        }
                    }
                    ClientMsg::SetBackend(b) => {
                        env.set_backend(match b {
                            Backend::Cpu => BackendKind::Cpu,
                            Backend::Cuda => BackendKind::Cuda,
                        });
                        env.invalidate_sim();
                    }
                    ClientMsg::SetQuota(q) => {
                        quota_bytes = q;
                        *env.quota_bytes.borrow_mut() = q;
                    }
                    ClientMsg::SetDtMs(dt) => {
                        if crate::valid_dt_ms(dt) {
                            env.cfg.borrow_mut().dt_ms = dt;
                            env.invalidate_sim();
                        } else {
                            emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!(
                                    "dt {dt} out of range {}..{} ms",
                                    crate::MIN_DT_MS,
                                    crate::MAX_DT_MS
                                )),
                            );
                        }
                    }
                    ClientMsg::SetNoise { hz, kick } => {
                        env.cfg.borrow_mut().noise_rate_hz = hz;
                        env.cfg.borrow_mut().noise_kick_mv = kick;
                        if let Ok(h) = env.sim_handle() {
                            h.borrow_mut().set_noise(hz, kick);
                        }
                    }
                    ClientMsg::SetInputConst { name, value } => {
                        let idx = env.net.borrow().input_index(&name);
                        if let (Some(i), Ok(h)) = (idx, env.sim_handle()) {
                            h.borrow_mut().set_input_constant(i, value);
                        }
                        let mut p = pub_rc.borrow_mut();
                        publish_env(&env, &mut p, &emit);
                    }
                    ClientMsg::PlayInput { name, series } => {
                        let idx = env.net.borrow().input_index(&name);
                        if let (Some(i), Ok(h)) = (idx, env.sim_handle()) {
                            h.borrow_mut().play_input(i, series);
                        }
                    }
                    ClientMsg::SetExternal { neuron, amps } => {
                        if let Ok(h) = env.sim_handle() {
                            h.borrow_mut().set_external(neuron as usize, amps);
                        }
                    }
                    // UI file pickers/read/write — confined to --data-dir by
                    // `resolve` (or verbatim in local mode) so the UI never
                    // needs its own std::fs.
                    ClientMsg::ListFiles { path } => {
                        let r = resolve(&env, &path).and_then(|p| {
                            let mut entries: Vec<proto::FileEntry> = std::fs::read_dir(&p)
                                .map_err(|e| e.to_string())?
                                .filter_map(|e| e.ok())
                                .filter_map(|e| {
                                    let md = e.metadata().ok()?;
                                    Some(proto::FileEntry {
                                        name: e.file_name().to_string_lossy().into_owned(),
                                        is_dir: md.is_dir(),
                                        size: md.len(),
                                        mtime_ms: md
                                            .modified()
                                            .ok()
                                            .and_then(|t| {
                                                t.duration_since(std::time::UNIX_EPOCH).ok()
                                            })
                                            .map(|d| d.as_millis() as u64)
                                            .unwrap_or(0),
                                    })
                                })
                                .collect();
                            entries
                                .sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
                            Ok(entries)
                        });
                        match r {
                            Ok(entries) => {
                                emit_to(&emit, client, ServerMsg::FileList { path, entries })
                            }
                            Err(e) => {
                                emit_to(&emit, client, ServerMsg::Error(format!("ls {path}: {e}")))
                            }
                        }
                    }
                    ClientMsg::ReadFile { path } => {
                        match resolve(&env, &path)
                            .and_then(|p| std::fs::read(&p).map_err(|e| e.to_string()))
                        {
                            Ok(data) => emit_to(&emit, client, ServerMsg::FileData { path, data }),
                            Err(e) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!("read {path}: {e}")),
                            ),
                        }
                    }
                    ClientMsg::WriteFile { path, data } => {
                        let r = resolve(&env, &path).and_then(|p| {
                            quota_check(
                                &env,
                                quota_bytes,
                                data.len() as u64,
                                std::path::Path::new(&p),
                            )?;
                            std::fs::write(&p, &data).map_err(|e| e.to_string())
                        });
                        match r {
                            Ok(()) => emit_to(&emit, client, ServerMsg::FileWritten { path }),
                            Err(e) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!("write {path}: {e}")),
                            ),
                        }
                    }
                    // A dataset push: bytes land under `datasets/<name>/`
                    // through the same resolve rules; the server caps the
                    // signal size (`--max-upload-mb`).
                    ClientMsg::UploadDataset {
                        name,
                        signal_ext,
                        signal,
                        meta_json,
                        labels_json,
                    } => {
                        let ext = if signal_ext == "npy" { "npy" } else { "f32" };
                        let r = if let Err(e) = crate::dataset::Dataset::check_name(&name) {
                            Err(e)
                        } else if signal.len() > max_upload_bytes {
                            Err(format!(
                                "upload {}: {} B over the {} B limit",
                                name,
                                signal.len(),
                                max_upload_bytes
                            ))
                        } else if serde_json::from_str::<serde_json::Value>(&meta_json).is_err()
                            || serde_json::from_str::<serde_json::Value>(&labels_json).is_err()
                        {
                            Err(format!("upload {name}: meta/labels not JSON"))
                        } else {
                            resolve(&env, &format!("datasets/{name}")).and_then(|dir| {
                                let dir = std::path::PathBuf::from(dir);
                                quota_check(
                                    &env,
                                    quota_bytes,
                                    (signal.len() + meta_json.len() + labels_json.len()) as u64,
                                    &dir,
                                )?;
                                std::fs::create_dir_all(&dir)
                                    .and_then(|()| {
                                        std::fs::write(dir.join(format!("signal.{ext}")), &signal)?;
                                        std::fs::write(dir.join("meta.json"), &meta_json)?;
                                        std::fs::write(dir.join("labels.json"), &labels_json)
                                    })
                                    .map_err(|e| e.to_string())
                                    // sanity: the file must load as a dataset
                                    .and_then(|()| {
                                        crate::dataset::Dataset::load(
                                            dir.parent().unwrap_or(&dir),
                                            &name,
                                        )
                                        .map(|_| ())
                                    })
                            })
                        };
                        match r {
                            Ok(()) => emit_to(
                                &emit,
                                client,
                                ServerMsg::FileWritten {
                                    path: format!("datasets/{name}"),
                                },
                            ),
                            Err(e) => {
                                emit_to(&emit, client, ServerMsg::Error(format!("upload: {e}")))
                            }
                        }
                    }
                    ClientMsg::Mkdir { path } => {
                        let r = resolve(&env, &path).and_then(|p| {
                            // Directories consume quota too (inode space) —
                            // the check is the same helper as file writes.
                            quota_check(&env, quota_bytes, 4096, std::path::Path::new(&p))?;
                            std::fs::create_dir_all(&p).map_err(|e| e.to_string())
                        });
                        match r {
                            Ok(()) => emit_to(&emit, client, ServerMsg::DirCreated { path }),
                            Err(e) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!("mkdir {path}: {e}")),
                            ),
                        }
                    }
                    ClientMsg::DeleteFile { path } => {
                        let r = resolve(&env, &path).and_then(|p| {
                            if is_sandbox_root(&env, &p) {
                                // `DeleteFile ""`/`"."` resolves to the user
                                // root — removing it breaks every later file
                                // op until the run restarts. Refuse outright.
                                return Err("refusing to delete the root directory".into());
                            }
                            let p = std::path::PathBuf::from(&p);
                            if p.is_dir() {
                                std::fs::remove_dir_all(&p).map_err(|e| e.to_string())
                            } else {
                                std::fs::remove_file(&p).map_err(|e| e.to_string())
                            }
                        });
                        match r {
                            Ok(()) => emit_to(&emit, client, ServerMsg::FileDeleted { path }),
                            Err(e) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!("delete {path}: {e}")),
                            ),
                        }
                    }
                    ClientMsg::RenameFile { from, to } => {
                        let r = resolve(&env, &from).and_then(|f| {
                            if is_sandbox_root(&env, &f) {
                                return Err("refusing to rename the root directory".into());
                            }
                            resolve(&env, &to)
                                .and_then(|t| std::fs::rename(&f, &t).map_err(|e| e.to_string()))
                        });
                        match r {
                            Ok(()) => emit_to(&emit, client, ServerMsg::FileRenamed { from, to }),
                            Err(e) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!("rename {from} → {to}: {e}")),
                            ),
                        }
                    }
                    ClientMsg::FileUsage => {
                        let used = env
                            .data_dir
                            .borrow()
                            .as_ref()
                            .map(|d| dir_size(d))
                            .unwrap_or(0);
                        emit_to(
                            &emit,
                            client,
                            ServerMsg::Usage {
                                used,
                                quota: quota_bytes,
                            },
                        );
                    }
                    ClientMsg::UploadStart { path, size } => {
                        let r = resolve(&env, &path).and_then(|p| {
                            if size > max_upload_bytes as u64 {
                                return Err(format!(
                                    "{size} B over the {} B per-file limit",
                                    max_upload_bytes
                                ));
                            }
                            let p = std::path::PathBuf::from(&p);
                            quota_check(&env, quota_bytes, size, &p)?;
                            // Fresh upload: replace any stale part file.
                            let _ = std::fs::remove_file(part_path(&p));
                            if let Some(parent) = p.parent() {
                                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                            }
                            Ok(())
                        });
                        match r {
                            Ok(()) => {
                                open_uploads.insert(path.clone(), (0, size));
                                emit_to(&emit, client, ServerMsg::UploadReady { path });
                            }
                            Err(e) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!("upload {path}: {e}")),
                            ),
                        }
                    }
                    ClientMsg::UploadChunk { path, data, last } => {
                        let r = open_uploads
                            .get_mut(&path)
                            .ok_or_else(|| format!("no UploadStart for {path}"))
                            .and_then(|(got, declared)| {
                                *got += data.len() as u64;
                                if *got > *declared {
                                    return Err(format!(
                                        "{} B exceeds the declared {} B",
                                        *got, declared
                                    ));
                                }
                                resolve(&env, &path)
                            })
                            .and_then(|p| {
                                let p = std::path::PathBuf::from(&p);
                                let part = part_path(&p);
                                use std::io::Write;
                                let mut f = std::fs::OpenOptions::new()
                                    .create(true)
                                    .append(true)
                                    .open(&part)
                                    .map_err(|e| e.to_string())?;
                                f.write_all(&data).map_err(|e| e.to_string())?;
                                drop(f);
                                if last {
                                    std::fs::rename(&part, &p).map_err(|e| e.to_string())?;
                                }
                                Ok(())
                            });
                        match r {
                            Ok(()) if last => {
                                open_uploads.remove(&path);
                                emit_to(&emit, client, ServerMsg::FileWritten { path });
                            }
                            Ok(()) => {}
                            Err(e) => {
                                // No partial file: drop the part on any error.
                                open_uploads.remove(&path);
                                if let Ok(p) = resolve(&env, &path) {
                                    let _ =
                                        std::fs::remove_file(part_path(std::path::Path::new(&p)));
                                }
                                emit_to(
                                    &emit,
                                    client,
                                    ServerMsg::Error(format!("upload {path}: {e}")),
                                )
                            }
                        }
                    }
                    ClientMsg::DownloadFile { path } => {
                        match resolve(&env, &path)
                            .and_then(|p| std::fs::read(&p).map_err(|e| e.to_string()))
                        {
                            Ok(data) => {
                                const CHUNK: usize = 256 * 1024;
                                let mut i = 0;
                                while i < data.len() {
                                    let end = (i + CHUNK).min(data.len());
                                    emit_to(
                                        &emit,
                                        client,
                                        ServerMsg::FileChunk {
                                            path: path.clone(),
                                            data: data[i..end].to_vec(),
                                            done: false,
                                        },
                                    );
                                    i = end;
                                }
                                emit_to(
                                    &emit,
                                    client,
                                    ServerMsg::FileChunk {
                                        path,
                                        data: Vec::new(),
                                        done: true,
                                    },
                                );
                            }
                            Err(e) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!("download {path}: {e}")),
                            ),
                        }
                    }
                    // Jobs are a `server` feature — they need a shared
                    // broadcast fan-out the local transport doesn't have.
                    ClientMsg::StartJob { .. }
                    | ClientMsg::CancelJob { .. }
                    | ClientMsg::ListJobs
                    | ClientMsg::OpenJob { .. } => emit_to(
                        &emit,
                        client,
                        ServerMsg::Error("jobs run on a `server` — connect with --connect".into()),
                    ),
                }
            }));
            if res.is_err() {
                // The panic may have left a half-updated state — clear the
                // busy flag so clients don't sit on a stale "working".
                status_busy = false;
                emit_b(
                    &emit,
                    con("! internal error handling that command (worker survived)"),
                );
            }
        }
        status!();

        // ---- continuous run: one chunk, throttle to `speed` × realtime ----
        if running {
            match env.sim_handle() {
                Ok(h) => {
                    let t0 = Instant::now();
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        h.borrow_mut().run_steps(chunk)
                    }));
                    match res {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            running = false;
                            emit_b(&emit, con(format!("run failed: {e}")));
                            continue;
                        }
                        Err(_) => {
                            running = false;
                            emit_b(&emit, con("run panicked (worker survived)"));
                            continue;
                        }
                    }
                    let el = t0.elapsed().as_secs_f32().max(1e-6);
                    {
                        let mut p = pub_rc.borrow_mut();
                        p.sps = chunk as f32 / el;
                        publish_sim(
                            &env.net,
                            env.net_version.get(),
                            &mut h.borrow_mut(),
                            &mut p,
                            &emit,
                        );
                    }
                    let dt = env.cfg.borrow().dt_ms;
                    let target_sps = (speed * 1000.0 / dt).max(1.0);
                    let target_el = chunk as f32 / target_sps;
                    if target_el > el {
                        std::thread::sleep(Duration::from_secs_f32(target_el - el));
                    }
                    chunk = ((target_sps / 4.0) as usize).clamp(50, 20_000);
                }
                Err(e) => {
                    running = false;
                    emit_b(&emit, con(format!("sim build failed: {e}")));
                }
            }
        }
    }
}

fn bump(env: &Env) {
    env.invalidate_sim();
    env.net_version.set(env.net_version.get() + 1);
}

fn cfg_wire(c: &SimConfig) -> Config {
    Config {
        dt_ms: c.dt_ms,
        synapse_tau_ms: c.synapse_tau_ms,
        readout_tau_ms: c.readout_tau_ms,
        noise_rate_hz: c.noise_rate_hz,
        noise_kick_mv: c.noise_kick_mv,
        seed: c.seed,
        int_weights: c.int_weights,
    }
}

/// Mutable publish state shared by the progress callback and the cmd loop.
struct Pub {
    /// Watched neuron ids for v(t) plots.
    watch: Vec<u32>,
    /// `sim.step_count()` at the previous publish — `counts` covers the
    /// interval since then.
    prev_step: u64,
    /// `sim.time_ms()` at the previous publish — the plane-decay window.
    prev_time_ms: f64,
    sps: f32,
    /// net version the cached grid member lists were built from.
    grid_ver: u64,
    /// texel-ordered member lists, one per grid group (`geo.grids` order).
    grid_lists: Vec<Vec<u32>>,
    /// Current u8 trace planes — delta base + `GridTraces` snapshots.
    trace_u8: Vec<Vec<u8>>,
    /// net_gen seen — a change zeroes the dynamic state.
    net_gen: u64,
    /// A sim has been built & published at least once — Status.sim_built.
    sim_built: bool,
    /// Activity sequence counter.
    seq: u64,
    /// Non-grid neuron ids (Geo.non_grid order) for the glow vector.
    non_grid: Vec<u32>,
    /// Rate-limit clock for Activity publishes.
    next_pub: Instant,
}

impl Default for Pub {
    fn default() -> Self {
        Self {
            watch: Vec::new(),
            prev_step: 0,
            prev_time_ms: 0.0,
            sps: 0.0,
            grid_ver: u64::MAX,
            grid_lists: Vec::new(),
            trace_u8: Vec::new(),
            net_gen: 0,
            sim_built: false,
            seq: 0,
            non_grid: Vec::new(),
            next_pub: Instant::now(),
        }
    }
}

impl Pub {
    /// Current trace planes sized to `geo.grids` — resizes the u8
    /// accumulators when geometry shifted (values carry over where the
    /// plane didn't change length).
    fn trace_snapshot(&mut self, geo: &Geo) -> Vec<Vec<u8>> {
        self.trace_u8.resize_with(geo.grids.len(), Vec::new);
        for (i, g) in geo.grids.iter().enumerate() {
            let n = (g.dims[0] * g.dims[1] * g.dims[2]) as usize;
            if self.trace_u8[i].len() != n {
                self.trace_u8[i] = vec![0u8; n];
            }
        }
        self.trace_u8.clone()
    }

    fn reset_dyn(&mut self, gen: u64) {
        self.net_gen = gen;
        self.prev_step = 0;
        self.prev_time_ms = 0.0;
        self.sps = 0.0;
        self.sim_built = false;
        // planes describe the replaced net — drop them; `trace_snapshot`
        // repopulates zeros at the new sizes on the next Geo.
        self.trace_u8.clear();
    }
}

/// Publish a snapshot NOW — callers use this at command/eval boundaries
/// where the rate-limited progress hook may have skipped the tail.
fn publish_env(env: &Env, p: &mut Pub, emit: &Emit) {
    if let Ok(h) = env.sim_handle() {
        p.next_pub = Instant::now();
        publish_sim(
            &env.net,
            env.net_version.get(),
            &mut h.borrow_mut(),
            p,
            emit,
        );
    }
}

/// Publish dynamic state read back from `sim` as an `Activity` frame.
/// Rate-limited: calls before `next_pub` just fold into the next frame.
fn publish_sim(net: &NetHandle, net_ver: u64, sim: &mut Sim, p: &mut Pub, emit: &Emit) {
    if Instant::now() < p.next_pub {
        return;
    }
    p.next_pub = Instant::now() + Duration::from_millis(MIN_ACTIVITY_MS);

    // Counts since previous publish: sum the record-ring rows covering the
    // elapsed steps. Intervals longer than the ring depth undercount the
    // oldest steps, which is fine for raster/particles.
    let back = sim
        .step_count()
        .saturating_sub(p.prev_step)
        .min(COUNTS_WINDOW as u64) as usize;
    let counts = sim.spike_counts(back).unwrap_or_default();
    p.prev_step = sim.step_count();
    // Sim-time covered by this frame — the client decays its planes by
    // the same factor, so the mirror stays faithful.
    let dms = (sim.time_ms() - p.prev_time_ms).max(0.0);
    p.prev_time_ms = sim.time_ms();
    let f_decay = (-dms / PLANE_TAU_MS as f64).exp();

    // Refresh cached non-grid index on geo bumps.
    if p.grid_ver != net_ver {
        p.grid_lists = grid_member_lists(&net.borrow());
        p.non_grid = non_grid(&net.borrow());
        p.grid_ver = net_ver;
    }

    // spike bitmap over the whole neuron-id space — raster and particles
    // need per-id spikes including grid members (whose glow rides the
    // texel deltas instead, which can't distinguish spikes from decay).
    let n = net.borrow().neurons.len();
    let mut bits = vec![0u8; n.div_ceil(8)];
    for (id, &c) in counts.iter().enumerate() {
        if c > 0 {
            bits[id / 8] |= 1 << (id % 8);
        }
    }
    let bit = |id: u32| bits[id as usize / 8] >> (id as usize % 8) & 1 == 1;

    // per-step state recording happens in the sim; (re)apply the watch
    // list every publish so a rebuilt sim gets it back, then drain the
    // samples recorded since the last publish — the plot gets every
    // step, not one point per chunk.
    sim.set_v_watch(&p.watch);
    let states = sim.drain_watch().unwrap_or_default();
    // states: [state slot][watch][samples] → the wire carries, per
    // watched neuron, one series per state slot (protocol 2).
    let watched: Vec<(u32, Vec<Vec<f32>>)> = p
        .watch
        .iter()
        .enumerate()
        .map(|(k, &id)| {
            let series = (0..states.len())
                .map(|s| {
                    states
                        .get(s)
                        .and_then(|slot| slot.get(k))
                        .cloned()
                        .unwrap_or_default()
                })
                .collect();
            (id, series)
        })
        .collect();

    // Grid-plane traces: (re)apply the watch lists after a rebuild, then
    // update `p.trace_u8` — the mirror of what clients display. The wire
    // carries NO texel data per-frame: clients decay their planes by
    // `f_decay` (same LUT) and stamp 255 on texels whose member spiked —
    // derivable from `spike_bits` + `Geo.grids[].members`.
    sim.set_grid_watches(&p.grid_lists);
    let traces = sim.grid_traces(PLANE_TAU_MS).unwrap_or_default();
    if p.trace_u8.len() != traces.len() {
        p.trace_u8 = vec![Vec::new(); traces.len()];
    }
    let plane_lut: [u8; 256] = std::array::from_fn(|v| (v as f64 * f_decay).round() as u8);
    let mut resync = false;
    for (gi, tr) in traces.iter().enumerate() {
        if p.trace_u8[gi].len() != tr.len() {
            // resized/new plane → resync from sim truth after the frame
            p.trace_u8[gi] = tr.iter().map(|&v| quantize(v)).collect();
            resync = true;
            continue;
        }
        for v in p.trace_u8[gi].iter_mut() {
            *v = plane_lut[*v as usize];
        }
        for (k, &m) in p.grid_lists[gi].iter().enumerate() {
            if bit(m) {
                p.trace_u8[gi][k] = 255;
            }
        }
    }

    p.seq += 1;
    p.sim_built = true;
    emit_b(
        emit,
        ServerMsg::Activity(Box::new(Activity {
            seq: p.seq,
            t_steps: sim.step_count(),
            time_ms: sim.time_ms(),
            spikes_total: sim.spikes_total(),
            rate_hz: sim.mean_rate_hz(),
            steps_per_sec: p.sps,
            spike_bits: bits,
            watched,
            input_values: sim.input_values(),
        })),
    );
    if resync {
        emit_b(
            emit,
            ServerMsg::GridTraces {
                base_seq: p.seq,
                grids: p.trace_u8.clone(),
            },
        );
    }
}

fn quantize(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn run_published(
    env: &Env,
    h: &Rc<std::cell::RefCell<Sim>>,
    mut steps: usize,
    pub_rc: &Rc<std::cell::RefCell<Pub>>,
    abort: &AtomicBool,
    emit: &Emit,
) {
    let t0 = Instant::now();
    let mut done = 0usize;
    // The watch list is only pushed into the sim at publish time —
    // apply it before stepping so a Watch→RunMs pair records.
    {
        let p = pub_rc.borrow();
        h.borrow_mut().set_v_watch(&p.watch);
    }
    while steps > 0 && !abort.load(Ordering::Relaxed) {
        let n = steps.min(200);
        let res =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.borrow_mut().run_steps(n)));
        match res {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                emit_b(emit, con(format!("run failed: {e}")));
                return;
            }
            Err(_) => {
                emit_b(emit, con("run panicked (worker survived)"));
                return;
            }
        }
        steps -= n;
        done += n;
        let mut p = pub_rc.borrow_mut();
        p.sps = done as f32 / t0.elapsed().as_secs_f32().max(1e-6);
        publish_env(env, &mut p, emit);
    }
    // Always flush the tail — the rate-limiter may have skipped the last
    // partial window otherwise.
    let mut p = pub_rc.borrow_mut();
    p.next_pub = Instant::now();
    publish_env(env, &mut p, emit);
}

/// Resolve a user-supplied path inside `env.data_dir` when set.
/// Refuses `..` components and absolute paths outright.
pub fn resolve(env: &Env, p: &str) -> Result<String, String> {
    env.resolve_path(p)
}

/// `resolved` is the sandbox root itself — `""` and `"."` both land
/// there; deleting/renaming it would wipe the user's whole directory
/// (every later file op then fails until the run restarts).
fn is_sandbox_root(env: &Env, resolved: &str) -> bool {
    let Some(d) = env.data_dir.borrow().clone() else {
        return false;
    };
    let root = d.canonicalize().unwrap_or(d);
    std::path::Path::new(resolved) == root
}

/// `.upload` staging name next to the target — committed by rename on
/// the last chunk, removed on any failure so a failed upload leaves
/// no partial file behind.
fn part_path(p: &std::path::Path) -> std::path::PathBuf {
    let mut s = p.as_os_str().to_os_string();
    s.push(".upload");
    std::path::PathBuf::from(s)
}

/// Quota gate for protocol writes: `usage(data_dir) + extra` must fit
/// inside `quota` (0 = unlimited). Subtracts the bytes `target`
/// already occupies when the write overwrites it. The `.upload` part
/// file counts toward usage, so chunk-by-chunk growth stays honest.
fn quota_check(env: &Env, quota: u64, extra: u64, target: &std::path::Path) -> Result<(), String> {
    if quota == 0 {
        return Ok(());
    }
    let Some(dir) = env.data_dir.borrow().clone() else {
        return Ok(());
    };
    let existing = std::fs::metadata(target).map(|m| m.len()).unwrap_or(0);
    let used = dir_size(&dir).saturating_sub(existing);
    if used.saturating_add(extra) > quota {
        return Err(format!(
            "over quota: {} + {} B would exceed the {} B limit",
            used, extra, quota
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------------

/// Build the immutable geometry snapshot of the network.
fn geo_snap(env: &Env) -> Geo {
    let net = env.net.borrow();
    let dt = env.cfg.borrow().dt_ms.max(1e-6);
    let mut g = geo_from_net(&net, dt);
    g.version = env.net_version.get();
    g.gen = env.net_gen.get();
    g
}

/// Member list of one grid group in texel order (position k = lattice
/// index k), computed from positions rather than list order so a reordered
/// `members` vec still maps correctly. `None` when the lattice is broken.
fn texel_order(gs: &net_format::GridShape, members: &[u32], net: &Network) -> Option<Vec<u32>> {
    let mut ord = vec![u32::MAX; gs.texels()];
    for &m in members {
        let neu = net.neurons.get(m as usize)?;
        ord[gs.member_index(gs.lattice_at(neu.pos))] = m;
    }
    if ord.contains(&u32::MAX) {
        return None;
    }
    Some(ord)
}

/// All grid groups' texel-ordered member lists, in `Geo.grids` order.
fn grid_member_lists(net: &Network) -> Vec<Vec<u32>> {
    grid_snaps(net).into_iter().map(|g| g.members).collect()
}

/// `Geo.grids` content: every group with a valid lattice, members in
/// texel order.
fn grid_snaps(net: &Network) -> Vec<Grid> {
    net.groups
        .iter()
        .enumerate()
        .filter_map(|(gi, g)| {
            let gs = g.grid?;
            let members = texel_order(&gs, &g.members, net)?;
            Some(Grid {
                group: gi,
                dims: gs.dims,
                axes: gs.axes,
                origin: gs.origin,
                spacing: gs.spacing,
                members,
            })
        })
        .collect()
}

/// Neuron ids covered by no grid group — the Activity bitmap index space.
fn non_grid(net: &Network) -> Vec<u32> {
    let mut in_grid = vec![false; net.neurons.len()];
    for g in &net.groups {
        if g.grid.is_some() {
            for &m in &g.members {
                if let Some(f) = in_grid.get_mut(m as usize) {
                    *f = true;
                }
            }
        }
    }
    in_grid
        .iter()
        .enumerate()
        .filter_map(|(i, &g)| (!g).then_some(i as u32))
        .collect()
}

/// Geometry snapshot straight from a `Network` (headless screenshot path).
/// `net.types` → the wire type table.
pub fn types_proto(net: &Network) -> Vec<proto::TypeRow> {
    net.types
        .iter()
        .map(|t| proto::TypeRow {
            name: t.name.clone(),
            model: t.model.clone(),
            params: t.params.clone(),
            kind: t.kind.into(),
            color: t.color,
        })
        .collect()
}

/// The model registry → `Hello.models` descriptors — everything the
/// client needs to build its inspector and plot tabs without knowing
/// a single model by name.
pub fn model_descs() -> Vec<proto::ModelDesc> {
    crate::model::MODELS
        .iter()
        .map(|m| proto::ModelDesc {
            id: m.id().into(),
            label: m.label().into(),
            params: m
                .params()
                .iter()
                .map(|p| proto::ParamDesc {
                    name: p.name.into(),
                    unit: p.unit.into(),
                    default: p.default,
                    min: p.min,
                    max: p.max,
                    doc: p.doc.into(),
                })
                .collect(),
            state: m
                .state()
                .iter()
                .map(|s| proto::StateDesc {
                    name: s.name.into(),
                    unit: s.unit.into(),
                    plot_range: s.plot_range,
                })
                .collect(),
            presets: m
                .presets()
                .iter()
                .map(|(n, p)| (n.to_string(), p.to_vec()))
                .collect(),
            map: m.param_map().map(|pm| proto::MapDesc {
                planes: pm
                    .planes
                    .iter()
                    .map(|p| proto::MapPlaneDesc {
                        x: p.x,
                        y: p.y,
                        x_range: p.x_range,
                        y_range: p.y_range,
                        spots: p
                            .spots
                            .iter()
                            .map(|s| proto::MapSpot {
                                label: s.label.into(),
                                name: s.name.into(),
                                params: s.params.to_vec(),
                            })
                            .collect(),
                    })
                    .collect(),
            }),
        })
        .collect()
}

pub fn geo_from_net(net: &Network, dt: f32) -> Geo {
    let dt = dt.max(1e-6);
    let n = net.neurons.len();

    let mut pos = Vec::with_capacity(n);
    let mut kind = Vec::with_capacity(n);
    let mut names = Vec::with_capacity(n);
    let mut ty = Vec::with_capacity(n);
    let mut params = Vec::with_capacity(n);
    for (i, neu) in net.neurons.iter().enumerate() {
        pos.push(neu.pos);
        kind.push(match net.neuron_kind(i as u32) {
            net_format::NeuronKind::Inhibitory => 1,
            _ => 0,
        });
        names.push(neu.name.clone());
        ty.push(neu.ty);
        params.push(neu.params.as_deref().map(|p| p.to_vec()));
    }

    let mut syn = Vec::with_capacity(net.synapses.len());
    let mut delay = Vec::with_capacity(net.synapses.len());
    let mut per_src: Vec<Vec<(u32, u16)>> = vec![Vec::new(); n];
    for s in &net.synapses {
        syn.push((s.src, s.tgt));
        let d = (s.delay_ms / dt).round().max(1.0) as u16;
        delay.push(d);
        let de = &mut per_src[s.src as usize];
        if de.len() < GEO_EDGES_PER_NEURON {
            de.push((s.tgt, d));
        }
    }
    let mut disp_ptr = Vec::with_capacity(n + 1);
    let mut disp_edges = Vec::new();
    for de in &per_src {
        disp_ptr.push(disp_edges.len() as u32);
        disp_edges.extend_from_slice(de);
    }
    disp_ptr.push(disp_edges.len() as u32);

    Geo {
        version: 0,
        gen: 0,
        pos,
        kind,
        names,
        types: types_proto(net),
        ty,
        params,
        syn,
        syn_delay_steps: delay,
        disp_edges,
        disp_ptr,
        groups: net
            .groups
            .iter()
            .map(|g| (g.name.clone(), g.members.clone()))
            .collect(),
        inputs: inputs_proto(net),
        readouts: net
            .readouts
            .iter()
            .map(|r| (r.name.clone(), r.units.clone(), r.weights.clone()))
            .collect(),
        samples: net
            .samples
            .iter()
            .map(|s| (s.name.clone(), s.times.clone(), s.values.clone()))
            .collect(),
        scripts: net.scripts.clone(),
        grids: grid_snaps(net),
        non_grid: non_grid(net),
        dt_ms: dt,
    }
}

// ---------------------------------------------------------------------------
// edits → deltas
// ---------------------------------------------------------------------------

enum Applied {
    /// Emit a `GeoDelta`.
    Delta(GeoDelta),
    /// Emit a full `Geo` — the edit renumbered neuron ids.
    Full,
    /// Nothing observable changed (e.g. duplicate disconnect).
    Noop,
    /// A cancellable builder stopped early — the delta carries the
    /// partial work so every client still syncs to the net as it
    /// stands (the Rhai builder did the same: kept what it built).
    Stopped(GeoDelta),
}

/// The `net.*` script call this edit replaced, formatted exactly as
/// the old `format!`-built `Eval` produced it — echoed to the console
/// as `> net.…`. `None` for edits that never went through a script.
fn script_equiv(e: &NetEdit) -> Option<String> {
    match e {
        NetEdit::CreateConnections {
            group,
            scaling,
            lambda,
            w,
        } => Some(format!(
            "net.create_connections({group:?}, {scaling:?}, {lambda:?}, {w:?})"
        )),
        NetEdit::SetSynapseWeights { group, mean, sd } => Some(format!(
            "net.set_synapse_weights({group:?}, {mean:?}, {sd:?})"
        )),
        NetEdit::SetSynTauDist { src, dst, mean, sd } => Some(format!(
            "net.set_syn_tau_dist({src:?}, {dst:?}, {mean:?}, {sd:?})"
        )),
        NetEdit::SetExcInh { group, frac } => Some(format!("net.set_exc_inh({group:?}, {frac:?})")),
        NetEdit::ScriptAdd { name, src } => Some(format!("net.script_add({name:?}, {src:?})")),
        _ => None,
    }
}

/// Apply a `NetEdit` to `net` and describe what changed as a delta.
/// `dt` converts stored ms delays to sim steps.
fn apply_edit(
    net: &mut Network,
    dt: f32,
    rng: &mut net_format::Rng,
    e: &NetEdit,
    cancel: &dyn Fn() -> bool,
) -> Result<Applied, String> {
    let mut d = GeoDelta::default();
    let steps = |ms: f32| (ms / dt.max(1e-6)).round().max(1.0) as u16;
    let row = |net: &Network, id: u32| {
        let neu = net.neuron(id);
        NeuronRow {
            id,
            pos: neu.pos,
            kind: match net.neuron_kind(id) {
                net_format::NeuronKind::Inhibitory => 1,
                _ => 0,
            },
            name: neu.name.clone(),
            ty: neu.ty,
            params: neu.params.as_deref().map(|p| p.to_vec()),
        }
    };
    let grids_now = |net: &Network| -> Vec<usize> {
        net.groups
            .iter()
            .enumerate()
            .filter_map(|(i, g)| g.grid.map(|_| i))
            .collect()
    };
    match e {
        NetEdit::AddNeuron { pos, kind } => {
            let id = net.add_neuron(*pos, (*kind).into());
            d.neurons_added.push(NeuronRow {
                id,
                pos: *pos,
                kind: match kind {
                    proto::NeuronKind::Inhibitory => 1,
                    _ => 0,
                },
                name: net.neuron(id).name.clone(),
                ty: net.neuron(id).ty,
                params: None,
            });
        }
        NetEdit::DeleteNeurons { ids } => {
            check_ids(net, ids)?;
            net.delete_neurons(ids.clone());
            return Ok(Applied::Full);
        }
        NetEdit::Connect { from, to, w } => {
            check_ids(net, &[*from, *to])?;
            if from == to {
                return Err(format!("self-synapse on neuron {from}"));
            }
            net.connect(*from, *to, *w);
            let s = net.synapses.last().unwrap();
            d.syn_added.push(SynRow {
                src: s.src,
                tgt: s.tgt,
                delay_steps: steps(s.delay_ms),
            });
            d.disp_changed.push((*from, disp_list(net, *from, dt)));
        }
        NetEdit::Disconnect { from, to } => {
            let before = net.synapses.len();
            net.synapses.retain(|s| !(s.src == *from && s.tgt == *to));
            if net.synapses.len() == before {
                return Ok(Applied::Noop);
            }
            d.syn_removed.push((*from, *to));
            d.disp_changed.push((*from, disp_list(net, *from, dt)));
        }
        NetEdit::MoveNeurons { items } => {
            let ids: Vec<u32> = items.iter().map(|(i, _)| *i).collect();
            check_ids(net, &ids)?;
            let grid_before = grids_now(net);
            for (id, pos) in items {
                net.neuron_mut(*id).pos = *pos;
            }
            // delays of every synapse touching a moved neuron may change;
            // sources whose TARGET moved recompute too — their display
            // edge lists carry the delay.
            let moved: HashSet<u32> = ids.into_iter().collect();
            net.refresh_delays_moved(&moved);
            net.invalidate_grids();
            grid_delta_membership(net, &grid_before, &mut d);
            for (id, _) in items {
                d.neurons_changed.push(row(net, *id));
            }
            let mut disp_srcs: HashSet<u32> = moved.iter().copied().collect();
            for (i, s) in net.synapses.iter().enumerate() {
                if moved.contains(&s.src) || moved.contains(&s.tgt) {
                    d.syn_delays.push((i as u32, steps(s.delay_ms)));
                }
                if moved.contains(&s.tgt) {
                    disp_srcs.insert(s.src);
                }
            }
            d.disp_changed = disp_lists(net, &disp_srcs, dt);
        }
        NetEdit::SetNeuronName { id, name } => {
            check_ids(net, &[*id])?;
            net.neuron_mut(*id).name = name.clone();
            d.neurons_changed.push(row(net, *id));
        }
        NetEdit::SetNeuronKind { id, kind } => {
            check_ids(net, &[*id])?;
            // set_kind may append a canonical type row (e.g. the first
            // inhibitory neuron on a net that only had "exc").
            let n_types = net.types.len();
            net.set_kind(*id, (*kind).into());
            d.neurons_changed.push(row(net, *id));
            if net.types.len() != n_types {
                d.types = Some(types_proto(net));
            }
        }
        NetEdit::UpsertType {
            name,
            model,
            params,
            kind,
            color,
        } => {
            let (_, m) =
                crate::model::by_id(model).ok_or_else(|| format!("unknown model {model:?}"))?;
            if params.len() != m.params().len() {
                return Err(format!(
                    "type {name:?}: {} params, model {} wants {}",
                    params.len(),
                    m.id(),
                    m.params().len()
                ));
            }
            let ty = net.upsert_type(net_format::NeuronType {
                name: name.clone(),
                model: model.clone(),
                params: params.clone(),
                kind: (*kind).into(),
                color: *color,
            });
            d.types = Some(types_proto(net));
            // Every neuron on this type sees a new kind byte and possibly
            // a pruned override — resend their rows.
            for (i, n) in net.neurons.iter().enumerate() {
                if n.ty == ty {
                    d.neurons_changed.push(row(net, i as u32));
                }
            }
        }
        NetEdit::SetNeuronType { ids, ty } => {
            check_ids(net, ids)?;
            let t = net
                .type_index(ty)
                .ok_or_else(|| format!("unknown type {ty:?}"))?;
            net.set_type(ids, t);
            for &id in ids {
                d.neurons_changed.push(row(net, id));
            }
        }
        NetEdit::SetNeuronParams { ids, params } => {
            check_ids(net, ids)?;
            if let Some(p) = params {
                // One width check per id up front — the override row must
                // match the neuron's type param width exactly.
                for &id in ids {
                    let want = net.neuron_type(id).map(|t| t.params.len()).unwrap_or(0);
                    if p.len() != want {
                        return Err(format!(
                            "neuron {id}: {} params, its type wants {want}",
                            p.len()
                        ));
                    }
                }
                for &id in ids {
                    net.neuron_mut(id).params = Some(p.clone().into_boxed_slice());
                }
            } else {
                for &id in ids {
                    net.neuron_mut(id).params = None;
                }
            }
            for &id in ids {
                d.neurons_changed.push(row(net, id));
            }
        }
        NetEdit::SetSynapse {
            from,
            to,
            w,
            tau_ms,
            delay_ms,
        } => {
            check_ids(net, &[*from, *to])?;
            let mut found = false;
            let mut delay_changed = false;
            for (i, s) in net.synapses.iter_mut().enumerate() {
                if s.src == *from && s.tgt == *to {
                    if let Some(v) = w {
                        s.w = *v;
                    }
                    if let Some(v) = tau_ms {
                        s.tau_ms = v.max(0.0);
                    }
                    if let Some(v) = delay_ms {
                        s.delay_ms = v.max(1e-3);
                        d.syn_delays.push((i as u32, steps(s.delay_ms)));
                        delay_changed = true;
                    }
                    found = true;
                }
            }
            if !found {
                return Err(format!("no synapse {from}→{to}"));
            }
            if delay_changed {
                d.disp_changed.push((*from, disp_list(net, *from, dt)));
            }
        }
        NetEdit::SetNeuronPos { id, pos } => {
            check_ids(net, &[*id])?;
            let grid_before = grids_now(net);
            net.neuron_mut(*id).pos = *pos;
            net.refresh_delays_of(*id);
            net.invalidate_grids();
            grid_delta_membership(net, &grid_before, &mut d);
            d.neurons_changed.push(row(net, *id));
            let mut disp_srcs = vec![*id];
            for (i, s) in net.synapses.iter().enumerate() {
                if s.src == *id || s.tgt == *id {
                    d.syn_delays.push((i as u32, steps(s.delay_ms)));
                }
                if s.tgt == *id {
                    disp_srcs.push(s.src);
                }
            }
            for src in disp_srcs {
                let dl = disp_list(net, src, dt);
                if !dl.is_empty() {
                    d.disp_changed.push((src, dl));
                }
            }
        }
        NetEdit::InsertSubnetJson { json, offset } => {
            let sub = net_format::io::from_json(json).map_err(|e| format!("paste: {e}"))?;
            let (s0, g0) = (net.synapses.len(), net.groups.len());
            // insert_subnet merges into same-named existing groups — diff
            // member counts to catch those, not just appended groups.
            let prev_lens: Vec<usize> = net.groups.iter().map(|g| g.members.len()).collect();
            let grid_before = grids_now(net);
            let map = net.insert_subnet(&sub, *offset);
            for &id in &map {
                d.neurons_added.push(row(net, id));
            }
            for s in &net.synapses[s0..] {
                d.syn_added.push(SynRow {
                    src: s.src,
                    tgt: s.tgt,
                    delay_steps: steps(s.delay_ms),
                });
            }
            for (i, g) in net.groups.iter().enumerate() {
                if i >= g0 {
                    d.groups_added
                        .push((i as u32, g.name.clone(), g.members.clone()));
                } else if g.members.len() != prev_lens[i] {
                    d.groups_changed
                        .push((i as u32, g.name.clone(), g.members.clone()));
                }
            }
            for &src in map.iter() {
                let dl = disp_list(net, src, dt);
                if !dl.is_empty() {
                    d.disp_changed.push((src, dl));
                }
            }
            net.invalidate_grids();
            grid_delta_membership(net, &grid_before, &mut d);
        }
        NetEdit::CreateGrid2d {
            group,
            w,
            h,
            spacing,
            origin,
        } => {
            net_format::check_grid_dims([*w as i64, *h as i64, 1])?;
            let (n0, g0) = (net.neurons.len(), net.groups.len());
            net_format::builders::create_2d_grid(
                net,
                group,
                *w as usize,
                *h as usize,
                *spacing,
                *origin,
            );
            for id in n0..net.neurons.len() {
                d.neurons_added.push(row(net, id as u32));
            }
            for (i, g) in net.groups[g0..].iter().enumerate() {
                d.groups_added
                    .push((g0 as u32 + i as u32, g.name.clone(), g.members.clone()));
            }
            d.grids_changed = grid_snaps(net);
        }
        NetEdit::CreateGrid3d {
            group,
            w,
            h,
            d: depth,
            spacing,
            origin,
        } => {
            net_format::check_grid_dims([*w as i64, *h as i64, *depth as i64])?;
            let (n0, g0) = (net.neurons.len(), net.groups.len());
            net_format::builders::create_3d_grid(
                net,
                group,
                *w as usize,
                *h as usize,
                *depth as usize,
                *spacing,
                *origin,
            );
            for id in n0..net.neurons.len() {
                d.neurons_added.push(row(net, id as u32));
            }
            for (i, g) in net.groups[g0..].iter().enumerate() {
                d.groups_added
                    .push((g0 as u32 + i as u32, g.name.clone(), g.members.clone()));
            }
            d.grids_changed = grid_snaps(net);
        }
        NetEdit::AddInput { name, targets, pos } => {
            // Q4: dangling ids/non-finite weights must not reach the net.
            check_ids(net, &targets.iter().map(|(id, _)| *id).collect::<Vec<_>>())?;
            for (_, w) in targets {
                if !w.is_finite() {
                    return Err(format!("input {name}: non-finite weight {w}"));
                }
            }
            net.add_input(name, targets.clone(), *pos);
            d.inputs = Some(inputs_proto(net));
        }
        NetEdit::AddInputTarget { name, target, w } => {
            check_ids(net, &[*target])?;
            if !w.is_finite() {
                return Err(format!("input {name}: non-finite weight {w}"));
            }
            let k = net
                .inputs
                .iter()
                .position(|i| &i.name == name)
                .ok_or_else(|| format!("no input named {name}"))?;
            net.inputs[k].targets.push((*target, *w));
            d.inputs = Some(inputs_proto(net));
        }
        NetEdit::MoveInputs { items } => {
            for (k, pos) in items {
                let inp = net
                    .inputs
                    .get_mut(*k as usize)
                    .ok_or_else(|| format!("no input #{k}"))?;
                inp.pos = *pos;
            }
            if !items.is_empty() {
                d.inputs = Some(inputs_proto(net));
            }
        }
        NetEdit::AddSamples { name } => {
            net.samples_mut(name);
            d.samples = Some(
                net.samples
                    .iter()
                    .map(|s| (s.name.clone(), s.times.clone(), s.values.clone()))
                    .collect(),
            );
        }
        NetEdit::AddReadout { name, units } => {
            check_ids(net, units)?; // Q4: dangling readout units refused
            net.add_readout(name, units.clone());
            d.readouts = Some(
                net.readouts
                    .iter()
                    .map(|r| (r.name.clone(), r.units.clone(), r.weights.clone()))
                    .collect(),
            );
        }
        NetEdit::AddToGroup { name, ids } => {
            check_ids(net, ids)?;
            let g0 = net.groups.len();
            let grid_before = grids_now(net);
            let gi = net.add_group(name);
            if gi as usize == g0 {
                d.groups_added.push((gi as u32, name.clone(), Vec::new()));
            }
            {
                let g = &mut net.groups[gi];
                // O(ids × members) contains() → box-select on a big
                // net stalls the worker; dedupe with a set.
                let have: HashSet<u32> = g.members.iter().copied().collect();
                for id in ids {
                    if !have.contains(id) {
                        g.members.push(*id);
                    }
                }
                d.groups_changed
                    .push((gi as u32, name.clone(), g.members.clone()));
            }
            net.invalidate_grids();
            grid_delta_membership(net, &grid_before, &mut d);
        }
        NetEdit::CreateConnections {
            group,
            scaling,
            lambda,
            w,
        } => {
            let members = group_members(net, group)?;
            let s0 = net.synapses.len();
            // The checked builder polls the worker's abort flag —
            // the same Stop the Rhai `create_connections` honoured.
            let stopped = net_format::builders::create_connections_checked(
                net, group, *scaling, *lambda, *w, rng, cancel,
            );
            for s in &net.synapses[s0..] {
                d.syn_added.push(SynRow {
                    src: s.src,
                    tgt: s.tgt,
                    delay_steps: steps(s.delay_ms),
                });
            }
            for &m in &members {
                d.disp_changed.push((m, disp_list(net, m, dt)));
            }
            if stopped {
                return Ok(Applied::Stopped(d));
            }
        }
        NetEdit::SetSynapseWeights { group, mean, sd } => {
            group_members(net, group)?;
            net_format::builders::set_synapse_weights(net, group, *mean, *sd, rng);
            // Weights are sim-side state — no wire field carries them;
            // the otherwise-empty delta still bumps the client's version.
        }
        NetEdit::SetSynTauDist { src, dst, mean, sd } => {
            group_members(net, src)?;
            group_members(net, dst)?;
            net_format::builders::set_synapse_tau_dist(net, src, dst, *mean, *sd, rng);
            // tau_ms is sim-side too — version-bump delta only.
        }
        NetEdit::SetExcInh { group, frac } => {
            let members = group_members(net, group)?;
            net_format::builders::set_exc_inh(net, group, *frac, rng);
            for m in members {
                d.neurons_changed.push(row(net, m));
            }
        }
        NetEdit::ScriptAdd { name, src } => {
            if let Some(s) = net.scripts.iter_mut().find(|(nm, _)| nm == name) {
                s.1 = src.clone();
            } else {
                net.scripts.push((name.clone(), src.clone()));
            }
            d.scripts = Some(net.scripts.clone());
        }
    }
    Ok(Applied::Delta(d))
}

/// `net.inputs` → wire rows (delta `inputs` replacements + full Geo).
fn inputs_proto(net: &Network) -> Vec<Input> {
    net.inputs
        .iter()
        .map(|i| Input {
            name: i.name.clone(),
            pos: i.pos,
            targets: i.targets.iter().map(|t| t.0).collect(),
        })
        .collect()
}

/// Group members, or the same refusal the Rhai `group_or_err` produced.
fn group_members(net: &Network, name: &str) -> Result<Vec<u32>, String> {
    net.group(name)
        .map(|g| g.members.clone())
        .ok_or_else(|| format!("unknown group \"{name}\""))
}

fn check_ids(net: &Network, ids: &[u32]) -> Result<(), String> {
    for &id in ids {
        if id as usize >= net.neurons.len() {
            return Err(format!("unknown neuron id {id}"));
        }
    }
    Ok(())
}

/// After grid-invalidating mutations: grids still valid → `grids_changed`;
/// previously-valid group indices now missing → `grids_removed`.
fn grid_delta_membership(net: &Network, before: &[usize], d: &mut GeoDelta) {
    let now = grid_snaps(net);
    let now_idx: HashSet<usize> = now.iter().map(|g| g.group).collect();
    for &b in before {
        if !now_idx.contains(&b) {
            d.grids_removed.push(b as u32);
        }
    }
    d.grids_changed = now;
}

/// First-K display synapses for one source, same shape geo_from_net emits.
fn disp_list(net: &Network, src: u32, dt: f32) -> Vec<(u32, u16)> {
    let mut out = Vec::with_capacity(GEO_EDGES_PER_NEURON);
    for s in &net.synapses {
        if s.src == src {
            if out.len() >= GEO_EDGES_PER_NEURON {
                break;
            }
            out.push((s.tgt, (s.delay_ms / dt.max(1e-6)).round().max(1.0) as u16));
        }
    }
    out
}

/// `disp_list` for a set of sources in one synapse pass — batch moves
/// would otherwise rescan the synapse table per source.
fn disp_lists(net: &Network, srcs: &HashSet<u32>, dt: f32) -> Vec<(u32, Vec<(u32, u16)>)> {
    let mut map: std::collections::HashMap<u32, Vec<(u32, u16)>> =
        srcs.iter().map(|&s| (s, Vec::new())).collect();
    let dtm = dt.max(1e-6);
    for s in &net.synapses {
        if let Some(v) = map.get_mut(&s.src) {
            if v.len() < GEO_EDGES_PER_NEURON {
                v.push((s.tgt, (s.delay_ms / dtm).round().max(1.0) as u16));
            }
        }
    }
    let mut out: Vec<(u32, Vec<(u32, u16)>)> =
        map.into_iter().filter(|(_, v)| !v.is_empty()).collect();
    out.sort_by_key(|(src, _)| *src);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small net snapshot for stack-shape tests.
    fn snap(tag: u32) -> UndoSnap {
        let mut net = Network::new();
        net.add_neuron([tag as f32; 3], net_format::NeuronKind::Excitatory);
        UndoSnap {
            net,
            rng: net_format::Rng::new(tag as u64),
            gen: tag as u64,
        }
    }

    /// History is bounded in steps AND bytes: past either, the oldest
    /// entries are evicted so memory can't grow with the session length.
    /// Sabotage: removing the pop_front loop (or the `> 1` guard) leaves
    /// either 70 entries or an empty stack — both asserted against.
    #[test]
    fn undo_stack_is_bounded() {
        let mut s: std::collections::VecDeque<UndoEntry> = Default::default();
        for i in 0..70 {
            push_step(
                &mut s,
                UndoEntry {
                    label: format!("e{i}"),
                    snap: snap(i),
                },
            );
        }
        assert_eq!(s.len(), UNDO_MAX_STEPS, "step cap evicts the oldest");
        assert_eq!(s.front().unwrap().label, "e6");
        assert_eq!(s.back().unwrap().label, "e69");
        // Byte cap: sized so three entries overfill — evicts down to the
        // newest one; a lone oversized entry is kept regardless (a step
        // is never dropped while it's the only undo point).
        let one = est_net_bytes(&snap(0).net);
        let mut s2: std::collections::VecDeque<UndoEntry> = Default::default();
        for i in 0..4u32 {
            push_step_bounded(
                &mut s2,
                UndoEntry {
                    label: format!("e{i}"),
                    snap: snap(i),
                },
                usize::MAX,
                one * 2 + one / 2,
            );
        }
        assert!(
            s2.len() <= 2,
            "byte cap must evict overflow, got {}",
            s2.len()
        );
        assert_eq!(s2.back().unwrap().label, "e3");
        let mut s3: std::collections::VecDeque<UndoEntry> = Default::default();
        push_step_bounded(
            &mut s3,
            UndoEntry {
                label: "huge".into(),
                snap: snap(0),
            },
            usize::MAX,
            1,
        );
        assert_eq!(s3.len(), 1, "a lone oversized step is still kept");
    }

    /// Menu labels name the action — "Undo move 12 neurons" is the
    /// target shape.
    #[test]
    fn undo_labels_name_actions() {
        assert_eq!(
            edit_label(&NetEdit::MoveNeurons {
                items: (0..12).map(|i| (i, [0.0; 3])).collect()
            }),
            "move 12 neurons"
        );
        assert_eq!(
            edit_label(&NetEdit::DeleteNeurons { ids: vec![3] }),
            "delete 1 neuron"
        );
        assert_eq!(
            eval_label("  net.create_2d_grid(\"g\", 4, 4, 8.0);\n  sim.run(10)"),
            "eval net.create_2d_grid(\"g\", 4, 4, 8.0); sim.run(10)"
        );
    }

    /// 20k-neuron net with `syn` random synapses, no groups.
    fn big_net(n: usize, syn: usize) -> Network {
        let mut net = Network::new();
        for i in 0..n {
            net.add_neuron(
                [
                    (i % 100) as f32,
                    ((i / 100) % 100) as f32,
                    (i / 10_000) as f32,
                ],
                net_format::NeuronKind::Excitatory,
            );
        }
        let mut s = 1u64;
        for _ in 0..syn {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let a = (s >> 33) as u32 % n as u32;
            let b = (s >> 17) as u32 % n as u32;
            if a != b {
                net.connect(a, b, 1.0);
            }
        }
        net
    }

    /// Q13: `AddToGroup` on a crowded group must not do a
    /// `Vec::contains` per incoming id. Times the old pattern inline so
    /// the before/after lives next to the fix.
    #[test]
    fn add_to_group_large_net_is_not_quadratic() {
        let mut net = big_net(100_000, 0);
        net.add_group("g");
        net.groups[0].members = (0..50_000).collect();
        let ids: Vec<u32> = (40_000..90_000).collect();

        // old shape: contains() per id — measured, not run through apply.
        let t = Instant::now();
        for id in &ids {
            if !net.groups[0].members.contains(id) {
                net.groups[0].members.push(*id);
            }
        }
        let old = t.elapsed();

        let mut net = big_net(100_000, 0);
        net.add_group("g");
        net.groups[0].members = (0..50_000).collect();
        let mut rng = net_format::Rng::new(0);
        let t = Instant::now();
        apply_edit(
            &mut net,
            0.1,
            &mut rng,
            &NetEdit::AddToGroup {
                name: "g".into(),
                ids: ids.clone(),
            },
            &|| false,
        )
        .unwrap();
        let new = t.elapsed();
        eprintln!("Q13 AddToGroup 50k ids on 50k members: old {old:?} new {new:?}");
        assert_eq!(net.groups[0].members.len(), 90_000);
        assert!(
            new < old / 10,
            "set-based add should be much faster: {old:?} vs {new:?}"
        );
    }

    /// Q13: dragging many neurons must refresh delays in one synapse
    /// pass, not one pass per neuron.
    #[test]
    fn move_neurons_large_net_is_one_pass() {
        let (n, syn) = (20_000usize, 400_000usize);
        let moved: Vec<u32> = (0..5_000).collect();
        let items: Vec<(u32, [f32; 3])> = moved.iter().map(|&i| (i, [0.0, 0.0, 9.0])).collect();

        // old shape: refresh_delays_of per moved neuron.
        let mut net = big_net(n, syn);
        let t = Instant::now();
        for (id, pos) in &items {
            net.neuron_mut(*id).pos = *pos;
        }
        for id in &moved {
            net.refresh_delays_of(*id);
        }
        let old = t.elapsed();
        let old_delays: Vec<f32> = net.synapses.iter().map(|s| s.delay_ms).collect();

        let mut net = big_net(n, syn);
        let mut rng = net_format::Rng::new(0);
        let t = Instant::now();
        apply_edit(
            &mut net,
            0.1,
            &mut rng,
            &NetEdit::MoveNeurons { items },
            &|| false,
        )
        .unwrap();
        let new = t.elapsed();
        eprintln!("Q13 MoveNeurons 5k of 20k/400k: old {old:?} new {new:?}");
        assert_eq!(
            old_delays,
            net.synapses.iter().map(|s| s.delay_ms).collect::<Vec<_>>()
        );
        assert!(
            new < old / 5,
            "one-pass refresh should be much faster: {old:?} vs {new:?}"
        );
    }
}
