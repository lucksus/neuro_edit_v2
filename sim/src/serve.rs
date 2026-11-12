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
    /// Shared stop flag — UI Stop button / server shutdown.
    pub abort: std::sync::Arc<AtomicBool>,
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
        let mut t = s[..MAX_CONSOLE].to_string();
        while !t.is_char_boundary(t.len()) {
            t.pop();
        }
        t.push_str(&format!(" … ({} chars truncated)", s.len() - t.len()));
        t
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
        abort,
    } = opts;
    let mut env = Env::with_print(net, backend, seed, {
        let e = emit.clone();
        Rc::new(move |s: &str| emit_b(&e, con(s)))
    });
    *env.data_dir.borrow_mut() = data_dir.clone();
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
        let (neth, nver) = (env.net.clone(), env.net_version.clone());
        let cb: ProgressFn = Rc::new(std::cell::RefCell::new(move |sim: &mut Sim| -> bool {
            let mut p = pr.borrow_mut();
            publish_sim(&neth, nver.get(), sim, &mut p, &ep);
            !abp.load(Ordering::Relaxed)
        }));
        env.set_progress(200, cb);
    }

    // Stop also lands inside slow native builders (create_connections,
    // grid builders) — they poll this every few thousand rows.
    {
        let ab = abort.clone();
        env.set_cancel(Rc::new(move || ab.load(Ordering::Relaxed)));
    }

    // Rhai op-level abort: the chunked `sim.run` hook only fires between
    // chunks, so a pure-script loop (`while true {}`) would never see the
    // abort flag. on_progress runs inside the eval loop — a `Some` return
    // terminates the script with ErrorTerminated.
    {
        let ab = abort.clone();
        env.engine.on_progress(move |ops| {
            if ops % 64 == 0 && ab.load(Ordering::Relaxed) {
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
    // Last-emitted Status — changes only.
    let mut last_status: Option<ServerMsg> = None;
    let mut status_busy = false;

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
            };
            if last_status.as_ref() != Some(&msg) {
                last_status = Some(msg.clone());
                emit_ev(&emit, ServerEvent::broadcast(msg));
            }
        }};
    }

    while !quit {
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
            match c {
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
                    match resolve(&env, &path)
                        .and_then(|p| net_format::io::load(&p).map_err(|e| e.to_string()))
                    {
                        Ok(n) => {
                            *env.net.borrow_mut() = n;
                            env.net_gen.set(env.net_gen.get() + 1);
                            bump(&env);
                            emit_b(&emit, con(format!("loaded {path}")));
                        }
                        Err(e) => emit_b(&emit, con(format!("load {path}: {e}"))),
                    }
                }
                ClientMsg::NewNet => {
                    *env.net.borrow_mut() = Network::new();
                    env.net_gen.set(env.net_gen.get() + 1);
                    bump(&env);
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
                        // Scope the net borrow: the Delta arm needs fresh
                        // reads (non_grid, geo_snap) — a match scrutinee
                        // RefMut would live across the arms.
                        let applied = {
                            let mut n = env.net.borrow_mut();
                            let dt = env.cfg.borrow().dt_ms;
                            apply_edit(&mut n, dt, &edit)
                        };
                        match applied {
                            Ok(applied) => {
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
                                            ServerEvent::broadcast(ServerMsg::Geo(Box::new(geo))),
                                        );
                                        emit_ev(
                                            &emit,
                                            ServerEvent::broadcast(ServerMsg::GridTraces {
                                                base_seq: seq,
                                                grids,
                                            }),
                                        );
                                    }
                                    Applied::Delta(mut d) => {
                                        published_geo_version = ver;
                                        d.version = ver;
                                        d.gen = gen;
                                        d.dt_ms = env.cfg.borrow().dt_ms;
                                        d.non_grid = non_grid(&env.net.borrow());
                                        emit_ev(
                                            &emit,
                                            ServerEvent::broadcast(ServerMsg::GeoDelta(Box::new(
                                                d,
                                            ))),
                                        );
                                    }
                                    Applied::Noop => {}
                                }
                            }
                            Err(e) => emit_to(
                                &emit,
                                client,
                                ServerMsg::Error(format!("edit refused: {e}")),
                            ),
                        }
                    }
                }
                ClientMsg::CopySubnet { ids } => {
                    let set: HashSet<u32> = ids.into_iter().collect();
                    let sub = env.net.borrow().subnet(&set);
                    match net_format::io::to_json(&sub) {
                        Ok(txt) => emit_to(&emit, client, ServerMsg::Clipboard(txt)),
                        Err(e) => {
                            emit_to(&emit, client, ServerMsg::Error(format!("copy failed: {e}")))
                        }
                    }
                }
                ClientMsg::SetBackend(b) => {
                    env.set_backend(match b {
                        Backend::Cpu => BackendKind::Cpu,
                        Backend::Cuda => BackendKind::Cuda,
                    });
                    env.invalidate_sim();
                }
                ClientMsg::SetDtMs(dt) => {
                    env.cfg.borrow_mut().dt_ms = dt;
                    env.invalidate_sim();
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
                                })
                            })
                            .collect();
                        entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
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
                        Err(e) => {
                            emit_to(&emit, client, ServerMsg::Error(format!("read {path}: {e}")))
                        }
                    }
                }
                ClientMsg::WriteFile { path, data } => {
                    match resolve(&env, &path)
                        .and_then(|p| std::fs::write(&p, &data).map_err(|e| e.to_string()))
                    {
                        Ok(()) => emit_to(&emit, client, ServerMsg::FileWritten { path }),
                        Err(e) => emit_to(
                            &emit,
                            client,
                            ServerMsg::Error(format!("write {path}: {e}")),
                        ),
                    }
                }
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

    // per-step v recording happens in the sim; (re)apply the watch list
    // every publish so a rebuilt sim gets it back, then drain the samples
    // recorded since the last publish — the plot gets every step, not one
    // point per chunk.
    sim.set_v_watch(&p.watch);
    let hists = sim.drain_v().unwrap_or_default();
    let mut watched_v = Vec::new();
    for (k, &id) in p.watch.iter().enumerate() {
        watched_v.push((id, hists.get(k).cloned().unwrap_or_default()));
    }

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
            watched_v,
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
pub fn geo_from_net(net: &Network, dt: f32) -> Geo {
    let dt = dt.max(1e-6);
    let n = net.neurons.len();

    let mut pos = Vec::with_capacity(n);
    let mut kind = Vec::with_capacity(n);
    let mut names = Vec::with_capacity(n);
    for neu in &net.neurons {
        pos.push(neu.pos);
        kind.push(match neu.kind {
            net_format::NeuronKind::Inhibitory => 1,
            _ => 0,
        });
        names.push(neu.name.clone());
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
        syn,
        syn_delay_steps: delay,
        disp_edges,
        disp_ptr,
        groups: net
            .groups
            .iter()
            .map(|g| (g.name.clone(), g.members.clone()))
            .collect(),
        inputs: net
            .inputs
            .iter()
            .map(|i| Input {
                name: i.name.clone(),
                pos: i.pos,
                targets: i.targets.iter().map(|t| t.0).collect(),
            })
            .collect(),
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
}

/// Apply a `NetEdit` to `net` and describe what changed as a delta.
/// `dt` converts stored ms delays to sim steps.
fn apply_edit(net: &mut Network, dt: f32, e: &NetEdit) -> Result<Applied, String> {
    let mut d = GeoDelta::default();
    let steps = |ms: f32| (ms / dt.max(1e-6)).round().max(1.0) as u16;
    let row = |net: &Network, id: u32| {
        let neu = net.neuron(id);
        NeuronRow {
            id,
            pos: neu.pos,
            kind: match neu.kind {
                net_format::NeuronKind::Inhibitory => 1,
                _ => 0,
            },
            name: neu.name.clone(),
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
            for id in &ids {
                net.refresh_delays_of(*id);
            }
            net.invalidate_grids();
            grid_delta_membership(net, &grid_before, &mut d);
            for (id, _) in items {
                d.neurons_changed.push(row(net, *id));
            }
            // delays of every synapse touching a moved neuron may change;
            // sources whose TARGET moved recompute too — their display
            // edge lists carry the delay.
            let moved: HashSet<u32> = ids.into_iter().collect();
            let mut disp_srcs: HashSet<u32> = moved.iter().copied().collect();
            for (i, s) in net.synapses.iter().enumerate() {
                if moved.contains(&s.src) || moved.contains(&s.tgt) {
                    d.syn_delays.push((i as u32, steps(s.delay_ms)));
                }
                if moved.contains(&s.tgt) {
                    disp_srcs.insert(s.src);
                }
            }
            for id in &disp_srcs {
                let dl = disp_list(net, *id, dt);
                if !dl.is_empty() {
                    d.disp_changed.push((*id, dl));
                }
            }
        }
        NetEdit::SetNeuronName { id, name } => {
            check_ids(net, &[*id])?;
            net.neuron_mut(*id).name = name.clone();
            d.neurons_changed.push(row(net, *id));
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
            net.add_input(name, targets.clone(), *pos);
            d.inputs = Some(
                net.inputs
                    .iter()
                    .map(|i| Input {
                        name: i.name.clone(),
                        pos: i.pos,
                        targets: i.targets.iter().map(|t| t.0).collect(),
                    })
                    .collect(),
            );
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
                for id in ids {
                    if !g.members.contains(id) {
                        g.members.push(*id);
                    }
                }
                d.groups_changed
                    .push((gi as u32, name.clone(), g.members.clone()));
            }
            net.invalidate_grids();
            grid_delta_membership(net, &grid_before, &mut d);
        }
    }
    Ok(Applied::Delta(d))
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
