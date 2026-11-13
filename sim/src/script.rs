//! Rhai scripting API — the embedded engine shared by the headless `sim`
//! binary and the editor's script console.
//!
//! Globals in scope: `net` (the Network), `sim` (the simulator, lazily built
//! on first `sim.run`), `rng`. Run granularity is `sim.run(ms)` /
//! `sim.run_steps(n)` — never one GPU step per script call. Arithmetic-heavy
//! work (readouts, metrics, stream generation) is exposed as Rust functions.

use std::cell::RefCell;
use std::rc::Rc;

use net_format::{builders, Network, NeuronKind, Rng};
use rhai::{Dynamic, Engine};

use crate::readout::{accuracy, auc, fit_ridge, ridge_output, DeltaReadout};
use crate::{BackendKind, Sim, SimConfig};

pub type NetHandle = Rc<RefCell<Network>>;
pub type RngHandle = Rc<RefCell<Rng>>;
/// `sim` in scripts is a lazily-built slot; builds on first use.
pub type SimSlot = Rc<RefCell<Option<Rc<RefCell<Sim>>>>>;
pub type CfgHandle = Rc<RefCell<SimConfig>>;

/// Where `print` goes. Headless: stdout; editor: the console pane.
pub type PrintFn = Rc<dyn Fn(&str)>;

/// Called between run chunks (editor only): publish snapshots, check pause.
/// Return `false` to abort the rest of a chunked run (pause pressed).
pub type ProgressFn = Rc<RefCell<dyn FnMut(&mut Sim) -> bool>>;

/// Polled inside slow native builders (editor's Stop button); `true` =
/// abort now. Defaults to never.
pub type CancelFn = Rc<dyn Fn() -> bool>;

/// `job_progress(#{…})` sink — the interactive worker prints the fields
/// to the console; the server job runner turns them into `JobProgress`
/// messages instead.
pub type JobSinkFn = Box<dyn FnMut(Vec<(String, f64)>)>;

/// Scripting environment: engine + shared handles.
pub struct Env {
    pub engine: Engine,
    pub net: NetHandle,
    pub rng: RngHandle,
    pub cfg: CfgHandle,
    sim: SimSlot,
    pub on_print: PrintFn,
    /// Bumped on every net mutation (any fn that invalidates the sim).
    pub net_version: Rc<std::cell::Cell<u64>>,
    /// Bumped only on wholesale net REPLACEMENT (clear/load/New) — the
    /// editor resets UI state (selection, watches, plots) on this, not on
    /// ordinary edits.
    pub net_gen: Rc<std::cell::Cell<u64>>,
    /// Between-chunks callback; `None` in headless use.
    pub progress: Rc<RefCell<Option<ProgressFn>>>,
    /// Chunk length for progress callbacks, in steps.
    pub progress_chunk: Rc<std::cell::Cell<usize>>,
    /// Live backend choice; write via `set_backend`.
    pub backend: BackendCell,
    /// Stop check for slow builders; editor plugs its abort flag in via
    /// `set_cancel`. Wrapped in `RefCell` so it can be swapped after the
    /// API closures captured it.
    pub cancel: Rc<RefCell<CancelFn>>,
    /// Sandbox root for every file path a script touches (`net.save`,
    /// `net.load`, `samples_*`, `write_to_file`, `read_to_vec`). Set by
    /// the server; `None` = local mode, paths used verbatim.
    pub data_dir: Rc<RefCell<Option<std::path::PathBuf>>>,
    /// `job_progress(#{…})` fields land here (console print locally,
    /// `ServerMsg::JobProgress` under a server job).
    pub job_sink: Rc<RefCell<JobSinkFn>>,
    /// `job_dir()` — the `runs/<id>` path scripts write job outputs to;
    /// `None` outside a job (interactive eval returns "").
    pub job_dir: Rc<RefCell<Option<String>>>,
    /// Server-side sim-size ceiling, consulted before every sim build
    /// (`build_sim`, script `sim.*` handles). All-zeros = unlimited —
    /// the local editor never sets it.
    pub sim_limits: Rc<RefCell<crate::SimLimits>>,
    /// Per-run storage quota for script file writes (`net.save`,
    /// `samples_save`, `write_to_file`, `dataset_gen`); the server sets
    /// it, 0 = unlimited (local mode). Same accounting the protocol
    /// file commands use.
    pub quota_bytes: Rc<RefCell<u64>>,
    /// Drive program saved by the last sim drop, reapplied to the next
    /// build — see `DriveSlot`.
    pub drive: DriveSlot,
}

impl Env {
    pub fn new(net: Network, backend: BackendKind, seed: u64) -> anyhow::Result<Env> {
        let print_fn: PrintFn = Rc::new(|s: &str| println!("{s}"));
        Ok(Self::with_print(net, backend, seed, print_fn))
    }

    pub fn with_print(net: Network, backend: BackendKind, seed: u64, on_print: PrintFn) -> Env {
        let net = Rc::new(RefCell::new(net));
        let rng = Rc::new(RefCell::new(Rng::new(seed)));
        let sim: SimSlot = Rc::new(RefCell::new(None));
        let cfg = Rc::new(RefCell::new(SimConfig {
            seed,
            record_spikes: true,
            ..Default::default()
        }));
        let mut engine = Engine::new();
        // Memory bounds (S5): a `loop { a.push(0.0) }` must die inside
        // the eval, not OOM the process. Generous ceilings — these are
        // per-value bounds, not totals; legit scripts stay far below.
        // No `max_operations`: long simulations are legitimate — Abort
        // (or shutdown) is the time-control mechanism.
        engine.set_max_array_size(10_000_000);
        engine.set_max_string_size(64 * 1024 * 1024);
        engine.set_max_map_size(1_000_000);
        {
            let p = on_print.clone();
            engine.on_print(move |s| p(s));
        }
        let net_version = Rc::new(std::cell::Cell::new(1u64));
        let net_gen = Rc::new(std::cell::Cell::new(1u64));
        let progress: Rc<RefCell<Option<ProgressFn>>> = Rc::new(RefCell::new(None));
        let progress_chunk = Rc::new(std::cell::Cell::new(200usize));
        let backend: BackendCell = Rc::new(std::cell::Cell::new(backend));
        let cancel: Rc<RefCell<CancelFn>> = Rc::new(RefCell::new(Rc::new(|| false)));
        let data_dir: Rc<RefCell<Option<std::path::PathBuf>>> = Rc::new(RefCell::new(None));
        let sim_limits: Rc<RefCell<crate::SimLimits>> =
            Rc::new(RefCell::new(crate::SimLimits::default()));
        let quota_bytes: Rc<RefCell<u64>> = Rc::new(RefCell::new(0));
        // Interactive default: job_progress fields print to the console.
        let job_sink: Rc<RefCell<JobSinkFn>> = {
            let p = on_print.clone();
            Rc::new(RefCell::new(Box::new(move |fields: Vec<(String, f64)>| {
                let s = fields
                    .iter()
                    .map(|(k, v)| format!("{k}={v:.4}"))
                    .collect::<Vec<_>>()
                    .join("  ");
                p(&format!("job: {s}"));
            }) as JobSinkFn))
        };
        let job_dir: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
        let drive: DriveSlot = Rc::new(RefCell::new(None));
        register_api(
            &mut engine,
            net.clone(),
            sim.clone(),
            rng.clone(),
            cfg.clone(),
            backend.clone(),
            net_version.clone(),
            net_gen.clone(),
            progress.clone(),
            progress_chunk.clone(),
            cancel.clone(),
            data_dir.clone(),
            job_sink.clone(),
            job_dir.clone(),
            sim_limits.clone(),
            quota_bytes.clone(),
            drive.clone(),
        );
        Env {
            engine,
            net,
            backend,
            rng,
            cfg,
            sim,
            on_print,
            net_version,
            net_gen,
            progress,
            progress_chunk,
            cancel,
            data_dir,
            job_sink,
            job_dir,
            sim_limits,
            quota_bytes,
            drive,
        }
    }

    /// Resolve a user-supplied path inside `data_dir` when set; `..`
    /// components and absolute paths are refused. `None` = verbatim.
    pub fn resolve_path(&self, p: &str) -> Result<String, String> {
        resolve_in(&self.data_dir.borrow(), p)
    }

    /// Editor mode: `cb` runs between run chunks so the UI can stream state.
    pub fn set_progress(&mut self, chunk_steps: usize, cb: ProgressFn) {
        self.progress_chunk.set(chunk_steps);
        *self.progress.borrow_mut() = Some(cb);
    }

    /// Plug in a stop check polled by the slow native builders.
    pub fn set_cancel(&self, f: CancelFn) {
        *self.cancel.borrow_mut() = f;
    }

    /// Job mode: swap the `job_progress` sink (server → `JobProgress`).
    pub fn set_job_sink(&self, f: JobSinkFn) {
        *self.job_sink.borrow_mut() = f;
    }

    /// Switch the backend scripts build sims on (invalidates any built sim).
    pub fn set_backend(&self, b: BackendKind) {
        self.backend.set(b);
        self.invalidate_sim();
    }

    /// Build (or rebuild) the simulator from the current network + config.
    pub fn build_sim(&self) -> anyhow::Result<Rc<RefCell<Sim>>> {
        let cfg = self.cfg.borrow().clone();
        self.sim_limits
            .borrow()
            .check(&self.net.borrow(), cfg.dt_ms)
            .map_err(anyhow::Error::msg)?;
        let mut s = Sim::new(&self.net.borrow(), cfg, self.backend.get())?;
        apply_drive(&self.drive, &mut s, &self.net);
        let h = Rc::new(RefCell::new(s));
        *self.sim.borrow_mut() = Some(h.clone());
        Ok(h)
    }

    /// The live sim handle, building lazily.
    pub fn sim_handle(&self) -> anyhow::Result<Rc<RefCell<Sim>>> {
        if let Some(h) = self.sim.borrow().as_ref() {
            return Ok(h.clone());
        }
        self.build_sim()
    }

    /// The sim handle only if one is already built — callers that publish
    /// state after an eval use this so a read-only `net.*` eval doesn't
    /// trigger a sim build nobody asked for.
    pub fn existing_sim(&self) -> Option<Rc<RefCell<Sim>>> {
        self.sim.borrow().as_ref().cloned()
    }

    /// Mark the sim stale — stash its drive program, then next `sim.run`
    /// rebuilds from the edited net with the drive re-applied.
    pub fn invalidate_sim(&self) {
        stash_drive(&self.sim, &self.drive, &self.net);
    }

    fn scope(&self) -> rhai::Scope<'_> {
        let mut scope = rhai::Scope::new();
        scope.push("net", self.net.clone());
        scope.push("sim", self.sim.clone());
        scope.push("rng", self.rng.clone());
        // Bare `help` (no parens) evaluates as a variable — bind it to
        // the overview so it prints instead of erroring.
        scope.push("help", help_text(""));
        scope
    }

    pub fn eval_str(&mut self, code: &str) -> Result<Dynamic, String> {
        self.engine
            .eval_with_scope::<Dynamic>(&mut self.scope(), code)
            .map_err(|e| match *e {
                // A user abort shows up as rhai's generic "Script
                // terminated" — say what actually happened.
                rhai::EvalAltResult::ErrorTerminated(_, pos) => match pos.line() {
                    Some(l) => format!("stopped by user at line {l}"),
                    None => "stopped by user".into(),
                },
                _ => e.to_string(),
            })
    }

    /// Evaluate a script file's contents.
    pub fn eval_file(&mut self, path: &str) -> Result<Dynamic, String> {
        let code = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        self.eval_str(&code)
    }

    pub fn time_ms(&self) -> f64 {
        self.sim
            .borrow()
            .as_ref()
            .map(|s| s.borrow().time_ms())
            .unwrap_or(0.0)
    }

    pub fn run_ms(&mut self, ms: f32) -> Result<(), String> {
        if let Ok(h) = self.sim_handle() {
            h.borrow_mut().run_ms(ms).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn spikes_total(&self) -> u64 {
        self.sim
            .borrow()
            .as_ref()
            .map(|s| s.borrow().spikes_total())
            .unwrap_or(0)
    }
}

#[inline]
fn d_f64(d: &Dynamic) -> f64 {
    if d.is_int() {
        d.as_int().unwrap_or(0) as f64
    } else {
        d.as_float().unwrap_or(0.0)
    }
}

#[inline]
fn d_i64(d: &Dynamic) -> i64 {
    if d.is_int() {
        d.as_int().unwrap_or(0)
    } else {
        d.as_float().unwrap_or(0.0) as i64
    }
}

#[inline]
fn dyn_f64(x: f64) -> Dynamic {
    Dynamic::from(x)
}

/// Error for a group-name parameter that doesn't match — silent no-ops on
/// typos were a review complaint.
fn group_or_err(n: &NetHandle, g: &str) -> Result<(), Box<rhai::EvalAltResult>> {
    if n.borrow().group(g).is_some() {
        Ok(())
    } else {
        Err(s_err(format!("unknown group \"{g}\"")))
    }
}

/// Script error shorthand.
fn s_err(s: impl Into<String>) -> Box<rhai::EvalAltResult> {
    s.into().into()
}

/// `help`/`help()`/`help("topic")` — topic slices of the script API,
/// the flat overview on `""`. Kept as a free fn so tests can call it
/// without an engine.
fn help_text(topic: &str) -> String {
    const TOPICS: &[(&str, &str)] = &[
        (
            "net",
            "net: add_neuron(x,y,z)  connect(a,b,w)  clear()  neuron_count()  synapse_count()\n  \
             neuron(name)  group_names()  neurons_of(g)  save(path)  load(path)",
        ),
        (
            "types",
            "types: models()  def_type(name,model,#{p:v},kind)  set_type(g_or_id,name)\n  \
             set_param(id,p,v)  type_name(id)  set_izh(id,a,b,c,d)  kind(id)  set_kind(id,k)",
        ),
        (
            "grid",
            "grid: create_2d_grid(g,w,h,d)  create_3d_grid(g,w,h,d,sp)\n  \
             create_connections(g,scaling,lambda,w)  set_exc_inh(g,frac)",
        ),
        (
            "syn",
            "syn: set_synapse_weights(g,mean,sd)  set_syn_tau(g1,g2,tau)\n  \
             set_syn_tau_dist(g1,g2,mean,sd)  set_syn_tau_one(src,tgt,tau)  syn_tau(src,tgt)",
        ),
        (
            "input",
            "input: add_input(name,[[n,w],..])  input_count()\n  \
             sim.set_input(name,x)  sim.play_input(name,arr)  sim.set_external(id,amps)",
        ),
        (
            "sim",
            "sim: run(ms)  run_steps(n)  time()  reset()  rate()  spikes()\n  \
             trace()  trace_of(ids)  v(id)  set_noise(hz,kick)\n  \
             rebuild()  set_dt(dt)  set_int_weights(on)",
        ),
        (
            "readout",
            "readout: add_readout(name,units)  set_readout_weights(name,w)  readout_weights(name)\n  \
             delta_new(n)  r.learn(trace,target,rate)  r.output(trace)\n  \
             fit_ridge(x_flat,T,F,y,lambda)  ridge_output(w,x)  accuracy(s,y)  auc(s,y)",
        ),
        (
            "io",
            "io: net.save(path)  net.load(path)  samples_write(n,t,v)\n  \
             samples_save(n,p)  samples_load(n,p)  script_add(n,code)  script_remove(n)",
        ),
        (
            "streams",
            "streams: rng_stream(seed,n_src,p) .window(steps,label,pairs,strength)",
        ),
        ("misc", "misc: rand(a,b)  gauss(m,sd)  seed(s)  print(x)"),
        (
            "examples",
            "examples: first_spikes.rhai  grid_wave.rhai  lsm_readout.rhai —\n  \
             runnable from the console's examples picker",
        ),
    ];
    let t = topic.trim().to_lowercase();
    let aliases: &[&str] = match t.as_str() {
        "synapse" | "synapses" => &["syn"],
        "inputs" | "stim" | "stimulus" => &["input"],
        "readouts" | "train" | "training" => &["readout"],
        "file" | "files" | "save" | "load" | "samples" | "scripts" => &["io"],
        "stream" | "noise" | "rng" => &["streams"],
        "example" | "demo" | "demos" => &["examples"],
        "network" | "neuron" | "neurons" => &["net"],
        "type" | "model" | "models" | "params" | "param" => &["types"],
        "grids" | "2d" | "3d" => &["grid"],
        "simulate" | "run" | "v" | "u" => &["sim"],
        _ => &[t.as_str()],
    };
    for want in aliases {
        for (name, sec) in TOPICS {
            if name == want {
                return (*sec).to_string();
            }
        }
    }
    if t.is_empty() || t == "all" {
        let mut s = format!(
            "API — help(\"topic\"): {}\n",
            TOPICS.iter().map(|t| t.0).collect::<Vec<_>>().join(" ")
        );
        for (_, sec) in TOPICS {
            s.push_str(sec);
            s.push('\n');
        }
        return s;
    }
    format!(
        "unknown topic '{topic}' — try help() for the overview or a topic: {}",
        TOPICS.iter().map(|t| t.0).collect::<Vec<_>>().join(" ")
    )
}

/// Bounds-check a script-supplied neuron index → u32 or a script error.
fn nid(id: i64, n: usize) -> Result<u32, Box<rhai::EvalAltResult>> {
    if id < 0 || id as usize >= n {
        return Err(s_err(format!(
            "neuron index {id} out of range ({n} neurons)"
        )));
    }
    Ok(id as u32)
}

/// Shared backend cell — `Cmd::SetBackend` writes here and the Rhai
/// `sim.*` closures read it live, so a backend switch is honoured by the
/// next script-run sim build (previously the choice was captured once at
/// `Env` creation and scripts kept using the old backend).
pub type BackendCell = Rc<std::cell::Cell<BackendKind>>;

/// The drive program saved by the last sim drop — inputs keyed by name,
/// ext by neuron (`stim::DriveState`). Reapplied to the next built sim so
/// a backend switch / dt change / net edit doesn't silently un-drive the
/// network (2026-11-13, Nico's report: toggling CUDA→CPU killed all
/// excitation because the stimulus only lived inside the dropped sim).
pub type DriveSlot = Rc<RefCell<Option<crate::stim::DriveState>>>;

/// Stash `slot`'s sim drive into `drive` before dropping it — every
/// sim-drop path funnels here or through `Env::invalidate_sim`.
fn stash_drive(slot: &SimSlot, drive: &DriveSlot, net: &NetHandle) {
    if let Some(h) = slot.borrow_mut().take() {
        *drive.borrow_mut() = Some(crate::stim::DriveState::capture(&h.borrow(), &net.borrow()));
    }
}

/// Re-apply a stashed drive onto a just-built sim, if one is waiting.
fn apply_drive(drive: &DriveSlot, sim: &mut Sim, net: &NetHandle) {
    if let Some(d) = drive.borrow_mut().take() {
        d.apply(sim, &net.borrow());
    }
}

/// Build or fetch the sim inside the slot (mirrors `Env::sim_handle`).
fn ensure_sim(
    slot: &SimSlot,
    net: &NetHandle,
    cfg: &CfgHandle,
    kind: &BackendCell,
    limits: &Rc<RefCell<crate::SimLimits>>,
    drive: &DriveSlot,
) -> Result<Rc<RefCell<Sim>>, Box<rhai::EvalAltResult>> {
    if let Some(h) = slot.borrow().as_ref() {
        return Ok(h.clone());
    }
    let c = cfg.borrow().clone();
    limits
        .borrow()
        .check(&net.borrow(), c.dt_ms)
        .map_err(s_err)?;
    let mut s = Sim::new(&net.borrow(), c, kind.get())
        .map_err(|e| -> Box<rhai::EvalAltResult> { e.to_string().into() })?;
    apply_drive(drive, &mut s, net);
    let h = Rc::new(RefCell::new(s));
    *slot.borrow_mut() = Some(h.clone());
    Ok(h)
}

/// Net mutated: stash the drive, drop the stale sim, bump the version
/// (editor republish).
fn invalidate(s: &SimSlot, n: &NetHandle, d: &DriveSlot, ver: &Rc<std::cell::Cell<u64>>) {
    stash_drive(s, d, n);
    ver.set(ver.get() + 1);
}

/// Net changed in a way the sim does not see (readout weights, samples,
/// script list): bump the version only, so the editor republishes geometry.
fn bump(ver: &Rc<std::cell::Cell<u64>>) {
    ver.set(ver.get() + 1);
}

/// Step `h` `steps` times, calling `progress` between chunks when set.
fn run_chunked(
    h: &Rc<RefCell<Sim>>,
    mut steps: usize,
    progress: &Rc<RefCell<Option<ProgressFn>>>,
    chunk: &Rc<std::cell::Cell<usize>>,
) -> Result<(), String> {
    let mut p = progress.borrow_mut();
    match p.as_mut() {
        Some(cb) => {
            let n_chunk = chunk.get().max(1);
            while steps > 0 {
                let n = steps.min(n_chunk);
                h.borrow_mut().run_steps(n).map_err(|e| e.to_string())?;
                steps -= n;
                if !(*cb.borrow_mut())(&mut h.borrow_mut()) {
                    break;
                }
            }
            Ok(())
        }
        None => h.borrow_mut().run_steps(steps).map_err(|e| e.to_string()),
    }
}

/// Shared resolver: a path relative to `dir`, refusing `..` and
/// absolute paths. `None` = local mode, verbatim. The result is
/// canonicalized through the deepest existing ancestor so a symlink
/// anywhere in the chain can't lead outside.
fn resolve_in(dir: &Option<std::path::PathBuf>, p: &str) -> Result<String, String> {
    match dir {
        None => Ok(p.to_string()),
        Some(dir) => {
            let rel = std::path::Path::new(p);
            if rel.is_absolute()
                || rel
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                return Err(format!("path {p:?} escapes --data-dir"));
            }
            let root = dir
                .canonicalize()
                .map_err(|e| format!("--data-dir {}: {e}", dir.display()))?;
            let joined = root.join(rel);
            // Walk up to the deepest existing ancestor and canonicalize
            // it — a symlinked component (dir or file) resolves to its
            // target, which must still sit under the root.
            let mut probe: &std::path::Path = &joined;
            let mut tail: Vec<&std::ffi::OsStr> = Vec::new();
            let mut out = loop {
                match probe.canonicalize() {
                    Ok(c) => {
                        if !c.starts_with(&root) {
                            return Err(format!("path {p:?} escapes --data-dir"));
                        }
                        break c;
                    }
                    Err(_) => match probe.file_name() {
                        Some(name) => {
                            tail.push(name);
                            probe = probe.parent().unwrap_or(root.as_path());
                        }
                        None => break joined.clone(),
                    },
                }
            };
            for name in tail.iter().rev() {
                out.push(name);
            }
            Ok(out.to_string_lossy().into_owned())
        }
    }
}

/// Script-facing resolver — escape attempts are script errors, not silent.
fn rdir(
    dd: &Rc<RefCell<Option<std::path::PathBuf>>>,
    p: &str,
) -> Result<String, Box<rhai::EvalAltResult>> {
    resolve_in(&dd.borrow(), p).map_err(s_err)
}

/// Quota gate for script-side writes: `usage(data_dir) - existing(path)
/// + bytes` must fit `quota` (0 = unlimited). Mirrors the protocol-side
/// `quota_check` in serve.rs — same accounting, same message, so a user
/// can't reach around the wire limits through `net.save` and friends.
pub fn quota_fits(
    dir: &Option<std::path::PathBuf>,
    resolved_path: &str,
    bytes: u64,
    quota: u64,
) -> Result<(), String> {
    if quota == 0 {
        return Ok(());
    }
    let Some(d) = dir else { return Ok(()) };
    let existing = std::fs::metadata(resolved_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let used = crate::serve::dir_size(d).saturating_sub(existing);
    if used.saturating_add(bytes) > quota {
        return Err(format!(
            "over quota: {used} + {bytes} B would exceed the {quota} B limit"
        ));
    }
    Ok(())
}

/// `rdir` + `quota_fits` + `std::fs::write` — the one helper every
/// script file write goes through (S7).
fn checked_write(
    dd: &Rc<RefCell<Option<std::path::PathBuf>>>,
    quota: &Rc<RefCell<u64>>,
    path: &str,
    bytes: &[u8],
) -> Result<String, Box<rhai::EvalAltResult>> {
    let p = rdir(dd, path)?;
    quota_fits(&dd.borrow(), &p, bytes.len() as u64, *quota.borrow()).map_err(s_err)?;
    std::fs::write(&p, bytes).map_err(|e| s_err(format!("write {path}: {e}")))?;
    Ok(p)
}

/// `import` resolver confined to a server's `--data-dir`: the path gets
/// the same `resolve_in` rules as every other script file fn (no `..`,
/// no absolute, no symlink escapes), then an inner `FileModuleResolver`
/// rooted at the dir does the loading. Nested imports resolve
/// dir-relative, not source-relative.
pub struct ConfinedResolver {
    dir: std::path::PathBuf,
    inner: rhai::module_resolvers::FileModuleResolver,
}

impl ConfinedResolver {
    pub fn new(dir: std::path::PathBuf) -> Self {
        Self {
            inner: rhai::module_resolvers::FileModuleResolver::new_with_path(dir.clone()),
            dir,
        }
    }
    /// What the inner resolver would open: `dir/path(.rhai)` — refuse
    /// unless `resolve_in` okays it.
    fn confine(&self, path: &str) -> Result<(), Box<rhai::EvalAltResult>> {
        let mut logical = std::path::PathBuf::from(path);
        logical.set_extension("rhai");
        resolve_in(&Some(self.dir.clone()), &logical.to_string_lossy()).map_err(|e| {
            Box::new(rhai::EvalAltResult::ErrorSystem(
                e.clone(),
                std::io::Error::new(std::io::ErrorKind::PermissionDenied, e).into(),
            ))
        })?;
        Ok(())
    }
}

impl rhai::ModuleResolver for ConfinedResolver {
    fn resolve(
        &self,
        engine: &Engine,
        source: Option<&str>,
        path: &str,
        pos: rhai::Position,
    ) -> Result<rhai::Shared<rhai::Module>, Box<rhai::EvalAltResult>> {
        self.confine(path)?;
        self.inner.resolve(engine, source, path, pos)
    }
}

/// Register the whole API on an engine, capturing the shared handles.
#[allow(clippy::too_many_arguments)]
pub fn register_api(
    engine: &mut Engine,
    net: NetHandle,
    sim: SimSlot,
    rng: RngHandle,
    cfg: CfgHandle,
    backend: BackendCell,
    net_version: Rc<std::cell::Cell<u64>>,
    net_gen: Rc<std::cell::Cell<u64>>,
    progress: Rc<RefCell<Option<ProgressFn>>>,
    progress_chunk: Rc<std::cell::Cell<usize>>,
    cancel: Rc<RefCell<CancelFn>>,
    data_dir: Rc<RefCell<Option<std::path::PathBuf>>>,
    job_sink: Rc<RefCell<JobSinkFn>>,
    job_dir: Rc<RefCell<Option<String>>>,
    sim_limits: Rc<RefCell<crate::SimLimits>>,
    quota_bytes: Rc<RefCell<u64>>,
    drive: DriveSlot,
) {
    engine
        .register_type_with_name::<NetHandle>("Net")
        .register_type_with_name::<SimSlot>("Sim")
        .register_type_with_name::<RngHandle>("Rng")
        .register_type_with_name::<RngStream>("RngStream")
        .register_type_with_name::<DeltaReadout>("DeltaReadout")
        .register_type_with_name::<crate::dataset::Dataset>("Dataset");

    // ---- help: `help`, `help()` and `help("topic")` all work ----
    engine.register_fn("help", || help_text(""));
    engine.register_fn("help", |topic: &str| help_text(topic));
    engine.register_fn("help", |topic: rhai::INT| help_text(&topic.to_string()));

    // ---- net methods ----
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "add_neuron",
            move |n: &mut NetHandle, x: Dynamic, y: Dynamic, z: Dynamic| -> rhai::INT {
                invalidate(&s, n, &dr, &v);
                let p = [d_f64(&x) as f32, d_f64(&y) as f32, d_f64(&z) as f32];
                n.borrow_mut().add_neuron(p, NeuronKind::Excitatory) as rhai::INT
            },
        );
    }
    {
        let (s, v, g, dr) = (
            sim.clone(),
            net_version.clone(),
            net_gen.clone(),
            drive.clone(),
        );
        engine.register_fn("clear", move |n: &mut NetHandle| {
            *n.borrow_mut() = Network::new();
            g.set(g.get() + 1);
            invalidate(&s, n, &dr, &v);
        });
    }
    engine.register_fn("neuron_count", |n: &mut NetHandle| -> rhai::INT {
        n.borrow().neurons.len() as rhai::INT
    });
    engine.register_fn("synapse_count", |n: &mut NetHandle| -> rhai::INT {
        n.borrow().synapses.len() as rhai::INT
    });
    engine.register_fn("input_count", |n: &mut NetHandle| -> rhai::INT {
        n.borrow().inputs.len() as rhai::INT
    });
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "connect",
            move |n: &mut NetHandle,
                  a: rhai::INT,
                  b: rhai::INT,
                  w: Dynamic|
                  -> Result<rhai::INT, Box<rhai::EvalAltResult>> {
                let len = n.borrow().neurons.len();
                let (a, b) = (nid(a, len)?, nid(b, len)?);
                if a == b {
                    return Err(format!("self-synapse on neuron {a}").into());
                }
                invalidate(&s, n, &dr, &v);
                Ok(n.borrow_mut().connect(a, b, d_f64(&w) as f32) as rhai::INT)
            },
        );
    }
    {
        let (s, v, c, dr) = (
            sim.clone(),
            net_version.clone(),
            cancel.clone(),
            drive.clone(),
        );
        engine.register_fn(
            "create_2d_grid",
            move |n: &mut NetHandle,
                  g: &str,
                  w: rhai::INT,
                  h: rhai::INT,
                  d: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                net_format::check_grid_dims([w, h, 1]).map_err(s_err)?;
                invalidate(&s, n, &dr, &v);
                let cancelled = builders::create_2d_grid_checked(
                    &mut n.borrow_mut(),
                    g,
                    w as usize,
                    h as usize,
                    d_f64(&d) as f32,
                    [0.0; 3],
                    &|| (c.borrow())(),
                );
                if cancelled {
                    Err(s_err("stopped by user"))
                } else {
                    Ok(())
                }
            },
        );
    }
    {
        let (s, v, c, dr) = (
            sim.clone(),
            net_version.clone(),
            cancel.clone(),
            drive.clone(),
        );
        engine.register_fn(
            "create_3d_grid",
            move |n: &mut NetHandle,
                  g: &str,
                  w: rhai::INT,
                  h: rhai::INT,
                  dep: rhai::INT,
                  d: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                net_format::check_grid_dims([w, h, dep]).map_err(s_err)?;
                invalidate(&s, n, &dr, &v);
                let cancelled = builders::create_3d_grid_checked(
                    &mut n.borrow_mut(),
                    g,
                    w as usize,
                    h as usize,
                    dep as usize,
                    d_f64(&d) as f32,
                    [0.0; 3],
                    &|| (c.borrow())(),
                );
                if cancelled {
                    Err(s_err("stopped by user"))
                } else {
                    Ok(())
                }
            },
        );
    }
    {
        let (r, s, v, c, dr) = (
            rng.clone(),
            sim.clone(),
            net_version.clone(),
            cancel.clone(),
            drive.clone(),
        );
        engine.register_fn(
            "create_connections",
            move |n: &mut NetHandle,
                  g: &str,
                  scaling: Dynamic,
                  lambda: Dynamic,
                  w: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                group_or_err(n, g)?;
                invalidate(&s, n, &dr, &v);
                let cancelled = builders::create_connections_checked(
                    &mut n.borrow_mut(),
                    g,
                    d_f64(&scaling),
                    d_f64(&lambda),
                    d_f64(&w) as f32,
                    &mut r.borrow_mut(),
                    &|| (c.borrow())(),
                );
                if cancelled {
                    Err(s_err("stopped by user"))
                } else {
                    Ok(())
                }
            },
        );
    }
    {
        let (r, s, v, dr) = (rng.clone(), sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_synapse_weights",
            move |n: &mut NetHandle,
                  g: &str,
                  mean: Dynamic,
                  sd: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                group_or_err(n, g)?;
                invalidate(&s, n, &dr, &v);
                builders::set_synapse_weights(
                    &mut n.borrow_mut(),
                    g,
                    d_f64(&mean),
                    d_f64(&sd),
                    &mut r.borrow_mut(),
                );
                Ok(())
            },
        );
    }
    // ---- per-synapse tau ----
    // tau semantics: >0 gives that synapse its own exponential current
    // trace; 0 returns it to the merged global synapse_tau path.
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_syn_tau",
            move |n: &mut NetHandle,
                  g1: &str,
                  g2: &str,
                  tau: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                group_or_err(n, g1)?;
                group_or_err(n, g2)?;
                invalidate(&s, n, &dr, &v);
                builders::set_synapse_tau(&mut n.borrow_mut(), g1, g2, d_f64(&tau));
                Ok(())
            },
        );
    }
    {
        let (r, s, v, dr) = (rng.clone(), sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_syn_tau_dist",
            move |n: &mut NetHandle,
                  g1: &str,
                  g2: &str,
                  mean: Dynamic,
                  sd: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                group_or_err(n, g1)?;
                group_or_err(n, g2)?;
                invalidate(&s, n, &dr, &v);
                builders::set_synapse_tau_dist(
                    &mut n.borrow_mut(),
                    g1,
                    g2,
                    d_f64(&mean),
                    d_f64(&sd),
                    &mut r.borrow_mut(),
                );
                Ok(())
            },
        );
    }
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_syn_tau_one",
            move |n: &mut NetHandle,
                  src: rhai::INT,
                  tgt: rhai::INT,
                  tau: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let len = n.borrow().neurons.len();
                // Validate before casting — a negative id would wrap.
                let (src, tgt) = (nid(src, len)?, nid(tgt, len)?);
                invalidate(&s, n, &dr, &v);
                let mut nn = n.borrow_mut();
                for syn in &mut nn.synapses {
                    if syn.src == src && syn.tgt == tgt {
                        syn.tau_ms = (d_f64(&tau) as f32).max(0.0);
                    }
                }
                Ok(())
            },
        );
    }
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_syn_tau_idx",
            move |n: &mut NetHandle, idx: rhai::INT, tau: Dynamic| {
                invalidate(&s, n, &dr, &v);
                if let Some(syn) = n.borrow_mut().synapses.get_mut(idx as usize) {
                    syn.tau_ms = (d_f64(&tau) as f32).max(0.0);
                }
            },
        );
    }
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_syn_tau_inh",
            move |n: &mut NetHandle,
                  g: &str,
                  tau: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                group_or_err(n, g)?;
                invalidate(&s, n, &dr, &v);
                builders::set_synapse_tau_on_inhibitory(&mut n.borrow_mut(), g, d_f64(&tau));
                Ok(())
            },
        );
    }
    engine.register_fn(
        "syn_tau",
        |n: &mut NetHandle, src: rhai::INT, tgt: rhai::INT| {
            let nn = n.borrow();
            let (src, tgt) = (src as u32, tgt as u32);
            nn.synapses
                .iter()
                .find(|s| s.src == src && s.tgt == tgt)
                .map(|s| s.tau_ms)
                .unwrap_or(0.0)
        },
    );
    {
        let (r, s, v, dr) = (rng.clone(), sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_exc_inh",
            move |n: &mut NetHandle,
                  g: &str,
                  frac: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                group_or_err(n, g)?;
                invalidate(&s, n, &dr, &v);
                builders::set_exc_inh(&mut n.borrow_mut(), g, d_f64(&frac), &mut r.borrow_mut());
                Ok(())
            },
        );
    }
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_izh",
            move |n: &mut NetHandle,
                  id: rhai::INT,
                  a: Dynamic,
                  b: Dynamic,
                  c: Dynamic,
                  d: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let id = nid(id, n.borrow().neurons.len())?;
                invalidate(&s, n, &dr, &v);
                n.borrow_mut().set_izh(
                    id,
                    d_f64(&a) as f32,
                    d_f64(&b) as f32,
                    d_f64(&c) as f32,
                    d_f64(&d) as f32,
                );
                Ok(())
            },
        );
    }
    // ---- neuron types / models (review 1.3e) -----------------------
    // `models()` lists the registry; `net.def_type` adds a type row,
    // `net.set_type` points a group or neuron at one, `net.set_param`
    // sets one parameter of a neuron's override row (set_izh's generic
    // form), `net.type_name` reads back the type name (the review's
    // `type_of` — Rhai reserves that symbol for its own built-in).
    engine.register_fn("models", || -> rhai::Array {
        crate::model::MODELS
            .iter()
            .map(|m| {
                let mut mm = rhai::Map::new();
                mm.insert("id".into(), m.id().into());
                mm.insert("label".into(), m.label().into());
                mm.insert(
                    "params".into(),
                    m.params()
                        .iter()
                        .map(|p| {
                            let mut pm = rhai::Map::new();
                            pm.insert("name".into(), p.name.into());
                            pm.insert("unit".into(), p.unit.into());
                            pm.insert("default".into(), Dynamic::from(p.default));
                            pm.insert("min".into(), Dynamic::from(p.min));
                            pm.insert("max".into(), Dynamic::from(p.max));
                            pm.insert("doc".into(), p.doc.into());
                            Dynamic::from(pm)
                        })
                        .collect::<rhai::Array>()
                        .into(),
                );
                mm.insert(
                    "state".into(),
                    m.state()
                        .iter()
                        .map(|s| {
                            let mut sm = rhai::Map::new();
                            sm.insert("name".into(), s.name.into());
                            sm.insert("unit".into(), s.unit.into());
                            Dynamic::from(sm)
                        })
                        .collect::<rhai::Array>()
                        .into(),
                );
                mm.insert(
                    "presets".into(),
                    m.presets()
                        .iter()
                        .map(|(n, p)| {
                            let mut pm = rhai::Map::new();
                            pm.insert("name".into(), (*n).into());
                            pm.insert(
                                "params".into(),
                                p.iter()
                                    .map(|&v| Dynamic::from(v))
                                    .collect::<rhai::Array>()
                                    .into(),
                            );
                            Dynamic::from(pm)
                        })
                        .collect::<rhai::Array>()
                        .into(),
                );
                Dynamic::from(mm)
            })
            .collect()
    });
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        // def_type(name, model, #{param: v,…} | [row], kind?)
        let def_type = move |n: &mut NetHandle,
                             name: &str,
                             model: &str,
                             spec: Dynamic,
                             kind: &str|
              -> Result<(), Box<rhai::EvalAltResult>> {
            let kind = match kind {
                "exc" | "excitatory" => net_format::NeuronKind::Excitatory,
                "inh" | "inhibitory" => net_format::NeuronKind::Inhibitory,
                "regular" => net_format::NeuronKind::Regular,
                other => {
                    return Err(s_err(format!(
                        "kind must be \"exc\", \"inh\" or \"regular\", got {other:?}"
                    )))
                }
            };
            let (_, m) = crate::model::by_id(model)
                .ok_or_else(|| s_err(format!("unknown model {model:?}")))?;
            // Full row: a map fills descriptor defaults for the kind,
            // an array is taken verbatim (checked against the width).
            let mut p = m.default_for(kind).to_vec();
            if let Some(map) = spec.clone().try_cast::<rhai::Map>() {
                for (i, pd) in m.params().iter().enumerate() {
                    if let Some(v) = map.get(pd.name) {
                        p[i] = d_f64(v) as f32;
                    }
                }
            } else if let Some(arr) = spec.clone().try_cast::<rhai::Array>() {
                if arr.len() != m.params().len() {
                    return Err(s_err(format!(
                        "def_type: {} params, model {} wants {}",
                        arr.len(),
                        model,
                        m.params().len()
                    )));
                }
                for (i, v) in arr.iter().enumerate() {
                    p[i] = d_f64(v) as f32;
                }
            } else {
                return Err(s_err(
                    "def_type: params must be a #{name: v} map or an array",
                ));
            }
            invalidate(&s, n, &dr, &v);
            n.borrow_mut().upsert_type(net_format::NeuronType {
                name: name.into(),
                model: model.into(),
                params: p,
                kind,
                color: None,
            });
            Ok(())
        };
        {
            let f = def_type.clone();
            engine.register_fn(
                "def_type",
                move |n: &mut NetHandle,
                      name: &str,
                      model: &str,
                      spec: Dynamic|
                      -> Result<(), Box<rhai::EvalAltResult>> {
                    f(n, name, model, spec, "exc")
                },
            );
        }
        engine.register_fn(
            "def_type",
            move |n: &mut NetHandle,
                  name: &str,
                  model: &str,
                  spec: Dynamic,
                  kind: &str|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                def_type(n, name, model, spec, kind)
            },
        );
    }
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_type",
            move |n: &mut NetHandle,
                  what: Dynamic,
                  ty: &str|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let ids: Vec<u32> = if let Some(gname) = what.clone().try_cast::<String>() {
                    n.borrow()
                        .group(&gname)
                        .ok_or_else(|| s_err(format!("unknown group {gname:?}")))?
                        .members
                        .clone()
                } else {
                    vec![nid(
                        what.as_int().map_err(|_| {
                            s_err("set_type: first arg must be a group name or a neuron id")
                        })?,
                        n.borrow().neurons.len(),
                    )?]
                };
                let t = n
                    .borrow()
                    .type_index(ty)
                    .ok_or_else(|| s_err(format!("unknown type {ty:?}")))?;
                invalidate(&s, n, &dr, &v);
                n.borrow_mut().set_type(&ids, t);
                Ok(())
            },
        );
        engine.register_fn(
            "type_name",
            |n: &mut NetHandle, id: rhai::INT| -> Result<String, Box<rhai::EvalAltResult>> {
                let id = nid(id, n.borrow().neurons.len())?;
                let g = n.borrow();
                let t = g
                    .neuron_type(id)
                    .ok_or_else(|| s_err(format!("neuron {id} has no type")))?;
                Ok(t.name.clone())
            },
        );
    }
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_param",
            move |n: &mut NetHandle,
                  id: rhai::INT,
                  pname: &str,
                  val: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let g = n.borrow_mut();
                let id = nid(id, g.neurons.len())?;
                let t = g
                    .neuron_type(id)
                    .ok_or_else(|| s_err(format!("neuron {id} has no type")))?;
                let (_, m) = crate::model::by_id(&t.model)
                    .ok_or_else(|| s_err(format!("unknown model {:?}", t.model)))?;
                let i = m
                    .params()
                    .iter()
                    .position(|p| p.name == pname)
                    .ok_or_else(|| {
                        s_err(format!(
                            "model {} has no param {pname:?} ({})",
                            t.model,
                            m.params()
                                .iter()
                                .map(|p| p.name)
                                .collect::<Vec<_>>()
                                .join(" ")
                        ))
                    })?;
                // Effective row → override with the one field changed.
                let mut row = g.neuron_params(id).to_vec();
                row[i] = d_f64(&val) as f32;
                drop(g);
                invalidate(&s, n, &dr, &v);
                n.borrow_mut().neuron_mut(id).params = Some(row.into_boxed_slice());
                Ok(())
            },
        );
    }
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_kind",
            move |n: &mut NetHandle,
                  id: rhai::INT,
                  kind: &str|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let id = nid(id, n.borrow().neurons.len())?;
                let kind = match kind {
                    "exc" | "excitatory" => net_format::NeuronKind::Excitatory,
                    "inh" | "inhibitory" => net_format::NeuronKind::Inhibitory,
                    "regular" => net_format::NeuronKind::Regular,
                    _ => {
                        return Err(s_err(format!(
                            "kind must be \"exc\", \"inh\" or \"regular\", got \"{kind}\""
                        )))
                    }
                };
                // Dale's law: flipping the kind re-signs the outgoing
                // weights at the next sim build — invalidate first.
                invalidate(&s, n, &dr, &v);
                n.borrow_mut().set_kind(id, kind);
                Ok(())
            },
        );
        engine.register_fn(
            "kind",
            |n: &mut NetHandle, id: rhai::INT| -> Result<String, Box<rhai::EvalAltResult>> {
                let id = nid(id, n.borrow().neurons.len())?;
                Ok(match n.borrow().neuron_kind(id) {
                    net_format::NeuronKind::Excitatory => "exc",
                    net_format::NeuronKind::Inhibitory => "inh",
                    net_format::NeuronKind::Regular => "regular",
                }
                .to_string())
            },
        );
        // M2: conduction speed and per-edge delays are script-tunable;
        // both invalidate so the next build re-derives delay steps.
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_conduction_speed",
            move |n: &mut NetHandle, speed: f64| -> Result<(), Box<rhai::EvalAltResult>> {
                if !(speed.is_finite() && speed > 0.0) {
                    return Err(s_err("conduction speed must be > 0"));
                }
                invalidate(&s, n, &dr, &v);
                let mut g = n.borrow_mut();
                g.meta.conduction_speed = speed as f32;
                g.refresh_delays();
                Ok(())
            },
        );
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_delay",
            move |n: &mut NetHandle,
                  src: rhai::INT,
                  tgt: rhai::INT,
                  ms: f64|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                if !(ms.is_finite() && ms >= 0.0) {
                    return Err(s_err("delay must be a finite ms >= 0"));
                }
                let mut g = n.borrow_mut();
                let src = nid(src, g.neurons.len())?;
                let tgt = nid(tgt, g.neurons.len())?;
                let mut found = false;
                for sy in g.synapses.iter_mut() {
                    if sy.src == src && sy.tgt == tgt {
                        sy.delay_ms = ms as f32;
                        found = true;
                    }
                }
                drop(g);
                if !found {
                    return Err(s_err(format!("no synapse {src} -> {tgt}")));
                }
                invalidate(&s, n, &dr, &v);
                Ok(())
            },
        );
    }
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "delete_neuron",
            move |n: &mut NetHandle, id: rhai::INT| -> Result<(), Box<rhai::EvalAltResult>> {
                let id = nid(id, n.borrow().neurons.len())?;
                invalidate(&s, n, &dr, &v);
                n.borrow_mut().delete_neuron(id);
                Ok(())
            },
        );
    }
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "delete_neurons",
            move |n: &mut NetHandle, ids: rhai::Array| -> Result<(), Box<rhai::EvalAltResult>> {
                let len = n.borrow().neurons.len();
                let ids: Vec<u32> = ids
                    .iter()
                    .map(|d| nid(d_i64(d), len))
                    .collect::<Result<_, _>>()?;
                invalidate(&s, n, &dr, &v);
                n.borrow_mut().delete_neurons(ids);
                Ok(())
            },
        );
    }
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "set_pos",
            move |n: &mut NetHandle,
                  id: rhai::INT,
                  x: Dynamic,
                  y: Dynamic,
                  z: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let id = nid(id, n.borrow().neurons.len())?;
                invalidate(&s, n, &dr, &v);
                let mut g = n.borrow_mut();
                g.neuron_mut(id).pos = [d_f64(&x) as f32, d_f64(&y) as f32, d_f64(&z) as f32];
                g.refresh_delays_of(id);
                g.invalidate_grids();
                Ok(())
            },
        );
    }
    engine.register_fn(
        "pos",
        |n: &mut NetHandle, id: rhai::INT| -> Result<rhai::Array, Box<rhai::EvalAltResult>> {
            let id = nid(id, n.borrow().neurons.len())?;
            let p = n.borrow().neuron(id).pos;
            Ok([p[0], p[1], p[2]]
                .iter()
                .map(|&x| dyn_f64(x as f64))
                .collect())
        },
    );
    engine.register_fn(
        "neuron_name",
        |n: &mut NetHandle, id: rhai::INT| -> Result<String, Box<rhai::EvalAltResult>> {
            let id = nid(id, n.borrow().neurons.len())?;
            Ok(n.borrow().neuron(id).name.clone())
        },
    );
    {
        let v = net_version.clone();
        engine.register_fn(
            "set_neuron_name",
            move |n: &mut NetHandle,
                  id: rhai::INT,
                  name: &str|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let id = nid(id, n.borrow().neurons.len())?;
                // Names don't change the sim — bump only so the editor
                // sees the rename on its next Geo delta.
                bump(&v);
                n.borrow_mut().neuron_mut(id).name = name.to_string();
                Ok(())
            },
        );
    }
    engine.register_fn("neuron", |n: &mut NetHandle, name: &str| -> rhai::INT {
        n.borrow()
            .neuron_index(name)
            .map(|i| i as rhai::INT)
            .unwrap_or(-1)
    });
    engine.register_fn(
        "neurons_of",
        |n: &mut NetHandle, g: &str| -> Result<rhai::Array, Box<rhai::EvalAltResult>> {
            group_or_err(n, g)?;
            let b = n.borrow();
            Ok(b.group(g)
                .unwrap()
                .members
                .iter()
                .map(|&m| Dynamic::from(m as rhai::INT))
                .collect())
        },
    );
    engine.register_fn("group_names", |n: &mut NetHandle| -> rhai::Array {
        n.borrow()
            .groups
            .iter()
            .map(|g| Dynamic::from(g.name.clone()))
            .collect()
    });
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "add_to_group",
            move |n: &mut NetHandle,
                  g: &str,
                  ids: rhai::Array|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let len = n.borrow().neurons.len();
                let ids: Vec<u32> = ids
                    .iter()
                    .map(|d| nid(d_i64(d), len))
                    .collect::<Result<_, _>>()?;
                // Membership feeds grid readouts — drop the sim and bump.
                invalidate(&s, n, &dr, &v);
                let mut b = n.borrow_mut();
                let gi = b.add_group(g);
                b.groups[gi].members.extend(ids);
                b.invalidate_grids();
                Ok(())
            },
        );
    }
    {
        let (s, v, dr) = (sim.clone(), net_version.clone(), drive.clone());
        engine.register_fn(
            "add_input",
            move |n: &mut NetHandle,
                  name: &str,
                  targets: rhai::Array|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let len = n.borrow().neurons.len();
                let mut tw = Vec::new();
                for t in targets {
                    if let Some(arr) = t.try_cast::<rhai::Array>() {
                        if arr.len() >= 2 {
                            tw.push((nid(d_i64(&arr[0]), len)?, d_f64(&arr[1]) as f32));
                        }
                    }
                }
                invalidate(&s, n, &dr, &v);
                n.borrow_mut().add_input(name, tw, [0.0; 3]);
                Ok(())
            },
        );
    }
    {
        let v = net_version.clone();
        engine.register_fn(
            "add_readout",
            move |n: &mut NetHandle,
                  name: &str,
                  units: rhai::Array|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let len = n.borrow().neurons.len();
                let us: Vec<u32> = units
                    .iter()
                    .map(|d| nid(d_i64(d), len))
                    .collect::<Result<_, _>>()?;
                bump(&v);
                n.borrow_mut().add_readout(name, us);
                Ok(())
            },
        );
    }
    {
        // Trained DeltaReadout weights → the network object (persisted in
        // .ne2, shown in the editor's weights plot).
        let v = net_version.clone();
        engine.register_fn(
            "set_readout_weights",
            move |n: &mut NetHandle,
                  name: &str,
                  w: rhai::Array|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                bump(&v);
                let mut b = n.borrow_mut();
                let mut w: Vec<f32> = w.iter().map(|d| d_f64(d) as f32).collect();
                let Some(r) = b.readouts.iter_mut().find(|r| r.name == name) else {
                    return Ok(());
                };
                // fit_ridge returns the weights with the bias weight
                // trailing — split it off so the file stays valid
                // (weights must be one per unit; bias_weight is separate).
                if w.len() == r.units.len() + 1 {
                    r.bias_weight = w.pop().unwrap();
                }
                if w.len() != r.units.len() {
                    return Err(format!(
                        "set_readout_weights: readout {name:?} has {} units, got {} weights",
                        r.units.len(),
                        w.len()
                    )
                    .into());
                }
                r.weights = w;
                Ok(())
            },
        );
    }
    engine.register_fn(
        "readout_weights",
        |n: &mut NetHandle, name: &str| -> rhai::Array {
            n.borrow()
                .readouts
                .iter()
                .find(|r| r.name == name)
                .map(|r| r.weights.iter().map(|&w| dyn_f64(w as f64)).collect())
                .unwrap_or_default()
        },
    );
    {
        let v = net_version.clone();
        engine.register_fn(
            "script_add",
            move |n: &mut NetHandle, name: &str, code: &str| {
                bump(&v);
                let mut b = n.borrow_mut();
                if let Some(s) = b.scripts.iter_mut().find(|(nm, _)| nm == name) {
                    s.1 = code.to_string();
                } else {
                    b.scripts.push((name.to_string(), code.to_string()));
                }
            },
        );
    }
    {
        let v = net_version.clone();
        engine.register_fn("script_remove", move |n: &mut NetHandle, name: &str| {
            bump(&v);
            n.borrow_mut().scripts.retain(|(nm, _)| nm != name);
        });
    }
    {
        let (dd, q, c) = (data_dir.clone(), quota_bytes.clone(), cfg.clone());
        engine.register_fn(
            "save",
            move |n: &mut NetHandle, path: &str| -> Result<(), Box<rhai::EvalAltResult>> {
                // M3: the live sim config is persisted with the net.
                crate::cfg_to_meta(&c.borrow(), &mut n.borrow_mut().meta);
                let bytes = net_format::io::to_bytes(&n.borrow())
                    .map_err(|e| s_err(format!("save {path}: {e}")))?;
                checked_write(&dd, &q, path, &bytes)?;
                Ok(())
            },
        );
    }
    {
        let (s, v, g, dd, c, dr) = (
            sim.clone(),
            net_version.clone(),
            net_gen.clone(),
            data_dir.clone(),
            cfg.clone(),
            drive.clone(),
        );
        engine.register_fn(
            "load",
            move |n: &mut NetHandle, path: &str| -> Result<(), Box<rhai::EvalAltResult>> {
                let p = rdir(&dd, path)?;
                let loaded =
                    net_format::io::load(&p).map_err(|e| s_err(format!("load {path}: {e}")))?;
                // M3: the file's sim settings become live.
                crate::meta_to_cfg(&loaded.meta, &mut c.borrow_mut());
                *n.borrow_mut() = loaded;
                g.set(g.get() + 1);
                invalidate(&s, n, &dr, &v);
                Ok(())
            },
        );
    }
    {
        let dd = data_dir.clone();
        engine.register_fn(
            "net_load",
            move |path: &str| -> Result<NetHandle, Box<rhai::EvalAltResult>> {
                let p = rdir(&dd, path)?;
                Ok(Rc::new(RefCell::new(
                    net_format::io::load(&p).map_err(|e| s_err(format!("load {path}: {e}")))?,
                )))
            },
        );
    }

    // ---- sim methods ----
    {
        let (s, n, c, lim, dr) = (
            sim.clone(),
            net.clone(),
            cfg.clone(),
            sim_limits.clone(),
            drive.clone(),
        );
        let b = backend.clone();
        let (p, pc) = (progress.clone(), progress_chunk.clone());
        engine.register_fn(
            "run",
            move |_: &mut SimSlot, ms: Dynamic| -> Result<(), Box<rhai::EvalAltResult>> {
                let h = ensure_sim(&s, &n, &c, &b, &lim, &dr)?;
                let steps = (d_f64(&ms) as f32 / c.borrow().dt_ms).round().max(0.0) as usize;
                run_chunked(&h, steps, &p, &pc).map_err(s_err)
            },
        );
    }
    {
        let (s, n, c, lim, dr) = (
            sim.clone(),
            net.clone(),
            cfg.clone(),
            sim_limits.clone(),
            drive.clone(),
        );
        let b = backend.clone();
        let (p, pc) = (progress.clone(), progress_chunk.clone());
        engine.register_fn(
            "run_steps",
            move |_: &mut SimSlot, steps: rhai::INT| -> Result<(), Box<rhai::EvalAltResult>> {
                let h = ensure_sim(&s, &n, &c, &b, &lim, &dr)?;
                run_chunked(&h, steps.max(0) as usize, &p, &pc).map_err(s_err)
            },
        );
    }
    {
        let (s, n, c, lim, dr) = (
            sim.clone(),
            net.clone(),
            cfg.clone(),
            sim_limits.clone(),
            drive.clone(),
        );
        let b = backend.clone();
        engine.register_fn(
            "build",
            move |_: &mut SimSlot| -> Result<(), Box<rhai::EvalAltResult>> {
                ensure_sim(&s, &n, &c, &b, &lim, &dr)?;
                Ok(())
            },
        );
    }
    {
        let (s, n, dr) = (sim.clone(), net.clone(), drive.clone());
        engine.register_fn("rebuild", move |_: &mut SimSlot| {
            stash_drive(&s, &dr, &n);
        });
    }
    engine.register_fn("time", |s: &mut SimSlot| -> f64 {
        s.borrow()
            .as_ref()
            .map(|h| h.borrow().time_ms())
            .unwrap_or(0.0)
    });
    engine.register_fn("reset", |s: &mut SimSlot| {
        if let Some(h) = s.borrow().as_ref() {
            h.borrow_mut().reset_state();
        }
    });
    engine.register_fn("rate", |s: &mut SimSlot| -> f64 {
        s.borrow()
            .as_ref()
            .map(|h| h.borrow().mean_rate_hz())
            .unwrap_or(0.0)
    });
    engine.register_fn("spikes", |s: &mut SimSlot| -> rhai::INT {
        s.borrow()
            .as_ref()
            .map(|h| h.borrow().spikes_total() as rhai::INT)
            .unwrap_or(0)
    });
    engine.register_fn("trace", |s: &mut SimSlot| -> rhai::Array {
        s.borrow()
            .as_ref()
            .map(|h| {
                h.borrow()
                    .trace()
                    .into_iter()
                    .map(|f| dyn_f64(f as f64))
                    .collect()
            })
            .unwrap_or_default()
    });
    engine.register_fn(
        "trace_of",
        |s: &mut SimSlot, ids: rhai::Array| -> Result<rhai::Array, Box<rhai::EvalAltResult>> {
            let Some(h) = s.borrow().as_ref().cloned() else {
                return Ok(Vec::new());
            };
            let tr = h.borrow().trace();
            ids.iter()
                .map(|d| Ok(dyn_f64(tr[nid(d_i64(d), tr.len())? as usize] as f64)))
                .collect()
        },
    );
    engine.register_fn(
        "v",
        |s: &mut SimSlot, id: rhai::INT| -> Result<f64, Box<rhai::EvalAltResult>> {
            let Some(h) = s.borrow().as_ref().cloned() else {
                return Ok(0.0);
            };
            let b = h.borrow();
            let v = b.v();
            Ok(v[nid(id, v.len())? as usize] as f64)
        },
    );
    engine.register_fn(
        "spike_counts",
        |s: &mut SimSlot, rows: rhai::INT| -> Result<rhai::Array, Box<rhai::EvalAltResult>> {
            let Some(h) = s.borrow().as_ref().cloned() else {
                return Ok(Vec::new());
            };
            let c = h
                .borrow()
                .spike_counts(rows.max(0) as usize)
                .map_err(s_err)?;
            Ok(c.into_iter().map(|f| dyn_f64(f as f64)).collect())
        },
    );
    engine.register_fn(
        "spike_counts_of",
        |s: &mut SimSlot,
         rows: rhai::INT,
         ids: rhai::Array|
         -> Result<rhai::Array, Box<rhai::EvalAltResult>> {
            let Some(h) = s.borrow().as_ref().cloned() else {
                return Ok(Vec::new());
            };
            let c = h
                .borrow()
                .spike_counts(rows.max(0) as usize)
                .map_err(s_err)?;
            ids.iter()
                .map(|d| Ok(dyn_f64(c[nid(d_i64(d), c.len())? as usize] as f64)))
                .collect()
        },
    );
    engine.register_fn("vs", |s: &mut SimSlot| -> rhai::Array {
        s.borrow()
            .as_ref()
            .map(|h| {
                h.borrow()
                    .v()
                    .into_iter()
                    .map(|f| dyn_f64(f as f64))
                    .collect()
            })
            .unwrap_or_default()
    });
    {
        let (s, n, c, lim, dr) = (
            sim.clone(),
            net.clone(),
            cfg.clone(),
            sim_limits.clone(),
            drive.clone(),
        );
        let b = backend.clone();
        engine.register_fn(
            "set_input",
            move |_: &mut SimSlot,
                  name: &str,
                  x: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let idx = match n.borrow().input_index(name) {
                    Some(i) => i,
                    None => return Err(s_err(format!("unknown input \"{name}\""))),
                };
                let h = ensure_sim(&s, &n, &c, &b, &lim, &dr)?;
                h.borrow_mut().set_input_constant(idx, d_f64(&x) as f32);
                Ok(())
            },
        );
    }
    {
        let (s, n, c, lim, dr) = (
            sim.clone(),
            net.clone(),
            cfg.clone(),
            sim_limits.clone(),
            drive.clone(),
        );
        let b = backend.clone();
        engine.register_fn(
            "play_input",
            move |_: &mut SimSlot,
                  name: &str,
                  arr: rhai::Array|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let idx = match n.borrow().input_index(name) {
                    Some(i) => i,
                    None => return Err(s_err(format!("unknown input \"{name}\""))),
                };
                let series: Vec<f32> = arr.iter().map(|d| d_f64(d) as f32).collect();
                let h = ensure_sim(&s, &n, &c, &b, &lim, &dr)?;
                h.borrow_mut().play_input(idx, series);
                Ok(())
            },
        );
    }
    {
        let (s, n, c, lim, dr) = (
            sim.clone(),
            net.clone(),
            cfg.clone(),
            sim_limits.clone(),
            drive.clone(),
        );
        let b = backend.clone();
        engine.register_fn(
            "set_external",
            move |_: &mut SimSlot,
                  id: rhai::INT,
                  amps: Dynamic|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let h = ensure_sim(&s, &n, &c, &b, &lim, &dr)?;
                let n_len = h.borrow().v().len();
                h.borrow_mut()
                    .set_external(nid(id, n_len)? as usize, d_f64(&amps) as f32);
                Ok(())
            },
        );
    }
    {
        let (c, s) = (cfg.clone(), sim.clone());
        engine.register_fn(
            "set_noise",
            move |_: &mut SimSlot, hz: Dynamic, kick: Dynamic| {
                let (hz, kick) = (d_f64(&hz) as f32, d_f64(&kick) as f32);
                c.borrow_mut().noise_rate_hz = hz;
                c.borrow_mut().noise_kick_mv = kick;
                if let Some(h) = s.borrow().as_ref() {
                    h.borrow_mut().set_noise(hz, kick);
                }
            },
        );
    }
    {
        let c = cfg.clone();
        engine.register_fn("set_synapse_tau", move |_: &mut SimSlot, tau: Dynamic| {
            c.borrow_mut().synapse_tau_ms = d_f64(&tau) as f32;
        });
    }
    {
        let c = cfg.clone();
        engine.register_fn("set_readout_tau", move |_: &mut SimSlot, tau: Dynamic| {
            c.borrow_mut().readout_tau_ms = d_f64(&tau) as f32;
        });
    }
    {
        let (c, s, n, dr) = (cfg.clone(), sim.clone(), net.clone(), drive.clone());
        engine.register_fn("set_dt", move |_: &mut SimSlot, dt: Dynamic| {
            let d = d_f64(&dt) as f32;
            if !crate::valid_dt_ms(d) {
                return Err(s_err(format!(
                    "dt {d} out of range {}..{} ms",
                    crate::MIN_DT_MS,
                    crate::MAX_DT_MS
                )));
            }
            c.borrow_mut().dt_ms = d;
            stash_drive(&s, &dr, &n); // ring depth depends on dt
            Ok(())
        });
    }
    {
        let (c, s, n, dr) = (cfg.clone(), sim.clone(), net.clone(), drive.clone());
        engine.register_fn("set_int_weights", move |_: &mut SimSlot, on: rhai::INT| {
            c.borrow_mut().int_weights = on != 0;
            stash_drive(&s, &dr, &n); // switches the kernel + ring semantics
        });
    }
    // M3: the persisted settings are readable, so a script can confirm
    // what a loaded file actually adopted.
    {
        let c = cfg.clone();
        engine.register_fn("synapse_tau", move |_: &mut SimSlot| -> f64 {
            c.borrow().synapse_tau_ms as f64
        });
    }
    {
        let c = cfg.clone();
        engine.register_fn("readout_tau", move |_: &mut SimSlot| -> f64 {
            c.borrow().readout_tau_ms as f64
        });
    }
    {
        let c = cfg.clone();
        engine.register_fn("dt", move |_: &mut SimSlot| -> f64 {
            c.borrow().dt_ms as f64
        });
    }

    // ---- readouts ----
    engine.register_fn(
        "delta_new",
        |n: rhai::INT| -> Result<DeltaReadout, Box<rhai::EvalAltResult>> {
            if !(0..=10_000_000).contains(&n) {
                return Err(s_err(format!("delta_new({n}): units must be 0..10M")));
            }
            Ok(DeltaReadout::new(n as usize))
        },
    );
    {
        let r = rng.clone();
        engine.register_fn(
            "init_random",
            move |ro: &mut DeltaReadout, lo: Dynamic, hi: Dynamic| {
                ro.init_weights_random(d_f64(&lo) as f32, d_f64(&hi) as f32, &mut r.borrow_mut());
            },
        );
    }
    engine.register_fn(
        "output",
        |ro: &mut DeltaReadout, trace: rhai::Array| -> f64 {
            let t: Vec<f32> = trace.iter().map(|d| d_f64(d) as f32).collect();
            ro.output(&t) as f64
        },
    );
    engine.register_fn(
        "learn",
        |ro: &mut DeltaReadout, trace: rhai::Array, target: Dynamic, rate: Dynamic| {
            let t: Vec<f32> = trace.iter().map(|d| d_f64(d) as f32).collect();
            ro.learn(&t, d_f64(&target) as f32, d_f64(&rate) as f32);
        },
    );
    engine.register_fn("set_proper", |ro: &mut DeltaReadout, p: rhai::INT| {
        ro.proper = p != 0;
    });
    engine.register_fn("weights", |ro: &mut DeltaReadout| -> rhai::Array {
        ro.weights.iter().map(|&w| dyn_f64(w as f64)).collect()
    });
    engine.register_fn(
        "fit_ridge",
        |x: rhai::Array,
         n_rows: rhai::INT,
         n_feat: rhai::INT,
         y: rhai::Array,
         lambda: Dynamic|
         -> Result<rhai::Array, Box<rhai::EvalAltResult>> {
            let (t, f) = (n_rows.max(0) as usize, n_feat.max(0) as usize);
            if x.len() < t.saturating_mul(f) {
                return Err(s_err(format!(
                    "fit_ridge: x has {} values, need T*F = {t}*{f}",
                    x.len()
                )));
            }
            if y.len() < t {
                return Err(s_err(format!(
                    "fit_ridge: y has {} values, need T = {t}",
                    y.len()
                )));
            }
            let xf: Vec<f32> = x.iter().map(|d| d_f64(d) as f32).collect();
            let yf: Vec<f32> = y.iter().map(|d| d_f64(d) as f32).collect();
            let w = fit_ridge(&xf, t, f, &yf, d_f64(&lambda) as f32);
            Ok(w.iter().map(|&v| dyn_f64(v)).collect())
        },
    );
    engine.register_fn("ridge_output", |w: rhai::Array, x: rhai::Array| -> f64 {
        let wf: Vec<f64> = w.iter().map(d_f64).collect();
        let xf: Vec<f32> = x.iter().map(|d| d_f64(d) as f32).collect();
        ridge_output(&wf, &xf)
    });
    engine.register_fn(
        "accuracy",
        |scores: rhai::Array, labels: rhai::Array| -> f64 {
            let s: Vec<f64> = scores.iter().map(d_f64).collect();
            let y: Vec<u8> = labels.iter().map(|d| d_i64(d) as u8).collect();
            accuracy(&s, &y)
        },
    );
    engine.register_fn("auc", |scores: rhai::Array, labels: rhai::Array| -> f64 {
        let s: Vec<f64> = scores.iter().map(d_f64).collect();
        let y: Vec<u8> = labels.iter().map(|d| d_i64(d) as u8).collect();
        auc(&s, &y)
    });

    // ---- RNG ----
    {
        let r = rng.clone();
        engine.register_fn("rand", move |lo: Dynamic, hi: Dynamic| -> f64 {
            r.borrow_mut().uniform(d_f64(&lo), d_f64(&hi))
        });
    }
    {
        let r = rng.clone();
        engine.register_fn("gauss", move |m: Dynamic, s: Dynamic| -> f64 {
            r.borrow_mut().gauss(d_f64(&m), d_f64(&s))
        });
    }
    {
        let r = rng.clone();
        engine.register_fn("seed", move |x: rhai::INT| {
            *r.borrow_mut() = Rng::new(x as u64);
        });
    }

    // ---- RNG bit-stream generator (hidden-signal source) ----
    engine.register_fn(
        "rng_stream",
        |seed: rhai::INT,
         n_src: rhai::INT,
         p: Dynamic|
         -> Result<RngStream, Box<rhai::EvalAltResult>> {
            if !(0..=10_000_000).contains(&n_src) {
                return Err(s_err(format!(
                    "rng_stream(…, {n_src}, …): sources must be 0..10M"
                )));
            }
            Ok(RngStream::new(
                seed as u64,
                n_src as usize,
                d_f64(&p) as f32,
            ))
        },
    );
    engine.register_fn(
        "window",
        |rs: &mut RngStream,
         steps: rhai::INT,
         label: rhai::INT,
         pairs: rhai::Array,
         strength: Dynamic|
         -> Result<rhai::Array, Box<rhai::EvalAltResult>> {
            if !(0..=100_000_000).contains(&steps) {
                return Err(s_err(format!("window steps {steps}: must be 0..100M")));
            }
            let mut pair_idx = Vec::new();
            for p in pairs {
                if let Some(a) = p.try_cast::<rhai::Array>() {
                    if a.len() >= 2 {
                        pair_idx.push((d_i64(&a[0]) as usize, d_i64(&a[1]) as usize));
                    }
                }
            }
            let m = rs.window(
                steps as usize,
                label != 0,
                &pair_idx,
                d_f64(&strength) as f32,
            );
            Ok(m.iter()
                .map(|row| {
                    row.iter()
                        .map(|&v| dyn_f64(v as f64))
                        .collect::<rhai::Array>()
                        .into()
                })
                .collect())
        },
    );

    // ---- samples / files ----
    {
        let v = net_version.clone();
        engine.register_fn(
            "samples_write",
            move |n: &mut NetHandle, name: &str, times: rhai::Array, values: rhai::Array| {
                bump(&v);
                let t: Vec<f32> = times.iter().map(|d| d_f64(d) as f32).collect();
                let vv: Vec<f32> = values.iter().map(|d| d_f64(d) as f32).collect();
                let mut b = n.borrow_mut();
                let s = b.samples_mut(name);
                s.times = t;
                s.values = vv;
            },
        );
    }
    {
        let (v, dd, q) = (net_version.clone(), data_dir.clone(), quota_bytes.clone());
        engine.register_fn(
            "samples_save",
            move |n: &mut NetHandle,
                  name: &str,
                  path: &str|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let b = n.borrow();
                let Some(s) = b.samples(name) else {
                    return Err(s_err(format!("unknown samples \"{name}\"")));
                };
                let text = net_format::io::samples_to_string(s);
                checked_write(&dd, &q, path, text.as_bytes())
                    .map_err(|e| s_err(format!("samples_save {path}: {e}")))?;
                bump(&v);
                Ok(())
            },
        );
    }
    {
        let (v, dd) = (net_version.clone(), data_dir.clone());
        engine.register_fn(
            "samples_load",
            move |n: &mut NetHandle,
                  name: &str,
                  path: &str|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let p = rdir(&dd, path).map_err(|e| s_err(format!("samples_load {path}: {e}")))?;
                let s = net_format::io::load_samples(&p, name)
                    .map_err(|e| s_err(format!("samples_load {path}: {e}")))?;
                bump(&v);
                let mut b = n.borrow_mut();
                let slot = b.samples_mut(name);
                slot.times = s.times;
                slot.values = s.values;
                Ok(())
            },
        );
    }
    // `net.samples(name)` → [[t,v],…] pairs.
    engine.register_fn("samples", |n: &mut NetHandle, name: &str| -> rhai::Array {
        n.borrow()
            .samples(name)
            .map(|s| {
                s.times
                    .iter()
                    .zip(s.values.iter())
                    .map(|(&t, &v)| rhai::Dynamic::from(vec![dyn_f64(t as f64), dyn_f64(v as f64)]))
                    .collect()
            })
            .unwrap_or_default()
    });
    {
        let (dd, q) = (data_dir.clone(), quota_bytes.clone());
        engine.register_fn(
            "write_to_file",
            move |path: &str, values: rhai::Array| -> Result<(), Box<rhai::EvalAltResult>> {
                let mut out = String::new();
                for v in values {
                    out.push_str(&format!("{}\n", d_f64(&v)));
                }
                checked_write(&dd, &q, path, out.as_bytes())
                    .map(|_| ())
                    .map_err(|e| s_err(format!("write_to_file {path}: {e}")))
            },
        );
    }
    {
        let dd = data_dir.clone();
        engine.register_fn("read_to_vec", move |path: &str| -> rhai::Array {
            let p = match rdir(&dd, path) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("read_to_vec {path}: {e}");
                    return Vec::new();
                }
            };
            std::fs::read_to_string(p)
                .map(|s| {
                    s.lines()
                        .filter_map(|l| l.trim().parse::<f64>().ok())
                        .map(dyn_f64)
                        .collect()
                })
                .unwrap_or_default()
        });
    }

    // ---- datasets (`datasets/` under the sandbox root) ----
    {
        let dd = data_dir.clone();
        engine.register_fn(
            "dataset",
            move |name: &str| -> Result<crate::dataset::Dataset, Box<rhai::EvalAltResult>> {
                let root = std::path::PathBuf::from(rdir(&dd, "datasets")?);
                crate::dataset::Dataset::load(&root, name).map_err(s_err)
            },
        );
        let (dd, q) = (data_dir.clone(), quota_bytes.clone());
        engine.register_fn(
            "dataset_gen",
            move |spec: rhai::Map| -> Result<crate::dataset::Dataset, Box<rhai::EvalAltResult>> {
                let spec = crate::dataset::GenSpec::from_map(&spec).map_err(s_err)?;
                let root = std::path::PathBuf::from(rdir(&dd, "datasets")?);
                let quota = *q.borrow();
                if quota > 0 {
                    // Post-check: generated size isn't known up front —
                    // measure the delta the call wrote and roll it back
                    // if it pushed the sandbox over quota.
                    let dir = crate::dataset::Dataset::dir_under(&root, &spec.name);
                    let existed = dir.exists();
                    let before = crate::serve::dir_size(&root);
                    let d =
                        crate::dataset::Dataset::generate_or_load(&root, &spec).map_err(s_err)?;
                    let added = crate::serve::dir_size(&root).saturating_sub(before);
                    let used = dd
                        .borrow()
                        .as_ref()
                        .map(|d| crate::serve::dir_size(d))
                        .unwrap_or(0);
                    if added > 0 && used > quota {
                        if !existed {
                            let _ = std::fs::remove_dir_all(&dir);
                        }
                        return Err(s_err(format!(
                            "over quota: dataset '{}' needs {added} B past the {quota} B limit",
                            spec.name
                        )));
                    }
                    Ok(d)
                } else {
                    crate::dataset::Dataset::generate_or_load(&root, &spec).map_err(s_err)
                }
            },
        );
        let dd = data_dir.clone();
        engine.register_fn("datasets", move || -> rhai::Array {
            let root = match rdir(&dd, "datasets") {
                Ok(p) => std::path::PathBuf::from(p),
                Err(_) => return Vec::new(),
            };
            std::fs::read_dir(root)
                .map(|rd| {
                    rd.flatten()
                        .filter(|e| e.path().is_dir())
                        .map(|e| e.file_name().to_string_lossy().into_owned().into())
                        .collect()
                })
                .unwrap_or_default()
        });
        engine.register_fn(
            "channel",
            |d: &mut crate::dataset::Dataset,
             i: rhai::INT|
             -> Result<rhai::Array, Box<rhai::EvalAltResult>> {
                if i < 0 || i as usize >= d.channels.len() {
                    return Err(s_err(format!("channel {i} out of 0..{}", d.channels.len())));
                }
                Ok(d.channels[i as usize]
                    .iter()
                    .map(|&v| rhai::FLOAT::from(v).into())
                    .collect())
            },
        );
        engine.register_fn("events", |d: &mut crate::dataset::Dataset| -> rhai::Array {
            d.events_ms.iter().map(|&v| v.into()).collect()
        });
        engine.register_fn(
            "events",
            |d: &mut crate::dataset::Dataset, lo: rhai::FLOAT, hi: rhai::FLOAT| -> rhai::Array {
                d.events_ms
                    .iter()
                    .filter(|&&v| v >= lo && v < hi)
                    .map(|&v| v.into())
                    .collect()
            },
        );
        engine.register_get(
            "n_channels",
            |d: &mut crate::dataset::Dataset| -> rhai::INT { d.channels.len() as rhai::INT },
        );
        engine.register_get("steps", |d: &mut crate::dataset::Dataset| -> rhai::INT {
            d.steps() as rhai::INT
        });
        engine.register_get("len_ms", |d: &mut crate::dataset::Dataset| -> rhai::FLOAT {
            d.len_ms()
        });
        engine.register_get("dt_ms", |d: &mut crate::dataset::Dataset| -> rhai::FLOAT {
            d.dt_ms as rhai::FLOAT
        });
        engine.register_get("name", |d: &mut crate::dataset::Dataset| -> String {
            d.name.clone()
        });
        engine.register_get("spec_json", |d: &mut crate::dataset::Dataset| -> String {
            d.spec_json.clone()
        });
    }

    // ---- training jobs (`job_progress` streams fields; the server job
    // runner swaps the sink for `JobProgress` messages) ----
    {
        let js = job_sink.clone();
        engine.register_fn("job_progress", move |m: rhai::Map| {
            let fields: Vec<(String, f64)> = m
                .iter()
                .filter_map(|(k, v)| {
                    v.as_float()
                        .ok()
                        .map(|f| (k.to_string(), f as f64))
                        .or_else(|| v.as_int().ok().map(|i| (k.to_string(), i as f64)))
                })
                .collect();
            (js.borrow_mut())(fields);
        });
        let jd = job_dir.clone();
        engine.register_fn("job_dir", move || -> String {
            jd.borrow().clone().unwrap_or_default()
        });
    }
}

/// Seeded multi-source bit-stream generator for the hidden-signal example.
/// Each source is i.i.d. Bernoulli(p). During labelled event windows the
/// listed pairs share a common coin with probability `strength`, which
/// correlates each pair while leaving every stream's marginal Bernoulli(p).
#[derive(Clone, Debug)]
pub struct RngStream {
    pub n_src: usize,
    pub p: f32,
    rng: Rng,
}

impl RngStream {
    pub fn new(seed: u64, n_src: usize, p: f32) -> Self {
        RngStream {
            n_src,
            p,
            rng: Rng::new(seed),
        }
    }

    /// `steps` × `n_src` matrix of 0/1. If `event`, each listed pair shares a
    /// common Bernoulli(p) bit with probability `strength` per step — a joint
    /// pattern that keeps every stream's own mean/variance unchanged.
    pub fn window(
        &mut self,
        steps: usize,
        event: bool,
        pairs: &[(usize, usize)],
        strength: f32,
    ) -> Vec<Vec<f32>> {
        let mut out = vec![vec![0.0f32; self.n_src]; steps];
        let p = self.p as f64;
        for step in out.iter_mut().take(steps) {
            for x in step.iter_mut().take(self.n_src) {
                *x = if self.rng.f64() < p { 1.0 } else { 0.0 };
            }
            if event && strength > 0.0 {
                for &(i, j) in pairs {
                    if i >= self.n_src || j >= self.n_src {
                        continue;
                    }
                    if self.rng.f64() < strength as f64 {
                        let bit = if self.rng.f64() < p { 1.0 } else { 0.0 };
                        step[i] = bit;
                        step[j] = bit;
                    }
                }
            }
        }
        out
    }
}
