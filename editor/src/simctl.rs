//! Sim worker thread: owns the network, script Env and simulator; streams
//! snapshots into `Shared`. The UI sends `Cmd`s over an mpsc channel and reads
//! `Shared` each frame — the sim never waits on the UI.

use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, RwLock};
use std::time::{Duration, Instant};

use net_format::Network;
use sim::script::{Env, ProgressFn};
use sim::{BackendKind, Sim, SimConfig};

use crate::snap::{GeoSnap, InputSnap, Shared, SharedRef, Snap};

/// Edits from the UI applied to the worker's network.
pub type NetEdit = Box<dyn FnOnce(&mut Network) + Send>;

#[allow(dead_code)]
pub enum Cmd {
    /// Run a Rhai script on the worker (console, load file).
    Eval(String),
    /// Replace the whole network from a .ne2 file.
    LoadNet(String),
    /// Replace with an empty network.
    NewNet,
    /// Continuous run at `speed × realtime`.
    AutoRun(bool),
    SetSpeed(f32),
    /// Single synchronous run of `ms` (toolbar Step).
    RunMs(f32),
    /// Abort a chunked run in flight (pause button).
    Abort,
    Reset,
    /// Neuron ids to record v(t) for (plots).
    Watch(Vec<u32>),
    /// Apply an edit to the network (invalidates the sim).
    EditNet(NetEdit),
    SetBackend(BackendKind),
    SetDtMs(f32),
    SetNoise(f32, f32),
    SetInputConst {
        name: String,
        value: f32,
    },
    PlayInput {
        name: String,
        series: Vec<f32>,
    },
    SetExternal {
        neuron: u32,
        amps: f32,
    },
    Quit,
}

#[allow(dead_code)]
pub struct SimCtl {
    pub tx: mpsc::Sender<Cmd>,
    pub shared: SharedRef,
    /// Ask a running chunked run to stop early.
    pub abort: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

const GEO_EDGES_PER_NEURON: usize = 4;
const WATCH_CAP: usize = 4096;
const COUNTS_WINDOW: usize = 4096;

struct Watch {
    ids: Vec<u32>,
    ring: Vec<VecDeque<f32>>,
}

/// Mutable publish state shared by the progress callback and the cmd loop.
struct Pub {
    watch: Watch,
    /// `sim.step_count()` at the previous publish — `counts` covers the
    /// interval since then.
    prev_step: u64,
    sps: f32,
}

impl SimCtl {
    pub fn start(net: Network, backend: BackendKind, seed: u64) -> SimCtl {
        Self::start_cfg(net, backend, seed, SimConfig::default())
    }

    /// Like `start` but lets the caller set extra `SimConfig` fields
    /// (e.g. `int_weights`); `seed` still overrides the config's.
    pub fn start_cfg(net: Network, backend: BackendKind, seed: u64, cfg: SimConfig) -> SimCtl {
        let shared: SharedRef = Arc::new(RwLock::new(Shared::default()));
        let abort = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<Cmd>();
        let sh2 = shared.clone();
        let ab2 = abort.clone();
        let handle = std::thread::Builder::new()
            .name("sim-worker".into())
            .spawn(move || worker(net, backend, seed, cfg, rx, sh2, ab2))
            .expect("spawn sim worker");
        SimCtl {
            tx,
            shared,
            abort,
            handle: Some(handle),
        }
    }

    pub fn send(&self, c: Cmd) {
        let _ = self.tx.send(c);
    }
}

impl Drop for SimCtl {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Quit);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker(
    net: Network,
    backend: BackendKind,
    seed: u64,
    cfg: SimConfig,
    rx: mpsc::Receiver<Cmd>,
    shared: SharedRef,
    abort: Arc<AtomicBool>,
) {
    let sh = shared.clone();
    let print: sim::script::PrintFn = Rc::new(move |s: &str| {
        if let Ok(mut w) = sh.write() {
            w.log(s);
        }
    });
    let mut env = Env::with_print(net, backend, seed, print);
    // The worker's whole job is publishing snapshots; raster, particles and
    // the activity counts all need the rolling spike bitmap, so force it on
    // regardless of the caller's cfg.
    *env.cfg.borrow_mut() = SimConfig {
        record_spikes: true,
        spike_record_steps: cfg.spike_record_steps.max(COUNTS_WINDOW),
        seed,
        ..cfg
    };

    let pub_rc = Rc::new(std::cell::RefCell::new(Pub {
        watch: Watch {
            ids: Vec::new(),
            ring: Vec::new(),
        },
        prev_step: 0,
        sps: 0.0,
    }));

    // Progress hook: publish between run chunks inside script `sim.run`s;
    // returning false aborts the rest of the run (pause pressed).
    {
        let (shp, abp, pr) = (shared.clone(), abort.clone(), pub_rc.clone());
        let cb: ProgressFn = Rc::new(std::cell::RefCell::new(move |sim: &Sim| -> bool {
            let mut p = pr.borrow_mut();
            publish_sim(sim, &shp, &mut p);
            !abp.load(Ordering::Relaxed)
        }));
        env.set_progress(200, cb);
    }

    let mut running = false;
    let mut speed = 1.0f32;
    let mut published_geo_version = 0u64;
    let mut chunk = 200usize;

    let mut quit = false;
    while !quit {
        // ---- geometry republish on version bump ----
        let gv = env.net_version.get();
        if gv != published_geo_version {
            let geo = geo_snap(&env);
            if let Ok(mut w) = shared.write() {
                w.geo = Some(Arc::new(geo));
            }
            published_geo_version = gv;
        }

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
        for c in pending {
            match c {
                Cmd::Quit => quit = true,
                Cmd::Abort => {
                    abort.store(true, Ordering::Relaxed);
                    running = false;
                }
                Cmd::Eval(code) => {
                    abort.store(false, Ordering::Relaxed);
                    set_busy(&shared, true);
                    log(&shared, format!("> {code}"));
                    match env.eval_str(&code) {
                        Ok(d) if !d.is_unit() => log(&shared, format!("= {d}")),
                        Ok(_) => {}
                        Err(e) => log(&shared, format!("! {e}")),
                    }
                    set_busy(&shared, false);
                }
                Cmd::LoadNet(path) => match net_format::io::load(&path) {
                    Ok(n) => {
                        *env.net.borrow_mut() = n;
                        bump(&env);
                        log(&shared, format!("loaded {path}"));
                    }
                    Err(e) => log(&shared, format!("load {path}: {e}")),
                },
                Cmd::NewNet => {
                    *env.net.borrow_mut() = Network::new();
                    bump(&env);
                }
                Cmd::AutoRun(b) => {
                    abort.store(false, Ordering::Relaxed);
                    running = b;
                }
                Cmd::SetSpeed(s) => speed = s.max(0.01),
                Cmd::RunMs(ms) => {
                    abort.store(false, Ordering::Relaxed);
                    if let Ok(h) = env.sim_handle() {
                        let dt = env.cfg.borrow().dt_ms;
                        let steps = (ms / dt).round().max(0.0) as usize;
                        run_published(&env, &h, steps, &shared, &pub_rc, &abort);
                    }
                }
                Cmd::Reset => {
                    if let Ok(h) = env.sim_handle() {
                        h.borrow_mut().reset_state();
                    }
                    let mut p = pub_rc.borrow_mut();
                    p.prev_step = 0;
                    publish_env(&env, &shared, &mut p);
                }
                Cmd::Watch(ids) => {
                    let mut p = pub_rc.borrow_mut();
                    p.watch.ids = ids;
                    p.watch.ring = vec![VecDeque::with_capacity(WATCH_CAP); p.watch.ids.len()];
                }
                Cmd::EditNet(f) => {
                    f(&mut env.net.borrow_mut());
                    bump(&env);
                }
                Cmd::SetBackend(b) => {
                    env.backend = b;
                    env.invalidate_sim();
                }
                Cmd::SetDtMs(dt) => {
                    env.cfg.borrow_mut().dt_ms = dt;
                    env.invalidate_sim();
                }
                Cmd::SetNoise(hz, kick) => {
                    env.cfg.borrow_mut().noise_rate_hz = hz;
                    env.cfg.borrow_mut().noise_kick_mv = kick;
                    if let Ok(h) = env.sim_handle() {
                        h.borrow_mut().set_noise(hz, kick);
                    }
                }
                Cmd::SetInputConst { name, value } => {
                    let idx = env.net.borrow().input_index(&name);
                    if let (Some(i), Ok(h)) = (idx, env.sim_handle()) {
                        h.borrow_mut().set_input_constant(i, value);
                    }
                }
                Cmd::PlayInput { name, series } => {
                    let idx = env.net.borrow().input_index(&name);
                    if let (Some(i), Ok(h)) = (idx, env.sim_handle()) {
                        h.borrow_mut().play_input(i, series);
                    }
                }
                Cmd::SetExternal { neuron, amps } => {
                    if let Ok(h) = env.sim_handle() {
                        h.borrow_mut().set_external(neuron as usize, amps);
                    }
                }
            }
        }

        // ---- continuous run: one chunk, throttle to `speed` × realtime ----
        if running {
            match env.sim_handle() {
                Ok(h) => {
                    let t0 = Instant::now();
                    h.borrow_mut().run_steps(chunk);
                    let el = t0.elapsed().as_secs_f32().max(1e-6);
                    {
                        let mut p = pub_rc.borrow_mut();
                        p.sps = chunk as f32 / el;
                        publish_sim(&h.borrow(), &shared, &mut p);
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
                    log(&shared, format!("sim build failed: {e}"));
                }
            }
        }
        if let Ok(mut w) = shared.write() {
            w.running = running;
        }
    }
}

fn bump(env: &Env) {
    env.invalidate_sim();
    env.net_version.set(env.net_version.get() + 1);
}

fn log(shared: &SharedRef, s: String) {
    if let Ok(mut w) = shared.write() {
        w.log(s);
    }
}

fn set_busy(shared: &SharedRef, b: bool) {
    if let Ok(mut w) = shared.write() {
        w.busy = b;
    }
}

fn publish_env(env: &Env, shared: &SharedRef, p: &mut Pub) {
    if let Ok(h) = env.sim_handle() {
        publish_sim(&h.borrow(), shared, p);
    }
}

/// Publish dynamic state read back from `sim` into `shared`.
fn publish_sim(sim: &Sim, shared: &SharedRef, p: &mut Pub) {
    let activity = sim.trace();
    // Counts since previous publish: sum the record-ring rows covering the
    // elapsed steps. Intervals longer than the ring depth undercount the
    // oldest steps, which is fine for raster/particles.
    let back = sim
        .step_count()
        .saturating_sub(p.prev_step)
        .min(COUNTS_WINDOW as u64) as usize;
    let counts = sim.spike_counts(back);
    p.prev_step = sim.step_count();

    let mut watched_v = Vec::new();
    if !p.watch.ids.is_empty() {
        let v = sim.v();
        for (k, &id) in p.watch.ids.iter().enumerate() {
            let val = v.get(id as usize).copied().unwrap_or(0.0);
            let r = &mut p.watch.ring[k];
            r.push_back(val);
            if r.len() > WATCH_CAP {
                r.pop_front();
            }
            watched_v.push((id, Arc::new(vec![val])));
        }
    }

    if let Ok(mut w) = shared.write() {
        w.snap = Snap {
            seq: w.snap.seq + 1,
            t_steps: sim.step_count(),
            time_ms: sim.time_ms(),
            spikes_total: sim.spikes_total(),
            rate_hz: sim.mean_rate_hz(),
            activity: Arc::new(activity),
            counts: Arc::new(counts),
            watched_v,
            steps_per_sec: p.sps,
        };
        w.sim_built = true;
    }
}

fn run_published(
    env: &Env,
    h: &Rc<std::cell::RefCell<Sim>>,
    mut steps: usize,
    shared: &SharedRef,
    pub_rc: &Rc<std::cell::RefCell<Pub>>,
    abort: &Arc<AtomicBool>,
) {
    let t0 = Instant::now();
    let mut done = 0usize;
    while steps > 0 && !abort.load(Ordering::Relaxed) {
        let n = steps.min(200);
        h.borrow_mut().run_steps(n);
        steps -= n;
        done += n;
        let mut p = pub_rc.borrow_mut();
        p.sps = done as f32 / t0.elapsed().as_secs_f32().max(1e-6);
        publish_env(env, shared, &mut p);
    }
}

/// Build an immutable geometry snapshot of the network.
fn geo_snap(env: &Env) -> GeoSnap {
    let net = env.net.borrow();
    let dt = env.cfg.borrow().dt_ms.max(1e-6);
    let mut g = geo_from_net(&net, dt);
    g.version = env.net_version.get();
    g
}

/// Geometry snapshot straight from a `Network` (headless screenshot path).
pub fn geo_from_net(net: &Network, dt: f32) -> GeoSnap {
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

    GeoSnap {
        version: 0,
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
            .map(|i| InputSnap {
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
        dt_ms: dt,
    }
}
