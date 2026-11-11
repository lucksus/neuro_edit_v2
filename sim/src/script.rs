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
pub type ProgressFn = Rc<RefCell<dyn FnMut(&Sim) -> bool>>;

/// Scripting environment: engine + shared handles.
pub struct Env {
    pub engine: Engine,
    pub net: NetHandle,
    pub backend: BackendKind,
    pub rng: RngHandle,
    pub cfg: CfgHandle,
    sim: SimSlot,
    pub on_print: PrintFn,
    /// Bumped on every net mutation (any fn that invalidates the sim).
    pub net_version: Rc<std::cell::Cell<u64>>,
    /// Between-chunks callback; `None` in headless use.
    pub progress: Rc<RefCell<Option<ProgressFn>>>,
    /// Chunk length for progress callbacks, in steps.
    pub progress_chunk: Rc<std::cell::Cell<usize>>,
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
        {
            let p = on_print.clone();
            engine.on_print(move |s| p(s));
        }
        let net_version = Rc::new(std::cell::Cell::new(1u64));
        let progress: Rc<RefCell<Option<ProgressFn>>> = Rc::new(RefCell::new(None));
        let progress_chunk = Rc::new(std::cell::Cell::new(200usize));
        register_api(
            &mut engine,
            net.clone(),
            sim.clone(),
            rng.clone(),
            cfg.clone(),
            backend,
            net_version.clone(),
            progress.clone(),
            progress_chunk.clone(),
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
            progress,
            progress_chunk,
        }
    }

    /// Editor mode: `cb` runs between run chunks so the UI can stream state.
    pub fn set_progress(&mut self, chunk_steps: usize, cb: ProgressFn) {
        self.progress_chunk.set(chunk_steps);
        *self.progress.borrow_mut() = Some(cb);
    }

    /// Build (or rebuild) the simulator from the current network + config.
    pub fn build_sim(&self) -> anyhow::Result<Rc<RefCell<Sim>>> {
        let cfg = self.cfg.borrow().clone();
        let s = Sim::new(&self.net.borrow(), cfg, self.backend)?;
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

    /// Mark the sim stale — next `sim.run` rebuilds from the edited net.
    pub fn invalidate_sim(&self) {
        *self.sim.borrow_mut() = None;
    }

    fn scope(&self) -> rhai::Scope<'_> {
        let mut scope = rhai::Scope::new();
        scope.push("net", self.net.clone());
        scope.push("sim", self.sim.clone());
        scope.push("rng", self.rng.clone());
        scope
    }

    pub fn eval_str(&mut self, code: &str) -> Result<Dynamic, String> {
        self.engine
            .eval_with_scope::<Dynamic>(&mut self.scope(), code)
            .map_err(|e| e.to_string())
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

    pub fn run_ms(&mut self, ms: f32) {
        if let Ok(h) = self.sim_handle() {
            h.borrow_mut().run_ms(ms);
        }
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

/// Build or fetch the sim inside the slot (mirrors `Env::sim_handle`).
fn ensure_sim(
    slot: &SimSlot,
    net: &NetHandle,
    cfg: &CfgHandle,
    kind: BackendKind,
) -> Result<Rc<RefCell<Sim>>, Box<rhai::EvalAltResult>> {
    if let Some(h) = slot.borrow().as_ref() {
        return Ok(h.clone());
    }
    let c = cfg.borrow().clone();
    let s = Sim::new(&net.borrow(), c, kind)
        .map_err(|e| -> Box<rhai::EvalAltResult> { e.to_string().into() })?;
    let h = Rc::new(RefCell::new(s));
    *slot.borrow_mut() = Some(h.clone());
    Ok(h)
}

/// Net mutated: drop the stale sim and bump the version (editor republish).
fn invalidate(s: &SimSlot, ver: &Rc<std::cell::Cell<u64>>) {
    *s.borrow_mut() = None;
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
) {
    let mut p = progress.borrow_mut();
    match p.as_mut() {
        Some(cb) => {
            let n_chunk = chunk.get().max(1);
            while steps > 0 {
                let n = steps.min(n_chunk);
                h.borrow_mut().run_steps(n);
                steps -= n;
                if !(*cb.borrow_mut())(&h.borrow()) {
                    break;
                }
            }
        }
        None => h.borrow_mut().run_steps(steps),
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
    backend: BackendKind,
    net_version: Rc<std::cell::Cell<u64>>,
    progress: Rc<RefCell<Option<ProgressFn>>>,
    progress_chunk: Rc<std::cell::Cell<usize>>,
) {
    engine
        .register_type_with_name::<NetHandle>("Net")
        .register_type_with_name::<SimSlot>("Sim")
        .register_type_with_name::<RngHandle>("Rng")
        .register_type_with_name::<RngStream>("RngStream")
        .register_type_with_name::<DeltaReadout>("DeltaReadout");

    // ---- help ----
    engine.register_fn("help", || -> String {
        "API:\n  net.add_neuron(x,y,z)  net.connect(a,b,w)  net.clear()\n  net.create_2d_grid(g,w,h,d)  net.create_3d_grid(g,w,h,d,sp)\n  net.create_connections(g,scaling,lambda,w)  net.set_synapse_weights(g,mean,sd)\n  net.set_exc_inh(g,frac)  net.neurons_of(g)  net.add_input(name,[[n,w],..])\n  net.neuron_count()  net.synapse_count()  net.save(path)  net.load(path)\n  sim.run(ms)  sim.run_steps(n)  sim.time()  sim.reset()  sim.rate()  sim.spikes()\n  sim.trace()  sim.trace_of(ids)  sim.v(id)  sim.set_input(name,x)\n  sim.play_input(name,arr)  sim.set_external(id,amps)  sim.set_noise(hz,kick)\n  sim.rebuild()  sim.set_dt(dt)  sim.set_int_weights(on)\n  net.add_readout(name,units)  net.set_readout_weights(name,w)  net.readout_weights(name)\n  net.samples_write(n,t,v)  net.samples_save(n,p)  net.samples_load(n,p)\n  net.script_add(n,code)  net.script_remove(n)\n  readouts: delta_new(n) r.learn(trace,target,rate) r.output(trace)\n  fit_ridge(x_flat,T,F,y,lambda)  ridge_output(w,x)  accuracy(s,y)  auc(s,y)\n  streams: rng_stream(seed,n_src,p) .window(steps,label,pairs,strength)\n  misc: rand(a,b)  gauss(m,sd)  seed(s)  print(x)"
            .to_string()
    });

    // ---- net methods ----
    {
        let (s, v) = (sim.clone(), net_version.clone());
        engine.register_fn(
            "add_neuron",
            move |n: &mut NetHandle, x: f64, y: f64, z: f64| -> rhai::INT {
                invalidate(&s, &v);
                n.borrow_mut()
                    .add_neuron([x as f32, y as f32, z as f32], NeuronKind::Excitatory)
                    as rhai::INT
            },
        );
    }
    {
        let (s, v) = (sim.clone(), net_version.clone());
        engine.register_fn("clear", move |n: &mut NetHandle| {
            *n.borrow_mut() = Network::new();
            invalidate(&s, &v);
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
    engine.register_fn(
        "connect",
        |n: &mut NetHandle, a: rhai::INT, b: rhai::INT, w: f64| -> rhai::INT {
            n.borrow_mut().connect(a as u32, b as u32, w as f32) as rhai::INT
        },
    );
    {
        let (s, v) = (sim.clone(), net_version.clone());
        engine.register_fn(
            "create_2d_grid",
            move |n: &mut NetHandle, g: &str, w: rhai::INT, h: rhai::INT, d: f64| {
                invalidate(&s, &v);
                builders::create_2d_grid(
                    &mut n.borrow_mut(),
                    g,
                    w as usize,
                    h as usize,
                    d as f32,
                    [0.0; 3],
                );
            },
        );
    }
    {
        let (s, v) = (sim.clone(), net_version.clone());
        engine.register_fn(
            "create_3d_grid",
            move |n: &mut NetHandle,
                  g: &str,
                  w: rhai::INT,
                  h: rhai::INT,
                  dep: rhai::INT,
                  d: f64| {
                invalidate(&s, &v);
                builders::create_3d_grid(
                    &mut n.borrow_mut(),
                    g,
                    w as usize,
                    h as usize,
                    dep as usize,
                    d as f32,
                    [0.0; 3],
                );
            },
        );
    }
    {
        let (r, s, v) = (rng.clone(), sim.clone(), net_version.clone());
        engine.register_fn(
            "create_connections",
            move |n: &mut NetHandle, g: &str, scaling: f64, lambda: f64, w: f64| {
                invalidate(&s, &v);
                builders::create_connections(
                    &mut n.borrow_mut(),
                    g,
                    scaling,
                    lambda,
                    w as f32,
                    &mut r.borrow_mut(),
                );
            },
        );
    }
    {
        let (r, s, v) = (rng.clone(), sim.clone(), net_version.clone());
        engine.register_fn(
            "set_synapse_weights",
            move |n: &mut NetHandle, g: &str, mean: f64, sd: f64| {
                invalidate(&s, &v);
                builders::set_synapse_weights(
                    &mut n.borrow_mut(),
                    g,
                    mean,
                    sd,
                    &mut r.borrow_mut(),
                );
            },
        );
    }
    {
        let (r, s, v) = (rng.clone(), sim.clone(), net_version.clone());
        engine.register_fn(
            "set_exc_inh",
            move |n: &mut NetHandle, g: &str, frac: f64| {
                invalidate(&s, &v);
                builders::set_exc_inh(&mut n.borrow_mut(), g, frac, &mut r.borrow_mut());
            },
        );
    }
    {
        let (s, v) = (sim.clone(), net_version.clone());
        engine.register_fn(
            "set_izh",
            move |n: &mut NetHandle, id: rhai::INT, a: f64, b: f64, c: f64, d: f64| {
                invalidate(&s, &v);
                n.borrow_mut().neuron_mut(id as u32).model = net_format::NeuronModel::Izhikevich {
                    a: a as f32,
                    b: b as f32,
                    c: c as f32,
                    d: d as f32,
                };
            },
        );
    }
    {
        let (s, v) = (sim.clone(), net_version.clone());
        engine.register_fn("delete_neuron", move |n: &mut NetHandle, id: rhai::INT| {
            invalidate(&s, &v);
            n.borrow_mut().delete_neuron(id as u32);
        });
    }
    {
        let (s, v) = (sim.clone(), net_version.clone());
        engine.register_fn(
            "delete_neurons",
            move |n: &mut NetHandle, ids: rhai::Array| {
                invalidate(&s, &v);
                n.borrow_mut()
                    .delete_neurons(ids.iter().map(|d| d_i64(d) as u32).collect());
            },
        );
    }
    {
        let (s, v) = (sim.clone(), net_version.clone());
        engine.register_fn(
            "set_pos",
            move |n: &mut NetHandle, id: rhai::INT, x: f64, y: f64, z: f64| {
                invalidate(&s, &v);
                let mut g = n.borrow_mut();
                g.neuron_mut(id as u32).pos = [x as f32, y as f32, z as f32];
                g.refresh_delays_of(id as u32);
            },
        );
    }
    engine.register_fn("pos", |n: &mut NetHandle, id: rhai::INT| -> rhai::Array {
        let p = n.borrow().neuron(id as u32).pos;
        [p[0], p[1], p[2]]
            .iter()
            .map(|&x| dyn_f64(x as f64))
            .collect()
    });
    engine.register_fn(
        "neuron_name",
        |n: &mut NetHandle, id: rhai::INT| -> String { n.borrow().neuron(id as u32).name.clone() },
    );
    engine.register_fn(
        "set_neuron_name",
        |n: &mut NetHandle, id: rhai::INT, name: &str| {
            n.borrow_mut().neuron_mut(id as u32).name = name.to_string();
        },
    );
    engine.register_fn("neuron", |n: &mut NetHandle, name: &str| -> rhai::INT {
        n.borrow()
            .neuron_index(name)
            .map(|i| i as rhai::INT)
            .unwrap_or(-1)
    });
    engine.register_fn("neurons_of", |n: &mut NetHandle, g: &str| -> rhai::Array {
        match n.borrow().group(g) {
            Some(gr) => gr
                .members
                .iter()
                .map(|&m| Dynamic::from(m as rhai::INT))
                .collect(),
            None => Vec::new(),
        }
    });
    engine.register_fn("group_names", |n: &mut NetHandle| -> rhai::Array {
        n.borrow()
            .groups
            .iter()
            .map(|g| Dynamic::from(g.name.clone()))
            .collect()
    });
    engine.register_fn(
        "add_to_group",
        |n: &mut NetHandle, g: &str, ids: rhai::Array| {
            let mut b = n.borrow_mut();
            let gi = b.add_group(g);
            for d in ids {
                b.groups[gi].members.push(d_i64(&d) as u32);
            }
        },
    );
    {
        let (s, v) = (sim.clone(), net_version.clone());
        engine.register_fn(
            "add_input",
            move |n: &mut NetHandle, name: &str, targets: rhai::Array| {
                invalidate(&s, &v);
                let mut tw = Vec::new();
                for t in targets {
                    if let Some(arr) = t.try_cast::<rhai::Array>() {
                        if arr.len() >= 2 {
                            tw.push((d_i64(&arr[0]) as u32, d_f64(&arr[1]) as f32));
                        }
                    }
                }
                n.borrow_mut().add_input(name, tw, [0.0; 3]);
            },
        );
    }
    {
        let v = net_version.clone();
        engine.register_fn(
            "add_readout",
            move |n: &mut NetHandle, name: &str, units: rhai::Array| {
                bump(&v);
                let us: Vec<u32> = units.iter().map(|d| d_i64(d) as u32).collect();
                n.borrow_mut().add_readout(name, us);
            },
        );
    }
    {
        // Trained DeltaReadout weights → the network object (persisted in
        // .ne2, shown in the editor's weights plot).
        let v = net_version.clone();
        engine.register_fn(
            "set_readout_weights",
            move |n: &mut NetHandle, name: &str, w: rhai::Array| {
                bump(&v);
                let mut b = n.borrow_mut();
                let w: Vec<f32> = w.iter().map(|d| d_f64(d) as f32).collect();
                if let Some(r) = b.readouts.iter_mut().find(|r| r.name == name) {
                    r.weights = w;
                }
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
    engine.register_fn("save", |n: &mut NetHandle, path: &str| {
        if let Err(e) = net_format::io::save(&n.borrow(), path) {
            eprintln!("save {path}: {e}");
        }
    });
    {
        let (s, v) = (sim.clone(), net_version.clone());
        engine.register_fn(
            "load",
            move |n: &mut NetHandle, path: &str| match net_format::io::load(path) {
                Ok(loaded) => {
                    *n.borrow_mut() = loaded;
                    invalidate(&s, &v);
                }
                Err(e) => eprintln!("load {path}: {e}"),
            },
        );
    }
    engine.register_fn("net_load", move |path: &str| -> NetHandle {
        Rc::new(RefCell::new(
            net_format::io::load(path).unwrap_or_else(|_| Network::new()),
        ))
    });

    // ---- sim methods ----
    {
        let (s, n, c) = (sim.clone(), net.clone(), cfg.clone());
        let (p, pc) = (progress.clone(), progress_chunk.clone());
        engine.register_fn(
            "run",
            move |_: &mut SimSlot, ms: f64| -> Result<(), Box<rhai::EvalAltResult>> {
                let h = ensure_sim(&s, &n, &c, backend)?;
                let steps = (ms as f32 / c.borrow().dt_ms).round().max(0.0) as usize;
                run_chunked(&h, steps, &p, &pc);
                Ok(())
            },
        );
    }
    {
        let (s, n, c) = (sim.clone(), net.clone(), cfg.clone());
        let (p, pc) = (progress.clone(), progress_chunk.clone());
        engine.register_fn(
            "run_steps",
            move |_: &mut SimSlot, steps: rhai::INT| -> Result<(), Box<rhai::EvalAltResult>> {
                let h = ensure_sim(&s, &n, &c, backend)?;
                run_chunked(&h, steps.max(0) as usize, &p, &pc);
                Ok(())
            },
        );
    }
    {
        let (s, n, c) = (sim.clone(), net.clone(), cfg.clone());
        engine.register_fn(
            "build",
            move |_: &mut SimSlot| -> Result<(), Box<rhai::EvalAltResult>> {
                ensure_sim(&s, &n, &c, backend)?;
                Ok(())
            },
        );
    }
    {
        let s = sim.clone();
        engine.register_fn("rebuild", move |_: &mut SimSlot| {
            *s.borrow_mut() = None;
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
        |s: &mut SimSlot, ids: rhai::Array| -> rhai::Array {
            let Some(h) = s.borrow().as_ref().cloned() else {
                return Vec::new();
            };
            let tr = h.borrow().trace();
            ids.iter()
                .map(|d| dyn_f64(tr[d_i64(d) as usize] as f64))
                .collect()
        },
    );
    engine.register_fn("v", |s: &mut SimSlot, id: rhai::INT| -> f64 {
        s.borrow()
            .as_ref()
            .map(|h| h.borrow().v()[id as usize] as f64)
            .unwrap_or(0.0)
    });
    engine.register_fn(
        "spike_counts",
        |s: &mut SimSlot, rows: rhai::INT| -> rhai::Array {
            s.borrow()
                .as_ref()
                .map(|h| {
                    h.borrow()
                        .spike_counts(rows as usize)
                        .into_iter()
                        .map(|f| dyn_f64(f as f64))
                        .collect()
                })
                .unwrap_or_default()
        },
    );
    engine.register_fn(
        "spike_counts_of",
        |s: &mut SimSlot, rows: rhai::INT, ids: rhai::Array| -> rhai::Array {
            let Some(h) = s.borrow().as_ref().cloned() else {
                return Vec::new();
            };
            let c = h.borrow().spike_counts(rows as usize);
            ids.iter()
                .map(|d| dyn_f64(c[d_i64(d) as usize] as f64))
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
        let (s, n, c) = (sim.clone(), net.clone(), cfg.clone());
        engine.register_fn(
            "set_input",
            move |_: &mut SimSlot, name: &str, x: f64| -> Result<(), Box<rhai::EvalAltResult>> {
                let idx = match n.borrow().input_index(name) {
                    Some(i) => i,
                    None => return Ok(()),
                };
                let h = ensure_sim(&s, &n, &c, backend)?;
                h.borrow_mut().set_input_constant(idx, x as f32);
                Ok(())
            },
        );
    }
    {
        let (s, n, c) = (sim.clone(), net.clone(), cfg.clone());
        engine.register_fn(
            "play_input",
            move |_: &mut SimSlot,
                  name: &str,
                  arr: rhai::Array|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let idx = match n.borrow().input_index(name) {
                    Some(i) => i,
                    None => return Ok(()),
                };
                let series: Vec<f32> = arr.iter().map(|d| d_f64(d) as f32).collect();
                let h = ensure_sim(&s, &n, &c, backend)?;
                h.borrow_mut().play_input(idx, series);
                Ok(())
            },
        );
    }
    {
        let (s, n, c) = (sim.clone(), net.clone(), cfg.clone());
        engine.register_fn(
            "set_external",
            move |_: &mut SimSlot,
                  id: rhai::INT,
                  amps: f64|
                  -> Result<(), Box<rhai::EvalAltResult>> {
                let h = ensure_sim(&s, &n, &c, backend)?;
                h.borrow_mut().set_external(id as usize, amps as f32);
                Ok(())
            },
        );
    }
    {
        let (c, s) = (cfg.clone(), sim.clone());
        engine.register_fn("set_noise", move |_: &mut SimSlot, hz: f64, kick: f64| {
            c.borrow_mut().noise_rate_hz = hz as f32;
            c.borrow_mut().noise_kick_mv = kick as f32;
            if let Some(h) = s.borrow().as_ref() {
                h.borrow_mut().set_noise(hz as f32, kick as f32);
            }
        });
    }
    {
        let c = cfg.clone();
        engine.register_fn("set_synapse_tau", move |_: &mut SimSlot, tau: f64| {
            c.borrow_mut().synapse_tau_ms = tau as f32;
        });
    }
    {
        let c = cfg.clone();
        engine.register_fn("set_readout_tau", move |_: &mut SimSlot, tau: f64| {
            c.borrow_mut().readout_tau_ms = tau as f32;
        });
    }
    {
        let (c, s) = (cfg.clone(), sim.clone());
        engine.register_fn("set_dt", move |_: &mut SimSlot, dt: f64| {
            c.borrow_mut().dt_ms = dt as f32;
            *s.borrow_mut() = None; // ring depth depends on dt
        });
    }
    {
        let (c, s) = (cfg.clone(), sim.clone());
        engine.register_fn("set_int_weights", move |_: &mut SimSlot, on: rhai::INT| {
            c.borrow_mut().int_weights = on != 0;
            *s.borrow_mut() = None; // switches the kernel + ring semantics
        });
    }

    // ---- readouts ----
    engine.register_fn("delta_new", |n: rhai::INT| -> DeltaReadout {
        DeltaReadout::new(n as usize)
    });
    {
        let r = rng.clone();
        engine.register_fn(
            "init_random",
            move |ro: &mut DeltaReadout, lo: f64, hi: f64| {
                ro.init_weights_random(lo as f32, hi as f32, &mut r.borrow_mut());
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
        |ro: &mut DeltaReadout, trace: rhai::Array, target: f64, rate: f64| {
            let t: Vec<f32> = trace.iter().map(|d| d_f64(d) as f32).collect();
            ro.learn(&t, target as f32, rate as f32);
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
         lambda: f64|
         -> rhai::Array {
            let xf: Vec<f32> = x.iter().map(|d| d_f64(d) as f32).collect();
            let yf: Vec<f32> = y.iter().map(|d| d_f64(d) as f32).collect();
            let w = fit_ridge(&xf, n_rows as usize, n_feat as usize, &yf, lambda as f32);
            w.iter().map(|&v| dyn_f64(v)).collect()
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
        engine.register_fn("rand", move |lo: f64, hi: f64| -> f64 {
            r.borrow_mut().uniform(lo, hi)
        });
    }
    {
        let r = rng.clone();
        engine.register_fn("gauss", move |m: f64, s: f64| -> f64 {
            r.borrow_mut().gauss(m, s)
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
        |seed: rhai::INT, n_src: rhai::INT, p: f64| -> RngStream {
            RngStream::new(seed as u64, n_src as usize, p as f32)
        },
    );
    engine.register_fn(
        "window",
        |rs: &mut RngStream,
         steps: rhai::INT,
         label: rhai::INT,
         pairs: rhai::Array,
         strength: f64|
         -> rhai::Array {
            let mut pair_idx = Vec::new();
            for p in pairs {
                if let Some(a) = p.try_cast::<rhai::Array>() {
                    if a.len() >= 2 {
                        pair_idx.push((d_i64(&a[0]) as usize, d_i64(&a[1]) as usize));
                    }
                }
            }
            let m = rs.window(steps as usize, label != 0, &pair_idx, strength as f32);
            m.iter()
                .map(|row| {
                    row.iter()
                        .map(|&v| dyn_f64(v as f64))
                        .collect::<rhai::Array>()
                        .into()
                })
                .collect()
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
        let v = net_version.clone();
        engine.register_fn(
            "samples_save",
            move |n: &mut NetHandle, name: &str, path: &str| {
                bump(&v);
                let b = n.borrow();
                if let Some(s) = b.samples(name) {
                    if let Err(e) = net_format::io::save_samples(s, path) {
                        eprintln!("samples_save {path}: {e}");
                    }
                }
            },
        );
    }
    {
        let v = net_version.clone();
        engine.register_fn(
            "samples_load",
            move |n: &mut NetHandle, name: &str, path: &str| match net_format::io::load_samples(
                path, name,
            ) {
                Ok(s) => {
                    bump(&v);
                    let mut b = n.borrow_mut();
                    let slot = b.samples_mut(name);
                    slot.times = s.times;
                    slot.values = s.values;
                }
                Err(e) => eprintln!("samples_load {path}: {e}"),
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
    engine.register_fn("write_to_file", |path: &str, values: rhai::Array| {
        let mut out = String::new();
        for v in values {
            out.push_str(&format!("{}\n", d_f64(&v)));
        }
        if let Err(e) = std::fs::write(path, out) {
            eprintln!("write_to_file {path}: {e}");
        }
    });
    engine.register_fn("read_to_vec", |path: &str| -> rhai::Array {
        std::fs::read_to_string(path)
            .map(|s| {
                s.lines()
                    .filter_map(|l| l.trim().parse::<f64>().ok())
                    .map(dyn_f64)
                    .collect()
            })
            .unwrap_or_default()
    });
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
