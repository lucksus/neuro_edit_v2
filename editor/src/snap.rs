//! Shared state between the sim worker thread and the UI.
//!
//! The worker owns the network + simulator + script engine. The UI never
//! touches them; it renders `Shared` (written under a short write lock once
//! per publish chunk) and sends `Cmd`s. The sim is never blocked by the UI:
//! publishing is the worker's choice, and heavy payloads (geometry) are
//! `Arc`-shared snapshots replaced wholesale.

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

/// Immutable geometry snapshot of the whole network.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct GeoSnap {
    pub version: u64,
    pub pos: Vec<[f32; 3]>,
    /// 0 = excitatory, 1 = inhibitory.
    pub kind: Vec<u8>,
    pub names: Vec<String>,
    /// (src, tgt) pairs, source order.
    pub syn: Vec<(u32, u32)>,
    /// Per-synapse delay in sim steps (for spike-dot travel time).
    pub syn_delay_steps: Vec<u16>,
    /// First K display synapses per neuron: (tgt, delay_steps) flattened.
    pub disp_edges: Vec<(u32, u16)>,
    /// Start index of neuron i's display edges in `disp_edges`.
    pub disp_ptr: Vec<u32>,
    pub groups: Vec<(String, Vec<u32>)>,
    /// (name, targets) — inputs drawn as markers.
    pub inputs: Vec<InputSnap>,
    /// (name, units, weights) — weights may be empty until a script pushes
    /// them via `net.set_readout_weights`.
    pub readouts: Vec<(String, Vec<u32>, Vec<f32>)>,
    /// (name, times, values) — plotted in the samples tab.
    pub samples: Vec<(String, Vec<f32>, Vec<f32>)>,
    /// (name, code) — script list saved in the file.
    pub scripts: Vec<(String, String)>,
    /// Groups whose members form a regular lattice (`Group::grid`), with
    /// the member list reordered into texel order. The plane view draws
    /// these instead of the member cubes.
    pub grids: Vec<GridSnap>,
    pub dt_ms: f32,
}

impl GeoSnap {
    /// Floor-grid height: 20 units under the deepest neuron (the original
    /// hard-codes y=0, which cuts through nets whose lattice extends below
    /// zero and makes deep nets unfocusable).
    pub fn floor_y(&self) -> f32 {
        self.pos
            .iter()
            .map(|p| p[1])
            .reduce(f32::min)
            .unwrap_or(0.0)
            - 20.0
    }
}

#[derive(Clone)]
pub struct InputSnap {
    pub name: String,
    pub pos: [f32; 3],
    pub targets: Vec<u32>,
}

/// A grid group turned renderable: lattice metadata plus the member list
/// re-ordered into texel order (position k = lattice index k).
#[derive(Clone)]
pub struct GridSnap {
    /// Index into `groups`.
    pub group: usize,
    /// Texel counts per lattice dim `[w, h, d]` — `d == 1` is a 2D grid.
    pub dims: [u32; 3],
    /// World axis (0=x, 1=y, 2=z) each lattice dim runs along.
    pub axes: [u8; 3],
    /// World position of lattice texel (0,0,0).
    pub origin: [f32; 3],
    /// World distance between neighbouring texels.
    pub spacing: f32,
    /// Texel-ordered member list (neuron ids), `len == w*h*d`.
    pub members: Vec<u32>,
}

/// Dynamic state published every chunk while the sim runs.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct Snap {
    pub seq: u64,
    pub t_steps: u64,
    pub time_ms: f64,
    pub spikes_total: u64,
    pub rate_hz: f64,
    /// Exp-filtered activity trace per neuron (drives instance colour).
    pub activity: Arc<Vec<f32>>,
    /// Per-neuron spike counts since the previous snap (particles, raster).
    pub counts: Arc<Vec<u32>>,
    /// New `v` samples for watched neurons, per id.
    pub watched_v: Vec<(u32, Arc<Vec<f32>>)>,
    /// New `u` samples for watched neurons — same ids/lengths as
    /// `watched_v`, feeding the v–u phase plot.
    pub watched_u: Vec<(u32, Arc<Vec<f32>>)>,
    /// Spike-trace value per texel for each `GeoSnap::grids` entry
    /// (same order; `exp` decay, tau ~20 ms — see `simctl::PLANE_TAU_MS`).
    pub grid_traces: Vec<Arc<Vec<f32>>>,
    /// Live constant drive per input (same order as `GeoSnap::inputs`).
    pub input_values: Arc<Vec<f32>>,
    /// Measured wall-time speed (steps/s) while running.
    pub steps_per_sec: f32,
}

#[allow(dead_code)]
pub struct Shared {
    pub geo: Option<Arc<GeoSnap>>,
    /// Net-replacement generation (New / load / `net.clear()`) — the UI
    /// resets selection, watches and plots when this changes.
    pub net_gen: u64,
    /// Last net version the server confirmed (Geo / GeoDelta) — the base
    /// for `ClientMsg::Edit`.
    pub net_version: u64,
    /// Edits sent but not yet confirmed — `send_edit` bases are
    /// `net_version + edits_in_flight`.
    pub edits_in_flight: u64,
    pub snap: Snap,
    /// Worker is in continuous run mode.
    pub running: bool,
    /// A `sim` instance exists on the worker.
    pub sim_built: bool,
    /// A blocking eval/run is in flight (UI shows spinner, input muted).
    pub busy: bool,
    /// Console output lines (print + results + errors + log).
    pub console: VecDeque<String>,
    /// Live sim configuration — the toolbar reads these so script calls
    /// (`sim.set_dt`, `sim.set_noise`) show up instead of stale values.
    pub cfg: proto::Config,
    /// Live backend choice on the worker.
    pub backend: proto::Backend,
    /// Worker noticed the backend failed and fell back.
    pub notice: Option<String>,
    /// JSON subnet staged by a copy/cut command for the UI to put on the
    /// system clipboard.
    pub clipboard: Option<String>,
    /// Last `ListFiles` reply (dir path + entries) — feeds the file
    /// pickers; client-side fs is never touched.
    pub files: Option<(String, Vec<proto::FileEntry>)>,
    /// `ReadFile` reply, consumed once by the script panel.
    pub file_data: Option<(String, String)>,
    /// `WriteFile` ack for status display.
    pub file_written: Option<String>,
    /// `Usage` reply — (used_bytes, quota_bytes); quota 0 = unlimited.
    pub usage: Option<(u64, u64)>,
    /// Completed file op that should trigger a list refresh
    /// (`DirCreated`/`FileDeleted`/`FileRenamed`/`UploadReady` paths).
    pub file_op: Option<String>,
    /// Streaming `FileChunk` replies accumulate here until `done` —
    /// consumed by the Files window's download path.
    pub download: Option<(String, Vec<u8>, bool)>,
    /// Server job list (`ListJobs` refreshed, JobStarted appended,
    /// JobDone re-stamps status).
    pub jobs: Vec<proto::JobInfo>,
    /// Accumulated `JobProgress` rows per job id — feeds the panel's
    /// progress plot.
    pub job_progress: std::collections::BTreeMap<u64, Vec<Vec<(String, f64)>>>,
    /// `OpenJob` reply, consumed once by the jobs panel.
    pub job_detail: Option<JobDetailSnap>,
}

/// Snapshot of an `OpenJob` reply for the jobs panel.
pub struct JobDetailSnap {
    pub id: u64,
    pub name: String,
    pub status: String,
    pub script: String,
    pub result_json: Option<String>,
    pub progress: Vec<Vec<(String, f64)>>,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            geo: None,
            net_gen: 0,
            net_version: 0,
            edits_in_flight: 0,
            snap: Snap::default(),
            running: false,
            sim_built: false,
            busy: false,
            console: VecDeque::with_capacity(512),
            cfg: proto::Config::default(),
            backend: proto::Backend::Cpu,
            notice: None,
            clipboard: None,
            files: None,
            file_data: None,
            file_written: None,
            usage: None,
            file_op: None,
            download: None,
            jobs: Vec::new(),
            job_progress: std::collections::BTreeMap::new(),
            job_detail: None,
        }
    }
}

pub type SharedRef = Arc<RwLock<Shared>>;

// ---------------------------------------------------------------------------
// ServerMsg → Shared folding (the local transport's read model)
// ---------------------------------------------------------------------------

/// Client-side fold state: things the wire sends incrementally but the UI
/// keeps as whole values (u8 trace accumulators, per-source display edges,
/// the non-grid index space).
#[derive(Default)]
pub struct Fold {
    /// Per-grid u8 trace planes — `GridTraces` sets them whole, each
    /// `Activity` decays them by `exp(-Δms / PLANE_TAU)` and stamps the
    /// spiked texels' absolute values (`grid_delta`).
    trace_u8: Vec<Vec<u8>>,
    /// Per-source display edges — regenerated into `GeoSnap`'s flat
    /// `disp_edges`/`disp_ptr` on every `disp_changed`.
    disp: Vec<Vec<(u32, u16)>>,
    /// `Geo.non_grid` — index space of `Activity::non_grid_spiked`.
    non_grid: Vec<u32>,
    /// u8 spike-trace glow per neuron id — decays by `(1-dt/tau_r)^Δsteps`
    /// per frame; `non_grid_spiked` stamps absolute values; grid members
    /// read their texel instead.
    act_u8: Vec<u8>,
    /// `Activity` anchors for the decay windows.
    last_time_ms: f64,
    last_t_steps: u64,
}

impl Fold {
    /// Apply one `ServerMsg` to `Shared`. `to:` routing is ignored — the
    /// local client is the only subscriber.
    pub fn apply(&mut self, sh: &mut Shared, msg: proto::ServerMsg) {
        match msg {
            proto::ServerMsg::Hello {
                proto,
                net_version,
                net_gen,
            } => {
                if proto != proto::PROTO_VERSION {
                    sh.log(format!(
                        "! protocol mismatch: server speaks {proto}, we speak {}",
                        proto::PROTO_VERSION
                    ));
                }
                sh.net_version = net_version;
                sh.net_gen = net_gen;
                sh.edits_in_flight = 0;
            }
            proto::ServerMsg::Geo(g) => {
                self.disp = split_disp(&g.disp_ptr, &g.disp_edges);
                self.non_grid = g.non_grid.clone();
                // planes re-arrive via GridTraces; until then, zeros
                self.trace_u8 = g
                    .grids
                    .iter()
                    .map(|gr| vec![0u8; gr.members.len()])
                    .collect();
                self.act_u8 = vec![0u8; g.pos.len()];
                self.last_time_ms = 0.0;
                self.last_t_steps = 0;
                sh.geo = Some(Arc::new(geo_snap_from(&g)));
                sh.net_version = g.version;
                sh.net_gen = g.gen;
                sh.edits_in_flight = 0;
            }
            proto::ServerMsg::GeoDelta(d) => {
                let gen = d.gen;
                if let Some(g) = sh.geo.as_deref().cloned() {
                    let mut g = g;
                    apply_delta(&mut g, &mut self.disp, &mut self.non_grid, *d);
                    sh.geo = Some(Arc::new(g));
                }
                if let Some(g) = &sh.geo {
                    sh.net_version = g.version;
                }
                // Deltas carry the wire gen — a lagging client must still
                // reset neuron-id-keyed state if a replacement slipped in.
                sh.net_gen = gen;
                sh.edits_in_flight = 0;
            }
            proto::ServerMsg::Status {
                running,
                busy,
                sim_built,
                backend,
                cfg,
                notice,
            } => {
                sh.running = running;
                sh.busy = busy;
                sh.sim_built = sim_built;
                sh.backend = backend;
                sh.cfg = cfg;
                sh.notice = notice;
            }
            proto::ServerMsg::Console(s) => sh.log(s),
            proto::ServerMsg::Activity(a) => self.fold_activity(sh, *a),
            proto::ServerMsg::GridTraces { grids, .. } => {
                self.trace_u8 = grids;
                sh.snap.grid_traces = self.traces_f32();
            }
            proto::ServerMsg::Clipboard(txt) => sh.clipboard = Some(txt),
            proto::ServerMsg::FileList { path, entries } => {
                sh.files = Some((path, entries));
            }
            proto::ServerMsg::FileData { path, data } => {
                sh.file_data = Some((path, String::from_utf8_lossy(&data).into_owned()));
            }
            proto::ServerMsg::FileWritten { path } => {
                sh.file_written = Some(path.clone());
                sh.log(format!("wrote {path}"));
            }
            proto::ServerMsg::JobStarted { id, name } => {
                sh.jobs.push(proto::JobInfo {
                    id,
                    name: name.clone(),
                    status: "running".into(),
                    started_ms: 0,
                });
                sh.job_progress.insert(id, Vec::new());
                sh.log(format!("job {id} started: {name}"));
            }
            proto::ServerMsg::JobProgress { id, fields } => {
                sh.job_progress.entry(id).or_default().push(fields);
            }
            proto::ServerMsg::JobDone { id, status, error } => {
                match sh.jobs.iter_mut().find(|j| j.id == id) {
                    Some(j) => j.status = status.clone(),
                    None => sh.jobs.push(proto::JobInfo {
                        id,
                        name: format!("job{id}"),
                        status: status.clone(),
                        started_ms: 0,
                    }),
                }
                match error {
                    Some(e) => sh.log(format!("job {id} {status}: {e}")),
                    None => sh.log(format!("job {id} {status}")),
                }
            }
            proto::ServerMsg::JobList { jobs } => {
                sh.jobs = jobs;
                for j in &sh.jobs {
                    sh.job_progress.entry(j.id).or_default();
                }
            }
            proto::ServerMsg::JobDetail {
                id,
                name,
                status,
                script,
                result_json,
                progress,
            } => {
                sh.job_detail = Some(JobDetailSnap {
                    id,
                    name,
                    status,
                    script,
                    result_json,
                    progress,
                });
            }
            proto::ServerMsg::DirCreated { path } => {
                sh.file_op = Some(format!("mkdir {path}"));
            }
            proto::ServerMsg::FileDeleted { path } => {
                sh.file_op = Some(format!("deleted {path}"));
            }
            proto::ServerMsg::FileRenamed { from, to } => {
                sh.file_op = Some(format!("renamed {from} → {to}"));
            }
            proto::ServerMsg::Usage { used, quota } => {
                sh.usage = Some((used, quota));
            }
            proto::ServerMsg::UploadReady { path } => {
                sh.file_op = Some(format!("upload ready {path}"));
            }
            proto::ServerMsg::FileChunk { path, data, done } => {
                let entry = sh
                    .download
                    .get_or_insert_with(|| (path.clone(), Vec::new(), false));
                if entry.0 == path {
                    entry.1.extend_from_slice(&data);
                    entry.2 = done;
                }
            }
            proto::ServerMsg::Error(e) => sh.log(format!("! {e}")),
        }
    }

    fn traces_f32(&self) -> Vec<Arc<Vec<f32>>> {
        self.trace_u8
            .iter()
            .map(|t| Arc::new(t.iter().map(|&v| v as f32 / 255.0).collect()))
            .collect()
    }

    fn fold_activity(&mut self, sh: &mut Shared, a: proto::Activity) {
        let n = sh.geo.as_ref().map(|g| g.pos.len()).unwrap_or(0);
        // ---- client-side decay (the wire only carries spiked entries) ----
        let dms = (a.time_ms - self.last_time_ms).max(0.0);
        let dsteps = a.t_steps.saturating_sub(self.last_t_steps);
        self.last_time_ms = a.time_ms;
        self.last_t_steps = a.t_steps;
        let f_plane = (-dms / proto::PLANE_TAU_MS as f64).exp();
        let f_act = {
            let dt = sh.cfg.dt_ms.max(1e-6) as f64;
            let tau = sh.cfg.readout_tau_ms.max(1e-6) as f64;
            (1.0 - dt / tau).powi(dsteps as i32)
        };
        let lut = |f: f64| -> [u8; 256] { std::array::from_fn(|v| (v as f64 * f).round() as u8) };
        let (plane_lut, act_lut) = (lut(f_plane), lut(f_act));
        for t in self.trace_u8.iter_mut() {
            for v in t.iter_mut() {
                *v = plane_lut[*v as usize];
            }
        }
        if self.act_u8.len() < n {
            self.act_u8.resize(n, 0);
        }
        for v in self.act_u8.iter_mut() {
            *v = act_lut[*v as usize];
        }

        // ---- stamp spiked texels/neurons (derivable from spike_bits) ----
        let bit = |id: u32| {
            let i = id as usize;
            i / 8 < a.spike_bits.len() && a.spike_bits[i / 8] >> (i % 8) & 1 == 1
        };
        if let Some(g) = &sh.geo {
            for (gi, gr) in g.grids.iter().enumerate() {
                if let Some(t) = self.trace_u8.get_mut(gi) {
                    for (k, &m) in gr.members.iter().enumerate() {
                        if bit(m) {
                            t[k] = 255;
                        }
                    }
                }
            }
        }
        for &id in &self.non_grid {
            if bit(id) {
                if let Some(dst) = self.act_u8.get_mut(id as usize) {
                    *dst = 255;
                }
            }
        }

        let nb = a.spike_bits.len() * 8;
        let mut counts = vec![0u32; nb];
        for (i, c) in counts.iter_mut().enumerate() {
            if a.spike_bits[i / 8] >> (i % 8) & 1 == 1 {
                *c = 1;
            }
        }

        // per-neuron glow: non-grid slots from the client-side decay +
        // stamped spikes; grid members decode from their texel
        let mut activity = vec![0f32; counts.len()];
        for (id, &q) in self.act_u8.iter().enumerate() {
            if let Some(dst) = activity.get_mut(id) {
                *dst = q as f32 / 255.0;
            }
        }
        if let Some(g) = &sh.geo {
            for (gi, gr) in g.grids.iter().enumerate() {
                let Some(t) = self.trace_u8.get(gi) else {
                    continue;
                };
                for (k, &m) in gr.members.iter().enumerate() {
                    if let (Some(&v), Some(dst)) = (t.get(k), activity.get_mut(m as usize)) {
                        *dst = v as f32 / 255.0;
                    }
                }
            }
        }

        sh.snap = Snap {
            seq: a.seq,
            t_steps: a.t_steps,
            time_ms: a.time_ms,
            spikes_total: a.spikes_total,
            rate_hz: a.rate_hz,
            activity: Arc::new(activity),
            counts: Arc::new(counts),
            watched_v: a
                .watched_v
                .into_iter()
                .map(|(id, v)| (id, Arc::new(v)))
                .collect(),
            watched_u: a
                .watched_u
                .into_iter()
                .map(|(id, v)| (id, Arc::new(v)))
                .collect(),
            grid_traces: self.traces_f32(),
            input_values: Arc::new(a.input_values),
            steps_per_sec: a.steps_per_sec,
        };
        sh.sim_built = true;
    }
}

fn split_disp(ptr: &[u32], edges: &[(u32, u16)]) -> Vec<Vec<(u32, u16)>> {
    ptr.windows(2)
        .map(|w| edges[w[0] as usize..w[1] as usize].to_vec())
        .collect()
}

fn flatten_disp(disp: &[Vec<(u32, u16)>]) -> (Vec<(u32, u16)>, Vec<u32>) {
    let mut edges = Vec::new();
    let mut ptr = Vec::with_capacity(disp.len() + 1);
    for d in disp {
        ptr.push(edges.len() as u32);
        edges.extend_from_slice(d);
    }
    ptr.push(edges.len() as u32);
    (edges, ptr)
}

/// Wire `Geo` → the UI's `GeoSnap` (identical fields, editor-side types).
pub fn geo_snap_from(g: &proto::Geo) -> GeoSnap {
    GeoSnap {
        version: g.version,
        pos: g.pos.clone(),
        kind: g.kind.clone(),
        names: g.names.clone(),
        syn: g.syn.clone(),
        syn_delay_steps: g.syn_delay_steps.clone(),
        disp_edges: g.disp_edges.clone(),
        disp_ptr: g.disp_ptr.clone(),
        groups: g.groups.clone(),
        inputs: g
            .inputs
            .iter()
            .map(|i| InputSnap {
                name: i.name.clone(),
                pos: i.pos,
                targets: i.targets.clone(),
            })
            .collect(),
        readouts: g.readouts.clone(),
        samples: g.samples.clone(),
        scripts: g.scripts.clone(),
        grids: g
            .grids
            .iter()
            .map(|gr| GridSnap {
                group: gr.group,
                dims: gr.dims,
                axes: gr.axes,
                origin: gr.origin,
                spacing: gr.spacing,
                members: gr.members.clone(),
            })
            .collect(),
        dt_ms: g.dt_ms,
    }
}

/// Apply a wire `GeoDelta` to a `GeoSnap` in place. `disp`/`non_grid` are
/// the per-source edge list and index space this fold maintains —
/// `disp_changed` patches `disp` which is then re-flattened into the
/// snapshot.
fn apply_delta(
    g: &mut GeoSnap,
    disp: &mut Vec<Vec<(u32, u16)>>,
    non_grid: &mut Vec<u32>,
    d: proto::GeoDelta,
) {
    g.version = d.version;
    if !d.non_grid.is_empty() {
        *non_grid = d.non_grid;
    }

    for r in d.neurons_added {
        // appended ids must line up — a gap means we missed a Geo
        debug_assert_eq!(r.id as usize, g.pos.len());
        g.pos.push(r.pos);
        g.kind.push(r.kind);
        g.names.push(r.name);
        disp.push(Vec::new());
    }
    for r in d.neurons_changed {
        let i = r.id as usize;
        if i < g.pos.len() {
            g.pos[i] = r.pos;
            g.kind[i] = r.kind;
            g.names[i] = r.name;
        }
    }
    for s in d.syn_added {
        g.syn.push((s.src, s.tgt));
        g.syn_delay_steps.push(s.delay_steps);
    }
    for (src, tgt) in d.syn_removed {
        if let Some(i) = g.syn.iter().position(|&(a, b)| a == src && b == tgt) {
            g.syn.remove(i);
            g.syn_delay_steps.remove(i);
        }
    }
    for (i, steps) in d.syn_delays {
        if let Some(ds) = g.syn_delay_steps.get_mut(i as usize) {
            *ds = steps;
        }
    }
    for (src, edges) in d.disp_changed {
        if let Some(slot) = disp.get_mut(src as usize) {
            *slot = edges;
        }
    }
    let (edges, ptr) = flatten_disp(disp);
    g.disp_edges = edges;
    g.disp_ptr = ptr;
    for (i, name, members) in d.groups_added {
        let i = i as usize;
        if i <= g.groups.len() {
            g.groups.insert(i, (name, members));
        }
    }
    for (i, name, members) in d.groups_changed {
        if let Some(gr) = g.groups.get_mut(i as usize) {
            *gr = (name, members);
        }
    }
    for rem in d.grids_removed {
        g.grids.retain(|gr| gr.group != rem as usize);
    }
    for gr in d.grids_changed {
        let snap = GridSnap {
            group: gr.group,
            dims: gr.dims,
            axes: gr.axes,
            origin: gr.origin,
            spacing: gr.spacing,
            members: gr.members,
        };
        match g.grids.iter_mut().find(|x| x.group == snap.group) {
            Some(slot) => *slot = snap,
            None => g.grids.push(snap),
        }
    }
    if let Some(v) = d.inputs {
        g.inputs = v
            .into_iter()
            .map(|i| InputSnap {
                name: i.name,
                pos: i.pos,
                targets: i.targets,
            })
            .collect();
    }
    if let Some(v) = d.readouts {
        g.readouts = v;
    }
    if let Some(v) = d.samples {
        g.samples = v;
    }
    if let Some(v) = d.scripts {
        g.scripts = v;
    }
    g.dt_ms = d.dt_ms;
}

impl Shared {
    pub fn log(&mut self, s: impl Into<String>) {
        let mut s = s.into();
        // Eval results like `net.neurons_of("g")` can return multi-KB
        // arrays — shorten the line and keep the tail count honest.
        const MAX: usize = 400;
        if s.len() > MAX {
            let extra = s.len() - MAX;
            s.truncate(MAX);
            // don't leave half a UTF-8 char at the cut point
            while !s.is_char_boundary(s.len()) {
                s.pop();
            }
            s.push_str(&format!(" … ({extra} chars truncated)"));
        }
        if self.console.len() >= 500 {
            self.console.pop_front();
        }
        self.console.push_back(s);
    }
}

#[cfg(test)]
mod floor_tests {
    use super::*;

    #[test]
    fn floor_tracks_deepest_neuron() {
        let g = GeoSnap {
            pos: vec![[10.0, 30.0, 0.0], [-5.0, -200.0, 4.0], [0.0, -80.0, 0.0]],
            ..Default::default()
        };
        assert_eq!(g.floor_y(), -220.0);
    }

    #[test]
    fn empty_net_floors_below_origin() {
        assert_eq!(GeoSnap::default().floor_y(), -20.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geo_delta_propagates_gen() {
        let mut fold = Fold::default();
        let mut sh = Shared::default();
        let geo = proto::Geo {
            version: 1,
            gen: 1,
            pos: vec![[0.0; 3]; 3],
            ..Default::default()
        };
        fold.apply(&mut sh, proto::ServerMsg::Geo(Box::new(geo)));
        assert_eq!(sh.net_gen, 1);
        // A delta crossing a generation boundary must move the client's
        // generation too — otherwise neuron-id-keyed state (selection,
        // watches, plots) is not reset.
        let d = proto::GeoDelta {
            version: 2,
            gen: 2,
            ..Default::default()
        };
        fold.apply(&mut sh, proto::ServerMsg::GeoDelta(Box::new(d)));
        assert_eq!(sh.net_gen, 2);
    }
}
