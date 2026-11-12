use serde::{Deserialize, Serialize};

use crate::{
    DEFAULT_CONDUCTION_SPEED, DEFAULT_DT_MS, DEFAULT_READOUT_TAU_MS, DEFAULT_SYNAPSE_TAU_MS,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum NeuronModel {
    /// Izhikevich 2003 model. v' = 0.04v^2+5v+140-u+I, u' = a(bv-u),
    /// spike at v>=30 then v=c, u+=d on the next step.
    Izhikevich { a: f32, b: f32, c: f32, d: f32 },
}

impl Default for NeuronModel {
    /// Regular-spiking defaults, as in the original Neuron constructor.
    fn default() -> Self {
        Self::regular_spiking()
    }
}

impl NeuronModel {
    /// Regular-spiking defaults (cortical RS cell).
    pub fn regular_spiking() -> Self {
        NeuronModel::Izhikevich {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
        }
    }
    /// Fast-spiking defaults — the standard inhibitory interneuron
    /// parameters; applied by `Network::set_kind` when a neuron's model
    /// still holds untouched defaults (see that function).
    pub fn fast_spiking() -> Self {
        NeuronModel::Izhikevich {
            a: 0.1,
            b: 0.2,
            c: -65.0,
            d: 2.0,
        }
    }
}

impl NeuronModel {
    pub fn params(&self) -> [f32; 4] {
        match *self {
            NeuronModel::Izhikevich { a, b, c, d } => [a, b, c, d],
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum NeuronKind {
    Excitatory,
    Inhibitory,
    /// Non-spiking annotation nodes drawn in the editor (inputs, readouts are
    /// separate lists, not neurons).
    Regular,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Neuron {
    pub name: String,
    pub pos: [f32; 3],
    pub model: NeuronModel,
    pub kind: NeuronKind,
}

/// One synapse = the original's (axon, synapse) pair collapsed: the axon only
/// carried a delay (distance/speed, fixed at build or move time) and the
/// synapse only a weight, so one record suffices.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Synapse {
    pub src: u32,
    pub tgt: u32,
    pub w: f32,
    /// Axonal delay in ms (distance / conduction speed at creation time).
    pub delay_ms: f32,
    /// Per-synapse current-trace time constant in ms. `0.0` = follow the
    /// simulation-wide `synapse_tau_ms` (the merged-current fast path —
    /// a network where every synapse has `tau_ms == 0` runs exactly the
    /// pre-per-synapse code). `> 0` gives this synapse its own
    /// exponential current trace decaying with `x *= 1 - dt/tau_ms`.
    #[serde(default)]
    pub tau_ms: f32,
}

/// A named set of neuron indices — the original's `Group`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Group {
    pub name: String,
    /// Indices into `Network::neurons`, stored in creation order.
    pub members: Vec<u32>,
    /// Regular-lattice layout when the group was created as a grid. Purely
    /// descriptive metadata — the editor uses it to draw the group as a
    /// textured activity plane instead of per-neuron cubes. `serde(default)`
    /// keeps JSON written before the field existed loadable; the bincode
    /// path handles the same via its format version (see `io.rs`).
    #[serde(default)]
    pub grid: Option<GridShape>,
}

/// Regular-lattice layout of a group's members, as created by
/// `builders::create_2d_grid` / `create_3d_grid`.
///
/// Convention: `members[k]` sits at lattice index `k`, decomposed as
/// `k = (i2 * dims[0] + i0) * dims[1] + i1` — lattice dim 1 varies fastest,
/// then dim 0, then dim 2. Each lattice dim runs along a world axis given
/// by `axes` (for a 2D grid, `dims[2] == 1` and `axes[2]` is the plane
/// normal). World position of lattice index `i` is
/// `origin + spacing * (i0*unit(axes[0]) + i1*unit(axes[1]) + i2*unit(axes[2]))`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct GridShape {
    /// Texel counts per lattice dim: `[w, h, d]`. `d == 1` is a 2D grid.
    pub dims: [u32; 3],
    /// World position of lattice texel (0,0,0).
    pub origin: [f32; 3],
    /// World distance between neighbouring texels.
    pub spacing: f32,
    /// World axis (0=x, 1=y, 2=z) each lattice dim runs along.
    /// `create_2d_grid` produces `[2, 1, 0]` (w along +z, h along +y,
    /// normal +x); `create_3d_grid` produces `[0, 1, 2]`.
    pub axes: [u8; 3],
}

/// Upper bound on `w*h*d` for a single grid-create call (script or
/// wire edit). Guards the `as usize`/`Vec::with_capacity` paths in the
/// builders: a negative `rhai::INT` or a `u32::MAX` dim would otherwise
/// request an absurd allocation and abort the worker.
pub const MAX_GRID_TEXELS: u64 = 1 << 26; // ~67M neurons

/// `Ok(total)` when all dims are non-negative (callers pass `i64` so a
/// negative script arg is an error, not a giant `usize`) and their
/// product fits `MAX_GRID_TEXELS`.
pub fn check_grid_dims(dims: [i64; 3]) -> Result<u64, String> {
    for d in dims {
        if d <= 0 {
            return Err(format!("grid dims {dims:?}: each must be >= 1"));
        }
    }
    let total = dims[0] as u64 * dims[1] as u64 * dims[2] as u64;
    if total > MAX_GRID_TEXELS {
        return Err(format!(
            "grid dims {dims:?} = {total} texels (limit {MAX_GRID_TEXELS})"
        ));
    }
    Ok(total)
}

impl GridShape {
    /// Total texel count — must equal `members.len()` for a valid grid.
    pub fn texels(&self) -> usize {
        self.dims[0] as usize * self.dims[1] as usize * self.dims[2] as usize
    }

    /// Lattice index of texel (i0, i1, i2) — its position in the member list.
    pub fn member_index(&self, i: [u32; 3]) -> usize {
        (i[2] as usize * self.dims[0] as usize + i[0] as usize) * self.dims[1] as usize
            + i[1] as usize
    }

    /// Inverse of `member_index`.
    pub fn lattice_of(&self, k: usize) -> [u32; 3] {
        let (w, h) = (self.dims[0] as usize, self.dims[1] as usize);
        let i1 = k % h;
        let r = k / h;
        [(r % w) as u32, i1 as u32, (r / w) as u32]
    }

    /// World position of lattice texel `i`.
    pub fn world(&self, i: [u32; 3]) -> [f32; 3] {
        let mut p = self.origin;
        for k in 0..3 {
            p[self.axes[k] as usize] += self.spacing * i[k] as f32;
        }
        p
    }

    /// Lattice coords of a member whose world position is `pos`
    /// (snapped to the nearest texel, clamped to the lattice).
    pub fn lattice_at(&self, pos: [f32; 3]) -> [u32; 3] {
        let mut i = [0u32; 3];
        for k in 0..3 {
            let a = self.axes[k] as usize;
            let v = ((pos[a] - self.origin[a]) / self.spacing).round();
            i[k] = (v as i64).clamp(0, self.dims[k].saturating_sub(1) as i64) as u32;
        }
        i
    }

    /// Does this metadata still describe `members`? True when the member
    /// list has exactly one member per texel, in texel order, each at its
    /// lattice position (within a small tolerance for float rebuilds).
    pub fn matches(&self, neurons: &[Neuron], members: &[u32]) -> bool {
        if members.is_empty() || members.len() != self.texels() {
            return false;
        }
        let tol = (self.spacing.abs() * 1e-3).max(1e-4);
        for (k, &m) in members.iter().enumerate() {
            let Some(n) = neurons.get(m as usize) else {
                return false;
            };
            let w = self.world(self.lattice_of(k));
            for a in 0..3 {
                if (n.pos[a] - w[a]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}

/// A named scalar drive onto a set of neurons — the generalization of the
/// original's `Samples`+`CurrentInducer` pair. During a run the sim looks up
/// the current stimulus value of this input (from a stimulus matrix or a
/// script-set constant) and adds `weight * value` to each target.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Input {
    pub name: String,
    /// (neuron index, weight) pairs.
    pub targets: Vec<(u32, f32)>,
    /// Draw position for the editor.
    pub pos: [f32; 3],
}

/// A named time series — the original's `Samples`. Used both to record values
/// during a run and as scripted inputs (interpolated like the original).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Samples {
    pub name: String,
    pub times: Vec<f32>,
    pub values: Vec<f32>,
}

impl Samples {
    /// Linear interpolation, as the original's `Samples::update` did.
    /// Before the first sample returns the first value; past the end, the last.
    pub fn value_at(&self, t: f32) -> f32 {
        if self.times.is_empty() {
            return 0.0;
        }
        if t <= self.times[0] {
            return self.values[0];
        }
        if t >= *self.times.last().unwrap() {
            return *self.values.last().unwrap();
        }
        let i = self.times.partition_point(|&x| x <= t);
        let (t0, v0) = (self.times[i - 1], self.values[i - 1]);
        let (t1, v1) = (self.times[i], self.values[i]);
        if t1 <= t0 {
            return v1;
        }
        v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    }
}

/// LSM readout unit — the original's `LSMReadOutNeuron`. Its filtered-trace
/// semantics are reproduced in `sim::readout`; the weights are stored here so
/// a trained readout survives in the file, as it did in the original.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Readout {
    pub name: String,
    /// Liquid units: neuron indices the readout sees.
    pub units: Vec<u32>,
    /// One weight per unit. Empty = not yet initialized.
    pub weights: Vec<f32>,
    pub bias: f32,
    pub bias_weight: f32,
    /// Filtered-trace time constant (ms). Original default: tc=50 -> tau 2ms.
    pub tau_ms: f32,
    pub pos: [f32; 3],
}

/// File-level metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Meta {
    pub dt_ms: f32,
    pub seed: u64,
    pub conduction_speed: f32,
    pub synapse_tau_ms: f32,
    /// M3: these three came along later — serde defaults keep every
    /// .ne2 written before them loadable.
    #[serde(default = "default_readout_tau_ms")]
    pub readout_tau_ms: f32,
    #[serde(default)]
    pub noise_rate_hz: f32,
    #[serde(default = "default_noise_kick_mv")]
    pub noise_kick_mv: f32,
}

fn default_readout_tau_ms() -> f32 {
    DEFAULT_READOUT_TAU_MS
}
fn default_noise_kick_mv() -> f32 {
    20.0
}

impl Default for Meta {
    fn default() -> Self {
        Meta {
            dt_ms: DEFAULT_DT_MS,
            seed: 1,
            conduction_speed: DEFAULT_CONDUCTION_SPEED,
            synapse_tau_ms: DEFAULT_SYNAPSE_TAU_MS,
            readout_tau_ms: DEFAULT_READOUT_TAU_MS,
            noise_rate_hz: 0.0,
            noise_kick_mv: 20.0,
        }
    }
}

/// The whole editable network.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Network {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub groups: Vec<Group>,
    pub inputs: Vec<Input>,
    pub samples: Vec<Samples>,
    pub readouts: Vec<Readout>,
    /// Scripts stored inside the file, as in the original.
    pub scripts: Vec<(String, String)>,
    pub meta: Meta,
}

/// Source-sorted synapse layout for the simulators: CSR by source neuron.
#[derive(Clone, Debug, Default)]
pub struct SynCsr {
    /// Length n_neurons+1: outgoing synapses of neuron i are
    /// `src_ptr[i] .. src_ptr[i+1]`.
    pub src_ptr: Vec<u32>,
    pub tgt: Vec<u32>,
    pub w: Vec<f32>,
    /// Delay in steps of the dt this CSR was built for.
    pub delay_steps: Vec<u16>,
    /// Per-synapse tau in ms, CSR order (`0.0` = merged global path).
    pub tau_ms: Vec<f32>,
}

impl Network {
    pub fn new() -> Self {
        Network {
            meta: Meta::default(),
            ..Default::default()
        }
    }

    pub fn add_neuron(&mut self, pos: [f32; 3], kind: NeuronKind) -> u32 {
        let id = self.neurons.len() as u32;
        self.neurons.push(Neuron {
            name: format!("Neuron_{id}"),
            pos,
            model: NeuronModel::default(),
            kind,
        });
        self.apply_kind_default(id);
        id
    }

    /// Model defaults follow the kind while the parameters are untouched:
    /// setting `Inhibitory` on a neuron whose model is still the stock
    /// regular-spiking default swaps in fast-spiking defaults, and going
    /// back to `Excitatory` on a stock fast-spiking model restores
    /// regular-spiking. A model the user changed is never overwritten —
    /// "explicitly set" means "differs from either kind's stock params".
    fn apply_kind_default(&mut self, id: u32) {
        let n = &mut self.neurons[id as usize];
        n.model = match n.kind {
            NeuronKind::Inhibitory if n.model == NeuronModel::regular_spiking() => {
                NeuronModel::fast_spiking()
            }
            NeuronKind::Excitatory | NeuronKind::Regular
                if n.model == NeuronModel::fast_spiking() =>
            {
                NeuronModel::regular_spiking()
            }
            _ => n.model.clone(),
        };
    }

    /// Set a neuron's kind (Dale's-law label), applying kind-appropriate
    /// model defaults while the model is a stock default.
    pub fn set_kind(&mut self, id: u32, kind: NeuronKind) {
        self.neurons[id as usize].kind = kind;
        self.apply_kind_default(id);
    }

    pub fn neuron(&self, id: u32) -> &Neuron {
        &self.neurons[id as usize]
    }

    pub fn neuron_mut(&mut self, id: u32) -> &mut Neuron {
        &mut self.neurons[id as usize]
    }

    pub fn neuron_index(&self, name: &str) -> Option<u32> {
        self.neurons
            .iter()
            .position(|n| n.name == name)
            .map(|i| i as u32)
    }

    /// Distance-based delay for a src→tgt pair in ms (min dt*0.5 so that a
    /// same-site connection is still one step).
    pub fn conduction_delay_ms(&self, src: u32, tgt: u32) -> f32 {
        let a = self.neuron(src).pos;
        let b = self.neuron(tgt).pos;
        let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        (d / self.meta.conduction_speed).max(1e-3)
    }

    /// Create a synapse, computing the delay from current positions.
    pub fn connect(&mut self, src: u32, tgt: u32, w: f32) -> usize {
        let delay_ms = self.conduction_delay_ms(src, tgt);
        self.synapses.push(Synapse {
            src,
            tgt,
            w,
            delay_ms,
            tau_ms: 0.0,
        });
        self.synapses.len() - 1
    }

    /// Recompute all delays from positions (after moving neurons).
    pub fn refresh_delays(&mut self) {
        for i in 0..self.synapses.len() {
            let (src, tgt) = (self.synapses[i].src, self.synapses[i].tgt);
            self.synapses[i].delay_ms = self.conduction_delay_ms(src, tgt);
        }
    }

    /// Recompute the delay of a set of synapses touching `neuron`.
    pub fn refresh_delays_of(&mut self, neuron: u32) {
        for i in 0..self.synapses.len() {
            if self.synapses[i].src == neuron || self.synapses[i].tgt == neuron {
                let (src, tgt) = (self.synapses[i].src, self.synapses[i].tgt);
                self.synapses[i].delay_ms = self.conduction_delay_ms(src, tgt);
            }
        }
    }

    /// Source-sorted CSR for `dt` ms steps. Delay steps are rounded, min 1.
    pub fn synapse_csr(&self, dt_ms: f32) -> SynCsr {
        let n = self.neurons.len();
        let mut order: Vec<usize> = (0..self.synapses.len()).collect();
        order.sort_by_key(|&i| self.synapses[i].src);
        let mut csr = SynCsr {
            src_ptr: vec![0; n + 1],
            tgt: Vec::with_capacity(self.synapses.len()),
            w: Vec::with_capacity(self.synapses.len()),
            delay_steps: Vec::with_capacity(self.synapses.len()),
            tau_ms: Vec::with_capacity(self.synapses.len()),
        };
        for &i in &order {
            let s = &self.synapses[i];
            csr.tgt.push(s.tgt);
            // Dale's law: the source's kind sets the sign — the stored
            // weight is a magnitude. Both backends then inhibit without
            // any further code.
            let inh = self.neurons[s.src as usize].kind == NeuronKind::Inhibitory;
            csr.w.push(if inh { -s.w.abs() } else { s.w.abs() });
            csr.tau_ms.push(s.tau_ms);
            let steps = (s.delay_ms / dt_ms).round().max(1.0);
            csr.delay_steps.push(steps.min(u16::MAX as f32) as u16);
            csr.src_ptr[s.src as usize + 1] += 1;
        }
        for i in 0..n {
            csr.src_ptr[i + 1] += csr.src_ptr[i];
        }
        csr
    }

    pub fn group_index(&self, name: &str) -> Option<usize> {
        self.groups.iter().position(|g| g.name == name)
    }

    pub fn group(&self, name: &str) -> Option<&Group> {
        self.group_index(name).map(|i| &self.groups[i])
    }

    /// Add a named group; returns its index. Reuses an existing same-named group.
    pub fn add_group(&mut self, name: &str) -> usize {
        if let Some(i) = self.group_index(name) {
            return i;
        }
        self.groups.push(Group {
            name: name.to_string(),
            members: Vec::new(),
            grid: None,
        });
        self.groups.len() - 1
    }

    pub fn input_index(&self, name: &str) -> Option<usize> {
        self.inputs.iter().position(|i| i.name == name)
    }

    pub fn add_input(&mut self, name: &str, targets: Vec<(u32, f32)>, pos: [f32; 3]) -> usize {
        if let Some(i) = self.input_index(name) {
            self.inputs[i].targets = targets;
            self.inputs[i].pos = pos;
            return i;
        }
        self.inputs.push(Input {
            name: name.to_string(),
            targets,
            pos,
        });
        self.inputs.len() - 1
    }

    pub fn samples_index(&self, name: &str) -> Option<usize> {
        self.samples.iter().position(|s| s.name == name)
    }

    pub fn samples(&self, name: &str) -> Option<&Samples> {
        self.samples_index(name).map(|i| &self.samples[i])
    }

    pub fn samples_mut(&mut self, name: &str) -> &mut Samples {
        match self.samples_index(name) {
            Some(i) => &mut self.samples[i],
            None => {
                self.samples.push(Samples {
                    name: name.to_string(),
                    ..Default::default()
                });
                self.samples.last_mut().unwrap()
            }
        }
    }

    pub fn add_readout(&mut self, name: &str, units: Vec<u32>) -> usize {
        let i = self.readouts.len();
        self.readouts.push(Readout {
            name: name.to_string(),
            units,
            weights: Vec::new(),
            bias: 1.0,
            bias_weight: 1.0,
            tau_ms: crate::DEFAULT_READOUT_TAU_MS,
            pos: [0.0; 3],
        });
        i
    }

    /// Remove synapses that reference neurons >= n (used after truncating the
    /// neuron list) and drop synapses whose endpoints no longer exist.
    pub fn prune_synapses(&mut self) {
        let n = self.neurons.len() as u32;
        self.synapses.retain(|s| s.src < n && s.tgt < n);
    }

    pub fn delete_neuron(&mut self, id: u32) {
        self.neurons.remove(id as usize);
        // shift all indices > id down by one
        for s in &mut self.synapses {
            if s.src > id {
                s.src -= 1;
            }
            if s.tgt > id {
                s.tgt -= 1;
            }
        }
        self.synapses.retain(|s| s.src != id && s.tgt != id);
        for g in &mut self.groups {
            g.members.retain(|&m| m != id);
            for m in &mut g.members {
                if *m > id {
                    *m -= 1;
                }
            }
        }
        for inp in &mut self.inputs {
            inp.targets.retain(|&(t, _)| t != id);
            for t in &mut inp.targets {
                if t.0 > id {
                    t.0 -= 1;
                }
            }
        }
        for r in &mut self.readouts {
            r.units.retain(|&u| u != id);
            for u in &mut r.units {
                if *u > id {
                    *u -= 1;
                }
            }
        }
        self.invalidate_grids();
    }

    /// Drop grid metadata that no longer describes its group's members —
    /// e.g. after a member was deleted, moved, added, or reordered. Called
    /// by the mutating methods here; editor/script code that moves neurons
    /// through `neuron_mut` must call it too.
    pub fn invalidate_grids(&mut self) {
        let neurons = &self.neurons;
        for g in &mut self.groups {
            let ok = g.grid.is_some_and(|gs| gs.matches(neurons, &g.members));
            if g.grid.is_some() && !ok {
                g.grid = None;
            }
        }
    }

    /// Delete a set of neurons at once.
    pub fn delete_neurons(&mut self, mut ids: Vec<u32>) {
        ids.sort_unstable_by(|a, b| b.cmp(a));
        for id in ids {
            self.delete_neuron(id);
        }
    }

    /// Extract the given neurons (with synapses and group membership inside
    /// the set) as a standalone sub-network — the clipboard format for
    /// copy/paste.
    pub fn subnet(&self, ids: &std::collections::HashSet<u32>) -> Network {
        let mut sub = Network::new();
        let mut map = std::collections::HashMap::new();
        let mut ids: Vec<u32> = ids.iter().copied().collect();
        ids.sort_unstable();
        for id in ids {
            if (id as usize) >= self.neurons.len() {
                continue;
            }
            map.insert(id, sub.neurons.len() as u32);
            sub.neurons.push(self.neurons[id as usize].clone());
        }
        for s in &self.synapses {
            if let (Some(&a), Some(&b)) = (map.get(&s.src), map.get(&s.tgt)) {
                sub.synapses.push(Synapse {
                    src: a,
                    tgt: b,
                    ..*s
                });
            }
        }
        for g in &self.groups {
            let members: Vec<u32> = g
                .members
                .iter()
                .filter_map(|m| map.get(m).copied())
                .collect();
            if !members.is_empty() {
                // A grid survives a copy only when every member came along —
                // the subnet then has the same lattice at the same coords.
                let grid = g
                    .grid
                    .filter(|gs| members.len() == gs.texels())
                    .filter(|gs| gs.matches(&sub.neurons, &members));
                sub.groups.push(Group {
                    name: g.name.clone(),
                    members,
                    grid,
                });
            }
        }
        sub
    }

    /// Merge a sub-network (from `subnet`) into this one at `offset`; returns
    /// the new ids in `src` order. Delays of pasted synapses are kept as-is.
    pub fn insert_subnet(&mut self, src: &Network, offset: [f32; 3]) -> Vec<u32> {
        let map: Vec<u32> = src
            .neurons
            .iter()
            .map(|neu| {
                let mut neu = neu.clone();
                neu.pos[0] += offset[0];
                neu.pos[1] += offset[1];
                neu.pos[2] += offset[2];
                self.neurons.push(neu);
                self.neurons.len() as u32 - 1
            })
            .collect();
        for s in &src.synapses {
            let (Some(a), Some(b)) = (
                map.get(s.src as usize).copied(),
                map.get(s.tgt as usize).copied(),
            ) else {
                continue; // malformed subnet: skip the dangling synapse
            };
            self.synapses.push(Synapse {
                src: a,
                tgt: b,
                ..*s
            });
        }
        for g in &src.groups {
            let i = self.add_group(&g.name);
            let members: Vec<u32> = g
                .members
                .iter()
                .filter_map(|&m| map.get(m as usize).copied())
                .collect();
            self.groups[i].members.extend(members);
        }
        self.invalidate_grids();
        map
    }

    /// Structural integrity: every stored index must point at a real neuron.
    /// Called after file deserialization — a corrupt or hand-edited file must
    /// fail here rather than panic inside a sim build or an editor redraw.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.neurons.len() as u32;
        for (i, s) in self.synapses.iter().enumerate() {
            if s.src >= n || s.tgt >= n {
                return Err(format!(
                    "synapse {i} references {} -> {} but only {n} neurons exist",
                    s.src, s.tgt
                ));
            }
            if !(s.tau_ms >= 0.0 && s.tau_ms.is_finite()) {
                return Err(format!(
                    "synapse {i} has bad tau_ms {} (want >= 0, finite)",
                    s.tau_ms
                ));
            }
        }
        for g in &self.groups {
            for &m in &g.members {
                if m >= n {
                    return Err(format!(
                        "group '{}' references neuron {m} but only {n} exist",
                        g.name
                    ));
                }
            }
            if let Some(grid) = &g.grid {
                // axes must be a permutation of {0,1,2}: an index >= 3
                // panics in world()/lattice_at() (out-of-bounds on pos),
                // and duplicate axes fold two lattice dims onto one world
                // axis.
                let mut seen = [false; 3];
                for (k, &a) in grid.axes.iter().enumerate() {
                    if a as usize >= 3 || seen[a as usize] {
                        return Err(format!(
                            "group '{}' grid axes {a} at dim {k} is invalid \
                             (want each of 0,1,2 exactly once)",
                            g.name
                        ));
                    }
                    seen[a as usize] = true;
                }
                if !(grid.spacing.is_finite() && grid.spacing != 0.0) {
                    return Err(format!(
                        "group '{}' grid spacing {} must be finite and non-zero",
                        g.name, grid.spacing
                    ));
                }
                for &d in &grid.dims {
                    if d == 0 {
                        return Err(format!(
                            "group '{}' grid dims {:?} contains a zero",
                            g.name, grid.dims
                        ));
                    }
                }
            }
        }
        for inp in &self.inputs {
            for &(t, _) in &inp.targets {
                if t >= n {
                    return Err(format!(
                        "input '{}' targets neuron {t} but only {n} exist",
                        inp.name
                    ));
                }
            }
        }
        for r in &self.readouts {
            for &u in &r.units {
                if u >= n {
                    return Err(format!(
                        "readout '{}' references unit {u} but only {n} exist",
                        r.name
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Texel index <-> lattice coords <-> world position round-trip, for
    /// both builder layouts (2D in the y–z plane, 3D in xyz).
    #[test]
    fn grid_shape_index_math() {
        let g2 = GridShape {
            dims: [5, 3, 1],
            origin: [100.0, -15.0, -25.0],
            spacing: 10.0,
            axes: [2, 1, 0],
        };
        for i0 in 0..5 {
            for i1 in 0..3 {
                let k = g2.member_index([i0, i1, 0]);
                assert_eq!(g2.lattice_of(k), [i0, i1, 0]);
                let w = g2.world([i0, i1, 0]);
                // 2D lattice lives in the y–z plane: i0 along +z, i1 +y.
                assert_eq!(
                    w,
                    [100.0, -15.0 + i1 as f32 * 10.0, -25.0 + i0 as f32 * 10.0]
                );
                assert_eq!(g2.lattice_at(w), [i0, i1, 0]);
            }
        }
        let g3 = GridShape {
            dims: [4, 3, 2],
            origin: [0.0; 3],
            spacing: 5.0,
            axes: [0, 1, 2],
        };
        assert_eq!(g3.texels(), 24);
        assert_eq!(g3.member_index([2, 1, 1]), (1 * 4 + 2) * 3 + 1);
        assert_eq!(g3.world([2, 1, 1]), [10.0, 5.0, 5.0]);
        for k in 0..g3.texels() {
            assert_eq!(g3.member_index(g3.lattice_of(k)), k);
        }
    }

    #[test]
    fn builders_record_grid_metadata() {
        use crate::builders;
        let mut net = Network::new();
        builders::create_2d_grid(&mut net, "p", 5, 3, 10.0, [100.0, 0.0, 0.0]);
        let g = net.group("p").unwrap().grid.unwrap();
        assert_eq!(g.dims, [5, 3, 1]);
        assert_eq!(g.axes, [2, 1, 0]);
        assert_eq!(net.group("p").unwrap().members.len(), 15);

        builders::create_3d_grid(&mut net, "v", 4, 3, 2, 5.0, [0.0; 3]);
        let g = net.group("v").unwrap().grid.unwrap();
        assert_eq!(g.dims, [4, 3, 2]);
        assert_eq!(g.axes, [0, 1, 2]);
        // member k's world position == world(lattice_of(k)) — checked by
        // matches(), which the builder itself ran before storing.
        let members = &net.group("v").unwrap().members;
        assert!(g.matches(&net.neurons, members));
    }

    /// Edits that break the lattice drop the metadata; the plain member
    /// list and group survive.
    #[test]
    fn grid_drops_on_member_delete_and_move() {
        use crate::builders;
        let mut net = Network::new();
        builders::create_2d_grid(&mut net, "p", 4, 4, 10.0, [0.0; 3]);
        assert!(net.group("p").unwrap().grid.is_some());

        // delete a member -> no longer one member per texel
        let m = net.group("p").unwrap().members[5];
        net.delete_neuron(m);
        assert!(net.group("p").unwrap().grid.is_none());
        assert_eq!(net.group("p").unwrap().members.len(), 15);

        // move a member -> position no longer matches the lattice
        builders::create_3d_grid(&mut net, "v", 3, 3, 3, 10.0, [500.0; 3]);
        let m = net.group("v").unwrap().members[7];
        net.neuron_mut(m).pos[0] += 3.0;
        net.invalidate_grids();
        assert!(net.group("v").unwrap().grid.is_none());
        assert_eq!(net.group("v").unwrap().members.len(), 27);
    }

    #[test]
    fn grid_survives_full_copy_but_not_partial_or_offset_paste() {
        use crate::builders;
        let mut net = Network::new();
        builders::create_2d_grid(&mut net, "p", 3, 3, 10.0, [0.0; 3]);
        let all: std::collections::HashSet<u32> =
            net.group("p").unwrap().members.iter().copied().collect();
        let sub = net.subnet(&all);
        assert_eq!(sub.groups[0].grid, net.group("p").unwrap().grid);

        // partial copy keeps the group but drops the lattice metadata
        let part: std::collections::HashSet<u32> = all.iter().take(4).copied().collect();
        let sub = net.subnet(&part);
        assert!(sub.groups[0].grid.is_none());

        // pasting the full grid at an offset breaks positions -> dropped
        let sub = net.subnet(&all);
        net.insert_subnet(&sub, [50.0, 0.0, 0.0]);
        assert_eq!(net.groups.len(), 1); // same name merges into "p"
        assert!(net.group("p").unwrap().grid.is_none());
    }

    #[test]
    fn subnet_insert_roundtrip() {
        let mut net = Network::new();
        for i in 0..5 {
            net.add_neuron([i as f32 * 10.0, 0.0, 0.0], NeuronKind::Excitatory);
        }
        net.connect(0, 1, 2.0);
        net.connect(1, 2, 3.0);
        net.connect(3, 4, 4.0);
        let g = net.add_group("g");
        net.groups[g].members = vec![0, 1, 2];

        // copy {1,2} — internal synapse (1→2) kept, group clipped to members
        let ids = [1u32, 2].into_iter().collect();
        let sub = net.subnet(&ids);
        assert_eq!(sub.neurons.len(), 2);
        assert_eq!(sub.synapses.len(), 1);
        assert_eq!((sub.synapses[0].src, sub.synapses[0].tgt), (0, 1));
        assert_eq!(sub.groups[0].members, vec![0, 1]);

        // paste at +100 x-offset: new ids appended, remapped synapse + group
        let map = net.insert_subnet(&sub, [100.0, 0.0, 0.0]);
        assert_eq!(net.neurons.len(), 7);
        assert_eq!(net.synapses.len(), 4);
        let pasted = net.synapses[3];
        assert_eq!((pasted.src, pasted.tgt), (map[0], map[1]));
        assert_eq!(net.neurons[map[0] as usize].pos[0], 110.0);
        let g2 = net.group("g").unwrap();
        assert_eq!(g2.members, vec![0, 1, 2, map[0], map[1]]);
    }

    #[test]
    fn validate_rejects_each_dangling_kind() {
        let mut base = Network::new();
        base.add_neuron([0.0; 3], NeuronKind::Excitatory);
        assert!(base.validate().is_ok());
        base.connect(0, 0, 1.0);
        assert!(base.validate().is_ok());

        let mut n = base.clone();
        n.synapses.push(Synapse {
            src: 0,
            tgt: 5,
            w: 1.0,
            delay_ms: 1.0,
            tau_ms: 0.0,
        });
        assert!(n.validate().unwrap_err().contains("synapse"));

        let mut n = base.clone();
        let g = n.add_group("g");
        n.groups[g].members.push(5);
        assert!(n.validate().unwrap_err().contains("group"));

        let mut n = base.clone();
        n.add_input("i", vec![(5, 1.0)], [0.0; 3]);
        assert!(n.validate().unwrap_err().contains("input"));

        let mut n = base;
        n.add_readout("r", vec![5]);
        assert!(n.validate().unwrap_err().contains("readout"));
    }

    #[test]
    fn insert_subnet_skips_dangling_refs() {
        let mut net = Network::new();
        net.add_neuron([0.0; 3], NeuronKind::Excitatory);
        // subnet with a synapse + group member past its own neuron count
        let mut sub = Network::new();
        sub.add_neuron([0.0; 3], NeuronKind::Excitatory);
        sub.synapses.push(Synapse {
            src: 0,
            tgt: 7,
            w: 1.0,
            delay_ms: 1.0,
            tau_ms: 0.0,
        });
        let g = sub.add_group("g");
        sub.groups[g].members = vec![0, 7];
        net.insert_subnet(&sub, [0.0; 3]);
        // dangling synapse dropped, valid group member kept
        assert!(net.synapses.is_empty());
        assert_eq!(net.group("g").unwrap().members, vec![1]);
    }
}
