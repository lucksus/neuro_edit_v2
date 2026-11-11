use serde::{Deserialize, Serialize};

use crate::{DEFAULT_CONDUCTION_SPEED, DEFAULT_DT_MS, DEFAULT_SYNAPSE_TAU_MS};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum NeuronModel {
    /// Izhikevich 2003 model. v' = 0.04v^2+5v+140-u+I, u' = a(bv-u),
    /// spike at v>=30 then v=c, u+=d on the next step.
    Izhikevich { a: f32, b: f32, c: f32, d: f32 },
}

impl Default for NeuronModel {
    /// Regular-spiking defaults, as in the original Neuron constructor.
    fn default() -> Self {
        NeuronModel::Izhikevich {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
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
}

/// A named set of neuron indices — the original's `Group`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Group {
    pub name: String,
    pub members: Vec<u32>,
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
}

impl Default for Meta {
    fn default() -> Self {
        Meta {
            dt_ms: DEFAULT_DT_MS,
            seed: 1,
            conduction_speed: DEFAULT_CONDUCTION_SPEED,
            synapse_tau_ms: DEFAULT_SYNAPSE_TAU_MS,
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
        id
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
        };
        for &i in &order {
            let s = &self.synapses[i];
            csr.tgt.push(s.tgt);
            csr.w.push(s.w);
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
                sub.groups.push(Group {
                    name: g.name.clone(),
                    members,
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
            self.synapses.push(Synapse {
                src: map[s.src as usize],
                tgt: map[s.tgt as usize],
                ..*s
            });
        }
        for g in &src.groups {
            let i = self.add_group(&g.name);
            let members: Vec<u32> = g.members.iter().map(|&m| map[m as usize]).collect();
            self.groups[i].members.extend(members);
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
