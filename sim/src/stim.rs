//! Stimulus delivery: named scalar drives (the `Input` list of a `Network`)
//! turned into per-neuron current during a run. One stimulus "column" per
//! input; scripts either push a whole time series (played back step-locked) or
//! set a constant value.

/// A stimulus program: `columns[input]` holds `values[step]` applied to that
/// input's targets with their weights.
#[derive(Clone, Debug, Default)]
pub struct Stimulus {
    /// input index -> constant value used when no series is playing
    pub constants: Vec<f32>,
    /// input index -> time series (one value per step)
    pub series: Vec<Vec<f32>>,
    /// input index -> step at which that input's series playback starts.
    /// Per-input: playing input B must not move input A's playhead.
    pub play_start: Vec<u64>,
}

/// Per-neuron stimulus routing, flattened for the simulators:
/// neuron i receives `sum_k value_k * w` over its (input, w) entries.
#[derive(Clone, Debug, Default)]
pub struct StimMap {
    /// neuron -> range into `input_idx`/`weight`
    pub neuron_ptr: Vec<u32>,
    pub input_idx: Vec<u32>,
    pub weight: Vec<f32>,
}

impl StimMap {
    pub fn from_network(net: &net_format::Network) -> Self {
        let n = net.neurons.len();
        let mut map = StimMap {
            neuron_ptr: vec![0; n + 1],
            input_idx: Vec::new(),
            weight: Vec::new(),
        };
        let mut per_neuron: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
        for (k, input) in net.inputs.iter().enumerate() {
            for &(t, w) in &input.targets {
                if (t as usize) < n {
                    per_neuron[t as usize].push((k as u32, w));
                }
            }
        }
        for (i, list) in per_neuron.iter().enumerate() {
            map.neuron_ptr[i + 1] = map.neuron_ptr[i] + list.len() as u32;
            for &(k, w) in list {
                map.input_idx.push(k);
                map.weight.push(w);
            }
        }
        map
    }
}

impl Stimulus {
    pub fn for_network(net: &net_format::Network) -> Self {
        Stimulus {
            constants: vec![0.0; net.inputs.len()],
            series: vec![Vec::new(); net.inputs.len()],
            play_start: vec![0; net.inputs.len()],
        }
    }

    /// Start playing `series` on input `k` at step `t` — other inputs'
    /// playheads are unaffected.
    pub fn play(&mut self, k: usize, t: u64, series: Vec<f32>) {
        if k >= self.series.len() {
            self.series.resize(k + 1, Vec::new());
        }
        if k >= self.play_start.len() {
            self.play_start.resize(k + 1, 0);
        }
        self.series[k] = series;
        self.play_start[k] = t;
    }

    /// Value of input `k` at step `t` (0 outside a played series' range).
    pub fn value(&self, k: usize, t: u64) -> f32 {
        if let Some(series) = self.series.get(k) {
            if !series.is_empty() {
                let start = self.play_start.get(k).copied().unwrap_or(0);
                let idx = t.wrapping_sub(start) as usize;
                return if idx < series.len() { series[idx] } else { 0.0 };
            }
        }
        self.constants.get(k).copied().unwrap_or(0.0)
    }

    /// Packed per-step values [step][input] for the next `n_steps` steps —
    /// what the CUDA backend uploads before `run_steps`.
    pub fn pack(&self, start_step: u64, n_steps: usize, n_inputs: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n_steps * n_inputs];
        for s in 0..n_steps {
            for k in 0..n_inputs {
                out[s * n_inputs + k] = self.value(k, start_step + s as u64);
            }
        }
        out
    }
}

/// The drive programmed into a sim (`set_input` constants, `play_input`
/// series, `set_external` currents), detached so a sim rebuild — backend
/// switch, dt change, net edit — can re-apply it. Constants and series
/// are keyed by input NAME so they survive input-list edits; `ext` stays
/// neuron-indexed.
#[derive(Clone, Debug, Default)]
pub struct DriveState {
    /// input name -> constant value. Only entries with no queued series —
    /// a series shadows the constant the way `Stimulus::value` does.
    pub consts: Vec<(String, f32)>,
    /// input name -> series tail still to play, rebased to the new sim's
    /// step 0. A fully-consumed series is kept as `[0.0]` so the shadowed
    /// constant does not resurrect after the rebuild.
    pub series: Vec<(String, Vec<f32>)>,
    /// neuron index -> external current (nonzero entries only).
    pub ext: Vec<(u32, f32)>,
}

impl DriveState {
    /// Detach `sim`'s drive, keying inputs by `net`'s input names. A
    /// half-played series is truncated to its remaining tail so the
    /// rebuilt sim continues playback where the old one stopped.
    pub fn capture(sim: &crate::Sim, net: &net_format::Network) -> DriveState {
        let stim = sim.stimulus();
        let t = sim.step_count();
        let mut d = DriveState::default();
        for (i, inp) in net.inputs.iter().enumerate() {
            match stim.series.get(i) {
                Some(ser) if !ser.is_empty() => {
                    let start = stim.play_start.get(i).copied().unwrap_or(0);
                    let tail = if t > start {
                        let done = (t - start) as usize;
                        if done < ser.len() {
                            ser[done..].to_vec()
                        } else {
                            vec![0.0]
                        }
                    } else {
                        ser.clone()
                    };
                    d.series.push((inp.name.clone(), tail));
                }
                _ => {
                    let c = stim.constants.get(i).copied().unwrap_or(0.0);
                    if c != 0.0 {
                        d.consts.push((inp.name.clone(), c));
                    }
                }
            }
        }
        for (i, &a) in sim.external().iter().enumerate() {
            if a != 0.0 {
                d.ext.push((i as u32, a));
            }
        }
        d
    }

    /// Re-apply onto a freshly built sim (its step count is 0, so each
    /// saved tail starts playing immediately).
    pub fn apply(&self, sim: &mut crate::Sim, net: &net_format::Network) {
        for (name, v) in &self.consts {
            if let Some(i) = net.input_index(name) {
                sim.set_input_constant(i, *v);
            }
        }
        for (name, tail) in &self.series {
            if let Some(i) = net.input_index(name) {
                sim.play_input(i, tail.clone());
            }
        }
        if !self.ext.is_empty() {
            let mut ext = vec![0.0f32; net.neurons.len()];
            for &(nid, a) in &self.ext {
                if let Some(e) = ext.get_mut(nid as usize) {
                    *e = a;
                }
            }
            sim.set_external_all(&ext);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use net_format::Network;

    fn driven_sim() -> (Network, crate::Sim) {
        let mut net = Network::new();
        net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
        net.add_neuron([10.0; 3], net_format::NeuronKind::Excitatory);
        net.add_input("a", vec![(0, 10.0)], [0.0; 3]);
        net.add_input("b", vec![(1, 10.0)], [0.0; 3]);
        let mut s = crate::Sim::new(&net, crate::SimConfig::default(), crate::BackendKind::Cpu)
            .unwrap();
        s.set_input_constant(0, 1.0);
        s.play_input(1, vec![5.0; 10]); // plays from t=0
        s.set_external(1, 3.0);
        (net, s)
    }

    #[test]
    fn capture_keys_consts_by_name_and_trims_series_tail() {
        let (net, mut s) = driven_sim();
        s.run_steps(4).unwrap(); // "b"'s 10-step series is 6 from done
        let d = DriveState::capture(&s, &net);
        assert_eq!(d.consts, vec![("a".to_string(), 1.0)]);
        assert_eq!(d.series, vec![("b".to_string(), vec![5.0; 6])]);
        assert_eq!(d.ext, vec![(1, 3.0)]);
    }

    #[test]
    fn apply_restores_drive_onto_fresh_sim() {
        let (net, mut s) = driven_sim();
        s.run_steps(4).unwrap();
        let d = DriveState::capture(&s, &net);
        let mut s2 =
            crate::Sim::new(&net, crate::SimConfig::default(), crate::BackendKind::Cpu).unwrap();
        d.apply(&mut s2, &net);
        let st = s2.stimulus();
        assert_eq!(st.constants[0], 1.0);
        assert_eq!(st.series[1], vec![5.0; 6]);
        assert_eq!(s2.external()[1], 3.0);
    }

    #[test]
    fn finished_series_stays_zero_and_does_not_resurrect_constant() {
        let (net, mut s) = driven_sim();
        s.set_input_constant(1, 9.0); // shadowed by the queued series
        s.run_steps(12).unwrap(); // "b"'s series ran out
        let d = DriveState::capture(&s, &net);
        // a consumed series must capture as non-empty so `value` keeps
        // returning 0 instead of falling back to the constant
        assert_eq!(d.series, vec![("b".to_string(), vec![0.0])]);
        let mut s2 =
            crate::Sim::new(&net, crate::SimConfig::default(), crate::BackendKind::Cpu).unwrap();
        d.apply(&mut s2, &net);
        s2.run_steps(5).unwrap();
        let st = s2.stimulus();
        assert_eq!(st.value(1, s2.step_count()), 0.0);
    }

    #[test]
    fn apply_drops_inputs_the_new_net_lacks() {
        let (net, s) = driven_sim();
        let d = DriveState::capture(&s, &net);
        let mut net2 = Network::new();
        net2.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
        net2.add_input("a", vec![(0, 10.0)], [0.0; 3]); // no "b", one neuron
        let mut s2 =
            crate::Sim::new(&net2, crate::SimConfig::default(), crate::BackendKind::Cpu).unwrap();
        d.apply(&mut s2, &net2);
        let st = s2.stimulus();
        assert_eq!(st.constants[0], 1.0);
        assert_eq!(st.series.len(), 1); // "b" series silently dropped
        assert!(s2.external().iter().all(|&v| v == 0.0)); // neuron 1 gone, no panic
    }
}
