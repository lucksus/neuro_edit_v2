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
