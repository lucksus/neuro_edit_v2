//! CPU reference simulator. Faithful to the 2011 model, flattened:
//!
//! Per step (dt ms), for every neuron i:
//!   arrivals = ring[t mod D][i]           (spikes whose delay elapsed)
//!   s[i] += arrivals                      (each arrived spike adds its weight)
//!   I = s[i] + ext[i] + stimulus + noise
//!   if v >= 30 { v = c; u += d }          (reset for last step's crossing)
//!   v += dt * (0.04 v^2 + 5v + 140 - u + I)
//!   u += dt * a * (b v - u)
//!   if v >= 30 -> spike                   (propagates to ring at t + delay)
//!   s[i] *= (1 - dt/tau_syn)              (original's Euler decay, tc/100 law)
//!   trace[i] *= (1 - dt/tau_r); on spike trace[i] = 1
//!
//! The original applied the decay after using the current in the same update;
//! a freshly arrived spike therefore contributes its full weight immediately.

use crate::state::{noise_fires, ring_depth, NeuronParams, SimConfig};
use net_format::{Network, SynCsr};

pub struct CpuSim {
    pub n: usize,
    pub cfg: SimConfig,
    pub params: NeuronParams,
    pub csr: SynCsr,
    pub stim_map: crate::stim::StimMap,
    pub stimulus: crate::stim::Stimulus,

    // state
    pub v: Vec<f32>,
    pub u: Vec<f32>,
    /// decaying synaptic current
    pub s: Vec<f32>,
    /// filtered spike trace for readouts (1 on spike, exponential decay)
    pub trace: Vec<f32>,
    /// delay ring: ring[slot * n + i], D = ring_depth
    pub ring: Vec<f32>,
    pub d: usize,
    /// constant external current per neuron (clamped drive)
    pub ext: Vec<f32>,
    /// spike recording: bits per step in a rolling window
    pub spike_bits: Vec<u64>,
    pub spike_counts: Vec<u32>,
    /// per-neuron lifetime spike totals (answers windows >= t exactly)
    pub pn_total: Vec<u64>,
    pub words: usize,

    /// watched neuron ids whose v is recorded every step (editor v(t) plot)
    pub v_watch: Vec<u32>,
    /// per-watched-neuron per-step v history, capped at `V_HIST_CAP`
    pub v_hist: Vec<std::collections::VecDeque<f32>>,

    /// per-grid-group spike-trace bookkeeping (editor plane view)
    pub grid_watches: Vec<GridWatch>,

    pub t: u64,
    pub spikes_total: u64,
}

/// Max per-step v samples kept per watched neuron between drains.
pub const V_HIST_CAP: usize = 4096;

/// Per-grid-group spike-trace bookkeeping. `map[neuron]` is that neuron's
/// texel index (-1 when not a member); `ls[texel]` is the absolute step of
/// the texel's most recent spike (u32::MAX = never). The materialized
/// trace `exp(-(t - ls) * dt / tau)` is computed on read so the view's tau
/// stays a display choice rather than baked into the sim state.
pub struct GridWatch {
    /// member list in texel order (kept for the no-op check)
    pub members: Vec<u32>,
    /// neuron id -> texel index, -1 when not a member
    pub map: Vec<i32>,
    /// texel -> absolute step of last spike (u32::MAX = never spiked)
    pub ls: Vec<u32>,
}

impl CpuSim {
    pub fn new(net: &Network, cfg: SimConfig) -> Self {
        let n = net.neurons.len();
        let csr = net.synapse_csr(cfg.dt_ms);
        let d = ring_depth(&csr, cfg.dt_ms);
        let params = NeuronParams::from_network(net);
        let words = n.div_ceil(64);
        // >= 2 keeps the wrap-clear semantics uniform with the CUDA backend
        let spike_record_steps = if cfg.record_spikes {
            cfg.spike_record_steps.max(2)
        } else {
            0
        };
        CpuSim {
            n,
            csr,
            params,
            stim_map: crate::stim::StimMap::from_network(net),
            stimulus: crate::stim::Stimulus::for_network(net),
            v: vec![0.0; n],
            u: vec![0.0; n],
            s: vec![0.0; n],
            trace: vec![0.0; n],
            ring: vec![0.0; d * n],
            d,
            ext: vec![0.0; n],
            spike_bits: vec![0; spike_record_steps * words],
            spike_counts: vec![0; spike_record_steps],
            pn_total: vec![0; n],
            words,
            v_watch: Vec::new(),
            v_hist: Vec::new(),
            grid_watches: Vec::new(),
            t: 0,
            spikes_total: 0,
            cfg,
        }
        .init()
    }

    fn init(mut self) -> Self {
        self.reset_state();
        self
    }

    /// Reset dynamical state (v, u, currents, ring, time). Keeps synapses,
    /// stimulus and ext — the original's `sim.reset()`.
    pub fn reset_state(&mut self) {
        self.v.fill(self.params.init_v);
        self.u.copy_from_slice(&self.params.init_u);
        self.s.fill(0.0);
        self.trace.fill(0.0);
        self.ring.fill(0.0);
        self.spike_bits.fill(0);
        self.spike_counts.fill(0);
        self.pn_total.fill(0);
        self.t = 0;
        self.spikes_total = 0;
        for d in &mut self.v_hist {
            d.clear();
        }
        for w in &mut self.grid_watches {
            w.ls.fill(u32::MAX);
        }
    }

    #[inline]
    fn dt(&self) -> f32 {
        self.cfg.dt_ms
    }

    /// One dt step. Mirrors the CUDA kernel order exactly.
    pub fn step(&mut self) {
        let n = self.n;
        let dt = self.dt();
        let t = self.t;
        let slot = (t as usize) % self.d;
        let p_kick = self.cfg.noise_rate_hz * dt / 1000.0;
        let kick = if self.cfg.noise_kick_mv != 0.0 {
            self.cfg.noise_kick_mv / dt
        } else {
            0.0
        };
        let decay_s = 1.0 - dt / self.cfg.synapse_tau_ms.max(1e-6);
        let decay_t = 1.0 - dt / self.cfg.readout_tau_ms.max(1e-6);
        let record = self.cfg.record_spikes;

        let slot_base = slot * n;
        let mut spiked: Vec<u32> = Vec::new();
        for i in 0..n {
            let arr = self.ring[slot_base + i];
            self.ring[slot_base + i] = 0.0;
            self.s[i] += arr;
            let mut input = self.s[i] + self.ext[i];
            // stimulus: sum over this neuron's (input, weight) entries
            let (lo, hi) = (
                self.stim_map.neuron_ptr[i] as usize,
                self.stim_map.neuron_ptr[i + 1] as usize,
            );
            for j in lo..hi {
                let k = self.stim_map.input_idx[j] as usize;
                input += self.stim_map.weight[j] * self.stimulus.value(k, t);
            }
            if p_kick > 0.0 && noise_fires(i as u32, t, self.cfg.seed, p_kick) {
                input += kick;
            }
            let mut v = self.v[i];
            let mut u = self.u[i];
            if v >= 30.0 {
                v = self.params.c[i];
                u += self.params.d[i];
            }
            v += dt * (0.04 * v * v + 5.0 * v + 140.0 - u + input);
            u += dt * self.params.a[i] * (self.params.b[i] * v - u);
            self.v[i] = v;
            self.u[i] = u;
            if v >= 30.0 {
                spiked.push(i as u32);
                // original sets the readout trace to full on spike, no decay
                self.trace[i] = 1.0;
            } else {
                self.trace[i] *= decay_t;
            }
            // decays (original's subtractive rate law)
            self.s[i] *= decay_s;
        }

        // propagate spikes through source-sorted CSR into the delay ring
        for &i in &spiked {
            let (lo, hi) = (
                self.csr.src_ptr[i as usize] as usize,
                self.csr.src_ptr[i as usize + 1] as usize,
            );
            for j in lo..hi {
                let dst = (t + self.csr.delay_steps[j] as u64) as usize % self.d;
                self.ring[dst * n + self.csr.tgt[j] as usize] += self.csr.w[j];
            }
        }

        if record {
            let row = (t as usize) % self.spike_counts.len().max(1);
            self.spike_counts[row] = spiked.len() as u32;
            // the row is reused every `record_steps` steps — clear stale
            // bits from the previous wrap before OR-ing this step's spikes
            self.spike_bits[row * self.words..(row + 1) * self.words].fill(0);
            for &i in &spiked {
                self.spike_bits[row * self.words + (i as usize) / 64] |= 1 << (i % 64);
            }
        }
        for &i in &spiked {
            self.pn_total[i as usize] += 1;
        }
        if !self.grid_watches.is_empty() {
            let t32 = (t.min(u32::MAX as u64 - 1)) as u32;
            for &i in &spiked {
                for w in &mut self.grid_watches {
                    let k = w.map[i as usize];
                    if k >= 0 {
                        w.ls[k as usize] = t32;
                    }
                }
            }
        }
        self.spikes_total += spiked.len() as u64;
        self.t += 1;
    }

    pub fn run_steps(&mut self, steps: usize) -> anyhow::Result<()> {
        for _ in 0..steps {
            self.step();
            if !self.v_watch.is_empty() {
                for (k, &id) in self.v_watch.iter().enumerate() {
                    let val = self.v.get(id as usize).copied().unwrap_or(0.0);
                    let r = &mut self.v_hist[k];
                    r.push_back(val);
                    if r.len() > V_HIST_CAP {
                        r.pop_front();
                    }
                }
            }
        }
        Ok(())
    }

    /// Set the watched-neuron list; no-op when unchanged (safe to call
    /// every publish — history survives unchanged watch lists).
    pub fn set_v_watch(&mut self, ids: &[u32]) {
        if self.v_watch == ids {
            return;
        }
        self.v_watch = ids.to_vec();
        self.v_hist = vec![std::collections::VecDeque::with_capacity(256); ids.len()];
    }

    /// Drain per-step v samples recorded since the last drain, one Vec per
    /// watched neuron (oldest first).
    pub fn drain_v(&mut self) -> Vec<Vec<f32>> {
        self.v_hist
            .iter_mut()
            .map(|d| d.drain(..).collect())
            .collect()
    }

    /// Set the grid-group member lists (texel order) whose spike traces the
    /// plane view wants. No-op when unchanged; safe to call every publish.
    pub fn set_grid_watches(&mut self, groups: &[Vec<u32>]) {
        if self.grid_watches.len() == groups.len()
            && self
                .grid_watches
                .iter()
                .zip(groups)
                .all(|(w, g)| w.members == *g)
        {
            return;
        }
        self.grid_watches = groups
            .iter()
            .map(|members| {
                let mut map = vec![-1i32; self.n];
                for (k, &m) in members.iter().enumerate() {
                    if let Some(slot) = map.get_mut(m as usize) {
                        *slot = k as i32;
                    }
                }
                GridWatch {
                    ls: vec![u32::MAX; members.len()],
                    members: members.clone(),
                    map,
                }
            })
            .collect();
    }

    /// Materialized spike traces for each watched grid group: one f32 per
    /// texel (member order), `exp(-(t - last_spike) * dt / tau_ms)`.
    pub fn grid_traces(&self, tau_ms: f32) -> Vec<Vec<f32>> {
        let k = self.cfg.dt_ms / tau_ms.max(1e-6);
        self.grid_watches
            .iter()
            .map(|w| {
                w.ls.iter()
                    .map(|&l| {
                        if l == u32::MAX {
                            0.0
                        } else {
                            (-(self.t.saturating_sub(l as u64) as f32) * k).exp()
                        }
                    })
                    .collect()
            })
            .collect()
    }

    pub fn run_ms(&mut self, ms: f32) -> anyhow::Result<()> {
        let steps = (ms / self.cfg.dt_ms).round() as usize;
        self.run_steps(steps)
    }

    pub fn time_ms(&self) -> f64 {
        self.t as f64 * self.cfg.dt_ms as f64
    }

    /// Mean firing rate over the last `steps` steps, Hz.
    pub fn mean_rate_hz(&self) -> f64 {
        if self.t == 0 {
            return 0.0;
        }
        self.spikes_total as f64
            / self.n.max(1) as f64
            / (self.t as f64 * self.cfg.dt_ms as f64 / 1000.0)
    }

    /// Spike bitmap for a step in the recorded window (`back` steps ago).
    pub fn spike_row(&self, back: usize) -> Option<&[u64]> {
        if !self.cfg.record_spikes
            || self.t == 0
            || back >= self.t as usize
            || back >= self.spike_counts.len()
        {
            return None;
        }
        let t = self.t as usize - 1 - back;
        let row = t % self.spike_counts.len();
        Some(&self.spike_bits[row * self.words..(row + 1) * self.words])
    }

    /// Readout-trace of a subset of units.
    pub fn trace_of(&self, units: &[u32]) -> Vec<f32> {
        units.iter().map(|&i| self.trace[i as usize]).collect()
    }

    /// Per-neuron spike counts over the last `rows` steps.
    ///
    /// `rows >= t`: exact lifetime totals. `rows <= record ring depth`: exact
    /// sliding-window count. Anything in between cannot be answered (the ring
    /// already wrapped) and returns an error rather than silently capping.
    pub fn spike_counts(&self, rows: usize) -> Result<Vec<u32>, String> {
        let mut counts = vec![0u32; self.n];
        if !self.cfg.record_spikes || self.t == 0 {
            return Ok(counts);
        }
        if rows >= self.t as usize {
            for (c, &tot) in counts.iter_mut().zip(self.pn_total.iter()) {
                *c = tot.min(u32::MAX as u64) as u32;
            }
            return Ok(counts);
        }
        if rows > self.spike_counts.len() {
            return Err(format!(
                "spike_counts window {rows} exceeds the recorded {} steps",
                self.spike_counts.len()
            ));
        }
        for back in 0..rows {
            let Some(row) = self.spike_row(back) else {
                break;
            };
            for (wi, &w) in row.iter().enumerate() {
                let mut w = w;
                while w != 0 {
                    let b = w.trailing_zeros() as usize;
                    w &= w - 1;
                    let i = wi * 64 + b;
                    if i < self.n {
                        counts[i] += 1;
                    }
                }
            }
        }
        Ok(counts)
    }

    pub fn set_input_constant(&mut self, input: usize, value: f32) {
        self.stimulus.constants[input] = value;
    }

    pub fn play_input(&mut self, input: usize, series: Vec<f32>) {
        self.stimulus.play(input, self.t, series);
    }

    pub fn set_external(&mut self, neuron: usize, amps: f32) {
        self.ext[neuron] = amps;
    }
}
