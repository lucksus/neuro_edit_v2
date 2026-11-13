//! Shared simulation configuration and the flat parameter layout both
//! backends (CPU reference and CUDA) are built from.

use crate::model::{self, NeuronModel};
use net_format::{Network, SynCsr};

#[derive(Clone, Debug, PartialEq)]
pub struct SimConfig {
    /// Integration step, ms. Original default: 0.1.
    pub dt_ms: f32,
    /// Synaptic current decay tau, ms. Original default tc=50 -> 2 ms.
    pub synapse_tau_ms: f32,
    /// Readout filtered-trace decay tau, ms.
    pub readout_tau_ms: f32,
    /// Background Poisson kick rate per neuron, Hz (0 = quiet).
    pub noise_rate_hz: f32,
    /// Kick amplitude: the current is `kick/dt` for one step, i.e. about
    /// `kick` mV of instantaneous depolarisation (the bench used 20).
    pub noise_kick_mv: f32,
    /// Seed for the counter-based per-step noise.
    pub seed: u64,
    /// Keep a rolling spike bitmap for raster plots and the editor stream.
    pub record_spikes: bool,
    /// Rolling window depth, in steps.
    pub spike_record_steps: usize,
    /// CUDA only: accumulate the delay ring in 32-bit fixed point
    /// (scale 4096). Integer atomicAdd is order-independent, so runs are
    /// bitwise reproducible; the float ring is not (atomic order varies).
    pub int_weights: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            dt_ms: net_format::DEFAULT_DT_MS,
            synapse_tau_ms: net_format::DEFAULT_SYNAPSE_TAU_MS,
            readout_tau_ms: net_format::DEFAULT_READOUT_TAU_MS,
            noise_rate_hz: 0.0,
            noise_kick_mv: 20.0,
            seed: 1,
            record_spikes: false,
            spike_record_steps: 512,
            int_weights: false,
        }
    }
}

/// Flat per-neuron model/param/state layout both backends run.
///
/// `model[i]` indexes `models` — a per-net table in first-use order;
/// neuron i's param row is `p[i*n_params ..]` and its initial-state row
/// is `x0[i*n_state ..]`. Row widths are the max over the net's models —
/// a model only reads its own slots, so a wider global row is harmless.
#[derive(Clone)]
pub struct NeuronParams {
    /// the net's model table: `model[i]` → registry model
    pub models: Vec<&'static dyn NeuronModel>,
    /// per-neuron index into `models`
    pub model: Vec<u8>,
    /// max param-row width over `models`
    pub n_params: usize,
    /// max state-row width over `models`
    pub n_state: usize,
    /// flat param rows, n * n_params
    pub p: Vec<f32>,
    /// flat initial-state rows, n * n_state — `reset_state` restores it
    pub x0: Vec<f32>,
    /// Kind as 0/1 (excitatory/inhibitory) — colouring and synapse sign.
    pub inhibitory: Vec<u8>,
    pub pos: Vec<[f32; 3]>,
}

impl std::fmt::Debug for NeuronParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuronParams")
            .field(
                "models",
                &self.models.iter().map(|m| m.id()).collect::<Vec<_>>(),
            )
            .field("model", &self.model)
            .field("n_params", &self.n_params)
            .field("n_state", &self.n_state)
            .field("inhibitory", &self.inhibitory)
            .field("pos", &self.pos)
            .finish_non_exhaustive()
    }
}

impl NeuronParams {
    /// Pull the sim-facing blocks out of a v4 net: each neuron's type
    /// names a registered model, its effective param row (override or
    /// type) becomes its `p` row, and `x0` is the model's init of that
    /// row. `Err` on a dangling `ty`, an unregistered model id, or a
    /// param row that doesn't match the model's descriptor width.
    pub fn from_network(net: &Network) -> Result<Self, String> {
        let n = net.neurons.len();
        let mut params = NeuronParams {
            models: Vec::new(),
            model: Vec::with_capacity(n),
            n_params: 0,
            n_state: 0,
            p: Vec::new(),
            x0: Vec::new(),
            inhibitory: Vec::with_capacity(n),
            pos: Vec::with_capacity(n),
        };
        let mut rows: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(n);
        for (i, neuron) in net.neurons.iter().enumerate() {
            let ty = net
                .neuron_type(i as u32)
                .ok_or_else(|| format!("neuron {i}: ty {} out of range", neuron.ty))?;
            let (_, m) = model::by_id(&ty.model)
                .ok_or_else(|| format!("neuron {i}: unknown model {:?}", ty.model))?;
            let p = net.neuron_params(i as u32);
            if p.len() != m.params().len() {
                return Err(format!(
                    "neuron {i}: {} params, model {} wants {}",
                    p.len(),
                    m.id(),
                    m.params().len()
                ));
            }
            let mi = match params.models.iter().position(|e| e.id() == m.id()) {
                Some(k) => k as u8,
                None => {
                    params.models.push(m);
                    params.models.len() as u8 - 1
                }
            };
            params.n_params = params.n_params.max(m.params().len());
            params.n_state = params.n_state.max(m.state().len());
            params.model.push(mi);
            params
                .inhibitory
                .push((ty.kind == net_format::NeuronKind::Inhibitory) as u8);
            params.pos.push(neuron.pos);
            let mut x0 = vec![0.0; m.state().len()];
            m.init(p, &mut x0);
            rows.push((p.to_vec(), x0));
        }
        // Pad all rows to the widest — a model only reads its own slots.
        params.p = vec![0.0; n * params.n_params];
        params.x0 = vec![0.0; n * params.n_state];
        for (i, (p, x0)) in rows.into_iter().enumerate() {
            params.p[i * params.n_params..i * params.n_params + p.len()].copy_from_slice(&p);
            params.x0[i * params.n_state..i * params.n_state + x0.len()].copy_from_slice(&x0);
        }
        params.check(n)?;
        Ok(params)
    }

    /// Re-pack `p`/`x0` to new row widths, zero-padding grown slots —
    /// model rows keep their contents in the same order.
    pub fn grow(&mut self, n_params: usize, n_state: usize) {
        if n_params == self.n_params && n_state == self.n_state {
            return;
        }
        let n = self.model.len();
        let mut p = vec![0.0; n * n_params];
        let mut x0 = vec![0.0; n * n_state];
        for i in 0..n {
            let kp = self.n_params.min(n_params);
            p[i * n_params..i * n_params + kp]
                .copy_from_slice(&self.p[i * self.n_params..i * self.n_params + kp]);
            let kx = self.n_state.min(n_state);
            x0[i * n_state..i * n_state + kx]
                .copy_from_slice(&self.x0[i * self.n_state..i * self.n_state + kx]);
        }
        self.p = p;
        self.x0 = x0;
        self.n_params = n_params;
        self.n_state = n_state;
    }

    /// Point neuron `i` at `models[mi]` with param row `params` and
    /// re-init its state row. `grow` first when `params` outruns
    /// `n_params`; a shorter row zero-pads.
    pub fn set(&mut self, i: usize, mi: u8, params: &[f32]) {
        self.model[i] = mi;
        let (np, nx) = (self.n_params, self.n_state);
        {
            let row = &mut self.p[i * np..i * np + np];
            row.fill(0.0);
            let k = params.len().min(np);
            row[..k].copy_from_slice(&params[..k]);
        }
        let xrow = &mut self.x0[i * nx..i * nx + nx];
        xrow.fill(0.0);
        self.models[mi as usize].init(&self.p[i * np..i * np + np], xrow);
    }

    /// Sanity check a params block against a net before a sim build:
    /// model indices in range, rows sized right.
    pub fn check(&self, n: usize) -> Result<(), String> {
        if self.model.len() != n
            || self.p.len() != n * self.n_params
            || self.x0.len() != n * self.n_state
        {
            return Err(format!(
                "param/state rows ({}/{}/{}) don't match {n} neurons",
                self.model.len(),
                self.p.len(),
                self.x0.len()
            ));
        }
        for (i, &mi) in self.model.iter().enumerate() {
            if mi as usize >= self.models.len() {
                return Err(format!(
                    "neuron {i} names model {mi} outside the model table"
                ));
            }
        }
        Ok(())
    }
}

/// Ring-buffer depth (steps) needed for `csr` at step `dt_ms`.
pub fn ring_depth(csr: &SynCsr, _dt_ms: f32) -> usize {
    let max_steps = csr.delay_steps.iter().copied().max().unwrap_or(1) as usize;
    (max_steps + 1).max(2)
}

/// Per-synapse-state ("x") tables for custom-tau synapses: synapses with
/// `tau_ms > 0` get their own exponential current, all others keep the
/// merged `s[i]` path. Built once here so the CPU and CUDA backends can
/// not drift on the layout (3.2).
#[derive(Clone, Debug, Default)]
pub struct XSyn {
    /// synapse j → x-slot, `u32::MAX` = merged path.
    pub map: Vec<u32>,
    /// per-slot decay factor `exp(-dt/tau)`.
    pub decay: Vec<f32>,
    /// per-slot weight (CPU integrates directly; CUDA reads `csr.w`).
    pub w: Vec<f32>,
    /// target-major rows over slots: neuron i's incoming x-slots are
    /// `xt_idx[xt_ptr[i]..xt_ptr[i+1]]`.
    pub xt_ptr: Vec<u32>,
    pub xt_idx: Vec<u32>,
}

impl XSyn {
    pub fn build(csr: &SynCsr, dt_ms: f32) -> XSyn {
        let n = csr.src_ptr.len().saturating_sub(1);
        let mut map = vec![u32::MAX; csr.w.len()];
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n];
        let (mut decay, mut w) = (Vec::new(), Vec::new());
        for (j, &tau) in csr.tau_ms.iter().enumerate() {
            if tau > 0.0 {
                let slot = decay.len() as u32;
                map[j] = slot;
                decay.push((-dt_ms / tau.max(1e-6)).exp());
                w.push(csr.w[j]);
                rows[csr.tgt[j] as usize].push(slot);
            }
        }
        let mut xt_ptr = vec![0u32; n + 1];
        let mut xt_idx = Vec::with_capacity(decay.len());
        for row in &rows {
            xt_idx.extend_from_slice(row);
        }
        for i in 0..n {
            xt_ptr[i + 1] = xt_ptr[i] + rows[i].len() as u32;
        }
        XSyn {
            map,
            decay,
            w,
            xt_ptr,
            xt_idx,
        }
    }
}

/// Counter-based per-(neuron,step) hash for reproducible background noise —
/// the same mixing as the CUDA kernel uses, so both backends draw the same
/// kicks for a given seed. SplitMix64: the (i,t,seed) tuple is combined
/// with three independent 64-bit strides and run through the full
/// splitmix64 avalanche — the previous version truncated a linear combo
/// to u32 before mixing, which weakens avalanche diffusion.
#[inline(always)]
pub fn noise_hash(i: u32, t: u64, seed: u64) -> u32 {
    let mut h = (i as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(t.wrapping_mul(0xBF58476D1CE4E5B9))
        .wrapping_add(seed.wrapping_mul(0x94D049BB133111EB));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 31;
    (h >> 32) as u32
}

/// Kick flag for neuron `i` at step `t`: Poisson with rate `rate_hz`.
#[inline(always)]
pub fn noise_fires(i: u32, t: u64, seed: u64, p: f32) -> bool {
    (noise_hash(i, t, seed) as f32) * 2.3283064e-10 < p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xsyn_layout_only_custom_synapses_get_slots() {
        // 0→1 merged, 0→2 custom tau, 1→2 custom tau: two x-slots,
        // target-major rows must list each neuron's incoming slots.
        let mut net = net_format::Network::new();
        for i in 0..3 {
            net.add_neuron([i as f32, 0.0, 0.0], net_format::NeuronKind::Excitatory);
        }
        net.connect(0, 1, 1.0);
        net.connect(0, 2, 2.0);
        net.connect(1, 2, 3.0);
        net.synapses[1].tau_ms = 20.0;
        net.synapses[2].tau_ms = 10.0;
        let csr = net.synapse_csr(0.1).unwrap();
        let x = XSyn::build(&csr, 0.1);
        // CSR sorts by src: [0→1, 0→2, 1→2] keeps insertion order here.
        assert_eq!(x.map, vec![u32::MAX, 0, 1]);
        assert_eq!(x.w, vec![2.0, 3.0]);
        assert_eq!(x.decay.len(), 2);
        assert!((x.decay[0] - (-0.1f32 / 20.0).exp()).abs() < 1e-7);
        // neuron 2 receives both custom slots; neuron 1 gets none.
        assert_eq!(
            &x.xt_idx[x.xt_ptr[2] as usize..x.xt_ptr[3] as usize],
            &[0, 1]
        );
        assert_eq!(x.xt_ptr[1], x.xt_ptr[2]);
    }
}
