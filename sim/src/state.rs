//! Shared simulation configuration and the flat parameter layout both
//! backends (CPU reference and CUDA) are built from.

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

/// Flat neuron parameter/state arrays built from a `Network`.
#[derive(Clone, Debug)]
pub struct NeuronParams {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub d: Vec<f32>,
    /// Kind as 0/1 (excitatory/inhibitory) — used for colouring and defaults.
    pub inhibitory: Vec<u8>,
    pub pos: Vec<[f32; 3]>,
    pub init_v: f32,
    pub init_u: Vec<f32>,
}

impl NeuronParams {
    pub fn from_network(net: &Network) -> Self {
        let n = net.neurons.len();
        let mut p = NeuronParams {
            a: vec![0.0; n],
            b: vec![0.0; n],
            c: vec![0.0; n],
            d: vec![0.0; n],
            inhibitory: vec![0; n],
            pos: vec![[0.0; 3]; n],
            init_v: -65.0,
            init_u: vec![0.0; n],
        };
        for (i, neuron) in net.neurons.iter().enumerate() {
            let [a, b, c, d] = neuron.model.params();
            p.a[i] = a;
            p.b[i] = b;
            p.c[i] = c;
            p.d[i] = d;
            p.inhibitory[i] = (neuron.kind == net_format::NeuronKind::Inhibitory) as u8;
            p.pos[i] = neuron.pos;
        }
        for i in 0..n {
            p.init_u[i] = p.b[i] * p.init_v;
        }
        p
    }
}

/// Ring-buffer depth (steps) needed for `csr` at step `dt_ms`.
pub fn ring_depth(csr: &SynCsr, _dt_ms: f32) -> usize {
    let max_steps = csr.delay_steps.iter().copied().max().unwrap_or(1) as usize;
    (max_steps + 1).max(2)
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
