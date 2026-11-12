//! sim: the neuro_edit v2 simulator library + headless binary.
//!
//! Two backends behind one API: `cpu` (reference, always available) and
//! `cuda` (cudarc/NVRTC kernels). Both step the same flattened model built
//! from `net_format::Network`.

pub mod bench;
pub mod cpu;
pub mod dataset;
pub mod readout;
pub mod script;
pub mod serve;
pub mod state;
pub mod stim;

pub mod kernels;

#[cfg(feature = "cuda")]
pub mod cuda;

use net_format::Network;
pub use state::SimConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
}

/// Facade that dispatches to the chosen backend.
pub enum Sim {
    Cpu(Box<cpu::CpuSim>),
    #[cfg(feature = "cuda")]
    Cuda(Box<cuda::CudaSim>),
}

/// True when a CUDA device is usable (feature `cuda` + driver + device 0).
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        let mut n = Network::new();
        n.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
        cuda::CudaSim::new(&n, SimConfig::default()).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

impl Sim {
    pub fn new(net: &Network, cfg: SimConfig, kind: BackendKind) -> anyhow::Result<Sim> {
        match kind {
            BackendKind::Cpu => Ok(Sim::Cpu(Box::new(cpu::CpuSim::new(net, cfg)))),
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => Ok(Sim::Cuda(Box::new(cuda::CudaSim::new(net, cfg)?))),
            #[cfg(not(feature = "cuda"))]
            BackendKind::Cuda => anyhow::bail!("sim built without the `cuda` feature"),
        }
    }

    pub fn n(&self) -> usize {
        match self {
            Sim::Cpu(s) => s.n,
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.n,
        }
    }

    pub fn time_ms(&self) -> f64 {
        match self {
            Sim::Cpu(s) => s.time_ms(),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.time_ms(),
        }
    }

    pub fn step_count(&self) -> u64 {
        match self {
            Sim::Cpu(s) => s.t,
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.t,
        }
    }

    pub fn run_steps(&mut self, n: usize) -> anyhow::Result<()> {
        match self {
            Sim::Cpu(s) => s.run_steps(n),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.run_steps(n),
        }
    }

    pub fn run_ms(&mut self, ms: f32) -> anyhow::Result<()> {
        match self {
            Sim::Cpu(s) => s.run_ms(ms),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.run_ms(ms),
        }
    }

    pub fn reset_state(&mut self) {
        match self {
            Sim::Cpu(s) => s.reset_state(),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.reset_state(),
        }
    }

    /// Filtered readout trace (one value per neuron).
    pub fn trace(&self) -> Vec<f32> {
        match self {
            Sim::Cpu(s) => s.trace.clone(),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.trace(),
        }
    }

    /// Membrane potentials (one per neuron) — for plots and probes.
    pub fn v(&self) -> Vec<f32> {
        match self {
            Sim::Cpu(s) => s.v.clone(),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.v(),
        }
    }

    /// Spikes in the last recorded step (indices). CUDA returns an empty vec
    /// unless spike recording is on.
    pub fn last_spikes(&self) -> Vec<u32> {
        match self {
            Sim::Cpu(s) => {
                if let Some(row) = s.spike_row(0) {
                    let mut out = Vec::new();
                    for (wi, &w) in row.iter().enumerate() {
                        for b in 0..64 {
                            if w & (1 << b) != 0 {
                                let i = wi * 64 + b;
                                if i < s.n {
                                    out.push(i as u32);
                                }
                            }
                        }
                    }
                    out
                } else {
                    Vec::new()
                }
            }
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.last_spikes(),
        }
    }

    /// Per-neuron spike counts over the last `rows` steps. Exact for
    /// `rows <= record ring depth` and for `rows >= t` (lifetime totals);
    /// errors for windows in between (the ring already wrapped).
    pub fn spike_counts(&self, rows: usize) -> Result<Vec<u32>, String> {
        match self {
            Sim::Cpu(s) => s.spike_counts(rows),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.spike_counts(rows),
        }
    }

    /// Spike bitmap for `back` steps ago (u64 words), if recording.
    pub fn spike_row(&self, back: usize) -> Option<Vec<u64>> {
        match self {
            Sim::Cpu(s) => s.spike_row(back).map(|r| r.to_vec()),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.spike_row(back),
        }
    }

    pub fn spikes_total(&self) -> u64 {
        match self {
            Sim::Cpu(s) => s.spikes_total,
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.spikes_total(),
        }
    }

    /// Mean firing rate over the whole run so far, Hz.
    pub fn mean_rate_hz(&self) -> f64 {
        match self {
            Sim::Cpu(s) => s.mean_rate_hz(),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.mean_rate_hz(),
        }
    }

    pub fn set_input_constant(&mut self, input: usize, value: f32) {
        match self {
            Sim::Cpu(s) => s.stimulus.constants[input] = value,
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.set_input_constant(input, value),
        }
    }

    /// Current constant drive per input (same order as `net.inputs`) —
    /// what the editor's Inputs panel should display.
    pub fn input_values(&self) -> Vec<f32> {
        match self {
            Sim::Cpu(s) => s.stimulus.constants.clone(),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.stimulus.constants.clone(),
        }
    }

    /// Play a per-step series on `input` starting at the current step.
    pub fn play_input(&mut self, input: usize, series: Vec<f32>) {
        match self {
            Sim::Cpu(s) => s.stimulus.play(input, s.t, series),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.play_input(input, series),
        }
    }

    /// Set watched-neuron ids whose per-step v is recorded (editor plot).
    /// No-op when unchanged; safe to call every publish.
    pub fn set_v_watch(&mut self, ids: &[u32]) {
        match self {
            Sim::Cpu(s) => s.set_v_watch(ids),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.set_v_watch(ids),
        }
    }

    /// Drain per-step v samples recorded since the last drain, one Vec per
    /// watched neuron (oldest first).
    pub fn drain_v(&mut self) -> anyhow::Result<Vec<Vec<f32>>> {
        match self {
            Sim::Cpu(s) => Ok(s.drain_v()),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.drain_v(),
        }
    }

    /// Drain per-step v AND u samples for watched neurons — one call so
    /// the CUDA backend can read both rings before it resets the shared
    /// sample counter (u first).
    pub fn drain_vu(&mut self) -> anyhow::Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        match self {
            Sim::Cpu(s) => Ok((s.drain_v(), s.drain_u())),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => {
                let u = s.drain_u()?;
                let v = s.drain_v()?;
                Ok((v, u))
            }
        }
    }

    /// Set the grid-group member lists (texel order) whose spike traces the
    /// editor's plane view wants. No-op when unchanged; safe to call every
    /// publish.
    pub fn set_grid_watches(&mut self, groups: &[Vec<u32>]) {
        match self {
            Sim::Cpu(s) => s.set_grid_watches(groups),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.set_grid_watches(groups),
        }
    }

    /// Materialized spike traces for each watched grid group — one f32 per
    /// texel (member order), `exp(-(t - last_spike) * dt / tau_ms)`.
    pub fn grid_traces(&mut self, tau_ms: f32) -> anyhow::Result<Vec<Vec<f32>>> {
        match self {
            Sim::Cpu(s) => Ok(s.grid_traces(tau_ms)),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.grid_traces(tau_ms),
        }
    }

    /// Constant external current on one neuron.
    pub fn set_external(&mut self, neuron: usize, amps: f32) {
        match self {
            Sim::Cpu(s) => {
                if let Some(e) = s.ext.get_mut(neuron) {
                    *e = amps;
                }
            }
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.set_external(neuron, amps),
        }
    }

    /// Change the background noise rate on a live sim.
    pub fn set_noise(&mut self, rate_hz: f32, kick_mv: f32) {
        match self {
            Sim::Cpu(s) => {
                s.cfg.noise_rate_hz = rate_hz;
                s.cfg.noise_kick_mv = kick_mv;
            }
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.set_noise(rate_hz, kick_mv),
        }
    }
}
