//! sim: the neuro_edit v2 simulator library + headless binary.
//!
//! Backends behind one API: `cpu` (reference, always available) and `cuda`
//! (cudarc/NVRTC kernels). Both step the same flattened model built from
//! `net_format::Network`.

pub mod cpu;
pub mod readout;
pub mod state;
pub mod stim;

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
}

impl Sim {
    pub fn new(net: &Network, cfg: SimConfig, kind: BackendKind) -> anyhow::Result<Sim> {
        match kind {
            BackendKind::Cpu => Ok(Sim::Cpu(Box::new(cpu::CpuSim::new(net, cfg)))),
            BackendKind::Cuda => anyhow::bail!("cuda backend not built yet"),
        }
    }

    pub fn n(&self) -> usize {
        match self {
            Sim::Cpu(s) => s.n,
        }
    }

    pub fn time_ms(&self) -> f64 {
        match self {
            Sim::Cpu(s) => s.time_ms(),
        }
    }

    pub fn step_count(&self) -> u64 {
        match self {
            Sim::Cpu(s) => s.t,
        }
    }

    pub fn run_steps(&mut self, n: usize) {
        match self {
            Sim::Cpu(s) => s.run_steps(n),
        }
    }

    pub fn run_ms(&mut self, ms: f32) {
        match self {
            Sim::Cpu(s) => s.run_ms(ms),
        }
    }

    pub fn reset_state(&mut self) {
        match self {
            Sim::Cpu(s) => s.reset_state(),
        }
    }

    /// Filtered readout trace (one value per neuron).
    pub fn trace(&self) -> Vec<f32> {
        match self {
            Sim::Cpu(s) => s.trace.clone(),
        }
    }

    /// Membrane potentials (one per neuron) — for plots and probes.
    pub fn v(&self) -> Vec<f32> {
        match self {
            Sim::Cpu(s) => s.v.clone(),
        }
    }

    /// Spikes in the last recorded step (indices).
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
        }
    }

    /// Spike bitmap for `back` steps ago (u64 words), if recording.
    pub fn spike_row(&self, back: usize) -> Option<Vec<u64>> {
        match self {
            Sim::Cpu(s) => s.spike_row(back).map(|r| r.to_vec()),
        }
    }

    pub fn spikes_total(&self) -> u64 {
        match self {
            Sim::Cpu(s) => s.spikes_total,
        }
    }

    pub fn mean_rate_hz(&self) -> f64 {
        match self {
            Sim::Cpu(s) => s.mean_rate_hz(),
        }
    }

    pub fn set_input_constant(&mut self, input: usize, value: f32) {
        match self {
            Sim::Cpu(s) => s.stimulus.constants[input] = value,
        }
    }

    /// Play a per-step series on `input` starting at the current step.
    pub fn play_input(&mut self, input: usize, series: Vec<f32>) {
        match self {
            Sim::Cpu(s) => {
                s.stimulus.series[input] = series;
                s.stimulus.play_start = s.t;
            }
        }
    }

    /// Constant external current on one neuron.
    pub fn set_external(&mut self, neuron: usize, amps: f32) {
        match self {
            Sim::Cpu(s) => s.ext[neuron] = amps,
        }
    }

    /// Change the background noise rate on a live sim.
    pub fn set_noise(&mut self, rate_hz: f32, kick_mv: f32) {
        match self {
            Sim::Cpu(s) => {
                s.cfg.noise_rate_hz = rate_hz;
                s.cfg.noise_kick_mv = kick_mv;
            }
        }
    }
}
