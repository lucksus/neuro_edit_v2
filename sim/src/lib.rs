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

/// Accepted integration step range, ms — `SetDtMs`/`sim.set_dt` validate
/// against it. A step of 0 makes `(ms / dt) as usize` allocate ~2^64
/// steps; negative/NaN/huge values are likewise meaningless here.
pub const MIN_DT_MS: f32 = 0.001;
pub const MAX_DT_MS: f32 = 100.0;
pub fn valid_dt_ms(dt: f32) -> bool {
    dt.is_finite() && (MIN_DT_MS..=MAX_DT_MS).contains(&dt)
}

/// M3: persisted sim settings round-trip through `Meta`. Loading a
/// network adopts the file's simulation settings; saving writes the
/// live config back so nothing is silently reset on reopen.
pub fn meta_to_cfg(meta: &net_format::Meta, cfg: &mut crate::state::SimConfig) {
    cfg.dt_ms = meta.dt_ms;
    cfg.synapse_tau_ms = meta.synapse_tau_ms;
    cfg.readout_tau_ms = meta.readout_tau_ms;
    cfg.noise_rate_hz = meta.noise_rate_hz;
    cfg.noise_kick_mv = meta.noise_kick_mv;
    cfg.seed = meta.seed;
}

/// Mirror of `meta_to_cfg` for the save path.
pub fn cfg_to_meta(cfg: &crate::state::SimConfig, meta: &mut net_format::Meta) {
    meta.dt_ms = cfg.dt_ms;
    meta.synapse_tau_ms = cfg.synapse_tau_ms;
    meta.readout_tau_ms = cfg.readout_tau_ms;
    meta.noise_rate_hz = cfg.noise_rate_hz;
    meta.noise_kick_mv = cfg.noise_kick_mv;
    meta.seed = cfg.seed;
}

/// Server-side ceiling on sim memory, checked once before a sim build.
/// All zeros = unlimited — the local desktop app has no server to
/// protect, so it runs with the default (unlimited) limits.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SimLimits {
    /// Neuron count cap (0 = unlimited).
    pub max_neurons: usize,
    /// Synapse count cap (0 = unlimited).
    pub max_synapses: usize,
    /// Delay-ring bytes cap: `ring_depth × neurons × 4` (0 = unlimited).
    pub max_ring_bytes: u64,
}

impl SimLimits {
    /// `Ok` when `net` fits at `dt_ms`; `Err` names the blown limit.
    /// The ring check builds the CSR — same order of work as the build
    /// itself, so it costs nothing extra to run it here once.
    pub fn check(&self, net: &Network, dt_ms: f32) -> Result<(), String> {
        let n = net.neurons.len();
        if self.max_neurons > 0 && n > self.max_neurons {
            return Err(format!(
                "net has {n} neurons (server limit {})",
                self.max_neurons
            ));
        }
        let s = net.synapses.len();
        if self.max_synapses > 0 && s > self.max_synapses {
            return Err(format!(
                "net has {s} synapses (server limit {})",
                self.max_synapses
            ));
        }
        if self.max_ring_bytes > 0 {
            let csr = net.synapse_csr(dt_ms)?;
            let depth = crate::state::ring_depth(&csr, dt_ms) as u64;
            let bytes = depth.saturating_mul(n as u64).saturating_mul(4);
            if bytes > self.max_ring_bytes {
                return Err(format!(
                    "delay ring needs {} MiB at dt {dt_ms} ms (server limit {} MiB) \
                     — fewer neurons, shorter delays or a bigger dt",
                    bytes >> 20,
                    self.max_ring_bytes >> 20
                ));
            }
        }
        Ok(())
    }
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
        // Device + driver probe only — CudaSim::new would NVRTC-compile
        // every kernel just to answer this question.
        cudarc::driver::CudaContext::new(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

impl Sim {
    pub fn new(net: &Network, cfg: SimConfig, kind: BackendKind) -> anyhow::Result<Sim> {
        match kind {
            BackendKind::Cpu => Ok(Sim::Cpu(Box::new(
                cpu::CpuSim::new(net, cfg).map_err(anyhow::Error::msg)?,
            ))),
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

    /// The full stimulus program (constants + series + playheads) —
    /// cloned for carryover when the sim is rebuilt on another
    /// backend/config.
    pub fn stimulus(&self) -> crate::stim::Stimulus {
        match self {
            Sim::Cpu(s) => s.stimulus.clone(),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.stimulus.clone(),
        }
    }

    /// External currents per neuron (host copy; on CUDA this is one
    /// device→host read — meant for rebuild snapshots, not hot paths).
    pub fn external(&self) -> Vec<f32> {
        match self {
            Sim::Cpu(s) => s.ext.clone(),
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.external(),
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

    /// Set the whole external-current vector at once — the rebuild
    /// carryover path (`DriveState::apply`).
    pub fn set_external_all(&mut self, ext: &[f32]) {
        match self {
            Sim::Cpu(s) => {
                let n = s.ext.len().min(ext.len());
                s.ext[..n].copy_from_slice(&ext[..n]);
            }
            #[cfg(feature = "cuda")]
            Sim::Cuda(s) => s.set_external_all(ext),
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
