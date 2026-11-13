//! sim: the neuro_edit v2 simulator library + headless binary.
//!
//! Two backends behind one API: `cpu` (reference, always available) and
//! `cuda` (cudarc/NVRTC kernels). Both step the same flattened model built
//! from `net_format::Network`.

pub mod bench;
pub mod cpu;
pub mod dataset;
pub mod model;
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

/// The simulator implementations behind `Sim` — cpu (reference) and
/// cuda (cudarc/NVRTC). Every call `Sim` makes on a backend goes through
/// this trait; a third backend is one new impl, not 20 edits to `Sim`.
///
/// State is addressed by slot: slot 0 is the membrane potential on every
/// registered model (`NeuronModel::state()[0]`). `Sim::v()` stays as the
/// slot-0 alias callers already use.
pub trait Backend {
    /// Which `BackendKind` built this sim — tests and rebuild logic use
    /// it where they used to match the `Sim` enum.
    fn kind(&self) -> BackendKind;
    fn n(&self) -> usize;
    fn step_count(&self) -> u64;
    fn time_ms(&self) -> f64;
    fn run_steps(&mut self, n: usize) -> anyhow::Result<()>;
    fn run_ms(&mut self, ms: f32) -> anyhow::Result<()>;
    /// Reset dynamical state (keeps synapses, stimulus and ext).
    fn reset_state(&mut self);
    /// Drain queued device work; a no-op on synchronous backends.
    fn synchronize(&self) {}
    /// Approximate device memory, for reporting (0 on CPU).
    fn dev_mem_bytes(&self) -> usize {
        0
    }
    /// Filtered readout trace (one value per neuron).
    fn trace(&self) -> Vec<f32>;
    /// Model state slot column, one value per neuron.
    fn state(&self, slot: usize) -> Vec<f32>;
    /// Spikes in the last recorded step (indices). Empty unless spike
    /// recording is on.
    fn last_spikes(&self) -> Vec<u32>;
    /// Per-neuron spike counts over the last `rows` steps. Exact for
    /// `rows <= record ring depth` and for `rows >= t` (lifetime totals);
    /// errors for windows in between (the ring already wrapped).
    fn spike_counts(&self, rows: usize) -> Result<Vec<u32>, String>;
    /// Spike bitmap for `back` steps ago (u64 words), if recording.
    fn spike_row(&self, back: usize) -> Option<Vec<u64>>;
    fn spikes_total(&self) -> u64;
    /// Mean firing rate over the whole run so far, Hz.
    fn mean_rate_hz(&self) -> f64;
    fn set_input_constant(&mut self, input: usize, value: f32);
    /// Current constant drive per input (same order as `net.inputs`).
    fn input_values(&self) -> Vec<f32>;
    /// The full stimulus program — cloned for carryover when the sim is
    /// rebuilt on another backend/config.
    fn stimulus(&self) -> crate::stim::Stimulus;
    /// External currents per neuron (host copy).
    fn external(&self) -> Vec<f32>;
    /// Play a per-step series on `input` starting at the current step.
    fn play_input(&mut self, input: usize, series: Vec<f32>);
    /// Set watched-neuron ids whose per-step state is recorded (editor
    /// plots). No-op when unchanged; safe to call every publish.
    fn set_v_watch(&mut self, ids: &[u32]);
    /// Drain per-step samples for watched neurons recorded since the last
    /// drain — `[state slot][watch][samples oldest-first]`. One call so a
    /// backend reading a shared ring resets the counter once.
    fn drain_watch(&mut self) -> anyhow::Result<Vec<Vec<Vec<f32>>>>;
    /// Set the grid-group member lists (texel order) whose spike traces
    /// the editor's plane view wants.
    fn set_grid_watches(&mut self, groups: &[Vec<u32>]);
    /// Materialized spike traces for each watched grid group — one f32
    /// per texel (member order), `exp(-(t - last_spike) * dt / tau_ms)`.
    fn grid_traces(&mut self, tau_ms: f32) -> anyhow::Result<Vec<Vec<f32>>>;
    /// Constant external current on one neuron.
    fn set_external(&mut self, neuron: usize, amps: f32);
    /// Set the whole external-current vector at once — the rebuild
    /// carryover path (`DriveState::apply`).
    fn set_external_all(&mut self, ext: &[f32]);
    /// Change the background noise rate on a live sim.
    fn set_noise(&mut self, rate_hz: f32, kick_mv: f32);
}

/// Facade over a `Box<dyn Backend>` chosen by `BackendKind`.
pub struct Sim {
    inner: Box<dyn Backend>,
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
        Self::with_params(
            net,
            cfg,
            kind,
            state::NeuronParams::from_network(net).map_err(anyhow::Error::msg)?,
        )
    }

    /// Build with explicit params — the file layer's per-neuron
    /// overrides and test-only models take this path.
    pub fn with_params(
        net: &Network,
        cfg: SimConfig,
        kind: BackendKind,
        params: state::NeuronParams,
    ) -> anyhow::Result<Sim> {
        let inner: Box<dyn Backend> = match kind {
            BackendKind::Cpu => {
                Box::new(cpu::CpuSim::with_params(net, cfg, params).map_err(anyhow::Error::msg)?)
            }
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => Box::new(cuda::CudaSim::with_params(net, cfg, params)?),
            #[cfg(not(feature = "cuda"))]
            BackendKind::Cuda => anyhow::bail!("sim built without the `cuda` feature"),
        };
        Ok(Sim { inner })
    }

    /// Membrane potentials (one per neuron) — model state slot 0, for
    /// plots and probes.
    pub fn v(&self) -> Vec<f32> {
        self.state(0)
    }
}

impl std::ops::Deref for Sim {
    type Target = dyn Backend;
    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}
impl std::ops::DerefMut for Sim {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.inner
    }
}
