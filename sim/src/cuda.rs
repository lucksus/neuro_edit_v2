//! CUDA backend: `kernels.rs` source via NVRTC, driven by cudarc.
//!
//! Per step: `neuron_kernel` (one thread per neuron) then `prop_kernel` (one
//! thread per possible spike, scattering weights into the delay ring). The
//! spike counter lives in a ping-pong pair of device ints that prop resets
//! for the next step, so `run_steps` never syncs or reads back per step.

use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc;

use crate::kernels;
use crate::state::{ring_depth, NeuronParams, SimConfig};
use net_format::{Network, SynCsr};

const BLOCK: u32 = 256;

/// Ring depth (steps) of the per-step v recording for watched neurons.
/// Matches the editor's plot cap — older samples are overwritten.
const VREC_D: usize = 4096;

pub struct CudaSim {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    f_neuron: CudaFunction,
    f_prop: CudaFunction,
    f_gtrace: CudaFunction,
    f_xsyn: CudaFunction,

    // device state — generic model blocks: neuron i's state row is
    // `x[i*nx ..]`, param row `p[i*np ..]`, model index `model[i]`
    // (`params` on the host mirror holds the same tables).
    x: CudaSlice<f32>,
    s: CudaSlice<f32>,
    trace: CudaSlice<f32>,
    ext: CudaSlice<f32>,
    /// Host mirror of `ext` — every writer keeps both in sync so
    /// `external()` (rebuild snapshots) costs no device readback.
    ext_host: Vec<f32>,
    /// Delay ring: f32 elements, or i32 fixed-point (scale 4096) when
    /// `cfg.int_weights` — same 4-byte words either way.
    ring: CudaSlice<u32>,
    p: CudaSlice<f32>,
    model: CudaSlice<u8>,
    // synapses (CSR by source)
    src_ptr: CudaSlice<u32>,
    tgt: CudaSlice<u32>,
    w: CudaSlice<f32>,
    delay: CudaSlice<u16>,
    // per-synapse current traces (C custom synapses; empty = merged path)
    xsyn_map: CudaSlice<u32>,
    xsyn: CudaSlice<f32>,
    xdecay: CudaSlice<f32>,
    xt_ptr: CudaSlice<u32>,
    xt_idx: CudaSlice<u32>,
    /// per-(slot, custom-synapse) arrival accumulation, rt_t elements
    xring: CudaSlice<u32>,
    n_xsyn: usize,
    // stimulus routing
    stim_ptr: CudaSlice<u32>,
    stim_input: CudaSlice<u32>,
    stim_w: CudaSlice<f32>,
    stim: CudaSlice<f32>,
    stim_t0: i64,
    n_inputs: usize,
    // spikes
    spk: CudaSlice<i32>,
    cnt: [CudaSlice<i32>; 2], // ping-pong by step parity
    total: CudaSlice<u64>,
    spk_bits: CudaSlice<u64>,
    /// per-neuron lifetime spike totals (exact answer for windows >= t)
    pn_total: CudaSlice<u64>,
    // per-step state recording for watched neurons (editor v(t) plot)
    /// host copy of the watched ids (compared on every set_v_watch call)
    pub v_watch: Vec<u32>,
    /// device copy of `v_watch` (W entries)
    vwatch_dev: CudaSlice<u32>,
    /// device ring VREC_D x W x X of per-step state rows for watched
    /// neurons — one row carries all state slots
    xrec_dev: CudaSlice<f32>,
    /// device buffers stale after set_v_watch — uploaded lazily in run_steps
    vrec_dirty: bool,
    /// steps recorded into vrec_dev since the last drain
    vrec_valid: u64,

    // grid-group spike traces (editor plane view): last-spike-step per
    // texel, marked inside neuron_kernel and materialized on readback
    /// host copy of each group's member list in texel order
    pub grid_watches: Vec<Vec<u32>>,
    /// concat neuron->texel map, n entries per group (-1 = not a member)
    gmap: CudaSlice<i32>,
    /// concat per-texel last-spike-step (u32::MAX = never spiked)
    gls: CudaSlice<u32>,
    /// per-group offset into gls
    goff: CudaSlice<u32>,
    /// scratch for `grid_trace_kernel` output (largest group's texel count)
    gtr: CudaSlice<f32>,
    grid_dirty: bool,

    // host-side mirrors
    pub n: usize,
    pub cfg: SimConfig,
    pub params: NeuronParams,
    pub csr: SynCsr,
    pub stimulus: crate::stim::Stimulus,
    pub ring_d: usize,
    words: usize,
    record_rows: usize,
    pub t: u64,
    /// host copy of the initial state rows — `reset_state` restores it
    x0: Vec<f32>,
}

impl CudaSim {
    pub fn new(net: &Network, cfg: SimConfig) -> Result<Self> {
        Self::with_params(net, cfg, NeuronParams::from_network(net))
    }

    /// Build with explicit params — the file layer's per-neuron
    /// overrides and test-only models take this path.
    pub fn with_params(net: &Network, cfg: SimConfig, params: NeuronParams) -> Result<Self> {
        let n = net.neurons.len();
        params.check(n).map_err(anyhow::Error::msg)?;
        let csr = net.synapse_csr(cfg.dt_ms).map_err(anyhow::Error::msg)?;
        let d = ring_depth(&csr, cfg.dt_ms);
        let stim_map = crate::stim::StimMap::from_network(net);

        let ctx = CudaContext::new(0).context("no CUDA device / driver")?;
        let stream = ctx.default_stream();

        // NVRTC compile for this device's arch: one dispatch case per
        // entry in the net's model table; int_weights selects the
        // fixed-point delay ring.
        let src = kernels::kernel_source(&params.models, cfg.int_weights);
        let ptx = nvrtc::compile_ptx(&src).context("NVRTC compile")?;
        let module: Arc<CudaModule> = ctx.load_module(ptx).context("load_module")?;
        let f_neuron = module.load_function("neuron_kernel")?;
        let f_prop = module.load_function("prop_kernel")?;
        let f_gtrace = module.load_function("grid_trace_kernel")?;
        let f_xsyn = module.load_function("xsyn_arrive_kernel")?;

        let stim_ptr_host: Vec<u32> = stim_map.neuron_ptr.clone();
        let stim_input_host = stim_map.input_idx.clone();
        let stim_w_host = stim_map.weight.clone();

        let x0 = params.x0.clone();
        let words = n.div_ceil(64);
        // >= 2 rows: prop_kernel clears the row the NEXT step will record
        // into; a depth-1 ring would clear the row just written
        let record_rows = if cfg.record_spikes {
            cfg.spike_record_steps.max(2)
        } else {
            0
        };
        // cudarc rejects 0-len buffers; on an empty net allocate one dummy
        // lane per buffer (nothing ever launches on it).
        fn pad<T: Default + Clone>(mut v: Vec<T>) -> Vec<T> {
            if v.is_empty() {
                v.push(T::default());
            }
            v
        }
        let na = n.max(1);

        // Per-synapse traces: custom synapses (tau_ms > 0) get their own
        // exponential current; all others keep the merged s path.
        let x = crate::state::XSyn::build(&csr, cfg.dt_ms);
        let n_xsyn = x.decay.len();

        let sim = CudaSim {
            x: stream.clone_htod(&pad(params.x0.clone()))?,
            s: stream.alloc_zeros(na)?,
            trace: stream.alloc_zeros(na)?,
            ext: stream.alloc_zeros(na)?,
            ext_host: vec![0.0; na],
            ring: stream.alloc_zeros((d * n).max(1))?,
            p: stream.clone_htod(&pad(params.p.clone()))?,
            model: stream.clone_htod(&pad(params.model.clone()))?,
            src_ptr: stream.clone_htod(&pad(csr.src_ptr.clone()))?,
            tgt: stream.clone_htod(&pad(csr.tgt.clone()))?,
            w: stream.clone_htod(&pad(csr.w.clone()))?,
            delay: stream.clone_htod(&pad(csr.delay_steps.clone()))?,
            xsyn_map: stream.clone_htod(&pad(x.map.clone()))?,
            xsyn: stream.alloc_zeros(n_xsyn.max(1))?,
            xdecay: stream.clone_htod(&pad(x.decay.clone()))?,
            xt_ptr: stream.clone_htod(&pad(x.xt_ptr.clone()))?,
            xt_idx: stream.clone_htod(&pad(x.xt_idx.clone()))?,
            xring: stream.alloc_zeros((d * n_xsyn).max(1))?,
            n_xsyn,
            stim_ptr: stream.clone_htod(&pad(stim_ptr_host.clone()))?,
            stim_input: stream.clone_htod(&pad(stim_input_host.clone()))?,
            stim_w: stream.clone_htod(&pad(stim_w_host.clone()))?,
            stim: stream.alloc_zeros(1)?,
            stim_t0: i64::MAX,
            n_inputs: net.inputs.len(),
            spk: stream.alloc_zeros(na)?,
            cnt: [stream.alloc_zeros(1)?, stream.alloc_zeros(1)?],
            total: stream.alloc_zeros(1)?,
            spk_bits: stream.alloc_zeros((record_rows * words).max(1))?,
            pn_total: stream.alloc_zeros(na)?,
            v_watch: Vec::new(),
            vwatch_dev: stream.alloc_zeros(1)?,
            xrec_dev: stream.alloc_zeros(1)?,
            vrec_dirty: false,
            vrec_valid: 0,
            grid_watches: Vec::new(),
            gmap: stream.alloc_zeros(1)?,
            gls: stream.alloc_zeros(1)?,
            goff: stream.alloc_zeros(1)?,
            gtr: stream.alloc_zeros(1)?,
            grid_dirty: false,
            n,
            cfg,
            params,
            csr,
            stimulus: crate::stim::Stimulus::for_network(net),
            ring_d: d,
            words,
            record_rows,
            t: 0,
            x0,
            ctx,
            stream,
            f_neuron,
            f_prop,
            f_gtrace,
            f_xsyn,
        };
        Ok(sim)
    }

    fn upload_stim(&mut self, n_steps: usize) -> Result<i64> {
        if self.n_inputs == 0 || n_steps == 0 {
            return Ok(i64::MAX);
        }
        let packed = self.stimulus.pack(self.t, n_steps, self.n_inputs);
        let need = packed.len();
        if self.stim.len() < need {
            self.stim = self.stream.alloc_zeros(need)?;
        }
        self.stream.memcpy_htod(&packed, &mut self.stim)?;
        Ok(self.t as i64)
    }

    /// Launch one step (async, on the sim's stream).
    fn step_async(&mut self) -> Result<()> {
        let t = self.t;
        let n_i32 = self.n as i32;
        let d_i32 = self.ring_d as i32;
        let dt = self.cfg.dt_ms;
        let pkick = self.cfg.noise_rate_hz * dt / 1000.0;
        let kick = if self.cfg.noise_kick_mv != 0.0 {
            self.cfg.noise_kick_mv / dt
        } else {
            0.0
        };
        let decay_s = (-dt / self.cfg.synapse_tau_ms.max(1e-6)).exp();
        let decay_t = (-dt / self.cfg.readout_tau_ms.max(1e-6)).exp();
        let grid = (self.n as u32).div_ceil(BLOCK);
        let cfg_launch = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_inputs_i32 = self.n_inputs as i32;
        let n_params_i32 = self.params.n_params as i32;
        let n_state_i32 = self.params.n_state as i32;
        let nwatch_i32 = self.v_watch.len() as i32;
        let vrec_d_i32 = VREC_D as i32;
        let n_grids_i32 = self.grid_watches.len() as i32;
        let stim_t0 = self.stim_t0;
        let cur = (t % 2) as usize;
        let record = self.cfg.record_spikes && self.record_rows > 0;
        let row = (t as usize) % self.record_rows.max(1);

        let mut row_view = if record {
            Some(
                self.spk_bits
                    .slice_mut(row * self.words..(row + 1) * self.words),
            )
        } else {
            None
        };
        let [c0, c1] = self.cnt.each_mut();
        let (cnt_cur, cnt_next) = if cur == 0 { (c0, c1) } else { (c1, c0) };

        unsafe {
            // per-synapse arrivals: x[j] += w, clear the slot (skip when
            // no synapse carries its own tau)
            if self.n_xsyn > 0 {
                let c_i32 = self.n_xsyn as i32;
                let slot_i32 = (t as usize % self.ring_d) as i32;
                let grid_x = (self.n_xsyn as u32).div_ceil(BLOCK);
                let cfg_x = LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut bx = self.stream.launch_builder(&self.f_xsyn);
                bx.arg(&c_i32)
                    .arg(&slot_i32)
                    .arg(&mut self.xring)
                    .arg(&mut self.xsyn);
                bx.launch(cfg_x)?;
            }

            // neuron
            let mut b = self.stream.launch_builder(&self.f_neuron);
            b.arg(&n_i32)
                .arg(&t)
                .arg(&d_i32)
                .arg(&dt)
                .arg(&pkick)
                .arg(&kick)
                .arg(&mut self.x)
                .arg(&self.p)
                .arg(&self.model)
                .arg(&n_params_i32)
                .arg(&n_state_i32)
                .arg(&mut self.s)
                .arg(&mut self.trace)
                .arg(&mut self.ring)
                .arg(&self.ext)
                .arg(&self.stim)
                .arg(&stim_t0)
                .arg(&n_inputs_i32)
                .arg(&self.stim_ptr)
                .arg(&self.stim_input)
                .arg(&self.stim_w)
                .arg(&decay_s)
                .arg(&decay_t)
                .arg(&self.cfg.seed)
                .arg(&mut self.spk)
                .arg(&mut *cnt_cur);
            match &mut row_view {
                Some(v) => {
                    b.arg(v);
                }
                None => {
                    b.arg(&0u64); // NULL spk_row
                }
            }
            b.arg(&mut self.pn_total)
                .arg(&self.vwatch_dev)
                .arg(&nwatch_i32)
                .arg(&mut self.xrec_dev)
                .arg(&vrec_d_i32)
                .arg(&self.gmap)
                .arg(&n_grids_i32)
                .arg(&mut self.gls)
                .arg(&self.goff)
                .arg(&self.xt_ptr)
                .arg(&self.xt_idx)
                .arg(&mut self.xsyn)
                .arg(&self.xdecay);
            b.launch(cfg_launch)?;
        }

        // Clear the NEXT record row so stale bits from the previous wrap
        // never reach spike_row/spike_counts (the row is reused every
        // record_rows steps and neuron_kernel only OR-s into it). Ordered on
        // the stream, so it lands after this step's reads and before the
        // next step's writes.
        if record {
            let next_row = (t as usize + 1) % self.record_rows;
            let mut next_view = self
                .spk_bits
                .slice_mut(next_row * self.words..(next_row + 1) * self.words);
            self.stream.memset_zeros(&mut next_view)?;
        }

        unsafe {
            // prop: warp-per-spike with grid-stride; 4 warps per neuron covers
            // bursts, extra warps exit on the broadcast counter read
            let prop_grid = (self.n as u32).saturating_mul(4).div_ceil(BLOCK);
            let cfg_prop = LaunchConfig {
                grid_dim: (prop_grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let c_i32 = self.n_xsyn as i32;
            let mut b = self.stream.launch_builder(&self.f_prop);
            b.arg(&n_i32)
                .arg(&t)
                .arg(&d_i32)
                .arg(&self.spk)
                .arg(&*cnt_cur)
                .arg(cnt_next)
                .arg(&mut self.total)
                .arg(&self.src_ptr)
                .arg(&self.tgt)
                .arg(&self.w)
                .arg(&self.delay)
                .arg(&mut self.ring)
                .arg(&self.xsyn_map)
                .arg(&c_i32)
                .arg(&mut self.xring);
            b.launch(cfg_prop)?;
        }
        self.t += 1;
        if !self.v_watch.is_empty() {
            self.vrec_valid += 1;
        }
        Ok(())
    }

    /// Upload the watched-id list + allocate the v record ring (lazy, only
    /// after `set_v_watch` changed the list).
    fn upload_vwatch(&mut self) -> Result<()> {
        let w = self.v_watch.len().max(1);
        self.vwatch_dev = self.stream.clone_htod(&{
            let mut v = self.v_watch.clone();
            if v.is_empty() {
                v.push(0);
            }
            v
        })?;
        self.xrec_dev = self.stream.alloc_zeros(VREC_D * w * self.params.n_state)?;
        self.vrec_dirty = false;
        self.vrec_valid = 0;
        Ok(())
    }

    /// Upload the concat maps + offsets for grid watches (lazy, only after
    /// `set_grid_watches` changed the member lists).
    fn upload_grid(&mut self) -> Result<()> {
        let mut map = vec![-1i32; self.grid_watches.len() * self.n];
        let mut off: Vec<u32> = Vec::with_capacity(self.grid_watches.len());
        let mut m_max = 0usize;
        let mut acc = 0u32;
        for (g, members) in self.grid_watches.iter().enumerate() {
            off.push(acc);
            acc += members.len() as u32;
            for (k, &m) in members.iter().enumerate() {
                if let Some(slot) = map.get_mut(g * self.n + m as usize) {
                    *slot = k as i32;
                }
            }
            m_max = m_max.max(members.len());
        }
        let total: usize = self.grid_watches.iter().map(|m| m.len()).sum();
        self.gmap = self.stream.clone_htod(&{
            let mut v = map;
            if v.is_empty() {
                v.push(-1);
            }
            v
        })?;
        self.gls = self.stream.clone_htod(&vec![u32::MAX; total.max(1)])?;
        self.goff = self.stream.clone_htod(&{
            let mut v = off;
            if v.is_empty() {
                v.push(0);
            }
            v
        })?;
        if self.gtr.len() < m_max.max(1) {
            self.gtr = self.stream.alloc_zeros(m_max.max(1))?;
        }
        self.grid_dirty = false;
        Ok(())
    }

    /// Set the grid-group member lists (texel order) whose spike traces the
    /// plane view wants. No-op when unchanged; safe to call every publish.
    pub fn set_grid_watches(&mut self, groups: &[Vec<u32>]) {
        if self.grid_watches == groups {
            return;
        }
        self.grid_watches = groups.to_vec();
        self.grid_dirty = true;
    }

    /// Materialized spike traces for each watched grid group: one f32 per
    /// texel (member order), `exp(-(t - last_spike) * dt / tau_ms)` —
    /// computed on device by `grid_trace_kernel`, read back per group.
    pub fn grid_traces(&mut self, tau_ms: f32) -> Result<Vec<Vec<f32>>> {
        if self.grid_dirty {
            self.upload_grid()?;
        }
        self.stream.synchronize()?;
        let k = self.cfg.dt_ms / tau_ms.max(1e-6);
        let mut out = Vec::with_capacity(self.grid_watches.len());
        let mut off = 0usize;
        for members in &self.grid_watches {
            let m = members.len();
            let mut host = vec![0.0f32; m];
            if m > 0 {
                let m_i32 = m as i32;
                let ls_view = self.gls.slice(off..off + m);
                let mut tr_view = self.gtr.slice_mut(0..m);
                let cfg = LaunchConfig {
                    grid_dim: (m.div_ceil(BLOCK as usize) as u32, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    let mut b = self.stream.launch_builder(&self.f_gtrace);
                    b.arg(&ls_view)
                        .arg(&m_i32)
                        .arg(&self.t)
                        .arg(&k)
                        .arg(&mut tr_view);
                    b.launch(cfg)?;
                }
                self.stream.memcpy_dtoh(&tr_view, &mut host)?;
            }
            out.push(host);
            off += m;
        }
        Ok(out)
    }

    /// Set the watched-neuron list; no-op when unchanged (safe to call
    /// every publish — history survives unchanged watch lists).
    pub fn set_v_watch(&mut self, ids: &[u32]) {
        if self.v_watch == ids {
            return;
        }
        self.v_watch = ids.to_vec();
        self.vrec_dirty = true;
        self.vrec_valid = 0;
    }

    /// Read the per-step record ring back and split every state slot —
    /// `[slot][watch][samples oldest-first]`. One readback per publish;
    /// resets the shared `vrec_valid` counter.
    pub fn drain_xrec(&mut self) -> Result<Vec<Vec<Vec<f32>>>> {
        let w = self.v_watch.len();
        let nx = self.params.n_state;
        if w == 0 || self.vrec_valid == 0 || self.vrec_dirty {
            return Ok(vec![vec![Vec::new(); w]; nx]);
        }
        self.stream.synchronize()?;
        let k = (self.vrec_valid as usize).min(VREC_D);
        let r0 = (self.t as usize).wrapping_sub(k) % VREC_D;
        let rowlen = w * nx;
        let mut host = vec![0.0f32; k * rowlen];
        if r0 + k <= VREC_D {
            let view = self.xrec_dev.slice(r0 * rowlen..(r0 + k) * rowlen);
            self.stream.memcpy_dtoh(&view, &mut host)?;
        } else {
            let first = VREC_D - r0;
            let view = self.xrec_dev.slice(r0 * rowlen..VREC_D * rowlen);
            self.stream
                .memcpy_dtoh(&view, &mut host[..first * rowlen])?;
            let view = self.xrec_dev.slice(0..(k - first) * rowlen);
            self.stream
                .memcpy_dtoh(&view, &mut host[first * rowlen..])?;
        }
        self.vrec_valid = 0;
        let mut out = vec![vec![Vec::with_capacity(k); w]; nx];
        for r in 0..k {
            for c in 0..w {
                for s in 0..nx {
                    out[s][c].push(host[r * rowlen + c * nx + s]);
                }
            }
        }
        Ok(out)
    }

    /// Drain per-step samples for state `slot`, one Vec per watched
    /// neuron (oldest first). At most `VREC_D` samples survive — and a
    /// drain empties the shared ring, so callers wanting several slots
    /// use `drain_xrec` once.
    pub fn drain_state(&mut self, slot: usize) -> Result<Vec<Vec<f32>>> {
        Ok(self
            .drain_xrec()?
            .into_iter()
            .nth(slot)
            .unwrap_or_else(|| vec![Vec::new(); self.v_watch.len()]))
    }

    pub fn run_steps(&mut self, n_steps: usize) -> Result<()> {
        if self.n == 0 {
            // Empty net: no kernels to launch (a 0-block grid is a driver
            // error anyway); just advance the clock like the CPU backend.
            self.t += n_steps as u64;
            return Ok(());
        }
        if self.vrec_dirty {
            self.upload_vwatch()?;
        }
        if self.grid_dirty {
            self.upload_grid()?;
        }
        // upload the stimulus window for this chunk, then fire-and-forget
        self.stim_t0 = self.upload_stim(n_steps)?;
        for _ in 0..n_steps {
            self.step_async()?;
        }
        // leave the stream running; reads sync as needed
        Ok(())
    }

    pub fn run_ms(&mut self, ms: f32) -> Result<()> {
        let steps = (ms / self.cfg.dt_ms).round() as usize;
        self.run_steps(steps)
    }

    pub fn synchronize(&self) {
        self.stream.synchronize().expect("cuda sync");
    }

    pub fn reset_state(&mut self) {
        self.synchronize();
        if self.n == 0 {
            self.t = 0;
            return;
        }
        self.stream
            .memcpy_htod(&self.x0.clone(), &mut self.x)
            .unwrap();
        self.stream.memset_zeros(&mut self.s).unwrap();
        self.stream.memset_zeros(&mut self.trace).unwrap();
        self.stream.memset_zeros(&mut self.ring).unwrap();
        if self.n_xsyn > 0 {
            self.stream.memset_zeros(&mut self.xsyn).unwrap();
            self.stream.memset_zeros(&mut self.xring).unwrap();
        }
        self.stream.memset_zeros(&mut self.cnt[0]).unwrap();
        self.stream.memset_zeros(&mut self.cnt[1]).unwrap();
        self.stream.memset_zeros(&mut self.total).unwrap();
        self.stream.memset_zeros(&mut self.spk_bits).unwrap();
        self.stream.memset_zeros(&mut self.pn_total).unwrap();
        self.vrec_valid = 0;
        if !self.grid_watches.is_empty() && !self.grid_dirty {
            let total: usize = self.grid_watches.iter().map(|m| m.len()).sum();
            self.stream
                .memcpy_htod(&vec![u32::MAX; total], &mut self.gls)
                .unwrap();
        }
        self.t = 0;
    }

    pub fn time_ms(&self) -> f64 {
        self.t as f64 * self.cfg.dt_ms as f64
    }

    pub fn trace(&self) -> Vec<f32> {
        if self.n == 0 {
            return Vec::new();
        }
        self.stream.clone_dtoh(&self.trace).unwrap()
    }

    /// Model state slot column, one value per neuron — one `x` readback
    /// for any slot, so `state(0)` (the membrane potential) costs the
    /// same as any other slot.
    pub fn state(&self, slot: usize) -> Vec<f32> {
        let nx = self.params.n_state;
        if slot >= nx || self.n == 0 {
            return Vec::new();
        }
        let x = self.stream.clone_dtoh(&self.x).unwrap();
        (0..self.n).map(|i| x[i * nx + slot]).collect()
    }

    pub fn spikes_total(&self) -> u64 {
        self.stream.clone_dtoh(&self.total).unwrap()[0]
    }

    pub fn mean_rate_hz(&self) -> f64 {
        if self.t == 0 {
            return 0.0;
        }
        self.spikes_total() as f64
            / self.n.max(1) as f64
            / (self.t as f64 * self.cfg.dt_ms as f64 / 1000.0)
    }

    pub fn spike_row(&self, back: usize) -> Option<Vec<u64>> {
        if !self.cfg.record_spikes
            || self.t == 0
            || self.record_rows == 0
            || self.words == 0
            || back >= self.t as usize
            || back >= self.record_rows
        {
            return None;
        }
        let t = self.t as usize - 1 - back;
        let row = t % self.record_rows;
        let view = self
            .spk_bits
            .slice(row * self.words..(row + 1) * self.words);
        Some(self.stream.clone_dtoh(&view).unwrap())
    }

    pub fn last_spikes(&self) -> Vec<u32> {
        let Some(bits) = self.spike_row(0) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (wi, w) in bits.iter().enumerate() {
            let mut w = *w;
            while w != 0 {
                let b = w.trailing_zeros() as usize;
                w &= w - 1;
                let i = wi * 64 + b;
                if i < self.n {
                    out.push(i as u32);
                }
            }
        }
        out
    }

    /// Per-neuron spike counts over the last `rows` steps.
    ///
    /// `rows >= t`: exact lifetime totals (`pn_total`). `rows <= record ring
    /// depth`: exact sliding-window count. Anything in between cannot be
    /// answered (the ring already wrapped) and returns an error rather than
    /// silently capping.
    pub fn spike_counts(&self, rows: usize) -> Result<Vec<u32>, String> {
        let mut counts = vec![0u32; self.n];
        if !self.cfg.record_spikes || self.t == 0 || self.record_rows == 0 {
            return Ok(counts);
        }
        if rows >= self.t as usize {
            let tot = self.stream.clone_dtoh(&self.pn_total).unwrap();
            for (c, &t) in counts.iter_mut().zip(tot.iter()) {
                *c = t.min(u32::MAX as u64) as u32;
            }
            return Ok(counts);
        }
        if rows > self.record_rows {
            return Err(format!(
                "spike_counts window {rows} exceeds the recorded {} steps",
                self.record_rows
            ));
        }
        let all = self.stream.clone_dtoh(&self.spk_bits).unwrap();
        for k in 0..rows {
            let t = self.t as usize - 1 - k;
            let row = t % self.record_rows;
            for (wi, &w) in all[row * self.words..(row + 1) * self.words]
                .iter()
                .enumerate()
            {
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
        if let Some(h) = self.ext_host.get_mut(neuron) {
            *h = amps;
            self.stream
                .memcpy_htod(&self.ext_host, &mut self.ext)
                .unwrap();
        }
    }

    /// Set the whole external-current vector at once (cheap path).
    pub fn set_external_all(&mut self, ext: &[f32]) {
        let n = self.ext_host.len().min(ext.len());
        self.ext_host[..n].copy_from_slice(&ext[..n]);
        self.stream
            .memcpy_htod(&self.ext_host, &mut self.ext)
            .unwrap();
    }

    /// Host copy of the external currents — for rebuild snapshots
    /// (`DriveState::capture`), not hot paths.
    pub fn external(&self) -> Vec<f32> {
        self.ext_host.clone()
    }

    /// Change the background noise rate on the live sim (kernel args read cfg).
    pub fn set_noise(&mut self, rate_hz: f32, kick_mv: f32) {
        self.cfg.noise_rate_hz = rate_hz;
        self.cfg.noise_kick_mv = kick_mv;
    }

    /// Approximate device memory held by this sim, for reporting.
    pub fn dev_mem_bytes(&self) -> usize {
        (self.x.len()
            + self.p.len()
            + self.model.len()
            + self.ring.len()
            + self.stim.len()
            + self.s.len()
            + self.trace.len()
            + self.ext.len())
            * 4
            + self.tgt.len() * 4
            + self.w.len() * 4
            + self.src_ptr.len() * 4
            + self.delay.len() * 2
            + self.spk_bits.len() * 8
            + self.spk.len() * 4
    }
}

impl crate::Backend for CudaSim {
    fn kind(&self) -> crate::BackendKind {
        crate::BackendKind::Cuda
    }
    fn n(&self) -> usize {
        self.n
    }
    fn step_count(&self) -> u64 {
        self.t
    }
    fn time_ms(&self) -> f64 {
        self.time_ms()
    }
    fn run_steps(&mut self, n: usize) -> anyhow::Result<()> {
        self.run_steps(n)
    }
    fn run_ms(&mut self, ms: f32) -> anyhow::Result<()> {
        self.run_ms(ms)
    }
    fn reset_state(&mut self) {
        self.reset_state()
    }
    fn synchronize(&self) {
        self.synchronize()
    }
    fn dev_mem_bytes(&self) -> usize {
        self.dev_mem_bytes()
    }
    fn trace(&self) -> Vec<f32> {
        self.trace()
    }
    fn state(&self, slot: usize) -> Vec<f32> {
        CudaSim::state(self, slot)
    }
    fn last_spikes(&self) -> Vec<u32> {
        self.last_spikes()
    }
    fn spike_counts(&self, rows: usize) -> Result<Vec<u32>, String> {
        self.spike_counts(rows)
    }
    fn spike_row(&self, back: usize) -> Option<Vec<u64>> {
        self.spike_row(back)
    }
    fn spikes_total(&self) -> u64 {
        self.spikes_total()
    }
    fn mean_rate_hz(&self) -> f64 {
        self.mean_rate_hz()
    }
    fn set_input_constant(&mut self, input: usize, value: f32) {
        self.set_input_constant(input, value)
    }
    fn input_values(&self) -> Vec<f32> {
        self.stimulus.constants.clone()
    }
    fn stimulus(&self) -> crate::stim::Stimulus {
        self.stimulus.clone()
    }
    fn external(&self) -> Vec<f32> {
        self.external()
    }
    fn play_input(&mut self, input: usize, series: Vec<f32>) {
        self.play_input(input, series)
    }
    fn set_v_watch(&mut self, ids: &[u32]) {
        self.set_v_watch(ids)
    }
    /// One ring readback, all slots — the counter resets inside
    /// `drain_xrec`.
    fn drain_watch(&mut self) -> anyhow::Result<Vec<Vec<Vec<f32>>>> {
        self.drain_xrec()
    }
    fn set_grid_watches(&mut self, groups: &[Vec<u32>]) {
        self.set_grid_watches(groups)
    }
    fn grid_traces(&mut self, tau_ms: f32) -> anyhow::Result<Vec<Vec<f32>>> {
        self.grid_traces(tau_ms)
    }
    fn set_external(&mut self, neuron: usize, amps: f32) {
        self.set_external(neuron, amps)
    }
    fn set_external_all(&mut self, ext: &[f32]) {
        self.set_external_all(ext)
    }
    fn set_noise(&mut self, rate_hz: f32, kick_mv: f32) {
        self.set_noise(rate_hz, kick_mv)
    }
}
