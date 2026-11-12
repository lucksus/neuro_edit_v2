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

use crate::kernels::KERNELS_SRC;
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

    // device state
    v: CudaSlice<f32>,
    u: CudaSlice<f32>,
    s: CudaSlice<f32>,
    trace: CudaSlice<f32>,
    ext: CudaSlice<f32>,
    /// Delay ring: f32 elements, or i32 fixed-point (scale 4096) when
    /// `cfg.int_weights` — same 4-byte words either way.
    ring: CudaSlice<u32>,
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    c: CudaSlice<f32>,
    d: CudaSlice<f32>,
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
    // per-step v recording for watched neurons (editor v(t) plot)
    /// host copy of the watched ids (compared on every set_v_watch call)
    pub v_watch: Vec<u32>,
    /// device copy of `v_watch` (W entries)
    vwatch_dev: CudaSlice<u32>,
    /// device ring VREC_D x W of per-step v for watched neurons
    vrec_dev: CudaSlice<f32>,
    /// device ring VREC_D x W of per-step u (recovery) — the phase plot
    urec_dev: CudaSlice<f32>,
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
    init_v: f32,
    init_u: Vec<f32>,
}

impl CudaSim {
    pub fn new(net: &Network, cfg: SimConfig) -> Result<Self> {
        let n = net.neurons.len();
        let csr = net.synapse_csr(cfg.dt_ms).map_err(anyhow::Error::msg)?;
        let d = ring_depth(&csr, cfg.dt_ms);
        let params = NeuronParams::from_network(net);
        let stim_map = crate::stim::StimMap::from_network(net);

        let ctx = CudaContext::new(0).context("no CUDA device / driver")?;
        let stream = ctx.default_stream();

        // NVRTC compile for this device's arch. int_weights selects the
        // fixed-point delay ring (bitwise-reproducible accumulation).
        let src = if cfg.int_weights {
            format!("#define INT_RING 1\n{KERNELS_SRC}")
        } else {
            KERNELS_SRC.to_string()
        };
        let ptx = nvrtc::compile_ptx(&src).context("NVRTC compile")?;
        let module: Arc<CudaModule> = ctx.load_module(ptx).context("load_module")?;
        let f_neuron = module.load_function("neuron_kernel")?;
        let f_prop = module.load_function("prop_kernel")?;
        let f_gtrace = module.load_function("grid_trace_kernel")?;
        let f_xsyn = module.load_function("xsyn_arrive_kernel")?;

        let stim_ptr_host: Vec<u32> = stim_map.neuron_ptr.clone();
        let stim_input_host = stim_map.input_idx.clone();
        let stim_w_host = stim_map.weight.clone();

        let init_v = params.init_v;
        let init_u = params.init_u.clone();
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
            v: stream.clone_htod(&vec![params.init_v; na])?,
            u: stream.clone_htod(&pad(params.init_u.clone()))?,
            s: stream.alloc_zeros(na)?,
            trace: stream.alloc_zeros(na)?,
            ext: stream.alloc_zeros(na)?,
            ring: stream.alloc_zeros((d * n).max(1))?,
            a: stream.clone_htod(&pad(params.a.clone()))?,
            b: stream.clone_htod(&pad(params.b.clone()))?,
            c: stream.clone_htod(&pad(params.c.clone()))?,
            d: stream.clone_htod(&pad(params.d.clone()))?,
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
            vrec_dev: stream.alloc_zeros(1)?,
            urec_dev: stream.alloc_zeros(1)?,
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
            init_v,
            init_u,
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
        let model: i32 = 0;
        let n_inputs_i32 = self.n_inputs as i32;
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
                .arg(&mut self.v)
                .arg(&mut self.u)
                .arg(&mut self.s)
                .arg(&mut self.trace)
                .arg(&self.a)
                .arg(&self.b)
                .arg(&self.c)
                .arg(&self.d)
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
                .arg(&model)
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
                .arg(&mut self.vrec_dev)
                .arg(&mut self.urec_dev)
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
        self.vrec_dev = self.stream.alloc_zeros(VREC_D * w)?;
        self.urec_dev = self.stream.alloc_zeros(VREC_D * w)?;
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

    /// Shared body for `drain_v`/`drain_u` — the two rings have the same
    /// rows and cadence and share `vrec_valid` bookkeeping; `clear` marks
    /// which drain owns the counter reset.
    fn drain_rec(&mut self, dev: &CudaSlice<f32>, clear: bool) -> Result<Vec<Vec<f32>>> {
        let w = self.v_watch.len();
        if w == 0 || self.vrec_valid == 0 || self.vrec_dirty {
            return Ok(vec![Vec::new(); w]);
        }
        self.stream.synchronize()?;
        let k = (self.vrec_valid as usize).min(VREC_D);
        let r0 = (self.t as usize).wrapping_sub(k) % VREC_D;
        let mut host = vec![0.0f32; k * w];
        if r0 + k <= VREC_D {
            let view = dev.slice(r0 * w..(r0 + k) * w);
            self.stream.memcpy_dtoh(&view, &mut host)?;
        } else {
            let first = VREC_D - r0;
            let view = dev.slice(r0 * w..VREC_D * w);
            self.stream.memcpy_dtoh(&view, &mut host[..first * w])?;
            let view = dev.slice(0..(k - first) * w);
            self.stream.memcpy_dtoh(&view, &mut host[first * w..])?;
        }
        if clear {
            self.vrec_valid = 0;
        }
        let mut out = vec![Vec::with_capacity(k); w];
        for r in 0..k {
            for (c, o) in out.iter_mut().enumerate() {
                o.push(host[r * w + c]);
            }
        }
        Ok(out)
    }

    /// Drain per-step v samples recorded since the last drain, one Vec per
    /// watched neuron (oldest first). At most `VREC_D` samples survive.
    pub fn drain_v(&mut self) -> Result<Vec<Vec<f32>>> {
        let dev = self.vrec_dev.clone();
        self.drain_rec(&dev, true)
    }

    /// Drain per-step u samples — same rows/cadence as `drain_v` — for
    /// the editor's v–u phase plot. NOTE: shares `vrec_valid`'s
    /// bookkeeping (both rings record in the same loop), so this must
    /// be called BEFORE `drain_v` at each publish or it reports empty
    /// after the counter resets. Callers: `Sim::drain_vu`.
    pub fn drain_u(&mut self) -> Result<Vec<Vec<f32>>> {
        let dev = self.urec_dev.clone();
        self.drain_rec(&dev, false)
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
            .memcpy_htod(&vec![self.init_v; self.n], &mut self.v)
            .unwrap();
        self.stream
            .memcpy_htod(&self.init_u.clone(), &mut self.u)
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

    pub fn v(&self) -> Vec<f32> {
        if self.n == 0 {
            return Vec::new();
        }
        self.stream.clone_dtoh(&self.v).unwrap()
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
        let mut host = self.stream.clone_dtoh(&self.ext).unwrap();
        if let Some(h) = host.get_mut(neuron) {
            *h = amps;
            self.stream.memcpy_htod(&host, &mut self.ext).unwrap();
        }
    }

    /// Set the whole external-current vector at once (cheap path).
    pub fn set_external_all(&mut self, ext: &[f32]) {
        self.stream
            .memcpy_htod(&ext.to_vec(), &mut self.ext)
            .unwrap();
    }

    /// Change the background noise rate on the live sim (kernel args read cfg).
    pub fn set_noise(&mut self, rate_hz: f32, kick_mv: f32) {
        self.cfg.noise_rate_hz = rate_hz;
        self.cfg.noise_kick_mv = kick_mv;
    }

    /// Approximate device memory held by this sim, for reporting.
    pub fn dev_mem_bytes(&self) -> usize {
        (self.v.len() * 6
            + self.a.len() * 4
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
