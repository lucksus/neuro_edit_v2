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

pub struct CudaSim {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    f_neuron: CudaFunction,
    f_prop: CudaFunction,

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
        let csr = net.synapse_csr(cfg.dt_ms);
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

        let sim = CudaSim {
            v: stream.clone_htod(&vec![params.init_v; n])?,
            u: stream.clone_htod(&params.init_u)?,
            s: stream.alloc_zeros(n)?,
            trace: stream.alloc_zeros(n)?,
            ext: stream.alloc_zeros(n)?,
            ring: stream.alloc_zeros(d * n)?,
            a: stream.clone_htod(&params.a)?,
            b: stream.clone_htod(&params.b)?,
            c: stream.clone_htod(&params.c)?,
            d: stream.clone_htod(&params.d)?,
            src_ptr: stream.clone_htod(&csr.src_ptr)?,
            tgt: stream.clone_htod(&csr.tgt)?,
            w: stream.clone_htod(&csr.w)?,
            delay: stream.clone_htod(&csr.delay_steps)?,
            stim_ptr: stream.clone_htod(&stim_ptr_host)?,
            stim_input: stream.clone_htod(&stim_input_host)?,
            stim_w: stream.clone_htod(&stim_w_host)?,
            stim: stream.alloc_zeros(1)?,
            stim_t0: i64::MAX,
            n_inputs: net.inputs.len(),
            spk: stream.alloc_zeros(n.max(1))?,
            cnt: [stream.alloc_zeros(1)?, stream.alloc_zeros(1)?],
            total: stream.alloc_zeros(1)?,
            spk_bits: stream.alloc_zeros((record_rows * words).max(1))?,
            pn_total: stream.alloc_zeros(n.max(1))?,
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
        let decay_s = 1.0 - dt / self.cfg.synapse_tau_ms.max(1e-6);
        let decay_t = 1.0 - dt / self.cfg.readout_tau_ms.max(1e-6);
        let grid = (self.n as u32).div_ceil(BLOCK);
        let cfg_launch = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let model: i32 = 0;
        let n_inputs_i32 = self.n_inputs as i32;
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
            b.arg(&mut self.pn_total);
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
                .arg(&mut self.ring);
            b.launch(cfg_prop)?;
        }
        self.t += 1;
        Ok(())
    }

    pub fn run_steps(&mut self, n_steps: usize) {
        // upload the stimulus window for this chunk, then fire-and-forget
        self.stim_t0 = self.upload_stim(n_steps).expect("stim upload");
        for _ in 0..n_steps {
            self.step_async().expect("kernel launch");
        }
        // leave the stream running; reads sync as needed
    }

    pub fn run_ms(&mut self, ms: f32) {
        let steps = (ms / self.cfg.dt_ms).round() as usize;
        self.run_steps(steps);
    }

    pub fn synchronize(&self) {
        self.stream.synchronize().expect("cuda sync");
    }

    pub fn reset_state(&mut self) {
        self.synchronize();
        self.stream
            .memcpy_htod(&vec![self.init_v; self.n], &mut self.v)
            .unwrap();
        self.stream
            .memcpy_htod(&self.init_u.clone(), &mut self.u)
            .unwrap();
        self.stream.memset_zeros(&mut self.s).unwrap();
        self.stream.memset_zeros(&mut self.trace).unwrap();
        self.stream.memset_zeros(&mut self.ring).unwrap();
        self.stream.memset_zeros(&mut self.cnt[0]).unwrap();
        self.stream.memset_zeros(&mut self.cnt[1]).unwrap();
        self.stream.memset_zeros(&mut self.total).unwrap();
        self.stream.memset_zeros(&mut self.spk_bits).unwrap();
        self.stream.memset_zeros(&mut self.pn_total).unwrap();
        self.t = 0;
    }

    pub fn time_ms(&self) -> f64 {
        self.t as f64 * self.cfg.dt_ms as f64
    }

    pub fn trace(&self) -> Vec<f32> {
        self.stream.clone_dtoh(&self.trace).unwrap()
    }

    pub fn v(&self) -> Vec<f32> {
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
        self.stimulus.series[input] = series;
        self.stimulus.play_start = self.t;
    }

    pub fn set_external(&mut self, neuron: usize, amps: f32) {
        let mut host = self.stream.clone_dtoh(&self.ext).unwrap();
        host[neuron] = amps;
        self.stream.memcpy_htod(&host, &mut self.ext).unwrap();
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
