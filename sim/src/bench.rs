//! Benchmark network + timing, mirroring `izh_bench.py`: 80% excitatory,
//! 20% inhibitory, fixed out-degree K, random targets, delays 1..20 ms.

use net_format::{Network, NeuronKind, Rng};
use std::time::Instant;

use crate::{BackendKind, Sim, SimConfig};

/// Polychronization-style random network like the reference benchmark.
pub fn bench_net(n: usize, k: usize, seed: u64, w_scale: f32, w_inh: f32, dt: f32) -> Network {
    let mut rng = Rng::new(seed);
    let mut net = Network::new();
    net.meta.conduction_speed = 1000.0; // we'll set delays directly instead
    for i in 0..n {
        let id = net.add_neuron(
            [
                rng.uniform(-500.0, 500.0) as f32,
                rng.uniform(-500.0, 500.0) as f32,
                rng.uniform(-500.0, 500.0) as f32,
            ],
            if i < n * 4 / 5 {
                NeuronKind::Excitatory
            } else {
                NeuronKind::Inhibitory
            },
        );
        if i >= n * 4 / 5 {
            // fast-spiking inhibitory, as in the bench
            net.neuron_mut(id).model = net_format::NeuronModel::Izhikevich {
                a: 0.1,
                b: 0.2,
                c: -65.0,
                d: 2.0,
            };
        }
    }
    let _ = dt;
    for src in 0..n as u32 {
        let exc = (src as usize) < n * 4 / 5;
        // synaptic weight is a current amplitude (original scale ~6/-5)
        let w = if exc { 6.0 * w_scale } else { -w_inh };
        for _ in 0..k {
            let tgt = rng.below(n) as u32;
            let delay_ms = if exc {
                rng.uniform(1.0, 21.0) as f32
            } else {
                1.0
            };
            net.synapses.push(net_format::Synapse {
                src,
                tgt,
                w,
                delay_ms,
            });
        }
    }
    net
}

pub struct BenchResult {
    pub wall_s: f64,
    pub realtime_x: f64,
    pub rate_hz: f64,
    pub spikes: u64,
    pub gpu_mem_mb: usize,
}

pub fn run_bench(
    net: &Network,
    kind: BackendKind,
    mut cfg: SimConfig,
    sim_ms: f32,
    warm_ms: f32,
) -> BenchResult {
    cfg.record_spikes = false;
    let mut sim = Sim::new(net, cfg, kind).expect("sim init");
    let mut gpu_mem_mb = 0;
    #[cfg(feature = "cuda")]
    if let Sim::Cuda(ref s) = sim {
        gpu_mem_mb = s.dev_mem_bytes() / (1 << 20);
    }
    // warm up (NVRTC compile + first launches + settle)
    sim.run_ms(warm_ms);
    #[cfg(feature = "cuda")]
    if let Sim::Cuda(ref s) = sim {
        s.synchronize();
    }
    let t0 = Instant::now();
    sim.run_ms(sim_ms);
    #[cfg(feature = "cuda")]
    if let Sim::Cuda(ref s) = sim {
        s.synchronize();
    }
    let wall = t0.elapsed().as_secs_f64();
    let spikes = sim.spikes_total();
    let sim_s = (sim_ms / 1000.0) as f64;
    BenchResult {
        wall_s: wall,
        realtime_x: sim_s / wall,
        rate_hz: spikes as f64 / net.neurons.len().max(1) as f64 / sim_s,
        spikes,
        gpu_mem_mb,
    }
}
