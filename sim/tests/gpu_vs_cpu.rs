//! CUDA vs CPU verification: same network, same seed, same steps.
//! Exact bit-equality is NOT expected: the GPU sums ring contributions with
//! float atomics in arbitrary order. We require identical spike *counts* on a
//! deterministic network without noise, and close values on a noisy one.

#![cfg(feature = "cuda")]

use net_format::{builders, Network, Rng};
use sim::{BackendKind, Sim, SimConfig};

fn lsm_like_net(n_side: usize, seed: u64) -> Network {
    let mut net = Network::new();
    let mut rng = Rng::new(seed);
    builders::create_3d_grid(&mut net, "liquid", n_side, n_side, n_side, 60.0, [0.0; 3]);
    builders::set_exc_inh_exact(&mut net, "liquid", n_side * n_side * n_side / 5, &mut rng);
    builders::create_connections(&mut net, "liquid", 0.4, 100.0, 5.0, &mut rng);
    net
}

fn gpu_available() -> bool {
    cudarc::driver::CudaContext::new(0).is_ok()
}

#[test]
fn gpu_matches_cpu_spike_counts_quiet() {
    if !gpu_available() {
        eprintln!("no GPU, skipping");
        return;
    }
    let net = lsm_like_net(8, 42);
    let cfg = SimConfig {
        record_spikes: true,
        ..Default::default()
    };
    let mut cpu = Sim::new(&net, cfg.clone(), BackendKind::Cpu).unwrap();
    let mut gpu = Sim::new(&net, cfg, BackendKind::Cuda).unwrap();
    // drive a few neurons so there is activity without noise
    for i in 0..8 {
        cpu.set_external(i, 12.0);
        gpu.set_external(i, 12.0);
    }
    cpu.run_ms(50.0);
    gpu.run_ms(50.0);
    let cpu_total = cpu.spikes_total();
    let gpu_total = gpu.spikes_total();
    eprintln!("cpu spikes={cpu_total} gpu spikes={gpu_total}");
    assert!(cpu_total > 0);
    assert_eq!(cpu_total, gpu_total, "spike counts differ");
}

#[test]
fn gpu_cpu_close_with_noise() {
    if !gpu_available() {
        eprintln!("no GPU, skipping");
        return;
    }
    let net = lsm_like_net(8, 7);
    let cfg = SimConfig {
        noise_rate_hz: 1000.0,
        noise_kick_mv: 20.0,
        seed: 99,
        record_spikes: true,
        ..Default::default()
    };
    let mut cpu = Sim::new(&net, cfg.clone(), BackendKind::Cpu).unwrap();
    let mut gpu = Sim::new(&net, cfg, BackendKind::Cuda).unwrap();
    cpu.run_ms(100.0);
    gpu.run_ms(100.0);
    let a = cpu.spikes_total() as f64;
    let g = gpu.spikes_total() as f64;
    eprintln!("cpu={a} gpu={g}");
    // float-atomic order can drift trajectories; allow loose agreement on
    // counts and check traces/v in range
    assert!(a > 0.0);
    let rel = (a - g).abs() / a.max(1.0);
    assert!(rel < 0.25, "spike counts diverge too much: cpu={a} gpu={g}");
    let tr = gpu.trace();
    assert!(tr.iter().all(|x| (0.0..=1.0).contains(x)));
    let v = gpu.v();
    assert!(v.iter().all(|x| *x < 100.0 && *x > -200.0));
}

/// Integer fixed-point ring: two identical seeded runs must produce bitwise
/// identical state — int atomicAdd is order-independent, float is not.
#[test]
fn gpu_int_weights_bitwise_deterministic() {
    if !gpu_available() {
        eprintln!("no GPU, skipping");
        return;
    }
    let net = lsm_like_net(8, 5);
    let cfg = SimConfig {
        noise_rate_hz: 2000.0,
        noise_kick_mv: 20.0,
        seed: 7,
        int_weights: true,
        record_spikes: true,
        ..Default::default()
    };
    let mut a = Sim::new(&net, cfg.clone(), BackendKind::Cuda).unwrap();
    let mut b = Sim::new(&net, cfg, BackendKind::Cuda).unwrap();
    a.run_ms(80.0);
    b.run_ms(80.0);
    assert_eq!(a.spikes_total(), b.spikes_total());
    // v(u8-quantized or not) and the recorded spike bitmap must match exactly
    assert_eq!(a.v(), b.v(), "v differs between identical int-ring runs");
    assert_eq!(a.spike_counts(64), b.spike_counts(64));
    // sanity: it actually ran and spiked
    assert!(a.spikes_total() > 0);
}

/// The int-ring result should stay close to the float ring (same regime,
/// small quantization drift allowed).
#[test]
fn gpu_int_weights_close_to_float() {
    if !gpu_available() {
        eprintln!("no GPU, skipping");
        return;
    }
    let net = lsm_like_net(8, 11);
    let base = SimConfig {
        noise_rate_hz: 2000.0,
        noise_kick_mv: 20.0,
        seed: 3,
        ..Default::default()
    };
    let mut f = Sim::new(&net, base.clone(), BackendKind::Cuda).unwrap();
    let mut i = Sim::new(
        &net,
        SimConfig {
            int_weights: true,
            ..base
        },
        BackendKind::Cuda,
    )
    .unwrap();
    f.run_ms(80.0);
    i.run_ms(80.0);
    let (tf, ti) = (f.spikes_total() as f64, i.spikes_total() as f64);
    eprintln!("float={tf} int={ti}");
    assert!(tf > 0.0 && ti > 0.0);
    assert!(
        (tf - ti).abs() / tf.max(1.0) < 0.3,
        "int ring diverged from float: float={tf} int={ti}"
    );
}
