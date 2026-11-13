//! Stage 21's second real model: AdEx, registered in `model::MODELS`.
//! A net mixing izh and adex neurons (via `net.upsert_type`/`set_type`
//! — the real user path, not hand-built params) must produce identical
//! spike counts on CPU and CUDA, and every adex neuron must fire AND
//! reset — the Stage 20 LIF lesson was that a test which never drives
//! the new model's neurons can pass with a broken GPU reset.

use net_format::{Network, NeuronKind, NeuronType};
use sim::model::{self, adex};
use sim::{BackendKind, Sim, SimConfig};

/// 8 neurons in a row; `adex_ids` get the "adex" type (model "adex"),
/// the rest keep the stock izh type — through the type table, so the
/// sim's params come from `NeuronParams::from_network` like a real net.
fn mixed_net(adex_ids: &[u32]) -> Network {
    let mut net = Network::new();
    for i in 0..8 {
        net.add_neuron([i as f32 * 50.0, 0.0, 0.0], NeuronKind::Excitatory);
    }
    let ty = net.upsert_type(NeuronType {
        name: "adex".into(),
        model: "adex".into(),
        params: adex::TONIC.to_vec(),
        kind: NeuronKind::Excitatory,
        color: None,
    });
    net.set_type(adex_ids, ty);
    net
}

/// The registry line works: `by_id` finds adex with full descriptors —
/// params with units/ranges/docs, state slots, presets — and `init`
/// lands on the leak potential with no adaptation.
#[test]
fn adex_is_registered_with_descriptors() {
    let (_, m) = model::by_id("adex").expect("adex must be in MODELS");
    assert_eq!(m.label(), "AdEx (adaptive exponential IF)");
    assert_eq!(m.params().len(), 10);
    assert_eq!(m.state().len(), 2);
    assert_eq!(m.state()[0].name, "v");
    assert_eq!(m.state()[1].name, "w");
    assert!(!m.presets().is_empty());
    for pd in m.params() {
        assert!(pd.min < pd.max, "param {} range", pd.name);
    }
    let mut x = [0.0f32; 2];
    m.init(&adex::TONIC, &mut x);
    assert_eq!(x, [-70.6, 0.0], "init: v=EL, w=0");
}

/// CPU-vs-CUDA on a mixed net: identical lifetime spike counts, and
/// every adex neuron fires repeatedly — repeated firing is impossible
/// without a working reset on both backends.
#[test]
#[cfg(feature = "cuda")]
fn adex_mixed_cpu_matches_cuda() {
    if !sim::cuda_available() {
        eprintln!("no GPU, skipping");
        return;
    }
    let adex_ids = [1u32, 3, 4, 6];
    let mut net = mixed_net(&adex_ids);
    net.connect(0, 1, 8.0); // izh -> adex
    net.connect(1, 2, 8.0); // adex -> izh
    net.connect(3, 4, 5.0); // adex -> adex
    let cfg = SimConfig {
        record_spikes: true,
        noise_rate_hz: 0.0,
        ..Default::default()
    };
    let mut cpu = Sim::with_params(
        &net,
        cfg.clone(),
        BackendKind::Cpu,
        sim::state::NeuronParams::from_network(&net).unwrap(),
    )
    .unwrap();
    let mut gpu = Sim::with_params(
        &net,
        cfg,
        BackendKind::Cuda,
        sim::state::NeuronParams::from_network(&net).unwrap(),
    )
    .unwrap();
    // Drive every adex neuron (plus izh 0) — pA-scale currents; ~1 nA
    // puts a tonic AdEx well past threshold.
    for (i, s) in [
        (0usize, 15.0),
        (1, 900.0),
        (3, 800.0),
        (4, 1000.0),
        (6, 1200.0),
    ] {
        cpu.set_external(i, s);
        gpu.set_external(i, s);
    }
    cpu.run_steps(3000).unwrap();
    gpu.run_steps(3000).unwrap();
    gpu.synchronize();

    let cc = cpu.spike_counts(usize::MAX).unwrap();
    let gc = gpu.spike_counts(usize::MAX).unwrap();
    eprintln!("per-neuron counts: cpu={cc:?} gpu={gc:?}");
    assert_eq!(cc, gc, "mixed-net spike counts diverged");
    for &i in &adex_ids {
        assert!(
            cc[i as usize] >= 2,
            "adex neuron {i} must fire and reset: {cc:?}"
        );
    }

    // state slot 0 close on every neuron, izh or adex
    let vc = cpu.state(0);
    let vg = gpu.state(0);
    for i in 0..8 {
        assert!(
            (vc[i] - vg[i]).abs() < 0.5,
            "neuron {i}: cpu={} gpu={}",
            vc[i],
            vg[i]
        );
    }
}

/// Same mixed net on CPU alone: the registered model must reach the
/// sim through `from_network` and every adex neuron must fire and
/// reset (counts ≥ 2) — keeps the acceptance green on GPU-less hosts.
#[test]
fn adex_mixed_net_fires_on_cpu() {
    let adex_ids = [1u32, 3, 6];
    let net = mixed_net(&adex_ids);
    let mut sim = Sim::with_params(
        &net,
        SimConfig {
            record_spikes: true,
            noise_rate_hz: 0.0,
            ..Default::default()
        },
        BackendKind::Cpu,
        sim::state::NeuronParams::from_network(&net).unwrap(),
    )
    .unwrap();
    for &i in &adex_ids {
        sim.set_external(i as usize, 1000.0);
    }
    sim.run_steps(3000).unwrap();
    let counts = sim.spike_counts(usize::MAX).unwrap();
    eprintln!("cpu counts: {counts:?}");
    for &i in &adex_ids {
        assert!(
            counts[i as usize] >= 2,
            "adex neuron {i} must fire and reset: {counts:?}"
        );
    }
}
