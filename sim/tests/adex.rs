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

/// Stage 22B regression on the REVIEW SCENE itself: in
/// `stage22_c_adex_*.png` the inspector's v(mV)/w(pA) traces sat flat
/// while the Plots window looked like it was spiking — that spike was
/// Auto Y stretching a ±0.03 mV resting noise floor, not a routing bug
/// (the inspector reads the same `PlotHist` map as the tabs). The real
/// defect: the scene's shared `kick` delivered ~14 pA to the AdEx cell,
/// ~40× below what a tonic AdEx needs to cross VT, so it never fired.
///
/// This evals the actual `review/stage22_adex.rhai`, watches the
/// converted cell exactly the way the inspector does (selection →
/// `Watch` → `set_v_watch`), and requires BOTH state slots to sweep a
/// real range — a spike is tens of mV and each reset steps w by b.
/// Sabotage: drop `adex_drive` in the script back to the shared kick
/// (or `sim.set_input("adex_drive", 0.0)`) → v range collapses to the
/// noise floor and both asserts fail.
#[test]
fn adex_scene_watch_series_change_over_time() {
    let mut env = sim::script::Env::new(Network::new(), BackendKind::Cpu, 1).unwrap();
    let _ = env
        .eval_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../review/stage22_adex.rhai"
        ))
        .unwrap();
    let h = env.sim_handle().unwrap();
    let mut sim = h.borrow_mut();
    sim.set_v_watch(&[0]);
    sim.run_steps(1000).unwrap();
    let states = sim.drain_watch().unwrap();
    assert!(
        states.len() >= 2,
        "adex publishes two state slots (v, w): {states:?}"
    );
    let range = |s: &[Vec<f32>]| -> (f32, f32) {
        s[0].iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)))
    };
    let (vmin, vmax) = range(&states[0]);
    let (wmin, wmax) = range(&states[1]);
    assert!(
        vmax - vmin > 40.0,
        "adex v must swing across a spike ({vmin:.2}..{vmax:.2}) — the scene's drive is too weak"
    );
    assert!(
        wmax - wmin > 10.0,
        "adex w must build adaptation ({wmin:.2}..{wmax:.2})"
    );
}

/// Stage 24d root cause: `noise_kick_mv` is mV of depolarisation — a
/// one-step current pulse scaled by the model's `noise_gain` (AdEx: C,
/// since its input is pA into `dv/dt = (… + I)/C`). Before the gain,
/// the example's 200 Hz × 700 setting gave ~2.5 mV per kick (700/281)
/// and the liquid died in seconds. One AdEx neuron on noise alone must
/// fire at the example's setting, and firing must rise with the noise
/// rate — on both backends.
fn lone_adex_net() -> Network {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], NeuronKind::Excitatory);
    let ty = net.upsert_type(NeuronType {
        name: "ax".into(),
        model: "adex".into(),
        params: adex::TONIC.to_vec(),
        kind: NeuronKind::Excitatory,
        color: None,
    });
    net.set_type(&[0], ty);
    net
}

/// Mean Hz of the lone cell over `ms` of Poisson kicks at `rate_hz`.
fn lone_rate(backend: BackendKind, rate_hz: f32, kick_mv: f32, ms: f32) -> f64 {
    let net = lone_adex_net();
    let mut sim = Sim::with_params(
        &net,
        SimConfig {
            noise_rate_hz: rate_hz,
            noise_kick_mv: kick_mv,
            seed: 5,
            ..Default::default()
        },
        backend,
        sim::state::NeuronParams::from_network(&net).unwrap(),
    )
    .unwrap();
    sim.run_ms(ms).unwrap();
    sim.spikes_total() as f64 / (ms as f64 / 1000.0)
}

fn assert_noise_drives_adex(backend: BackendKind) {
    // The example tail's setting (60 Hz x 14 mV) must fire the cell.
    let at_example = lone_rate(backend, 60.0, 14.0, 20_000.0);
    assert!(
        at_example > 0.5,
        "{backend:?}: lone AdEx is quiet at the example's noise setting ({at_example} Hz)"
    );
    // and the firing rate must rise with the noise rate
    let lo = lone_rate(backend, 30.0, 18.0, 20_000.0);
    let hi = lone_rate(backend, 120.0, 18.0, 20_000.0);
    assert!(
        hi > lo * 1.5,
        "{backend:?}: rate must rise with noise rate ({lo} -> {hi} Hz)"
    );
}

#[test]
fn adex_noise_drives_lone_neuron_cpu() {
    assert_noise_drives_adex(BackendKind::Cpu);
}

#[test]
#[cfg(feature = "cuda")]
fn adex_noise_drives_lone_neuron_cuda() {
    if !sim::cuda_available() {
        eprintln!("no GPU, skipping");
        return;
    }
    assert_noise_drives_adex(BackendKind::Cuda);
}

/// Stage 25 (Data's 4b): the AdEx Euler step ends ~400–500 mV above
/// Vpeak on the crossing step; the stored v must be cut at Vpeak
/// (Brette & Gerstner) so watched traces show the peak, not the
/// runaway. A driven AdEx cell must spike ≥ 2 times and every watched
/// sample must stay ≤ Vpeak — on both backends. Sabotage: remove the
/// `x[0] > p[9]` cut in adex.rs → vmax lands in the hundreds.
fn vpeak_cut(backend: BackendKind) {
    let net = lone_adex_net();
    let mut sim = Sim::with_params(
        &net,
        SimConfig {
            record_spikes: true,
            noise_rate_hz: 0.0,
            ..Default::default()
        },
        backend,
        sim::state::NeuronParams::from_network(&net).unwrap(),
    )
    .unwrap();
    sim.set_external(0, 1000.0);
    sim.set_v_watch(&[0]);
    sim.run_steps(3000).unwrap();
    let counts = sim.spike_counts(usize::MAX).unwrap();
    assert!(counts[0] >= 2, "{backend:?}: cell must fire: {counts:?}");
    let states = sim.drain_watch().unwrap();
    let vmax = states[0]
        .iter()
        .flatten()
        .fold(f32::MIN, |a, &v| a.max(v));
    assert!(
        vmax <= adex::TONIC[9],
        "{backend:?}: v must be cut at Vpeak={} — got {vmax}",
        adex::TONIC[9]
    );
}

#[test]
fn adex_v_cut_at_vpeak_cpu() {
    vpeak_cut(BackendKind::Cpu);
}

#[test]
#[cfg(feature = "cuda")]
fn adex_v_cut_at_vpeak_cuda() {
    if !sim::cuda_available() {
        eprintln!("no GPU, skipping");
        return;
    }
    vpeak_cut(BackendKind::Cuda);
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
