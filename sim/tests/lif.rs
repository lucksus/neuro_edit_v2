//! Test-only second model: leaky integrate-and-fire. Proves the generic
//! model/param/state blocks end to end — a hand-computed CPU trace, and
//! CPU-vs-CUDA parity on a net that mixes izh and LIF neurons.
//!
//! Not registered in `model::MODELS`: LIF here is a plumbing test, not a
//! user-facing model (protocol/editor exposure is Stage 21's call).

use net_format::{Network, NeuronKind};
use sim::cpu::CpuSim;
use sim::model::{NeuronModel, ParamDef, StateDef};
use sim::state::NeuronParams;
use sim::{BackendKind, Sim, SimConfig};

/// Params [tau, e, r, vth, vreset]; state [v]. Euler:
/// `v += dt * (e - v + r*I) / tau`; a step ending at `v >= vth` spikes
/// and `v = vreset` on the NEXT step's entry — same ordering rule as izh.
struct Lif;

static LIF_P: &[ParamDef] = &[
    ParamDef {
        name: "tau",
        unit: "ms",
        default: 20.0,
        min: 1.0,
        max: 100.0,
        doc: "membrane time constant",
    },
    ParamDef {
        name: "e",
        unit: "mV",
        default: -65.0,
        min: -80.0,
        max: 0.0,
        doc: "rest potential",
    },
    ParamDef {
        name: "r",
        unit: "",
        default: 1.0,
        min: 0.0,
        max: 10.0,
        doc: "input resistance",
    },
    ParamDef {
        name: "vth",
        unit: "mV",
        default: -50.0,
        min: -70.0,
        max: 0.0,
        doc: "spike threshold",
    },
    ParamDef {
        name: "vreset",
        unit: "mV",
        default: -70.0,
        min: -80.0,
        max: 0.0,
        doc: "post-spike reset",
    },
];
static LIF_S: &[StateDef] = &[StateDef {
    name: "v",
    unit: "mV",
    plot_range: (-80.0, 0.0),
}];
/// [tau, e, r, vth, vreset]
static LIF_DEF: &[f32] = &[20.0, -65.0, 1.0, -50.0, -70.0];

static LIF: Lif = Lif;

impl NeuronModel for Lif {
    fn id(&self) -> &'static str {
        "lif"
    }
    fn label(&self) -> &'static str {
        "leaky integrate-and-fire"
    }
    fn params(&self) -> &'static [ParamDef] {
        LIF_P
    }
    fn state(&self) -> &'static [StateDef] {
        LIF_S
    }
    fn presets(&self) -> &'static [(&'static str, &'static [f32])] {
        &[]
    }
    fn default_for(&self, _: NeuronKind) -> &'static [f32] {
        LIF_DEF
    }
    fn init(&self, p: &[f32], x: &mut [f32]) {
        x[0] = p[1];
    }
    fn step(&self, p: &[f32], x: &mut [f32], input: f32, dt: f32) -> bool {
        if x[0] >= p[3] {
            x[0] = p[4];
        }
        x[0] += dt * (p[1] - x[0] + p[2] * input) / p[0];
        x[0] >= p[3]
    }
    fn cuda_step(&self) -> &'static str {
        "if(x[0] >= p[3]){ x[0] = p[4]; }\n\
         x[0] += dt * (p[1] - x[0] + p[2] * I) / p[0];\n\
         spiked = x[0] >= p[3];"
    }
}

/// Params where neurons `lif_ids` run the test LIF (table index 1) and
/// the rest run izh — exercises mixed P/X rows on both backends.
fn mixed_params(net: &Network, lif_ids: &[usize]) -> NeuronParams {
    let mut p = NeuronParams::from_network(net).unwrap();
    p.models.push(&LIF);
    p.grow(5, 2);
    for &i in lif_ids {
        p.set(i, 1, LIF_DEF);
    }
    p
}

fn net_with(n: usize) -> Network {
    let mut net = Network::new();
    for i in 0..n {
        net.add_neuron([i as f32 * 50.0, 0.0, 0.0], NeuronKind::Excitatory);
    }
    net
}

/// Hand-computed Euler trace on the CPU: the exact values a LIF neuron
/// must hit under a constant drive, including the reset one step after
/// a crossing (same entry-reset order as izh).
#[test]
fn lif_hand_computed_trace() {
    let net = net_with(1);
    let mut sim = CpuSim::with_params(
        &net,
        SimConfig {
            dt_ms: 1.0,
            record_spikes: true,
            ..Default::default()
        },
        mixed_params(&net, &[0]),
    )
    .unwrap();

    // quiet: v starts at e=-65 and stays
    sim.run_steps(3).unwrap();
    assert_eq!(sim.state(0)[0], -65.0);

    // drive 10: v += (e - v + r*I)/tau each step — hand values:
    // v1 = -65 + (-65 +65 +10)/20 = -64.5; v2 = -64.5 + (0.5+10)/20 = -64.0? no:
    // v2 = -64.5 + (-65+64.5+10)/20 = -64.5 + 9.5/20 = -64.025
    sim.set_external(0, 10.0);
    sim.run_steps(1).unwrap();
    assert!(
        (sim.state(0)[0] - (-64.5)).abs() < 1e-6,
        "v={:?}",
        sim.state(0)
    );
    sim.run_steps(1).unwrap();
    assert!(
        (sim.state(0)[0] - (-64.025)).abs() < 1e-6,
        "v={:?}",
        sim.state(0)
    );

    // drive 100: v_inf = -65 + 100 = 35 > vth — crosses, spikes, resets
    // next step to -70 (then integrates once more).
    sim.set_external(0, 100.0);
    let mut spike_step = None;
    for k in 0..40 {
        sim.run_steps(1).unwrap();
        if let Some(row) = sim.spike_row(0) {
            if row[0] & 1 == 1 {
                spike_step = Some(sim.t - 1);
                break;
            }
        }
        let _ = k;
    }
    let t_spike = spike_step.expect("LIF must spike under drive 100");
    // the crossing step leaves v above threshold...
    assert!(sim.state(0)[0] >= -50.0, "post-spike v={:?}", sim.state(0));
    sim.run_steps(1).unwrap();
    // ...the NEXT step resets first, then integrates: v = -70 +
    // (-65 +70 +100)/20 = -70 + 105/20 = -64.75
    assert!(
        (sim.state(0)[0] - (-64.75)).abs() < 1e-6,
        "post-reset v={:?} (spike at t={t_spike})",
        sim.state(0)
    );
}

/// CPU vs CUDA on a mixed net: izh neurons and LIF neurons in one sim
/// (model table [izh, lif], global P=5/X=2 rows) must produce the same
/// spikes and near-identical state on both backends.
#[test]
#[cfg(feature = "cuda")]
fn lif_mixed_cpu_matches_cuda() {
    if !sim::cuda_available() {
        eprintln!("no GPU, skipping");
        return;
    }
    let mut net = net_with(8);
    net.connect(0, 1, 8.0); // izh -> lif
    net.connect(1, 2, 8.0); // lif -> izh
    net.connect(3, 4, 5.0); // lif -> lif
    let params = mixed_params(&net, &[1, 3, 4, 6]);
    let cfg = SimConfig {
        record_spikes: true,
        noise_rate_hz: 0.0,
        ..Default::default()
    };

    let mut cpu = Sim::with_params(&net, cfg.clone(), BackendKind::Cpu, params.clone()).unwrap();
    let mut gpu = Sim::with_params(&net, cfg, BackendKind::Cuda, params).unwrap();
    for s in [0.0, 25.0, 18.0, 30.0] {
        cpu.set_external(s as usize, 15.0 + s);
        gpu.set_external(s as usize, 15.0 + s);
    }
    cpu.run_steps(300).unwrap();
    gpu.run_steps(300).unwrap();
    gpu.synchronize();

    // lifetime spike counts are exact on both backends
    assert_eq!(
        cpu.spike_counts(usize::MAX).unwrap(),
        gpu.spike_counts(usize::MAX).unwrap(),
        "mixed-net spike counts diverged"
    );
    assert!(cpu.spikes_total() > 0, "the mixed net must actually fire");

    // state slot 0 (membrane potential) close on every neuron, izh or lif
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
    // watched recording: slot 0 samples step-aligned and close
    cpu.set_v_watch(&[1, 4]);
    gpu.set_v_watch(&[1, 4]);
    cpu.run_steps(50).unwrap();
    gpu.run_steps(50).unwrap();
    let hc = cpu.drain_watch().unwrap();
    let hg = gpu.drain_watch().unwrap();
    assert_eq!(hc.len(), 2, "two state slots (max X) recorded");
    assert_eq!(hc.len(), hg.len());
    for s in 0..2 {
        assert_eq!(hc[s].len(), hg[s].len(), "slot {s} watch count");
        for k in 0..hc[s].len() {
            assert_eq!(hc[s][k].len(), hg[s][k].len(), "slot {s} watch {k} samples");
            for (j, (a, b)) in hc[s][k].iter().zip(&hg[s][k]).enumerate() {
                assert!(
                    (a - b).abs() < 0.5,
                    "slot {s} watch {k} step {j}: cpu={a} gpu={b}"
                );
            }
        }
    }
}
