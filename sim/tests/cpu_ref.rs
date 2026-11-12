//! CPU reference tests: Izhikevich against hand-computed traces, delay ring,
//! exponential synaptic current, stimulus routing, noise determinism.

use net_format::{Network, NeuronKind};
use sim::cpu::CpuSim;
use sim::SimConfig;

fn net_with(n: usize) -> Network {
    let mut net = Network::new();
    for i in 0..n {
        net.add_neuron([i as f32 * 50.0, 0.0, 0.0], NeuronKind::Excitatory);
    }
    net
}

#[test]
fn single_neuron_step_matches_hand_computed() {
    // Izhikevich a=0.02 b=0.2 c=-65 d=8, v=-65, u=-13, dt=0.1, I=10
    let net = net_with(1);
    let mut sim = CpuSim::new(&net, SimConfig::default()).unwrap();
    sim.set_external(0, 10.0);
    // step 1 (v=-65 < 30 so no reset):
    // v += 0.1*(0.04*4225 -325 +140 +13 +10) = 0.1*(169-162) = +0.7 -> -64.3
    // u += 0.1*0.02*(0.2*-64.3 - (-13)) = 0.002*(-12.86+13) = +0.00028 -> -12.99972
    sim.run_steps(1).unwrap();
    assert!((sim.v[0] - (-64.3)).abs() < 1e-4, "v={}", sim.v[0]);
    assert!((sim.u[0] - (-12.99972)).abs() < 1e-4, "u={}", sim.u[0]);
    // step 2: v=-64.3, u=-12.99972
    // v += 0.1*(0.04*4134.49 -321.5 +140 +12.99972 +10) = 0.1*(165.3796-168.50028)
    sim.run_steps(1).unwrap();
    let v2 = -64.3f64 + 0.1 * (0.04 * 64.3f64 * 64.3 - 5.0 * 64.3 + 140.0 + 12.99972 + 10.0);
    assert!(
        (sim.v[0] as f64 - v2).abs() < 1e-3,
        "v2={} vs {}",
        sim.v[0],
        v2
    );
}

#[test]
fn spike_resets_next_step() {
    let net = net_with(1);
    let mut sim = CpuSim::new(&net, SimConfig::default()).unwrap();
    sim.set_external(0, 30.0);
    // run until first spike
    let mut spike_step = None;
    for i in 0..200 {
        sim.run_steps(1).unwrap();
        if sim.trace[0] > 0.9 {
            spike_step = Some(i);
            break;
        }
    }
    let _t = spike_step.expect("neuron should spike");
    // on the spike step v is still at its peak; the reset happens next step
    let trace_on_spike = sim.trace[0];
    sim.run_steps(1).unwrap();
    // one step after the spike: v was reset to c and integrated once
    assert!(sim.v[0] < -60.0, "v after spike {}", sim.v[0]);
    assert!(sim.trace[0] < trace_on_spike);
}

#[test]
fn synapse_arrives_after_delay_and_decays() {
    // A(0) --w=30, delay 2 steps--> B(1). Force A to spike by strong ext.
    let mut net = net_with(2);
    net.synapses.push(net_format::Synapse {
        src: 0,
        tgt: 1,
        w: 30.0,
        delay_ms: 0.2,
        tau_ms: 0.0,
    });
    // dt=0.1 -> delay 2 steps
    let mut sim = CpuSim::new(&net, SimConfig::default()).unwrap();
    // fire A exactly once: strong current until first spike then off
    sim.set_external(0, 25.0);
    let mut first_spike_t = None;
    for _ in 0..500 {
        let spikes_before = sim.spikes_total;
        sim.run_steps(1).unwrap();
        if sim.spikes_total > spikes_before {
            first_spike_t = Some(sim.t - 1);
            break;
        }
    }
    let _t_spike = first_spike_t.unwrap();
    sim.set_external(0, 0.0);
    // B's synaptic current must be 0 for delay-1 steps, then jump by w
    sim.run_steps(1).unwrap(); // t_spike+1: still in flight (arrives at t_spike+2)
    assert_eq!(sim.s[1], 0.0, "s too early");
    sim.run_steps(1).unwrap(); // arrival step: s += w, neuron reads it, then decays
    assert!(
        (sim.s[1] - 30.0 * (-0.1f32 / 2.0).exp()).abs() < 1e-3,
        "s={}",
        sim.s[1]
    );
    sim.run_steps(1).unwrap();
    let expected = 30.0f32 * (-0.05f32).exp().powi(2);
    assert!(
        (sim.s[1] - expected).abs() < 1e-2,
        "s={} vs {}",
        sim.s[1],
        expected
    );
}

#[test]
fn custom_tau_synapse_arrives_then_decays_its_own_trace() {
    // Same wiring as synapse_arrives_after_delay_and_decays but tau_ms=20:
    // the spike's current must live in xsyn (not the merged s), decay with
    // the synapse's own constant, and s[1] must stay 0.
    let mut net = net_with(2);
    net.synapses.push(net_format::Synapse {
        src: 0,
        tgt: 1,
        w: 30.0,
        delay_ms: 0.2,
        tau_ms: 20.0,
    });
    let mut sim = CpuSim::new(&net, SimConfig::default()).unwrap();
    assert_eq!(sim.xsyn.len(), 1, "one custom synapse expected");
    sim.set_external(0, 25.0);
    let mut first_spike_t = None;
    for _ in 0..500 {
        let before = sim.spikes_total;
        sim.run_steps(1).unwrap();
        if sim.spikes_total > before {
            first_spike_t = Some(sim.t - 1);
            break;
        }
    }
    first_spike_t.unwrap();
    sim.set_external(0, 0.0);
    // in flight: still nothing anywhere
    sim.run_steps(1).unwrap();
    assert_eq!(sim.xsyn[0], 0.0);
    assert_eq!(sim.s[1], 0.0);
    // arrival step: x += w, neuron reads it, then the trace decays once
    sim.run_steps(1).unwrap();
    let dec = 1.0 - 0.1 / 20.0;
    assert_eq!(sim.s[1], 0.0, "custom synapse must not feed merged s");
    assert!(
        (sim.xsyn[0] - 30.0 * dec).abs() < 1e-3,
        "xsyn={} vs {}",
        sim.xsyn[0],
        30.0 * dec
    );
    sim.run_steps(1).unwrap();
    let expected = 30.0f32 * dec.powi(2);
    assert!(
        (sim.xsyn[0] - expected).abs() < 1e-2,
        "xsyn={} vs {}",
        sim.xsyn[0],
        expected
    );
}

#[test]
fn slow_tau_keeps_target_firing_longer() {
    // One spike into B: a 50 ms synaptic trace must drive the target
    // noticeably longer than the default 2 ms merged current.
    let make = |tau: f32| {
        let mut net = net_with(2);
        net.synapses.push(net_format::Synapse {
            src: 0,
            tgt: 1,
            w: 15.0,
            delay_ms: 0.2,
            tau_ms: tau,
        });
        CpuSim::new(&net, SimConfig::default()).unwrap()
    };
    let mut def = make(0.0);
    let mut slow = make(50.0);
    for sim in [&mut def, &mut slow] {
        sim.set_external(0, 25.0);
        // fire neuron 0 exactly once
        loop {
            let before = sim.spikes_total;
            sim.run_steps(1).unwrap();
            if sim.spikes_total > before {
                break;
            }
        }
        sim.set_external(0, 0.0);
    }
    def.run_steps(400).unwrap();
    slow.run_steps(400).unwrap();
    let d_spikes = def.pn_total[1];
    let s_spikes = slow.pn_total[1];
    assert!(
        s_spikes > d_spikes,
        "slow tau should prolong target firing: default={d_spikes} slow={s_spikes}"
    );
    // and the custom trace itself must still be nonzero well past arrival
    assert!(slow.xsyn[0] > 0.5, "xsyn={}", slow.xsyn[0]);
}

#[test]
fn default_tau_is_bit_exact() {
    // Pinned: a net where every synapse has tau_ms == 0 runs the merged
    // pre-per-synapse code path — the spike train must reproduce the
    // pre-change bits on a fixed seed.
    use net_format::builders;
    let mut net = Network::new();
    net.meta.dt_ms = 0.1;
    let mut rng = net_format::Rng::new(3);
    builders::create_2d_grid(&mut net, "g", 8, 8, 30.0, [0.0; 3]);
    builders::create_connections(&mut net, "g", 0.35, 80.0, 6.0, &mut rng);
    let cfg = SimConfig {
        record_spikes: true,
        spike_record_steps: 512,
        noise_rate_hz: 20.0,
        noise_kick_mv: 0.8,
        seed: 7,
        ..Default::default()
    };
    let mut sim = CpuSim::new(&net, cfg).unwrap();
    sim.run_steps(400).unwrap();
    // fingerprint: spike bits + v for the whole recorded window
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut fold = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    };
    for back in 0..400 {
        for &w in sim.spike_row(back).unwrap() {
            fold(w);
        }
    }
    for &x in &sim.v {
        fold(x.to_bits() as u64);
    }
    assert_eq!(h, 0x9d79_2bd2_3313_2f5d, "default-path fingerprint changed");
}

#[test]
fn no_self_connection_and_csr_order() {
    let mut net = net_with(3);
    net.connect(0, 1, 5.0);
    net.connect(0, 2, 6.0);
    net.connect(2, 0, 7.0);
    let csr = net.synapse_csr(0.1).unwrap();
    assert_eq!(csr.src_ptr, vec![0, 2, 2, 3]);
    assert_eq!(csr.tgt, vec![1, 2, 0]);
}

#[test]
fn stimulus_reaches_only_its_targets() {
    let mut net = net_with(3);
    net.add_input("drive", vec![(0, 1.0), (2, 2.0)], [0.0; 3]);
    let mut sim = CpuSim::new(&net, SimConfig::default()).unwrap();
    sim.set_input_constant(0, 5.0);
    // single step: neuron1 untouched
    let s1 = sim.s[1];
    sim.run_steps(1).unwrap();
    assert_eq!(sim.s[1], s1);
    // v of neuron0 and neuron2 must diverge (2 gets double current)
    let dv = (sim.v[2] - sim.v[0]).abs();
    assert!(dv > 0.05, "dv={}", dv);
    assert!((sim.v[0] - -64.8).abs() < 0.2, "v0={}", sim.v[0]);
}

#[test]
fn played_series_is_step_locked() {
    let mut net = net_with(1);
    net.add_input("x", vec![(0, 1.0)], [0.0; 3]);
    let mut sim = CpuSim::new(&net, SimConfig::default()).unwrap();
    // 10 units of drive for exactly one step, then silence
    sim.play_input(0, vec![10.0, 0.0, 0.0]);
    sim.run_steps(1).unwrap();
    // same as the hand-computed I=10 step: v = -64.3
    assert!((sim.v[0] - (-64.3)).abs() < 1e-4, "v={}", sim.v[0]);
    // series exhausted -> later steps get 0, not the constant
    assert_eq!(sim.stimulus.value(0, 3), 0.0);
    // with sustained drive the neuron spikes within a few steps
    sim.play_input(0, vec![25.0; 20]);
    let mut spiked = false;
    for _ in 0..20 {
        sim.run_steps(1).unwrap();
        if sim.trace[0] == 1.0 {
            spiked = true;
        }
    }
    assert!(spiked);
}

#[test]
fn noise_is_deterministic_per_seed() {
    let net = net_with(64);
    let cfg = SimConfig {
        noise_rate_hz: 500.0,
        ..Default::default()
    };
    let mut a = CpuSim::new(&net, cfg.clone()).unwrap();
    let mut b = CpuSim::new(&net, cfg.clone()).unwrap();
    let mut c = CpuSim::new(&net, SimConfig { seed: 2, ..cfg }).unwrap();
    a.run_steps(100).unwrap();
    b.run_steps(100).unwrap();
    c.run_steps(100).unwrap();
    assert_eq!(a.v, b.v);
    assert_ne!(a.v, c.v);
}

#[test]
fn reset_clears_state() {
    let mut net = net_with(4);
    net.connect(0, 1, 5.0);
    let mut sim = CpuSim::new(&net, SimConfig::default()).unwrap();
    sim.set_external(0, 30.0);
    sim.run_steps(50).unwrap();
    sim.reset_state();
    assert_eq!(sim.t, 0);
    assert!(sim.v.iter().all(|&v| v == -65.0));
    assert!(sim.ring.iter().all(|&x| x == 0.0));
    assert!(sim.s.iter().all(|&x| x == 0.0));
}

/// M1: a synapse from an inhibitory neuron inhibits — the CSR carries
/// -|w| from the source's kind, so the target fires LESS with the
/// synapse than without it (before the fix the kind was colour only).
#[test]
fn inhibitory_source_suppresses_target() {
    fn pair(inh: bool, synapse: bool) -> (Network, u32, u32) {
        let mut net = Network::new();
        let src = net.add_neuron(
            [0.0; 3],
            if inh {
                NeuronKind::Inhibitory
            } else {
                NeuronKind::Excitatory
            },
        );
        let tgt = net.add_neuron([60.0, 0.0, 0.0], NeuronKind::Excitatory);
        if synapse {
            net.connect(src, tgt, 6.0);
        }
        (net, src, tgt)
    }
    let run = |net: &Network| -> u32 {
        let mut sim = CpuSim::new(
            net,
            SimConfig {
                record_spikes: true,
                ..Default::default()
            },
        )
        .unwrap();
        // source fires hard, target driven into a steady rhythm on its own
        sim.set_external(0, 30.0);
        sim.set_external(1, 12.0);
        sim.run_ms(400.0).unwrap();
        sim.spike_counts(usize::MAX).unwrap()[1]
    };

    let (free_net, _, _) = pair(true, false);
    let (inh_net, _, _) = pair(true, true);
    let (exc_net, _, _) = pair(false, true);
    let free = run(&free_net);
    let inh = run(&inh_net);
    let exc = run(&exc_net);
    assert!(free > 0, "driven target must fire");
    assert!(
        inh < free,
        "inhibitory synapse must reduce firing: {inh} !< {free}"
    );
    assert!(
        exc > inh,
        "same-weight excitatory synapse must not inhibit: {exc} !> {inh}"
    );
    // and the CSR itself carries the signed weight
    let (net, _, _) = pair(true, true);
    let csr = net.synapse_csr(0.1).unwrap();
    assert!(csr.w.iter().all(|&w| w < 0.0), "inh src must emit -|w|");
    let (net, _, _) = pair(false, true);
    assert!(net.synapse_csr(0.1).unwrap().w.iter().all(|&w| w > 0.0));
}

/// M1: set_kind flips the sign at the next build and swaps in
/// fast-spiking defaults while the model is stock — a user-tuned model
/// is left alone.
#[test]
fn set_kind_reweights_and_defaults() {
    let mut net = Network::new();
    let a = net.add_neuron([0.0; 3], NeuronKind::Excitatory);
    let b = net.add_neuron([10.0, 0.0, 0.0], NeuronKind::Excitatory);
    net.connect(a, b, 6.0);
    assert!(net.synapse_csr(0.1).unwrap().w[0] > 0.0);

    net.set_kind(a, NeuronKind::Inhibitory);
    assert!(
        net.synapse_csr(0.1).unwrap().w[0] < 0.0,
        "kind flip must re-sign"
    );
    assert_eq!(
        net.neuron(a).model.params(),
        [0.1, 0.2, -65.0, 2.0],
        "stock RS model swaps to FS defaults"
    );

    // user-set params survive the flip back and forth
    net.neuron_mut(a).model = net_format::NeuronModel::Izhikevich {
        a: 0.03,
        b: 0.25,
        c: -60.0,
        d: 4.0,
    };
    net.set_kind(a, NeuronKind::Excitatory);
    assert_eq!(net.neuron(a).model.params()[0], 0.03, "explicit model kept");
    net.set_kind(a, NeuronKind::Inhibitory);
    assert_eq!(net.neuron(a).model.params()[0], 0.03);
}

/// M2: the tutorial liquid must get real delay structure — with the
/// default conduction speed (100 u/ms, spacing 60) neighbours sit at
/// several different step counts instead of one flat step.
#[test]
fn tutorial_liquid_has_multiple_delay_steps() {
    use net_format::{builders, Rng};
    let mut net = Network::new();
    let mut rng = Rng::new(1);
    builders::create_3d_grid(&mut net, "liquid", 6, 6, 6, 60.0, [0.0; 3]);
    builders::create_connections(&mut net, "liquid", 0.4, 400.0, 5.0, &mut rng);
    let csr = net.synapse_csr(0.1).unwrap();
    let steps: std::collections::BTreeSet<u16> = csr.delay_steps.iter().copied().collect();
    assert!(
        steps.len() > 1,
        "expected several distinct delay steps, got {steps:?}"
    );
    assert!(steps.iter().max() > Some(&1));
}

/// M4: a delay past the u16 ring is an error, not a silent clamp that
/// would deliver spikes early.
#[test]
fn delay_over_ring_is_an_error() {
    let mut net = Network::new();
    let a = net.add_neuron([0.0; 3], NeuronKind::Excitatory);
    let b = net.add_neuron([10.0, 0.0, 0.0], NeuronKind::Excitatory);
    net.synapses.push(net_format::Synapse {
        src: a,
        tgt: b,
        w: 1.0,
        delay_ms: 7000.0, // 70000 steps at dt 0.1
        tau_ms: 0.0,
    });
    let err = net.synapse_csr(0.1).unwrap_err();
    assert!(err.contains("exceeds"), "{err}");
    assert!(CpuSim::new(&net, SimConfig::default()).is_err());
}

/// M4: tau < dt used to produce a NEGATIVE decay factor (1 - dt/tau),
/// which flips the current's sign each step and can oscillate to
/// infinity. exp(-dt/tau) stays in (0,1].
#[test]
fn tau_below_dt_decays_safely() {
    let mut net = Network::new();
    let a = net.add_neuron([0.0; 3], NeuronKind::Excitatory);
    let b = net.add_neuron([10.0, 0.0, 0.0], NeuronKind::Excitatory);
    net.synapses.push(net_format::Synapse {
        src: a,
        tgt: b,
        w: 5.0,
        delay_ms: 1.0,
        tau_ms: 0.05, // half of dt — Euler decay would be -1.0
    });
    let mut sim = CpuSim::new(
        &net,
        SimConfig {
            record_spikes: true,
            ..Default::default()
        },
    )
    .unwrap();
    sim.set_external(0, 30.0);
    sim.run_ms(30.0).unwrap();
    assert!(sim.v[1].is_finite(), "v went non-finite: {}", sim.v[1]);
    // current must not accumulate sign-flipped energy
    for &x in &sim.s {
        assert!(x.is_finite() && x >= 0.0, "negative synaptic current {x}");
    }
}
