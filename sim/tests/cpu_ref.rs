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
    let mut sim = CpuSim::new(&net, SimConfig::default());
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
    let mut sim = CpuSim::new(&net, SimConfig::default());
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
    });
    // dt=0.1 -> delay 2 steps
    let mut sim = CpuSim::new(&net, SimConfig::default());
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
        (sim.s[1] - 30.0 * (1.0 - 0.1 / 2.0)).abs() < 1e-3,
        "s={}",
        sim.s[1]
    );
    sim.run_steps(1).unwrap();
    let expected = 30.0f32 * (1.0f32 - 0.05).powi(2);
    assert!(
        (sim.s[1] - expected).abs() < 1e-2,
        "s={} vs {}",
        sim.s[1],
        expected
    );
}

#[test]
fn no_self_connection_and_csr_order() {
    let mut net = net_with(3);
    net.connect(0, 1, 5.0);
    net.connect(0, 2, 6.0);
    net.connect(2, 0, 7.0);
    let csr = net.synapse_csr(0.1);
    assert_eq!(csr.src_ptr, vec![0, 2, 2, 3]);
    assert_eq!(csr.tgt, vec![1, 2, 0]);
}

#[test]
fn stimulus_reaches_only_its_targets() {
    let mut net = net_with(3);
    net.add_input("drive", vec![(0, 1.0), (2, 2.0)], [0.0; 3]);
    let mut sim = CpuSim::new(&net, SimConfig::default());
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
    let mut sim = CpuSim::new(&net, SimConfig::default());
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
    let mut a = CpuSim::new(&net, cfg.clone());
    let mut b = CpuSim::new(&net, cfg.clone());
    let mut c = CpuSim::new(&net, SimConfig { seed: 2, ..cfg });
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
    let mut sim = CpuSim::new(&net, SimConfig::default());
    sim.set_external(0, 30.0);
    sim.run_steps(50).unwrap();
    sim.reset_state();
    assert_eq!(sim.t, 0);
    assert!(sim.v.iter().all(|&v| v == -65.0));
    assert!(sim.ring.iter().all(|&x| x == 0.0));
    assert!(sim.s.iter().all(|&x| x == 0.0));
}
