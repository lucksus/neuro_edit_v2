//! P2-8 regression: `Stimulus.play_start` used to be one shared value —
//! calling `play_input` on input B reset input A's series playhead to the
//! current step. Now each input keeps its own playhead.

use net_format::{Network, NeuronKind};
use sim::cpu::CpuSim;
use sim::SimConfig;

#[test]
fn playing_one_input_does_not_restart_others() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], NeuronKind::Excitatory);
    net.add_input("a", vec![(0, 1.0)], [0.0; 3]);
    net.add_input("b", vec![(0, 1.0)], [0.0; 3]);
    let mut sim = CpuSim::new(&net, SimConfig::default());
    let ramp: Vec<f32> = (0..100).map(|i| i as f32).collect();
    sim.play_input(0, ramp);
    sim.run_steps(50).unwrap();
    let before = sim.stimulus.value(0, sim.t);
    sim.play_input(1, vec![0.0; 10]);
    let after = sim.stimulus.value(0, sim.t);
    assert_eq!(before, after, "playing b moved a's playhead");
    assert_eq!(after, 50.0, "a should be mid-series at t=50");
}

/// Two inputs with staggered playheads stay correctly phased over time.
#[test]
fn staggered_playheads_run_independently() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], NeuronKind::Excitatory);
    net.add_input("a", vec![(0, 1.0)], [0.0; 3]);
    net.add_input("b", vec![(0, 1.0)], [0.0; 3]);
    let mut sim = CpuSim::new(&net, SimConfig::default());
    sim.play_input(0, vec![1.0; 100]);
    sim.run_steps(10).unwrap();
    sim.play_input(1, vec![7.0; 100]);
    sim.run_steps(5).unwrap();
    // at t=15: a is at series index 5, b is at index 5
    assert_eq!(sim.stimulus.value(0, sim.t), 1.0);
    assert_eq!(sim.stimulus.value(1, sim.t), 7.0);
}
