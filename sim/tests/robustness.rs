//! Empty networks must construct and run on every backend (no 0-block CUDA
//! launches, no 0-len buffers), and `run_steps`/`run_ms` errors are Results.

use net_format::Network;
use sim::{BackendKind, Sim, SimConfig};

#[test]
fn empty_net_runs_cpu() {
    let net = Network::new();
    let mut sim = Sim::new(&net, SimConfig::default(), BackendKind::Cpu).unwrap();
    sim.run_steps(10).unwrap();
    assert_eq!(sim.step_count(), 10);
    assert!(sim.v().is_empty());
    assert!(sim.trace().is_empty());
    assert_eq!(sim.spike_counts(5).unwrap(), Vec::<u32>::new());
    sim.reset_state();
    assert_eq!(sim.step_count(), 0);
}

#[test]
#[cfg(feature = "cuda")]
fn empty_net_runs_cuda() {
    if !sim::cuda_available() {
        return;
    }
    let net = Network::new();
    let mut sim = Sim::new(&net, SimConfig::default(), BackendKind::Cuda).unwrap();
    sim.run_steps(10).unwrap();
    assert_eq!(sim.step_count(), 10);
    assert!(sim.v().is_empty());
    assert!(sim.trace().is_empty());
    assert_eq!(sim.spike_counts(5).unwrap(), Vec::<u32>::new());
    sim.reset_state();
    assert_eq!(sim.step_count(), 0);
}
