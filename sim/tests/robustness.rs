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

/// P2-7: `Cmd::SetBackend` (editor backend switch) used to be ignored by
/// scripts — the Rhai `sim.*` closures captured the backend chosen at Env
/// creation. `Env::set_backend` must make the *next* script-built sim use
/// the new backend.
#[test]
#[cfg(feature = "cuda")]
fn set_backend_is_respected_by_scripts() {
    if !sim::cuda_available() {
        return;
    }
    let mut env = sim::script::Env::new(Network::new(), BackendKind::Cpu, 1).unwrap();
    env.set_backend(BackendKind::Cuda);
    env.eval_str("net.add_neuron(0.0,0.0,0.0); sim.run(1.0);")
        .unwrap();
    let h = env.sim_handle().unwrap();
    let kind = match &*h.borrow() {
        sim::Sim::Cuda(_) => "Cuda",
        #[allow(unreachable_patterns)]
        _ => "not Cuda",
    };
    assert_eq!(kind, "Cuda", "script-built sim should honour SetBackend");

    // and the reverse: switching back to CPU rebuilds a CPU sim
    env.set_backend(BackendKind::Cpu);
    env.eval_str("sim.run(1.0);").unwrap();
    let h = env.sim_handle().unwrap();
    assert!(
        matches!(&*h.borrow(), sim::Sim::Cpu(_)),
        "switching back should rebuild a CPU sim"
    );
}
