//! Robustness: script-supplied indices become script errors (not panics),
//! and malformed files fail to load instead of silently producing empty nets.

use net_format::Network;
use sim::script::Env;
use sim::BackendKind;

fn env() -> Env {
    Env::new(Network::new(), BackendKind::Cpu, 1).unwrap()
}

#[test]
fn script_index_out_of_range_is_an_error() {
    let mut env = env();
    env.eval_str("net.add_neuron(0, 0, 0)").unwrap();
    for code in [
        "net.connect(0, 99, 1.0)",
        "net.connect(99, 0, 1.0)",
        "net.connect(-1, 0, 1.0)",
        "net.set_izh(5, 0.02, 0.2, -65.0, 8.0)",
        "net.delete_neuron(9)",
        "net.delete_neurons([0, 9])",
        "net.set_pos(9, 0, 0, 0)",
        "net.pos(9)",
        "net.neuron_name(9)",
        "net.set_neuron_name(9, \"x\")",
        "net.add_to_group(\"g\", [9])",
        "net.add_input(\"i\", [[9, 1.0]])",
        "net.add_readout(\"r\", [9])",
    ] {
        let e = env.eval_str(code).expect_err(code);
        assert!(e.contains("out of range"), "{code}: {e}");
    }
}

#[test]
fn script_sim_index_out_of_range_is_an_error() {
    let mut env = env();
    env.eval_str("net.add_neuron(0, 0, 0); sim.run(1.0)")
        .unwrap();
    for code in [
        "sim.v(9)",
        "sim.trace_of([9])",
        "sim.spike_counts_of(10, [9])",
        "sim.set_external(9, 1.0)",
    ] {
        let e = env.eval_str(code).expect_err(code);
        assert!(e.contains("out of range"), "{code}: {e}");
    }
}

#[test]
fn net_load_failure_is_an_error_not_an_empty_net() {
    let mut env = env();
    assert!(env.eval_str("net_load(\"/nonexistent/x.ne2\")").is_err());
    assert!(env.eval_str("net.load(\"/nonexistent/x.ne2\")").is_err());
}
