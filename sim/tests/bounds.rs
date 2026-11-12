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

#[test]
fn script_unknown_input_name_is_an_error() {
    let mut env = env();
    env.eval_str("net.add_neuron(0, 0, 0); sim.run(1.0)")
        .unwrap();
    for code in [
        "sim.set_input(\"nope\", 1.0)",
        "sim.play_input(\"nope\", [1.0])",
    ] {
        let e = env.eval_str(code).expect_err(code);
        assert!(e.contains("unknown input"), "{code}: {e}");
    }
    env.eval_str("net.add_input(\"in\", [[0, 1.0]])").unwrap();
    env.eval_str("sim.set_input(\"in\", 1.0)").unwrap();
}

#[test]
fn script_net_mutations_bump_net_version() {
    let mut env = env();
    env.eval_str("net.add_neuron(0, 0, 0)").unwrap();
    let v0 = env.net_version.get();
    env.eval_str("net.set_neuron_name(0, \"x\")").unwrap();
    let v1 = env.net_version.get();
    assert!(v1 > v0, "set_neuron_name left net_version at {v0}");
    env.eval_str("net.add_to_group(\"g\", [0])").unwrap();
    assert!(
        env.net_version.get() > v1,
        "add_to_group left net_version at {v1}"
    );
}

#[test]
fn script_set_syn_tau_one_rejects_negative_ids() {
    let mut env = env();
    env.eval_str("net.add_neuron(0, 0, 0); net.add_neuron(1, 0, 0); net.connect(0, 1, 1.0)")
        .unwrap();
    for code in [
        "net.set_syn_tau_one(-1, 0, 5.0)",
        "net.set_syn_tau_one(0, -1, 5.0)",
        "net.set_syn_tau_one(0, 99, 5.0)",
    ] {
        let e = env.eval_str(code).expect_err(code);
        assert!(e.contains("out of range"), "{code}: {e}");
    }
}

#[test]
fn script_fit_ridge_rejects_short_arrays() {
    let mut env = env();
    let e = env
        .eval_str("fit_ridge([1.0, 2.0], 4, 2, [0, 1, 0, 1], 0.1)")
        .expect_err("short x");
    assert!(e.contains("T*F"), "{e}");
    let e = env
        .eval_str("fit_ridge([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 4, 2, [0, 1], 0.1)")
        .expect_err("short y");
    assert!(e.contains("T = 4"), "{e}");
    env.eval_str(
        "fit_ridge([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 4, 2, [0.0, 0.0, 0.0, 0.0], 0.1)",
    )
    .unwrap();
}

#[test]
fn script_file_errors_reach_the_script() {
    let mut env = env();
    let e = env
        .eval_str("net.samples_save(\"nope\", \"x.txt\")")
        .expect_err("unknown samples");
    assert!(e.contains("unknown samples"), "{e}");
    // No data dir: paths pass through — so aim at a directory that
    // cannot exist and check the io error becomes a script error.
    let e = env
        .eval_str("write_to_file(\"/nonexistent_dir_xyz/f.txt\", [1.0])")
        .expect_err("unwritable path");
    assert!(e.contains("write_to_file"), "{e}");
    let e = env
        .eval_str("net.samples_load(\"s\", \"/nonexistent_dir_xyz/f.txt\")")
        .expect_err("missing file");
    assert!(e.contains("samples_load"), "{e}");
}
