//! Stage 13 F: console help — `help`, `help()` and `help("topic")`.

use net_format::Network;
use sim::script::Env;
use sim::BackendKind;

fn env() -> Env {
    Env::new(Network::new(), BackendKind::Cpu, 1).unwrap()
}

#[test]
fn bare_help_evaluates_to_overview() {
    // No parens: `help` is a scope binding holding the overview text —
    // before Stage 13 this was a rhai "variable not found" error.
    let r = env().eval_str("help").unwrap();
    let s = r.to_string();
    assert!(s.contains("run(ms)"), "{s}");
    assert!(s.contains("add_neuron"), "{s}");
}

#[test]
fn help_parens_returns_overview() {
    let r = env().eval_str("help()").unwrap();
    let s = r.to_string();
    assert!(s.contains("run(ms)"), "{s}");
    // and the bare + parens forms agree
    let bare = env().eval_str("help").unwrap().to_string();
    assert_eq!(s, bare);
}

#[test]
fn help_topic_gives_a_section() {
    let s = env().eval_str("help(\"sim\")").unwrap().to_string();
    assert!(s.contains("run(ms)"), "{s}");
    assert!(
        !s.contains("add_neuron"),
        "topic slice must not be the dump"
    );
    // aliases work too
    let s2 = env().eval_str("help(\"synapse\")").unwrap().to_string();
    assert!(s2.contains("set_syn_tau"), "{s2}");
}

#[test]
fn help_unknown_topic_lists_topics() {
    let s = env().eval_str("help(\"bogus\")").unwrap().to_string();
    assert!(s.contains("unknown topic"), "{s}");
    assert!(s.contains("net"), "{s}");
    assert!(s.contains("sim"), "{s}");
}
