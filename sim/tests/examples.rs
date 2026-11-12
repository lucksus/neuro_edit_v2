//! Stage 13 H: every bundled example must eval headless without error,
//! and the LSM demo must print a readout accuracy above chance.

use std::cell::RefCell;
use std::rc::Rc;

use net_format::Network;
use sim::script::Env;
use sim::BackendKind;

const EXAMPLES: &[(&str, &str)] = &[
    (
        "first_spikes",
        include_str!("../../examples/first_spikes.rhai"),
    ),
    ("grid_wave", include_str!("../../examples/grid_wave.rhai")),
    (
        "lsm_readout",
        include_str!("../../examples/lsm_readout.rhai"),
    ),
];

fn run(code: &str) -> (Result<rhai::Dynamic, String>, Vec<String>) {
    let out = Rc::new(RefCell::new(Vec::new()));
    let o = out.clone();
    let mut env = Env::with_print(
        Network::new(),
        BackendKind::Cpu,
        1,
        Rc::new(move |s: &str| o.borrow_mut().push(s.to_string())),
    );
    let r = env.eval_str(code);
    drop(env); // releases the Rc inside on_print
    (r, Rc::try_unwrap(out).unwrap().into_inner())
}

#[test]
fn examples_eval_without_error() {
    for (name, src) in EXAMPLES {
        let (r, _) = run(src);
        assert!(r.is_ok(), "{name}: {}", r.err().unwrap_or_default());
    }
}

#[test]
fn first_spikes_actually_spikes() {
    let (r, out) = run(EXAMPLES[0].1);
    assert!(r.is_ok(), "{}", r.err().unwrap_or_default());
    // "spikes so far: N" — the demo must show activity.
    let line = out
        .iter()
        .find(|l| l.contains("spikes so far"))
        .expect("spike print");
    let n: f64 = line.rsplit(' ').next().unwrap().parse().unwrap();
    assert!(n > 0.0, "first_spikes produced no spikes: {out:?}");
}

#[test]
fn lsm_readout_beats_chance() {
    let (r, out) = run(EXAMPLES[2].1);
    assert!(r.is_ok(), "{}", r.err().unwrap_or_default());
    let line = out
        .iter()
        .find(|l| l.contains("readout accuracy"))
        .expect("accuracy print");
    let acc: f64 = line.rsplit(' ').next().unwrap().parse().unwrap();
    assert!(acc > 0.55, "readout at chance: {acc} ({out:?})");
}
