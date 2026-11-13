//! Stage 13 H: every bundled example must eval headless without error,
//! and the LSM demo must print a readout accuracy above chance.
//!
//! 2026-11-13 (Nico's report): the comments make *dynamical* promises that
//! an eval-check can't see — the chain must propagate, the wave must cross
//! the sheet, the "reservoir" must have synapses. These tests run each
//! script on every reachable backend and assert what the comment promises.

use std::cell::RefCell;
use std::rc::Rc;

use net_format::Network;
use sim::script::Env;
use sim::{BackendKind, Sim};

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

/// Every backend the scripts must hold up on — CPU always, CUDA when a
/// device is reachable (Nico ran the picker on CUDA).
fn backends() -> Vec<BackendKind> {
    let mut v = vec![BackendKind::Cpu];
    #[cfg(feature = "cuda")]
    if sim::cuda_available() {
        v.push(BackendKind::Cuda);
    }
    v
}

/// Result of a script run: the Env stays alive so tests can interrogate
/// the built sim; `out` collects the script's `print` lines.
struct Run {
    env: Env,
    res: Result<rhai::Dynamic, String>,
    out: Rc<RefCell<Vec<String>>>,
}

/// Eval on `backend`; `record_steps` sizes the spike ring so `spike_row`
/// scans reach back over the script's whole run.
fn eval_env(code: &str, backend: BackendKind, record_steps: usize) -> Run {
    let out = Rc::new(RefCell::new(Vec::new()));
    let o = out.clone();
    let mut env = Env::with_print(
        Network::new(),
        backend,
        1,
        Rc::new(move |s: &str| o.borrow_mut().push(s.to_string())),
    );
    env.cfg.borrow_mut().spike_record_steps = record_steps;
    let r = env.eval_str(code);
    Run { env, res: r, out }
}

/// Step index of `neuron`'s first recorded spike (needs a ring at least
/// `t` deep — `record_steps` above).
fn first_spike_step(sim: &Sim, neuron: usize) -> Option<u64> {
    let t = sim.step_count();
    for s in 0..t {
        let row = sim.spike_row((t - 1 - s) as usize)?;
        if row[neuron / 64] & (1u64 << (neuron % 64)) != 0 {
            return Some(s);
        }
    }
    None
}

#[test]
fn examples_eval_without_error() {
    for (name, src) in EXAMPLES {
        for b in backends() {
            let run = eval_env(src, b, 512);
            assert!(
                run.res.is_ok(),
                "{name} on {b:?}: {}",
                run.res.err().unwrap_or_default()
            );
        }
    }
}

#[test]
fn first_spikes_actually_spikes() {
    let run = eval_env(EXAMPLES[0].1, BackendKind::Cpu, 512);
    assert!(run.res.is_ok(), "{}", run.res.err().unwrap_or_default());
    // "spikes so far: N" — the demo must show activity.
    let out = run.out.borrow();
    let line = out
        .iter()
        .find(|l| l.contains("spikes so far"))
        .expect("spike print");
    let n: f64 = line.rsplit(' ').next().unwrap().parse().unwrap();
    assert!(n > 0.0, "first_spikes produced no spikes: {out:?}");
}

/// The comment: "Chain them so spikes propagate left to right." — every
/// neuron 0..4 must spike, and their first spikes must land in chain order.
#[test]
fn first_spikes_propagates_along_chain() {
    for b in backends() {
        let run = eval_env(EXAMPLES[0].1, b, 30_000);
        assert!(
            run.res.is_ok(),
            "{b:?}: {} ({:?})",
            run.res.err().unwrap_or_default(),
            run.out.borrow()
        );
        let h = run.env.existing_sim().expect("script builds a sim");
        let s = h.borrow();
        let counts = s.spike_counts(usize::MAX).unwrap();
        assert_eq!(counts.len(), 5);
        for i in 0..5 {
            assert!(
                counts[i] > 0,
                "{b:?}: neuron {i} never spiked — chain broken at {i}: {counts:?}"
            );
        }
        let firsts: Vec<u64> = (0..5)
            .map(|i| first_spike_step(&s, i).expect("spiked above"))
            .collect();
        assert!(
            firsts.windows(2).all(|w| w[0] < w[1]),
            "{b:?}: first spikes not in chain order: {firsts:?}"
        );
    }
}

/// The comment: "a 20x20 sheet with distance-decaying random connections,
/// kicked on the left edge — activity rolls across the plane." The sheet
/// must actually be connected, and activity must reach the right half.
#[test]
fn grid_wave_reaches_right_half() {
    for b in backends() {
        let run = eval_env(EXAMPLES[1].1, b, 35_000);
        assert!(
            run.res.is_ok(),
            "{b:?}: {} ({:?})",
            run.res.err().unwrap_or_default(),
            run.out.borrow()
        );
        let n_syn = run.env.net.borrow().synapses.len();
        assert!(
            n_syn > 500,
            "{b:?}: sheet has {n_syn} synapses — the \"distance-decaying random \
             connections\" promised by the comment don't exist"
        );
        let h = run.env.existing_sim().expect("script builds a sim");
        let s = h.borrow();
        let counts = s.spike_counts(usize::MAX).unwrap();
        assert_eq!(counts.len(), 400);
        // lattice: member k sits at (i0=k/20 along +z, i1=k%20); the kick
        // drives i0=0. "Right half" = i0 >= 10  <=>  k >= 200.
        let active = |lo: usize, hi: usize| (lo..hi).filter(|&k| counts[k] > 0).count();
        let (left, right) = (active(0, 200), active(200, 400));
        assert!(
            right >= 100,
            "{b:?}: only {right}/200 right-half neurons spiked — the wave \
             never crossed the sheet (left {left})"
        );
        // "rolls across": the first spike on the far edge must come after
        // the driven column's first spike.
        let col_first = |i0: usize| -> Option<u64> {
            (0..20)
                .filter_map(|i1| first_spike_step(&s, i0 * 20 + i1))
                .min()
        };
        let l = col_first(0).expect("driven column spikes");
        let r_edge = col_first(19).expect("right edge spiked");
        assert!(
            r_edge > l,
            "{b:?}: right edge fired at {r_edge}, driven column at {l} — not a wave"
        );
    }
}

/// The comment promises a "3D reservoir" — with zero synapses it's just a
/// bag of neurons. Assert connectivity exists AND the readout still beats
/// chance (the original test's promise).
#[test]
fn lsm_readout_beats_chance() {
    for b in backends() {
        let run = eval_env(EXAMPLES[2].1, b, 512);
        assert!(
            run.res.is_ok(),
            "{b:?}: {} ({:?})",
            run.res.err().unwrap_or_default(),
            run.out.borrow()
        );
        let n_syn = run.env.net.borrow().synapses.len();
        assert!(
            n_syn > 200,
            "{b:?}: \"reservoir\" has {n_syn} synapses — disconnected neurons"
        );
        let out = run.out.borrow();
        let line = out
            .iter()
            .find(|l| l.contains("readout accuracy"))
            .expect("accuracy print");
        let acc: f64 = line.rsplit(' ').next().unwrap().parse().unwrap();
        assert!(acc > 0.55, "{b:?}: readout at chance: {acc} ({out:?})");
    }
}
