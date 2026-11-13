//! Stage 13 H: every bundled example must eval headless without error,
//! and each one must show what its header comment claims.
//!
//! Stage 24: every example runs at reduced scale via `quick()` (the
//! `NEURO_QUICK` cell — tests set it on the Env; scripts read it and
//! shrink sizes/splits). For each example the test names the sabotage
//! that would make it fail.

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
    ("lsm_big", include_str!("../../examples/lsm_big.rhai")),
    (
        "lsm_big_off",
        include_str!("../../examples/lsm_big_off.rhai"),
    ),
    ("lsm_adex", include_str!("../../examples/lsm_adex.rhai")),
    ("lsm_hidden", include_str!("../../examples/lsm_hidden.rhai")),
    ("detector", include_str!("../../examples/detector.rhai")),
    (
        "memory_window",
        include_str!("../../examples/memory_window.rhai"),
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

/// Eval on `backend` at QUICK scale; `record_steps` sizes the spike
/// ring so `spike_row` scans reach back over the script's whole run.
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
    env.quick.set(true); // Stage 24: examples shrink under quick()
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

/// Parse "key<ws>=<ws><num>" out of a printed line (spaces tolerated).
fn metric(line: &str, key: &str) -> f64 {
    let i = line
        .find(key)
        .unwrap_or_else(|| panic!("{line} lacks {key}"));
    let rest = &line[i + key.len()..];
    let rest = rest.trim_start_matches(|c: char| c == ' ' || c == '=');
    rest.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .next()
        .unwrap()
        .parse()
        .unwrap_or_else(|_| panic!("{line}: bad {key} number"))
}

fn line_with<'a>(out: &'a [String], needle: &str) -> &'a str {
    out.iter()
        .find(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("missing line containing {needle:?}: {out:?}"))
}

/// Every example evals without error on every reachable backend.
/// Sabotage: delete the `quick()` branch in any example (full-size
/// `lsm_big` would still pass but take minutes — the assertion here is
/// `res.is_ok()`, so the real guard is the alive/auc checks below).
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

/// Activity without edits: every example that claims to "show activity"
/// leaves a spiking sim behind — the "alive:"/final print carries the
/// spike count and the live sim still has recorded spikes.
/// Sabotage: removing the noise tail from a script makes its sim dead
/// on arrival — `alive: spikes=0` and this fails.
#[test]
fn examples_leave_a_spiking_sim() {
    for (name, src) in EXAMPLES {
        let run = eval_env(src, BackendKind::Cpu, 4096);
        assert!(
            run.res.is_ok(),
            "{name}: {}",
            run.res.err().unwrap_or_default()
        );
        let out = run.out.borrow();
        // evidence of firing, either printed by the script or recorded
        // in the live sim's spike counts.
        let printed = out.iter().any(|l| {
            (l.contains("alive: spikes=") || l.contains("spikes=")) && metric(l, "spikes") > 0.0
        });
        let recorded = run
            .env
            .existing_sim()
            .map(|h| {
                h.borrow()
                    .spike_counts(usize::MAX)
                    .map(|c| c.iter().map(|&v| v as f64).sum::<f64>())
                    .unwrap_or(0.0)
            })
            .unwrap_or(0.0);
        assert!(
            printed || recorded > 0.0,
            "{name}: no spikes anywhere (printed={printed}, recorded={recorded}): {out:?}"
        );
    }
}

#[test]
fn first_spikes_actually_spikes() {
    let run = eval_env(EXAMPLES[0].1, BackendKind::Cpu, 512);
    assert!(run.res.is_ok(), "{}", run.res.err().unwrap_or_default());
    let out = run.out.borrow();
    let line = line_with(&out, "spikes so far");
    let n: f64 = line.rsplit(' ').next().unwrap().parse().unwrap();
    assert!(n > 0.0, "first_spikes produced no spikes: {out:?}");
}

/// The comment: "Chain them so spikes propagate left to right." — every
/// neuron 0..4 must spike, and their first spikes must land in chain order.
/// Sabotage: drop `net.connect(i, i+1, 30)` → counts[>0]==0 fails.
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

/// The comment: "a 20x20 sheet … activity rolls across the plane."
/// Sabotage: `create_connections(sheet, 0.0, …)` empties the sheet →
/// the synapse-count assert fails; zero kick → right-half assert fails.
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

/// Held-out metrics common to all three LSM variants.
struct LsmOut {
    mlp_auc: f64,
    ridge_auc: f64,
    raw_auc: f64,
    alive_spikes: f64,
}

fn parse_lsm(out: &[String]) -> LsmOut {
    LsmOut {
        mlp_auc: metric(line_with(out, "test mlp"), "auc"),
        ridge_auc: metric(line_with(out, "test ridge"), "auc"),
        raw_auc: metric(line_with(out, "test raw"), "auc"),
        alive_spikes: metric(line_with(out, "alive: spikes"), "spikes"),
    }
}

/// lsm_readout (216 neurons): the comment claims the MLP detects the
/// hidden correlation above chance on held-out windows while the raw
/// baseline cannot.
/// Sabotage: `STRENGTH = 0.0` collapses mlp_auc to ~0.5; `W_IN = 0`
/// kills the drive → alive_spikes == 0.
#[test]
fn lsm_readout_mlp_beats_chance() {
    for b in backends() {
        let run = eval_env(EXAMPLES[2].1, b, 512);
        assert!(
            run.res.is_ok(),
            "{b:?}: {} ({:?})",
            run.res.err().unwrap_or_default(),
            run.out.borrow()
        );
        assert!(
            run.env.net.borrow().synapses.len() > 200,
            "{b:?}: \"reservoir\" is disconnected"
        );
        let o = parse_lsm(&run.out.borrow());
        assert!(
            o.alive_spikes > 0.0,
            "{b:?}: no activity on the probe window"
        );
        assert!(o.mlp_auc > 0.6, "{b:?}: mlp at chance: {}", o.mlp_auc);
        assert!(
            o.raw_auc < 0.75,
            "{b:?}: raw baseline should sit near chance: {}",
            o.raw_auc
        );
        // the trained MLP is stored on the readout (file persistence)
        assert!(
            run.env.net.borrow().readout_mlp("ro").is_some(),
            "{b:?}: readout \"ro\" lost its mlp"
        );
    }
}

/// lsm_big (4096 → 512 under quick): the same claims, plus the
/// coincidence baseline is allowed to win — we only pin the liquid's
/// side of the contract: mlp above chance, raw at chance.
/// Sabotage: `STRENGTH = 0` → mlp_auc ~0.5; dropping `set_noise` →
/// the post-eval sim is dead.
#[test]
fn lsm_big_mlp_beats_chance_on_heldout() {
    for b in backends() {
        let run = eval_env(EXAMPLES[3].1, b, 2048);
        assert!(
            run.res.is_ok(),
            "{b:?}: {} ({:?})",
            run.res.err().unwrap_or_default(),
            run.out.borrow()
        );
        let o = parse_lsm(&run.out.borrow());
        assert!(o.alive_spikes > 0.0, "{b:?}: dead liquid");
        assert!(
            o.mlp_auc > 0.65,
            "{b:?}: big-liquid mlp at chance: {}",
            o.mlp_auc
        );
        assert!(
            (o.raw_auc - 0.5).abs() < 0.2,
            "{b:?}: raw baseline should sit near chance: {}",
            o.raw_auc
        );
        assert!(
            run.env.net.borrow().readout_mlp("ro").is_some(),
            "{b:?}: readout \"ro\" lost its mlp"
        );
    }
}

/// lsm_big_off is the negative control: identical pipeline, STRENGTH=0.
/// EVERY reported AUC — mlp included — must sit at chance; a high mlp
/// AUC here means the pipeline leaks the label, which is exactly what
/// this control exists to catch.
/// Sabotage: any label leak (e.g. label mixed into features, or the
/// test set trained on) pushes mlp_auc over 0.7 and fails this.
#[test]
fn lsm_big_off_stays_at_chance() {
    for b in backends() {
        let run = eval_env(EXAMPLES[4].1, b, 2048);
        assert!(
            run.res.is_ok(),
            "{b:?}: {} ({:?})",
            run.res.err().unwrap_or_default(),
            run.out.borrow()
        );
        let o = parse_lsm(&run.out.borrow());
        assert!(o.alive_spikes > 0.0, "{b:?}: dead liquid");
        for (k, v) in [
            ("mlp", o.mlp_auc),
            ("ridge", o.ridge_auc),
            ("raw", o.raw_auc),
        ] {
            assert!(
                (v - 0.5).abs() < 0.2,
                "{b:?}: signal-off {k} auc {v} — the control found a signal"
            );
        }
    }
}

/// lsm_adex: same task on an AdEx reservoir — the variant must also
/// carry the signal above chance (it may be weaker than izh; the
/// threshold is lower).
/// Sabotage: `W_IN = 0` or `def_type` swapped for a silent model →
/// alive_spikes == 0; `STRENGTH = 0` → mlp_auc ~0.5.
#[test]
fn lsm_adex_carries_the_signal() {
    for b in backends() {
        let run = eval_env(EXAMPLES[5].1, b, 2048);
        assert!(
            run.res.is_ok(),
            "{b:?}: {} ({:?})",
            run.res.err().unwrap_or_default(),
            run.out.borrow()
        );
        let o = parse_lsm(&run.out.borrow());
        assert!(o.alive_spikes > 0.0, "{b:?}: dead AdEx liquid");
        assert!(
            o.mlp_auc > 0.6,
            "{b:?}: AdEx-liquid mlp at chance: {}",
            o.mlp_auc
        );
    }
}

/// lsm_hidden at quick scale: the honest-eval contract — the liquid's
/// summary line must beat chance while the signal-off summary sits at
/// chance, and the cooc baseline is allowed (expected!) to win.
/// Sabotage: `STRENGTH = 0` → main ~0.5; a leaked label → off > 0.7.
#[test]
fn lsm_hidden_honest_summary() {
    let run = eval_env(EXAMPLES[6].1, BackendKind::Cpu, 512);
    assert!(run.res.is_ok(), "{}", run.res.err().unwrap_or_default());
    let out = run.out.borrow();
    let ridge_line = |prefix: &str| {
        out.iter()
            .find(|l| l.starts_with(prefix) && l.contains("ridge_auc"))
            .unwrap_or_else(|| panic!("no {prefix} ridge_auc line: {out:?}"))
    };
    let main = metric(ridge_line("main"), "ridge_auc");
    let off = metric(ridge_line("signal off"), "ridge_auc");
    assert!(main > 0.6, "lsm_hidden main ridge_auc at chance: {main}");
    assert!(off < 0.7, "lsm_hidden signal-off found a signal: {off}");
}

/// detector + memory_window at quick scale: they must eval and stream
/// their result rows (job:/horizon= lines). These are job-style
/// experiments — quick mode only proves the pipeline runs and reports.
/// Sabotage: deleting the dataset/drive leaves the prints absent.
#[test]
fn job_examples_report() {
    for (i, needle) in [(7usize, "liquid"), (8usize, "horizon=")] {
        let run = eval_env(EXAMPLES[i].1, BackendKind::Cpu, 1024);
        assert!(
            run.res.is_ok(),
            "{}: {}",
            EXAMPLES[i].0,
            run.res.err().unwrap_or_default()
        );
        let out = run.out.borrow();
        assert!(
            out.iter().any(|l| l.contains(needle)),
            "{}: no {needle} line: {out:?}",
            EXAMPLES[i].0
        );
    }
}
