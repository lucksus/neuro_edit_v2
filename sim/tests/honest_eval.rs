//! Regression test for the honest LSM evaluation (P2-4).
//!
//! The eval in examples/lsm_hidden.rhai reports, next to the liquid result:
//!   - a permutation test: >=500 label shuffles, fraction of null draws
//!     reaching the observed ridge AUC = p-value;
//!   - a 28-feature pairwise-coincidence baseline (equal-bit counts per
//!     source pair) — which the reviewer showed beats the liquid on this
//!     synthetic task;
//!   - a held-out validation split.
//! This test locks in the machinery at reduced scale: the cooc baseline
//! must detect the planted pairs almost perfectly, and the permutation
//! null must (a) sit at ~0.5 and (b) put almost no mass at/above the
//! observed AUC, i.e. a small p-value — while signal-off data yields a
//! p-value consistent with chance.

use net_format::Rng;
use sim::readout::{auc, fit_ridge, ridge_output};
use sim::script::RngStream;

const N_SRC: usize = 8;
const WIN: usize = 50;
const PAIRS: [(usize, usize); 4] = [(0, 1), (2, 3), (4, 5), (6, 7)];

/// C(8,2)=28 features: fraction of window steps where sources i,j agree.
fn cooc(m: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::with_capacity(28);
    for i in 0..N_SRC {
        for j in (i + 1)..N_SRC {
            let c = m.iter().filter(|r| r[i] == r[j]).count();
            out.push(c as f32 / WIN as f32);
        }
    }
    out
}

fn shuffle(labels: &[f32], rng: &mut Rng) -> Vec<f32> {
    let mut l = labels.to_vec();
    for i in (1..l.len()).rev() {
        let j = (rng.f64() * (i + 1) as f64) as usize;
        l.swap(i, j);
    }
    l
}

/// Generate a train/test feature set for a mode at a strength, like the
/// script's gen_set (same rng_stream offsets).
fn gen(
    seed: u64,
    strength: f32,
    feat: &dyn Fn(&[Vec<f32>]) -> Vec<f32>,
) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<u8>) {
    let mut gtr = RngStream::new(seed + 11, N_SRC, 0.5);
    let mut gte = RngStream::new(seed + 77, N_SRC, 0.5);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for it in 0..200usize {
        let l = it % 2;
        x.extend(feat(&gtr.window(WIN, l == 1, &PAIRS, strength)));
        y.push(l as f32);
    }
    let mut xte = Vec::new();
    let mut labs = Vec::new();
    for it in 0..120usize {
        let l = it % 2;
        xte.push(feat(&gte.window(WIN, l == 1, &PAIRS, strength)));
        labs.push(l as u8);
    }
    (x, y, xte, labs)
}

fn ridge_auc(x: &[f32], y: &[f32], f: usize, xte: &[Vec<f32>], labs: &[u8]) -> f64 {
    let w = fit_ridge(x, y.len(), f, y, 1.0);
    let s: Vec<f64> = xte.iter().map(|r| ridge_output(&w, r)).collect();
    auc(&s, labs)
}

/// The trivial baseline the liquid must be compared against: pairwise
/// coincidence counts detect the planted pairs essentially perfectly at
/// the strengths the script sweeps.
#[test]
fn coincidence_baseline_detects_planted_pairs() {
    for (strength, min_auc) in [(0.5f32, 0.9), (0.8, 0.9), (1.0, 0.9)] {
        let (x, y, xte, labs) = gen(1, strength, &cooc);
        let a = ridge_auc(&x, &y, 28, &xte, &labs);
        assert!(
            a > min_auc,
            "cooc baseline should nail strength {strength}: auc={a:.3}"
        );
    }
    // and stays at chance with no planted signal
    let (x, y, xte, labs) = gen(1, 0.0, &cooc);
    let a = ridge_auc(&x, &y, 28, &xte, &labs);
    assert!(
        (0.35..=0.65).contains(&a),
        "cooc at strength 0 should sit near chance: auc={a:.3}"
    );
}

/// The permutation test: refit the ridge on permuted labels and count how
/// many draws reach the observed AUC. With real signal that fraction is a
/// small p-value; with strength-0 features it is unremarkable.
#[test]
fn permutation_test_gives_small_p_only_with_signal() {
    const PERMS: usize = 100;
    let mut rng = Rng::new(99);

    for (strength, want_small) in [(0.8f32, true), (0.0, false)] {
        let (x, y, xte, labs) = gen(1, strength, &cooc);
        let observed = ridge_auc(&x, &y, 28, &xte, &labs);
        let mut above = 0usize;
        let mut sum = 0.0;
        let mut sq = 0.0;
        for _ in 0..PERMS {
            let a = ridge_auc(&x, &shuffle(&y, &mut rng), 28, &xte, &labs);
            sum += a;
            sq += a * a;
            if a >= observed {
                above += 1;
            }
        }
        let mean = sum / PERMS as f64;
        let sd = (sq / PERMS as f64 - mean * mean).sqrt();
        let p = (1 + above) as f64 / (PERMS + 1) as f64;
        eprintln!("strength {strength}: observed={observed:.3} null={mean:.3}±{sd:.3} p={p:.3}");
        assert!(
            (0.4..=0.6).contains(&mean),
            "permutation null should centre at 0.5: mean={mean:.3}"
        );
        if want_small {
            assert!(
                p < 0.05,
                "with signal the observed auc should beat almost all null draws: p={p:.3}"
            );
        } else {
            assert!(
                p > 0.05,
                "without signal the observed auc should be unremarkable: p={p:.3}"
            );
        }
    }
}
