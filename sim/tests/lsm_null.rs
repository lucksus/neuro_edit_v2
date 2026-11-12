//! Regression test for the "below-chance shuffled-label control".
//!
//! The shuffled control in examples/lsm_hidden.rhai used to retrain the
//! readouts on ONE random label permutation and report the held-out AUC.
//! On this pipeline that statistic is a draw from a *wide* null
//! distribution (sd ≈ 0.2 over permutations): with ~500 features and a
//! strong planted signal, a readout fit on permuted labels still learns a
//! large-magnitude weight vector whose projection onto the true class
//! direction is randomly positive or negative. Single draws land far below
//! 0.5 with high probability (seeds 7/42 gave ridge AUC 0.28/0.32 — not a
//! bias bug, just the left tail of a too-coarse control).
//!
//! The correct permutation null averages over many permutations; the mean
//! over all permutations is exactly 0.5 by complement symmetry
//! (w(σ̄) = −w(σ) ⇒ scores negate ⇒ AUC → 1−AUC).
//!
//! This test mirrors the Rhai example's pipeline natively at reduced scale
//! (5³ liquid, 120/60 windows — same RNG consumption order), reproduces the
//! single-draw failure, and verifies the fixed (multi-permutation) control.

use net_format::builders;
use net_format::{Network, Rng};
use sim::cpu::CpuSim;
use sim::readout::{accuracy, auc, fit_ridge, ridge_output};
use sim::script::RngStream;
use sim::SimConfig;

const SEED: u64 = 1;
const N_SRC: usize = 8;
const P: f32 = 0.5;
const WIN_MS: usize = 50;
const SPB: usize = 10;
const STRENGTH: f32 = 0.8;
const EPOCHS: usize = 10;
const LR: f32 = 0.5;
const RIDGE_L: f32 = 1.0;
const LIQ_SIDE: usize = 5;
const LIQ_SPACING: f32 = 60.0;
const CONN_SCALING: f64 = 0.35;
const CONN_LAMBDA: f64 = 90.0;
const W_REC: f32 = 3.0;
const W_IN: f32 = 12.0;
const SHARED_PER_PAIR: usize = 24;
const PRIV_PER_SRC: usize = 12;
const TRAIN_N: usize = 120;
const TEST_N: usize = 60;
const N_PERM: usize = 60;

const PAIRS: [(usize, usize); 4] = [(0, 1), (2, 3), (4, 5), (6, 7)];

/// Same construction order as lsm_hidden.rhai (the global rng draws must
/// line up for the numbers to match the script's).
fn build_liquid(rng: &mut Rng) -> (Network, usize) {
    let mut net = Network::new();
    builders::create_3d_grid(
        &mut net,
        "liquid",
        LIQ_SIDE,
        LIQ_SIDE,
        LIQ_SIDE,
        LIQ_SPACING,
        [0.0; 3],
    );
    builders::set_exc_inh(&mut net, "liquid", 0.2, rng);
    builders::create_connections(&mut net, "liquid", CONN_SCALING, CONN_LAMBDA, W_REC, rng);
    let liq = net.group("liquid").unwrap().members.clone();
    let n_liq = liq.len();
    net.add_readout("ro", liq.clone());
    for p in PAIRS {
        let pool: Vec<u32> = (0..SHARED_PER_PAIR)
            .map(|_| liq[rng.uniform(0.0, 1.0e9) as usize % n_liq])
            .collect();
        for s in [p.0, p.1] {
            let mut tgts: Vec<(u32, f32)> = pool.iter().map(|&n| (n, W_IN)).collect();
            for _ in 0..PRIV_PER_SRC {
                tgts.push((liq[rng.uniform(0.0, 1.0e9) as usize % n_liq], W_IN));
            }
            net.add_input(&format!("src{s}"), tgts, [0.0; 3]);
        }
    }
    (net, n_liq)
}

/// One labelled window through the liquid → per-neuron rates (Hz × WIN_MS
/// as in the script: counts / WIN_MS).
fn play_window(sim: &mut CpuSim, net: &Network, m: &[Vec<f32>]) -> Vec<f32> {
    sim.reset_state();
    for s in 0..N_SRC {
        let mut series = Vec::with_capacity(WIN_MS * SPB);
        for row in m.iter().take(WIN_MS) {
            for _ in 0..SPB {
                series.push(row[s]);
            }
        }
        let idx = net.input_index(&format!("src{s}")).unwrap();
        sim.play_input(idx, series);
    }
    sim.run_steps(WIN_MS * SPB);
    sim.spike_counts(WIN_MS * SPB)
        .unwrap()
        .iter()
        .map(|&c| c as f32 / WIN_MS as f32)
        .collect()
}

/// Raw-stream features: per-source bit counts.
fn counts_of(m: &[Vec<f32>]) -> Vec<f32> {
    let mut c = [0.0f32; N_SRC];
    for row in m.iter().take(WIN_MS) {
        for (s, v) in row.iter().enumerate().take(N_SRC) {
            c[s] += *v;
        }
    }
    c.iter().map(|v| v / WIN_MS as f32).collect()
}

fn shuffle(labels: &[f32], rng: &mut Rng) -> Vec<f32> {
    let mut l = labels.to_vec();
    for i in (1..l.len()).rev() {
        let j = rng.uniform(0.0, 1.0e9) as usize % (i + 1);
        l.swap(i, j);
    }
    l
}

/// Feature passes for one experiment variant. `raw` swaps liquid features
/// for per-source bit counts (no sim needed but we keep the signature).
fn gen_features(
    sim: &mut CpuSim,
    net: &Network,
    strength: f32,
    raw: bool,
) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<u8>) {
    let mut gen_tr = RngStream::new(SEED + 11, N_SRC, P);
    let mut gen_te = RngStream::new(SEED + 77, N_SRC, P);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for it in 0..TRAIN_N {
        let label = (it % 2) as i64;
        let m = gen_tr.window(WIN_MS, label != 0, &PAIRS, strength);
        let f = if raw {
            counts_of(&m)
        } else {
            play_window(sim, net, &m)
        };
        x.extend_from_slice(&f);
        y.push(label as f32);
    }
    let mut xte = Vec::new();
    let mut labs = Vec::new();
    for it in 0..TEST_N {
        let label = (it % 2) as i64;
        let m = gen_te.window(WIN_MS, label != 0, &PAIRS, strength);
        let f = if raw {
            counts_of(&m)
        } else {
            play_window(sim, net, &m)
        };
        xte.push(f);
        labs.push(label as u8);
    }
    (x, y, xte, labs)
}

fn ridge_scores(x: &[f32], n_feat: usize, ytr: &[f32], xte: &[Vec<f32>]) -> Vec<f64> {
    let w = fit_ridge(x, TRAIN_N, n_feat, ytr, RIDGE_L);
    xte.iter().map(|f| ridge_output(&w, f)).collect()
}

fn delta_scores(
    x: &[f32],
    n_feat: usize,
    ytr: &[f32],
    xte: &[Vec<f32>],
    rng: &mut Rng,
) -> Vec<f64> {
    let mut ro = sim::readout::DeltaReadout::new(n_feat);
    ro.init_weights_random(-0.02, 0.02, rng);
    ro.proper = true;
    for _ in 0..EPOCHS {
        for it in 0..TRAIN_N {
            ro.learn(&x[it * n_feat..(it + 1) * n_feat], ytr[it], LR);
        }
    }
    xte.iter().map(|f| ro.output(f) as f64).collect()
}

#[test]
fn lsm_null_is_wide_single_draw_is_not_the_null() {
    let mut rng = Rng::new(SEED);
    let (net, n_liq) = build_liquid(&mut rng);
    let mut sim = CpuSim::new(
        &net,
        SimConfig {
            record_spikes: true,
            ..Default::default()
        },
    );

    // main / signal-off / raw — same windows as the script (strength 0.8
    // for main+raw, 0.0 for off), generated in the script's order.
    let (x, y, xte, labs) = gen_features(&mut sim, &net, STRENGTH, false);
    let (x0, y0, xte0, labs0) = gen_features(&mut sim, &net, 0.0, false);
    let (xr, yr, xter, labsr) = gen_features(&mut sim, &net, STRENGTH, true);

    // global-rng bookkeeping to stay aligned with the script: each
    // experiment call consumed 513 draws in init_random before the shuffle.
    let burn = |rng: &mut Rng| {
        let mut d = sim::readout::DeltaReadout::new(n_liq);
        d.init_weights_random(-0.02, 0.02, rng);
    };

    burn(&mut rng); // main's init_random
    let sr = ridge_scores(&x, n_liq, &y, &xte);
    let main_auc = auc(&sr, &labs);
    burn(&mut rng); // signal off's init_random
    let s0 = ridge_scores(&x0, n_liq, &y0, &xte0);
    let off_auc = auc(&s0, &labs0);
    // raw baseline (its own init_random in the script — comes after
    // shuffled's, but consumption order only matters for the perm draw).
    let sraw = ridge_scores(&xr, N_SRC, &yr, &xter);
    let raw_auc = auc(&sraw, &labsr);

    burn(&mut rng); // shuffled experiment's init_random
    let ytr1 = shuffle(&y, &mut rng);
    let s1 = ridge_scores(&x, n_liq, &ytr1, &xte);
    let single_perm_auc = auc(&s1, &labs);

    // the actual liquid separates the classes
    assert!(
        main_auc > 0.7,
        "main ridge AUC too low at this scale: {main_auc}"
    );
    // the bug: a single permutation draw lands far below chance
    assert!(
        single_perm_auc < 0.45,
        "expected the single-draw control to fail; got {single_perm_auc}"
    );
    assert!(
        (0.3..=0.7).contains(&off_auc),
        "signal-off AUC out of range: {off_auc}"
    );
    assert!(
        (0.3..=0.7).contains(&raw_auc),
        "raw-baseline AUC out of range: {raw_auc}"
    );

    // The fixed control: mean over many permutations must sit at 0.5.
    let mut perms = Vec::with_capacity(N_PERM);
    for _ in 0..N_PERM {
        let ytr = shuffle(&y, &mut rng);
        let s = ridge_scores(&x, n_liq, &ytr, &xte);
        perms.push(auc(&s, &labs));
    }
    let mean = perms.iter().sum::<f64>() / N_PERM as f64;
    let sd = (perms.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / N_PERM as f64).sqrt();
    eprintln!("main={main_auc:.3} off={off_auc:.3} raw={raw_auc:.3}");
    eprintln!(
        "single-perm={single_perm_auc:.3}  null mean={mean:.3} sd={sd:.3} range=[{:.3},{:.3}]",
        perms.iter().cloned().fold(f64::INFINITY, f64::min),
        perms.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    assert!(
        sd > 0.08,
        "permutation null should be wide at this scale; sd={sd}"
    );
    assert!(
        (0.42..=0.58).contains(&mean),
        "permutation-null mean must be ~0.5; got {mean}"
    );

    // delta readout null behaves the same way
    let mut dperms = Vec::new();
    for _ in 0..20 {
        let ytr = shuffle(&y, &mut rng);
        let s = delta_scores(&x, n_liq, &ytr, &xte, &mut rng);
        dperms.push(auc(&s, &labs));
    }
    let dmean = dperms.iter().sum::<f64>() / dperms.len() as f64;
    assert!(
        (0.4..=0.6).contains(&dmean),
        "delta permutation-null mean must be ~0.5; got {dmean}"
    );
}

/// Accuracy at a 0.5 threshold should track AUC for the shuffled control
/// (no threshold tuning happens anywhere).
#[test]
fn accuracy_threshold_is_fixed() {
    let scores = [0.1, 0.4, 0.6, 0.9];
    let labels = [0u8, 0, 1, 1];
    assert_eq!(accuracy(&scores, &labels), 1.0);
}
