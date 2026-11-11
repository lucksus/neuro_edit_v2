// Liquid-state-machine experiment: detect hidden pairwise correlations in
// random bit streams. The signal is invisible to any single stream's
// statistics (each stays Bernoulli(p)) — only coincident activity between
// paired sources carries it. The liquid turns coincidences into rate
// differences a linear readout can separate.
//
//   sim --script examples/lsm_hidden.rhai [--cuda|--cpu]
//
// Loops over SEEDS, rebuilding the liquid per seed, and prints
// accuracy/AUC for the trained readouts plus three controls:
//   signal off  — same pipeline, strength 0 (no planted correlation)
//   shuffled    — readouts retrained on permuted labels, averaged over
//                 NPERM permutation pairs (each permutation and its label
//                 complement). A single permutation is NOT a valid null:
//                 the readout interpolates 200 windows in ~500 features, so
//                 the fitted noise direction has a large random projection
//                 onto the true class axis and one draw can sit anywhere in
//                 [0,1] (measured single-draw sd ≈ 0.2 — that's what made
//                 seeds 7/42 report AUC ~0.3 before). The mean over
//                 permutations is the real null: exactly 0.5 by symmetry.
//   raw         — per-source bit counts instead of liquid features
// Summary table reports mean ± sd across seeds; the controls must sit at
// 0.5 within noise.

// ---------- parameters ----------
let SEEDS = [1, 7, 42, 2, 3, 4, 5, 6, 8, 9, 10, 11];
let N_SRC = 8;               // 5-10 sources
let P = 0.5;                 // Bernoulli bit probability
let WIN_MS = 50;             // window length
let SPB = 10;                // steps per bit (1 ms at dt=0.1)
let PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]];
let STRENGTH = 0.8;          // planted-signal strength (pairwise shared coin)
let LR = 0.5;                // delta-rule learn rate (proper rule, rate feats)
let EPOCHS = 10;             // delta-rule passes over the training set
let RIDGE_L = 1.0;
let NPERM = 20;              // permutation pairs per shuffled control (40 draws)

let LIQ_SIDE = 8;
let LIQ_SPACING = 60.0;
let CONN_SCALING = 0.35;
let CONN_LAMBDA = 90.0;
let W_REC = 3.0;
let W_IN = 12.0;             // input weight: 1 source sub-threshold, 2 spike
let SHARED_PER_PAIR = 24;    // pool neurons shared inside a pair
let PRIV_PER_SRC = 12;       // private neurons per source

let TRAIN_N = 200;
let TEST_N = 120;

// ---------- helpers (anonymous fns capture the scope above) ----------

let build_liquid = |seed_val| {
    net.clear();
    seed(seed_val);
    net.create_3d_grid("liquid", LIQ_SIDE, LIQ_SIDE, LIQ_SIDE, LIQ_SPACING);
    net.set_exc_inh("liquid", 0.2);
    net.create_connections("liquid", CONN_SCALING, CONN_LAMBDA, W_REC);
    let liq = net.neurons_of("liquid");
    let n_liq = liq.len();
    net.add_readout("ro", liq);
    // each pair shares a pool; each source also gets a private pool so all
    // streams stay equally represented individually
    for p in PAIRS {
        let pool = [];
        for k in 0..SHARED_PER_PAIR {
            pool.push(liq[rand(0.0, 1.0e9).to_int() % n_liq]);
        }
        for s in p {
            let tgts = [];
            for n in pool {
                tgts.push([n, W_IN]);
            }
            for k in 0..PRIV_PER_SRC {
                tgts.push([liq[rand(0.0, 1.0e9).to_int() % n_liq], W_IN]);
            }
            net.add_input("src" + s, tgts);
        }
    }
    n_liq
};

// Play a T x N_SRC bit window: each bit held for SPB steps (1 bit/ms).
let play_window = |m| {
    sim.reset();
    for s in 0..N_SRC {
        let series = [];
        for t in 0..WIN_MS {
            let b = m[t][s];
            for k in 0..SPB {
                series.push(b);
            }
        }
        sim.play_input("src" + s, series);
    }
    sim.run_steps(WIN_MS * SPB);
    sim.spike_counts(WIN_MS * SPB).map(|c| c / WIN_MS)
};

// Raw-stream baseline features: per-source bit counts (linear => chance).
let counts_of = |m| {
    let c = [];
    for s in 0..N_SRC {
        let a = 0.0;
        for t in 0..WIN_MS {
            a += m[t][s];
        }
        c.push(a / WIN_MS);
    }
    c
};

// Fisher–Yates permutation of the label vector.
let shuffled = |labels| {
    let l = labels;
    let n = l.len();
    for k in 0..n {
        let i = n - 1 - k;
        let j = rand(0.0, 1.0e9).to_int() % (i + 1);
        let tmp = l[i];
        l[i] = l[j];
        l[j] = tmp;
    }
    l
};

let complement = |labels| labels.map(|v| 1.0 - v);

// Generate one feature set: [X flat, Y, Xte (array of rows), labs].
let gen_set = |seed_val, strength, raw| {
    let gen_tr = rng_stream(seed_val + 11, N_SRC, P);
    let gen_te = rng_stream(seed_val + 77, N_SRC, P);
    let X = [];
    let Y = [];
    let Xte = [];
    let labs = [];
    for it in 0..TRAIN_N {
        let label = it % 2;
        let m = gen_tr.window(WIN_MS, label, PAIRS, strength);
        let f = if raw { counts_of.call(m) } else { play_window.call(m) };
        for v in f {
            X.push(v);
        }
        Y.push(label.to_float());
    }
    for it in 0..TEST_N {
        let label = it % 2;
        let m = gen_te.window(WIN_MS, label, PAIRS, strength);
        Xte.push(if raw { counts_of.call(m) } else { play_window.call(m) });
        labs.push(label);
    }
    [X, Y, Xte, labs]
};

// Fit both readouts on (X, ytr) and score Xte against labs.
// Returns [delta_acc, delta_auc, ridge_acc, ridge_auc].
let eval_fit = |X, n_feat, ytr, Xte, labs| {
    let ro = delta_new(n_feat);
    ro.init_random(-0.02, 0.02);
    ro.set_proper(1);          // corrected delta rule; quirk mode can't grow
                               // weights away from init (update ~ w*f)
    for ep in 0..EPOCHS {
        for it in 0..TRAIN_N {
            ro.learn(X.extract(it * n_feat, n_feat), ytr[it], LR);
        }
    }
    let wr = fit_ridge(X, TRAIN_N, n_feat, ytr, RIDGE_L);
    let sd = [];
    let sr = [];
    for f in Xte {
        sd.push(ro.output(f));
        sr.push(ridge_output(wr, f));
    }
    [accuracy(sd, labs), auc(sd, labs), accuracy(sr, labs), auc(sr, labs)]
};

let mean_of = |rows, k| {
    let s = 0.0;
    for r in rows {
        s += r[k];
    }
    s / rows.len()
};
let sd_of = |rows, k, m| {
    let s = 0.0;
    for r in rows {
        s += (r[k] - m) * (r[k] - m);
    }
    (s / rows.len()).sqrt()
};
let f3 = |v| (v * 1000.0).to_int() / 1000.0;

let report = |name, r| {
    print(name + "  delta: acc=" + f3.call(r[0]) + " auc=" + f3.call(r[1]) +
          "  ridge: acc=" + f3.call(r[2]) + " auc=" + f3.call(r[3]));
};

// ---------- run ----------
print("=== LSM hidden-correlation experiment ===");
print("sources=" + N_SRC + " pairs=" + PAIRS.len() + " p=" + P +
      " win=" + WIN_MS + "ms strength=" + STRENGTH +
      " train=" + TRAIN_N + " test=" + TEST_N +
      " seeds=" + SEEDS.len() + " perm_pairs=" + NPERM);

let rows_main = [];
let rows_off = [];
let rows_shuf = [];
let rows_raw = [];
let null_draws = [];     // pooled single-draw ridge AUCs (null spread)
let n_liq_last = 0;

for si in 0..SEEDS.len() {
    let seed_val = SEEDS[si];
    let n_liq = build_liquid.call(seed_val);
    n_liq_last = n_liq;
    if si == 0 {
        print("seed " + seed_val + ": liquid " + n_liq +
              " neurons, " + net.synapse_count() + " synapses");
    }

    let S1 = gen_set.call(seed_val, STRENGTH, false);
    let S0 = gen_set.call(seed_val, 0.0, false);
    let SR = gen_set.call(seed_val, STRENGTH, true);

    let r_main = eval_fit.call(S1[0], n_liq, S1[1], S1[2], S1[3]);
    let r_off = eval_fit.call(S0[0], n_liq, S0[1], S0[2], S0[3]);
    let r_raw = eval_fit.call(SR[0], N_SRC, SR[1], SR[2], SR[3]);

    // shuffled-label control: NPERM permutations + complements
    let draws = [];
    for b in 0..NPERM {
        let ytr = shuffled.call(S1[1]);
        draws.push(eval_fit.call(S1[0], n_liq, ytr, S1[2], S1[3]));
        draws.push(eval_fit.call(S1[0], n_liq, complement.call(ytr), S1[2], S1[3]));
    }
    let r_shuf = [mean_of.call(draws, 0), mean_of.call(draws, 1),
                  mean_of.call(draws, 2), mean_of.call(draws, 3)];
    for d in draws {
        null_draws.push(d[3]);
    }

    if si == 0 {
        // keep trained weights in the saved file (editor readouts plot)
        let wr = fit_ridge(S1[0], TRAIN_N, n_liq, S1[1], RIDGE_L);
        net.set_readout_weights("ro", wr);
        net.save("examples/lsm.ne2");
    }

    rows_main.push(r_main);
    rows_off.push(r_off);
    rows_shuf.push(r_shuf);
    rows_raw.push(r_raw);

    print("seed " + seed_val +
          "  main auc=" + f3.call(r_main[3]) +
          "  off=" + f3.call(r_off[3]) +
          "  shuf=" + f3.call(r_shuf[3]) + " (1-draw sd=" +
            f3.call(sd_of.call(draws, 3, r_shuf[3])) + ")" +
          "  raw=" + f3.call(r_raw[3]));
}

// ---------- summary ----------
print("\n=== mean ± sd over " + SEEDS.len() + " seeds ===");
let summary = |name, rows| {
    for k in [1, 3] {
        let m = mean_of.call(rows, k);
        let s = sd_of.call(rows, k, m);
        let what = if k == 1 { "delta_auc" } else { "ridge_auc" };
        print(name + "  " + what + " = " + f3.call(m) + " ± " + f3.call(s));
    }
};
summary.call("main        ", rows_main);
summary.call("signal off  ", rows_off);
summary.call("shuffled    ", rows_shuf);
summary.call("raw baseline", rows_raw);
let nm = mean_of.call(null_draws.map(|v| [0, 0, 0, v]), 3);
print("shuffled single-draw ridge AUC: mean=" + f3.call(nm) +
      " sd=" + f3.call(sd_of.call(null_draws.map(|v| [0, 0, 0, v]), 3, nm)) +
      " (the old control was ONE of these draws)");

// strength sweep on the last-built (last seed) liquid, for reference
print("\nstrength sweep (seed " + SEEDS[SEEDS.len() - 1] + " liquid):");
for s in [0.0, 0.5, 0.8, 1.0] {
    let Ss = gen_set.call(SEEDS[SEEDS.len() - 1], s, false);
    let r = eval_fit.call(Ss[0], n_liq_last, Ss[1], Ss[2], Ss[3]);
    print("  s=" + s + "  delta_auc=" + f3.call(r[1]) + " ridge_auc=" + f3.call(r[3]));
}
print("done");
