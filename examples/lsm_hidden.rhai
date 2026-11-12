// Liquid-state-machine experiment: detect hidden pairwise correlations in
// random bit streams. The signal is invisible to any single stream's
// statistics (each stays Bernoulli(p)) — only coincident activity between
// paired sources carries it. The liquid turns coincidences into rate
// differences a linear readout can separate.
//
//   sim --script examples/lsm_hidden.rhai [--cuda|--cpu]
//
// Loops over SEEDS, rebuilding the liquid per seed, and evaluates four
// feature pipelines on the SAME bit windows:
//   liquid   — per-neuron spike rates of the 512-neuron liquid (~500 dims)
//   signal off — liquid, but strength 0 (no planted correlation)
//   raw      — per-source bit counts (8 dims, linear => must stay chance)
//   cooc     — 28 pairwise coincidence features; the trivial non-liquid
//              baseline that directly counts equal bits per source pair
//
// Null model: a real permutation test — the main labels are shuffled NPERM
// times, the ridge refit each time, and the fraction of shuffled AUCs
// reaching the observed AUC is the p-value (>=500 draws => resolution
// ~0.002). Null mean/sd are reported too; a single permutation is NOT a
// valid null (one draw's fitted noise direction can land anywhere in
// [0,1], sd ~ 0.2).
//
// Splits: TRAIN_N train / VAL_N validation / TEST_N test windows from
// three independent generator streams. Validation is never trained or
// tuned on — it cross-checks the test number.
//
// Summary reports mean ± sd across seeds. Controls must sit at 0.5 within
// noise; the coincidence baseline is reported honestly even where it
// beats the liquid.

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
let NPERM = 500;             // label permutations per seed (permutation test)

let LIQ_SIDE = 8;
let LIQ_SPACING = 60.0;
let CONN_SCALING = 0.35;
let CONN_LAMBDA = 90.0;
let W_REC = 3.0;
let W_IN = 12.0;             // input weight: 1 source sub-threshold, 2 spike
let SHARED_PER_PAIR = 24;    // pool neurons shared inside a pair
let PRIV_PER_SRC = 12;       // private neurons per source

let TRAIN_N = 200;
let VAL_N = 120;
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

// Pairwise coincidence features: for each of the C(8,2)=28 source pairs,
// the fraction of window steps with equal bits. Directly detects the
// planted structure with NO liquid — the honest baseline to beat.
let cooc_of = |m| {
    let f = [];
    for i in 0..N_SRC {
        for j in (i + 1)..N_SRC {
            let c = 0.0;
            for t in 0..WIN_MS {
                if m[t][i] == m[t][j] {
                    c += 1.0;
                }
            }
            f.push(c / WIN_MS);
        }
    }
    f
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

// Feature extractor selector for gen_set: 0 liquid, 1 raw counts, 2 cooc.
let MODE_LIQ = 0;
let MODE_RAW = 1;
let MODE_COOC = 2;
let feat_of = |mode, m| {
    if mode == MODE_RAW {
        counts_of.call(m)
    } else if mode == MODE_COOC {
        cooc_of.call(m)
    } else {
        play_window.call(m)
    }
};

// Generate one feature set: [X flat, Y, Xte, labs, Xval, labsval].
// Train/val/test windows come from three independent generator streams.
let gen_set = |seed_val, strength, mode| {
    let gen_tr = rng_stream(seed_val + 11, N_SRC, P);
    let gen_va = rng_stream(seed_val + 55, N_SRC, P);
    let gen_te = rng_stream(seed_val + 77, N_SRC, P);
    let X = [];
    let Y = [];
    let Xte = [];
    let labs = [];
    let Xval = [];
    let labsval = [];
    for it in 0..TRAIN_N {
        let label = it % 2;
        let m = gen_tr.window(WIN_MS, label, PAIRS, strength);
        for v in feat_of.call(mode, m) {
            X.push(v);
        }
        Y.push(label.to_float());
    }
    for it in 0..VAL_N {
        let label = it % 2;
        let m = gen_va.window(WIN_MS, label, PAIRS, strength);
        Xval.push(feat_of.call(mode, m));
        labsval.push(label);
    }
    for it in 0..TEST_N {
        let label = it % 2;
        let m = gen_te.window(WIN_MS, label, PAIRS, strength);
        Xte.push(feat_of.call(mode, m));
        labs.push(label);
    }
    [X, Y, Xte, labs, Xval, labsval]
};

// Fit both readouts on (X, ytr) and score val+test against their labels.
// Returns [delta_acc, delta_auc, ridge_acc, ridge_auc, ridge_auc_val].
let eval_fit = |X, n_feat, ytr, Xte, labs, Xval, labsval| {
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
    let sv = [];
    for f in Xval {
        sv.push(ridge_output(wr, f));
    }
    [accuracy(sd, labs), auc(sd, labs), accuracy(sr, labs), auc(sr, labs),
     auc(sv, labsval)]
};

// Ridge AUC of one permutation draw (null model for the permutation test).
let perm_auc = |X, n_feat, ytr_perm, Xte, labs| {
    let w = fit_ridge(X, TRAIN_N, n_feat, ytr_perm, RIDGE_L);
    let s = [];
    for f in Xte {
        s.push(ridge_output(w, f));
    }
    auc(s, labs)
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

// ---------- run ----------
print("=== LSM hidden-correlation experiment (honest eval) ===");
print("sources=" + N_SRC + " pairs=" + PAIRS.len() + " p=" + P +
      " win=" + WIN_MS + "ms strength=" + STRENGTH +
      " train=" + TRAIN_N + " val=" + VAL_N + " test=" + TEST_N +
      " seeds=" + SEEDS.len() + " perms/seed=" + NPERM);

let rows_main = [];
let rows_val = [];         // main ridge AUC on the validation split
let rows_off = [];
let rows_raw = [];
let rows_cooc = [];
let rows_p = [];           // per-seed permutation p-values
let null_draws = [];       // pooled null ridge AUCs
let null_above = 0;        // pooled count of null draws >= the observed auc
let n_liq_last = 0;

for si in 0..SEEDS.len() {
    let seed_val = SEEDS[si];
    let n_liq = build_liquid.call(seed_val);
    n_liq_last = n_liq;
    if si == 0 {
        print("seed " + seed_val + ": liquid " + n_liq +
              " neurons, " + net.synapse_count() + " synapses");
    }

    let S1 = gen_set.call(seed_val, STRENGTH, MODE_LIQ);
    let S0 = gen_set.call(seed_val, 0.0, MODE_LIQ);
    let SR = gen_set.call(seed_val, STRENGTH, MODE_RAW);
    let SC = gen_set.call(seed_val, STRENGTH, MODE_COOC);

    let r_main = eval_fit.call(S1[0], n_liq, S1[1], S1[2], S1[3], S1[4], S1[5]);
    let r_off = eval_fit.call(S0[0], n_liq, S0[1], S0[2], S0[3], S0[4], S0[5]);
    let r_raw = eval_fit.call(SR[0], N_SRC, SR[1], SR[2], SR[3], SR[4], SR[5]);
    let r_cooc = eval_fit.call(SC[0], 28, SC[1], SC[2], SC[3], SC[4], SC[5]);

    // Permutation test: NPERM label shuffles on the main feature set.
    let draws = [];
    let above = 0;
    for b in 0..NPERM {
        let a = perm_auc.call(S1[0], n_liq, shuffled.call(S1[1]), S1[2], S1[3]);
        draws.push(a);
        if a >= r_main[3] {
            above += 1;
            null_above += 1;
        }
    }
    let p_seed = (1.0 + above) / (NPERM + 1.0);
    let nm = mean_of.call(draws.map(|v| [0, 0, 0, v]), 3);
    let ns = sd_of.call(draws.map(|v| [0, 0, 0, v]), 3, nm);
    for d in draws {
        null_draws.push(d);
    }

    if si == 0 {
        // keep trained weights in the saved file (editor readouts plot)
        let wr = fit_ridge(S1[0], TRAIN_N, n_liq, S1[1], RIDGE_L);
        net.set_readout_weights("ro", wr);
        net.save("examples/lsm.ne2");
    }

    rows_main.push(r_main);
    rows_val.push([0, 0, 0, r_main[4]]);
    rows_off.push(r_off);
    rows_raw.push(r_raw);
    rows_cooc.push(r_cooc);
    rows_p.push(p_seed);

    print("seed " + seed_val +
          "  main auc=" + f3.call(r_main[3]) + " (val=" + f3.call(r_main[4]) +
          ")  p=" + f3.call(p_seed) +
          "  off=" + f3.call(r_off[3]) +
          "  raw=" + f3.call(r_raw[3]) +
          "  cooc=" + f3.call(r_cooc[3]) +
          "  null=" + f3.call(nm) + "±" + f3.call(ns));
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
let mv = mean_of.call(rows_val, 3);
print("main (val)   ridge_auc = " + f3.call(mv) +
      " ± " + f3.call(sd_of.call(rows_val, 3, mv)));
summary.call("signal off  ", rows_off);
summary.call("raw baseline", rows_raw);
summary.call("cooc (28-d) ", rows_cooc);
let nm = mean_of.call(null_draws.map(|v| [0, 0, 0, v]), 3);
let ns = sd_of.call(null_draws.map(|v| [0, 0, 0, v]), 3, nm);
let p_pool = (1.0 + null_above) / (null_draws.len() + 1.0);
let pm = mean_of.call(rows_p.map(|v| [0, 0, 0, v]), 3);
print("permutation null: mean=" + f3.call(nm) + " sd=" + f3.call(ns) +
      "  pooled p=" + f3.call(p_pool) +
      "  mean per-seed p=" + f3.call(pm) +
      " (" + null_draws.len() + " draws)");

// strength sweep on the last-built liquid: liquid vs coincidence features
print("\nstrength sweep (seed " + SEEDS[SEEDS.len() - 1] + " liquid):");
for s in [0.0, 0.5, 0.8, 1.0] {
    let Sl = gen_set.call(SEEDS[SEEDS.len() - 1], s, MODE_LIQ);
    let rl = eval_fit.call(Sl[0], n_liq_last, Sl[1], Sl[2], Sl[3], Sl[4], Sl[5]);
    let Sc = gen_set.call(SEEDS[SEEDS.len() - 1], s, MODE_COOC);
    let rc = eval_fit.call(Sc[0], 28, Sc[1], Sc[2], Sc[3], Sc[4], Sc[5]);
    print("  s=" + s +
          "  liquid ridge_auc=" + f3.call(rl[3]) +
          "  cooc ridge_auc=" + f3.call(rc[3]));
}
print("done");
