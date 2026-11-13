// Signal-off control for lsm_big — the SAME build, the SAME pipeline,
// but the planted pairwise correlation is switched off (strength = 0).
// The "event" label is now unrelated to anything in the streams or the
// liquid state, so every held-out number MUST land at chance (~0.5).
//
//   Why it exists: an experiment that can't fail isn't an experiment.
//   If this ever prints liquid/mlp AUC far above 0.5, the pipeline —
//   not the liquid — is leaking the label (a real result of this exact
//   shape is why the lsm_hidden example added its permutation null).
//   Compare line for line with lsm_big.rhai: only STRENGTH and this
//   header differ.
//
//   What to look for: alive spikes prove the liquid still fires (the
//   control is off in the *signal*, not the dynamics); every test AUC
//   should sit near 0.5 — mlp, ridge, raw and even cooc, because with
//   no planted bit there is nothing for any of them to find.
//
//   sim --script examples/lsm_big_off.rhai [--cuda|--cpu]

// ---------- parameters (identical to lsm_big except STRENGTH) ----------
let SEED = 7;
let Q = quick();
let SIDE = if Q { 8 } else { 16 };
let N_SRC = 8;
let P = 0.5;
let WIN_MS = 40;
let SPB = 10;
let PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]];
let STRENGTH = 0.0;                   // <- the only functional change:
                                      //    no shared bit is planted
let TRAIN_N = if Q { 80 } else { 200 };
let TEST_N = if Q { 48 } else { 120 };

let LIQ_SPACING = 60.0;
let CONN_SCALING = 0.35;
let CONN_LAMBDA = 90.0;
let W_REC = 3.0;
let W_IN = 12.0;
let SHARED_PER_PAIR = 48;
let PRIV_PER_SRC = 24;
let RIDGE_L = 1.0;
let EPOCHS = if Q { 60 } else { 120 };
let MLP_OPTS = #{hidden: 24, epochs: EPOCHS,
                 lr: 0.05, wd: 0.0001, seed: SEED};

// ---------- build (identical to lsm_big) ----------
net.clear();
seed(SEED);
net.create_3d_grid("res", SIDE, SIDE, SIDE, LIQ_SPACING);
net.set_exc_inh("res", 0.2);
net.create_connections("res", CONN_SCALING, CONN_LAMBDA, W_REC);
let res = net.neurons_of("res");
let n_liq = res.len();
net.add_readout("ro", res);
for p in PAIRS {
    let pool = [];
    for k in 0..SHARED_PER_PAIR {
        pool.push(res[rand(0.0, 1.0e9).to_int() % n_liq]);
    }
    for s in p {
        let tgts = [];
        for n in pool { tgts.push([n, W_IN]); }
        for k in 0..PRIV_PER_SRC {
            tgts.push([res[rand(0.0, 1.0e9).to_int() % n_liq], W_IN]);
        }
        net.add_input("src" + s, tgts);
    }
}

let play_window = |m| {
    sim.reset();
    for s in 0..N_SRC {
        let series = [];
        for t in 0..WIN_MS {
            let b = m[t][s];
            for k in 0..SPB { series.push(b); }
        }
        sim.play_input("src" + s, series);
    }
    sim.run_steps(WIN_MS * SPB);
    sim.spike_counts(WIN_MS * SPB).map(|c| c / WIN_MS)
};
let counts_of = |m| {
    let c = [];
    for s in 0..N_SRC {
        let a = 0.0;
        for t in 0..WIN_MS { a += m[t][s]; }
        c.push(a / WIN_MS);
    }
    c
};
let cooc_of = |m| {
    let f = [];
    for i in 0..N_SRC {
        for j in (i + 1)..N_SRC {
            let c = 0.0;
            for t in 0..WIN_MS {
                if m[t][i] == m[t][j] { c += 1.0; }
            }
            f.push(c / WIN_MS);
        }
    }
    f
};
let gen = |seed_val, strength| {
    let g_tr = rng_stream(seed_val + 11, N_SRC, P);
    let g_te = rng_stream(seed_val + 77, N_SRC, P);
    let X = [];
    let Y = [];
    let Xte = [];
    let labs = [];
    for it in 0..TRAIN_N {
        let label = it % 2;
        let m = g_tr.window(WIN_MS, label, PAIRS, strength);
        for v in play_window.call(m) { X.push(v); }
        Y.push(label.to_float());
    }
    for it in 0..TEST_N {
        let label = it % 2;
        let m = g_te.window(WIN_MS, label, PAIRS, strength);
        Xte.push(play_window.call(m));
        labs.push(label);
    }
    [X, Y, Xte, labs]
};
let gen_side = |gen_s, strength, f_of| {
    let X = [];
    let Y = [];
    let Xte = [];
    let labs = [];
    for it in 0..TRAIN_N {
        let label = it % 2;
        let m = gen_s.window(WIN_MS, label, PAIRS, strength);
        for v in f_of.call(m) { X.push(v); }
        Y.push(label.to_float());
    }
    let g2 = rng_stream(SEED + 99, N_SRC, P);
    for it in 0..TEST_N {
        let label = it % 2;
        let m = g2.window(WIN_MS, label, PAIRS, strength);
        Xte.push(f_of.call(m));
        labs.push(label);
    }
    [X, Y, Xte, labs]
};

print("=== lsm_big_off: SIGNAL OFF control (strength=0) ===");
print("backend=" + sim.backend() + "  quick=" + Q +
      "  synapses=" + net.synapse_count() +
      "  train=" + TRAIN_N + " test=" + TEST_N + " win=" + WIN_MS + "ms");
{
    let m = rng_stream(SEED + 5, N_SRC, P).window(WIN_MS, 1, PAIRS, 0.0);
    let f = play_window.call(m);
    print("alive: spikes=" + sim.spikes() +
          " rate=" + sim.rate() + " Hz (control is off in the signal, not the dynamics)");
}

let S = gen.call(SEED, STRENGTH);
let m_mlp = fit_mlp(S[0], TRAIN_N, n_liq, S[1], MLP_OPTS);
let wr = fit_ridge(S[0], TRAIN_N, n_liq, S[1], RIDGE_L);
net.set_readout_mlp("ro", m_mlp);

let s_mlp = [];
let s_ridge = [];
for f in S[2] {
    s_mlp.push(mlp_output(m_mlp, f));
    s_ridge.push(ridge_output(wr, f));
}
let labs = S[3];
let f3 = |v| (v * 1000.0).to_int() / 1000.0;
print("test mlp   auc=" + f3.call(auc(s_mlp, labs)) +
      " acc=" + f3.call(accuracy(s_mlp, labs)) + "   (expect ~0.5)");
print("test ridge auc=" + f3.call(auc(s_ridge, labs)) +
      " acc=" + f3.call(accuracy(s_ridge, labs)) + "   (expect ~0.5)");

let SR = gen_side.call(rng_stream(SEED + 3, N_SRC, P), STRENGTH, counts_of);
let SC = gen_side.call(rng_stream(SEED + 4, N_SRC, P), STRENGTH, cooc_of);
for blk in [["raw", SR, N_SRC], ["cooc", SC, 28]] {
    let w = fit_ridge(blk[1][0], TRAIN_N, blk[2], blk[1][1], RIDGE_L);
    let s = [];
    for f in blk[1][2] { s.push(ridge_output(w, f)); }
    print("test " + blk[0] + "   auc=" + f3.call(auc(s, blk[1][3])) +
          "   (expect ~0.5)");
}

sim.reset();
sim.set_noise(60.0, 10.0);
sim.run(400.0);
print("done: spikes=" + sim.spikes() +
      " — signal-off liquid still fires on noise; press Run to continue");
