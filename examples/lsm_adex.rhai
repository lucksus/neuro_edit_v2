// LSM variant — AdEx liquid, different size + E/I ratio: same
// hidden-correlation task as lsm_big, but the reservoir is 12^3 = 1728
// adaptive-exponential (Brette–Gerstner) neurons at 35% inhibitory.
// AdEx works in pA: one stream at 400 pA stays under threshold, the
// pair's shared bit arriving together (800 pA) drives it — so the
// planted correlation becomes a *firing* difference, not a rate hint.
//
//   What to look for: the same four held-out numbers as lsm_big.
//   Compare the mlp AUC against the izh liquid — does a more realistic
//   adapting cell keep the memory as well? The alive line + the noise
//   tail prove this liquid fires the whole time.
//
//   sim --script examples/lsm_adex.rhai [--cuda|--cpu]

// ---------- parameters ----------
let SEED = 11;
let Q = quick();
let SIDE = if Q { 6 } else { 12 };    // 12^3 = 1728 AdEx neurons
let N_SRC = 8;
let P = 0.5;
let WIN_MS = 40;
let SPB = 30;                         // AdEx integrates slowly — hold each bit 3 ms
let PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]];
let STRENGTH = 0.85;
let TRAIN_N = if Q { 80 } else { 160 };
let TEST_N = if Q { 48 } else { 100 };

let LIQ_SPACING = 60.0;
let CONN_SCALING = 0.35;
let CONN_LAMBDA = 90.0;
let W_REC = 150.0;                    // AdEx synapses carry pA, not mV
let W_IN = 700.0;                     // sustained pA per stream bit —
                                      // tonic AdEx needs ~600+ to fire
let SHARED_PER_PAIR = 48;
let PRIV_PER_SRC = 24;
let EI = 0.35;                        // variant: 35% inhibitory
let RIDGE_L = 1.0;
let EPOCHS = if Q { 60 } else { 100 };
let MLP_OPTS = #{hidden: 24, epochs: EPOCHS,
                 lr: 0.05, wd: 0.0001, seed: SEED};

// ---------- build the AdEx liquid ----------
net.clear();
seed(SEED);
// Naud et al. tonic-spiking AdEx for excitatory, non-adapting for inh.
net.def_type("ax", "adex", #{b: 80.5}, "exc");
net.def_type("axi", "adex", #{a: 0.0, b: 0.0}, "inh");
net.create_3d_grid("res", SIDE, SIDE, SIDE, LIQ_SPACING);
net.set_exc_inh("res", EI);
let res = net.neurons_of("res");
let n_liq = res.len();
for n in res {
    net.set_type(n, if net.kind(n) == "exc" { "ax" } else { "axi" });
}
net.create_connections("res", CONN_SCALING, CONN_LAMBDA, W_REC);
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

// ---------- features (same trio as lsm_big) ----------
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

print("=== lsm_adex: " + n_liq + " AdEx neurons, EI=" + EI +
      ", hidden-correlation task ===");
print("backend=" + sim.backend() + "  quick=" + Q +
      "  synapses=" + net.synapse_count() +
      "  train=" + TRAIN_N + " test=" + TEST_N + " win=" + WIN_MS + "ms");
{
    let m = rng_stream(SEED + 5, N_SRC, P).window(WIN_MS, 1, PAIRS, STRENGTH);
    let f = play_window.call(m);
    print("alive: spikes=" + sim.spikes() +
          " rate=" + sim.rate() + " Hz over the probe window");
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
print("test ridge auc=" + f3.call(auc(s_ridge, labs)) +
      " acc=" + f3.call(accuracy(s_ridge, labs)));

let SR = gen_side.call(rng_stream(SEED + 3, N_SRC, P), STRENGTH, counts_of);
let SC = gen_side.call(rng_stream(SEED + 4, N_SRC, P), STRENGTH, cooc_of);
for blk in [["raw", SR, N_SRC], ["cooc", SC, 28]] {
    let w = fit_ridge(blk[1][0], TRAIN_N, blk[2], blk[1][1], RIDGE_L);
    let s = [];
    for f in blk[1][2] { s.push(ridge_output(w, f)); }
    print("test " + blk[0] + "   auc=" + f3.call(auc(s, blk[1][3])));
}
print("test mlp   auc=" + f3.call(auc(s_mlp, labs)) +
      " acc=" + f3.call(accuracy(s_mlp, labs)) + "   <- the headline");

sim.reset();
// AdEx noise kick is a current pulse — a bigger kick value is needed
// than the izh examples use to keep the liquid visibly firing.
sim.set_noise(200.0, 700.0);
sim.run(400.0);
print("done: spikes=" + sim.spikes() +
      " — AdEx liquid still firing on noise; press Run to continue");
