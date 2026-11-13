// Big LSM end to end: a 16^3 = 4096-neuron liquid reads 8 random bit
// streams. Inside each pair, the two streams share a hidden common bit
// on "event" windows — invisible to any single stream's rate, the
// signature only exists between them. A NON-SPIKING multi-layer
// perceptron trained on the liquid's per-neuron spike rates separates
// event from noise on held-out windows; ridge, a raw-stream baseline
// and the 28-dim coincidence baseline print next to it.
//
//   Run: sim --script examples/lsm_big.rhai [--cuda|--cpu]
//   In the editor the reservoir loads as a "block" group glyph —
//   four thousand dots would only smear; the block still flashes with
//   firing and the Inputs panel shows the streams live.
//
//   What to look for: "alive:" proves the liquid is spiking before
//   training starts; then the HELD-OUT test numbers print — mlp vs
//   ridge on liquid features, raw (must stay ~0.5) and cooc (the cheap
//   baseline the liquid has to beat). quick() shrinks everything for
//   the examples test / smoke runs (NEURO_QUICK=1).

// ---------- parameters ----------
let SEED = 7;
let Q = quick();
let SIDE = if Q { 8 } else { 16 };    // 16^3 = 4096 liquid neurons
let N_SRC = 8;
let P = 0.5;
let WIN_MS = 40;                      // ms per bit window
let SPB = 10;                         // sim steps per ms (dt = 0.1)
let PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]];
let STRENGTH = 0.85;
let TRAIN_N = if Q { 80 } else { 160 };
let TEST_N = if Q { 48 } else { 96 };

let LIQ_SPACING = 60.0;
let CONN_SCALING = 0.35;
let CONN_LAMBDA = 90.0;
let W_REC = 3.0;
let W_IN = 12.0;
let SHARED_PER_PAIR = 48;             // pool shared inside a pair
let PRIV_PER_SRC = 24;                // private pool per stream
let RIDGE_L = 1.0;
let EPOCHS = if Q { 60 } else { 120 };
let MLP_OPTS = #{hidden: 24, epochs: EPOCHS,
                 lr: 0.05, wd: 0.0001, seed: SEED};

// ---------- build the liquid (editor shows group "res" as a block) ----------
net.clear();
seed(SEED);
net.create_3d_grid("res", SIDE, SIDE, SIDE, LIQ_SPACING);
net.set_exc_inh("res", 0.2);
net.create_connections("res", CONN_SCALING, CONN_LAMBDA, W_REC);
let res = net.neurons_of("res");
let n_liq = res.len();
net.add_readout("ro", res);

// Each pair's streams share a big pool (the hidden common bit lands on
// the same neurons); every stream also owns a private pool so all eight
// stay equally represented individually.
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

// ---------- feature extractors ----------
// liquid: per-neuron spike rates over the replayed window
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
// raw: per-source bit counts — linear, must stay at chance
let counts_of = |m| {
    let c = [];
    for s in 0..N_SRC {
        let a = 0.0;
        for t in 0..WIN_MS { a += m[t][s]; }
        c.push(a / WIN_MS);
    }
    c
};
// cooc: for each of the 28 source pairs, fraction of equal bits —
// detects the planted structure with no liquid at all (the baseline
// to beat; on the small experiment it wins outright — see NOTES).
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

print("=== lsm_big: " + n_liq + "-neuron liquid, hidden-correlation task ===");
print("backend=" + sim.backend() + "  quick=" + Q +
      "  synapses=" + net.synapse_count() +
      "  train=" + TRAIN_N + " test=" + TEST_N + " win=" + WIN_MS + "ms");

// alive proof BEFORE training: replay one event window and report.
{
    let m = rng_stream(SEED + 5, N_SRC, P).window(WIN_MS, 1, PAIRS, STRENGTH);
    let f = play_window.call(m);
    print("alive: spikes=" + sim.spikes() +
          " rate=" + sim.rate() + " Hz over the probe window");
}

// ---------- train + held-out test ----------
let S = gen.call(SEED, STRENGTH);
let m_mlp = fit_mlp(S[0], TRAIN_N, n_liq, S[1], MLP_OPTS);
let wr = fit_ridge(S[0], TRAIN_N, n_liq, S[1], RIDGE_L);
net.set_readout_mlp("ro", m_mlp); // persists with the net

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

// side baselines on the same windows (no liquid features involved)
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

// Stay visibly alive: background noise keeps the liquid firing after
// the script ends, so Run in the editor shows ongoing activity.
sim.reset();
sim.set_noise(60.0, 10.0);
sim.run(400.0);
print("done: spikes=" + sim.spikes() +
      " — reservoir is still firing on noise; press Run to continue");
