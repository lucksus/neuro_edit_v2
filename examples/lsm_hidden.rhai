// Liquid-state-machine experiment: detect hidden pairwise correlations in
// random bit streams. The signal is invisible to any single stream's
// statistics (each stays Bernoulli(p)) — only coincident activity between
// paired sources carries it. The liquid turns coincidences into rate
// differences a linear readout can separate.
//
//   sim --script examples/lsm_hidden.rhai [--cuda|--cpu]
//
// Prints accuracy/AUC for the trained readouts, the three controls
// (signal off, shuffled labels, raw-stream baseline) and a strength sweep.

// ---------- parameters ----------
let SEED = 1;
let N_SRC = 8;               // 5-10 sources
let P = 0.5;                 // Bernoulli bit probability
let WIN_MS = 50;             // window length
let SPB = 10;                // steps per bit (1 ms at dt=0.1)
let PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]];
let STRENGTH = 0.8;          // planted-signal strength (pairwise shared coin)
let LR = 0.5;                // delta-rule learn rate (proper rule, rate feats)
let EPOCHS = 10;             // delta-rule passes over the training set
let RIDGE_L = 1.0;

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

seed(SEED);

// ---------- build the liquid ----------
net.create_3d_grid("liquid", LIQ_SIDE, LIQ_SIDE, LIQ_SIDE, LIQ_SPACING);
net.set_exc_inh("liquid", 0.2);
net.create_connections("liquid", CONN_SCALING, CONN_LAMBDA, W_REC);
let liq = net.neurons_of("liquid");
let n_liq = liq.len();
print("liquid: " + n_liq + " neurons, " + net.synapse_count() + " synapses");

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
net.save("examples/lsm.ne2");

// ---------- helpers (anonymous fns capture the scope above) ----------

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

// One full experiment: [delta_acc, delta_auc, ridge_acc, ridge_auc].
//   strength        planted-correlation strength
//   do_shuffle      permute training labels (control)
//   raw             raw-stream counts instead of the liquid (control)
let experiment = |strength, do_shuffle, raw| {
    let gen_tr = rng_stream(SEED + 11, N_SRC, P);
    let gen_te = rng_stream(SEED + 77, N_SRC, P);
    let n_feat = if raw { N_SRC } else { n_liq };
    let ro = delta_new(n_feat);
    ro.init_random(-0.02, 0.02);
    ro.set_proper(1);          // corrected delta rule; quirk mode can't grow
                               // weights away from init (update ~ w*f)
    let X = [];
    let Y = [];
    for it in 0..TRAIN_N {
        let label = it % 2;
        let m = gen_tr.window(WIN_MS, label, PAIRS, strength);
        let f = if raw { counts_of.call(m) } else { play_window.call(m) };
        for v in f {
            X.push(v);
        }
        Y.push(label.to_float());
    }
    let Ytr = if do_shuffle { shuffled.call(Y) } else { Y };
    for ep in 0..EPOCHS {
        for it in 0..TRAIN_N {
            let tr = X.extract(it * n_feat, n_feat);
            ro.learn(tr, Ytr[it], LR);
        }
    }
    let wr = fit_ridge(X, TRAIN_N, n_feat, Ytr, RIDGE_L);
    let sd = [];
    let sr = [];
    let labs = [];
    for it in 0..TEST_N {
        let label = it % 2;
        let m = gen_te.window(WIN_MS, label, PAIRS, strength);
        let f = if raw { counts_of.call(m) } else { play_window.call(m) };
        sd.push(ro.output(f));
        sr.push(ridge_output(wr, f));
        labs.push(label);
    }
    [accuracy(sd, labs), auc(sd, labs), accuracy(sr, labs), auc(sr, labs)]
};

let report = |name, r| {
    print(name + "  delta: acc=" + r[0] + " auc=" + r[1] +
          "  ridge: acc=" + r[2] + " auc=" + r[3]);
};

// ---------- run ----------
print("=== LSM hidden-correlation experiment ===");
print("sources=" + N_SRC + " pairs=" + PAIRS.len() + " p=" + P +
      " win=" + WIN_MS + "ms strength=" + STRENGTH +
      " train=" + TRAIN_N + " test=" + TEST_N + " seed=" + SEED);

report.call("main        ", experiment.call(STRENGTH, false, false));
report.call("signal off  ", experiment.call(0.0, false, false));
report.call("shuffled    ", experiment.call(STRENGTH, true, false));
report.call("raw baseline", experiment.call(STRENGTH, false, true));

print("strength sweep:");
for s in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0] {
    let r = experiment.call(s, false, false);
    print("  s=" + s + "  delta_acc=" + r[0] + " delta_auc=" + r[1] +
          " ridge_acc=" + r[2] + " ridge_auc=" + r[3]);
}
print("done");
