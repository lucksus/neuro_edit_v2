// Deep readout on a big local liquid: a 20^3 = 8000-neuron izh liquid
// with random distance-dependent (local) wiring and Dale's law reads 8
// random bit streams. In every window exactly ONE of four stream pairs
// secretly shares its bits (stream b copies stream a, with a couple of
// swaps); the task is 4-way: WHICH pair? Every stream is balanced —
// exactly half its 40 bits are 1 — so per-stream rates carry NOTHING,
// not even "these two counts are suspiciously equal". Only the
// spike-time coincidences differ, and the liquid has to turn those
// into rates.
// A NON-SPIKING multi-layer perceptron, trained by backprop on the
// liquid's per-neuron spike rates, answers on held-out windows.
//
//   Wiring: p(edge) = 0.35 * exp(-(d/90)^2) on a 60-unit lattice —
//   nearest neighbours only, mean fan-in ~5.5 (printed below).
//
//   Readouts printed side by side (held-out accuracy, chance = 0.25):
//     deep    MLP, hidden [128, 64] (or the readout's layers — select
//             the readout in the editor, edit its layers, press Train)
//     1-hid   MLP, hidden [128], same optimizer — depth is the only change
//     ridge   one-vs-rest ridge on the same liquid features
//     raw     no liquid: the deep MLP on per-stream bit rates
//     bits    no liquid: the deep MLP on the raw 8x40 input bits
//     cooc    hand-built side channel: 28 pairwise equal-bit rates
//             (knows the task's structure — the ceiling, not a rival)
//
//   Run: sim --script examples/lsm_deep.rhai [--cuda|--cpu]
//   In the editor the liquid loads as a "block" group glyph; the
//   readout "ro" draws as a layered stack once trained.
//   quick() shrinks everything for tests (NEURO_QUICK=1).

// ---------- readout layers: the readout's own config wins ----------
// Read BEFORE net.clear(): the editor's inspector stores layers on the
// readout, Train re-runs this script, and they must survive the rebuild.
let RO_LAYERS = net.readout_layers("ro");
let DEFAULT_LAYERS = [128, 64];
let LAYERS = if type_of(RO_LAYERS) == "array" { RO_LAYERS } else { DEFAULT_LAYERS };

// ---------- parameters ----------
let SEED = 11;
let Q = quick();
let SIDE = if Q { 8 } else { 20 };    // 20^3 = 8000 liquid neurons
let N_SRC = 8;
let WIN_MS = 40;
let SPB = 10;                         // sim steps per ms (dt = 0.1)
let PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]];
let K = 4;
let SWAPS = 2;                        // 1↔0 swaps applied to the copy
let TRAIN_N = if Q { 160 } else { 960 };
let TEST_N = if Q { 80 } else { 400 };

let LIQ_SPACING = 60.0;
let CONN_SCALING = 0.35;
let CONN_LAMBDA = 90.0;
let W_REC = 3.0;
let W_IN = 12.0;
let SHARED_PER_PAIR = 48;
let PRIV_PER_SRC = 24;
let RIDGE_L = 1.0;
let EPOCHS = if Q { 80 } else { 150 };
let opts_for = |layers| #{layers: layers, classes: K, opt: "adam",
                          lr: 0.001, wd: 0.001, batch: 32, epochs: EPOCHS,
                          patience: 20, seed: SEED};

// ---------- build the liquid ----------
net.clear();
seed(SEED);
net.create_3d_grid("res", SIDE, SIDE, SIDE, LIQ_SPACING);
net.set_exc_inh("res", 0.2);
net.create_connections("res", CONN_SCALING, CONN_LAMBDA, W_REC);
let res = net.neurons_of("res");
let n_liq = res.len();
net.add_readout("ro", res);
if type_of(RO_LAYERS) == "array" { net.set_readout_layers("ro", RO_LAYERS); }

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

// ---------- features ----------
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
let bits_of = |m| {
    let f = [];
    for t in 0..WIN_MS { for s in 0..N_SRC { f.push(m[t][s]); } }
    f
};
let cooc_of = |m| {
    let f = [];
    for i in 0..N_SRC {
        for j in (i + 1)..N_SRC {
            let c = 0.0;
            for t in 0..WIN_MS { if m[t][i] == m[t][j] { c += 1.0; } }
            f.push(c / WIN_MS);
        }
    }
    f
};

// ---------- balanced streams ----------
// `n` bits, exactly n/2 ones, shuffled (seeded via rand()).
let balanced = |n| {
    let a = [];
    for t in 0..n { a.push(if t < n / 2 { 1.0 } else { 0.0 }); }
    for t in 0..(n - 1) {
        let j = t + rand(0.0, 1.0e9).to_int() % (n - t);
        let tmp = a[t]; a[t] = a[j]; a[j] = tmp;
    }
    a
};
// One window m[t][s]: pair PAIRS[label] = (a, b) has b = a's bits with
// SWAPS random 1↔0 swaps — still exactly balanced.
let window_of = |label| {
    let S = [];
    for s in 0..N_SRC { S.push(balanced.call(WIN_MS)); }
    let a = PAIRS[label][0];
    let b = PAIRS[label][1];
    let c = S[a];
    for k in 0..SWAPS {
        let i = rand(0.0, 1.0e9).to_int() % WIN_MS;
        while c[i] < 0.5 { i = rand(0.0, 1.0e9).to_int() % WIN_MS; }
        let j = rand(0.0, 1.0e9).to_int() % WIN_MS;
        while c[j] > 0.5 { j = rand(0.0, 1.0e9).to_int() % WIN_MS; }
        c[i] = 0.0; c[j] = 1.0;
    }
    S[b] = c;
    let m = [];
    for t in 0..WIN_MS {
        let row = [];
        for s in 0..N_SRC { row.push(S[s][t]); }
        m.push(row);
    }
    m
};

let syn = net.synapse_count();
print("=== lsm_deep: " + n_liq + "-neuron local liquid, 4-way hidden-pair task ===");
print("backend=" + sim.backend() + "  quick=" + Q + "  synapses=" + syn +
      "  mean fan-in=" + ((syn * 10 / n_liq).to_int() / 10.0) +
      "  lambda=" + CONN_LAMBDA + " (spacing " + LIQ_SPACING + ")");
print("train=" + TRAIN_N + " test=" + TEST_N + " windows of " + WIN_MS +
      " ms; chance = 0.25; readout layers " + LAYERS +
      (if type_of(RO_LAYERS) == "array" { " (from the readout)" } else { " (default)" }));

{
    play_window.call(window_of.call(0));
    print("alive: spikes=" + sim.spikes() + " rate=" + sim.rate() +
          " Hz over the probe window");
}

// ---------- one pass over the windows: every feature set at once ----------
let gen = |n| {
    let D = #{liq: [], raw: [], bits: [], cooc: [], y: []};
    for it in 0..n {
        let label = it % K;
        let m = window_of.call(label);
        D.liq.push(play_window.call(m));
        D.raw.push(counts_of.call(m));
        D.bits.push(bits_of.call(m));
        D.cooc.push(cooc_of.call(m));
        D.y.push(label);
    }
    D
};
let flat = |rows| { let x = []; for r in rows { x += r; } x };

let TR = gen.call(TRAIN_N);
let TE = gen.call(TEST_N);

let f3 = |v| (v * 1000.0).to_int() / 1000.0;
let acc_mlp = |m, rows, y| {
    let pred = [];
    for r in rows { pred.push(mlp_predict(m, r)); }
    class_accuracy(pred, y)
};
let acc_ridge = |ws, rows, y| {
    let pred = [];
    for r in rows { pred.push(ridge_predict(ws, r)); }
    class_accuracy(pred, y)
};

let X = flat.call(TR.liq);
let m_deep = net.train_readout_mlp("ro", X, TRAIN_N, n_liq, TR.y, opts_for.call(LAYERS));
let m_one = fit_mlp(X, TRAIN_N, n_liq, TR.y, opts_for.call([128]));
let w_ridge = fit_ridge_multi(X, TRAIN_N, n_liq, TR.y, K, RIDGE_L);
let a_deep = acc_mlp.call(m_deep, TE.liq, TE.y);
let a_one = acc_mlp.call(m_one, TE.liq, TE.y);
let a_ridge = acc_ridge.call(w_ridge, TE.liq, TE.y);

let m_raw = fit_mlp(flat.call(TR.raw), TRAIN_N, N_SRC, TR.y, opts_for.call(LAYERS));
let m_bits = fit_mlp(flat.call(TR.bits), TRAIN_N, N_SRC * WIN_MS, TR.y, opts_for.call(LAYERS));
let w_cooc = fit_ridge_multi(flat.call(TR.cooc), TRAIN_N, 28, TR.y, K, RIDGE_L);
let a_raw = acc_mlp.call(m_raw, TE.raw, TE.y);
let a_bits = acc_mlp.call(m_bits, TE.bits, TE.y);
let a_cooc = acc_ridge.call(w_cooc, TE.cooc, TE.y);

print("held-out accuracy (chance 0.25):");
print("test deep  acc=" + f3.call(a_deep) + "   MLP " + mlp_sizes(m_deep) + " on the liquid  <- the headline");
print("test 1hid  acc=" + f3.call(a_one) + "   MLP " + mlp_sizes(m_one) + " on the liquid");
print("test ridge acc=" + f3.call(a_ridge) + "   ridge on the liquid");
print("test raw   acc=" + f3.call(a_raw) + "   no liquid: deep MLP on per-stream rates");
print("test bits  acc=" + f3.call(a_bits) + "   no liquid: deep MLP on the raw input bits");
print("test cooc  acc=" + f3.call(a_cooc) + "   hand-built pairwise side channel (knows the task)");
let best_nl = if a_raw > a_bits { a_raw } else { a_bits };
print(if a_deep > a_one + 0.02 {
    "depth: the deep readout beats the 1-hidden one here"
} else if a_one > a_deep + 0.02 {
    "depth: the 1-hidden readout beats the deep one here — depth does not help on this task"
} else {
    "depth: deep and 1-hidden are within 0.02 — depth does not clearly help on this task"
});
print(if a_deep > best_nl + 0.1 {
    "liquid: the liquid readouts clearly beat both no-liquid baselines"
} else {
    "liquid: the liquid does NOT clearly beat the no-liquid baselines"
});

// Stay visibly alive after the experiment.
sim.reset();
sim.set_noise(60.0, 10.0);
let s0 = sim.spikes();
sim.run(2000.0);
let tail_hz = (sim.spikes() - s0) / n_liq / 2.0;
print("done: tail rate=" + f3.call(tail_hz) + " Hz over " + n_liq +
      " neurons on noise; press Run to keep it going");
