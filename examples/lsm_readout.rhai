// Reservoir + MLP readout (liquid state machine): a small 3D reservoir
// is driven by 6 random bit streams; on "event" windows two pairs share
// a hidden common bit — invisible to any single stream's rate. An MLP
// trained on the reservoir's spike counts separates event from noise on
// held-out windows; the raw stream baseline stays at chance. Keep it
// small — this is the intro demo; lsm_big.rhai is the real experiment
// and lsm_hidden.rhai the full honest evaluation.
//
//   What to look for: "alive:" shows the reservoir firing on a probe
//   window before any training; the test AUCs compare mlp vs ridge on
//   reservoir features vs a raw-input baseline that must sit at 0.5.
//   After the script ends the liquid keeps firing on background noise,
//   so pressing Run in the editor still shows activity.

let SEED = 7;
let SIDE = 6;                  // 6^3 = 216 reservoir neurons
let N_SRC = 6;
let PAIRS = [[0, 1], [2, 3]];
let WIN = 30;                  // ms per window (1 ms bits, 0.1 ms steps)
let TRAIN = 80;
let TEST = 50;
// Coincidence-detector regime: Izhikevich RS fires only above ~4 mV of
// sustained current, so a single stream at 2.8 stays sub-threshold but
// the pair's shared bit arriving together (5.6) spikes the pool — the
// planted correlation becomes a *firing* difference, not a rate hint.
// (Stage 17 retune: with real Dale's-law inhibition + fast-spiking
// inhibitory cells, the old W_IN=9 saturated the whole reservoir.)
let W_IN = 2.8;

net.clear();
seed(SEED);
net.create_3d_grid("res", SIDE, SIDE, SIDE, 20.0);
net.set_exc_inh("res", 0.2);
// lambda must clear the 20-unit spacing or the reservoir has no synapses
// at all (exp(-(20/lambda)^2)); weak recurrence keeps the shared-pool
// signal readable instead of washing it out.
net.create_connections("res", 1.0, 30.0, 1.0);
let res = net.neurons_of("res");
let n_res = res.len();
net.add_readout("ro", res);

// Each pair shares a big pool (the planted common bit hits the SAME
// reservoir neurons).
let SHARED = 80;
for p in PAIRS {
    let pool = [];
    for k in 0..SHARED {
        pool.push(res[rand(0.0, 1.0e9).to_int() % n_res]);
    }
    for s in p {
        let tgts = [];
        for n in pool {
            tgts.push([n, W_IN]);
        }
        net.add_input("src" + s, tgts);
    }
}

// Reservoir features after playing a bit window: per-neuron spike
// counts. Only paired sources have inputs — the rest of the stream is
// pure noise by design and stays silent.
let driven = [];
for p in PAIRS {
    for s in p {
        driven.push(s);
    }
}
let replay = |m| {
    sim.reset();
    for s in driven {
        let series = [];
        for t in 0..WIN {
            for k in 0..10 { series.push(m[t][s]); } // 0.1 ms steps
        }
        sim.play_input("src" + s, series);
    }
    sim.run_steps(WIN * 10);
    sim.spike_counts(WIN * 10)
};

// alive proof before training: the reservoir fires on a probe window.
{
    let m = rng_stream(SEED + 5, N_SRC, 0.4).window(WIN, 1, PAIRS, 0.85);
    let f = replay.call(m);
    print("alive: spikes=" + sim.spikes() +
          " rate=" + sim.rate() + " Hz over the probe window");
}

// --- collect train / held-out test feature sets ---
let g_tr = rng_stream(SEED + 11, N_SRC, 0.4);
let g_te = rng_stream(SEED + 77, N_SRC, 0.4);
let X = [];
let Xr = [];
let Y = [];
let Xte = [];
let Xte_raw = [];
let labs = [];
for i in 0..TRAIN {
    let ev = (i % 2) == 0;
    let m = g_tr.window(WIN, if ev { 1 } else { 0 }, PAIRS, 0.85);
    for v in replay.call(m) { X.push(v); }
    for s in 0..N_SRC {
        let a = 0.0;
        for t in 0..WIN { a += m[t][s]; }
        Xr.push(a / WIN);
    }
    Y.push(if ev { 1.0 } else { 0.0 });
}
for i in 0..TEST {
    let ev = (i % 2) == 0;
    let m = g_te.window(WIN, if ev { 1 } else { 0 }, PAIRS, 0.85);
    Xte.push(replay.call(m));
    let raw = [];
    for s in 0..N_SRC {
        let a = 0.0;
        for t in 0..WIN { a += m[t][s]; }
        raw.push(a / WIN);
    }
    Xte_raw.push(raw);
    labs.push(if ev { 1 } else { 0 });
}

// --- train the non-spiking MLP on reservoir counts; ridge baseline ---
let m_mlp = fit_mlp(X, TRAIN, n_res, Y,
                    #{hidden: 16, epochs: 120, lr: 0.05, seed: SEED});
let wr = fit_ridge(X, TRAIN, n_res, Y, 1.0);
net.set_readout_mlp("ro", m_mlp); // trained weights persist in the .ne2

let s_mlp = [];
let s_ridge = [];
for f in Xte {
    s_mlp.push(mlp_output(m_mlp, f));
    s_ridge.push(ridge_output(wr, f));
}
let f3 = |v| (v * 1000.0).to_int() / 1000.0;
print("test ridge auc=" + f3.call(auc(s_ridge, labs)) +
      " acc=" + f3.call(accuracy(s_ridge, labs)));

// raw-stream baseline: mean bit per source over the SAME windows,
// linear readout — it cannot see pair structure, so it must sit at
// chance.
let wr_raw = fit_ridge(Xr, TRAIN, N_SRC, Y, 1.0);
let s_raw = [];
for m2 in Xte_raw {
    s_raw.push(ridge_output(wr_raw, m2));
}
print("test raw   auc=" + f3.call(auc(s_raw, labs)) + "   (expect ~0.5)");
print("test mlp   auc=" + f3.call(auc(s_mlp, labs)) +
      " acc=" + f3.call(accuracy(s_mlp, labs)) + "   <- the headline");

// Stay visibly alive: background noise keeps the reservoir firing after
// the script ends, so Run in the editor still shows activity.
sim.reset();
sim.set_noise(60.0, 8.0);
sim.run(400.0);
print("done: spikes=" + sim.spikes() +
      " — reservoir is still firing on noise; press Run to continue");
