// Reservoir + readout (liquid state machine): a small 3D reservoir is
// driven by 6 random bit streams; on "event" windows two pairs share a
// hidden common bit — invisible to any single stream's rate. A delta
// readout trained on the reservoir's own spike trace separates event
// from noise; the raw stream baseline stays at chance. Keep it small —
// this is a demo, not the paper experiment (see lsm_hidden.rhai).

let SEED = 7;
let SIDE = 6;                  // 6^3 = 216 reservoir neurons
let N_SRC = 6;
let PAIRS = [[0, 1], [2, 3]];
let WIN = 30;                  // ms per window (1 ms steps)
let TRAIN = 60;
let TEST = 40;
let W_IN = 9.0;

net.clear();
seed(SEED);
net.create_3d_grid("res", SIDE, SIDE, SIDE, 20.0);
net.set_exc_inh("res", 0.2);
net.create_connections("res", 1.0, 2.5, 0.8);
let res = net.neurons_of("res");
let n_res = res.len();

// Each pair shares a pool (so the planted common bit hits the SAME
// reservoir neurons) plus a private pool per source.
let SHARED = 12;
let PRIV = 6;
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
        for k in 0..PRIV {
            tgts.push([res[rand(0.0, 1.0e9).to_int() % n_res], W_IN]);
        }
        net.add_input("src" + s, tgts);
    }
}

// Reservoir features after playing a bit window: raw per-neuron spike
// counts (not rates — the delta rule needs features of order ~1).
let replay = |m| {
    sim.reset();
    for s in 0..N_SRC {
        let series = [];
        for t in 0..WIN {
            for k in 0..10 { series.push(m[t][s]); } // 0.1 ms steps
        }
        sim.play_input("src" + s, series);
    }
    sim.run_steps(WIN * 10);
    sim.spike_counts(WIN * 10)
};

let stream = rng_stream(SEED, N_SRC, 0.4);
let ro = delta_new(n_res);
ro.set_proper(1);
let LR = 0.15;

// --- train ---
for i in 0..TRAIN {
    let ev = (i % 2) == 0;
    let m = stream.window(WIN, if ev { 1 } else { 0 }, PAIRS, 0.85);
    ro.learn(replay.call(m), if ev { 1.0 } else { 0.0 }, LR);
}

// --- test (fresh windows from the same stream) ---
let n = 0.0;
let ok = 0.0;
let base_ok = 0.0;
for i in 0..TEST {
    let ev = (i % 2) == 0;
    let m = stream.window(WIN, if ev { 1 } else { 0 }, PAIRS, 0.85);
    let s = ro.output(replay.call(m));
    if (s > 0.5) == ev { ok += 1.0; }
    // linear baseline on the raw bits: per-source mean, majority vote —
    // it cannot see pair structure, so it should sit at chance.
    let b = (m[0][0] + m[0][1]) > 0.5;
    if b == ev { base_ok += 1.0; }
    n += 1.0;
}
print("readout accuracy on reservoir trace: " + (ok / n));
print("raw-stream baseline (expect ~0.5): " + (base_ok / n));
