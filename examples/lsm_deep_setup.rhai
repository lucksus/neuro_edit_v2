// Deep LSM — build only, live inputs, NO training. Loads the same
// 20^3 = 8000-neuron izh layout as lsm_deep (local random wiring,
// Dale's law) plus SIX input nodes standing in a column OUTSIDE the
// cube, each with its own random spike train (different seed and rate
// — "different RNG streams"). Each input projects into its own patch of
// the near face, so spikes enter from the side and spread through the
// liquid. The trains loop, so activity keeps flowing for as long as Run
// is on (Reset restarts it — it never runs dry).
//
//   Demo: load it, talk about the net, press Run, watch spikes travel.
//   Training is a separate step: Examples > "Deep readout" (lsm_deep)
//   builds this layout (at the sparser weight 3) and trains the readouts; or call the API
//   yourself from the console (help("readout")).

let SEED = 11;
let Q = quick();
let SIDE = if Q { 8 } else { 20 };    // 20^3 = 8000 liquid neurons
let SPACING = 60.0;
let N_IN = 6;
let SPB = 10;                         // sim steps per ms (dt = 0.1)
let LOOP_MS = 8000;                   // each train repeats every 8 s
let RATES = [12.0, 18.0, 25.0, 9.0, 30.0, 15.0];   // Hz, one per input
let PULSE_STEPS = 20;                 // 2 ms pulse per input spike
let W_IN = 14.0;
let PER_INPUT = 60;                   // liquid targets per input
let W_REC = 12.0;                     // recurrent weight: lsm_deep trains at 3.0 (sparse,
                                      // spikes stay local); 12 lets them travel — ~7 Hz
                                      // while driven, dies within ~1 s once inputs stop

net.clear();
seed(SEED);
net.create_3d_grid("res", SIDE, SIDE, SIDE, SPACING);
net.set_exc_inh("res", 0.2);
net.create_connections("res", 0.35, 90.0, W_REC);
let res = net.neurons_of("res");
let n_liq = res.len();
net.add_readout("ro", res);

// Liquid bounds, measured (the grid is centred near the origin).
let lo = [1.0e9, 1.0e9, 1.0e9];
let hi = [-1.0e9, -1.0e9, -1.0e9];
for id in res {
    let p = net.pos(id);
    for a in 0..3 {
        if p[a] < lo[a] { lo[a] = p[a]; }
        if p[a] > hi[a] { hi[a] = p[a]; }
    }
}
let ext = hi[0] - lo[0];
let near = [];                        // the two layers facing the inputs
for id in res {
    if net.pos(id)[0] <= lo[0] + SPACING * 1.5 { near.push(id); }
}

// Inputs: a column at 45% of the cube width left of it, evenly spaced
// in y, centred in z; input k feeds the near-face neurons in its own
// y band.
for k in 0..N_IN {
    let y_lo = lo[1] + (hi[1] - lo[1]) * k / N_IN;
    let y_hi = lo[1] + (hi[1] - lo[1]) * (k + 1) / N_IN;
    let band = [];
    for id in near {
        let y = net.pos(id)[1];
        if y >= y_lo && y <= y_hi { band.push(id); }
    }
    let tgts = [];
    for j in 0..PER_INPUT {
        tgts.push([band[rand(0.0, 1.0e9).to_int() % band.len()], W_IN]);
    }
    net.add_input("in" + k, tgts,
                  [lo[0] - 0.45 * ext, (y_lo + y_hi) * 0.5, (lo[2] + hi[2]) * 0.5]);
}

// A looping Poisson spike train: `rate` Hz, LOOP_MS long, each spike a
// PULSE_STEPS-step pulse. Seeded through rand() → reproducible.
let train = |rate| {
    let n = LOOP_MS * SPB;
    let a = [];
    a.pad(n, 0.0);
    let t = 0.0;
    loop {
        t += -(rand(1.0e-9, 1.0).ln()) * 1000.0 / rate;
        if t >= LOOP_MS.to_float() { break; }
        let i = (t * SPB).to_int();
        for d in 0..PULSE_STEPS { if i + d < n { a[i + d] = 1.0; } }
    }
    a
};
for k in 0..N_IN {
    sim.loop_input("in" + k, train.call(RATES[k]));
}

print("=== lsm_deep_setup: " + n_liq + "-neuron local liquid + " + N_IN +
      " looping input trains (no training) ===");
print("synapses=" + net.synapse_count() + "  inputs: in0..in" + (N_IN - 1) +
      " at " + RATES + " Hz, each into its own patch of the near face");
// A short look so the console shows it is alive; Run keeps it going.
sim.run(1000.0);
let s1 = sim.spikes();
sim.run(1000.0);
let hz = (sim.spikes() - s1).to_float() / n_liq;
print("alive: liquid rate " + ((hz * 100.0).to_int() / 100.0) + " Hz over the 2nd second (t=" +
      sim.time() + " ms) — press Run, spikes keep flowing; Reset restarts the trains");
print("next: Examples > Deep readout (lsm_deep) trains the readouts on this liquid");
