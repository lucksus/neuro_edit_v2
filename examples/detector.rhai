// Event detector on a generated random series: an LSM liquid reads 8
// Poisson-ish channels; "corr" events (a simultaneous burst shared by
// all channels, invisible in any single channel's rate) are planted at
// random-ish times. A ridge readout on the liquid's per-neuron spike
// rates answers "did an event just start?" — label 1 within a short
// readout window after each event time, 0 at event-free positions.
//
//   server + editor: start it as a job (progress streams to the jobs
//   panel); or `sim --script examples/detector.rhai` locally.
//
// Baselines are reported honestly on the SAME windows:
//   raw  — per-channel spike counts (linear over channels)
//   cooc — 28 pairwise coincidence features; the non-liquid baseline
//          that directly counts the planted coincidence signature
// Null model: permutation test over train labels (NPERM shuffles, ridge
// refit each time) — the observed AUC is compared against the null
// distribution, not a single shuffle.
//
// Splits: event times land in train / val / test ranges of the series.
// Validation is never tuned on; it cross-checks the test number.

// ---------- parameters ---------- [TEST-BLOCK]
let SEEDS = [1, 7, 42];
let DS_NAME = "det_corr";
let NCH = 8;
let LEN_MS = 45000;        // 45 s of series
let DT_MS = 1.0;           // dataset sample = 1 ms
let SPB = 10;              // sim steps per sample (dt=0.1 ms)
let RATE_HZ = 25.0;        // base channel rate
let EV_EVERY = 500.0;      // mean gap between events
let EV_JITTER = 150.0;     // ± jitter on the gap
let EV_WIN = 12.0;         // event footprint
let EV_STR = 35.0;         // corr-burst strength — subtle: per-channel
                           // rate barely moves; coincidence carries it

let DELAY_MS = 0.0;        // readout window anchored at event start
let WIN_MS = 20;           // readout window length
let MARGIN = 100;          // ms of stream played before the anchor
let NEG_PRE = 250.0;       // negatives: no event this far back...
let NEG_POST = 60.0;       // ...and this far forward of the window
let RIDGE_L = 1.0;
let NPERM = 200;

let LIQ_SIDE = 8;
let LIQ_SPACING = 60.0;
let CONN_SCALING = 0.35;
let CONN_LAMBDA = 90.0;
let W_REC = 3.0;
let W_IN = 12.0;
let TAU_REC = 0.0;         // per-synapse trace tau on liquid->liquid
                           // synapses (0 = merged 2ms global path)
let TAU_INH = 0.0;         // per-synapse tau on inhibitory sources
let PER_SRC = 24;          // liquid neurons each channel drives

// ranges inside the series (ms): train / val / test
let TRAIN = [0.0, 22500.0];
let VAL = [22500.0, 34000.0];
let TEST = [34000.0, 45000.0];
// ---------- /parameters ---------- [TEST-BLOCK]

// ---------- dataset ----------
let d = dataset_gen(#{name: DS_NAME, channels: NCH, len_ms: LEN_MS,
    dt_ms: DT_MS, dist: "poisson", rate_hz: RATE_HZ, seed: 901,
    events: #{kind: "corr", every_ms: EV_EVERY, jitter_ms: EV_JITTER,
             window_ms: EV_WIN, strength: EV_STR}});
let CHS = [];
for c in 0..NCH { CHS.push(d.channel(c)); }
let EVS = d.events();
print("dataset " + DS_NAME + ": " + d.n_channels + " ch x " + d.steps +
      " steps (" + d.len_ms + " ms), " + EVS.len() + " events");

// ---------- helpers ----------
let f3 = |v| (v * 1000.0).to_int() / 1000.0;
let mean_of = |rows, k| { let s = 0.0; for r in rows { s += r[k]; } s / rows.len() };
let sd_of = |rows, k, m| { let s = 0.0; for r in rows { s += (r[k]-m)*(r[k]-m); } (s / rows.len()).sqrt() };

let build_liquid = |seed_val| {
    net.clear();
    seed(seed_val);
    net.create_3d_grid("liquid", LIQ_SIDE, LIQ_SIDE, LIQ_SIDE, LIQ_SPACING);
    net.set_exc_inh("liquid", 0.2);
    net.create_connections("liquid", CONN_SCALING, CONN_LAMBDA, W_REC);
    if TAU_REC > 0.0 { net.set_syn_tau("liquid", "liquid", TAU_REC); }
    if TAU_INH > 0.0 { net.set_syn_tau_inh("liquid", TAU_INH); }
    let liq = net.neurons_of("liquid");
    let n_liq = liq.len();
    net.add_readout("ro", liq);
    for c in 0..NCH {
        let tgts = [];
        for k in 0..PER_SRC {
            tgts.push([liq[rand(0.0, 1.0e9).to_int() % n_liq], W_IN]);
        }
        net.add_input("src" + c, tgts);
    }
    n_liq
};

// Liquid features for one anchored window: play [t0-MARGIN, t0+DELAY+WIN]
// through the liquid and read per-neuron rates over the readout window.
let liq_feat = |t0| {
    let s0 = (t0 - MARGIN).max(0.0).to_int();
    let s1 = (t0 + DELAY_MS + WIN_MS.to_float()).to_int();
    sim.reset();
    for c in 0..NCH {
        let ch = CHS[c];
        let series = [];
        for s in s0..s1 {
            for k in 0..SPB { series.push(ch[s]); }
        }
        sim.play_input("src" + c, series);
    }
    sim.run_steps((s1 - s0) * SPB);
    sim.spike_counts(WIN_MS * SPB).map(|c| c / WIN_MS)
};

// Raw baseline: per-channel sample counts in the readout window.
let raw_feat = |t0| {
    let lo = (t0 + DELAY_MS).to_int();
    let hi = lo + WIN_MS;
    let f = [];
    for c in 0..NCH {
        let a = 0.0;
        for s in lo..hi { a += CHS[c][s]; }
        f.push(a / WIN_MS);
    }
    f
};

// Coincidence baseline: all C(8,2) pairwise co-occurrence counts over
// the readout window — the direct non-liquid measure of the planted
// signal.
let cooc_feat = |t0| {
    let lo = (t0 + DELAY_MS).to_int();
    let hi = lo + WIN_MS;
    let f = [];
    for i in 0..NCH {
        for j in (i + 1)..NCH {
            let c = 0.0;
            for s in lo..hi {
                if CHS[i][s] > 0.0 && CHS[j][s] > 0.0 { c += 1.0; }
            }
            f.push(c / WIN_MS);
        }
    }
    f
};

let MODE_LIQ = 0;
let MODE_RAW = 1;
let MODE_COOC = 2;
let feat_of = |mode, t0| {
    if mode == MODE_RAW { raw_feat.call(t0) }
    else if mode == MODE_COOC { cooc_feat.call(t0) }
    else { liq_feat.call(t0) }
};

// Negatives: positions in [lo+margin, hi-(delay+win+pad)] with no event
// inside NEG_PRE back / NEG_POST forward — "no recent event" windows.
let neg_ok = |t| {
    let lo = t - NEG_PRE;
    let hi = t + DELAY_MS + WIN_MS.to_float() + NEG_POST;
    let ok = true;
    for e in EVS {
        if e >= lo && e <= hi { ok = false; }
    }
    ok
};

// Feature set for a time range: [X flat, y, n_feat].
let gen_set = |mode, lo, hi| {
    // positives = events whose full readout window fits in the range
    let pos = [];
    for e in EVS {
        if e >= lo + MARGIN && e + DELAY_MS + WIN_MS.to_float() < hi {
            pos.push(e);
        }
    }
    // negatives = event-free positions, same count, evenly spread
    let neg = [];
    let t = lo + MARGIN + 50.0;
    while neg.len() < pos.len() && t < hi - DELAY_MS - WIN_MS.to_float() - 50.0 {
        if neg_ok.call(t) { neg.push(t); }
        t += 47.0;
    }
    let X = [];
    let y = [];
    for t in pos {
        for v in feat_of.call(mode, t) { X.push(v); }
        y.push(1.0);
    }
    for t in neg {
        for v in feat_of.call(mode, t) { X.push(v); }
        y.push(0.0);
    }
    [X, y, pos.len(), neg.len()]
};

// Fisher–Yates shuffle for the permutation null.
let shuffled = |labels| {
    let l = labels;
    let n = l.len();
    for k in 0..n {
        let i = n - 1 - k;
        let j = rand(0.0, 1.0e9).to_int() % (i + 1);
        let tmp = l[i]; l[i] = l[j]; l[j] = tmp;
    }
    l
};

// Fit ridge on [X,y] (flat T × n_feat), score flat test + val sets row
// by row; return [test_auc, test_acc, val_auc] (val_auc = 0.5 when no
// val rows pass).
let eval_fit = |X, n_feat, y, Xt, yt, Xv, yv| {
    let w = fit_ridge(X, y.len(), n_feat, y, RIDGE_L);
    let st = [];
    for i in 0..yt.len() {
        st.push(ridge_output(w, Xt.extract(i * n_feat, n_feat)));
    }
    let va = 0.5;
    if Xv.len() > 0 {
        let sv = [];
        for i in 0..yv.len() {
            sv.push(ridge_output(w, Xv.extract(i * n_feat, n_feat)));
        }
        va = auc(sv, yv);
    }
    [auc(st, yt), accuracy(st, yt), va]
};

// ---------- run ----------
print("=== detector: corr events on " + NCH + " ch, " + f3.call(EV_EVERY) +
      "ms apart, readout +" + DELAY_MS + "ms/" + WIN_MS + "ms, tau_rec=" +
      TAU_REC + " tau_inh=" + TAU_INH + " ===");

let rows = [];
let null_draws = [];
let null_above = 0;
for si in 0..SEEDS.len() {
    let seed_val = SEEDS[si];
    let n_liq = build_liquid.call(seed_val);
    if si == 0 {
        print("seed " + seed_val + ": liquid " + n_liq + " neurons, " +
              net.synapse_count() + " synapses");
    }
    // generate once per mode; train/val/test by time range
    let mliq = gen_set.call(MODE_LIQ, TRAIN[0], TRAIN[1]);
    let mliq_v = gen_set.call(MODE_LIQ, VAL[0], VAL[1]);
    let mliq_t = gen_set.call(MODE_LIQ, TEST[0], TEST[1]);
    if si == 0 {
        print("windows: train " + mliq[2] + "+" + mliq[3] +
              "  val " + mliq_v[2] + "+" + mliq_v[3] +
              "  test " + mliq_t[2] + "+" + mliq_t[3]);
    }
    let r = eval_fit.call(mliq[0], n_liq, mliq[1], mliq_t[0], mliq_t[1],
                          mliq_v[0], mliq_v[1]);

    let mraw_t = gen_set.call(MODE_RAW, TEST[0], TEST[1]);
    let mraw = gen_set.call(MODE_RAW, TRAIN[0], TRAIN[1]);
    let r_raw = eval_fit.call(mraw[0], NCH, mraw[1], mraw_t[0], mraw_t[1],
                              [], []);

    let mc_t = gen_set.call(MODE_COOC, TEST[0], TEST[1]);
    let mc = gen_set.call(MODE_COOC, TRAIN[0], TRAIN[1]);
    let r_cooc = eval_fit.call(mc[0], (NCH*(NCH-1)/2).to_int(), mc[1],
                               mc_t[0], mc_t[1], [], []);

    // permutation null on the liquid features
    let above = 0;
    for b in 0..NPERM {
        let w = fit_ridge(mliq[0], mliq[1].len(), n_liq,
                          shuffled.call(mliq[1]), RIDGE_L);
        let s = [];
        for i in 0..mliq_t[1].len() {
            s.push(ridge_output(w, mliq_t[0].extract(i * n_liq, n_liq)));
        }
        let a = auc(s, mliq_t[1]);
        null_draws.push(a);
        if a >= r[0] { above += 1; null_above += 1; }
    }
    let p = (1.0 + above) / (NPERM + 1.0);

    rows.push([r[0], r_raw[0], r_cooc[0], r[2]]);
    print("seed " + seed_val +
          "  liquid auc=" + f3.call(r[0]) + " (val=" + f3.call(r[2]) +
          ", acc=" + f3.call(r[1]) + ")" +
          "  raw=" + f3.call(r_raw[0]) + "  cooc=" + f3.call(r_cooc[0]) +
          "  p=" + f3.call(p));
    job_progress(#{seed: seed_val, auc: r[0], val_auc: r[2],
                  raw_auc: r_raw[0], cooc_auc: r_cooc[0], p: p});

    if si == 0 && job_dir() != "" {
        let w = fit_ridge(mliq[0], mliq[1].len(), n_liq, mliq[1], RIDGE_L);
        net.set_readout_weights("ro", w);
        net.save(job_dir() + "/detector.ne2");
    }
}

print("\n=== mean ± sd over " + SEEDS.len() + " seeds ===");
let rep = |name, k| {
    let m = mean_of.call(rows, k);
    print(name + " ridge_auc = " + f3.call(m) + " ± " + f3.call(sd_of.call(rows, k, m)));
};
rep.call("liquid    ", 0);
rep.call("raw       ", 1);
rep.call("cooc (28d)", 2);
let mval = mean_of.call(rows, 3);
print("liquid val  ridge_auc = " + f3.call(mval) +
      " ± " + f3.call(sd_of.call(rows, 3, mval)));
let nm = mean_of.call(null_draws.map(|v| [v]), 0);
let ns = sd_of.call(null_draws.map(|v| [v]), 0, nm);
print("permutation null: mean=" + f3.call(nm) + " sd=" + f3.call(ns) +
      "  pooled p=" + f3.call((1.0 + null_above) / (null_draws.len() + 1.0)) +
      " (" + null_draws.len() + " draws)");
print("done");
