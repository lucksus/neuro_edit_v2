// Memory window: how long does a liquid retain "an event just
// happened"? Same corr-event task as detector.rhai, but the readout
// window is placed DELAY ms *after* each event — so at delay > 0 the
// input stream contains nothing; only the liquid's fading state can
// carry the answer. Sweep DELAYS × SETTINGS and print the AUC table.
//
//   start as a server job; every (setting, delay) cell streams a
//   job_progress row the jobs panel plots.
//
// Baselines (raw per-channel counts) ride along — with no event in the
// readout window they must sit at ~0.5, which is what "the liquid
// remembers" actually means.

// ---------- parameters ---------- [TEST-BLOCK]
let SEED = 5;
let DS_NAME = "mem_corr";
let NCH = 8;
let LEN_MS = 45000;
let DT_MS = 1.0;
let SPB = 10;
let RATE_HZ = 25.0;
let EV_EVERY = 900.0;      // sparse events — negatives need clean stretches
let EV_JITTER = 200.0;
let EV_WIN = 12.0;
let EV_STR = 60.0;         // strong enough to start near ceiling —
                           // the decay curve is the measurement

let WIN_MS = 20;
let MARGIN = 100;
let NEG_PRE = 500.0;       // negatives: event-free for this long back
let NEG_POST = 50.0;       // ...and this far forward of the window end
let RIDGE_L = 1.0;

let DELAYS = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0];
// (name, side, lambda, w_rec, w_in) — size × connectivity × strength
let SETTINGS = [
    ["base", 8, 90.0, 3.0, 12.0],
    ["weak", 8, 45.0, 1.2, 12.0],
    ["big", 12, 120.0, 4.5, 12.0],
];
let LIQ_SPACING = 60.0;
let CONN_SCALING = 0.35;
let PER_SRC = 24;

let TRAIN = [0.0, 27000.0];
let TEST = [27000.0, 45000.0];
// ---------- /parameters ---------- [TEST-BLOCK]

// ---------- dataset ----------
let d = dataset_gen(#{name: DS_NAME, channels: NCH, len_ms: LEN_MS,
    dt_ms: DT_MS, dist: "poisson", rate_hz: RATE_HZ, seed: 902,
    events: #{kind: "corr", every_ms: EV_EVERY, jitter_ms: EV_JITTER,
             window_ms: EV_WIN, strength: EV_STR}});
let CHS = [];
for c in 0..NCH { CHS.push(d.channel(c)); }
let EVS = d.events();
print("dataset " + DS_NAME + ": " + EVS.len() + " events in " +
      d.len_ms + " ms");

// ---------- helpers ----------
let f3 = |v| (v * 1000.0).to_int() / 1000.0;

let build_liquid = |side, lambda, w_rec, w_in, seed_val| {
    net.clear();
    seed(seed_val);
    net.create_3d_grid("liquid", side, side, side, LIQ_SPACING);
    net.set_exc_inh("liquid", 0.2);
    net.create_connections("liquid", CONN_SCALING, lambda, w_rec);
    let liq = net.neurons_of("liquid");
    let n_liq = liq.len();
    for c in 0..NCH {
        let tgts = [];
        for k in 0..PER_SRC {
            tgts.push([liq[rand(0.0, 1.0e9).to_int() % n_liq], w_in]);
        }
        net.add_input("src" + c, tgts);
    }
    n_liq
};

// Liquid features: play [t0-MARGIN, t0+delay+WIN] and read rates over
// the readout window — the liquid saw the event DELAY ms ago.
let liq_feat = |t0, delay| {
    let s0 = (t0 - MARGIN).max(0.0).to_int();
    let s1 = (t0 + delay + WIN_MS.to_float()).to_int();
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

// Raw baseline over the readout window (honest: at delay > 0 the stream
// itself carries no event → should sit at ~0.5).
let raw_feat = |t0, delay| {
    let lo = (t0 + delay).to_int();
    let hi = lo + WIN_MS;
    let f = [];
    for c in 0..NCH {
        let a = 0.0;
        for s in lo..hi { a += CHS[c][s]; }
        f.push(a / WIN_MS);
    }
    f
};

let neg_ok = |t, delay| {
    let lo = t - NEG_PRE;
    let hi = t + delay + WIN_MS.to_float() + NEG_POST;
    let ok = true;
    for e in EVS {
        if e >= lo && e <= hi { ok = false; }
    }
    ok
};

// [X flat, y, n_pos, n_neg] for a range + delay.
let gen_set = |mode_liq, delay, lo, hi| {
    let pos = [];
    for e in EVS {
        if e >= lo + MARGIN && e + delay + WIN_MS.to_float() < hi {
            pos.push(e);
        }
    }
    let neg = [];
    let t = lo + MARGIN + 80.0;
    while neg.len() < pos.len() && t < hi - delay - WIN_MS.to_float() - 80.0 {
        if neg_ok.call(t, delay) { neg.push(t); }
        t += 61.0;
    }
    let X = [];
    let y = [];
    for t in pos {
        let f = if mode_liq { liq_feat.call(t, delay) } else { raw_feat.call(t, delay) };
        for v in f { X.push(v); }
        y.push(1.0);
    }
    for t in neg {
        let f = if mode_liq { liq_feat.call(t, delay) } else { raw_feat.call(t, delay) };
        for v in f { X.push(v); }
        y.push(0.0);
    }
    [X, y, pos.len(), neg.len()]
};

let eval_auc = |X, n_feat, y, Xt, yt| {
    let w = fit_ridge(X, y.len(), n_feat, y, RIDGE_L);
    let st = [];
    for i in 0..yt.len() {
        st.push(ridge_output(w, Xt.extract(i * n_feat, n_feat)));
    }
    auc(st, yt)
};

// ---------- run ----------
print("=== memory window: AUC vs readout delay ===");
let dl_hdr = "";
for x in DELAYS {
    let c = x.to_string();
    c.pad(9, ' ');
    dl_hdr += c;
}
print("delay(ms)     " + dl_hdr);

let table = [];
for st in 0..SETTINGS.len() {
    let s = SETTINGS[st];
    let n_liq = build_liquid.call(s[1].to_int(), s[2], s[3], s[4],
                                  SEED + st * 1000);
    print("\nsetting " + s[0] + ": " + n_liq + " neurons, " +
          net.synapse_count() + " syn (lambda=" + s[2] + " w=" + s[3] + ")");
    let row = [];
    for dl in 0..DELAYS.len() {
        let delay = DELAYS[dl];
        let tr = gen_set.call(true, delay, TRAIN[0], TRAIN[1]);
        let te = gen_set.call(true, delay, TEST[0], TEST[1]);
        let a_liq = eval_auc.call(tr[0], n_liq, tr[1], te[0], te[1]);
        // raw baseline on the same windows
        let tr_r = gen_set.call(false, delay, TRAIN[0], TRAIN[1]);
        let te_r = gen_set.call(false, delay, TEST[0], TEST[1]);
        let a_raw = eval_auc.call(tr_r[0], NCH, tr_r[1], te_r[0], te_r[1]);
        row.push(a_liq);
        print("  +" + delay + "ms: liquid=" + f3.call(a_liq) +
              "  raw=" + f3.call(a_raw) +
              "  (train " + tr[2] + "+" + tr[3] + ")");
        job_progress(#{setting: st.to_float(), delay_ms: delay,
                      auc: a_liq, raw_auc: a_raw});
    }
    table.push(row);
}

// ---------- table ----------
print("\ndelay(ms)     " + dl_hdr);
for st in 0..SETTINGS.len() {
    let line = SETTINGS[st][0].to_string();
    line.pad(13, ' ');
    for dl in 0..DELAYS.len() {
        let c = f3.call(table[st][dl]).to_string();
        c.pad(9, ' ');
        line += c;
    }
    print(line);
}
print("done");
