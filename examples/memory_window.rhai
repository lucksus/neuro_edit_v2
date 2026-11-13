// Memory window v2 (Stage 7): how long does a liquid retain "an event
// just happened"? Corr-event task; the readout window sits DELAY ms
// after event onset, so at delay > 0 the stream alone can't answer —
// only the liquid's state (membrane + per-synapse current traces) can.
//
//   vs stage 6: >=200 windows/split (300 s dataset), 5 liquid seeds
//   (mean ± sd per cell), a label-permutation null per delay, and the
//   per-synapse tau sweep feeding in from build_liquid.
//
//   ONE replay serves all delays: run to each (t0+delay+WIN) boundary
//   and read sim.spike_counts(WIN*SPB) — exact per-window counts.
//
//   start as a server job; every (setting, delay) cell streams a
//   job_progress row the jobs panel plots.

// ---------- parameters ---------- [TEST-BLOCK]
let SEED = 5;
let DS_NAME = "mem_corr2";
let NCH = 8;
let LEN_MS = 300000;         // ~500 events -> ~250 windows/split
let DT_MS = 1.0;
let SPB = 10;
let RATE_HZ = 25.0;
let EV_EVERY = 600.0;
let EV_JITTER = 180.0;
let EV_WIN = 12.0;
let EV_STR = 60.0;

let WIN_MS = 20;
let MARGIN = 100;
let NEG_PRE = 300.0;         // negatives: no event this far back
let NEG_POST = 30.0;         // ...nor this far after the readout window
let RIDGE_L = 1.0;
let NPERM = 200;             // label-shuffle draws for the per-delay null

let DELAYS = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0];
let SEEDS = [5, 7, 11, 19, 42];
// (name, side, lambda, w_rec, w_in, tau_rec, tau_inh)
// tau_rec: per-synapse trace tau on liquid->liquid edges (0 = merged 2ms)
// tau_inh: per-synapse tau on inhibitory-source edges (slow inhibition)
let SETTINGS = [
    ["merged",      8, 90.0, 3.0, 12.0,   0.0,  0.0],
    ["rec5",        8, 90.0, 3.0, 12.0,   5.0,  0.0],
    ["rec20",       8, 90.0, 3.0, 12.0,  20.0,  0.0],
    ["rec50",       8, 90.0, 3.0, 12.0,  50.0,  0.0],
    ["rec100",      8, 90.0, 3.0, 12.0, 100.0,  0.0],
    ["rec50-w2",    8, 90.0, 6.0, 12.0,  50.0,  0.0],
    ["slow-inh",    8, 90.0, 3.0, 12.0,  20.0, 50.0],
];
let LIQ_SPACING = 60.0;
let CONN_SCALING = 0.35;
let PER_SRC = 24;

let TRAIN = [0.0, 150000.0];
let TEST = [150000.0, 300000.0];
// ---------- /parameters ---------- [TEST-BLOCK]

// quick(): the examples test / smoke path — same experiment, tiny scale.
if quick() {
    LEN_MS = 16000;
    SEEDS = [5];
    DELAYS = [0.0, 20.0, 100.0];
    SETTINGS = [
        ["merged",      6, 90.0, 3.0, 12.0,   0.0,  0.0],
        ["rec50",       6, 90.0, 3.0, 12.0,  50.0,  0.0],
    ];
    NPERM = 30;
    TRAIN = [0.0, 8000.0];
    TEST = [8000.0, 16000.0];
}

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

let build_liquid = |side, lambda, w_rec, w_in, tau_rec, tau_inh, seed_val| {
    net.clear();
    seed(seed_val);
    net.create_3d_grid("liquid", side, side, side, LIQ_SPACING);
    net.set_exc_inh("liquid", 0.2);
    net.create_connections("liquid", CONN_SCALING, lambda, w_rec);
    if tau_rec > 0.0 {
        net.set_syn_tau("liquid", "liquid", tau_rec);
    }
    if tau_inh > 0.0 {
        net.set_syn_tau_inh("liquid", tau_inh);
    }
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

// One replay produces EVERY delay's features: run up to each
// (t0 + delay + WIN) boundary, then read trailing-window counts.
let liq_feats = |t0| {
    let cur = (t0 - MARGIN).max(0.0).to_int();
    sim.reset();
    let out = [];
    for dl in 0..DELAYS.len() {
        let s1 = (t0 + DELAYS[dl] + WIN_MS.to_float()).to_int();
        for c in 0..NCH {
            let ch = CHS[c];
            let series = [];
            for s in cur..s1 {
                for k in 0..SPB { series.push(ch[s]); }
            }
            sim.play_input("src" + c, series);
        }
        sim.run_steps((s1 - cur) * SPB);
        out.push(sim.spike_counts(WIN_MS * SPB).map(|c| c / WIN_MS));
        cur = s1;
    }
    out
};

// Raw per-channel counts in each delay's window (no sim).
let raw_feats = |t0| {
    let out = [];
    for dl in 0..DELAYS.len() {
        let lo = (t0 + DELAYS[dl]).to_int();
        let hi = lo + WIN_MS;
        let f = [];
        for c in 0..NCH {
            let a = 0.0;
            for s in lo..hi { a += CHS[c][s]; }
            f.push(a / WIN_MS);
        }
        out.push(f);
    }
    out
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

// Per-delay flat feature sets over [lo,hi]: ->
// [Xl[dl][flat], Xr[dl][flat], y, pos_n, neg_n]  (one replay per window)
let gen_sets = |lo, hi| {
    let pos = [];
    for e in EVS {
        if e >= lo + MARGIN && e + DELAYS[DELAYS.len()-1] + WIN_MS.to_float() < hi {
            pos.push(e);
        }
    }
    let neg = [];
    let t = lo + MARGIN + 80.0;
    while neg.len() < pos.len() && t < hi - DELAYS[DELAYS.len()-1] - WIN_MS.to_float() - 80.0 {
        // a negative must be clean at EVERY delay it will serve
        let ok = true;
        for dl in 0..DELAYS.len() {
            if !neg_ok.call(t, DELAYS[dl]) { ok = false; }
        }
        if ok { neg.push(t); }
        t += 61.0;
    }
    let Xl = [];
    let Xr = [];
    for dl in 0..DELAYS.len() { Xl.push([]); Xr.push([]); }
    let y = [];
    for t in pos {
        let lf = liq_feats.call(t);
        let rf = raw_feats.call(t);
        for dl in 0..DELAYS.len() {
            for v in lf[dl] { Xl[dl].push(v); }
            for v in rf[dl] { Xr[dl].push(v); }
        }
        y.push(1.0);
    }
    for t in neg {
        let lf = liq_feats.call(t);
        let rf = raw_feats.call(t);
        for dl in 0..DELAYS.len() {
            for v in lf[dl] { Xl[dl].push(v); }
            for v in rf[dl] { Xr[dl].push(v); }
        }
        y.push(0.0);
    }
    [Xl, Xr, y, pos.len(), neg.len()]
};

// scores for a fit on a per-delay set
let score = |X, n_feat, y, Xt, n_t| {
    let w = fit_ridge(X, y.len(), n_feat, y, RIDGE_L);
    let st = [];
    for i in 0..n_t {
        st.push(ridge_output(w, Xt.extract(i * n_feat, n_feat)));
    }
    st
};

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

let mean = |a| { let s = 0.0; for x in a { s += x; } s / a.len().max(1) };
let sd = |a, m| {
    let s = 0.0;
    for x in a { s += (x - m) * (x - m); }
    (s / a.len().max(1)).sqrt()
};

// ---------- run ----------
print("=== memory window v2: AUC vs readout delay (5 seeds, null per delay) ===");
let dl_hdr = "delay(ms)     ";
for x in DELAYS {
    let c = x.to_string();
    c.pad(9, ' ');
    dl_hdr += c;
}
print(dl_hdr + "   null mean±sd   horizon");

// train/test sets are built once per (setting,seed) — every delay's
// features come out of the same replays.
for st in 0..SETTINGS.len() {
    let s = SETTINGS[st];
    let auc_rows = [];              // [seed][delay]
    let raw_rows = [];
    let null_m = [];
    let null_s = [];
    for dl in 0..DELAYS.len() { null_m.push(0.0); null_s.push(0.0); }
    for si in 0..SEEDS.len() {
        let seed_val = SEEDS[si];
        let n_liq = build_liquid.call(s[1].to_int(), s[2], s[3], s[4],
                                      s[5], s[6], seed_val);
        if si == 0 {
            print("\nsetting " + s[0] + ": " + n_liq + " neurons, " +
                  net.synapse_count() + " syn (tau_rec=" + s[5] +
                  " tau_inh=" + s[6] + ")");
        }
        let tr = gen_sets.call(TRAIN[0], TRAIN[1]);
        let te = gen_sets.call(TEST[0], TEST[1]);
        let a_row = [];
        let r_row = [];
        for dl in 0..DELAYS.len() {
            let stl = score.call(tr[0][dl], n_liq, tr[2], te[0][dl], te[2].len());
            let a_liq = auc(stl, te[2]);
            let str_ = score.call(tr[1][dl], NCH, tr[2], te[1][dl], te[2].len());
            let a_raw = auc(str_, te[2]);
            a_row.push(a_liq);
            r_row.push(a_raw);
            // label-permutation null on the liquid scores (seed 0 only —
            // the null is a label property, not a liquid property)
            if si == 0 {
                let draws = [];
                for p in 0..NPERM {
                    draws.push(auc(stl, shuffled.call(te[2])));
                }
                let nm = mean.call(draws);
                null_m[dl] = nm;
                null_s[dl] = sd.call(draws, nm);
            }
            job_progress(#{setting: st.to_float(), seed: seed_val.to_float(),
                          delay_ms: DELAYS[dl], auc: a_liq, raw_auc: a_raw});
        }
        auc_rows.push(a_row);
        raw_rows.push(r_row);
        print("  seed " + seed_val + ": train " + tr[3] + "+" + tr[4] +
              " test " + te[3] + "+" + te[4]);
    }
    // mean ± sd over seeds per delay; horizon = last delay where
    // mean_auc >= null_mean + 2*null_sd ("-" if it never holds)
    let line = s[0].to_string();
    line.pad(13, ' ');
    let horizon = "-";
    for dl in 0..DELAYS.len() {
        let col = [];
        for si in 0..SEEDS.len() { col.push(auc_rows[si][dl]); }
        let m = mean.call(col);
        let c = f3.call(m).to_string() + "±" + f3.call(sd.call(col, m)).to_string();
        c.pad(11, ' ');
        line += c;
        let thr = null_m[dl] + 2.0 * null_s[dl];
        if m >= thr {
            horizon = DELAYS[dl].to_string();
        }
    }
    print(line + "   " + f3.call(mean.call(null_m)) + "±" +
          f3.call(mean.call(null_s)) + "   horizon=" + horizon);
    // raw-count baseline row (same windows, same delays)
    let rline = "  raw      ";
    rline.pad(13, ' ');
    for dl in 0..DELAYS.len() {
        let col = [];
        for si in 0..SEEDS.len() { col.push(raw_rows[si][dl]); }
        let m = mean.call(col);
        let c = f3.call(m).to_string() + "±" + f3.call(sd.call(col, m)).to_string();
        c.pad(11, ' ');
        rline += c;
    }
    print(rline);
}
// Stay visibly alive: background noise keeps the last-built liquid
// firing after the experiment ends, so Run in the editor still shows
// activity (Stage 24 rule: every example shows activity on start).
sim.set_noise(40.0, 12.0);
sim.run(300.0);
print("done — liquid still firing on noise (spikes=" + sim.spikes() + ")");
