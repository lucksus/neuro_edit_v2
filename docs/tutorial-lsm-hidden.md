# Tutorial: the hidden-correlation LSM experiment

This walks through `examples/lsm_hidden.rhai` — the milestone use case —
line by line, then shows how to reproduce each stage interactively in the
editor. Run it first:

```sh
sim --script examples/lsm_hidden.rhai          # CUDA
sim --script examples/lsm_hidden.rhai --cpu    # same on the CPU backend
```

Takes ~10 min (12 seeds × (200 + 120 + 120) liquid windows × 50 ms of sim
time, plus 500 permutation fits per seed) and prints a per-seed line, then
a `mean ± sd` summary:

```
seed 1: liquid 512 neurons, 2256 synapses
seed 1  main auc=0.868 (val=0.84)  p=0.003  off=0.527  raw=0.501  cooc=1.0
        null=0.509±0.151
…
=== mean ± sd over 12 seeds ===
main         delta_auc = …   ridge_auc ≈ 0.88 ± 0.03
main (val)   ridge_auc ≈ 0.87
signal off   … ≈ 0.5
raw baseline … ≈ 0.5
cooc (28-d)  … = 1.000
permutation null: mean ≈ 0.5 sd ≈ 0.14  pooled p < 0.001
```

Exact numbers are in `NOTES-for-data.md`. Two honest takeaways: the main
result is real (permutation p ≈ 0.001–0.003 per seed), **and** the trivial
28-feature coincidence baseline solves this synthetic task perfectly —
the liquid's value is generality, not this particular task.

## 1. The problem it solves

Eight bit streams, each i.i.d. Bernoulli(0.5). During *event* windows,
fixed pairs of sources (0↔1, 2↔3, 4↔5, 6↔7) share a common coin with
probability `STRENGTH = 0.8` — so during events the two streams of a pair
are correlated. Every stream's own statistics are unchanged: the signal is
invisible to any per-source feature (that's the `raw` control) — only the
joint activity carries it.

A 512-neuron recurrent liquid with distance-based delays turns coincident
input spikes into liquid-state differences that a linear readout can
separate. If the liquid does its job, a readout trained on liquid features
detects the events; trained on raw per-source counts it cannot.

## 2. Script anatomy

### Parameters (`let SEEDS`, `N_SRC`, `PAIRS`, …)

```rhai
let SEEDS = [1, 7, 42, 2, 3, 4, 5, 6, 8, 9, 10, 11];   // 12-seed loop
let PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]];         // correlated pairs
let STRENGTH = 0.8;        // probability of the shared coin per step
let WIN_MS = 50; let SPB = 10;   // 50 ms windows, 10 steps per bit
```

`SPB=10` at `dt=0.1` means 1 bit per ms — each bit is held for 10 sim steps.

### `build_liquid(seed)` — fresh liquid per seed

```rhai
net.clear();
seed(seed_val);
net.create_3d_grid("liquid", 8, 8, 8, 60.0);   // 512 neurons, 60 unit spacing
net.set_exc_inh("liquid", 0.2);                // 20% inhibitory
net.create_connections("liquid", 0.35, 90.0, 3.0);  // p = 0.35·e^(−(d/90)²)
```

Then the wiring that makes the task *hard but learnable*: each pair shares
a pool of 24 neurons (so coincidences have somewhere to land), and each
source additionally gets 12 private neurons so all eight streams are
equally represented individually:

```rhai
net.add_readout("ro", liq);                    // units = all liquid neurons
for p in PAIRS {
    pool = 24 random liquid neurons (shared between the pair's sources)
    for s in p:
        tgts = pool@W_IN + 12 private random neurons@W_IN
        net.add_input("src" + s, tgts)         // W_IN=12: 1 hit sub-threshold
}
```

`net.clear()` per seed is what makes the loop sound — no state leaks
between seeds; `seed(seed_val)` reseeds both builder draws and `rand()`.

### `play_window(m)` — liquid features

```rhai
sim.reset();                                    // clear state, keep weights
for s in 0..8: sim.play_input("src"+s, bits-of-source-s ×10 steps each)
sim.run_steps(WIN_MS * SPB);                    // 500 steps
sim.spike_counts(WIN_MS * SPB) / WIN_MS         // per-neuron rate feature
```

The feature vector is the liquid's per-neuron spike rate over the window —
513-ish numbers (`n_liq` + nothing extra). `sim.reset()` between windows is
essential: without it, one window's leftover activity leaks into the next.

### `gen_set(seed, strength, mode)` — train/val/test sets

Uses three `rng_stream`s (`seed+11` train, `seed+55` validation, `seed+77`
test — disjoint draws so no window is shared across splits). Labels
alternate `it % 2`. `mode` picks the feature extractor: `MODE_LIQ` (liquid
rates), `MODE_RAW` (per-source counts), `MODE_COOC` (28 pairwise
coincidences). The validation split is never trained or tuned on — it only
cross-checks the test number.

### `eval_fit(X, n_feat, ytr, Xte, labs, Xval, labsval)` — the readouts

Two readouts fit on the same 200 windows and score the held-out splits:

- **delta**: `delta_new(n)` + `init_random` + `set_proper(1)` (the
  "proper" corrected delta rule), 10 epochs at `LR=0.5`.
- **ridge**: closed-form `fit_ridge(X, 200, n_feat, y, λ=1.0)`.

Both are scored by `accuracy` (0.5 threshold) and `auc` (threshold-free)
on the test split; the ridge additionally reports a validation AUC.

### The controls and the permutation test

```rhai
signal off : gen_set(seed, 0.0,      MODE_LIQ)  — no planted correlation
raw        : gen_set(seed, STRENGTH, MODE_RAW)  — per-source counts, no liquid
cooc       : gen_set(seed, STRENGTH, MODE_COOC) — 28 pairwise-coincidence
                                                features, no liquid
permutation: NPERM=500 shuffles of the main labels; the ridge is refit per
             draw and the fraction of draws reaching the observed AUC is
             the p-value
```

**Why a permutation test, not one shuffle?** A single permutation is not a
valid null here: with 200 training windows in ~500 dimensions the fitted
noise direction has a large random projection onto the true class axis, so
one draw can sit anywhere in [0,1] (measured sd ≈ 0.14; seeds 7 and 42 once
reported ~0.3 — that was the reviewer complaint). 500 draws per seed give a
real null distribution — mean ≈ 0.5, sd ≈ 0.14 — and a p-value with ~0.002
resolution. (An earlier version paired each permutation with its complement
to force the mean to exactly 0.5 by symmetry; the honest version just uses
more draws.)

**About `cooc` beating the liquid:** the coincidence features count equal
bits per source pair — exactly the statistic the signal is planted in. On
this synthetic task it reaches AUC 1.0 at strengths where the liquid gets
0.8–0.9. That's expected and reported, not hidden: the liquid is the
*general-purpose* machine (it would also pick up nonlinear mixtures the
linear coincidence features can't see); the coincidence baseline is a
task-matched decoder. The experiment shows the liquid genuinely detects
the hidden signal (p ~ 0.001–0.003), not that it's the best possible
decoder for this task.

## 3. Reproduce it interactively in the editor

```sh
editor --net examples/lsm.ne2     # the file saved by seed 1 of the script
```

or build it by hand:

1. **Insert → 3D grid…**: group `liquid`, 8×8×8, spacing 60 → Create.
2. **Group → Set exc/inh…**: group `liquid`, frac 0.2 → Apply.
3. **Group → Create connections…**: `liquid`, scaling 0.35, lambda 90,
   weight 3 → Create (2256 synapses at seed 1 — a few thousand, not tens
   of thousands: the Gaussian `p = 0.35·e^(−(d/90)²)` falls off fast).
4. Console: `net.add_input("src0", [[0,12],[1,12],[2,12]])` — or box-select
   some neurons first, then **Insert → Input…**. A new input starts at
   value 0, so drive it with `sim.set_input("src0", 1)`.
5. `sim.set_noise(60, 18)` — a Poisson kick ~18 mV per hit, so the raster
   actually lights up (`25, 0.5` stays sub-threshold). Box-select neurons
   and `sim.run(500)`: the first 8 selected neurons are the `v(t)` watch
   list (the plots header says how many), and the raster tab shows every
   spiking neuron.
6. Drive it like the script does:
   `sim.play_input("src0", [1,1,1,0,0,1,…])`, `sim.run_steps(500)`,
   `sim.spike_counts(500)`.
7. **File → Save as…** writes a `.ne2` you can reload with `net.load` or
   `sim --net`.

## 4. Knobs worth turning

- `STRENGTH` (0.0–1.0): the sweep at the end of the script shows the
  detection threshold — ~0.5 already works, 0.8 is comfortable.
- `W_IN`: raise it and single-source bits start spiking directly (task gets
  easier and the `raw` control creeps up); lower it and the liquid sees
  nothing.
- `TRAIN_N`/`TEST_N`: more windows → tighter, slower.
- `SEEDS`: the script is the multi-seed loop; keep ≥ 10 seeds or the sd is
  meaningless.
- `NPERM`: 500 shuffles gives p resolution ~0.002 per seed; it only refits
  the ridge on cached features, so it's cheap.
