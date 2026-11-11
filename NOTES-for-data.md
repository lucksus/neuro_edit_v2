# NOTES-for-data

Working notes for Nico on neuro_edit v2 (Rust + CUDA rewrite).

## What works (commands + numbers)

### Workspace
`cargo build --workspace`, `cargo test --workspace` (26 tests), `cargo clippy
--workspace --all-targets` all clean. Target dir lives at `../neuro_edit_v2-target`
(outside the repo; `CARGO_TARGET_DIR` env var).

### Headless sim

```
cargo run -p sim --release -- --bench --bench-k 100 --wscale 0.7 --w-inh 5 --cuda
```

- `--net`, `--script file.rhai`, repeated `--eval`, `--cpu`/`--cuda`,
  `--bench`, `--bench-k`, `--wscale`, `--w-inh`, `--dt`, `--ms`, `--warm`,
  `--noise`, `--seed`, `--int-weights` (fixed-point ring → bitwise
  reproducible runs; `sim.set_int_weights(1)` in scripts). Bare `sim` opens
  an interactive Rhai console; bare `--net x.ne2` runs the net for `--ms`.

Benchmarks (dt=0.1, K=100 synapses/neuron, wscale=0.7 → ~21 Hz healthy regime,
3000 ms simulated, 3090):

| N | synapses | wall | realtime | GPU mem |
|---|----------|------|----------|---------|
| 10k | 1M | 0.276 s | 10.9x | 18 MB |
| 100k | 10M | 0.549 s | 5.5x | 181 MB |

CPU reference at 10k/1M synapses, 500 ms: 2.06 s wall → 0.24x realtime.
Prop kernel is warp-per-spike (lanes stride the source's CSR row); switching
from thread-per-spike gave ~6x. Per-step launch floor ~9µs at 10k.

GPU-vs-CPU verification: `cargo test -p sim --features cuda` —
`gpu_matches_cpu_spike_counts_quiet` (exact spike counts, no noise) and
`gpu_cpu_close_with_noise` (same seed, rates within tolerance) pass.

### Hidden-correlation LSM example

```
cargo run -p sim --release -- --script examples/lsm_hidden.rhai --cuda
```

`examples/lsm_hidden.rhai` (also works on `--cpu`, slower): 8 random sources
drive a 512-neuron liquid (8x8x8 grid, distance-dependent connections). During
labelled "event" windows each of 4 source *pairs* flips a shared coin
(strength 0.8); per-stream marginals stay iid Bernoulli(0.5) — signal exists
only in pairwise correlation. 200 train / 120 held-out windows, 50 ms each.
Features = per-neuron spike rates over the window. The script loops over 12
seeds, rebuilding the liquid each seed (`net.clear()`).

Measured (12 seeds, CUDA; ~45 s):

```
main          delta_auc = 0.880 ± 0.030   ridge_auc = 0.874 ± 0.032
signal off    delta_auc = 0.484 ± 0.036   ridge_auc = 0.484 ± 0.035
shuffled      delta_auc = 0.500 ± 0.001   ridge_auc = 0.500 ± 0.000
raw baseline  delta_auc = 0.480 ± 0.051   ridge_auc = 0.482 ± 0.046
shuffled single-draw ridge AUC: mean 0.500, sd 0.185
sweep (seed 11): s=0.0→0.45  0.5→0.79  0.8→0.89  1.0→0.94 (ridge auc)
```

**The shuffled-label bug (reviewer's finding, now fixed).** The old control
retrained the readouts on ONE label permutation and reported that AUC. On
this pipeline that single draw is *not* a point estimate at 0.5 — it's a
draw from a wide null: 200 windows in 513 features are interpolated, so the
weight vector fitted to random labels is large and its projection onto the
true class-separation direction is randomly ±, scaling with the signal
strength. Measured: single-draw sd ≈ 0.19, 28% of draws < 0.4 — seeds 7/42
drew 0.28/0.32, which looked "~4 SE below chance" only against the wrong
(within-sample) SE. There is no leak and no metric bug: over the full
permutation group E[AUC] = 0.5 exactly, by complement symmetry
(w(1−y) = −w(y) for ridge ⇒ scores negate ⇒ AUC → 1−AUC). The fix: the
control now averages over NPERM=20 permutation *pairs* (each permutation
and its label complement — antithetic pairs, so the per-seed null lands
essentially exactly on 0.5) and the script also prints the single-draw sd
so the spread stays visible. Regression test: `lsm_null.rs` reproduces the
tail draw (0.344) and the ~0.5 mean at reduced scale.

Fully deterministic: identical numbers on CPU and CUDA runs (streams come
from `rng_stream`, a seeded counter-based Rust RNG, not the sim's noise
channel).

### Editor + screenshots (headless-verified)

```
editor --screenshot out.png --net examples/lsm.ne2 --after-ms 500
editor --screenshot big.png  --net big.ne2 --no-synapses
```

- Worker thread owns `Network`+`Sim`+Rhai (`Env`); UI sends `Cmd`s over a
  channel and reads immutable `Shared` snapshots — sim never blocks on UI.
  `sim.run`/`run_steps` are chunked with progress callbacks so pause/abort
  and live activity work during long scripts.
- Verified headless: `--screenshot` renders the full 3D scene offscreen
  (neurons instanced, synapse lines, group boxes, floor grid, inputs,
  activity colours) → `docs/screenshots/*.png`.
- Measured on 3090: 97,336 neurons + 1,119,357 synapses = **4.4 ms/scene
  render** (neurons only 1.9 ms) — well over 60 fps territory. 512-neuron
  liquid after 300 ms of live sim: activity colouring + spike particles work.
- Gotcha found+fixed: `lcol` was inferred `Vec<[f64;4]>` → buffer held f64
  bytes, shader read f32 → ghost-green synapses. Any all-literal `Vec` pushed
  into `cast_slice` needs an explicit `f32` annotation.

## What does NOT work yet

- Interactive GUI not verified: miguel has no display; `--screenshot` is the
  render-verification path. Picking/box-select/camera/drag code is written
  but untested interactively (marked `[~]` in FEATURE-PARITY).
- Missing vs original: XML .nex export/import, syntax highlighting in the
  script editor, per-synapse τ (v2 uses one global `synapse_tau_ms`), and
  the axon/dendrite *node* objects the original could place individually
  (v2 collapses trees to one delay per synapse pair, by design).
- No `.neb` import (out of scope unless trivial).
- STDP, AdEx/HH, dendritic compartments: out of scope by design.

## Decisions / things to know

- **Synaptic current**: decaying `s` per neuron, `+= w` on arrival, decayed
  after use — original semantics (tc=50 → τ≈2 ms). The reference `izh_bench.py`
  uses delta kicks; my decaying version delivers ~2x the charge, so the
  benchmark weights had to be scaled down (`--wscale 0.7`) to avoid a
  recurrent avalanche. Default `wscale=1.0` saturates everything — revisit
  defaults before shipping the bench as a headline number.
- **DeltaReadout**: original `learn` updates `w += rate*d*(w_i*f_i)` — the
  `w_i*f_i` quirk makes learning stall at small init. Added `proper` flag
  (Rhai `set_proper(1)`): uses bare `f`. Default preserves the original quirk
  for parity. The example sets `proper`.
- **Rhai scoping**: `fn`s and even `const` are not visible inside
  script-level functions in this eval mode; the example uses anonymous
  closures invoked via `.call(...)`. Same constraint will apply to editor
  console scripts.
- **GPU determinism**: `float atomics` in the prop kernel sum synaptic input
  in nondeterministic order → default (float) GPU runs are not bitwise
  identical between runs (or vs CPU). Implemented fix: `--int-weights` /
  `SimConfig::int_weights` / `sim.set_int_weights(1)` compiles a kernel
  variant whose delay ring is 32-bit fixed point (scale 4096); int
  atomicAdd is order-independent, so runs are bitwise reproducible.
  Verified: `gpu_int_weights_bitwise_deterministic` (identical v + spike
  bitmaps across two seeded runs) and `--bench 10000 --int-weights` twice
  → identical 104384 spikes / 20.9 Hz. Weights quantize to 1/4096 —
  `gpu_int_weights_close_to_float` checks the regime stays within 30%
  of the float run (measured ~same at 10k).
- **Sim threading for the editor**: implemented as described — worker owns
  `Env` (net+sim+Rhai), snapshots via `Arc<RwLock<Shared>>`; geometry only
  re-uploaded on `net_version` change; activity/spikes every progress chunk.
- `examples/lsm.ne2` is the tiny 2-population demo net from the Rhai smoke
  test; `lsm_hidden.rhai` builds its own liquid programmatically.

## Open FEATURE-PARITY items

Editor core is implemented: 3D view, selection (click/box/GPU pick + hover
pick), move/insert/connect/delete, copy/cut/paste subnet JSON (menu +
Ctrl+C/X/V + OS clipboard), right-click context menu, groups/grids/weights
dialogs, property inspector, run/step/pause/speed/dt/noise controls, v(t)
+ raster + readout-weights + samples plots, Rhai console + multiline
script editor (scripts saved into .ne2), new/load/save/recent/quit,
headless screenshots, live worker sim. Interactive-only items are `[~]`
pending a display test. Still open: XML I/O, syntax highlighting,
per-synapse τ, node-level axon/dendrite objects (`[-]` by design).
