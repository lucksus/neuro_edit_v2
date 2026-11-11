# NOTES-for-data

Working notes for Nico on neuro_edit v2 (Rust + CUDA rewrite).

## What works (commands + numbers)

### Workspace
`cargo build --workspace`, `cargo test --workspace` (19 tests), `cargo clippy
--workspace --all-targets` all clean. Target dir lives at `../neuro_edit_v2-target`
(outside the repo; set via `.cargo/config.toml`).

### Headless sim

```
cargo run -p sim --release -- --bench --bench-k 100 --wscale 0.7 --w-inh 5 --cuda
```

- `--net`, `--script file.rhai`, repeated `--eval`, `--cpu`/`--cuda`,
  `--bench`, `--bench-k`, `--wscale`, `--w-inh`, `--dt`, `--ms`, `--warm`,
  `--noise`, `--seed`. Bare `sim` opens an interactive Rhai console; bare
  `--net x.ne2` runs the net for `--ms`.

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
Features = per-neuron spike rates over the window.

Measured (seed=1, CUDA):

```
main          delta: acc=0.78 auc=0.88  ridge: acc=0.83 auc=0.88
signal off    delta: acc=0.52 auc=0.53  ridge: acc=0.51 auc=0.53
shuffled      delta: acc=0.50 auc=0.58  ridge: acc=0.46 auc=0.48
raw baseline  delta: acc=0.50 auc=0.51  ridge: acc=0.53 auc=0.50
sweep s:      0.0→auc~0.53  0.5→0.80  0.8→0.88  1.0→0.89 (ridge; delta similar)
```

All controls at chance, detection threshold visible in the sweep. Fully
deterministic: identical numbers on CPU and CUDA runs (streams come from
`rng_stream`, a seeded counter-based Rust RNG, not the sim's noise channel).

## What does NOT work yet

- The editor is a stub (`editor/src/main.rs`). No UI exists.
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
  in nondeterministic order → GPU runs are not bitwise identical between runs
  (or vs CPU). For the seeded example this doesn't matter (streams are
  host-generated). Integer/fixed-point weights would fix it; noted as an
  option, not implemented.
- **Sim threading for the editor**: plan is an in-process sim thread owning
  `Sim`, stepping under a speed governor, publishing a spike-bitmap ring +
  activity snapshot under a `RwLock`; UI never blocks sim. To be validated.
- `examples/lsm.ne2` is the tiny 2-population demo net from the Rhai smoke
  test; `lsm_hidden.rhai` builds its own liquid programmatically.

## Open FEATURE-PARITY items

Everything under "Editor" in FEATURE-PARITY.md is open; all sim-core and
script-API items are done (see checkboxes there).
