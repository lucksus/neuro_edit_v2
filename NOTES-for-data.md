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
only in pairwise correlation. 200 train / 120 validation / 120 test windows,
50 ms each, from three independent generator streams. Features = per-neuron
spike rates over the window. The script loops over 12 seeds, rebuilding the
liquid each seed (`net.clear()`).

Measured (12 seeds, CPU; ~15 min):

```
main          delta_auc = 0.880 ± 0.030   ridge_auc = 0.874 ± 0.032
main (val)                              ridge_auc = 0.860 ± 0.028
signal off    delta_auc = 0.484 ± 0.036   ridge_auc = 0.484 ± 0.035
raw baseline  delta_auc = 0.480 ± 0.051   ridge_auc = 0.482 ± 0.046
cooc (28-d)   delta_auc = 1.000 ± 0.000   ridge_auc = 1.000 ± 0.000
permutation null (6000 draws): mean 0.502, sd 0.189
pooled p = 0.001   per-seed p: [0.001, 0.005], mean 0.003
sweep (seed 11): s=0.0 liq 0.445 / cooc 0.533   s=0.5 liq 0.793 / cooc 1.000
                 s=0.8 liq 0.894 / cooc 1.000   s=1.0 liq 0.937 / cooc 1.000
```

**What the numbers say.** The liquid detects the hidden correlation, and the
effect is real: no permutation draw beat the observed AUC on 7 of 12 seeds
(p=0.001), worst case p=0.005, pooled p=0.001 over 6000 draws — while the
signal-off and raw-input controls sit at chance. The validation split
(0.860 vs test 0.874) confirms the test number is not overfit. **But the
28-feature pairwise-coincidence baseline solves this task outright (AUC
1.000 at every nonzero strength), clearly beating the liquid's 0.79–0.94.**
That's the honest result: the coincidence features are a task-matched
decoder — they count equal bits per source pair, which is exactly where
the signal lives. The liquid is the general-purpose machine (it would also
pick up structure linear coincidence features cannot see), but on this
synthetic task a trivial statistic wins. The experiment demonstrates that
the liquid genuinely detects hidden correlations, not that it is the best
decoder for this task.

**The shuffled-label bug (reviewer's finding, history).** The original
control retrained the readouts on ONE label permutation and reported that
AUC. On this pipeline that single draw is *not* a point estimate at 0.5 —
it's a draw from a wide null: 200 windows in 513 features are interpolated,
so the weight vector fitted to random labels is large and its projection
onto the true class-separation direction is randomly ±, scaling with the
signal strength. Measured: single-draw sd ≈ 0.19 — seeds 7/42 drew
0.28/0.32, which looked "~4 SE below chance" only against the wrong
(within-sample) SE. There is no leak and no metric bug: over the full
permutation group E[AUC] = 0.5 exactly, by complement symmetry
(w(1−y) = −w(y) for ridge ⇒ scores negate ⇒ AUC → 1−AUC). An intermediate
fix averaged over 20 permutation/complement pairs to pin the mean at 0.5;
the current version is a real permutation test — NPERM=500 independent
label shuffles per seed, ridge refit per draw, and the fraction of draws
reaching the observed AUC reported as the p-value together with the null
mean/sd. Regression test: `lsm_null.rs` reproduces the tail draw (0.344)
and the ~0.5 null mean at reduced scale.

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
  activity colours) → `docs/screenshots/*.png`. Extra flags:
  `--cam-dist/--cam-yaw/--cam-pitch` override the auto-framed camera.
- Measured on 3090: 97,336 neurons + 1,119,357 synapses = **4.4 ms/scene
  render** (neurons only 1.9 ms) — well over 60 fps territory. 512-neuron
  liquid after 300 ms of live sim: activity colouring + spike particles work.
- **Dense-net overdraw fixed (run 2).** At wide zoom the fixed alpha=0.25
  per synapse turned 1.1M lines into an opaque slab (old liquid_97k.png).
  `Gpu::syn_lod` now computes per-frame (a) a deterministic 1-of-`keep`
  synapse subset in `vs_syn` once mean neuron spacing drops below ~3 px,
  and (b) a density-aware alpha: mean line overdraw inside the projected
  silhouette (`n_drawn·len_px / max(screen_px, silhouette_px)`) is folded
  into a per-line alpha that accumulates to ~55 % coverage — zooming into
  a sparse region restores full alpha automatically (overdraw < 1 ⇒ mul 1).
  Re-rendered on a rebuilt 46³/1.15M-synapse liquid (`create_3d_grid` +
  `create_connections(0.7, 90)` ≈ the original 1.12M): translucent volume
  with surface neurons readable instead of a slab; verified at dist
  1400–14000 and unchanged on sparse nets (alpha_mul = keep = 1).
- **Camera floor clamp.** `Camera::clamp_above_floor` runs in every camera
  mutator: pitch is capped so `eye().y ≥ 2`, and a target dragged below the
  floor is lifted. Verified `--cam-pitch 85` renders level, not under-floor;
  unit tests cover orbit/pan/high-target cases.
- Gotcha found+fixed: `lcol` was inferred `Vec<[f64;4]>` → buffer held f64
  bytes, shader read f32 → ghost-green synapses. Any all-literal `Vec` pushed
  into `cast_slice` needs an explicit `f32` annotation.

## What does NOT work yet

- Interactive GUI verified via user-space Xvfb (see §"Interactive
  verification"); what remains unexercised there: WASD/QE pan, double-click
  focus, shift-add selection, cut/copy/paste keys, save/load dialogs,
  plane switching during Move. `--screenshot` remains the render CI path.
- Missing vs original: XML .nex export/import, per-synapse τ (v2 uses one
  global `synapse_tau_ms`; needs per-synapse trace state in both kernels),
  and the axon/dendrite *node* objects the original could place
  individually (v2 collapses trees to one delay per synapse pair, by
  design).
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
- `examples/lsm.ne2` is the 512-neuron liquid (8³ grid + inputs + trained
  readout) that `lsm_hidden.rhai` saves at the end of seed 1 — the same
  net the script builds programmatically per seed.

## Interactive verification (Xvfb, run 2)

No display/no sudo on miguel, but `apt download xvfb` + deps works without
root. Xvfb ran on `:99` (1600x1000x24); the editor renders through lavapipe
(`libvulkan_lvp.so`) — DRI3 warnings are harmless. Input driven with
`xdotool` (focus the window first, `windowfocus`; no WM so `windowactivate`
fails), captures via `ffmpeg -f x11grab` or `xwd` + PIL.

Verified with real input events (not code review):
- Click-pick neuron → inspector shows it (several neurons).
- Box select via incremental drag → 252 / 205 objects selected. (A drag
  teleported in one step yields a zero-area box — egui's drag threshold
  starts the box at the destination; must move in steps.)
- Right-click context menu on empty space and on a neuron; "Watch v(t)"
  → "watching 1 neuron".
- Run/pause, `sim.set_noise(15.0, 25.0)` via console → rate ~10 Hz.
- v(t) plot: 8 watched neurons, live traces with real spikes.
- raster tab: dense live raster.
- readouts tab: trained ridge weights bar chart (from lsm.ne2).
- Connect mode (`c`): click 445, click 413 → synapse created
  (`net.synapse_count()` 2256→2257; edit bumps net_version → sim rebuilds,
  t resets as designed).
- Move mode (`m`): drag on x-z plane, neuron #429 (60,60,120) →
  (−556.9,60,474.7).
- Insert (`i`) + click → neuron placed; Del deletes (512→511).
- Esc/shortcut keys work; typing in the console no longer fires them.

Four GUI-only bugs found + fixed along the way:
- **Blit surface format**: `Gpu::new` hardcoded `Rgba8UnormSrgb` but the
  egui surface is `Bgra8Unorm` → instant panic on first frame. Now takes
  the real `target_format` (headless still uses Rgba8UnormSrgb).
- **Click-pick starvation**: the every-frame hover pick kept the single
  pick slot busy, so click picks were silently dropped ~half the time.
  `viewport()` now polls the completed pick before issuing requests, so a
  completed hover pick frees the slot for the click pick in the same frame.
- **Shortcuts while typing**: global single-letter keys (`i`/`c`/`m`/WASD/QE)
  fired while a text field had focus (`i` in "sim." switched to Insert).
  `keys()` + camera pan now return early on `wants_keyboard_input()`.
- **Raster/particles permanently empty**: the worker overwrote the script
  env's `record_spikes: true` with `SimConfig::default()` (false), so
  `spike_counts` always returned zeros. Worker now forces `record_spikes`
  + a `COUNTS_WINDOW`-deep ring. Also replaced the sliding-window diff
  (underflowed to 0 when the ring slid between publishes) with a direct
  interval sum `spike_counts(steps_since_last_publish)` tracked by
  `Pub::prev_step`.

## Open FEATURE-PARITY items

Editor core is implemented and the interactive list above is verified:
3D view, selection (click/box/GPU pick + hover pick),
move/insert/connect/delete, copy/cut/paste subnet JSON (menu +
Ctrl+C/X/V + OS clipboard), right-click context menu, groups/grids/weights
dialogs, property inspector, run/step/pause/speed/dt/noise controls, v(t)
+ raster + readout-weights + samples plots, Rhai console + multiline
script editor (scripts saved into .ne2), new/load/save/recent/quit,
headless screenshots, live worker sim, Rhai syntax highlighting in the
script editor + console (`editor/src/syntax.rs`). Still open: XML I/O,
per-synapse τ, node-level axon/dendrite objects (`[-]` by design).

## Run 2 — GUI sweep continued + docs (this session)

Fixes on top of the earlier Xvfb findings:
- Right-drag orbit no longer draws a selection box (drag events are gated
  to the primary button: `drag_started_by/_stopped_by/dragged_by`).
- Cube faces fixed: the `CUBE` vertex table had the +y/-y faces wound
  inward (culled from outside — the "L shapes" / "no top face" report),
  +z missing entirely, +x duplicated; `CUBE_N` had -x/+z normals swapped.
- Pick targets fattened in the pick pass (`misc.w` flag → vs_neuron scales
  cubes up) so dense scenes are clickable.
- Connect mode: a missed click no longer drops the pending source
  (status "missed — click a target neuron"); Esc still cancels.
- Box select supports Shift-additive (matches click-select).
- Rhai float params accept int literals everywhere (Dynamic + d_f64);
  dialogs emit `{:?}` so `120.0` doesn't degrade to `120`.
- `net.connect` (script) now invalidates the built sim like every other
  mutating call.
- Original NeuroEdit toolbar artwork reused (add_neuron/connect/play/
  pause/load/save icons + About splash).

Systematic Xvfb sweep verified (real input events): all menus, every
dialog (2D/3D grid, input, samples, readout, connections, weights,
exc/inh, add-to-group, save-as, load, about), all context menus incl.
connect-from-here and frame-network, copy/cut/paste (liquid 519→509→519),
script editor (highlighting, Run, Save/Open round trip, Save-to-net),
console eval/history/Clear, backend combo cuda↔cpu, dt/noise fields,
speed slider 20×, plane selectors, synapses/grid/script checkboxes, WASD
pan, dbl-click focus, Shift-add select, Recent files.

Docs added: `docs/features.md` (full feature inventory + architecture),
`docs/scripting.md` (complete Rhai API reference), `docs/tutorial.md`
(LSM walkthrough), README screenshot (`docs/screenshots/editor_lsm.png`,
tracked via targeted .gitignore exception; `liquid_97k.png` now tracked
too).

Still open: per-synapse τ (needs per-synapse trace state in both kernels —
real physics change, deliberately not bolted on), XML I/O `[-]`,
node-level axon/dendrite objects `[-]` by design, MLP/MNIST/PHOTOSS `[-]`.

Checks: cargo fmt clean; cargo test --workspace 13/13 binaries pass with
and without `sim/cuda`.

## Stage 2 — review fixes (review/STAGE2.md, this run)

Done in the review's order; each item is a separate commit.

- **P1-1 spike ring** (`aa55bf4`): `CpuSim` `|=`/`CudaSim` `atomicOr`
  never cleared ring rows, so long windows double-counted and wrapped
  rows reported stale spikes. Both backends now memset the *next* row
  after each step, `spike_row` clamps `back` to `t`, and
  `spike_counts(rows)` is `Result`: `rows >= t` returns exact lifetime
  totals (new `pn_total` counter), `rows > ring` errors, else the exact
  window sum. `sim/tests/spike_ring.rs` reproduces red on both backends
  (true 0 reported as 14 on CPU) and is green on the fix, incl. a
  5000-step window.
- **P1-2 robustness** (`4904d81`, `519104d`): `.ne2` load validates
  synapse/input/readout/group indices and rejects malformed files
  (`net_format::validate`, wired into `io::load`/`load_json`);
  `insert_subnet` skips out-of-range refs instead of panicking. All
  script neuron ids are bounds-checked → script errors (not panics).
  CUDA host/device calls propagate `Result` through `run_steps`/`run_ms`
  instead of `expect` panics; empty nets are safe on both backends
  (padded allocations, guarded reset/read paths). The editor worker wraps
  script eval in `catch_unwind` and stays alive after panics/errors.
  Tests: `sim/tests/robustness.rs`, `bounds.rs`, net-format io/network
  unit tests.
- **P2-5 pause stops scripts** (`2e20d07`): the worker installs an Rhai
  `on_progress` hook (fires per-operation, incl. inside `while` loops)
  that checks a UI-side atomic abort flag — pause works even while the
  worker is blocked in `eval_str`. Pause button is shown whenever a run
  or script is busy, not only during continuous run. Test: infinite
  `while(true){}` loop aborts and the worker keeps serving commands.
- **P1-3 big scenes** (`e14b3b7`): the 16 B/synapse `lcol` buffer is
  gone — the source endpoint's high bit carries exc/inh and the shader
  picks the color (8 B/synapse total). `pairs` is uploaded/drawn in
  chunks under `max_storage_buffer_binding_size`, capped by
  `max_buffer_size`; `set_scene` indexes `geo.pos` defensively. Screenshot
  readback now uses padded row strides (odd widths no longer panic).
  `editor/tests/big_scene.rs`: 10 M synapses through `set_scene` + a real
  offscreen draw in ~0.85 s; synapse colors visually verified on the LSM
  liquid screenshot.
- **P2-4 honest eval** (`37d64c5`, `c6b400f`): see the LSM section above.
  500 independent permutations/seed (ridge refit each) → null mean 0.502,
  sd 0.189, pooled p=0.001; independent train/val/test streams; 28-feature
  coincidence baseline added — and it *beats* the liquid (1.000 vs 0.88),
  which is now stated plainly in README/tutorial/parity. Tests:
  `sim/tests/honest_eval.rs`, `lsm_null.rs` (kept).

Optional review items also done:

- **P2-7** (`01041ae`): `Env.backend` is an `Rc<Cell<BackendKind>>` read
  live by `ensure_sim` — the toolbar backend combo now affects
  script-built sims; `set_backend` invalidates the sim.
- **P2-8** (`fff6d00`): `Stimulus.play_start` is per-input —
  `play_input` on input B no longer re-zeroes input A.
- **P2-9** (`ec3e720`): noise hash is splitmix64-style full-64-bit
  avalanche mixing (was a weak linear pre-mix), identical on CPU/CUDA.
- **P2-11** (`58b4ea4`): v(t) watch records every simulated step
  (per-watch drain buffers), not one sample per published chunk; the app
  back-dates samples by dt. `ProgressFn` takes `&mut Sim` accordingly.

Checks: `cargo fmt` clean; `cargo test --workspace` all binaries pass
with and without `sim/cuda` (21 test targets, 0 failures on the final
head). Each of the four optional commits compiles standalone (worktree-
verified). Not pushed; no remote configured.

## Stage 3a — grid activity planes (Nico's plane view)

Regular lattices (`create_2d_grid` / `create_3d_grid`, Rhai or Insert
dialog) now render as ONE textured quad when zoomed out — each texel a
member neuron's fading spike trace — instead of millions of cubes.
Three commits: `bf25aac` (net-format), `0cd0bfd` (sim), editor+docs below.

- **Metadata** (`bf25aac`): `Group::grid: Option<GridShape>` = dims
  `[w,h,d]` + origin + spacing + world-axis map; member order = lattice
  order `(i2·dims0 + i0)·dims1 + i1`. `.ne2` is now version 2 (v1 files
  load with `grid: None`; JSON defaults it). `Network::invalidate_grids`
  drops the metadata whenever the lattice is structurally broken —
  `delete_neurons`, `subnet`/`insert_subnet` (partial paste survives on
  full copies only), `add_to_group`, every move path (`apply_move`,
  inspector pos edit, `net.set_pos`, `net.connect`). A broken grid falls
  back to cubes — never draws a wrong sheet.
- **Traces** (`0cd0bfd`): per-grid f32 spike-trace buffers,
  `set_grid_watches` + `grid_traces(tau)` on both backends. CPU accumulates
  host-side per step; CUDA writes device buffers in `neuron_kernel`
  (spike → 1.0, else × exp(−dt/τ)) and materializes on demand. Same decay
  semantics either side; `sim/tests/grid_traces.rs` checks the math and
  CPU≈CUDA.
- **Editor**: `editor/src/plane.rs` (renderer-free): `GridView`
  Auto/Cubes/Planes, `PlaneCfg` (collapse axis + slice/max/mean), lattice↔
  raster mapping, `quad`, `texel_px`. `GeoSnap::grids` publishes
  position-derived texel-ordered member lists; `Snap::grid_traces` carries
  the per-grid f32 buffers. `Gpu` gains plane textures (R32Float,
  256B-padded rows), a per-texel member-id pick map (R32Uint pick pass →
  click a texel = select that neuron), `PlaneParamsGpu` quads, and a
  shared hide-bit buffer that collapses member cubes to degenerates and
  culls member–member synapses (external synapses stay). Toolbar combo
  `auto|cubes|planes` (default auto → plane below ~4 px/texel); in auto a
  selection ≤256 touching the grid flips it back to cubes so editing
  keeps working. 3D grids get per-group collapse/slice/proj controls in
  the inspector. `--planes`/`--noise-hz`/`--noise-kick` flags on
  `--screenshot` exercise the same path headlessly.
- **Tests**: `plane::tests` (raster ordering, slice/max/mean, argmax pick,
  quad placement); `worker_publishes_grid_traces` (net → publish →
  `snap.grid_traces` + `geo.grids`); `million_neuron_grid_plane_draws` —
  a 1000×1000 grid builds/uploads/draws in <1 s offscreen (the cube path
  is 36 M instanced verts); net-format round-trip + v1-compat + metadata-
  invalidation tests.
- **Verified visually (Xvfb :99)**: 1600-neuron 40×40 grid with live
  noise — cubes at ~15 px/texel, zoom out → continuous viridis sheet
  (LOD flip), zoom back in → cubes return. Console eval, run, spike
  counts all live. Screenshots in `docs/screenshots/`: `lsm_plane.png`
  (the LSM liquid 8³ max-projected mid-noise), `grid_plane.png` (40×40),
  `grid_plane_3d.png` (3D projection). Plane *picking* is code-verified
  (`fs_plane_pick` writes member ids, shared pick path) but was not
  click-tested in the GUI.
- Docs: `docs/features.md` (view modes, LOD/selection rules, 3D controls,
  pick, format v2), `docs/scripting.md` (metadata + invalidation).

Checks: `cargo fmt` clean; `cargo test --workspace` 22 targets pass with
cuda, `cargo test -p sim --no-default-features` 14 targets pass without.

## Stage 3b — UX review fixes (review/STAGE3B.md, this run)

Ten small commits `c7d95ba`..`7a78bb6`, one per review item, in the
file's order. All verified live on Xvfb :99 unless noted.

- **Item 0 — plane pick click-test** (no commit): clicking a grid
  rendered as an activity plane selected the *correct* neuron — #1057,
  inspector pos `0,-12,24` matches the lattice math exactly. The test
  surfaced a real bug instead: a click landing while a hover pick was
  in flight got dropped by `request_pick`'s early return — fixed under
  #9.
- **`c7d95ba` #1 tutorial vs reality**: synapse count 2256 (seed 1,
  measured), Gaussian `scaling*exp(-(d/lambda)^2)` formula, added the
  missing `sim.set_input` step, noise `(60,18)` actually spikes (was
  `(25,0.5)` → 0 spikes), 8-watch cap documented, lsm.ne2 description
  fixed (the script writes it). Then ran the tutorial end-to-end in
  the GUI: `= 2256`, 0.7 input value, 3261 spikes, cap at 8 watches.
- **`eba1b05` #2 Inputs panel**: `Snap.input_values` publishes the
  worker's real constants — panel shows 0.7 where it used to show a
  placeholder; the New-input dialog asks for the value.
- **`df00cae` #3 Stop**: label is "Stop" (pause was a lie — busy
  scripts can't resume); abort surfaces as `stopped by user at line N`;
  `create_2d_grid`/`create_3d_grid`/`create_connections` poll
  `Env::cancel` (wired to the worker's abort flag) so a huge build
  stops mid-rows instead of running to completion.
- **`3adf697` #5 auto-frame**: empty→non-empty transitions and both
  grid dialogs re-frame the camera; `F` key + a real toolbar "Frame"
  button (was right-click-menu-only). Verified: console-built grid on
  a blank canvas appears centered; F re-frames after zooming away.
- **`3d20576` #7 group combos**: Connections/Weights/ExcInh dialogs
  list live groups in a combo; unknown group names in scripts return
  `Runtime error: unknown group "x"` (was silent no-op).
- **`9f50389` #8 raster**: x axis labels in real ms, y rows = the 8
  watched neurons (was raw-id scatter), explicit `height` drag field
  in the header (the panel-edge resize was too finicky to hit);
  no-watch state falls back to the raw-id scatter with auto y bounds.
  Debug print proved all 8 rows populated (`w=[...] pts=160`).
- **`4eba759` #9 picks+synapses**: click picks queue in
  `app.pick_click` behind an in-flight hover pick (`request_pick ->
  bool`) — first click on a plane now selects every time; self-synapse
  refused in both `Network::connect` callers (editor message + Rhai
  error); synapse lines brighter (alpha 0.4) with a source→target
  brightness gradient; synapses touching a selected neuron blend gold.
  Found while testing: cube half-size was a fixed 4.0 vs ~3-unit grid
  spacing → cubes merged into a solid mass hiding the synapses — now
  adaptive (0.4·nearest-neighbour distance, clamped [0.3, 4.0]). Same
  commit because the synapse fix was invisible without it.
- **`b633211` #11 net-replacement reset**: `net_gen` counter bumped
  by `net.clear()`/`net.load()`/New; on change the app drops
  selection, pending connect, move/box state, hover+queued picks,
  watches, v-history, raster, particles, activity, grid traces, input
  values, plane cfgs and requests a re-frame; the worker publishes a
  clean Snap (t/spikes zeroed, `sim_built=false`). New/Load/recent/
  Quit/OS-close are all guarded by a discard-unsaved dialog
  (`saved_version` vs published geo version). Verified: `net.clear()`
  → "nothing selected", watching 0, zeroed stats; dirty New → prompt
  → Discard resets.
- **`1257e63` #13 toolbar truth**: worker mirrors `env.cfg` +
  `env.backend` into `Shared` each loop (cheap change-check) so
  script-side `sim.set_dt`/`sim.set_noise` show up in the toolbar
  (verified: `dt 0.05`, `noise Hz 60` after script calls). One speed
  readout: `x1.0 realtime (max x9)` — requested factor + the best the
  machine reached; the old `sim speed = N steps/s` status line is gone.
- **`7a78bb6` #15 polish**: `→`/`Δ`/`⦿` glyphs → ASCII (`->`, `d(`,
  plain label — egui's bundled fonts render `·`/`—`/`…`/`×` but not
  those); Help menu moved after Group; console default 240px with the
  scroll area filling it (was a fixed 90px cap); console lines >400
  chars end `… (N chars truncated)` — a 3600-element `neurons_of` no
  longer floods; empty scene shows a centered "empty network" hint
  with the two ways to get content.

Still open by design (the [Nico]/[split] tags): real pause/resume for
scripts, raster density heat map, density-aware synapse alpha tuning,
backend fallback notice (the field exists but is never set — cuda
failure currently just errors the eval that built the sim).

## Stage 4 — server / UI split, part 1 (review/STAGE4.md, this run)

Seven commits `b306182`..`27e783a`. Part 1 only: the protocol, the
worker extraction, a headless WebSocket server, and `editor --connect`.
No wasm/browser (spec's own scope).

- **`b306182` proto crate**: serde types for everything crossing the
  line — `ClientMsg` (Hello/Subscribe/Eval/LoadNet/NewNet/AutoRun/
  SetSpeed/RunMs/Abort/Reset/Watch/versioned Edit/CopySubnet/backend +
  input controls/SetExternal/Bye), `ServerMsg` (Hello/Geo/GeoDelta/
  Status/Console/Activity/GridTraces/Clipboard/Error), `NetEdit` enum
  covering all 16 former closure edit sites, `ServerEvent{to}` for
  targeted-vs-broadcast, `PROTO_VERSION` in the hello. bincode codec
  with a 1-byte tag; payloads >256 B deflate (`flate2` fast). Round-trip
  tests for every type.
- **`0c5a38f` sim::serve**: the worker extracted from `simctl.rs` into
  `sim::serve::run(net, opts, rx, emit)` — consumes `(client_id,
  ClientMsg)`, emits `ServerEvent`. One `Emit` callback feeds print
  capture, the run-progress hook, and the main loop. Versioned edits:
  `Edit{base}` against `net_version` refuses stale bases with `Error` +
  Geo/Status/GridTraces resync. Deletions send full Geo (ids renumber);
  other edits send `GeoDelta` (incl. insert-subnet merging into
  same-named groups — diffed by member-count snapshots). Activity
  published rate-limited (`MIN_ACTIVITY_MS`); run/eval tails always
  flush so `time_ms`/totals never lag. Rhai file fns (`net.save/load`,
  `net_load`, `samples_save/load`, `write_to_file`, `read_to_vec`,
  `play_input` files) resolve inside `ServeOpts::data_dir` when set —
  absolute paths and `..` refused. 10 tests in `sim/tests/serve.rs`.
- **`6fec43c` wire size**: the first sketch put per-texel `(idx,val)`
  activity deltas + a `(idx,glow)` list on the wire — on a hot 1M grid
  that measured ~2.5 MB/frame compressed. Now `Activity` carries only
  `spike_bits` (whole-id-space bitmap), watched-v samples, input values,
  totals + `steps_per_sec`; grid/non-grid glow is reconstructed
  client-side (decay cached glow per frame, stamp spiked ids to 255 —
  the worker's own `trace_u8` does the same decay so client/server
  agree). Full-truth resync rides the existing `GridTraces`. Also
  removed a `sim.trace()` readback from the publish path (~4 MB/frame
  on CUDA, now dead).
- **`2290d38` server + `editor --connect`**: `server` crate/binary —
  one `sim::serve` worker + thread-per-client tungstenite (blocking WS;
  no tokio needed at a handful of clients). Binds 127.0.0.1, `--bind` +
  `--data-dir` flags. `ClientConn` typed test/tool client with
  `into_split`. `SimCtl::connect(addr)` splits the socket into
  reader/writer threads feeding the same `Shared` fold as the local
  transport — the UI can't tell where the sim lives. `--connect`
  ignores `--net`/`--cpu`/`--cuda`/`--int-weights` (server owns those,
  message printed); window title shows the remote address.
- **`27e783a` multi-client fixes**: `ClientMsg::Bye` from a client is
  swallowed server-side — a client leaving must not kill the shared
  worker (leaving = socket close; `Server::shutdown` injects Bye
  directly). `WsReader::shutdown_stream` lets `SimCtl::drop` unblock the
  parked recv thread (was a guaranteed hang on remote quit).

Divergence from the spec's wording, honestly: there's no `Transport`
trait object — the seam is the `mpsc::Sender<(u64, ClientMsg)>` +
`Fold`/`Shared` pair; `SimCtl::start_cfg` and `SimCtl::connect` are two
constructors onto it. Same effect, less plumbing.

### Tests (all green both ways)

`cargo test --workspace` (cuda) + `cargo test -p sim
--no-default-features`; `cargo fmt` clean.

- proto round trips for every wire type.
- stale-version edit refused (+ resync) — ws + serve levels.
- local ≡ remote: identical scripted session → identical `N`/`S`/`K`
  (net, synapse, spike counts) over channel and WS transports.
- path escape refused (`net.save("../escape.ne2")`; also live: the lsm
  script's `save examples/lsm.ne2` errored inside `--data-dir`).
- second client sees first's `GeoDelta`; client Bye isolation.
- `SimCtl::connect` hydrate + fold + hang-free drop.
- GUI smoke on Miguel: `server --bind 127.0.0.1:3090 --data-dir
  /tmp/ne_server_data` on the 3090 + `editor --connect` under Xvfb :99.
  `lsm_hidden.rhai` ran server-side ("liquid 512 neurons, 2256
  synapses"); a second wire client's eval (400-neuron grid + noise)
  appeared in the editor live; plane glow lit cyan under
  `spike_bits`-driven client decay. Screenshots /tmp/remote7-8.png.

### Measured bandwidth (server/examples/bandwidth.rs, RTX 3090 CUDA)

1,000,000-neuron `100x100x100` grid, encoded `Activity` bytes counted
over a 15 s wall window:

| scenario | frames | avg frame | activity B/s | cadence | set bits/frame |
|---|---|---|---|---|---|
| noise 60 Hz | ~30 | ~700 B | ~1.35 KB/s | ~2 fps | ~all neurons |
| noise 5 Hz  | ~30 | ~111 KB | ~0.22 MB/s  | ~2 fps | ~700,861 |

Both far under the 2 MB/s target. Honest caveats: (a) the cadence was
~2 fps under these loads — publish pacing is the sim's, not a transport
throttle; at a true 30 fps the per-frame spike window shrinks ~10x so
sparse-frame size shrinks roughly proportionally (bandwidth scales with
spikes/s, not frame size), and the all-ones dense bitmap deflates to
nothing. (b) Client glow within a publish window is an approximation —
a spiked texel stamps 255 rather than the true exp-of-last-spike value;
bounded by the window, and `GridTraces` restores truth on geo changes.

### Next steps (not in part 1, per spec)

wasm/browser build of the UI against the same protocol; dataset upload
+ training jobs; auth beyond localhost binding. Also open from 3b: real
script pause/resume, raster density heat map, synapse-alpha tuning,
backend-fallback `notice` (field exists, never set).

## Stage 5 — file confinement + browser UI (review/STAGE5.md, this run)

Six commits `63e3a5e`..`d54e009`. Both items: the `import` confinement
gap (item 0), and the wasm/browser editor served by `server --web`
(item 1). Not done, per spec: dataset upload/training jobs, auth.

- **`63e3a5e` import confinement (item 0)**: Rhai's default
  `FileModuleResolver` ignored `data_dir` entirely — `import` read from
  process cwd and absolute paths. `ConfinedResolver` wraps a resolver
  rooted at `data_dir` after pre-validating the path through the same
  `resolve_in` as the file fns. `resolve_in` now also canonicalizes the
  deepest existing ancestor of the resolved path, so a symlink planted
  inside `data_dir` pointing outside is refused for `net.save`,
  `net.load`, `import`, and the file messages alike (nonexistent save
  targets resolve through the canonicalized parent). Red→green over WS:
  `import "../x"`, `import "/abs/x"` and symlinked-dir escapes all
  refused as console errors while `import "ok"` inside the dir still
  loads (`imp=42`).
- **`6bc5788` wire file ops**: `ClientMsg::{ListFiles,ReadFile,WriteFile}`
  → `ServerMsg::{FileList,FileData,FileWritten}` + `proto::FileEntry`.
  Handled in `sim::serve` through the same `resolve` sandbox, targeted
  to the requesting client. The UI's last `std::fs` (script panel
  open/save, Load dialog) now goes through the wire — List renders
  server-side entries, dirs descend via `ListFiles`, and `Shared`
  carries the one-shot responses. `Shared`'s `SimConfig`/`BackendKind`
  became the `proto::` wire types, so the whole UI read model is
  `sim`-free (needed for wasm).
- **`73371fd` browser build**: `editor` features — `native` (default:
  sim/server/clap/pollster) and `web`; both are off on wasm32 where
  `SimCtl::connect` is the only transport (browser `WebSocket`, sends
  queued until `onopen`, `onmessage` folds into `Shared`, no threads,
  no `std::fs`, no `std::time::Instant` anywhere in the UI surface —
  egui supplies timing; `getrandom/js` for entropy). `main` on wasm is
  an eframe `WebRunner` pinned to `Backends::BROWSER_WEBGPU` (the scene
  pipelines need storage buffers; a WebGL2 fallback would just fail
  later) and, on failure, paints a plain-English "needs WebGPU" message
  on the page. `editor/web/build.sh` = `cargo build --target
  wasm32-unknown-unknown --release --no-default-features --features
  web` + `wasm-bindgen --target web` (CLI 0.2.118 matches the crate) →
  `editor/dist-web/` (gitignored) with `web/index.html`.
- **`508056b` server --web**: `handle_client` sniffs the request head —
  `Upgrade: websocket` answers with a hand-rolled 101 +
  `derive_accept_key`, anything else serves a static file from
  `web_dir` (component-wise traversal refusal; `index.html` at `/`;
  correct MIME incl. `application/wasm`). One port carries page, bundle
  and wire. `accept_web`/`spawn_acceptor_web`; `--web` requires
  `index.html` to exist at startup.
- **`d54e009` docs**: README + features.md gained the server/browser
  story (the features table never mentioned `server`; the old "a
  browser UI could…" line is now what actually exists).

### Tests (all green)

`cargo fmt` clean; `cargo test --workspace` — 30 targets (cuda);
`cargo test -p sim --no-default-features` — 15 targets; `cargo check
-p editor --no-default-features` (native, feature-less) and
`--target wasm32-unknown-unknown --features web` both clean;
`cargo build -p editor --release --target wasm32-unknown-unknown`
produces `editor.wasm`; `wasm-bindgen` emits `editor_bg.wasm`
(~9.9 MB unoptimized-for-size).

- `ws_web_serves_http_and_ws_on_one_port`: GET /, /app.js with the
  right MIME, `/../secret.txt` and misses 404, then a real WS client on
  the same port completes Hello→Eval→Console.
- Item-0 confinement tests stay green.

### Browser smoke (this box, port 3090)

`server --bind 127.0.0.1:3090 --data-dir /tmp/ne_web_data --web
editor/dist-web`. curl: `/`, `/editor.js`, `/editor_bg.wasm` all 200
with correct types. Headless Chromium (snap + playwright
chrome-headless-shell) and windowed Chromium under Xvfb :99:

- page loads, wasm compiles, eframe takes over `<canvas>` (egui's
  hidden text input present; `#loading` removed) — DOM verified via
  `--dump-dom`;
- the browser opened `ws://127.0.0.1:3090/` and got
  `HTTP/1.1 101 Switching Protocols` — netlog verified (the manual
  upgrade path works);
- WebGPU adapter/device/surface acquired — `WebGPUSwapBufferProvider`
  created the swapchain under `--use-webgpu-adapter=swiftshader`;
- **pixels could not be screenshotted here**: every headless/snap path
  on this box lacks a `SharedImageBackingFactory` for
  `WebgpuSwapChainTexture` — the canvas renders but can't be
  display-composited (also with `--use-vulkan`, real RTX 3090 Vulkan
  path, and ANGLE-swiftshader). Environment limit, not an app failure;
  needs a real Chrome on a display-capable host.
- the no-WebGPU path IS screenshotted (`--disable-gpu`): the page shows
  "neuro_edit v2 needs WebGPU (the 3D view uses storage buffers).
  Adapter/device request failed: …" on dark background —
  /home/data/web_no_webgpu.png.

### Next steps

Painted-pixel verification of the browser editor on a WebGPU-capable
Chrome (any recent Chrome on a normal desktop works — headless shells
on this box can't composite the swapchain). `wasm-opt` for the ~9.9 MB
bundle. Then spec-deferred items: dataset upload + training jobs, auth,
and the 3b leftovers (script pause/resume, raster density map,
synapse-alpha tuning, backend-fallback `notice`).
