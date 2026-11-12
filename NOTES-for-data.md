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

## Stage 6 — datasets, background jobs, detector + memory examples (review/STAGE6.md, this run)

### What landed

- **`sim::dataset`** — seeded multi-channel generator (poisson/gauss
  base + `rate`/`corr`/`motif` events), persisted under
  `datasets/<name>/` as `signal.f32` + `meta.json` + `labels.json`,
  cached by canonical spec JSON; `.npy` reader for uploads. Same spec +
  seed ⇒ identical bytes (test `same_seed_same_bytes`).
- **Wire upload**: `ClientMsg::UploadDataset` → `datasets/<name>`
  inside `data_dir`, name confined to a plain dir name (no `/`, `\`,
  `.`-prefix), byte cap via `server --max-upload-mb` (default 64 MiB;
  `ServeOpts::max_upload_bytes`). Server replies `DatasetUploaded` or
  `Error`; local transport accepts uploads too, so CLI runs share the
  same code path.
- **Rhai dataset API**: `dataset(name)` (gen-or-load by name),
  `dataset_gen(#{...spec})`, `datasets()`, `d.channel(i)`, `d.events()`,
  `d.events(lo,hi)`, `d.n_channels`, `d.steps`, `d.len_ms`, `d.dt_ms`,
  `d.name`, `d.spec_json`. Names go through the same confinement as
  file paths — `dataset("../x")` is refused.
- **Background jobs** (`server` only — the sim worker is shared, so a
  job gets its own `Env` on its own thread): `StartJob{name,script}` →
  `JobStarted{id}`; the runner writes `runs/<id>/{script.rhai,
  status,metrics.json,…}` under `data_dir`, streams
  `job_progress(#{k:v,…})` as `JobProgress` broadcasts, survives the
  launching client's disconnect, honours `CancelJob` (rhai
  `on_progress` → `Terminated`), and `ListJobs`/`OpenJob` report
  `JobInfo`/`JobDetail` (status + `result.json` + script). `job_dir()`
  gives scripts their artifact dir. The local transport rejects job
  messages with an explanatory error (jobs need the server).
- **Editor**: a *Jobs* plot tab — live job list (2 s poll while open),
  per-job status, cancel/open buttons, numeric `JobProgress` fields
  plotted as lines, and opened-job detail (status + `result.json` +
  script). The script editor gained a **Run as job** button.
- **Examples**: `examples/detector.rhai`, `examples/memory_window.rhai`
  (same rigor pattern as `lsm_hidden.rhai`: cached dataset, split by
  time, ridge readout on liquid spike counts, raw + coincidence
  baselines, permutation null).

### Detector numbers (detector.rhai, full params, CPU)

`det_corr`: 8 ch × 45 s poisson 25 Hz, `corr` bursts every ~500 ms
(±150 jitter), footprint 12 ms, strength 35 Hz — deliberately subtle:
per-channel marginal barely moves, pairwise coincidence carries it.
Windows = event anchors vs. clean negatives (≥250 ms event-free behind,
≥60 ms ahead); 3 liquid seeds, ridge λ=1, 200-label permutation.

| split | pos+neg |
|---|---|
| train | 44+44  val 25+25  test 21+21 |

| readout | test AUC (mean ± sd, 3 seeds) |
|---|---|
| **liquid** | **0.739 ± 0.014** (val 0.666 ± 0.051) |
| raw counts | 0.710 |
| coincidence counts (28d) | 0.819 |
| permutation null | 0.503 ± 0.185 (pooled p = 0.114) |

Per-seed liquid AUCs: 0.752 / 0.746 / 0.719 (p = 0.039 / 0.139 / 0.174).

**Honest read**: the null sits at chance (0.503) — the control is
clean. The event is genuinely detectable, but at readout delay 0 the
28-dim coincidence-count baseline is *stronger* than the liquid
(0.82 vs 0.74) — no memory is needed at zero delay, and the coincidence
feature is a direct match to the planted event structure. The liquid's
value-add is the temporal dimension, which the delay sweep below
probes. Marginal significance (p≈0.11) reflects the small test set —
more events would tighten it.

### Memory-window table (memory_window.rhai, full params, CPU)

Sparse `corr` events (every ~900 ms), ridge readout on liquid spike
counts in a 20 ms window starting `delay` ms after event onset; ~29+29
windows/split; three liquid settings.

| liquid | +0 | +5 | +10 | +20 | +50 | +100 | +200 ms |
|---|---|---|---|---|---|---|---|
| base (8³, λ=8, w=1.0) | .725 | .688 | .547 | .485 | .535 | .522 | .488 |
| weak (w_rec 0.5)      | .711 | .661 | .417 | .505 | .681 | .481 | .453 |
| big (12³, λ=8)        | .762 | .752 | .522 | .536 | .635 | .622 | .405 |

**Honest read**: usable readout horizon ≈ 5–20 ms at these settings —
by +50 ms the curves sit at chance, with a non-monotonic blip at
50–100 ms for `big` and several sub-0.5 cells (the readout
anti-predicts — overfit noise at ~30 windows/split, not real negative
information). The decay is real but the curve is noisy; longer
datasets / more windows would smooth it.

### Tests added

- `sim/src/dataset.rs` unit tests: `same_seed_same_bytes`,
  `labels_line_up_with_events`, `save_load_round_trip`, `npy_reads`.
- `sim/tests/serve.rs`: `dataset_gen_over_wire`,
  `dataset_name_escape_refused`, `upload_dataset_too_big_refused`,
  `upload_dataset_roundtrip` (+ earlier `import` confinement cases).
- `server/tests/ws.rs`: `ws_detector_example_small` (splice in small
  params, real job end-to-end — dataset → liquid → auc → JobDone +
  `runs/<id>` artifacts), `ws_job_progress_survives_client`
  (job keeps running + persists after the starter disconnects),
  `ws_job_cancel` (`job.status` → `cancelled`, run dir stamped).

## Stage 7 — per-synapse `tau_syn` + a memory curve that survives scrutiny

### Per-synapse synaptic time constants

Each `Synapse` carries `tau_ms`. `0.0` = the historical behaviour
*unchanged*: the synapse's current is folded into its target's single
merged current `s[i]` decaying at the global `synapse_tau` (2 ms).
`> 0` gives that synapse its own exponential current trace with its own
time constant — an independent `ds/dt = −s/τ + Σw·δ(t_arrive)` per
synapse, summed per target neuron.

- **Default bit-exact**: all-zero `tau_ms` bypasses the new path
  entirely. `cpu_ref.rs::default_tau_is_bitexact` pins the pre-change
  spike train fingerprint on a fixed seed; all stage ≤ 6 tests pass
  unmodified.
- **CPU + CUDA**: both carry per-synapse trace state;
  `gpu_matches_cpu_with_custom_tau` agrees within the existing
  tolerance (exact spike-count match on a deterministic net).
- **`.ne2` v3**: format version 2→3; v1/v2 files lift to `tau_ms=0.0`
  (old nets are byte-identical in behaviour). Round-trip + lift tests
  in `net-format/src/io.rs`.
- **Rhai**: `net.set_syn_tau{,_dist,_one,_idx}`, `net.set_syn_tau_inh`
  (slow-inhibition variant), `net.syn_tau(src,tgt)`. Editor: *Group →
  Set synapse tau…* dialog (uniform or Gaussian per edge).
- **Memory/bandwidth cost** (allocated only when any `tau > 0`; the
  merged path allocates nothing): ~12 B/synapse — `tau_ms` (4 B in the
  CSR), `xtrace` current value (4 B), one `xring` arrival-list slot
  entry (~4 B) — plus ~8 B/neuron for the `xrow` row ranges. Bandwidth:
  the neuron pass now also walks each neuron's custom-arriving
  synapses; cost scales with spike volume × trace count, not steps.

### Memory curve, tightened (memory_window.rhai v2)

300 s `mem_corr2` dataset (503 corr events), **249+249 train /
254+254 test windows** (≥ 200 ✓), **5 liquid seeds** (5/7/11/19/42),
mean ± sd per cell, and a **label-permutation null per delay**
(200 draws on seed-5 test labels — null mean 0.499 ± 0.025, so the
pass line is null + 2sd ≈ 0.549). One replay per window serves every
delay via `sim.spike_counts` checkpoints (exact per-window counts).

AUC (mean ± sd over seeds), readout window = 20 ms at `delay` after onset:

| setting | +0 | +5 | +10 | +20 | +50 | +100 | +200 ms | horizon |
|---|---|---|---|---|---|---|---|---|
| merged (τ=2ms)   | .741±.003 | .719±.009 | .637±.007 | .618±.012 | .574±.019 | .518±.021 | .517±.005 | **50** |
| rec τ=5ms        | .741±.005 | .737±.006 | .707±.012 | .679±.008 | .584±.021 | .517±.013 | .512±.014 | 50 |
| **rec τ=20ms**   | .724±.016 | .717±.020 | .707±.019 | .707±.011 | .645±.006 | **.570±.012** | .526±.015 | **100** |
| rec τ=50ms       | .656±.018 | .622±.038 | .622±.030 | .595±.025 | .551±.028 | .519±.031 | .521±.025 | 50 |
| rec τ=100ms      | .622±.028 | .593±.032 | .591±.021 | .557±.029 | .535±.041 | .529±.020 | .514±.028 | 20 |
| rec τ=50, w_rec×2| .600±.059 | .572±.047 | .529±.029 | .502±.021 | .502±.019 | .492±.026 | .499±.027 | 5 |
| **slow-inh (20/50)** | .715±.022 | .699±.023 | .697±.025 | .701±.025 | .638±.016 | **.608±.011** | .531±.013 | **100** |
| raw counts       | .747 | .666 | .551 | .516 | .513 | .527 | .481 | ~10 |

(Horizon = last delay with mean AUC ≥ null + 2sd.)

**What the sweep shows**: per-synapse tau on the recurrent edges is the
single change that extends the horizon — τ=20 ms roughly doubles it
(50 → 100 ms) while *costing* ~0.02 AUC at delay 0. Longer taus trade
immediate detectability for a flatter mid-range slope (τ=50/100 never
beat the +50 ms cell of τ=20). Doubling recurrent weight hurt
everywhere. Slow inhibition (τ_rec=20, τ_inh=50) matches τ=20's curve
and is best at +100 ms (0.608 vs 0.570) — the inhibitory tail keeps
event structure readable later. Nothing holds at +200 ms; by then the
membrane state has decayed past readout resolution — the limit is the
~50–100 ms envelope of the traces themselves, not the readout.

### Detector at the best memory setting (detector.rhai, delay sweep)

`slow-inh` liquid (τ_rec=20, τ_inh=50) on the `det_corr` task; raw and
coincidence baselines use the *same* delayed window.

| delay | liquid | raw | cooc | null | pooled p |
|---|---|---|---|---|---|
| **0 ms**   | 0.668 ± 0.018 | 0.710 | 0.819 | 0.500 ± 0.098 | 0.049 |
| **100 ms** | 0.634 ± 0.047 | **0.489** | 0.614 | 0.495 ± 0.077 | **0.038** |

**Honest read**: this is the result stage 6 promised. At delay 0 the
coincidence baseline still wins (the event is *in* the window — no
memory needed) and per-synapse traces cost the liquid ~0.07 AUC vs the
merged net (0.668 vs 0.739). At delay 100 ms the stream's raw counts
are pure chance (0.489) and even the coincidence feature has decayed to
0.614 — but the liquid holds 0.634 ± 0.047, beating raw outright and
matching cooc with 3× its sd. That's the liquid's temporal memory
paying off exactly where memory is required. The margin over cooc at
+100 ms is thin (0.634 vs 0.614 — within noise on 3 seeds), so the
claim is "the liquid retains what the stream has lost", not "beats the
best baseline by a wide margin".

### Remaining honest caveats

- AUCs at +100 ms are modest (~0.57–0.61) — real but not strong; the
  useful horizon is 50–100 ms, not the hundreds some LSM papers imply.
- The null is tight (sd ≈ 0.025 on 254-sample cells), so ±2sd is a
  fair bar; smaller datasets couldn't have drawn these lines at all.
- GPU/CPU agree on custom taus, but all numbers above are CPU runs
  (CUDA used only for speed checks).
- `dataset_gen` caches by canonical spec; `mem_corr2` is regenerable
  from `examples/memory_window.rhai`'s parameter block.

### Next steps

Auth for `server` (any connected client can run jobs), job
queueing/limits, `wasm-opt` for the 9.9 MB bundle. Science: per-synapse
tau made the +100 ms window readable but not strong — a larger liquid
or structured input mixing could push the horizon past 100 ms; the
detector's cooc baseline should also get the delay sweep directly
(here it rides along as a table row, not a sweep).

## Stage 8 — small server polish

Opening the wire port in a browser used to log `WebSocket protocol
error: No "Connection: upgrade" header` and show nothing. Now every
accepted connection's request head is classified the same way `--web`
mode already did: `Upgrade: websocket` is the wire protocol; a plain
GET on a WS-only port gets `426 Upgrade Required` + a text/plain hint
(build with `editor/web/build.sh`, serve with `server --web
editor/dist-web`, or attach with `editor --connect host:port`), and the
server log says the same in one line. `--web <dir>` without an
`index.html` fails at startup naming `editor/web/build.sh`; startup
prints `browser editor: open http://<bind>/` with `--web` or
`editor --connect <bind>` without. README gained a *Run in the browser*
section with the exact toolchain commands (wasm32 target,
`wasm-bindgen-cli` pinned to the Cargo.lock version). Tests:
`ws_plain_http_gets_hint_without_web` (426 + hint body, wire still
works after), `web_dir_without_index_fails_startup_check`. Verified
live: curl → 426 + hint, `--web /tmp/empty_web` → startup error,
`--web editor/dist-web` → the open-URL line.

## Stage 9 — WebGPU help page per browser

Nico's Brave-on-Linux report: the page died on the wasm
adapter/device error ("No suitable graphics adapter found … webgpu not
requested") — Brave ships WebGPU off on Linux, `navigator.gpu` is
absent, and users got a raw JsValue instead of a way forward.

`editor/web/webgpu-help.mjs` (plain ES module, no build step) runs
BEFORE the wasm loads: `navigator.gpu` missing → "WebGPU is off in
this browser"; `requestAdapter()` null/throws → "WebGPU found no
GPU"; adapter found → the loading line names the vendor/architecture
and the editor boots. Browser/OS detection covers Brave
(`navigator.brave.isBrave`), Edge/Opera/Vivaldi/Chrome/Chromium,
Firefox, Safari × Linux/Windows/macOS/Android/iOS; unknown gets
generic Chromium steps. Steps are real per-combo instructions —
`<scheme>://flags/#enable-unsafe-webgpu` + `#enable-vulkan` on
Chromium-Linux (scheme brave/chrome/edge/vivaldi/opera), update-first
on Windows/macOS (on by default since Chrome 113),
`settings/system` + `chrome://gpu` + `#ignore-gpu-blocklist` for the
no-GPU case, `about:config → dom.webgpu.enabled` for Firefox (on by
default Windows ≥141 / macOS ≥147 — verified against the gpuweb
implementation-status table, not guessed), Safari 26+/Feature Flags.
Every browser URL renders as copyable text + a Copy button (pages
can't open `chrome://` links), and every page ends with
`editor --connect <this host:port>`. The Rust fallback in `main.rs`
kept as the last resort: a device request that still fails calls
`window.neWebgpuHelp.show(err)` — same rendered page, raw JsValue in
a `<details>` line.

Tests: `webgpu-help.test.mjs` (`node --test`, 9 cases with real UA
strings — Brave/Chrome/Vivaldi Linux, Edge Windows, Firefox Linux,
Safari macOS, Android Chrome, unknown). Verified live in headless
Chrome: `server --web editor/dist-web` + no-GPU browser renders the
numbered-steps page (`review/stage9_webgpu_help.png`), not a blank
canvas. README's browser section now documents the check and the
node-test command.

## Stage 10 — new artwork

Direction B ("spike": one neuron + its action potential, cyan line,
amber tip) from Data's artwork set, put into the app. Masters (SVG)
live in `editor/assets/src/`; tool-icon names → labels in
`editor/assets/src/tool-icons/icons.tsv`.

- **App icon + About** (`25dbe99`): `editor/assets/icon_256.png` (window
  icon, `main.rs` unchanged) and `icon_32.png` swapped. `about.png` is
  now 740×246 and drawn at 370×123, so it stays sharp on 2x screens.
- **Web** (`4f05545`): `favicon.ico` (16/32/48), `favicon.svg`,
  `apple-touch-icon.png`, `icon-192/512.png`, `site.webmanifest` +
  `<link>` tags; `#loading` shows `loading-logo.svg` (slow pulse, off
  with reduced motion) and `webgpu-help.mjs` writes its line into a
  child `#loading-status` (`setLoadingStatus`), so the logo stays.
  `build.sh` copies the 7 files. Server (`4d7fa51`): `.webmanifest` →
  `application/manifest+json` (was octet-stream).
- **Tool icons** (`4ddd0b5`): `icons.rs` loads 39 white 48 px PNGs
  (`editor/assets/icons/`); old toolbar PNGs removed. Buttons use
  `image_tint_follows_text_color`, so icons take the text colour,
  hover/selected/disabled included (dark on the browser's light theme,
  light on native dark). Toolbar: Run/Stop, 10ms, 100ms, Reset keep
  their text; speed, backend, dt, noise get an icon with hover text
  instead of the "backend"/"dt"/"noise Hz" labels (the fields show
  " ms"/" Hz"); modes, planes and plot tabs are icon + text; synapses,
  grid and the four panels are icon + text toggle buttons (were
  checkboxes); Frame is icon-only (hover: "Frame network (F)"). File
  and Help menus have icons; Edit/Insert/Group menus stay text.
- **Mipmaps:** egui-wgpu 0.32.3 (Cargo.lock) makes every managed
  texture with `mip_level_count: 1` (`renderer.rs:629`) and never reads
  `TextureOptions::mipmap_mode` (`with_mipmap_mode` exists in epaint
  0.32.3 but does nothing on wgpu). So icons are box-filtered on load
  (`image::imageops::thumbnail`) to 16 px × `pixels_per_point` (16 at
  1x, 32 at 2x, max 48) and reloaded when the scale changes
  (`Icons::ensure`).

Tests: `icons::tests::tool_icons_are_scaled_to_16_points` (sizes at
ppp 1/1.5/2/3/4 + banner 740×246; red with resizing off);
`webgpu-help.test.mjs` +3 (status goes to `#loading-status` and the
`#loading` text is untouched — red when it writes `#loading`; fallback
without the span; `index.html` has the logo, span and favicon link);
`ws_web_serves_http_and_ws_on_one_port` checks the manifest MIME (red
without the fix). `cargo fmt` clean; `cargo test --workspace` 122
passed in 30 targets (cuda); `cargo test -p sim --no-default-features`
53 in 15; `node --test` 12/12; clippy warnings unchanged.

Verified live: native editor on Xvfb at 1x and at 2x
(`WINIT_X11_SCALE_FACTOR=2`); `editor/web/build.sh` + `server --bind
127.0.0.1:3197 --web editor/dist-web` in headed Chromium (SwiftShader
WebGPU): favicon in the tab, loading logo + "loading neuro_edit v2…
WebGPU adapter: google swiftshader", editor boots. Screenshots:
`review/stage10_*.png`.

Open: the x-y plane icon is a plain square and can read as an
unchecked checkbox. At 1x the toolbar is ~1160 px wide (window default
1440). Direction A or C would need the set redrawn.

## Stage 11 — UI review fixes (commits 7553b22..9e9eb73)

All 12 items from review/STAGE11.md (section 5 numbers), one commit
per tight group, on top of the Stage 10 artwork.

1. **Dark theme (1.6)** — `ctx.set_theme(egui::Theme::Dark)` in both
   native and wasm entries. `set_visuals` alone loses to eframe
   applying the OS light scheme after the app creator.
2. **Selection brackets (1.2)** — `screen_box_rect` projects the
   picked cube's 8 corners (centre ± inst_half·1.15) to screen and
   takes the bounding rect; >24 selected → ONE union bracket plus a
   "N selected" count label instead of 64 corner ticks.
3. **Move (1.3, 3)** — `MoveDrag.anchor` anchors the drag plane at
   the GRABBED object's position so it stays under the cursor;
   `CamU.preview` offsets selected cubes + their synapse endpoints in
   the shader at 50 % alpha for a live preview; Esc drops the drag
   (positions snap back) — commit only happens on release.
4. **Floor (1.1)** — `GeoSnap::floor_y() = min neuron y − 20`;
   `floor_grid` and `Camera.floor_y` clamp (`eye.y ≥ floor_y + 2`)
   follow it. `shot.rs` sets floor before framing/clamp.
5. **Cursors + hover (3)** — hover pick every frame rides `lod.z`;
   hovered cube brightens 1.4× in the shader; cursor icons: pointer
   over selectable, grab over selected, grabbing during move,
   crosshair in Insert/Connect.
6. **Gestures (3)** — `primary_drag_action`: Navigate drag on object
   (selected or not) → Move; empty → box select; click selects;
   Move mode = always move; click in Move mode selects the hit
   neuron first (shot 11). Matches glscene.cpp:93-123 mapping.
7. **Floating windows (option A)** — Inspector, Plots, Console,
   Script editor, Inputs are `egui::Window`s (movable, resizable,
   closable); View menu toggles each + "Reset layout"
   (`memory.reset_areas()`); visibility persisted via
   `Persisted.windows` (eframe storage, `#[serde(default)]` for
   older blobs). Ctrl+1..5 toggles. Verified: windows float over
   panels on native and browser.
8. **Shader grid (1.1)** — `vs_grid/fs_grid` on a
   camera-following 4000-unit quad at floor_y: fwidth-AA lines,
   brighter majors every 5, distance fade, no moiré at grazing
   angles, no hard tile edge (replaces the 1 px line grid).
9. **Connect from inputs (1.4)** — `NetEdit::AddInputTarget`
   (proto + serve arm appends, never replaces targets); Connect
   mode accepts an input as source; input→target links drawn as
   dashed yellow lines; input context menu gains "Connect from
   here". ws test `ws_connect_from_input_adds_target`.
11. **Toolbar wraps** — see bug note below; BOTH toolbar rows are
    now `horizontal_wrapped`.
12. **`?` overlay** — Shortcuts window (centered, non-resizable):
    modes, camera, mouse gestures, edit, window keys.
13. **Status + glyphs** — mode change clears mode-scoped status;
    ASCII "up/down" replaces missing ↑↓ glyphs in the console hint.

Real bug found while verifying #11: making only the MODE row
`horizontal_wrapped` was not enough — a sibling plain-`horizontal`
row that overflows grows the panel's `max_rect`
(`Placer::expand_to_include_x`) past the window edge, so the wrapped
row's wrap boundary sat off-screen and it still clipped. Both
toolbar rows are now `horizontal_wrapped`. Verified on Xvfb at
480 px: controls reflow onto 5 rows, nothing clipped
(`review/stage11_narrow_toolbar.png`).

Tests (all in `cargo test -p editor --lib`, 23 green):
`screen_box_rect` far/near bracket shrinks with distance;
`primary_drag_action` gesture→action mapping incl. hover-not-
selected → Move; `MoveDrag` plane through grabbed anchor keeps the
object under the cursor; `snap::floor_y` = min y − 20; proto
`NetEdit::AddInputTarget` round-trip; ws test for the edit path.
Each new test fails without its change (checked while writing).

Verification:
- `cargo fmt --all -- --check` clean.
- `cargo test --workspace`: 30 targets green (cuda features),
  `cargo test -p sim --no-default-features`: 15 green.
- `cargo test --workspace --release`: 30 targets green.
- wasm `cargo check --target wasm32-unknown-unknown
  --no-default-features --features web` clean; `editor/web/build.sh`
  rebuilt `dist-web`. No web JS touched → node tests unchanged
  (still 12/12 from Stage 10 run).

Screenshots (`review/stage11_*.png`, headful Chromium on Xvfb +
drive.js, server on :3197, OS light scheme):
- `stage11_dark_theme` / `stage11_floor_lsm`: lsm.ne2, OS light →
  UI dark; floor grid renders BELOW the 512-neuron liquid.
- `stage11_bracket_near`: single neuron, bracket hugs projected box.
- `stage11_bracket_big`: 57 objects → one union bracket + "57
  selected".
- `stage11_move_preview`: mid-drag — ghost cube 50 % alpha at the
  cursor, bracket left at origin, status "move d(-9,0,32)".
- `stage11_connect_input`: status "input stim -> 18", new yellow
  link drawn input→target (3rd link on the input).
- `stage11_shortcuts`: `?` overlay listing all keys/gestures.
- `stage11_view_menu`: View menu with Inspector/Plots/Console
  checked, Script editor/Inputs unchecked, Reset layout.
- `stage11_floating_windows`: native — Inspector/Plots/Console
  floating over the viewport.
- `stage11_narrow_toolbar`: native Xvfb at 480 px (browser DPR2
  equivalent) — toolbar wraps to 5 rows, nothing clipped. Browser
  at 800 CSS px was unusable for this shot because floating windows
  cover the whole viewport there — noted as a follow-up (window
  defaults could dodge the top-left at small sizes).

Skipped/deferred (per spec): 10 synapse look, 14 vertical-move
modifier, 15 docking — all waiting on Nico. The 3090 port server
belongs to Nico — untouched; all shots on 3197, all test
server/editor processes stopped after capture.

## Stage 12 — hosted web app: accounts, back office, per-user runs + files

Commits (`29edb33..e3f2c32`, all local, nothing pushed):

- `29edb33` proto + serve: `Mkdir`/`DeleteFile`/`RenameFile`/`FileUsage`/
  `UploadStart`/`UploadChunk`/`DownloadFile` + replies, `FileEntry.mtime_ms`,
  recursive quota accounting on `ServeOpts`, 256 KiB chunked down/upload
  (`.upload` staging file renamed on the last chunk).
- `62f1167` multi-user core: `accounts.rs` (rusqlite bundled SQLite,
  argon2id hashes, SHA-256-digested 256-bit session tokens, per-IP and
  per-email login rate limiter, admin CLI ops), `runs.rs` (per-user run
  registry: lazy start on first WS, idle stop, max-runs cap, drop closes
  the user's sockets), `pages.rs` (landing/login/signup/admin HTML in the
  Stage-10 dark+cyan+amber palette), `multi.rs` (HTTP front door, session
  cookie `ne_session`, Origin/CSRF checks, `--trust-proxy` allowlist for
  XFF/XFP), `main.rs` (new flags + `admin` subcommand + any-non-loopback
  bind refusal).
- `fca125b` `server/tests/multi.rs`: 11 integration tests + the fixes
  they forced (below).
- `73e6be7` editor: Files floating window (Ctrl+6, View menu, persisted),
  remote protocol vs local `dirs::data_dir()/neuro_edit` backend,
  mkdir/rename/delete/usage bar, chunked uploads (native file dialog +
  dnd; wasm picker via FileReader, download via Blob), `.ne2` load /
  `.rhai` script-open; login dialog for `--connect http(s)://` with
  remember-me token in eframe storage, `connect_secure` wss+Bearer;
  wasm kept clean (target-gated ureq/rfd/dirs).
- `622c249` `admin add|list|activate|disable` take `--accounts` after the
  action (spec's literal order, password stdin-only); `docs/deploy.md`
  (miguel/Marvin, systemd, nginx WS-upgrade block, `--trust-proxy`,
  backups); README pointer; clippy pass on new code.
- `e3f2c32` `/app` serves index.html with `<base href="/app/">` injected
  (bundle imports stay under the session-gated prefix — without it
  `editor.js` 404'd and the page hung on "checking WebGPU…"); landing +
  admin table show the effective quota instead of a raw `0 MB` row.

### The one big default I did not take

The brief defaulted to axum+tokio for HTTP. I kept **blocking I/O**:
a minimal HTTP/1.1 parser + one thread per connection on the existing
`TcpListener`, and the WebSocket upgrade reuses the exact same
reader/writer-thread bridge as shared mode. Reasons: the codebase is
entirely blocking/threaded (`sim::serve` worker via mpsc, jobs on
threads, `Rc<RefCell>` per run — none of it Send-safe across an async
boundary without wrapping); the required surface is small (six routes,
forms, cookies, one upgrade); and `Connection: close` per request makes
the per-connection-thread cost irrelevant at this scale. Adding axum/
tokio would have pulled ~80 crates of async runtime plus a
`Rc<RefCell>`→`Arc<Mutex>` rework of every serve state for zero
user-visible gain. If request volume ever matters, the HTTP layer is a
single file (`multi.rs`) behind a `handle(stream)` boundary — swapping
in an async front door does not touch runs/accounts/editor.

### Other choices where a default existed

- `--trust-proxy` as an **IP allowlist** (Update 14:5x), not a bool:
  XFF/XFP honoured only when the direct TCP peer is allowlisted; headers
  from other peers ignored. `expected_origin` uses XFP only from those
  peers too.
- Origin required on `/signup` and all `/admin/*` POSTs; `/login` and
  `/api/login` skip it (rate-limited credential endpoints — the native
  client sends no Origin). Missing Origin = reject, matching browsers
  (which always send it on POST).
- Admin quota row `0` = "use `--default-quota-mb`" (fallback in
  `quota_of`), so admin add needs no quota flag.
- User data dirs get created lazily at run start, not at signup.
- Login page errors: pending/disabled carry the spec'd texts; wrong
  password vs unknown email share "wrong email or password" — no
  account-existence leak. The `429` path returns the pending-style page
  with the wait text.

### Red checks (tests that failed before their fix/feature)

Genuinely observed red→green during development:

- `signup_pending_activate_login_ws` — login returned the error page
  because argon2's verify signature was wrong (0.6→0.5 downgrade caught
  by this test).
- `disable_kills_session_and_ws` — the run's `Arc::try_unwrap` never
  fired while a live client held an `Arc` ref, so the socket stayed
  open; fixed with `Mutex<Option<JoinHandle>>` stop + explicit
  `shutdown(Both)` on cloned client sockets.
- `quota_refuses_and_leaves_no_partial` — caught my own overflowing
  test-byte generator (wrapping mul needed), and verified the `.upload`
  stage file is cleaned, not just the target.
- `users_cannot_cross` — `..`, absolute paths, a planted symlink A→B,
  and cross-id protocol calls all refused.
- `trust_proxy_gates_xff` — spoofed XFF from an unlisted peer ignored,
  allowlisted peer honoured (written for the 14:5x update).
- `max_runs_cap_and_idle_stop`, `login_rate_limit`,
  `db_holds_no_secrets`, `chunked_round_trip` (multi-MB),
  `admin_gated_and_post_checks`, `ws_refused_without_session` — written
  against the implementation; the first two exercise timing/caps that
  would trivially fail on a shared-run build.
- Live-verified bug beyond the suite: `/app`'s module imports 404'd
  under multi mode (screenshot workflow caught it) → the `<base>`
  injection.

### Verification counts

- `cargo fmt --all` clean; `cargo clippy` — zero warnings in new code
  (warnings left are all pre-existing in dataset.rs/network.rs/etc).
- `cargo test --workspace --release`: all targets green —
  `server/tests/multi.rs` 11/11, `server/tests/ws.rs` 15/15 (shared
  mode unchanged), sim suites (lsm_null, honest_eval, gpu_vs_cpu
  custom-tau, serve 13/13…) green, `accounts::tests` 1/1.
- `cargo test -p sim --release --no-default-features`: all green.
- `cargo check` editor native + `--target wasm32-unknown-unknown
  --no-default-features --features web`: clean, zero warnings;
  `editor/web/build.sh` rebuilt dist-web.
- node: `node --test editor/web/webgpu-help.test.mjs` — 12/12 (count
  grew with Stage 10/11 additions).
- Admin CLI smoke: `admin add` (stdin pw) / `list` verified on a real
  state dir.

### Screenshots (`review/stage12_*.png`, gitignored, on disk)

Multi-user server on **:3412** (Nico's :3090 untouched throughout;
the first launch attempt also self-killed a stray `pkill` match once —
learned to scope the pattern), headful Chromium on Xvfb :99 +
playwright `drive.js`, seeded users: `nico@neuro.dev` (admin, active),
`ada@x.dev` (pending):

- `stage12_landing_desktop` / `stage12_landing_phone` — hero art +
  blurb + stacked Log in / Sign up cards; 375 px layout holds.
- `stage12_signup` — signup card scrolled into view.
- `stage12_pending` — "your account is waiting for activation" after
  logging in as the pending user (verified text via DOM eval too).
- `stage12_admin` — "Back office — 1 pending activation": ada pending
  on top (note + 2048 MB quota), nico active·admin; activate/make
  admin/delete/quota/disable buttons.
- `stage12_app_boot` — the wasm editor inside the session-gated `/app`.
- `stage12_browser_files` — Files window listing nico's server-side
  files (detector.rhai, lsm.ne2, memory_window.rhai) + "92.0 KB of
  2.0 GB" usage bar — full cookie → /app → /ws → protocol round-trip.
- `stage12_native_login` — "Log in — http://127.0.0.1:3412" dialog with
  email/password/remember-me on `--connect http://…`.
- `stage12_native_files` — native local file space at
  `~/.local/share/neuro_edit` (no --connect): dir + files + usage.
- `stage12_viewmenu` — View menu with the Files entry.

### Open / follow-ups

- Pending-user count badge shows on `/` for admins? Not implemented —
  count lives in the `/admin` header only (spec asks for it there).
- xdotool synthetic clicks could focus buttons but not the login
  text fields (worked around by screenshotting the dialog alone; the
  real flow is covered by the `/api/login` + `connect_secure` paths in
  tests). Native login UX verified by code path, not pixel-perfect.
- `UploadChunk` sends the whole remaining tail at once (bounded by
  `--max-upload-mb`); a progress UI during slow links is a nice-to-have.
- Nico's stage-10/14/15 items still wait on him, as before.

## Stage 13 — plots, spike dots, help, manual, examples (commits 7824c1f..7d01c45)

Nico's 15:08 browser feedback: plots don't restart on Reset, no time
window, no v–u plot, spike dots invisible, watching is hidden, no help,
no manual, no examples. Items A–K plus L (admin effective quota).

### What each finding turned out to be

1. **Reset really didn't clear plots** — `v_hist`/`raster` only cleared
   on `net_gen` bump (new/load/`net.clear()`); `ClientMsg::Reset` and
   `sim.reset()` both send the sim clock backwards but kept the old
   trace. Fixed inside `PlotHist.feed`: any backwards step in `t` clears
   all series first. Test: `plot_hist_resets_on_time_reversal`.

2. **No sliding window** — `egui_plot` default auto-bounds over the full
   4096-sample history; drag/zoom kills auto-bounds permanently so new
   data ran off the edge. Now: sliding window (default 200 ms, editable
   chip), Follow mode on by default and re-armed by the "Follow" button
   or double-click, Y fixed at −90..+40 mV for Izhikevich with an
   "Auto Y" toggle.

3. **Spike dots were indeed invisible** — `Particles::feed` keyed flights
   to `snap.time_ms`; a 1 ms flight vs ≥16 ms between frames meant every
   flight expired before one frame drew it. Rewrote `Particles` on
   `wall_ms`: each spike launches at sight-time and flies for
   `max(150 ms, delay*dt*20)` of wall clock, capped at 400 live dots, View
   toggle "Spike dots". Test in `scene.rs`.

4. **Watching was invisible** — inspector "Plot v(t)" button, in-plot
   empty hint, legend rows now show `Neuron_3`-style names (not raw ids),
   context-menu entry unchanged ("Watch v(t)").

5. **`help` was one flat string** — typing `help` (no parens) errored.
   Now `help` / `help()` / `help("topic")` all work, topics:
   net/grid/syn/input/sim/readout/io/streams/misc/examples, unknown
   topic lists them. `sim/tests/help.rs` covers every form.

6. **v–u plot added** — `PlotTab::Phase`, per-watch trajectory over the
   same window, newest point marked, shares the v(t) palette. Server
   side: `Snap.watched_u` + CUDA `urec_dev` ring mirroring `vrec_dev`,
   `Sim::drain_vu` drains u before v (before the counter reset).

7. **Examples** — `editor/src/examples.rs`, Help → Examples ▶ + an
   "Examples ▾" button in the Script editor. Three bundled scripts:
   first_spikes (5-neuron chain, watch 0-4), grid_wave (20×20 sheet),
   lsm_readout (216-cell reservoir + delta readout, prints accuracy).
   `sim/tests/examples.rs`: all eval headless; #3 prints 0.65 vs 0.35
   raw-stream baseline.

### Real bugs found by the browser verification

- **Auto-watch fired against the pre-run net** — the lsm.ne2 session
  already had ≥5 neurons, so `example_watch` applied to stale geometry,
  retired, then the eval's `net.clear` gen bump wiped the selection.
  Fix: `example_geo_covers(base_ver, geo, ids)` — only a Geo newer than
  the Run-time version counts, plus a 45 s deadline.

- **Stale click picks clobbered state** — a queued `pick_click` that
  starved behind hover picks resolved seconds later and cleared the
  just-applied watch. `pick_click` is now `(Vec2, wall_ms)` and expires
  after 800 ms.

- **`GeoDelta.gen` dropped by the fold** — deltas carry the wire gen but
  `snap.rs` never wrote it to `Shared.net_gen`; a delta crossing a
  generation boundary wouldn't reset id-keyed state. One-line fix + a
  fold test (was red before it).

- **HashSet churn resent Watch every frame** — rebuilding `selection`
  each frame re-shuffled the id order so `ids != watch_sent` kept
  flapping. Now rebuilt only when membership differs; `watch_sent`
  compared as a set.

- **Drag hint overlapped the mode hints** — both `painter.text` calls
  drew at `left_bottom+8,-8`; moved the drag hint up one line.

- **`default_pos` missing on Script editor** — its position varied run
  to run (that's why my synthetic Run clicks kept missing).

### Tests (all red-then-green per item)

`PlotHist` reversal, `plot_follow` bounds, phase points, particles
wall-clock, help topics (`sim/tests/help.rs`), orbit sign
(`camera.rs`), `MoveInputs` e2e + proto round-trip + save/load (`sim`),
`drag_plane`/`axis_hit`/re-anchor (`view.rs`), example evals
(`sim/tests/examples.rs`), `example_geo_covers` + `geo_delta_propagates_gen`
(`editor`), admin `(default)` quota marker (`sim`).

`cargo test --workspace --release` all green; `sim --no-default-features`
green; `webgpu-help.test.mjs` 12/12; `cargo fmt --check` clean.

### Screenshots (`review/stage13_*.png`, on disk)

Own port **:3513**, headful Chromium on Xvfb :99, shared `lsm.ne2` server
(each session sees whatever net the last eval left — that's why several
shots show the example nets rather than the lsm wall):

- `stage13_empty_hint` — "no watched neurons — click a neuron to watch
  it, or right-click 'Watch v(t)' / inspector 'Plot v(t)'".
- `stage13_console_help` — `help` output: API index +
  net/grid/syn topic lines.
- `stage13_manual` — floating Manual window, page list + "Getting
  started".
- `stage13_examples_menu` — Help → Manual/Shortcuts/Examples▶/About.
- `stage13_example_first_spikes` — 5-neuron chain + "watching 5",
  v(t) traces at t≈10 s, AutoRun on.
- `stage13_example_grid` — 20×20 sheet, 400 neurons, "spike dots" chip
  on, spikes=1600+.
- `stage13_example_lsm` — reservoir cubes, console tail shows
  "readout accuracy on reservoir trace: 0.65 · raw-stream baseline
  (expect ~0.5): 0.35".
- `stage13_plot_follow` — sliding 200 ms window at t≈6.2 s.
- `stage13_reset_follow` — after Reset + re-Run, fresh window t≈750 ms
  proves history restarted (item A+B).
- `stage13_phase` — v–u trajectory loops (v mV axis), per-watch colors.
- `stage13_spike_dots` — dots mid-flight along edges (grid_wave, yellow
  specks on the wire lines between cubes).
- `stage13_move_drag` — mid-drag: "move d(15,0,-17)" in status, plane
  grid through anchor, hint strip "plane: x-z · Shift: vertical ·
  X/Y/Z: axis lock · Esc: cancel" above the mode hints.

### Open

- Console input is a plain egui `TextEdit`; synthetic clicks need two
  presses (activate → focus) before `keyboard.type` lands. Not a product
  bug.
- `pick_click` TTL is 800 ms — a compromise between starving picks and
  responsiveness; watch for long-frame hitches on weak GPUs.

## Stage 13B — polish pass (admin UX, manual layout, glyph coverage, palette)

### A. `server admin add` password input (commit `3a54353`)

Bug: `stdin().read_to_string` waited for EOF — on a TTY the command
"hung" (really waiting for Ctrl+D), and the password echoed on screen.

Fix (`server/src/main.rs` + `server/src/lib.rs`):
- TTY (`IsTerminal`): `rpassword::prompt_password("Password: ")` →
  12-char minimum → `prompt_password("Repeat password: ")` → mismatch
  aborts. Echo disabled both times.
- Non-TTY: `first_password_line()` reads ONLY the first line via
  `BufRead::read_line`, strips `\r\n` — multi-line paste can't smuggle
  a second password in, and `echo pw | server admin add` scripts keep
  working.
- Test: `piped_password_reads_first_line_only` (lib).
- Live-verified both paths: piped multi-line input takes line 1 only;
  under a real pty the prompts show and typing is hidden.
- `docs/deploy.md` updated (prompt behaviour + min length + piping).

### B. Manual window layout (`92f1940`, `30612fc`, `edf70b4`)

Bug: body text ran off the window's right edge.

Two REAL bugs behind it (found by reproducing the window headlessly):
1. `ScrollArea` inside `ui.horizontal` claims the row's *unbounded* x
   extent → window resized to content each frame (~+200px/frame until
   it hit the screen edge). Headless probe: clip x2 grew 857→1161.
2. `Label::wrap()` inside `horizontal_wrapped` squeezes into the row's
   leftover sliver → word-per-line columns.

Fix: body column is now `allocate_ui_with_layout(vec2(w,h),
Layout::top_down)` with `w = (clip_w - 150).clamp(80, 480)` — bounded
and stable; each md line renders as ONE `LayoutJob` (bold/code inline
formats) that wraps as a whole paragraph. Default window 640×520.

Tests (`manual::layout_tests`):
- `body_text_wraps_to_window` — every embedded page, painted-text
  width ≤ 340px in a 360px window (was failing: text ran off).
- `manual_window_width_is_stable` — frame-over-frame growth regression.
- Verified in browser: `review/stage13b_manual_1440.png` (1440×900) and
  `review/stage13b_manual_800.png` (800×700) — list left, wrapped
  scrollable body right.

### C. Tofu glyphs (`47500eb`)

Probe (`Fonts::has_glyph` vs `FontDefinitions::default()`) showed the
bundled proportional font lacks: → (U+2192) ▾ ⇑ ✕ — all rendered as
boxes in proportional text (monospace is fine, console unaffected).

Replacements: `→`→`->` (empty-plot hint, file-rename status, manual
pages), `▾` dropped ("Examples" menu already draws its own arrow),
`↑`→`^` (Files up-button), `✕`→`x`, `⇒`→`=>` in docs.

Regression test `ui_strings_stay_in_font_coverage` scans every
string-bearing line of `editor/src/*.rs` + all manual pages against
the real `Fonts` — any future uncovered glyph fails the build.

### D. Manual colour legend (`480e7bd`)

The legend claimed "orange = excitatory, blue = inhibitory" — wrong on
both counts. Actual palette (now shared constants in `scene.rs`):
- Neurons: `NEURON_EXC` light blue `[0.55,0.8,1.0]`, `NEURON_INH` pale
  purple `[0.85,0.6,1.0]` — both flash `SPIKE_FLASH` orange on spike.
- Synapses: `SYN_EXC` salmon, `SYN_INH` blue, `SYN_SELECTED` golden.
- Spike dots: `DOT_EXC` yellow, `DOT_INH` light blue.

`gpu.rs` uniforms now use the constants; `palette_matches_shader` test
asserts each `vec3f(...)` literal in shader.wgsl still matches (wgsl
can't share Rust consts — the test pins them). getting-started.md
rewritten. All other manual pages checked — no other colour claims.

### E. Leftover Stage-12 items (verification only, no code)

- L (effective quota): done in `801d84a` — admin shows effective
  values, raw-0 rows marked `(default)`.
- M (disable kills sessions): already implemented —
  `MultiServer::disable` calls `set_status` + `delete_user_sessions` +
  `runs.drop_user`; `UserRun::stop` shuts down live sockets; test
  `disable_kills_session_and_ws` proves the WS dies and the old cookie
  redirects; `session_user` gates on `status=="active"`.

### Screenshots

- `stage13b_manual_1440` / `stage13b_manual_800` — Manual at 1440×900
  and 800×700: page list left, wrapped body right, new colour legend
  visible, ASCII arrows in the empty-plot hint.

## Stage 14 — input lag: measure first, then fix

Complaint: dragging egui windows and pulling a selection frame lags the
cursor, on both browser and desktop.

### Method

`update()` timestamps instrumented identically on both builds (scratch
worktree at Stage-13B HEAD vs the fixed tree; patch reverted after):
wasm pushed `wall_ms` to `window.__ft`, native printed `FT` lines to
stderr. Browser driven via playwright on own ports (:3515 before,
:3516 after); native under Xvfb :99 + xdotool, `--connect` to :3515
(lsm.ne2, 512 neurons, CUDA server).

### Why it lagged

Every `update()` unconditionally rendered the whole scene offscreen AND
queued a full-viewport pick pass whenever the cursor hovered the canvas
— submits pile into the wgpu present queue (default depth 2), so the
displayed frame carries input that's 1–2 frames old. It's present-queue
latency, not CPU frame time: on this hardware update() never exceeded
~1 ms CPU per frame unthrottled, and vsync-capped it holds 16.7 ms.

### Browser (headful Chromium under Xvfb, vsync 60 Hz)

`update()` calls per gesture window — p50 interval:

| gesture                | before            | after                    |
|------------------------|-------------------|--------------------------|
| hover                  | 150 @ 16.7 ms     | 150 @ 16.6 ms            |
| egui window drag       | 121 @ 16.7 ms     | 121 @ 16.6 ms            |
| selection-frame drag   | 120 @ 16.7 ms     | 120 @ 16.7 ms            |
| idle, sim paused       | 178 @ 16.7 ms     | **1 frame** (sleeps)     |
| `sim.run(4000)` eval   | 179 @ 16.8 ms     | **9 @ ~467 ms** (snap cadence) |

Unthrottled run (vsync off) confirmed update() cadence is bounded by
event/repaint rate, ~0.8 ms main-thread cost — GPU-side queue depth is
the lag lever, addressed by the latency cap below.

### Native (Xvfb, no vsync — interval = app cadence)

| phase                  | before                  | after                        |
|------------------------|-------------------------|------------------------------|
| idle, sim paused       | 4543 upd @ 0.84 ms spin | **0 frames**                 |
| hover                  | 4256 @ 0.86 ms          | 475 @ 3.67 ms (input-driven) |
| egui window drag       | 1804 @ 0.85 ms          | 1214 @ 0.85 ms               |
| selection-frame drag   | 1766 @ 0.86 ms          | 1703 @ 0.83 ms               |
| idle after drag        | 4420 @ 0.87 ms          | **2 frames**                 |
| during `sim.run` eval  | ~2.35 ms continuous     | ~2.31 ms continuous          |

Drags still repaint per input event — interaction cadence unchanged.
The win is ~4600 wasted renders/4 s at idle eliminated, and hover no
longer queues a full-viewport pick every frame.

### Fixes

- **B. SceneKey dirty-check** (`gpu.rs`, `view.rs`): fingerprint of
  size/camera/preview/hover/selection/activity/polylines/particles/
  planes/draw-params/geo-version; `render_offscreen` skips when
  unchanged, blit keeps showing `scene_tex`. `set_scene` invalidates.
  Sim activity changes every snap → still renders while running.
  Unit test: `scene_dirty_only_on_real_change` verifies every field
  invalidates and identical keys skip.
- **C. Pick scissor**: `request_pick` sets `set_scissor_rect(px,py,1,1)`
  — raster work confined to the target pixel.
- **D. Frame latency**: `desired_maximum_frame_latency: Some(1)` in
  native + web wgpu options — submit blocks until at most 1 frame is
  queued ahead, so a drag can't sit behind 2 presents.
- **E. On-demand repaint**: unconditional `ctx.request_repaint()`
  replaced by an animating check (running/busy/server traffic/camera
  tween/drags/particles/picks/uploads). Server traffic signals via
  `Shared.dirty_seq` (bumped per ServerMsg → `net_touched`). Hover
  picks now queue only when the target pixel/camera/geometry changed
  (`last_hover_pick`) — the old every-frame pick would otherwise hold
  `pick_in_flight` and keep the app awake forever under a still cursor.
- **A. Frame-stats readout**: View ▸ "frame stats" toggles a
  `frame ms / fps` overlay (viewport top-right; EMA, idle gaps >200 ms
  skipped).

### Notes

- Browser p50 is vsync-pinned at 16.7 ms in both builds — the win is
  idle→sleep, 20× fewer frames while streaming, scissored picks, and
  the queue-depth cap (end-to-end latency, not cadence).
- Native Xvfb numbers are unthrottled cadence, not GPU time; on a real
  desktop with vsync the before-build holds 60 fps but keeps the
  present queue saturated, which is the felt lag.
- `busy` is a local-worker concept; remote-server evals stream via
  `dirty_seq` → repaint per message (~snap cadence), which is also
  what keeps status/plots current.

## Stage 15 — layout correctness + palette + SceneKey audit

Review: `review/STAGE15.md`. Screenshots:
`review/stage15_{before,after}_{800,1440}.png`,
`review/stage15_after_800_manual.png` (browser build, `:3517`).

### A+C. Windows on-screen, hint strip reserved

- `usable_rect(ctx)` = `ctx.available_rect()` after the toolbar /
  status / hint panels are laid out — the strip between them.
- Every floating window (Inspector, Inputs, Files, Manual, Console,
  Script editor, Plots, all dialogs, Login, Shortcuts) now goes
  through `app_win` = `Window::new().constrain_to(usable_rect)`.
  egui re-applies the constraint **every frame**, so persisted
  off-screen positions and narrow resizes get pulled back; it also
  caps window size to the strip.
- Defaults now come from `clamp_window_rect(want, usable)` — position
  clamps inside and size shrinks to fit. The Manual additionally caps
  at 70% of the usable strip (was 640×520 covering ~80% of an
  800 px window).
- The mode/drag hint moved from in-canvas paint (which windows could
  cover) to a `TopBottomPanel::bottom("hints")` above the status bar.
  Because it's a panel it shrinks `usable_rect`, so no window can
  ever overlap it; the text wraps to a second line at narrow widths.
- Tests: `window_clamp_keeps_rect_inside_usable` (in/right/bottom/
  left/oversized cases).

### B. Toolbar wraps by groups

`ui.horizontal` inside `horizontal_wrapped` claims the whole remaining
row as its desired size, so egui's wrap check (`available < desired`)
never fires — the old rows split mid-control ("a" / "x1.0" /
"realtime" orphans at 800 px). `toolbar_group` allocates each group
with an explicit upper-bound width via `allocate_ui_with_layout`, so
the wrap check sees a real footprint and moves the group whole
(cursor still advances by the group's actual width — over-estimating
the bound only packs slightly early). Groups: run/step/reset, speed,
backend, dt, noise / modes, plane, view toggles, grid-view. Test:
`wrapped_toolbar_groups_never_split` (same-row atoms + no panel
overflow + wrap engages).

### D. Legend vs render — gold lines explained

The gold radial lines with "nothing selected" are **input→target
links** (`INPUT_LINK_COL`), not stale selection highlight:
`write_selection` zeroes the selection bitset every frame, and the
shader only tints synapses gold where a selected bit is set. So the
render was right and the text was wrong — the manual never mentioned
input links. Fixed in `docs/manual/getting-started.md`; the link
colour is now derived from a shared `scene::INPUT_COL` (same gold as
the input cubes' wgsl literal, pinned by `palette_matches_shader`).

### E. One colour rule: warm = excitatory, cool = inhibitory

Before, neurons inverted the convention the synapses/dots already
used (excitatory neurons were blue, excitatory synapses salmon).

| element | before | after |
|---|---|---|
| neuron excitatory | light blue `0.55,0.8,1.0` | warm amber `1.0,0.62,0.35` |
| neuron inhibitory | pale purple `0.85,0.6,1.0` | cool blue `0.55,0.68,1.0` |
| synapse exc/inh | salmon / blue | unchanged |
| spike dot exc/inh | yellow / light blue | unchanged |
| input node + links | gold / gold | gold, now one shared const |

`NEURON_EXC/INH` flow to the shader via the `exc_color`/`inh_color`
uniforms (`write_cam`), so only `scene.rs` changed; the legend in
getting-started.md states the rule. Visible in
`stage15_after_800.png` (amber excitatory, blue inhibitory cubes).

### F. SceneKey completeness

`scene_key` (view.rs) extracted from the `render_offscreen` call site
into a function with the audit as its doc comment. Reads of the
scene pass: viewport size, camera+aspect+vp-height, preview delta,
hover, selection/activity/polyline/particle buffers, plane state,
draw toggles (counts, show_syn, show_grid, planes_on), geo version —
all keyed. Deliberately unkeyed (compile-time or non-scene): clear
colour, palette constants, `misc.x/y` (cube size/glow), `misc.z`
(time — unread by the shader), `misc.w` (pick-mode flag — the pick
pass writes its own uniform and never touches the scene texture).
Test: `scene_key_changes_with_app_state` builds a real `EditorApp`
(`SimCtl::offline`) and flips each input — show_grid, hover,
preview_delta, plane_key, geo_version, camera orbit — asserting the
key moves, plus buffer-side inputs (sel/size/show_syn/planes_on).

## Stage 16 — hosted-server security and crash fixes (fable-review-1112)

Source: `review/fable-review-1112.md` §2 (S1–S22) and §3.1 (Q1–Q3, Q10).
Every item was re-confirmed against the code before fixing; "skipped"
below means the finding was wrong, already done, or deferred with a
reason. No changes below `f6581b6`.

### P1 — done

- **S1** `/ws` cookie upgrades now require `origin_ok`; only Bearer
  clients may omit `Origin`. `session_full` reports the auth source.
  Test: `ws_cookie_requires_our_origin` (foreign Origin→403, missing
  Origin→403, own Origin→101, Bearer+none→101).
- **S2** stop/freeze: `ServeOpts`/`Server`/`UserRun` carry a separate
  `shutdown` flag that nothing clears; once set, the worker drops every
  command except `Bye`. `drop_user`/`reap` remove the run from the map
  in a separate statement before `stop()` (lock released), and the join
  is bounded (`STOP_JOIN_TIMEOUT`, detach instead of block). Test:
  `drop_user_during_eval_is_fast` (two queued evals + stop + `GET /`
  answers).
- **S3** `ClientMsg::Abort` sets the run's abort atomic before the
  queued copy — a `while true {}` eval dies on the flag (rhai
  `on_progress` hook also checks `shutdown`). Test:
  `abort_over_socket_ends_loop_eval` (worker answers "alive" after).
- **S4** `admin add` on an existing email replaces password+name,
  forces `active`/`is_admin`, deletes that user's sessions, and prints
  "replaced". deploy.md now says to create the first admin before
  exposing the service. Test: `admin_add_replaces_existing_account`.
- **S5** memory ceilings: `SimLimits {max_neurons,max_synapses,
  max_ring_bytes}` checked in `ensure_sim` before any sim build
  (neurons, synapses, depth×neurons×4 all gated); flags
  `--max-neurons/--max-syns/--max-ring-mb`; rhai engine gets
  `set_max_array_size(10M)/set_max_string_size(64MiB)/set_max_map_size(1M)`;
  `MemoryMax=12G` added to the systemd unit. Tests:
  `sim_limits_refuse_oversized_build`, `grid_dims_validated`.

### P2 — done

- **S6** jobs: `--max-jobs` live-job cap per user (default 2),
  `--job-hours` wall-clock deadline (default 2h) checked in the rhai
  progress hook, `UserRun::stop` cancels that user's jobs via
  `JobRegistry::cancel_all`, progress rows capped at 2000, finished
  jobs pruned at 64 kept, `script.rhai`/`result.json`/`metrics.json`
  written through `checked_write`. Tests: `job_cap_and_cancel_all`,
  `job_wall_clock_limit`.
- **S7** quota on every write path: `checked_write`/`quota_fits`
  centralised; `Mkdir` counts entries. Test:
  `quota_covers_all_write_paths` (save/samples_save/write_to_file/
  dataset_gen/Mkdir/job output).
- **S8** `proto::decode` inflates through `.take(DECODE_LIMIT)` (256
  MiB) and deserializes with `bincode::options().with_limit(..)` —
  fixint+allow_trailing preserved. WS `max_message_size`/`max_frame_size`
  =16 MiB via `server_ws_config()`. Test:
  `decode_refuses_oversized_payloads`.
- **S9** login limits: `client_ip` takes the LAST `X-Forwarded-For`
  entry (attacker can't pick the bucket behind a trusted proxy);
  failure keys are ip + (ip,email) pair at 10/10min + email ceiling
  30/10min — a distributed attacker can't lock one account, a local
  brute force still dies. `limited()` no longer inserts on lookup;
  stale buckets sweep on touch. Tests: `trust_proxy_gates_xff`
  (updated to the pair semantics), `rate_limiter_lookup_does_not_insert`.
- **S10** signup: per-IP `signup:` bucket at 5, `pending_count()>=1000`
  refuses with 503, name≤80 chars, note≤500 chars. Test:
  `signup_rate_limited`. (Admin-table paging skipped — the pending cap
  bounds the flood; table renders all rows but they can't pile up.)
- **S11** live-socket invalidation: `ValidCheck` closure in
  `ws_client_loop` re-checks `accounts.session` + `status=='active'`
  per inbound frame (throttled 500ms); reaper also drops runs whose
  user left `active`. Tests: `cli_disable_drops_live_ws` (second
  `Accounts` handle = the CLI process),
  `reaper_drops_disabled_user`.
- **S12** `get_or_start` replaces a run whose worker `is_finished`;
  the whole dispatch is inside `catch_unwind` and reports the panic to
  the client console. Test: `dead_run_is_replaced`.
- **S13** per-client outbound `sync_channel(CLIENT_QUEUE=512)` —
  `fanout`/`send_to` `try_send`, drop the client when full, writer
  sends WS Close on exit; inbound `sync_channel(CMD_QUEUE=256)` — a
  flooding socket is closed on `Full`. Test:
  `fanout_drops_stalled_client`.
- **S20** `verify_login` reads the hash under the db lock, releases,
  then runs argon2 (no session lookups stall behind logins). Covered
  by `login_roundtrip` (no dedicated timing test).

### P3 — done

- **S14** HTML responses carry `X-Frame-Options: DENY`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
  `Cache-Control: no-store`, `Content-Security-Policy: default-src
  'self'; style-src 'self' 'unsafe-inline'`; non-HTML gets `nosniff`.
  Verified through existing `admin_gated_and_post_checks`/`signup_*`
  response paths.
- **S15** `url_decode_form` decodes on bytes — `%` before a
  multibyte char can't panic. Test: `url_decode_never_panics`.
- **S16** HTTP sockets get 120s read/write timeouts;
  `--max-connections` (default 512) answers 503+`Retry-After: 5`
  instead of spawning a thread. Test: `conn_cap_answers_503`.
- **S17** login POST requires `origin_ok`; Bearer `POST /logout` now
  deletes the bearer session (was cookie-only); accounts mode refuses
  a non-loopback bind without `--public-url`; `delete_expired_sessions`
  sweeps on login. Password change: skipped — `admin add` on an
  existing email is now a real reset path (S4), and a self-service
  flow needs mail verification we don't have.
- **S18** admin POSTs: unknown id → `did=no+such+user`; self-disable/
  delete/revoke refused; last-active-admin guarded via
  `has_other_admin`; `mb*(1<<20)` saturates; live quota pushed through
  `ClientMsg::SetQuota` to the run. Test: `admin_post_guards`.
- **S19** `server admin list` prints `default` for `quota_bytes==0`
  (page-side part was already done at stage 12).
- **S21** `DeleteFile`/`RenameFile` on `""` (the sandbox root) refuse.
  Test: `sandbox_root_cannot_be_deleted_or_renamed`.
- **S22** `UploadChunk` without `UploadStart` refused; open uploads
  track declared vs received bytes — over-declared data is an error
  and partial files are removed; `Hello` with wrong proto version
  emits an Error notice; >8 bad frames closes the conn. Test:
  `upload_chunks_need_start_and_count_bytes`. (`Watch`/`PlayInput`/
  `WriteFile`/`ReadFile` sizes now bounded by the 16 MiB ws cap +
  quota — no separate per-call limits added.)

### §3.1 crash fixes — done

- **Q1** echo/`Console` truncation walks back to a char boundary.
  Test: `eval_echo_truncates_on_char_boundary` (>400B echo with an
  umlaut at the cut — panicked before).
- **Q2** `Network::validate` rejects grids with axes out of range /
  duplicated dims / zero dims; `InsertSubnetJson` errors instead of
  panicking. Test: `paste_grid_bad_axes_refused_not_panic`.
- **Q3** `check_dt` shared by `ClientMsg::SetDtMs` and `sim.set_dt` —
  0, negative, NaN, or <0.01ms refuse. Test:
  `set_dt_validated_both_paths`.
- **Q10** the whole per-command dispatch in `serve::run` is inside
  `catch_unwind`; a panic reports `Error(panicked: …)` to the client
  and the worker survives. Verified structurally — every historically
  panicking input now has a dedicated test (Q1/Q2, S5 limits).

### Verification

- `cargo fmt --all -- --check` clean
- `cargo test --workspace --release` — all green (33 suites)
- `cargo test --workspace --release --features sim/cuda` — all green
- `cargo check --target wasm32-unknown-unknown -p editor
  --no-default-features --features web` — clean
- `node editor/web/webgpu-help.test.mjs` — 12/12

## Stage 16B — the strict CSP broke /app

### Proof (before the fix)

`respond()` put `default-src 'self'; style-src 'self' 'unsafe-inline'`
on every `text/html` response — including `/app`'s `index.html`, whose
inline `<script type="module">` bootstrap and wasm instantiate both
violate it. Accounts-mode server on :3519, real login cookie, headful
Chromium (the stage screenshot setup), `GET /app`:

```
CSP: default-src 'self'; style-src 'self' 'unsafe-inline'
[error] Executing inline script violates the following Content
Security Policy directive 'default-src 'self''. Either the
'unsafe-inline' keyword, a hash
('sha256-r1Ul51ihl2MdCj/B+HyQ94ZYGq+zcVX6JW8k3CUTFNg='), or a nonce
('nonce-...') is required to enable inline execution.
```

The page sat on the "checking WebGPU…" loading card forever — the
bootstrap never ran, so the wasm never even fetched.

### Fix

- `editor/web/index.html`: the bootstrap moved verbatim into
  `editor/web/boot.mjs` (`<script type="module" src="boot.mjs">`),
  added to `build.sh`'s copy list — the narrowest policy needs no
  hash and no `'unsafe-inline'`.
- `respond()` → `respond_csp(...)`: `CSP_STRICT` for the account and
  admin pages (no scripts at all), `CSP_APP` for `/app` and any html
  under `--web` (`serve_static`): `default-src 'self'; script-src
  'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'`.
  `connect-src` stays inside `default-src 'self'`, which covers the
  same-origin `ws(s)://` upgrade — confirmed live in the browser run.

Test: `app_csp_allows_wasm_not_inline_scripts` — `/app` and
`/app/index.html` carry `wasm-unsafe-eval`, their `script-src` clause
has no `'unsafe-inline'`, nothing has `'unsafe-eval'`; `/`, `/login`,
`/admin`, `/signup` keep the strict policy (no `script-src` at all).

### After

Same run: `GET /app` → the new CSP, **zero** console errors, editor
drawn (toolbar, grid, Inspector, Plots, Console) —
`review/stage16b_app_csp.png`.

## Stage 17 — sim semantics + editor/protocol fixes (review/fable-review-1112.md)

One commit per finding, above `88097bf`.

**M1 Dale's law** (`synapse_csr` signs outgoing weights by source kind:
exc `+|w|`, inh `-|w|`; stored `w` is a magnitude). `set_kind` gives
fast-spiking defaults (a .1 b .2 c -65 d 2) when the model still
matches the previous stock model — "explicit" means any parameter
the user touched, which survives the kind switch. `net.kind` /
`net.set_kind` script APIs; bench.rs manual sign removed. Tests:
csr sign table, inh-synapse suppresses a driven target (CPU), and
`gpu_vs_cpu` now covers inhibitory nets.
**M2 delays**: `DEFAULT_CONDUCTION_SPEED = 100` world units/ms
(typical tutorial grid spans several delay steps; old files keep
their stored `Meta.conduction_speed`). `net.set_conduction_speed`,
`net.set_delay`, status bar shows `delays 1..N steps`. Test:
tutorial grid produces >1 distinct delay step.
**M3 settings in the file**: `Meta` gains dt_ms, seed,
conduction_speed, synapse_tau_ms, readout_tau_ms, noise_rate_hz,
noise_kick_mv (serde defaults — old files load). Load → live
`SimConfig`; save ← live. Getters `sim.synapse_tau/readout_tau/dt`.
Test: tau 10 ms round-trips through save/load into a fresh session.
**M4**: decay `exp(-dt/tau)` everywhere (was Euler — chose accurate
decay over refusing tau<dt); `delay_steps` > u16 ring capacity is an
error (`synapse_csr`/`CpuSim::new` now `Result`), not a silent clamp.
Test: small-tau decay + overflow error.
**A5 honest rerun** (`lsm_hidden.rhai`, 8 sources/4 pairs/p=.5,
50 ms, strength .8, 200/120/120 split, 12 seeds × 500 perms):
main liquid AUC .895±.040, ridge .878±.039, val ridge .874±.041;
signal-off Δ .490±.040 (was the whole signal); raw baseline
.482±.046; coincidence baseline 1.0; permutation null .501±.142,
pooled p=0.0, mean per-seed p≈.002 — null centred at chance across
all 12 seeds (the reviewer's seeds-7-and-42 concern is gone).
`lsm_readout.rhai` retuned honestly under Dale dynamics: W_IN 9.0
saturated the reservoir at chance; the mechanism is coincidence
detection — W_IN=2.8, SHARED=80, PRIV=0 keeps single inputs
sub-threshold and pairs super-threshold.
**Q4** `AddInput`/`AddReadout` with dangling neuron ids refused
server-side. **Q5** a refused edit returns the optimistic
`edits_in_flight` slot (previously every later edit carried a stale
base and was refused forever). **Q6** typed `SaveNet`/`Saved`
ack. **Q7** `net_path`/recent move only on confirmed `NetLoaded`.
**Q8** upload slot frees on `FileWritten`/upload error (the
100%-forever stall). **Q9** `egui::Event::Paste` only pastes into
the net when no widget holds keyboard focus. **Q11**
`Network::validate()` rejects NaN/inf positions, model params,
weights, delays, tau, bad refs, degenerate grids, malformed
readouts/samples. **Q14** local-mode mkdir/rename/delete/upload
errors reach `files.status`.
**U1** last error in the status bar (light red) + unread dot on the
Console entry. **U3** Plots stacks above Console under ~930 px.
**U4** manual reflows paragraphs, h2 16 px. **U5** `Shared.connected`
+ disconnect banner; native reconnects with the stored token, wasm
reloads. **U7** input links dimmed (α .75→.35) + own toggle +
legend. **U8** rename button works, watch = 8 lowest ids + "N of M",
grid dialog ranges + count, connect weight by kind (exc 6/inh 8),
Alt = inhibitory insert, weak empty-plot hint, Auto-Y default,
terse hints <1100 px, '?' button dropped, Jobs tab fetch-on-open
(pushes keep it live), app.ron chmod 0600 + File > Log out.
**U2-kind** inspector exc/inh selector via `NetEdit::SetNeuronKind`.
**D** `drop_user` test asserts `run.dead()` and live `GET /` during
shutdown (red-checked: without the shutdown flag the test hangs).

Skips: pluggable-model refactor (waits for Nico per the stage doc);
v(t) fixed range kept as an opt-in pin (Auto-Y is now the default);
`examples/lsm.ne2` was regenerated by the A5 experiment and landed
inside `f08e42f` — valid file, exercises M3 Meta persistence.

## Stage 17B — screenshots + reconnect backoff

Browser shots on :9417 (Xvfb :99, headful Chromium — headless offers
no WebGPU adapter): `stage17_1440.png`/`stage17_800.png` show the
inhibitory neuron selected (kind selector `exc | inh` in the
Inspector) and `delays 5..6 steps` in the status bar;
`stage17_manual.png` shows reflowed paragraphs; `stage17_disconnected.png`
shows the banner (`connection to server lost · retrying in Ns · Retry
now`) plus the console's `! connection to server closed`.

U5b: the wasm retry was a page reload — a dead server meant a reload
loop. Replaced by an in-place reconnect on both transports (native
`connect_secure`, wasm fresh `WebSocket`) driven by a persistent
backoff: first attempt +2 s, doubling to 60 s, reset only on a fresh
login; a wasm retry whose pending socket immediately dies re-arms at
the grown delay — attempts march 2,4,8…60 and never tighten. The
banner counts down between attempts with a Retry-now button.

Two real bugs found by the screenshots: (1) `dirty_seq` is only read
inside `update()`, which needs a repaint — an idle page never woke
to notice a closed socket, so the banner could not appear; a 1 s
idle heartbeat fixes it (bounds the notice to ~1 s). (2) pgrep-style
kills during verification kept hitting shell wrappers, not the
server — the CDP run showed `WS OPEN→CLOSED→OPEN(retry)→ERR_REFUSED→CLOSED`
and a restart recovered the session in place with no reload.
