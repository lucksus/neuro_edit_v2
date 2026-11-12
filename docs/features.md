# Feature documentation

Two binaries share one `Network` model (`net-format`):

| binary | role |
|---|---|
| `sim` | headless simulator: run scripts, eval snippets, benchmarks |
| `editor` | interactive 3D editor/simulator (egui + wgpu) |

The editor does not launch `sim` as a subprocess — the simulator is compiled
in via the `sim` crate and runs on a worker thread (see *Architecture* below).
A browser UI could connect the same way by exposing the worker's `Cmd` channel
over a websocket and shipping the published snapshots; nothing in the design
requires the wgpu frontend.

## `sim` CLI

```sh
sim --net file.ne2 [--script x.rhai] [--eval '…' --eval '…']
    [--ms 1000] [--dt 0.1] [--seed 1]
    [--cuda | --cpu]
sim --bench N [--bench-k 100] [--wscale 1.0] [--w-inh 5.0]
    [--warm 100] [--ms 1000] [--noise 0.0]
sim --int-weights …        # CUDA only: fixed-point accumulation → bitwise
                           # reproducible runs (float atomics are not)
```

- `--net` loads `.ne2` (binary) or `.json`.
- `--script` runs a Rhai file; `--eval` snippets run in order, in the same
  scope (`net`, `sim`, `rng` globals — see `docs/scripting.md`).
- `--bench N` builds a random N-neuron net (~`bench-k` out-degree each) and
  reports steps/s.
- `--noise` adds Poisson kicks; `--seed` seeds everything.

## `editor` CLI

```sh
editor                                   # empty network, CUDA backend
editor --net file.ne2 [--cpu] [--script x.rhai]
editor --screenshot out.png [--net f.ne2] [--script x.rhai]
      [--after-ms 500] [--width 1280] [--height 800] [--no-synapses]
      [--planes] [--noise-hz 30 --noise-kick 20]
```

`--screenshot` renders one frame headlessly (no X needed — surfaceless
wgpu) and exits; `--after-ms` simulates first. `--planes` renders grid
groups as activity planes (below); `--noise-hz`/`--noise-kick` drive the
background Poisson kicks so the shot isn't dark.

## Editor UI

### Menu bar

- **File**: New, Load…, Recent files, Save, Save as…, Quit.
- **Edit**: Copy / Cut / Paste (Ctrl+C/X/V — subnets serialized as JSON to
  the system clipboard), Delete selected (Del), Add selection to group….
- **Insert**: Neuron (I), 2D grid…, 3D grid…, Input… (on the current
  selection), Samples…, Readout from selection….
- **Group**: Create connections…, Set synapse weights…, Set exc/inh….
- **Help**: About (original splash artwork).

### Toolbar row 1 — simulation

Run/Pause (play/pause icons), Step 10ms, Step 100ms, Reset, speed slider
(0.05–20×, logarithmic), backend combo (cuda/cpu), dt DragValue
(0.01–1.0 ms), noise Hz DragValue.

### Toolbar row 2 — modes and view

- Modes: **Navigate**, **Insert** (neuron icon), **Connect** (icon), **Move**.
- `plane x-z / x-y / y-z`: the grid plane + movement plane for drag ops.
- Checkboxes: `synapses`, `grid`, `console`, `plots`, `inspector`, `script`.
- Grid render combo `auto|cubes|planes`: how groups with lattice metadata
  draw (see *Grid activity planes*).

### Viewport

- Left-click: pick a neuron/input/readout (GPU ID pass; pick targets are
  drawn slightly fat so dense scenes stay clickable).
- Shift-click / Shift-drag: additive selection.
- Left-drag: box selection (Navigate) or move selected neurons on the
  current plane (Move mode); drag in empty space in Insert mode places a
  neuron on click.
- Right-drag: orbit camera. Mouse wheel: zoom. WASD/QE: pan. Double-click on
  a neuron: focus camera on it.
- `I`/`C`/`M`: switch to Insert/Connect/Move. `Del`: delete selection.
- Connect mode: click source neuron, then target neuron — status line shows
  progress; a missed click keeps the pending source (status "missed — click
  a target neuron"); Esc cancels.
- The camera is clamped above the floor grid.
- Synapses render as lines with density-aware alpha; below ~3 px neuron
  spacing a deterministic keep-every-Nth LOD thins them (screen-space, so
  zooming in restores detail). Sparse networks are unaffected.

#### Grid activity planes

Groups created by `create_2d_grid`/`create_3d_grid` carry persistent
lattice metadata (`Group::grid`). Such a group can render as **one
textured quad** instead of one cube per neuron:

- Each texel shows the member neuron's spike trace (1.0 on spike, exp
  decay τ ≈ 20 ms) in a viridis colormap — one draw call per group.
- The toolbar combo picks `cubes` (always instanced cubes), `planes`
  (always the sheet), or `auto`: per-grid zoom LOD that switches to the
  plane once texels shrink below ~4 px on screen. Zooming back in returns
  the cubes for editing — and so does a small selection (≤256) touching
  the grid's members, so selected neurons stay directly editable in
  auto mode.
- While a group is plane-rendered its member cubes and member–member
  synapses are culled in the vertex shader (the sheet already shows that
  wiring); synapses to outside neurons stay.
- Picking still lands on individual neurons: the plane's pick pass writes
  the member id of the clicked texel, so selection/inspect/move work on
  the neuron, not the sheet.
- 3D lattices get a per-group *collapse* setting in the inspector (Groups
  section): which lattice dim collapses and whether the plane shows a
  `slice` (with index), the `max` or the `mean` projection along it.
- Moving/deleting/copying part of a lattice drops the metadata — the
  group falls back to cubes instead of drawing a wrong sheet.

A 1,000,000-neuron `1000×1000` grid is a regression test
(`editor/tests/big_scene.rs`): build + upload + draw complete in well
under a second; the cube path would not fit comfortably.

#### Context menus (right-click)

- On a neuron: Watch v(t), Connect from here, Delete.
- On an input: Set value… (slider).
- On a readout: Plot weights.
- On empty space: Insert neuron here (switches to Insert mode), Paste,
  Frame network.

### Panels

- **Inspector** (right): selection info + Delete, Groups with neuron counts,
  Inputs (live values), Readouts (unit counts).
- **Plots** (bottom): `v(t)` membrane-potential traces of watched neurons,
  `raster` spike raster, `readouts` trained weights per named readout
  (dropdown), `samples` (t,value) channels.
- **Console** (bottom): Rhai REPL — Ctrl+Enter or Run evaluates, Up/Down
  recall history, Clear wipes the log. Results print as `= value`, errors
  as `! error`. Syntax highlighting (keywords, `net`/`sim`/`rng` globals,
  numbers, strings, comments).
- **Script editor** (`script` checkbox): named multiline scripts with the
  same highlighting; Run executes on the worker, Save/Open round-trip `.rhai`
  files, "Save to net" stores the script inside the `.ne2`.

### Status bar

`t` (ms), mean `rate` (Hz), `spikes` total, `sim speed` (steps/s), plus
mode hints and connect-mode progress.

## Network model and `.ne2` format

`net-format::Network` is flat arrays: neurons (position, Izhikevich model,
exc/inh kind, name), synapses (src, tgt, weight, delay — distance-based),
groups, inputs (named stimulus → weighted target list), samples (named
(t,value) series), readouts (name, unit ids, trained weights), and embedded
scripts. Groups may additionally carry `grid` lattice metadata
(`dims`, `origin`, `spacing`, `axes`) written by the grid builders — the
fuel for the activity-plane renderer.
`.ne2` = `NE2\0` magic + u32 version + bincode payload; version 2 adds
the grid field, version-1 files still load (grids come out empty) and
JSON simply defaults the absent field — both readable by either binary.

## Simulation semantics

Izhikevich neurons, `v ≥ 30 → spike`, exponentially decaying synaptic
currents (`synapse_tau`, default 2 ms), distance-based axonal delays on a
ring buffer (`delay_ms = dist / conduction_speed` (5000 u/ms), recomputed
when neurons move), optional Poisson noise
kicks, external current per neuron, named inputs (constant value or
per-step played series). Readout features are per-neuron traces: 1.0 on
spike decaying by `(1 − dt/readout_tau)` per step. CPU and CUDA backends
implement the same update order; `--int-weights` makes CUDA accumulation
fixed-point so repeated GPU runs are bitwise identical.

## Architecture

One `Network` is shared by everything. In the editor a **worker thread**
owns `Network` + `Sim` + Rhai env; the UI thread sends `Cmd` messages
(`EditNet`, `Eval`, `RunMs`, `AutoRun`, `SetBackend`, `SetDtMs`,
`SetNoise`, `Reset`, `Abort`, …) over a channel and reads immutable
snapshots (geometry, v, activity, spikes, readouts, per-grid-group spike
traces) published under an `RwLock`. Long `sim.run` calls chunk every
~200 steps so pause/abort stays
responsive. Rendering: one instanced draw for all neuron cubes, pull-based
line draws for synapses/overlays, a full-resolution ID pass for picking,
and one textured-quad draw per grid group (the CPU publishes the f32
trace buffer; the 3D→2D slice/projection happens host-side on upload).
Measured 97,336 neurons + 1.1 M synapses ≈ 4.4 ms/frame on an RTX 3090.
