# Feature parity: neuro_edit (C++/Qt4, 2011–2012) → neuro_edit v2 (Rust)

This is the parity checklist. Every user-facing feature of the original editor
(`NeuroEdit/`) and headless runner (`NeuroRunner/`), plus the model behaviour of
`NeuronModels/` that is visible through them.

Legend: `[ ]` open · `[x]` done · `[~]` partial (note) · `[-]` deliberately out of scope (note)

Checked against the C++ source at `~/code/neuro_edit-ref` (commit 8c4e7ce), 2026-11-11.

## 1. Network model objects (NeuronModels/)

| # | Original feature | v2 |
|---|------------------|----|
| 1.1 | Izhikevich point neuron (a,b,c,d; default RS 0.02/0.2/-65/8), forward Euler, v>=30 spike + reset | [ ] |
| 1.2 | Per-neuron position (3D, `Point`) | [ ] |
| 1.3 | Exponentially decaying synaptic current: spike adds `w` to a trace decaying at `time_constant/100` per ms (default tc=50 ⇒ τ≈2 ms); dendritic current = w·trace | [ ] |
| 1.4 | Axon conduction delay = 3D distance / speed (default 5 mm/ms); per-axon speed | [ ] |
| 1.5 | Axon tree of AxonNodes (branching, fan-out) — v2 note: trees collapse to one delay per synapse pair; explicit node editing is `[-]` unless drawn | [~] |
| 1.6 | Dendrite tree of DendriticNodes (no delay, sums current to root) — v2: root only, see 1.5 | [~] |
| 1.7 | Synapse object between axon and dendrite node, weight + time_constant properties | [ ] |
| 1.8 | CurrentInducer: injects a driven current into a dendritic node; `active` flag | [ ] |
| 1.9 | Samples: named (t,value) series; linear interpolation; drives CurrentInducers / discriminator outputs; write_value during sim; load/save `.nsamples`; clear; constant-value mode | [ ] |
| 1.10 | Group: named object set; user actions (see §5); add/remove objects; inputs | [ ] |
| 1.11 | LSMReadOutNeuron: sigmoid unit over exp-filtered spike traces (τ=2 ms default, same rate law as synapses), bias weight, `learn(target, rate)` delta rule, `init_weights_random_uniform`, `connect_with_group`, output written to attached Samples | [ ] |
| 1.12 | LinearDiscriminator + LDConnection (weighted sum of LD outputs → sigmoid), used for MLP | [-] MLP/backprop out of scope for the LSM use case |
| 1.13 | Group::do_backprop / MLP (`MLP/`, `create_mlp`) | [-] out of scope (LSM only) |
| 1.14 | PhotossSignalImporter (text file → bit_at_time) | [-] replaced by generic Samples/CSV input |
| 1.15 | MNISTData | [-] out of scope |
| 1.16 | RandomGenerator script API (uniform, gauss) | [ ] |

## 2. Simulation control (Simulation/Controller/Network)

| # | Original feature | v2 |
|---|------------------|----|
| 2.1 | simulation_step (dt, default 0.1 ms) editable | [ ] |
| 2.2 | real_step (wall-clock ms per sim step) ⇒ playback speed throttle; not_matching_speed signal | [ ] |
| 2.3 | Continuous run in a thread; start / pause (request_stop) / resume | [ ] |
| 2.4 | run(time_ms), run_one_step(step) from scripts | [ ] |
| 2.5 | reset() — network state + time to 0 | [ ] |
| 2.6 | sim.time_ms | [ ] |
| 2.7 | Simulation must be saved before first GUI run (filename guard) | [-] v2 sim does not require a file |
| 2.8 | Controller: new / load(.neb) / save(.neb) / export XML(.nex) / import XML / close | [~] .ne2 binary + JSON export; old .neb import is `[-]` (nice-to-have only) |
| 2.9 | Scripts stored inside the file (simulation scripts + network scripts) | [ ] |

## 3. 3D view (glscene.cpp / Drawable*)

| # | Original feature | v2 |
|---|------------------|----|
| 3.1 | 3D rendering of all objects: neuron (box/frustum), axon node, dendrite node, axon (line + travelling spike dots), synapse (cone), current inducer (cone), samples (cube), linear discriminator (cube), LD connection (line), group (handle/box) | [~] v2 draws neurons, synapse lines w/ spike dots, inputs, groups, readouts; node-level trees `[-]` |
| 3.2 | Camera: right-drag orbit (azimuth/elevation clamp ±90), wheel zoom (70–10000), WASD pan (x/z), double-click smooth-center on object | [ ] |
| 3.3 | Colour-buffer picking for object_under_cursor; box (rubber-band) selection; shift-add to selection; selection markers (corner brackets) | [ ] (GPU ID buffer instead of colour pass — same trick) |
| 3.4 | Move selected objects on a work plane; Shift/Ctrl switch plane (x-z default, x-y, y-z); drawn plane grid | [ ] |
| 3.5 | Insert mode: new objects follow mouse on plane, click to place, Esc abort | [ ] |
| 3.6 | Connect mode: drag from emitter to receiver, live line overlay; allowed pairs (axon→dendrite/axon-node, samples→inducer/group/LD, LD→LD/samples) | [~] v2 pairs: neuron→neuron synapse, input→group, readout→group |
| 3.7 | Floor grid (x-z, step 20, ±1000) | [ ] |
| 3.8 | Detail levels for neurons/axons/synapses (GuiSettings) — v2: LOD + toggle layers | [~] |
| 3.9 | Live visual feedback while sim runs: neuron colour by membrane potential, axon spikes travelling, synapse activity | [ ] (colour by v/spike; spike dots on lines) |
| 3.10 | ≥100k neurons rendered smoothly (new requirement; original could not) | [ ] |

## 4. Editing actions (mainwindow + menus)

| # | Original feature | v2 |
|---|------------------|----|
| 4.1 | Insert: Single Neuron, Axon Node, Dendrite Node, Synapse, Current Inducer, Samples, Linear Discriminator, Group | [~] neuron, input, samples/stream, readout, group; node-level insert `[-]` |
| 4.2 | Cut / Copy / Paste (serialize to clipboard, insert-moved) | [ ] |
| 4.3 | Delete selected (with cascade: group members, incoming axons/synapses, inducers) | [ ] |
| 4.4 | Connect (from §3.6) | [ ] |
| 4.5 | Selection menu: Add to Group; right-click per-object user actions | [ ] |
| 4.6 | File: New, Load (.neb/.js), Save, Save as, Export/Import XML, Recent files, Close, Quit | [~] .ne2 + .rhai scripts + recent files |
| 4.7 | Edit menu enabled/disable by selection context (axon node ⇒ axon node/synapse, etc.) | [~] |

## 5. Object user actions (right-click menus)

| # | Original feature | v2 |
|---|------------------|----|
| 5.1 | Group: Create MLP… (layers × units, LD + LDConnections) | [-] MLP out of scope |
| 5.2 | Group: Create LSM Column… (width × height × distance ⇒ 2D grid of neurons) | [ ] (+ 3D grid) |
| 5.3 | Group: Create connections… (scaling, lambda ⇒ p=scaling·e^-(d/λ)²) | [ ] |
| 5.4 | Group: Set synapse weights… (gaussian mean/var) | [ ] |
| 5.5 | Group: Do Backpropagation | [-] out of scope |
| 5.6 | Samples: Save/Load/PHOTOSS load/Plot/Clear | [~] save/load CSV-ish + plot; PHOTOSS `[-]` |
| 5.7 | LinearDiscriminator: Update, Create spiking dendrite node | [-] LD out of scope |
| 5.8 | ReadOut: show weights visualization | [ ] |

## 6. Property browser + dialogs

| # | Original feature | v2 |
|---|------------------|----|
| 6.1 | Property browser dock: edit QObject properties of selected objects (name, position, user_movable, weights, params…) | [ ] egui inspector panel |
| 6.2 | Izhikevich property widget (a,b,c,d; presets?) | [ ] |
| 6.3 | Simulation settings widget: simulation_step, real_step, time display | [ ] |
| 6.4 | Settings dialog (GUI settings, detail levels) | [~] view/perf options |
| 6.5 | Log window | [ ] |
| 6.6 | About dialog | [ ] |

## 7. Plots / visualization (Visualizer/ + widgets)

| # | Original feature | v2 |
|---|------------------|----|
| 7.1 | Neuron membrane potential plot (live v(t) trace) | [ ] |
| 7.2 | Izhikevich system-state plot (v–u phase portrait) | [~] v/u shown in inspector; phase plot optional |
| 7.3 | Samples plot via Visualizer2D (time series window) | [ ] egui_plot |
| 7.4 | ReadOut weights visualization | [ ] |
| 7.5 | MLP visualization | [-] out of scope |
| 7.6 | Spike raster / network activity view | [ ] (new, needed for large nets) |

## 8. Scripting

| # | Original feature | v2 |
|---|------------------|----|
| 8.1 | Embedded script engine (QtScript→**Rhai**), same API in GUI console and headless runner | [ ] |
| 8.2 | Script window: multi-line editor (QScintilla w/ JS highlighting), Run, output pane, Clear, load/save .js, script list from file, history up/down on command line | [ ] egui editor + console + history + load/save .rhai |
| 8.3 | Constructors: Neuron, Group, Point, Samples, CurrentInducer, ReadOut, MLP, MNISTData, PhotossSignalImporter | [~] neuron/group/point/samples/input/readout; MLP/MNIST/PHOTOSS `[-]` |
| 8.4 | Globals: print, show, simulation(), load(file), write_to_file(file, …), Controller, RandomGenerator | [~] print/sim/load/write_to_file/rng; show→panels `[-]` |
| 8.5 | Script debugger (QScriptEngineDebugger) | [-] Rhai has none |
| 8.6 | Script runs in its own thread (ScriptRunnerThread) so GUI stays live | [ ] |

## 9. NeuroRunner (headless)

| # | Original feature | v2 |
|---|------------------|----|
| 9.1 | `NeuroRunner` binary: eval each argv as script, or interactive console (REPL) | [ ] `sim` binary: `--script x.rhai`, `--eval`, REPL |
| 9.2 | Loads .neb in scripts (Controller.load_simulation) | [ ] load .ne2 |
| 9.3 | Cluster-job files (.pjob) | [-] superseded by plain CLI |

## 10. v2-only (not in original)

| # | Feature | v2 |
|---|---------|----|
| 10.1 | CUDA GPU backend (cudarc/NVRTC kernels) + CPU fallback | [ ] |
| 10.2 | Live spike/trace streaming from running GPU sim to editor, non-blocking | [ ] |
| 10.3 | Headless screenshot mode (`--screenshot`) | [ ] |
| 10.4 | Seeded, reproducible runs (counter-based RNG) | [ ] |
| 10.5 | Ridge/closed-form readout baseline | [ ] |
| 10.6 | Planted-hidden-signal LSM benchmark + controls | [ ] |
