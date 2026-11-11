# Feature parity: neuro_edit (C++/Qt4, 2011–2012) → neuro_edit v2 (Rust)

This is the parity checklist. Every user-facing feature of the original editor
(`NeuroEdit/`) and headless runner (`NeuroRunner/`), plus the model behaviour of
`NeuronModels/` that is visible through them.

Legend: `[ ]` open · `[x]` done · `[~]` partial (note) · `[-]` deliberately out of scope (note)

Checked against the C++ source at `~/code/neuro_edit-ref` (commit 8c4e7ce), 2026-11-11.

## 1. Network model objects (NeuronModels/)

| # | Original feature | v2 |
|---|------------------|----|
| 1.1 | Izhikevich point neuron (a,b,c,d; default RS 0.02/0.2/-65/8), forward Euler, v>=30 spike + reset | [x] |
| 1.2 | Per-neuron position (3D, `Point`) | [x] |
| 1.3 | Exponentially decaying synaptic current: spike adds `w` to a trace decaying at `time_constant/100` per ms (default tc=50 ⇒ τ≈2 ms); dendritic current = w·trace | [x] |
| 1.4 | Axon conduction delay = 3D distance / speed (default 5 mm/ms); per-axon speed | [x] speed is global, not per-axon |
| 1.5 | Axon tree of AxonNodes (branching, fan-out) — v2 note: trees collapse to one delay per synapse pair; explicit node editing is `[-]` unless drawn | [~] |
| 1.6 | Dendrite tree of DendriticNodes (no delay, sums current to root) — v2: root only, see 1.5 | [~] |
| 1.7 | Synapse object between axon and dendrite node, weight + time_constant properties | [~] weight + delay per synapse; τ is a global config |
| 1.8 | CurrentInducer: injects a driven current into a dendritic node; `active` flag | [x] `Input` objects + `set_external`/`set_input` |
| 1.9 | Samples: named (t,value) series; linear interpolation; drives CurrentInducers / discriminator outputs; write_value during sim; load/save `.nsamples`; clear; constant-value mode | [~] series + interpolation + play_input; no .nsamples file I/O yet |
| 1.10 | Group: named object set; user actions (see §5); add/remove objects; inputs | [x] |
| 1.11 | LSMReadOutNeuron: sigmoid unit over exp-filtered spike traces (τ=2 ms default, same rate law as synapses), bias weight, `learn(target, rate)` delta rule, `init_weights_random_uniform`, `connect_with_group`, output written to attached Samples | [x] DeltaReadout in Rust (proper delta rule + original quirk mode) |
| 1.12 | LinearDiscriminator + LDConnection (weighted sum of LD outputs → sigmoid), used for MLP | [-] MLP/backprop out of scope for the LSM use case |
| 1.13 | Group::do_backprop / MLP (`MLP/`, `create_mlp`) | [-] out of scope (LSM only) |
| 1.14 | PhotossSignalImporter (text file → bit_at_time) | [-] replaced by generic Samples/CSV input |
| 1.15 | MNISTData | [-] out of scope |
| 1.16 | RandomGenerator script API (uniform, gauss) | [x] |

## 2. Simulation control (Simulation/Controller/Network)

| # | Original feature | v2 |
|---|------------------|----|
| 2.1 | simulation_step (dt, default 0.1 ms) editable | [x] |
| 2.2 | real_step (wall-clock ms per sim step) ⇒ playback speed throttle; not_matching_speed signal | [x] speed slider in editor; steps_per_sec reported |
| 2.3 | Continuous run in a thread; start / pause (request_stop) / resume | [x] worker thread, non-blocking |
| 2.4 | run(time_ms), run_one_step(step) from scripts | [x] sim.run(ms) / sim.run_steps(n) |
| 2.5 | reset() — network state + time to 0 | [x] |
| 2.6 | sim.time_ms | [x] sim.time() |
| 2.7 | Simulation must be saved before first GUI run (filename guard) | [-] v2 sim does not require a file |
| 2.8 | Controller: new / load(.neb) / save(.neb) / export XML(.nex) / import XML / close | [~] .ne2 binary + JSON export; old .neb import is `[-]` (nice-to-have only) |
| 2.9 | Scripts stored inside the file (simulation scripts + network scripts) | [x] .ne2 `scripts` field |

## 3. 3D view (glscene.cpp / Drawable*)

| # | Original feature | v2 |
|---|------------------|----|
| 3.1 | 3D rendering of all objects: neuron (box/frustum), axon node, dendrite node, axon (line + travelling spike dots), synapse (cone), current inducer (cone), samples (cube), linear discriminator (cube), LD connection (line), group (handle/box) | [~] v2 draws neurons, synapse lines w/ spike dots, inputs, groups, readouts; node-level trees `[-]` |
| 3.2 | Camera: right-drag orbit (azimuth/elevation clamp ±90), wheel zoom (70–10000), WASD pan (x/z), double-click smooth-center on object | [~] implemented (orbit/zoom/pan/frame); interactive feel unverified headless |
| 3.3 | Colour-buffer picking for object_under_cursor; box (rubber-band) selection; shift-add to selection; selection markers (corner brackets) | [~] GPU ID pick + box select implemented; unverified headless |
| 3.4 | Move selected objects on a work plane; Shift/Ctrl switch plane (x-z default, x-y, y-z); drawn plane grid | [~] implemented |
| 3.5 | Insert mode: new objects follow mouse on plane, click to place, Esc abort | [~] implemented |
| 3.6 | Connect mode: drag from emitter to receiver, live line overlay; allowed pairs (axon→dendrite/axon-node, samples→inducer/group/LD, LD→LD/samples) | [~] v2 pairs: neuron→neuron synapse, input→group, readout→group |
| 3.7 | Floor grid (x-z, step 20, ±1000) | [x] visible in screenshots |
| 3.8 | Detail levels for neurons/axons/synapses (GuiSettings) — v2: LOD + toggle layers | [~] |
| 3.9 | Live visual feedback while sim runs: neuron colour by membrane potential, axon spikes travelling, synapse activity | [x] activity colouring + travelling spike particles; synapse glow |
| 3.10 | ≥100k neurons rendered smoothly (new requirement; original could not) | [x] 97k neurons + 1.1M syn = 4.4 ms/frame measured |

## 4. Editing actions (mainwindow + menus)

| # | Original feature | v2 |
|---|------------------|----|
| 4.1 | Insert: Single Neuron, Axon Node, Dendrite Node, Synapse, Current Inducer, Samples, Linear Discriminator, Group | [~] neuron, input, samples/stream, readout, group; node-level insert `[-]` |
| 4.2 | Cut / Copy / Paste (serialize to clipboard, insert-moved) | [ ] |
| 4.3 | Delete selected (with cascade: group members, incoming axons/synapses, inducers) | [~] delete_selection drops neurons+their synapses; no cascade to groups/inputs yet |
| 4.4 | Connect (from §3.6) | [x] |
| 4.5 | Selection menu: Add to Group; right-click per-object user actions | [~] Add-to-Group dialog; no right-click context menus yet |
| 4.6 | File: New, Load (.neb/.js), Save, Save as, Export/Import XML, Recent files, Close, Quit | [~] new/load/save-as .ne2; no recent-files list |
| 4.7 | Edit menu enabled/disable by selection context (axon node ⇒ axon node/synapse, etc.) | [~] menu always enabled; actions no-op without selection |

## 5. Object user actions (right-click menus)

| # | Original feature | v2 |
|---|------------------|----|
| 5.1 | Group: Create MLP… (layers × units, LD + LDConnections) | [-] MLP out of scope |
| 5.2 | Group: Create LSM Column… (width × height × distance ⇒ 2D grid of neurons) | [x] 2D + 3D grid dialogs |
| 5.3 | Group: Create connections… (scaling, lambda ⇒ p=scaling·e^-(d/λ)²) | [x] |
| 5.4 | Group: Set synapse weights… (gaussian mean/var) | [x] |
| 5.5 | Group: Do Backpropagation | [-] out of scope |
| 5.6 | Samples: Save/Load/PHOTOSS load/Plot/Clear | [~] save/load CSV-ish + plot; PHOTOSS `[-]` |
| 5.7 | LinearDiscriminator: Update, Create spiking dendrite node | [-] LD out of scope |
| 5.8 | ReadOut: show weights visualization | [ ] |

## 6. Property browser + dialogs

| # | Original feature | v2 |
|---|------------------|----|
| 6.1 | Property browser dock: edit QObject properties of selected objects (name, position, user_movable, weights, params…) | [x] inspector: name/pos/kind/params |
| 6.2 | Izhikevich property widget (a,b,c,d; presets?) | [x] params editable in inspector |
| 6.3 | Simulation settings widget: simulation_step, real_step, time display | [x] toolbar: dt, speed, noise, time |
| 6.4 | Settings dialog (GUI settings, detail levels) | [~] view/perf options |
| 6.5 | Log window | [x] console bottom panel shows log lines |
| 6.6 | About dialog | [ ] |

## 7. Plots / visualization (Visualizer/ + widgets)

| # | Original feature | v2 |
|---|------------------|----|
| 7.1 | Neuron membrane potential plot (live v(t) trace) | [~] implemented (watch selected neurons); unverified headless |
| 7.2 | Izhikevich system-state plot (v–u phase portrait) | [~] v/u shown in inspector; phase plot optional |
| 7.3 | Samples plot via Visualizer2D (time series window) | [ ] |
| 7.4 | ReadOut weights visualization | [ ] |
| 7.5 | MLP visualization | [-] out of scope |
| 7.6 | Spike raster / network activity view | [~] implemented; unverified headless |

## 8. Scripting

| # | Original feature | v2 |
|---|------------------|----|
| 8.1 | Embedded script engine (QtScript→**Rhai**), same API in GUI console and headless runner | [x] |
| 8.2 | Script window: multi-line editor (QScintilla w/ JS highlighting), Run, output pane, Clear, load/save .js, script list from file, history up/down on command line | [~] console + history + eval; no multiline editor pane / .rhai file picker yet |
| 8.3 | Constructors: Neuron, Group, Point, Samples, CurrentInducer, ReadOut, MLP, MNISTData, PhotossSignalImporter | [~] neuron/group/point/samples/input/readout; MLP/MNIST/PHOTOSS `[-]` |
| 8.4 | Globals: print, show, simulation(), load(file), write_to_file(file, …), Controller, RandomGenerator | [~] print/sim/load/write_to_file/rng; show→panels `[-]` |
| 8.5 | Script debugger (QScriptEngineDebugger) | [-] Rhai has none |
| 8.6 | Script runs in its own thread (ScriptRunnerThread) so GUI stays live | [x] worker thread |

## 9. NeuroRunner (headless)

| # | Original feature | v2 |
|---|------------------|----|
| 9.1 | `NeuroRunner` binary: eval each argv as script, or interactive console (REPL) | [~] `sim --script`/`--net`; no REPL yet |
| 9.2 | Loads .neb in scripts (Controller.load_simulation) | [x] net.load/save .ne2 |
| 9.3 | Cluster-job files (.pjob) | [-] superseded by plain CLI |

## 10. v2-only (not in original)

| # | Feature | v2 |
|---|---------|----|
| 10.1 | CUDA GPU backend (cudarc/NVRTC kernels) + CPU fallback | [x] verified vs CPU |
| 10.2 | Live spike/trace streaming from running GPU sim to editor, non-blocking | [x] snapshot channel, worker never blocks on UI |
| 10.3 | Headless screenshot mode (`--screenshot`) | [x] verified (docs/screenshots/) |
| 10.4 | Seeded, reproducible runs (counter-based RNG) | [x] |
| 10.5 | Ridge/closed-form readout baseline | [x] |
| 10.6 | Planted-hidden-signal LSM benchmark + controls | [x] acc/auc + controls + sweep in NOTES |
