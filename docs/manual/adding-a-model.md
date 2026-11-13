# Adding a model

A neuron model is **one new file under `sim/src/model/` plus one
registry line**. The simulator, the wire protocol, the script API and
the editor all read the model's descriptors — nothing else changes.
This page walks through the commit that added `adex` (adaptive
exponential integrate-and-fire) as the worked example.

## 1. The model file: `sim/src/model/adex.rs`

Implement the `NeuronModel` trait (`sim/src/model/mod.rs`):

- `id()` — the stable string files store (`"adex"`). Never renamed.
- `label()` — display name shown in the editor inspector.
- `params()` — `ParamDef` rows: `name`, `unit`, `default`, `min`,
  `max`, `doc`. The inspector builds a slider per row, validation uses
  the width, and `net.set_param` names params by index.
- `state()` — `StateDef` rows: `name`, `unit`, `plot_range`. Slot 0 is
  always the membrane potential (what `sim.v(i)` and the v(t) plot
  read). One editor plot tab per slot, automatically.
- `presets()` — named param rows for the inspector's preset picker.
- `default_for(kind)` — stock params per `NeuronKind` (adex gives
  inhibitory neurons the non-adapting row).
- `init(p, x)` — initial state for a param row.
- `step(p, x, input, dt)` — the CPU step. Convention, shared with izh:
  a crossing seen at step **entry** resets *before* integrating;
  return `true` when the step ends spiked.
- `cuda_step()` — the same step as CUDA C text. It sees `p[k]`, `x[k]`,
  `I`, `dt` in scope and sets `spiked` instead of returning. NVRTC has
  `expf`/`fminf` for transcendentals.

The CPU function and the C string must compute the same thing — the
mixed-net test checks spike counts and state on both backends.

## 2. The registry line: `sim/src/model/mod.rs`

```rust
pub mod adex;
// ...
pub static MODELS: &[&dyn NeuronModel] = &[&izh::Izhikevich, &adex::AdEx];
```

That is the whole integration. From there:

- `model::by_id("adex")` resolves file type rows and script calls.
- `NeuronParams::from_network` builds a per-net model table —
  `model[i]` indexes it, `p`/`x0` rows are padded to the widest model.
- The CUDA kernel (`kernels.rs` `//@MODEL_STEP@`) gets one
  `if(mi == k){ ... }` case per model in the table — a one-model net
  keeps the un-switched fast path.
- `Hello.models` ships the descriptors to every client; the editor
  inspector and plot tabs are built from them.
- `models()` in Rhai lists it; `net.def_type("t", "adex", [...])` and
  `net.set_type(ids, "t")` use it.

## 3. The tests: `sim/tests/adex.rs`

- `adex_is_registered_with_descriptors` — the registry line and the
  descriptor surface (param/state widths, presets, `init`).
- `adex_mixed_cpu_matches_cuda` (cuda feature) — an 8-neuron net where
  `net.upsert_type`/`net.set_type` assign adex to a subset, driven so
  every adex neuron fires **and resets** (`counts[i] >= 2`); asserts
  CPU and CUDA lifetime spike counts are exactly equal and state slot 0
  matches. This is the Stage 20 lesson: drive the *new* model's
  neurons, or a broken GPU reset passes silently.
- `adex_mixed_net_fires_on_cpu` — the same acceptance on GPU-less
  hosts.

Run: `cargo test -p sim --test adex` (CUDA host runs all three;
`--nocapture` prints per-neuron counts).

## 4. What a `.ne2` file sees

A saved net stores the type's `model` id string and param row; loading
resolves the id through `by_id`, so files written today keep working —
and a file naming an unregistered model fails `from_network` with a
clear error, never a mis-indexed row.
