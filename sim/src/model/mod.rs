//! Neuron models: one file per model, both step texts side by side.
//!
//! A model owns a parameter row (`p[k]`) and a state row (`x[k]`); state
//! slot 0 is always the membrane potential. The CPU step is a Rust
//! function, the CUDA step is a C snippet spliced into `neuron_kernel`
//! at build time — the two must compute the same thing, which the
//! cpu/gpu tests check.

use net_format::NeuronKind;

pub mod adex;
pub mod izh;

/// A parameter's descriptor — units/ranges for editors and validation.
#[derive(Clone, Debug)]
pub struct ParamDef {
    pub name: &'static str,
    pub unit: &'static str,
    pub default: f32,
    pub min: f32,
    pub max: f32,
    pub doc: &'static str,
}

/// A state slot's descriptor. Slot 0 is always the membrane potential
/// (what `sim.v()` and the editor's v(t) plot read).
#[derive(Clone, Debug)]
pub struct StateDef {
    pub name: &'static str,
    pub unit: &'static str,
    pub plot_range: (f32, f32),
}

/// One clickable spot on a parameter design map — a named region whose
/// param row is applied on click. The spot's position on its plane is
/// read off `params[x], params[y]`, so region coordinates can never
/// drift from the preset data.
#[derive(Clone, Debug)]
pub struct MapSpot {
    /// Short label drawn on the map ("RS", "IB", …).
    pub label: &'static str,
    /// Full region name for tooltips ("regular spiking").
    pub name: &'static str,
    /// Param row applied on click (full `params()` length).
    pub params: &'static [f32],
}

/// One 2-D plane of a design map (for izh: a–b and c–d).
#[derive(Clone, Debug)]
pub struct MapPlane {
    /// Param-row indices drawn on the x/y axes.
    pub x: usize,
    pub y: usize,
    /// Axis display ranges in parameter units.
    pub x_range: (f32, f32),
    pub y_range: (f32, f32),
    pub spots: Vec<MapSpot>,
}

/// A model's canonical parameter design map — the classic Izhikevich
/// a–b / c–d diagram is the example. Models without one return `None`
/// from `param_map()` and the inspector just skips the widget.
#[derive(Clone, Debug)]
pub struct ParamMap {
    pub planes: Vec<MapPlane>,
}

/// One neuron model: descriptors + CPU step + CUDA step text.
/// `Sync` — the registry hands out `&'static dyn` shared across threads.
pub trait NeuronModel: Sync {
    /// Stable id stored in files — never renamed. ("izh")
    fn id(&self) -> &'static str;
    /// Display name. ("Izhikevich")
    fn label(&self) -> &'static str;
    /// Parameter descriptors; a param row is laid out in `params()` order.
    fn params(&self) -> &'static [ParamDef];
    /// State slots; `state()[0]` is the membrane potential.
    fn state(&self) -> &'static [StateDef];
    /// Named parameter presets ("regular spiking", "fast spiking").
    fn presets(&self) -> &'static [(&'static str, &'static [f32])];
    /// The parameter design map the inspector draws, if the model has
    /// a canonical one. Default: none.
    fn param_map(&self) -> Option<ParamMap> {
        None
    }
    /// Stock params for a neuron kind (for izh, inhibitory = fast spiking).
    fn default_for(&self, kind: NeuronKind) -> &'static [f32];
    /// Initial state for param row `p` (izh: v = −65, u = b·v).
    fn init(&self, p: &[f32], x: &mut [f32]);
    /// One step for one neuron at `dt` ms with input current `input`.
    /// Does its own reset: a crossing seen at step entry resets BEFORE
    /// integrating (the original's order). Returns true when this step
    /// ends spiked.
    fn step(&self, p: &[f32], x: &mut [f32], input: f32, dt: f32) -> bool;
    /// The same step as CUDA C. Sees `p[k]`, `x[k]`, `I`, `dt` in scope;
    /// sets `spiked` (bool) instead of returning it.
    fn cuda_step(&self) -> &'static str;
}

/// Every model the sim can run, in registry order. A file names a model
/// by `id()`; a net's `model[]` column indexes the per-net table
/// `NeuronParams::models` built from this registry.
pub static MODELS: &[&dyn NeuronModel] = &[&izh::Izhikevich, &adex::AdEx];

/// Registry index + model for `id`, if known.
pub fn by_id(id: &str) -> Option<(u8, &'static dyn NeuronModel)> {
    MODELS
        .iter()
        .position(|m| m.id() == id)
        .map(|i| (i as u8, MODELS[i]))
}
