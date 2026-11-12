//! Shared state between the sim worker thread and the UI.
//!
//! The worker owns the network + simulator + script engine. The UI never
//! touches them; it renders `Shared` (written under a short write lock once
//! per publish chunk) and sends `Cmd`s. The sim is never blocked by the UI:
//! publishing is the worker's choice, and heavy payloads (geometry) are
//! `Arc`-shared snapshots replaced wholesale.

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

/// Immutable geometry snapshot of the whole network.
#[derive(Clone)]
#[allow(dead_code)]
pub struct GeoSnap {
    pub version: u64,
    pub pos: Vec<[f32; 3]>,
    /// 0 = excitatory, 1 = inhibitory.
    pub kind: Vec<u8>,
    pub names: Vec<String>,
    /// (src, tgt) pairs, source order.
    pub syn: Vec<(u32, u32)>,
    /// Per-synapse delay in sim steps (for spike-dot travel time).
    pub syn_delay_steps: Vec<u16>,
    /// First K display synapses per neuron: (tgt, delay_steps) flattened.
    pub disp_edges: Vec<(u32, u16)>,
    /// Start index of neuron i's display edges in `disp_edges`.
    pub disp_ptr: Vec<u32>,
    pub groups: Vec<(String, Vec<u32>)>,
    /// (name, targets) — inputs drawn as markers.
    pub inputs: Vec<InputSnap>,
    /// (name, units, weights) — weights may be empty until a script pushes
    /// them via `net.set_readout_weights`.
    pub readouts: Vec<(String, Vec<u32>, Vec<f32>)>,
    /// (name, times, values) — plotted in the samples tab.
    pub samples: Vec<(String, Vec<f32>, Vec<f32>)>,
    /// (name, code) — script list saved in the file.
    pub scripts: Vec<(String, String)>,
    /// Groups whose members form a regular lattice (`Group::grid`), with
    /// the member list reordered into texel order. The plane view draws
    /// these instead of the member cubes.
    pub grids: Vec<GridSnap>,
    pub dt_ms: f32,
}

#[derive(Clone)]
pub struct InputSnap {
    pub name: String,
    pub pos: [f32; 3],
    pub targets: Vec<u32>,
}

/// A grid group turned renderable: lattice metadata plus the member list
/// re-ordered into texel order (position k = lattice index k).
#[derive(Clone)]
pub struct GridSnap {
    /// Index into `groups`.
    pub group: usize,
    /// Texel counts per lattice dim `[w, h, d]` — `d == 1` is a 2D grid.
    pub dims: [u32; 3],
    /// World axis (0=x, 1=y, 2=z) each lattice dim runs along.
    pub axes: [u8; 3],
    /// World position of lattice texel (0,0,0).
    pub origin: [f32; 3],
    /// World distance between neighbouring texels.
    pub spacing: f32,
    /// Texel-ordered member list (neuron ids), `len == w*h*d`.
    pub members: Vec<u32>,
}

/// Dynamic state published every chunk while the sim runs.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct Snap {
    pub seq: u64,
    pub t_steps: u64,
    pub time_ms: f64,
    pub spikes_total: u64,
    pub rate_hz: f64,
    /// Exp-filtered activity trace per neuron (drives instance colour).
    pub activity: Arc<Vec<f32>>,
    /// Per-neuron spike counts since the previous snap (particles, raster).
    pub counts: Arc<Vec<u32>>,
    /// New `v` samples for watched neurons, per id.
    pub watched_v: Vec<(u32, Arc<Vec<f32>>)>,
    /// Spike-trace value per texel for each `GeoSnap::grids` entry
    /// (same order; `exp` decay, tau ~20 ms — see `simctl::PLANE_TAU_MS`).
    pub grid_traces: Vec<Arc<Vec<f32>>>,
    /// Live constant drive per input (same order as `GeoSnap::inputs`).
    pub input_values: Arc<Vec<f32>>,
    /// Measured wall-time speed (steps/s) while running.
    pub steps_per_sec: f32,
}

#[allow(dead_code)]
pub struct Shared {
    pub geo: Option<Arc<GeoSnap>>,
    /// Net-replacement generation (New / load / `net.clear()`) — the UI
    /// resets selection, watches and plots when this changes.
    pub net_gen: u64,
    pub snap: Snap,
    /// Worker is in continuous run mode.
    pub running: bool,
    /// A `sim` instance exists on the worker.
    pub sim_built: bool,
    /// A blocking eval/run is in flight (UI shows spinner, input muted).
    pub busy: bool,
    /// Console output lines (print + results + errors + log).
    pub console: VecDeque<String>,
    /// Live sim configuration — the toolbar reads these so script calls
    /// (`sim.set_dt`, `sim.set_noise`) show up instead of stale values.
    pub cfg: sim::SimConfig,
    /// Live backend choice on the worker.
    pub backend: sim::BackendKind,
    /// Worker noticed the backend failed and fell back.
    pub notice: Option<String>,
    /// JSON subnet staged by a copy/cut command for the UI to put on the
    /// system clipboard.
    pub clipboard: Option<String>,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            geo: None,
            net_gen: 0,
            snap: Snap::default(),
            running: false,
            sim_built: false,
            busy: false,
            console: VecDeque::with_capacity(512),
            cfg: sim::SimConfig::default(),
            backend: sim::BackendKind::Cpu,
            notice: None,
            clipboard: None,
        }
    }
}

pub type SharedRef = Arc<RwLock<Shared>>;

impl Shared {
    pub fn log(&mut self, s: impl Into<String>) {
        let mut s = s.into();
        // Eval results like `net.neurons_of("g")` can return multi-KB
        // arrays — shorten the line and keep the tail count honest.
        const MAX: usize = 400;
        if s.len() > MAX {
            let extra = s.len() - MAX;
            s.truncate(MAX);
            // don't leave half a UTF-8 char at the cut point
            while !s.is_char_boundary(s.len()) {
                s.pop();
            }
            s.push_str(&format!(" … ({extra} chars truncated)"));
        }
        if self.console.len() >= 500 {
            self.console.pop_front();
        }
        self.console.push_back(s);
    }
}
