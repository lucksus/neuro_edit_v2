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
    pub readouts: Vec<(String, Vec<u32>)>,
    pub dt_ms: f32,
}

#[derive(Clone)]
pub struct InputSnap {
    pub name: String,
    pub pos: [f32; 3],
    pub targets: Vec<u32>,
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
    /// Measured wall-time speed (steps/s) while running.
    pub steps_per_sec: f32,
}

#[allow(dead_code)]
pub struct Shared {
    pub geo: Option<Arc<GeoSnap>>,
    pub snap: Snap,
    /// Worker is in continuous run mode.
    pub running: bool,
    /// A `sim` instance exists on the worker.
    pub sim_built: bool,
    /// A blocking eval/run is in flight (UI shows spinner, input muted).
    pub busy: bool,
    /// Console output lines (print + results + errors + log).
    pub console: VecDeque<String>,
    /// Worker noticed the backend failed and fell back.
    pub notice: Option<String>,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            geo: None,
            snap: Snap::default(),
            running: false,
            sim_built: false,
            busy: false,
            console: VecDeque::with_capacity(512),
            notice: None,
        }
    }
}

pub type SharedRef = Arc<RwLock<Shared>>;

impl Shared {
    pub fn log(&mut self, s: impl Into<String>) {
        if self.console.len() >= 500 {
            self.console.pop_front();
        }
        self.console.push_back(s.into());
    }
}
