//! Loadable examples (Stage 13 H): Help > Examples and the console
//! window's "Examples" menu load one of these into the script editor; Run
//! sends it to the worker. `watch` lists neuron ids to auto-watch after
//! a Run so first_spikes shows traces without further setup.
//! `purpose` is the one-line summary shown in the menu; `groups` pins
//! editor group modes on load (Stage 24: the big liquids present their
//! reservoir as a Stage-23 "block" glyph, not thousands of smeared dots).

pub struct Example {
    /// Menu label.
    pub name: &'static str,
    /// One-line purpose shown next to the name in the Examples menu.
    pub purpose: &'static str,
    /// Script-editor name / saved-file name.
    pub file: &'static str,
    pub src: &'static str,
    /// Neuron ids to auto-watch on Run ([] = leave the watch alone).
    pub watch: &'static [u32],
    /// (group name, mode) pairs applied on load: "whole" | "inside" |
    /// "block" — same words as the inspector's group-mode selector.
    pub groups: &'static [(&'static str, &'static str)],
}

pub const EXAMPLES: &[Example] = &[
    Example {
        name: "First spikes",
        purpose: "five-neuron driven chain — v(t) and the raster light up",
        file: "first_spikes.rhai",
        src: include_str!("../../examples/first_spikes.rhai"),
        watch: &[0, 1, 2, 3, 4],
        groups: &[],
    },
    Example {
        name: "Grid wave",
        purpose: "20x20 sheet, left-edge kick — a wave rolls across",
        file: "grid_wave.rhai",
        src: include_str!("../../examples/grid_wave.rhai"),
        watch: &[],
        groups: &[],
    },
    Example {
        name: "Reservoir + MLP readout (small LSM)",
        purpose: "216-cell reservoir, MLP readout detects a hidden correlation",
        file: "lsm_readout.rhai",
        src: include_str!("../../examples/lsm_readout.rhai"),
        watch: &[],
        groups: &[],
    },
    Example {
        name: "Big LSM + MLP",
        purpose: "4096-neuron liquid (block group) vs hidden correlation, full baselines",
        file: "lsm_big.rhai",
        src: include_str!("../../examples/lsm_big.rhai"),
        watch: &[],
        groups: &[("res", "block")],
    },
    Example {
        name: "Big LSM — signal off (control)",
        purpose: "same big liquid with no planted signal — everything must sit at chance",
        file: "lsm_big_off.rhai",
        src: include_str!("../../examples/lsm_big_off.rhai"),
        watch: &[],
        groups: &[("res", "block")],
    },
    Example {
        name: "AdEx liquid + MLP",
        purpose: "hidden correlation on an adapting AdEx reservoir (1728 cells)",
        file: "lsm_adex.rhai",
        src: include_str!("../../examples/lsm_adex.rhai"),
        watch: &[],
        groups: &[("res", "block")],
    },
    Example {
        name: "Hidden-correlation LSM (honest eval)",
        purpose: "multi-seed study with permutation null — reports when baselines win",
        file: "lsm_hidden.rhai",
        src: include_str!("../../examples/lsm_hidden.rhai"),
        watch: &[],
        groups: &[],
    },
    Example {
        name: "Event detector",
        purpose: "planted corr events in Poisson channels; job-mode progress",
        file: "detector.rhai",
        src: include_str!("../../examples/detector.rhai"),
        watch: &[],
        groups: &[],
    },
    Example {
        name: "Memory window",
        purpose: "how long a liquid keeps an event readable — tau sweep",
        file: "memory_window.rhai",
        src: include_str!("../../examples/memory_window.rhai"),
        watch: &[],
        groups: &[],
    },
];
