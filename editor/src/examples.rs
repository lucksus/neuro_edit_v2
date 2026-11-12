//! Loadable examples (Stage 13 H): Help → Examples and the console
//! window's "Examples ▾" load one of these into the script editor; Run
//! sends it to the worker. `watch` lists neuron ids to auto-watch after
//! a Run so first_spikes shows traces without further setup.

pub struct Example {
    /// Menu label.
    pub name: &'static str,
    /// Script-editor name / saved-file name.
    pub file: &'static str,
    pub src: &'static str,
    /// Neuron ids to auto-watch on Run ([] = leave the watch alone).
    pub watch: &'static [u32],
}

pub const EXAMPLES: &[Example] = &[
    Example {
        name: "First spikes",
        file: "first_spikes.rhai",
        src: include_str!("../../examples/first_spikes.rhai"),
        watch: &[0, 1, 2, 3, 4],
    },
    Example {
        name: "Grid wave",
        file: "grid_wave.rhai",
        src: include_str!("../../examples/grid_wave.rhai"),
        watch: &[],
    },
    Example {
        name: "Reservoir + readout (LSM)",
        file: "lsm_readout.rhai",
        src: include_str!("../../examples/lsm_readout.rhai"),
        watch: &[],
    },
];
