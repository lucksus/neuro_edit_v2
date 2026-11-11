//! net-format: the neuro_edit v2 network model and `.ne2` file format.
//!
//! Flat arrays, not a pointer graph. Neurons carry position and Izhikevich
//! parameters; synapses are stored as (src, tgt, weight, delay_ms) and can be
//! exported to a source-sorted CSR layout for the simulators.
//!
//! Units: positions are micrometres (same arbitrary unit as the 2011 editor's
//! grids), conduction speed is micrometres per millisecond (the original's
//! default of 5 mm/ms = 5000 um/ms), times are milliseconds.

pub mod builders;
pub mod io;
pub mod rng;

mod network;

pub use network::{
    Group, Input, Meta, Network, Neuron, NeuronKind, NeuronModel, Readout, Samples, SynCsr, Synapse,
};
pub use rng::Rng;

/// Default synaptic decay time constant (ms). The original called its rate
/// `time_constant` with default 50, meaning decay of 50/100 per ms, i.e. tau 2ms.
pub const DEFAULT_SYNAPSE_TAU_MS: f32 = 2.0;

/// Default readout filtered-trace time constant (ms); same rate law.
pub const DEFAULT_READOUT_TAU_MS: f32 = 2.0;

/// Default conduction speed in um/ms (the original's 5 mm/ms).
pub const DEFAULT_CONDUCTION_SPEED: f32 = 5000.0;

/// Default simulation step (ms), as in the original.
pub const DEFAULT_DT_MS: f32 = 0.1;
