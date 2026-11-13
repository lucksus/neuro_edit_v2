//! Izhikevich (2003) — the 2011 original's neuron model.
//!
//! v' = 0.04v² + 5v + 140 − u + I, u' = a(bv − u). A step ending at
//! v ≥ 30 spikes; the reset (v = c, u += d) happens on the NEXT step's
//! entry, exactly the original's order. Param row [a, b, c, d], state
//! row [v, u].

use net_format::NeuronKind;

use super::{NeuronModel, ParamDef, StateDef};

pub struct Izhikevich;

/// Regular-spiking (cortical) params — the original's Neuron defaults.
pub const RS: [f32; 4] = [0.02, 0.2, -65.0, 8.0];
/// Fast-spiking params — the stock inhibitory model.
pub const FS: [f32; 4] = [0.1, 0.2, -65.0, 2.0];

static PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "a",
        unit: "1/ms",
        default: RS[0],
        min: 0.0,
        max: 0.2,
        doc: "recovery time scale",
    },
    ParamDef {
        name: "b",
        unit: "",
        default: RS[1],
        min: -1.0,
        max: 1.0,
        doc: "recovery sensitivity to subthreshold v",
    },
    ParamDef {
        name: "c",
        unit: "mV",
        default: RS[2],
        min: -80.0,
        max: 0.0,
        doc: "post-spike reset potential",
    },
    ParamDef {
        name: "d",
        unit: "",
        default: RS[3],
        min: 0.0,
        max: 20.0,
        doc: "post-spike recovery bump",
    },
];

static STATE: &[StateDef] = &[
    StateDef {
        name: "v",
        unit: "mV",
        plot_range: (-90.0, 40.0),
    },
    StateDef {
        name: "u",
        unit: "",
        plot_range: (-20.0, 20.0),
    },
];

static PRESETS: &[(&str, &[f32])] = &[("regular spiking", &RS), ("fast spiking", &FS)];

impl NeuronModel for Izhikevich {
    fn id(&self) -> &'static str {
        "izh"
    }
    fn label(&self) -> &'static str {
        "Izhikevich"
    }
    fn params(&self) -> &'static [ParamDef] {
        PARAMS
    }
    fn state(&self) -> &'static [StateDef] {
        STATE
    }
    fn presets(&self) -> &'static [(&'static str, &'static [f32])] {
        PRESETS
    }
    fn default_for(&self, kind: NeuronKind) -> &'static [f32] {
        match kind {
            NeuronKind::Inhibitory => &FS,
            _ => &RS,
        }
    }

    fn init(&self, p: &[f32], x: &mut [f32]) {
        x[0] = -65.0;
        x[1] = p[1] * x[0];
    }

    /// Identical order to the 2011 original and the CUDA text below:
    /// reset for a previous crossing, integrate v, integrate u (with the
    /// NEW v), then the spike check.
    #[inline(always)]
    fn step(&self, p: &[f32], x: &mut [f32], input: f32, dt: f32) -> bool {
        if x[0] >= 30.0 {
            x[0] = p[2];
            x[1] += p[3];
        }
        x[0] += dt * (0.04 * x[0] * x[0] + 5.0 * x[0] + 140.0 - x[1] + input);
        x[1] += dt * p[0] * (p[1] * x[0] - x[1]);
        x[0] >= 30.0
    }

    fn cuda_step(&self) -> &'static str {
        "if(x[0] >= 30.f){ x[0] = p[2]; x[1] += p[3]; }\n\
         x[0] += dt * (0.04f*x[0]*x[0] + 5.f*x[0] + 140.f - x[1] + I);\n\
         x[1] += dt * p[0] * (p[1]*x[0] - x[1]);\n\
         spiked = x[0] >= 30.f;"
    }
}
