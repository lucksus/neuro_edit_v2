//! Izhikevich (2003) — the 2011 original's neuron model.
//!
//! v' = 0.04v² + 5v + 140 − u + I, u' = a(bv − u). A step ending at
//! v ≥ 30 spikes; the reset (v = c, u += d) happens on the NEXT step's
//! entry, exactly the original's order. Param row [a, b, c, d], state
//! row [v, u].

use net_format::NeuronKind;

use super::{MapPlane, MapSpot, NeuronModel, ParamDef, ParamMap, StateDef};

pub struct Izhikevich;

/// Regular-spiking (cortical) params — the original's Neuron defaults.
pub const RS: [f32; 4] = [0.02, 0.2, -65.0, 8.0];
/// Intrinsically bursting.
pub const IB: [f32; 4] = [0.02, 0.2, -55.0, 4.0];
/// Chattering.
pub const CH: [f32; 4] = [0.02, 0.2, -50.0, 2.0];
/// Fast-spiking params — the stock inhibitory model.
pub const FS: [f32; 4] = [0.1, 0.2, -65.0, 2.0];
/// Low-threshold spiking.
pub const LTS: [f32; 4] = [0.02, 0.25, -65.0, 2.0];
/// Thalamo-cortical (rebound bursts).
pub const TC: [f32; 4] = [0.02, 0.25, -65.0, 0.05];
/// Resonator (subthreshold oscillations).
pub const RZ: [f32; 4] = [0.1, 0.26, -65.0, 2.0];

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

static PRESETS: &[(&str, &[f32])] = &[
    ("regular spiking", &RS),
    ("intrinsically bursting", &IB),
    ("chattering", &CH),
    ("fast spiking", &FS),
    ("low-threshold spiking", &LTS),
    ("thalamo-cortical", &TC),
    ("resonator", &RZ),
];

/// The classic design map: every cortical cell type on the (a,b) and
/// (c,d) planes. The map and the preset list share the const rows —
/// one source of truth. Several types coincide on one plane (RS/IB/CH
/// share a,b; FS/LTS/RZ share c,d); the inspector stacks their labels
/// at the shared point and each stays its own click target.
fn spots() -> Vec<MapSpot> {
    vec![
        MapSpot {
            label: "RS",
            name: "regular spiking",
            params: &RS,
        },
        MapSpot {
            label: "IB",
            name: "intrinsically bursting",
            params: &IB,
        },
        MapSpot {
            label: "CH",
            name: "chattering",
            params: &CH,
        },
        MapSpot {
            label: "FS",
            name: "fast spiking",
            params: &FS,
        },
        MapSpot {
            label: "LTS",
            name: "low-threshold spiking",
            params: &LTS,
        },
        MapSpot {
            label: "TC",
            name: "thalamo-cortical",
            params: &TC,
        },
        MapSpot {
            label: "RZ",
            name: "resonator",
            params: &RZ,
        },
    ]
}

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
    fn param_map(&self) -> Option<ParamMap> {
        Some(ParamMap {
            planes: vec![
                MapPlane {
                    x: 0,
                    y: 1,
                    x_range: (0.0, 0.15),
                    y_range: (0.0, 0.3),
                    spots: spots(),
                },
                MapPlane {
                    x: 2,
                    y: 3,
                    x_range: (-70.0, -45.0),
                    y_range: (-1.0, 10.0),
                    spots: spots(),
                },
            ],
        })
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
