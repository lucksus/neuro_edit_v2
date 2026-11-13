//! Adaptive exponential integrate-and-fire — Brette & Gerstner (2005).
//!
//! C·v' = −g_L(v−E_L) + g_L·Δ_T·exp((v−V_T)/Δ_T) − w + I,
//! τ_w·w' = a(v−E_L) − w. A step ending at v ≥ V_peak spikes; the reset
//! (v = V_r, w += b) happens on the NEXT step's entry, exactly the
//! izh/original order. The exponential argument is clamped at 10 —
//! past V_T + 10·Δ_T the runaway is already irreversible and the
//! spike is recorded on the same step anyway.
//!
//! Param row [C, gL, EL, VT, dT, a, tau_w, b, Vr, Vpeak], state row
//! [v, w]. Currents in pA, conductances in nS, capacitance in pF —
//! pA/pF = mV/ms, so `I` plugs straight in. Params are the tonic-spiking
//! set of Naud et al. (2008), table "tonic".

use net_format::NeuronKind;

use super::{NeuronModel, ParamDef, StateDef};

pub struct AdEx;

/// Tonic-spiking params (Naud et al. 2008): adapting cortical cell.
pub const TONIC: [f32; 10] = [
    281.0, 30.0, -70.6, -50.4, 2.0, 4.0, 144.0, 80.5, -70.6, 20.0,
];
/// Non-adapting variant — the stock inhibitory AdEx.
pub const NONADAP: [f32; 10] = [281.0, 30.0, -70.6, -50.4, 2.0, 0.0, 144.0, 0.0, -70.6, 20.0];

static PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "C",
        unit: "pF",
        default: TONIC[0],
        min: 50.0,
        max: 1000.0,
        doc: "membrane capacitance",
    },
    ParamDef {
        name: "gL",
        unit: "nS",
        default: TONIC[1],
        min: 1.0,
        max: 200.0,
        doc: "leak conductance",
    },
    ParamDef {
        name: "EL",
        unit: "mV",
        default: TONIC[2],
        min: -90.0,
        max: -40.0,
        doc: "leak reversal (rest) potential",
    },
    ParamDef {
        name: "VT",
        unit: "mV",
        default: TONIC[3],
        min: -70.0,
        max: -30.0,
        doc: "exponential threshold potential",
    },
    ParamDef {
        name: "dT",
        unit: "mV",
        default: TONIC[4],
        min: 0.1,
        max: 10.0,
        doc: "exponential slope factor",
    },
    ParamDef {
        name: "a",
        unit: "nS",
        default: TONIC[5],
        min: 0.0,
        max: 100.0,
        doc: "subthreshold adaptation conductance",
    },
    ParamDef {
        name: "tau_w",
        unit: "ms",
        default: TONIC[6],
        min: 1.0,
        max: 1000.0,
        doc: "adaptation time constant",
    },
    ParamDef {
        name: "b",
        unit: "pA",
        default: TONIC[7],
        min: 0.0,
        max: 500.0,
        doc: "spike-triggered adaptation bump",
    },
    ParamDef {
        name: "Vr",
        unit: "mV",
        default: TONIC[8],
        min: -90.0,
        max: -40.0,
        doc: "post-spike reset potential",
    },
    ParamDef {
        name: "Vpeak",
        unit: "mV",
        default: TONIC[9],
        min: -20.0,
        max: 60.0,
        doc: "spike cutoff (upswing detection)",
    },
];

static STATE: &[StateDef] = &[
    StateDef {
        name: "v",
        unit: "mV",
        plot_range: (-90.0, 40.0),
    },
    StateDef {
        name: "w",
        unit: "pA",
        plot_range: (-50.0, 500.0),
    },
];

static PRESETS: &[(&str, &[f32])] = &[("tonic spiking", &TONIC), ("non-adapting", &NONADAP)];

impl NeuronModel for AdEx {
    fn id(&self) -> &'static str {
        "adex"
    }
    fn label(&self) -> &'static str {
        "AdEx (adaptive exponential IF)"
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
            NeuronKind::Inhibitory => &NONADAP,
            _ => &TONIC,
        }
    }

    fn init(&self, p: &[f32], x: &mut [f32]) {
        x[0] = p[2];
        x[1] = 0.0;
    }

    /// Same order as the izh step and the CUDA text below: reset for a
    /// previous crossing, integrate v (clamped exponential), integrate
    /// w with the NEW v, then the spike check.
    #[inline(always)]
    fn step(&self, p: &[f32], x: &mut [f32], input: f32, dt: f32) -> bool {
        if x[0] >= p[9] {
            x[0] = p[8];
            x[1] += p[7];
        }
        let ex = ((x[0] - p[3]) / p[4]).min(10.0).exp();
        x[0] += dt * (-p[1] * (x[0] - p[2]) + p[1] * p[4] * ex - x[1] + input) / p[0];
        x[1] += dt * (p[5] * (x[0] - p[2]) - x[1]) / p[6];
        x[0] >= p[9]
    }

    fn cuda_step(&self) -> &'static str {
        "if(x[0] >= p[9]){ x[0] = p[8]; x[1] += p[7]; }\n\
         float ex = expf(fminf((x[0] - p[3]) / p[4], 10.f));\n\
         x[0] += dt * (-p[1]*(x[0] - p[2]) + p[1]*p[4]*ex - x[1] + I) / p[0];\n\
         x[1] += dt * (p[5]*(x[0] - p[2]) - x[1]) / p[6];\n\
         spiked = x[0] >= p[9];"
    }
}
