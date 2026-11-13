//! Writes the checked-in v1/v2/v3 `.ne2` fixtures under `tests/fixtures/`.
//! The layouts mirror what the old code paths at format-version 1/2/3
//! actually wrote — same field order, same encodings — only built from
//! the current crate's serializable pieces plus local copies of the old
//! row shapes (which is also what the migration loaders decode).
//!
//! `cargo run -p net-format --example gen_format_fixtures`

use net_format::{Group, Input, Meta, NeuronKind, NeuronModel, Readout, Samples, Synapse};
use serde::Serialize;

const NEURONS: &[(&str, [f32; 3], NeuronModel, NeuronKind)] = &[
    (
        "n0",
        [0.0, 0.0, 0.0],
        NeuronModel::Izhikevich {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
        },
        NeuronKind::Excitatory,
    ),
    (
        "n1",
        [60.0, 0.0, 0.0],
        NeuronModel::Izhikevich {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
        },
        NeuronKind::Excitatory,
    ),
    (
        "n2",
        [0.0, 60.0, 0.0],
        NeuronModel::Izhikevich {
            a: 0.1,
            b: 0.2,
            c: -65.0,
            d: 2.0,
        },
        NeuronKind::Inhibitory,
    ),
    // custom row — lifts to an izh#xxxx type on load
    (
        "n3",
        [60.0, 60.0, 0.0],
        NeuronModel::Izhikevich {
            a: 0.03,
            b: 0.25,
            c: -60.0,
            d: 4.0,
        },
        NeuronKind::Excitatory,
    ),
];

/// Old neuron row: model + kind inline (v1..v3).
#[derive(Serialize)]
struct NeuronV3 {
    name: String,
    pos: [f32; 3],
    model: NeuronModel,
    kind: NeuronKind,
}

/// Old synapse row: no `tau_ms` (v1/v2).
#[derive(Serialize)]
struct SynapseV2 {
    src: u32,
    tgt: u32,
    w: f32,
    delay_ms: f32,
}

fn neurons_v3() -> Vec<NeuronV3> {
    NEURONS
        .iter()
        .map(|(name, pos, model, kind)| NeuronV3 {
            name: name.to_string(),
            pos: *pos,
            model: model.clone(),
            kind: *kind,
        })
        .collect()
}

fn synapses() -> Vec<Synapse> {
    vec![
        Synapse {
            src: 0,
            tgt: 1,
            w: 1.5,
            delay_ms: 0.6,
            tau_ms: 7.5,
        },
        Synapse {
            src: 2,
            tgt: 0,
            w: 2.0,
            delay_ms: 0.6,
            tau_ms: 0.0,
        },
    ]
}

fn groups() -> Vec<Group> {
    vec![Group {
        name: "g".into(),
        members: vec![0, 1, 2],
        grid: None,
    }]
}

fn inputs() -> Vec<Input> {
    vec![Input {
        name: "i".into(),
        targets: vec![(0, 1.0)],
        pos: [0.0; 3],
    }]
}

fn tail(w: &mut Vec<u8>, include_group_grid: bool) {
    if include_group_grid {
        bincode::serialize_into(&mut *w, &groups()).unwrap();
    }
    bincode::serialize_into(&mut *w, &inputs()).unwrap();
    bincode::serialize_into(&mut *w, &Vec::<Samples>::new()).unwrap();
    bincode::serialize_into(&mut *w, &Vec::<Readout>::new()).unwrap();
    bincode::serialize_into(&mut *w, &vec![("s".to_string(), "print(1)".to_string())]).unwrap();
    bincode::serialize_into(&mut *w, &Meta::default()).unwrap();
}

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    std::fs::create_dir_all(&dir).unwrap();

    // v1: neurons, synapses(v2 row), groups as (name, members), tail.
    let mut w = Vec::new();
    w.extend_from_slice(b"NE2\0");
    w.extend_from_slice(&1u32.to_le_bytes());
    bincode::serialize_into(&mut w, &neurons_v3()).unwrap();
    let syns_v2: Vec<SynapseV2> = synapses()
        .iter()
        .map(|s| SynapseV2 {
            src: s.src,
            tgt: s.tgt,
            w: s.w,
            delay_ms: s.delay_ms,
        })
        .collect();
    bincode::serialize_into(&mut w, &syns_v2).unwrap();
    let g1: Vec<(String, Vec<u32>)> = groups()
        .iter()
        .map(|g| (g.name.clone(), g.members.clone()))
        .collect();
    bincode::serialize_into(&mut w, &g1).unwrap();
    tail(&mut w, false);
    std::fs::write(dir.join("v1.ne2"), &w).unwrap();

    // v2: same but groups carry the (possibly empty) grid field.
    let mut w = Vec::new();
    w.extend_from_slice(b"NE2\0");
    w.extend_from_slice(&2u32.to_le_bytes());
    bincode::serialize_into(&mut w, &neurons_v3()).unwrap();
    bincode::serialize_into(&mut w, &syns_v2).unwrap();
    tail(&mut w, true);
    std::fs::write(dir.join("v2.ne2"), &w).unwrap();

    // v3: full synapse row (with tau_ms), no types table yet.
    let mut w = Vec::new();
    w.extend_from_slice(b"NE2\0");
    w.extend_from_slice(&3u32.to_le_bytes());
    bincode::serialize_into(&mut w, &neurons_v3()).unwrap();
    bincode::serialize_into(&mut w, &synapses()).unwrap();
    tail(&mut w, true);
    std::fs::write(dir.join("v3.ne2"), &w).unwrap();

    println!("wrote v1/v2/v3 fixtures to {}", dir.display());
}
