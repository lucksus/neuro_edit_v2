//! `.ne2` file format: magic, version, then a bincode-serialized `Network`.
//! Both `sim` and `editor` use only this crate, so the file can never drift
//! between the two programs.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::Network;

const MAGIC: &[u8; 4] = b"NE2\0";
const FORMAT_VERSION: u32 = 3;

/// `Group` as serialized by format v1 — before the `grid` field existed.
#[derive(serde::Deserialize)]
struct GroupV1 {
    name: String,
    members: Vec<u32>,
}

/// `Synapse` as serialized by format v1/v2 — before `tau_ms` existed.
#[derive(serde::Serialize, serde::Deserialize)]
struct SynapseV2 {
    src: u32,
    tgt: u32,
    w: f32,
    delay_ms: f32,
}

/// Lift v1/v2 synapses (no `tau_ms`) into today's `Synapse` — `0.0`
/// keeps them on the merged global-tau path, exactly their old meaning.
fn lift_synapses(ss: Vec<SynapseV2>) -> Vec<crate::Synapse> {
    ss.into_iter()
        .map(|s| crate::Synapse {
            src: s.src,
            tgt: s.tgt,
            w: s.w,
            delay_ms: s.delay_ms,
            tau_ms: 0.0,
        })
        .collect()
}

/// Decode a v1 stream field-by-field (bincode has no defaults for missing
/// trailing struct fields), lifting each v1 `Group` into today's `Group`.
fn load_v1(r: &mut impl Read) -> Result<Network, LoadError> {
    fn de<T: serde::de::DeserializeOwned>(t: &str, r: &mut impl Read) -> Result<T, LoadError> {
        bincode::deserialize_from(r).map_err(|e| LoadError::Decode(format!("{t}: {e}")))
    }
    let net = Network {
        neurons: de("neurons", r)?,
        synapses: {
            let ss: Vec<SynapseV2> = de("synapses", r)?;
            lift_synapses(ss)
        },
        groups: {
            let gs: Vec<GroupV1> = de("groups", r)?;
            gs.into_iter()
                .map(|g| crate::Group {
                    name: g.name,
                    members: g.members,
                    grid: None,
                })
                .collect()
        },
        inputs: de("inputs", r)?,
        samples: de("samples", r)?,
        readouts: de("readouts", r)?,
        scripts: de("scripts", r)?,
        meta: de("meta", r)?,
    };
    Ok(net)
}

/// Decode a v2 stream field-by-field — same as v3 minus `Synapse::tau_ms`
/// (groups already carry `grid` in v2).
fn load_v2(r: &mut impl Read) -> Result<Network, LoadError> {
    fn de<T: serde::de::DeserializeOwned>(t: &str, r: &mut impl Read) -> Result<T, LoadError> {
        bincode::deserialize_from(r).map_err(|e| LoadError::Decode(format!("{t}: {e}")))
    }
    let net = Network {
        neurons: de("neurons", r)?,
        synapses: {
            let ss: Vec<SynapseV2> = de("synapses", r)?;
            lift_synapses(ss)
        },
        groups: de("groups", r)?,
        inputs: de("inputs", r)?,
        samples: de("samples", r)?,
        readouts: de("readouts", r)?,
        scripts: de("scripts", r)?,
        meta: de("meta", r)?,
    };
    Ok(net)
}

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    Decode(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "io error: {e}"),
            LoadError::BadMagic => write!(f, "not a .ne2 file (bad magic)"),
            LoadError::UnsupportedVersion(v) => write!(f, "unsupported .ne2 version {v}"),
            LoadError::Decode(e) => write!(f, "decode error: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        LoadError::Io(e)
    }
}

/// The exact bytes `save` would write — lets callers check a size or a
/// quota before touching the filesystem.
pub fn to_bytes(net: &Network) -> io::Result<Vec<u8>> {
    let mut w = Vec::new();
    w.write_all(MAGIC)?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    let mut w = BufWriter::new(w);
    bincode::serialize_into(&mut w, net).map_err(io::Error::other)?;
    w.flush()?;
    Ok(w.into_inner().map_err(io::Error::other)?)
}

pub fn save(net: &Network, path: impl AsRef<Path>) -> io::Result<()> {
    std::fs::write(path, to_bytes(net)?)
}

pub fn load(path: impl AsRef<Path>) -> Result<Network, LoadError> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(LoadError::BadMagic);
    }
    let mut ver = [0u8; 4];
    r.read_exact(&mut ver)?;
    let version = u32::from_le_bytes(ver);
    let mut net = match version {
        1 => load_v1(&mut r)?,
        2 => load_v2(&mut r)?,
        FORMAT_VERSION => {
            bincode::deserialize_from(&mut r).map_err(|e| LoadError::Decode(e.to_string()))?
        }
        other => return Err(LoadError::UnsupportedVersion(other)),
    };
    net.validate().map_err(LoadError::Decode)?;
    net.invalidate_grids(); // corrupt/stale metadata in a hand-edited file
    Ok(net)
}

/// Debug-friendly JSON export/import (not the primary format).
pub fn save_json(net: &Network, path: impl AsRef<Path>) -> io::Result<()> {
    let s = serde_json::to_string_pretty(net).map_err(io::Error::other)?;
    std::fs::write(path, s)
}

pub fn load_json(path: impl AsRef<Path>) -> Result<Network, LoadError> {
    let s = std::fs::read_to_string(path)?;
    let mut net: Network =
        serde_json::from_str(&s).map_err(|e| LoadError::Decode(e.to_string()))?;
    net.validate().map_err(LoadError::Decode)?;
    net.invalidate_grids();
    Ok(net)
}

/// Serialize to a JSON string (clipboard, network transport).
pub fn to_json(net: &Network) -> io::Result<String> {
    serde_json::to_string(net).map_err(io::Error::other)
}

/// Parse a JSON string produced by `to_json`.
pub fn from_json(s: &str) -> Result<Network, LoadError> {
    let mut net: Network = serde_json::from_str(s).map_err(|e| LoadError::Decode(e.to_string()))?;
    net.validate().map_err(LoadError::Decode)?;
    net.invalidate_grids();
    Ok(net)
}

/// `.nsamples` contents — the text `save_samples` writes.
pub fn samples_to_string(s: &crate::Samples) -> String {
    let mut out = String::new();
    for (t, v) in s.times.iter().zip(&s.values) {
        out.push_str(&format!("{t} {v}\n"));
    }
    out
}

/// `.nsamples`: the original's samples file — one "time value" pair per line.
pub fn save_samples(s: &crate::Samples, path: impl AsRef<Path>) -> io::Result<()> {
    std::fs::write(path, samples_to_string(s))
}

/// Load a `.nsamples` file ("time value" per line; blank/# lines ignored).
pub fn load_samples(path: impl AsRef<Path>, name: &str) -> Result<crate::Samples, LoadError> {
    let s = std::fs::read_to_string(path)?;
    let mut smp = crate::Samples {
        name: name.to_string(),
        ..Default::default()
    };
    for (i, line) in s.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let t = it
            .next()
            .and_then(|x| x.parse::<f32>().ok())
            .ok_or_else(|| LoadError::Decode(format!("line {}: bad time", i + 1)))?;
        let v = it
            .next()
            .and_then(|x| x.parse::<f32>().ok())
            .ok_or_else(|| LoadError::Decode(format!("line {}: bad value", i + 1)))?;
        smp.times.push(t);
        smp.values.push(v);
    }
    Ok(smp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders;
    use crate::Rng;

    fn tempdir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ne2-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn roundtrip() {
        let mut net = Network::new();
        let mut rng = Rng::new(1);
        builders::create_2d_grid(&mut net, "col", 5, 5, 40.0, [0.0; 3]);
        builders::create_connections(&mut net, "col", 0.5, 65.0, 30.0, &mut rng);
        net.add_input("stim", vec![(0, 1.0), (3, 0.5)], [-100.0; 3]);
        net.samples_mut("out").times = vec![0.0, 1.0];
        net.samples_mut("out").values = vec![0.0, 1.0];
        net.scripts.push(("main".into(), "print(1)".into()));
        let path = tempdir().join("rt.ne2");
        save(&net, &path).unwrap();
        let back = load(&path).unwrap();
        assert_eq!(net.neurons.len(), back.neurons.len());
        assert_eq!(net.synapses.len(), back.synapses.len());
        assert_eq!(back.groups[0].name, "col");
        assert_eq!(back.inputs[0].targets.len(), 2);
        assert_eq!(back.scripts[0].1, "print(1)");
        assert_eq!(back.meta.dt_ms, net.meta.dt_ms);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn synapse_tau_round_trips() {
        let mut net = Network::new();
        net.add_neuron([0.0; 3], crate::NeuronKind::Excitatory);
        net.add_neuron([50.0; 3], crate::NeuronKind::Excitatory);
        let s0 = net.connect(0, 1, 1.0);
        net.synapses[s0].tau_ms = 37.5;
        net.connect(0, 1, 2.0); // stays default
        let path = tempdir().join("tau.ne2");
        save(&net, &path).unwrap();
        let back = load(&path).unwrap();
        assert_eq!(back.synapses[0].tau_ms, 37.5);
        assert_eq!(back.synapses[1].tau_ms, 0.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn v2_file_lifts_synapses_with_default_tau() {
        // hand-build a v2 stream: same field order as v3 minus tau_ms
        let mut net = Network::new();
        net.add_neuron([0.0; 3], crate::NeuronKind::Excitatory);
        net.add_neuron([50.0; 3], crate::NeuronKind::Excitatory);
        net.connect(0, 1, 1.5);
        let path = tempdir().join("v2.ne2");
        {
            use std::io::Write;
            let mut w = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            w.write_all(MAGIC).unwrap();
            w.write_all(&2u32.to_le_bytes()).unwrap();
            bincode::serialize_into(&mut w, &net.neurons).unwrap();
            let v2syns: Vec<SynapseV2> = net
                .synapses
                .iter()
                .map(|s| SynapseV2 {
                    src: s.src,
                    tgt: s.tgt,
                    w: s.w,
                    delay_ms: s.delay_ms,
                })
                .collect();
            bincode::serialize_into(&mut w, &v2syns).unwrap();
            bincode::serialize_into(&mut w, &net.groups).unwrap();
            bincode::serialize_into(&mut w, &net.inputs).unwrap();
            bincode::serialize_into(&mut w, &net.samples).unwrap();
            bincode::serialize_into(&mut w, &net.readouts).unwrap();
            bincode::serialize_into(&mut w, &net.scripts).unwrap();
            bincode::serialize_into(&mut w, &net.meta).unwrap();
        }
        let back = load(&path).unwrap();
        assert_eq!(back.synapses.len(), 1);
        assert_eq!(back.synapses[0].w, 1.5);
        assert_eq!(back.synapses[0].tau_ms, 0.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn csr_sorted_and_delayed() {
        let mut net = Network::new();
        net.meta.conduction_speed = 1000.0; // 1 um -> 1 us = 0.001 ms
        for i in 0..4 {
            net.add_neuron([i as f32 * 100.0, 0.0, 0.0], crate::NeuronKind::Excitatory);
        }
        // 100um at 1000 um/ms = 0.1 ms -> 1 step at dt=0.1
        net.connect(2, 0, 5.0);
        net.connect(0, 1, 6.0);
        net.connect(0, 3, 7.0);
        net.connect(2, 3, 8.0);
        let csr = net.synapse_csr(0.1);
        // src order: 0->[1,3], 2->[0,3]
        assert_eq!(csr.tgt, vec![1, 3, 0, 3]);
        assert_eq!(csr.w, vec![6.0, 7.0, 5.0, 8.0]);
        // distances 100,300,200,100 um at 1000 um/ms -> 0.1,0.3,0.2,0.1 ms -> 1,3,2,1 steps
        assert_eq!(csr.delay_steps, vec![1, 3, 2, 1]);
        assert_eq!(csr.src_ptr, vec![0, 2, 2, 4, 4]);
        // dt=0.05 -> 2,6,4,2 steps
        let csr2 = net.synapse_csr(0.05);
        assert_eq!(csr2.delay_steps, vec![2, 6, 4, 2]);
    }

    #[test]
    fn interpolation() {
        let s = crate::Samples {
            times: vec![0.0, 10.0, 20.0],
            values: vec![0.0, 100.0, 0.0],
            ..Default::default()
        };
        assert_eq!(s.value_at(-5.0), 0.0);
        assert_eq!(s.value_at(0.0), 0.0);
        assert_eq!(s.value_at(5.0), 50.0);
        assert_eq!(s.value_at(15.0), 50.0);
        assert_eq!(s.value_at(25.0), 0.0);
    }

    #[test]
    fn grid_metadata_round_trips() {
        let mut net = Network::new();
        builders::create_2d_grid(&mut net, "p", 4, 3, 10.0, [100.0; 3]);
        builders::create_3d_grid(&mut net, "v", 2, 2, 2, 10.0, [-100.0; 3]);
        let path = tempdir().join("grid.ne2");
        save(&net, &path).unwrap();
        let back = load(&path).unwrap();
        assert_eq!(back.group("p").unwrap().grid, net.group("p").unwrap().grid);
        assert_eq!(back.group("v").unwrap().grid, net.group("v").unwrap().grid);
        std::fs::remove_file(&path).ok();

        // JSON keeps it too, and a file written before the field existed
        // (no "grid" key) still parses with grid == None.
        let s = to_json(&net).unwrap();
        assert_eq!(
            from_json(&s).unwrap().group("p").unwrap().grid,
            net.group("p").unwrap().grid
        );
        let mut v: serde_json::Value = serde_json::from_str(&s).unwrap();
        for g in v["groups"].as_array_mut().unwrap() {
            g.as_object_mut().unwrap().remove("grid");
        }
        let old = from_json(&serde_json::to_string(&v).unwrap()).unwrap();
        assert!(old.group("p").unwrap().grid.is_none());
    }

    /// A format-v1 `.ne2` (groups without the `grid` field) still loads.
    #[test]
    fn loads_v1_files() {
        let mut net = Network::new();
        builders::create_2d_grid(&mut net, "p", 3, 2, 10.0, [0.0; 3]);
        net.connect(0, 5, 1.0);
        net.add_input("i", vec![(0, 1.0)], [0.0; 3]);
        net.scripts.push(("s".into(), "print(1)".into()));

        // Hand-serialize the v1 layout: fields in Network order, groups as
        // (name, members) pairs — bincode encodes a struct as field order.
        let path = tempdir().join("v1.ne2");
        let mut w = Vec::new();
        w.extend_from_slice(b"NE2\0");
        w.extend_from_slice(&1u32.to_le_bytes());
        bincode::serialize_into(&mut w, &net.neurons).unwrap();
        let syns: Vec<SynapseV2> = net
            .synapses
            .iter()
            .map(|s| SynapseV2 {
                src: s.src,
                tgt: s.tgt,
                w: s.w,
                delay_ms: s.delay_ms,
            })
            .collect();
        bincode::serialize_into(&mut w, &syns).unwrap();
        let g1: Vec<(String, Vec<u32>)> = net
            .groups
            .iter()
            .map(|g| (g.name.clone(), g.members.clone()))
            .collect();
        bincode::serialize_into(&mut w, &g1).unwrap();
        bincode::serialize_into(&mut w, &net.inputs).unwrap();
        bincode::serialize_into(&mut w, &net.samples).unwrap();
        bincode::serialize_into(&mut w, &net.readouts).unwrap();
        bincode::serialize_into(&mut w, &net.scripts).unwrap();
        bincode::serialize_into(&mut w, &net.meta).unwrap();
        std::fs::write(&path, &w).unwrap();

        let back = load(&path).unwrap();
        assert_eq!(back.neurons.len(), net.neurons.len());
        assert_eq!(back.synapses.len(), net.synapses.len());
        assert_eq!(back.group("p").unwrap().members.len(), 6);
        assert!(back.group("p").unwrap().grid.is_none());
        assert_eq!(back.scripts.len(), 1);
        std::fs::remove_file(&path).ok();
    }

    /// A stale/wrong grid in a hand-edited file is dropped, not trusted.
    #[test]
    fn stale_grid_in_file_is_dropped() {
        let mut net = Network::new();
        builders::create_2d_grid(&mut net, "p", 3, 3, 10.0, [0.0; 3]);
        // corrupt the shape so it no longer matches the member list
        net.groups[0].grid.as_mut().unwrap().dims = [3, 3, 3];
        let path = tempdir().join("stale.ne2");
        save(&net, &path).unwrap();
        let back = load(&path).unwrap();
        assert!(back.group("p").unwrap().grid.is_none());
        assert_eq!(back.group("p").unwrap().members.len(), 9);
        std::fs::remove_file(&path).ok();
    }

    /// A net with one dangling reference of each kind — `validate` must
    /// reject all of them on every load path.
    fn malformed_nets() -> Vec<Network> {
        let mut base = Network::new();
        base.add_neuron([0.0; 3], crate::NeuronKind::Excitatory);
        vec![
            {
                let mut n = base.clone();
                n.synapses.push(crate::Synapse {
                    src: 0,
                    tgt: 9,
                    w: 1.0,
                    delay_ms: 1.0,
                    tau_ms: 0.0,
                });
                n
            },
            {
                let mut n = base.clone();
                let g = n.add_group("g");
                n.groups[g].members.push(9);
                n
            },
            {
                let mut n = base.clone();
                n.add_input("i", vec![(9, 1.0)], [0.0; 3]);
                n
            },
            {
                let mut n = base.clone();
                n.add_readout("r", vec![9]);
                n
            },
        ]
    }

    #[test]
    fn load_rejects_dangling_refs() {
        for (i, net) in malformed_nets().into_iter().enumerate() {
            let path = tempdir().join(format!("bad{i}.ne2"));
            save(&net, &path).unwrap(); // save doesn't validate; load must
            match load(&path) {
                Err(LoadError::Decode(e)) => assert!(e.contains("9"), "{e}"),
                other => panic!("case {i}: expected Decode error, got {other:?}"),
            }
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn json_load_rejects_dangling_refs() {
        for (i, net) in malformed_nets().into_iter().enumerate() {
            let s = to_json(&net).unwrap();
            assert!(from_json(&s).is_err(), "from_json case {i}");
            let path = tempdir().join(format!("bad{i}.json"));
            std::fs::write(&path, &s).unwrap();
            assert!(load_json(&path).is_err(), "load_json case {i}");
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn connections_respect_lambda() {
        let mut rng = Rng::new(3);
        let mut net = Network::new();
        // two tight clusters 1000um apart; lambda=65 -> almost no cross links
        builders::create_2d_grid(&mut net, "a", 3, 3, 10.0, [0.0; 3]);
        builders::create_2d_grid(&mut net, "b", 3, 3, 10.0, [0.0, 0.0, 1000.0]);
        let all: Vec<u32> = (0..18).collect();
        let gi = net.add_group("all");
        net.groups[gi].members = all;
        builders::create_connections(&mut net, "all", 0.9, 65.0, 30.0, &mut rng);
        let cross = net
            .synapses
            .iter()
            .filter(|s| (s.src < 9) != (s.tgt < 9))
            .count();
        let inside = net.synapses.len() - cross;
        assert!(inside > 30);
        assert_eq!(cross, 0);
    }
}
