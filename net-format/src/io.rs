//! `.ne2` file format: magic, version, then a bincode-serialized `Network`.
//! Both `sim` and `editor` use only this crate, so the file can never drift
//! between the two programs.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::Network;

const MAGIC: &[u8; 4] = b"NE2\0";
const FORMAT_VERSION: u32 = 1;

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

pub fn save(net: &Network, path: impl AsRef<Path>) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    bincode::serialize_into(&mut w, net).map_err(io::Error::other)?;
    w.flush()
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
    if version != FORMAT_VERSION {
        return Err(LoadError::UnsupportedVersion(version));
    }
    bincode::deserialize_from(&mut r).map_err(|e| LoadError::Decode(e.to_string()))
}

/// Debug-friendly JSON export/import (not the primary format).
pub fn save_json(net: &Network, path: impl AsRef<Path>) -> io::Result<()> {
    let s = serde_json::to_string_pretty(net).map_err(io::Error::other)?;
    std::fs::write(path, s)
}

pub fn load_json(path: impl AsRef<Path>) -> Result<Network, LoadError> {
    let s = std::fs::read_to_string(path)?;
    serde_json::from_str(&s).map_err(|e| LoadError::Decode(e.to_string()))
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
