//! Datasets for training jobs: a named multi-channel f32 series with
//! planted events and event-time labels, cached under
//! `data_dir/datasets/<name>/` (or `datasets/<name>/` verbatim in local
//! mode — the same `resolve_in` rules as every other file fn).
//!
//! Layout on disk:
//!   `signal.f32` or `signal.npy` — channel-major samples (C order)
//!   `meta.json`   — {name, channels, steps, dt_ms, spec}
//!   `labels.json` — {events_ms: […]}
//!
//! `generate` is deterministic in `spec.seed` — regenerate or reuse the
//! cached directory when the spec matches.

use std::path::{Path, PathBuf};

use net_format::Rng;

/// A loaded or generated dataset. `channels[c]` is one f32 series of
/// `steps` samples at `dt_ms` each; `events_ms` are event start times.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub name: String,
    pub channels: Vec<Vec<f32>>,
    pub dt_ms: f32,
    pub events_ms: Vec<f64>,
    /// Canonical JSON of the generator spec (cache key; empty for
    /// uploads/loads where no spec was recorded).
    pub spec_json: String,
}

impl Dataset {
    pub fn steps(&self) -> usize {
        self.channels.first().map(|c| c.len()).unwrap_or(0)
    }
    pub fn len_ms(&self) -> f64 {
        self.steps() as f64 * self.dt_ms as f64
    }

    /// The dataset's directory under a resolved `datasets/` root path.
    pub fn dir_under(datasets_root: &Path, name: &str) -> PathBuf {
        datasets_root.join(name)
    }

    /// Write `signal.f32` + `meta.json` + `labels.json` into
    /// `datasets_root/<name>/`.
    pub fn save(&self, datasets_root: &Path) -> Result<(), String> {
        let dir = Self::dir_under(datasets_root, &self.name);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity(4 * self.channels.len() * self.steps());
        for ch in &self.channels {
            for &v in ch {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(dir.join("signal.f32"), &bytes).map_err(|e| e.to_string())?;
        let meta = serde_json::json!({
            "name": self.name,
            "channels": self.channels.len(),
            "steps": self.steps(),
            "dt_ms": self.dt_ms,
            "spec": serde_json::from_str::<serde_json::Value>(&self.spec_json)
                .unwrap_or(serde_json::Value::Null),
        });
        std::fs::write(
            dir.join("meta.json"),
            serde_json::to_string_pretty(&meta).unwrap(),
        )
        .map_err(|e| e.to_string())?;
        let labels = serde_json::json!({ "events_ms": self.events_ms });
        std::fs::write(
            dir.join("labels.json"),
            serde_json::to_string_pretty(&labels).unwrap(),
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Read `datasets_root/<name>/` back (generated or uploaded files).
    pub fn load(datasets_root: &Path, name: &str) -> Result<Dataset, String> {
        let dir = Self::dir_under(datasets_root, name);
        let meta_s = std::fs::read_to_string(dir.join("meta.json"))
            .map_err(|e| format!("dataset {name:?}: meta.json: {e}"))?;
        let meta: serde_json::Value =
            serde_json::from_str(&meta_s).map_err(|e| format!("meta.json: {e}"))?;
        let n_ch = meta["channels"].as_u64().unwrap_or(0) as usize;
        let steps = meta["steps"].as_u64().unwrap_or(0) as usize;
        let dt_ms = meta["dt_ms"].as_f64().unwrap_or(0.1) as f32;
        if n_ch == 0 || steps == 0 {
            return Err(format!("dataset {name:?}: meta has channels=0 or steps=0"));
        }
        let spec_json = meta["spec"].to_string();

        // signal: raw f32 or .npy — channel-major, len = channels × steps.
        let flat: Vec<f32> = {
            let f32p = dir.join("signal.f32");
            let npyp = dir.join("signal.npy");
            if f32p.is_file() {
                let raw = std::fs::read(&f32p).map_err(|e| e.to_string())?;
                if raw.len() % 4 != 0 {
                    return Err(format!("signal.f32: {} B not a multiple of 4", raw.len()));
                }
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else if npyp.is_file() {
                let raw = std::fs::read(&npyp).map_err(|e| e.to_string())?;
                let (_dims, v) = read_npy_f32(&raw).map_err(|e| format!("signal.npy: {e}"))?;
                v
            } else {
                return Err(format!("dataset {name:?}: no signal.f32 or signal.npy"));
            }
        };
        if flat.len() != n_ch * steps {
            return Err(format!(
                "dataset {name:?}: signal has {} samples, meta wants {n_ch}×{steps}",
                flat.len()
            ));
        }
        let channels = flat
            .chunks_exact(steps)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>();

        let labels_s = std::fs::read_to_string(dir.join("labels.json"))
            .map_err(|e| format!("labels.json: {e}"))?;
        let labels: serde_json::Value =
            serde_json::from_str(&labels_s).map_err(|e| format!("labels.json: {e}"))?;
        let events_ms = labels["events_ms"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        Ok(Dataset {
            name: name.into(),
            channels,
            dt_ms,
            events_ms,
            spec_json,
        })
    }

    /// Load when `datasets_root/<name>` already matches `spec_json`,
    /// else generate + save. The cache path is what makes dataset_gen
    /// cheap on repeat runs — the spec echo decides reuse, not mtime.
    pub fn generate_or_load(datasets_root: &Path, spec: &GenSpec) -> Result<Dataset, String> {
        let dir = Self::dir_under(datasets_root, &spec.name);
        if let Ok(meta_s) = std::fs::read_to_string(dir.join("meta.json")) {
            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_s) {
                if meta["spec"].to_string() == spec.canonical_json() {
                    if let Ok(d) = Self::load(datasets_root, &spec.name) {
                        return Ok(d);
                    }
                }
            }
        }
        let d = generate(spec);
        d.save(datasets_root)?;
        Ok(d)
    }
}

// ---------------------------------------------------------------------------
// generator
// ---------------------------------------------------------------------------

/// Parsed generator spec (from the `dataset_gen(#{…})` map).
#[derive(Clone, Debug)]
pub struct GenSpec {
    pub name: String,
    pub channels: usize,
    pub len_ms: f64,
    pub dt_ms: f64,
    /// "poisson" (default) | "gauss".
    pub dist: String,
    /// poisson: per-channel rate; gauss: mean/sd.
    pub rate_hz: f64,
    /// poisson event amplitude / gauss baseline mean.
    pub mean: f64,
    pub sd: f64,
    pub seed: u64,
    pub events: Option<EventSpec>,
}

#[derive(Clone, Debug)]
pub struct EventSpec {
    /// "rate" (rate change in a window) | "corr" (shared drive added to a
    /// channel subset) | "motif" (a pattern wave across a subset).
    pub kind: String,
    /// Mean period between event starts; first event at `every_ms`.
    pub every_ms: f64,
    /// Uniform ± jitter on the period.
    pub jitter_ms: f64,
    /// Window length for "rate"/"corr".
    pub window_ms: f64,
    /// rate: rate multiplier; corr: added shared-drive Hz; motif: amplitude.
    pub strength: f64,
    /// Affected channel subset; empty = all.
    pub channels: Vec<usize>,
    /// motif: per-channel wave delay.
    pub gap_ms: f64,
    /// motif: amplitude envelope per step (empty → single sample).
    pub pattern: Vec<f32>,
}

impl GenSpec {
    /// Deterministic JSON used as the cache key for `generate_or_load`.
    pub fn canonical_json(&self) -> String {
        let ev = self.events.as_ref().map(|e| {
            serde_json::json!({
                "kind": e.kind, "every_ms": e.every_ms, "jitter_ms": e.jitter_ms,
                "window_ms": e.window_ms, "strength": e.strength,
                "channels": e.channels, "gap_ms": e.gap_ms, "pattern": e.pattern,
            })
        });
        serde_json::json!({
            "name": self.name, "channels": self.channels, "len_ms": self.len_ms,
            "dt_ms": self.dt_ms, "dist": self.dist, "rate_hz": self.rate_hz,
            "mean": self.mean, "sd": self.sd, "seed": self.seed, "events": ev,
        })
        .to_string()
    }
}

/// Deterministic generation from `spec.seed`. All randomness flows
/// through `net_format::Rng` in a fixed consumption order: base series
/// channel-by-channel, then event times, then per-event effects.
pub fn generate(spec: &GenSpec) -> Dataset {
    let steps = (spec.len_ms / spec.dt_ms).round().max(1.0) as usize;
    let mut rng = Rng::new(spec.seed);

    // --- base series ---
    let mut channels: Vec<Vec<f32>> = Vec::with_capacity(spec.channels);
    for _ in 0..spec.channels {
        let mut ch = Vec::with_capacity(steps);
        if spec.dist == "gauss" {
            for _ in 0..steps {
                ch.push(rng.gauss(spec.mean, spec.sd) as f32);
            }
        } else {
            let p = (spec.rate_hz * spec.dt_ms / 1000.0).clamp(0.0, 1.0);
            let amp = if spec.mean == 0.0 { 1.0 } else { spec.mean } as f32;
            for _ in 0..steps {
                ch.push(if rng.bernoulli(p) { amp } else { 0.0 });
            }
        }
        channels.push(ch);
    }

    // --- event times ---
    let mut events_ms = Vec::new();
    if let Some(ev) = &spec.events {
        // How long an event's footprint lasts, so it never spills past
        // the end of the series.
        let span_ms = match ev.kind.as_str() {
            "motif" => {
                let n_sub = if ev.channels.is_empty() {
                    spec.channels
                } else {
                    ev.channels.len()
                };
                (n_sub.max(1) - 1) as f64 * ev.gap_ms + ev.pattern.len().max(1) as f64 * spec.dt_ms
            }
            _ => ev.window_ms,
        };
        let mut t = ev.every_ms;
        while t >= 0.0 && t + span_ms <= spec.len_ms {
            events_ms.push(t);
            t += ev.every_ms + rng.uniform(-ev.jitter_ms, ev.jitter_ms);
        }

        // --- apply each event ---
        let sub: Vec<usize> = if ev.channels.is_empty() {
            (0..spec.channels).collect()
        } else {
            ev.channels
                .iter()
                .copied()
                .filter(|&c| c < spec.channels)
                .collect()
        };
        for &t0 in &events_ms {
            match ev.kind.as_str() {
                "rate" => {
                    // Change the subset's statistics inside the window:
                    // poisson scales its rate by `strength`, gauss shifts
                    // its mean by `strength`.
                    let s0 = (t0 / spec.dt_ms) as usize;
                    let s1 = ((t0 + ev.window_ms) / spec.dt_ms) as usize;
                    for &c in &sub {
                        for s in s0..s1.min(steps) {
                            if spec.dist == "gauss" {
                                channels[c][s] += ev.strength as f32;
                            } else {
                                let p = (spec.rate_hz * ev.strength * spec.dt_ms / 1000.0)
                                    .clamp(0.0, 1.0);
                                let amp = if spec.mean == 0.0 { 1.0 } else { spec.mean } as f32;
                                channels[c][s] = if rng.bernoulli(p) { amp } else { 0.0 };
                            }
                        }
                    }
                }
                "corr" => {
                    // A shared latent drive added identically to every
                    // subset channel — invisible per channel (rate goes
                    // up only by `strength`), visible as coincidence.
                    let s0 = (t0 / spec.dt_ms) as usize;
                    let s1 = ((t0 + ev.window_ms) / spec.dt_ms) as usize;
                    let p = (ev.strength * spec.dt_ms / 1000.0).clamp(0.0, 1.0);
                    let amp = if spec.mean == 0.0 { 1.0 } else { spec.mean } as f32;
                    for s in s0..s1.min(steps) {
                        if rng.bernoulli(p) {
                            for &c in &sub {
                                channels[c][s] += amp;
                            }
                        }
                    }
                }
                "motif" => {
                    // A temporal signature: `pattern` plays on each
                    // subset channel delayed by `gap_ms` — no rate or
                    // mean change at all.
                    let pat: Vec<f32> = if ev.pattern.is_empty() {
                        vec![1.0]
                    } else {
                        ev.pattern.clone()
                    };
                    for (i, &c) in sub.iter().enumerate() {
                        let start = ((t0 + i as f64 * ev.gap_ms) / spec.dt_ms) as usize;
                        for (k, &pv) in pat.iter().enumerate() {
                            let s = start + k;
                            if s < steps {
                                channels[c][s] += pv * ev.strength as f32;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    Dataset {
        name: spec.name.clone(),
        channels,
        dt_ms: spec.dt_ms as f32,
        events_ms,
        spec_json: spec.canonical_json(),
    }
}

// ---------------------------------------------------------------------------
// minimal .npy reader — v1/v2, C order, little-endian <f4/<f8 only
// ---------------------------------------------------------------------------

/// Returns `(shape, samples)`. Only what dataset uploads produce.
fn read_npy_f32(raw: &[u8]) -> Result<(Vec<usize>, Vec<f32>), String> {
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err("not an .npy file".into());
    }
    let (major, mut off) = (raw[6], 8usize);
    let hlen = match major {
        1 => {
            let l = u16::from_le_bytes([raw[off], raw[off + 1]]) as usize;
            off += 2;
            l
        }
        2 | 3 => {
            let l =
                u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as usize;
            off += 4;
            l
        }
        v => return Err(format!("npy version {v} unsupported")),
    };
    let header = std::str::from_utf8(raw.get(off..off + hlen).ok_or("truncated npy header")?)
        .map_err(|_| "npy header not utf-8")?;
    let data = raw.get(off + hlen..).ok_or("missing npy payload")?;

    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.trim_start().strip_prefix('\''))
        .and_then(|s| s.split('\'').next())
        .unwrap_or("");
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true") {
        return Err("fortran-order .npy unsupported".into());
    }
    let shape_s = header
        .split("'shape':")
        .nth(1)
        .ok_or("npy header lacks shape")?;
    let shape: Vec<usize> = shape_s
        .trim_start()
        .trim_start_matches('(')
        .split(')')
        .next()
        .unwrap_or("")
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    if shape.is_empty() {
        return Err("npy shape unparseable".into());
    }
    let n: usize = shape.iter().product();
    let vals: Vec<f32> = match descr {
        "<f4" | "f4" => {
            if data.len() < 4 * n {
                return Err("npy payload short".into());
            }
            data[..4 * n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        "<f8" | "f8" => {
            if data.len() < 8 * n {
                return Err("npy payload short".into());
            }
            data[..8 * n]
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect()
        }
        d => return Err(format!("npy dtype {d:?} unsupported (want <f4/<f8)")),
    };
    Ok((shape, vals))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, seed: u64) -> GenSpec {
        GenSpec {
            name: name.into(),
            channels: 4,
            len_ms: 2000.0,
            dt_ms: 0.5,
            dist: "poisson".into(),
            rate_hz: 50.0,
            mean: 0.0,
            sd: 1.0,
            seed,
            events: Some(EventSpec {
                kind: "corr".into(),
                every_ms: 250.0,
                jitter_ms: 40.0,
                window_ms: 15.0,
                strength: 150.0,
                channels: vec![0, 2],
                gap_ms: 0.0,
                pattern: vec![],
            }),
        }
    }

    #[test]
    fn same_seed_same_bytes() {
        let a = generate(&spec("d", 7));
        let b = generate(&spec("d", 7));
        assert_eq!(a.channels, b.channels);
        assert_eq!(a.events_ms, b.events_ms);
        let c = generate(&spec("d", 8));
        assert!(c.channels != a.channels || c.events_ms != a.events_ms);
    }

    #[test]
    fn labels_line_up_with_events() {
        let d = generate(&spec("d", 7));
        // events start at every_ms=250 and repeat ~250ms over 2000ms
        // minus one footprint: expect 6–8 labels, all inside range.
        assert!(d.events_ms.len() >= 6 && d.events_ms.len() <= 8);
        for &t in &d.events_ms {
            assert!(t >= 250.0 - 40.0 && t + 15.0 <= 2000.0);
        }
        // correlation check: in each event window channels 0 and 2 share
        // far more identical nonzero steps than out of window.
        let mut same_in = 0usize;
        let mut tot_in = 0usize;
        let mut same_out = 0usize;
        let mut tot_out = 0usize;
        let dt = d.dt_ms as f64;
        let win = (15.0 / dt) as usize;
        for (s, (&a, &b)) in d.channels[0].iter().zip(d.channels[2].iter()).enumerate() {
            let t = s as f64 * dt;
            if d.events_ms.iter().any(|&e| t >= e && t < e + 15.0) {
                tot_in += 1;
                if a != 0.0 && a == b {
                    same_in += 1;
                }
            } else {
                tot_out += 1;
                if a != 0.0 && a == b {
                    same_out += 1;
                }
            }
            let _ = win;
        }
        assert!(
            same_in as f64 / tot_in as f64 > 5.0 * (same_out as f64 / tot_out.max(1) as f64 + 1e-9)
        );
    }

    #[test]
    fn save_load_round_trip() {
        let root = std::env::temp_dir().join(format!("ne_ds_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let d = generate(&spec("rt", 3));
        d.save(&root).unwrap();
        let back = Dataset::load(&root, "rt").unwrap();
        assert_eq!(back.channels, d.channels);
        assert_eq!(back.events_ms, d.events_ms);
        assert_eq!(back.dt_ms, d.dt_ms);
        // cache hit: same spec loads, doesn't regenerate
        let again = Dataset::generate_or_load(&root, &spec("rt", 3)).unwrap();
        assert_eq!(again.channels, d.channels);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn npy_reads() {
        // hand-built v1 <f4 npy: shape (2,), values [1.5, -2.25]
        let mut f = b"\x93NUMPY\x01\x00".to_vec();
        let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
        let pad = 64 - ((10 + dict.len() + 1) % 64);
        let header = format!("{dict}{}{}\n", " ".repeat(pad), "");
        f.extend_from_slice(&(header.len() as u16).to_le_bytes());
        f.extend_from_slice(header.as_bytes());
        f.extend_from_slice(&1.5f32.to_le_bytes());
        f.extend_from_slice(&(-2.25f32).to_le_bytes());
        let (shape, v) = read_npy_f32(&f).unwrap();
        assert_eq!(shape, vec![2]);
        assert_eq!(v, vec![1.5, -2.25]);
    }
}
