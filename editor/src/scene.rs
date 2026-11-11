//! CPU-side scene helpers: overlay lines (floor grid, work plane, group
//! boxes, drag helpers) and the spike-particle system that walks dots along
//! display edges (an approximation of the original's travelling axon spikes).

use glam::Vec3;

use crate::gpu::{Part, Pv};
use crate::snap::GeoSnap;

const GRID_COL: [f32; 4] = [0.28, 0.32, 0.38, 0.8];
const PLANE_COL: [f32; 4] = [0.35, 0.5, 0.6, 0.5];
const GROUP_COL: [f32; 4] = [0.9, 0.8, 0.3, 0.6];
const CONN_COL: [f32; 4] = [1.0, 0.9, 0.2, 1.0];

pub fn pv(pos: [f32; 3], color: [f32; 4]) -> Pv {
    Pv {
        pos,
        color,
        _pad: 0.0,
    }
}

fn line(out: &mut Vec<Pv>, a: Vec3, b: Vec3, c: [f32; 4]) {
    out.push(pv(a.into(), c));
    out.push(pv(b.into(), c));
}

/// Floor grid on x-z (original: step 20, ±1000) centred under the camera
/// target so it stays useful at any zoom.
pub fn floor_grid(center: Vec3, out: &mut Vec<Pv>) {
    let step = 20.0f32;
    let ext = 1000.0f32;
    let x0 = (center.x / step).round() * step - ext;
    let z0 = (center.z / step).round() * step - ext;
    let x1 = (center.x / step).round() * step + ext;
    let z1 = (center.z / step).round() * step + ext;
    let y = 0.0;
    let mut x = x0;
    while x <= x1 {
        line(out, Vec3::new(x, y, z0), Vec3::new(x, y, z1), GRID_COL);
        x += step;
    }
    let mut z = z0;
    while z <= z1 {
        line(out, Vec3::new(x0, y, z), Vec3::new(x1, y, z), GRID_COL);
        z += step;
    }
}

/// Work-plane grid near the target (move/insert reference plane).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Plane {
    XZ,
    XY,
    YZ,
}

pub fn work_plane(center: Vec3, plane: Plane, out: &mut Vec<Pv>) {
    let step = 20.0f32;
    let ext = 200.0f32;
    let c = |v: f32| (v / step).round() * step;
    for i in -10..=10 {
        let o = i as f32 * step;
        let o0 = -ext;
        let o1 = ext;
        match plane {
            Plane::XZ => {
                line(
                    out,
                    Vec3::new(c(center.x) + o, center.y, c(center.z) + o0),
                    Vec3::new(c(center.x) + o, center.y, c(center.z) + o1),
                    PLANE_COL,
                );
                line(
                    out,
                    Vec3::new(c(center.x) + o0, center.y, c(center.z) + o),
                    Vec3::new(c(center.x) + o1, center.y, c(center.z) + o),
                    PLANE_COL,
                );
            }
            Plane::XY => {
                line(
                    out,
                    Vec3::new(c(center.x) + o, c(center.y) + o0, center.z),
                    Vec3::new(c(center.x) + o, c(center.y) + o1, center.z),
                    PLANE_COL,
                );
                line(
                    out,
                    Vec3::new(c(center.x) + o0, c(center.y) + o, center.z),
                    Vec3::new(c(center.x) + o1, c(center.y) + o, center.z),
                    PLANE_COL,
                );
            }
            Plane::YZ => {
                line(
                    out,
                    Vec3::new(center.x, c(center.y) + o, c(center.z) + o0),
                    Vec3::new(center.x, c(center.y) + o, c(center.z) + o1),
                    PLANE_COL,
                );
                line(
                    out,
                    Vec3::new(center.x, c(center.y) + o0, c(center.z) + o),
                    Vec3::new(center.x, c(center.y) + o1, c(center.z) + o),
                    PLANE_COL,
                );
            }
        }
    }
}

/// Group bounding boxes.
pub fn group_boxes(geo: &GeoSnap, out: &mut Vec<Pv>) {
    for (_, members) in &geo.groups {
        if members.is_empty() {
            continue;
        }
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for &m in members {
            if let Some(p) = geo.pos.get(m as usize) {
                let v = Vec3::from(*p);
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        if mn.x > mx.x {
            continue;
        }
        let pad = 12.0;
        mn -= Vec3::splat(pad);
        mx += Vec3::splat(pad);
        box_lines(out, mn, mx, GROUP_COL);
    }
}

pub fn box_lines(out: &mut Vec<Pv>, mn: Vec3, mx: Vec3, col: [f32; 4]) {
    let c = [
        Vec3::new(mn.x, mn.y, mn.z),
        Vec3::new(mx.x, mn.y, mn.z),
        Vec3::new(mx.x, mn.y, mx.z),
        Vec3::new(mn.x, mn.y, mx.z),
        Vec3::new(mn.x, mx.y, mn.z),
        Vec3::new(mx.x, mx.y, mn.z),
        Vec3::new(mx.x, mx.y, mx.z),
        Vec3::new(mn.x, mx.y, mx.z),
    ];
    const E: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    for (a, b) in E {
        line(out, c[a], c[b], col);
    }
}

/// Connect-mode drag line.
pub fn connect_line(out: &mut Vec<Pv>, a: Vec3, b: Vec3) {
    line(out, a, b, CONN_COL);
}

// ---------- spike particles ----------

struct Flight {
    t0_ms: f64,
    dur_ms: f64,
    src: Vec3,
    dst: Vec3,
    inhibitory: bool,
}

/// Dots travelling along a few display synapses per spiking neuron.
/// (Approximation of the original's per-axon-node travelling spikes — only
/// the first K outgoing synapses per neuron get dots; see NOTES.)
pub struct Particles {
    flights: Vec<Flight>,
    last_seen_t: f64,
}

const MAX_PARTICLES: usize = 60_000;
const PART_DOT: f32 = 1.6;

impl Particles {
    pub fn new() -> Self {
        Self {
            flights: Vec::with_capacity(1024),
            last_seen_t: 0.0,
        }
    }

    /// Feed a snapshot: for each neuron that spiked since last publish,
    /// launch dots on its display edges.
    pub fn feed(&mut self, geo: &GeoSnap, counts: &[u32], now_ms: f64) {
        if now_ms < self.last_seen_t {
            self.flights.clear(); // sim reset
        }
        self.last_seen_t = now_ms;
        let n = counts.len().min(geo.disp_ptr.len().saturating_sub(1));
        for (i, &c) in counts.iter().enumerate().take(n) {
            if c == 0 {
                continue;
            }
            let (s, e) = (geo.disp_ptr[i] as usize, geo.disp_ptr[i + 1] as usize);
            if s == e {
                continue;
            }
            // Per spike, pick one edge deterministically by index hash.
            for k in 0..c.min(4) {
                if self.flights.len() >= MAX_PARTICLES {
                    return;
                }
                let (tgt, delay) = geo.disp_edges[s + ((i + k as usize) % (e - s))];
                let dur = (delay as f64) * geo.dt_ms as f64;
                self.flights.push(Flight {
                    t0_ms: now_ms - dur * (k as f64 / c.max(1) as f64),
                    dur_ms: dur.max(0.5),
                    src: Vec3::from(geo.pos[i]),
                    dst: Vec3::from(geo.pos[tgt as usize]),
                    inhibitory: geo.kind.get(i) == Some(&1),
                });
            }
        }
    }

    /// Produce the instance list for this frame; drops finished flights.
    pub fn instances(&mut self, now_ms: f64) -> Vec<Part> {
        self.flights.retain(|f| now_ms - f.t0_ms < f.dur_ms + 0.001);
        self.flights
            .iter()
            .map(|f| {
                let k = ((now_ms - f.t0_ms) / f.dur_ms).clamp(0.0, 1.0) as f32;
                let p = f.src.lerp(f.dst, k);
                let glow = (1.0 - k * 0.6) * 0.9;
                let col = if f.inhibitory {
                    [0.6, 0.7, 1.0, glow]
                } else {
                    [1.0, 0.9, 0.4, glow]
                };
                Part {
                    pos: [p.x, p.y, p.z, PART_DOT],
                    color: col,
                }
            })
            .collect()
    }
}
