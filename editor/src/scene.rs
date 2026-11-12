//! CPU-side scene helpers: overlay lines (floor grid, work plane, group
//! boxes, drag helpers) and the spike-particle system that walks dots along
//! display edges (an approximation of the original's travelling axon spikes).

use glam::Vec3;

use crate::gpu::{Part, Pv};
use crate::snap::GeoSnap;

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
///
/// Flights are timed on the WALL clock, not the sim clock: a dot that
/// should read as "a spike travelling" must be visible for a human-scale
/// duration no matter the sim rate or frame cadence. Sim time only feeds
/// rewind detection (a Reset clears the air).
pub struct Particles {
    flights: Vec<Flight>,
    last_seen_t: f64,
}

const MAX_PARTICLES: usize = 60_000;
const PART_DOT: f32 = 1.6;
/// A flight never reads shorter than this — at sim speed 20 a 10 ms
/// axon delay was ~0.5 ms wall, gone before the next frame.
pub const MIN_FLIGHT_MS: f64 = 350.0;
/// Nor longer — a 2 s sim delay must not hang a dot for 2 wall seconds.
pub const MAX_FLIGHT_MS: f64 = 1500.0;

impl Default for Particles {
    fn default() -> Self {
        Self::new()
    }
}

impl Particles {
    pub fn new() -> Self {
        Self {
            flights: Vec::with_capacity(1024),
            last_seen_t: 0.0,
        }
    }

    /// Feed a snapshot: for each neuron that spiked since last publish,
    /// launch dots on its display edges. `sim_ms` is only the rewind
    /// detector; `wall_ms` stamps the flights.
    pub fn feed(&mut self, geo: &GeoSnap, counts: &[u32], sim_ms: f64, wall_ms: f64) {
        if sim_ms < self.last_seen_t {
            self.flights.clear(); // sim reset
        }
        self.last_seen_t = sim_ms;
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
                // Sim axon delay as the wall duration, clamped to a
                // visible window.
                let dur = ((delay as f64) * geo.dt_ms as f64)
                    .clamp(MIN_FLIGHT_MS, MAX_FLIGHT_MS);
                self.flights.push(Flight {
                    t0_ms: wall_ms - dur * (k as f64 / c.max(1) as f64),
                    dur_ms: dur,
                    src: Vec3::from(geo.pos[i]),
                    dst: Vec3::from(geo.pos[tgt as usize]),
                    inhibitory: geo.kind.get(i) == Some(&1),
                });
            }
        }
    }

    /// Produce the instance list for this frame; drops finished flights.
    /// `wall_ms` is the same clock `feed` stamps flights with.
    pub fn instances(&mut self, wall_ms: f64) -> Vec<Part> {
        self.flights.retain(|f| wall_ms - f.t0_ms < f.dur_ms + 0.001);
        self.flights
            .iter()
            .map(|f| {
                let k = ((wall_ms - f.t0_ms) / f.dur_ms).clamp(0.0, 1.0) as f32;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snap::GeoSnap;

    /// Two neurons at ±10 on x, one display edge 0→1 with a 1-step
    /// delay (the invisibility case: sim delay ≈ 1 ms).
    fn geo() -> GeoSnap {
        let mut g = GeoSnap::default();
        g.pos = vec![[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        g.kind = vec![0, 0];
        g.disp_ptr = vec![0, 1, 1];
        g.disp_edges = vec![(1, 1)];
        g.dt_ms = 1.0;
        g
    }

    /// Stage 13 D: a 1 ms sim delay must still fly for ≥ MIN_FLIGHT_MS
    /// of WALL time — before, flights keyed off the sim clock expired
    /// between frames at high sim speeds.
    #[test]
    fn dots_live_at_least_min_wall_ms() {
        let mut p = Particles::new();
        let g = geo();
        p.feed(&g, &[1, 0], /* sim */ 10.0, /* wall */ 1000.0);
        // mid-flight: one dot between src and dst
        let mid = p.instances(1000.0 + MIN_FLIGHT_MS / 2.0);
        assert_eq!(mid.len(), 1);
        assert!(mid[0].pos[0] > -10.0 && mid[0].pos[0] < 10.0);
        // still alive near the end, gone after dur
        assert_eq!(p.instances(1000.0 + MIN_FLIGHT_MS - 1.0).len(), 1);
        assert_eq!(p.instances(1000.0 + MAX_FLIGHT_MS + 1.0).len(), 0);
    }

    /// Sim-clock rewind clears flights even though the wall clock kept
    /// running forward.
    #[test]
    fn sim_rewind_clears_flights() {
        let mut p = Particles::new();
        let g = geo();
        p.feed(&g, &[1, 0], 10.0, 1000.0);
        p.feed(&g, &[1, 0], 5.0, 1100.0); // sim went backwards
        // only the second feed's flight exists (t0 = 1100)
        let mid = p.instances(1100.0 + MIN_FLIGHT_MS / 2.0);
        assert_eq!(mid.len(), 1);
    }

    /// Burst cap: the particle count never exceeds MAX_PARTICLES.
    #[test]
    fn particle_count_is_capped() {
        let mut p = Particles::new();
        let g = geo();
        for i in 0..(MAX_PARTICLES / 4 + 100) {
            p.feed(&g, &[4, 0], i as f64, 1000.0 + i as f64);
        }
        assert!(p.instances(2000.0).len() <= MAX_PARTICLES);
    }
}
