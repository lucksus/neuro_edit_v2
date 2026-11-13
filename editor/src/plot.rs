//! Plot history state: per-watch voltage traces + the spike raster,
//! plus the sliding-window/follow math shared by the plot tabs.
//!
//! Everything here is indexed by *simulation* milliseconds, which is
//! why `feed` watches for time going backwards — the Reset button, a
//! script's `sim.reset()`, or another client on a shared server can all
//! rewind the clock, and a trace indexed by the old run must not draw
//! on top of the new one.

use std::collections::{HashMap, VecDeque};

use eframe::egui::Color32;

/// Raster keeps the newest N published bins.
const RASTER_MAX: usize = 600;
/// Per-neuron sample cap (at 1 ms steps, ~4 s of history).
const V_HIST_MAX: usize = 4096;

/// Watched-neuron state histories plus the spike raster.
///
/// `slots[s]` is state slot `s` of the watched neurons' model (protocol
/// 2 carries one series per slot; the model's `StateDesc`s name them —
/// slot 0 is always the membrane potential).
pub struct PlotHist {
    /// state slot → neuron id → (t_ms, value) samples, oldest first.
    pub slots: Vec<HashMap<u32, VecDeque<(f64, f32)>>>,
    /// (t_ms, fired-ids) per published snapshot.
    pub raster: VecDeque<(f64, Vec<u32>)>,
    /// Sim time of the last fed snapshot.
    last_t: f64,
    /// How many times fed time went backwards (tests/status).
    pub resets: usize,
}

impl Default for PlotHist {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            raster: VecDeque::with_capacity(64),
            last_t: f64::NEG_INFINITY,
            resets: 0,
        }
    }
}

impl PlotHist {
    /// History of state slot `s`, if any watched data arrived for it.
    pub fn slot(&self, s: usize) -> Option<&HashMap<u32, VecDeque<(f64, f32)>>> {
        self.slots.get(s)
    }

    /// Fold one snapshot into the histories. `watched` is
    /// `(neuron id, per-slot sample series)` — each series ends at
    /// `t_ms` and is walked back by `dt_ms`, so all slots share the
    /// same sim-ms x axis (the phase plot pairs two of them on it).
    ///
    /// When `t_ms` is earlier than the last fed time the sim ran
    /// backwards: clear all histories so only the new run draws.
    pub fn feed(&mut self, t_ms: f64, fired: Vec<u32>, watched: &[(u32, &[Vec<f32>])], dt_ms: f64) {
        if t_ms < self.last_t {
            for s in &mut self.slots {
                s.clear();
            }
            self.raster.clear();
            self.resets += 1;
        }
        self.last_t = t_ms;
        self.raster.push_back((t_ms, fired));
        while self.raster.len() > RASTER_MAX {
            self.raster.pop_front();
        }
        let n_slots = watched.iter().map(|w| w.1.len()).max().unwrap_or(0);
        while self.slots.len() < n_slots {
            self.slots.push(HashMap::new());
        }
        for &(id, series) in watched {
            for (s, vs) in series.iter().enumerate() {
                let h = self.slots[s].entry(id).or_default();
                let n = vs.len();
                for (i, &v) in vs.iter().enumerate() {
                    let t = t_ms - (n - 1 - i) as f64 * dt_ms;
                    h.push_back((t, v));
                }
                while h.len() > V_HIST_MAX {
                    h.pop_front();
                }
            }
        }
    }

    /// Net was rebuilt or Reset was pressed locally — drop everything
    /// including the clock, so the next feed can't trip the reversal
    /// check on an unrelated earlier sim.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.raster.clear();
        self.last_t = f64::NEG_INFINITY;
    }
}

/// Fixed y range for the v(t) plot — the working span of the
/// Izhikevich model. Kept fixed so a spike entering or leaving the
/// window can't bounce the axis; "Auto Y" opts out per tab.
pub const V_Y_RANGE: (f64, f64) = (-90.0, 40.0);

/// egui_plot draws tick labels inside the plot rect: the lowest y label
/// lands in the same corner as the leftmost x label. A y formatter can
/// use this to blank the colliding mark — a y tick is suppressed when it
/// sits within half a grid step of the visible range start.
pub fn y_label_hits_corner(
    mark: &egui_plot::GridMark,
    range: &std::ops::RangeInclusive<f64>,
) -> bool {
    mark.value - *range.start() < 0.5 * mark.step_size
}

/// Auto Y never zooms tighter than this fraction of the descriptor's
/// `plot_range` — without it a resting neuron's ±0.03 mV noise floor
/// stretched across the whole axis and read as a fake spike train
/// (Stage 22B: the Plots window "spiked" while the inspector honestly
/// showed the same flat data pinned to its descriptor range).
pub const AUTO_Y_MIN_FRAC: f64 = 0.12;

/// `[lo, hi]` the Auto-Y plot axis should show for windowed data
/// spanning `dmin..dmax` on a state slot whose descriptor range is
/// `plot_range`: pad the data by ~15 %, but never narrower than
/// `AUTO_Y_MIN_FRAC` of the descriptor span (expanded around the data
/// mid). Real excursions still fit; noise can't fill the plot.
pub fn fit_auto_y(dmin: f64, dmax: f64, plot_range: (f64, f64)) -> (f64, f64) {
    let min_span = ((plot_range.1 - plot_range.0) * AUTO_Y_MIN_FRAC).max(1e-3);
    let span = ((dmax - dmin) * 1.15).max(min_span);
    let mid = (dmin + dmax) / 2.0;
    (mid - span / 2.0, mid + span / 2.0)
}

/// Stable per-watch color, shared by the v(t) and phase tabs — keyed
/// by the position in the sent watch list so the same neuron draws the
/// same color in both (the histories are HashMaps; iterating them
/// would not agree on an order).
pub fn watch_color(i: usize) -> Color32 {
    const P: [(u8, u8, u8); 8] = [
        (91, 192, 235),
        (255, 167, 66),
        (152, 227, 138),
        (231, 138, 195),
        (128, 203, 196),
        (255, 214, 102),
        (179, 157, 219),
        (244, 143, 177),
    ];
    let (r, g, b) = P[i % P.len()];
    Color32::from_rgb(r, g, b)
}

/// The (v, u) point pairs a phase trajectory draws inside `win`
/// (aligned by their shared sim-ms x axis). Returns the windowed
/// trajectory plus the newest point (marked in the plot).
pub fn phase_points(
    vs: &VecDeque<(f64, f32)>,
    us: &VecDeque<(f64, f32)>,
    win: Option<(f64, f64)>,
) -> (Vec<[f64; 2]>, Option<[f64; 2]>) {
    // u and v share t; zip by position (equal lengths by construction,
    // but tolerate drift) and filter on v's timestamp.
    let n = vs.len().min(us.len());
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let (t, v) = vs[vs.len() - n + i];
        let (_, u) = us[us.len() - n + i];
        if let Some((lo, hi)) = win {
            if !(lo..=hi).contains(&t) {
                continue;
            }
        }
        pts.push([v as f64, u as f64]);
    }
    let newest = pts.last().copied();
    (pts, newest)
}

/// Sliding-window state for the v(t), raster and phase plots.
///
/// `follow` pins the window's right edge to the newest fed sim time.
/// Dragging or zooming the plot turns follow off (egui reports it via
/// the plot response); pressing Follow or double-clicking turns it on.
pub struct PlotWindow {
    /// Window length in sim ms.
    pub len_ms: f64,
    /// Track the newest time; false once the user pans/zooms away.
    pub follow: bool,
    /// v(t) only: false pins the y axis to `V_Y_RANGE`.
    pub auto_y: bool,
}

impl Default for PlotWindow {
    fn default() -> Self {
        Self {
            len_ms: 200.0,
            follow: true,
            auto_y: true,
        }
    }
}

impl PlotWindow {
    /// `[lo, hi]` the plot should show for newest-time `t_ms`:
    /// the trailing `len_ms` window ending at `t_ms` when following,
    /// else `None` (the user's own pan/zoom stays).
    pub fn bounds(&self, t_ms: f64) -> Option<(f64, f64)> {
        self.follow.then(|| (t_ms - self.len_ms, t_ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A watched-neuron wire row in the protocol-2 shape: `(id, per-slot
    /// series)` — slot 0 = v, slot 1 = u.
    fn watch_run(v: f32, u: f32) -> Vec<(u32, Vec<Vec<f32>>)> {
        vec![(7, vec![vec![v], vec![u]])]
    }

    /// Stage 13 A: feeding times 0..50 then 0..10 must leave only the
    /// second run's data — v/u samples AND raster bins.
    #[test]
    fn feed_clears_history_when_time_goes_backwards() {
        let mut h = PlotHist::default();
        // `borrow` re-slices an owned watch row for one feed call.
        fn borrowed<'a>(o: &'a [(u32, Vec<Vec<f32>>)]) -> Vec<(u32, &'a [Vec<f32>])> {
            o.iter().map(|(id, s)| (*id, s.as_slice())).collect()
        }
        // First "run": 50 ms, v = -60/u = -10, neuron 7 fires every bin.
        for t in 0..50 {
            h.feed(t as f64, vec![7], &borrowed(&watch_run(-60.0, -10.0)), 1.0);
        }
        assert_eq!(h.raster.len(), 50);
        // Second run rewinds: 10 ms, v = -70/u = -20.
        for t in 0..10 {
            h.feed(t as f64, vec![3], &borrowed(&watch_run(-70.0, -20.0)), 1.0);
        }
        assert_eq!(h.resets, 1);
        assert_eq!(h.raster.len(), 10, "only second-run bins remain");
        assert!(h.raster.iter().all(|(t, f)| *t < 10.0 && f == &[3]));
        let v = h.slot(0).unwrap().get(&7).unwrap();
        assert_eq!(v.len(), 10);
        assert!(v.iter().all(|(t, val)| *t < 10.0 && *val == -70.0));
        let u = h.slot(1).unwrap().get(&7).unwrap();
        assert_eq!(u.len(), 10);
        assert!(u.iter().all(|(t, val)| *t < 10.0 && *val == -20.0));
        // And time continuing forward afterwards appends normally.
        h.feed(10.0, vec![9], &borrowed(&watch_run(-70.0, -20.0)), 1.0);
        assert_eq!(h.raster.len(), 11);
        assert_eq!(h.resets, 1);
    }

    /// Stage 13 C: phase pairs (v,u) by sample index, honours the time
    /// window, and reports the newest point.
    #[test]
    fn phase_points_pairs_and_windows() {
        let mut vs = VecDeque::new();
        let mut us = VecDeque::new();
        for i in 0..10 {
            vs.push_back((i as f64, i as f32 * 10.0));
            us.push_back((i as f64, -(i as f32)));
        }
        let (pts, newest) = phase_points(&vs, &us, Some((4.0, 7.0)));
        assert_eq!(pts.len(), 4);
        assert_eq!(pts[0], [40.0, -4.0]);
        assert_eq!(newest, Some([70.0, -7.0]));
        // No window → everything.
        let (all, _) = phase_points(&vs, &us, None);
        assert_eq!(all.len(), 10);
        // Unequal lengths are tolerated (zip by count min).
        us.pop_front();
        let (pts, _) = phase_points(&vs, &us, None);
        assert_eq!(pts.len(), 9);
    }

    #[test]
    fn corner_label_detection() {
        let r = -90.0..=40.0;
        // a mark at the range floor collides; the next step up doesn't.
        let mk = |v: f64| egui_plot::GridMark {
            value: v,
            step_size: 20.0,
        };
        assert!(y_label_hits_corner(&mk(-90.0), &r));
        assert!(y_label_hits_corner(&mk(-85.0), &r));
        assert!(!y_label_hits_corner(&mk(-80.0), &r));
        assert!(!y_label_hits_corner(&mk(40.0), &r));
    }

    /// Stage 22B: Auto Y must never stretch a resting noise floor into
    /// a full-height fake spike — the fitted span never goes below
    /// AUTO_Y_MIN_FRAC of the descriptor range. A real excursion still
    /// fits the data.
    /// Sabotage: `min_span` forced to 0 (or `.max()` dropped) → the
    /// flat-data case returns a ~0.1 span and fails the bound.
    #[test]
    fn auto_y_clamps_noise_floor() {
        let range = (-90.0, 40.0); // izh v descriptor
        let min_span = (range.1 - range.0) * AUTO_Y_MIN_FRAC;
        // Flat resting data (±0.035 mV) → at least min_span, centred on it.
        let (lo, hi) = fit_auto_y(-70.18, -70.11, range);
        assert!(
            hi - lo >= min_span - 1e-6,
            "noise floor stretched to {:.3} — fakes a spike",
            hi - lo
        );
        assert!(lo < -70.18 && hi > -70.11, "data must stay inside");
        // A real ~90 mV swing fits the data, wider than min_span.
        let (lo, hi) = fit_auto_y(-80.0, 25.0, range);
        assert!(hi - lo > min_span * 3.0, "real spikes should zoom in");
        assert!(lo < -80.0 && hi > 25.0);
        // Degenerate single-point data still yields a sane span.
        let (lo, hi) = fit_auto_y(-65.0, -65.0, range);
        assert!(hi - lo >= min_span - 1e-6);
    }

    /// Stage 13 B: follow mode yields the trailing window; once off,
    /// no bounds are imposed (the user's pan/zoom is kept).
    #[test]
    fn window_bounds_follow_then_off() {
        let mut w = PlotWindow::default();
        assert_eq!(w.bounds(1000.0), Some((800.0, 1000.0)));
        w.follow = false;
        assert_eq!(w.bounds(1000.0), None);
        w.follow = true;
        w.len_ms = 50.0;
        assert_eq!(w.bounds(1000.0), Some((950.0, 1000.0)));
    }
}
