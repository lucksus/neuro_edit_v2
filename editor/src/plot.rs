//! Plot history state: per-watch voltage traces + the spike raster,
//! plus the sliding-window/follow math shared by the plot tabs.
//!
//! Everything here is indexed by *simulation* milliseconds, which is
//! why `feed` watches for time going backwards — the Reset button, a
//! script's `sim.reset()`, or another client on a shared server can all
//! rewind the clock, and a trace indexed by the old run must not draw
//! on top of the new one.

use std::collections::{HashMap, VecDeque};

/// Raster keeps the newest N published bins.
const RASTER_MAX: usize = 600;
/// Per-neuron sample cap (at 1 ms steps, ~4 s of history).
const V_HIST_MAX: usize = 4096;

/// Watched-neuron v histories plus the spike raster.
pub struct PlotHist {
    /// neuron id → (t_ms, v) samples, oldest first.
    pub v: HashMap<u32, VecDeque<(f64, f32)>>,
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
            v: HashMap::new(),
            raster: VecDeque::with_capacity(64),
            last_t: f64::NEG_INFINITY,
            resets: 0,
        }
    }
}

impl PlotHist {
    /// Fold one snapshot into the histories. `watched` is
    /// `(neuron id, per-step v samples ending at t_ms)` — the samples
    /// are walked back by `dt_ms` so their x axis stays in sim ms.
    ///
    /// When `t_ms` is earlier than the last fed time the sim ran
    /// backwards: clear both histories so only the new run draws.
    pub fn feed(&mut self, t_ms: f64, fired: Vec<u32>, watched: &[(u32, &[f32])], dt_ms: f64) {
        if t_ms < self.last_t {
            self.v.clear();
            self.raster.clear();
            self.resets += 1;
        }
        self.last_t = t_ms;
        self.raster.push_back((t_ms, fired));
        while self.raster.len() > RASTER_MAX {
            self.raster.pop_front();
        }
        for &(id, vals) in watched {
            let h = self.v.entry(id).or_default();
            let n = vals.len();
            for (i, &v) in vals.iter().enumerate() {
                let t = t_ms - (n - 1 - i) as f64 * dt_ms;
                h.push_back((t, v));
            }
            while h.len() > V_HIST_MAX {
                h.pop_front();
            }
        }
    }

    /// Net was rebuilt or Reset was pressed locally — drop everything
    /// including the clock, so the next feed can't trip the reversal
    /// check on an unrelated earlier sim.
    pub fn clear(&mut self) {
        self.v.clear();
        self.raster.clear();
        self.last_t = f64::NEG_INFINITY;
    }
}

/// Fixed y range for the v(t) plot — the working span of the
/// Izhikevich model. Kept fixed so a spike entering or leaving the
/// window can't bounce the axis; "Auto Y" opts out per tab.
pub const V_Y_RANGE: (f64, f64) = (-90.0, 40.0);

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
            auto_y: false,
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

    fn watch_run(vals: &[f32]) -> Vec<(u32, &[f32])> {
        vec![(7, vals)]
    }

    /// Stage 13 A: feeding times 0..50 then 0..10 must leave only the
    /// second run's data — v samples AND raster bins.
    #[test]
    fn feed_clears_history_when_time_goes_backwards() {
        let mut h = PlotHist::default();
        // First "run": 50 ms, v always -60, neuron 7 fires every bin.
        for t in 0..50 {
            h.feed(t as f64, vec![7], &watch_run(&[-60.0]), 1.0);
        }
        assert_eq!(h.raster.len(), 50);
        // Second run rewinds: 10 ms, v always -70.
        for t in 0..10 {
            h.feed(t as f64, vec![3], &watch_run(&[-70.0]), 1.0);
        }
        assert_eq!(h.resets, 1);
        assert_eq!(h.raster.len(), 10, "only second-run bins remain");
        assert!(h.raster.iter().all(|(t, f)| *t < 10.0 && f == &[3]));
        let v = h.v.get(&7).unwrap();
        assert_eq!(v.len(), 10);
        assert!(v.iter().all(|(t, val)| *t < 10.0 && *val == -70.0));
        // And time continuing forward afterwards appends normally.
        h.feed(10.0, vec![9], &watch_run(&[-70.0]), 1.0);
        assert_eq!(h.raster.len(), 11);
        assert_eq!(h.resets, 1);
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
