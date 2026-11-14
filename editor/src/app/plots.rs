//! Plots window: v(t), raster, v–u phase, readout weights, samples and
//! the jobs tab. Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;
use egui_plot::{Line, PlotPoints};

use crate::gpu::Entity;
use crate::icons;

use super::layout::{app_win, default_rect, narrow_mode, usable_rect, DefWin};
use super::{EditorApp, PlotTab};

impl EditorApp {
    pub(super) fn plots(&mut self, ctx: &egui::Context) {
        // Stage 22B: on the first laid-out frame of a narrow window the
        // floating Plots window stays closed — it would squeeze into
        // the ~260 px strip between the docks. The user (or the
        // inspector's "Open in Plots window") can still open it; on
        // wide windows nothing changes.
        if !self.win.narrow_defaults_done {
            if ctx.screen_rect().width() > 0.0 {
                self.win.narrow_defaults_done = true;
                if narrow_mode(ctx) {
                    self.win.plots = false;
                }
            }
        }
        if !self.win.plots {
            return;
        }
        let mut open = self.win.plots;
        let u = usable_rect(ctx);
        let dr = default_rect(DefWin::Plots, u);
        app_win(ctx, "Plots")
            .open(&mut open)
            .default_size(dr.size())
            .default_pos(dr.min)
            .show(ctx, |ui| {
                // Wrapped so the tab bar reflows instead of clipping
                // when the strip between the docks is narrower than
                // the row (~460px at 800px wide) — Stage 25.
                ui.horizontal_wrapped(|ui| {
                    if let Some(ic) = &self.icons {
                        // One tab per state slot of the plot model —
                        // labels and ranges ride the descriptors, so a
                        // new model gets its tabs for free.
                        let n_states = self.plot_n_states();
                        let names: Vec<String> =
                            (0..n_states.max(2)).map(|s| self.plot_state_name(s)).collect();
                        let t = &mut self.plots.tab;
                        for (s, name) in names.iter().enumerate().take(n_states) {
                            let lbl = format!("{name}(t)");
                            icons::selectable_value(ui, t, PlotTab::State(s), &ic.voltage, &lbl);
                        }
                        if n_states >= 2 {
                            let lbl = format!("{}\u{2013}{}", names[0], names[1]);
                            icons::selectable_value(ui, t, PlotTab::Phase, &ic.phase, &lbl);
                        }
                        icons::selectable_value(ui, t, PlotTab::Raster, &ic.raster, "raster");
                        icons::selectable_value(ui, t, PlotTab::Readouts, &ic.readout, "readouts");
                        icons::selectable_value(ui, t, PlotTab::Samples, &ic.samples, "samples");
                        icons::selectable_value(ui, t, PlotTab::Jobs, &ic.jobs, "jobs");
                    }
                    if self.plots.tab == PlotTab::Jobs && !self.jobs.list.is_empty() {
                        ui.separator();
                        ui.label(format!("{} job(s)", self.jobs.list.len()));
                    }
                    ui.separator();
                    if matches!(
                        self.plots.tab,
                        PlotTab::State(_) | PlotTab::Phase | PlotTab::Raster
                    ) {
                        ui.separator();
                        ui.label("window");
                        ui.add(
                            egui::DragValue::new(&mut self.plots.win.len_ms)
                                .range(20.0..=5000.0)
                                .suffix("ms"),
                        );
                        if !self.plots.win.follow {
                            if ui.button("Follow").clicked() {
                                self.plots.win.follow = true;
                            }
                        }
                        if matches!(self.plots.tab, PlotTab::State(_) | PlotTab::Phase) {
                            ui.checkbox(&mut self.plots.win.auto_y, "Auto Y");
                        }
                    }
                    if matches!(self.plots.tab, PlotTab::State(_) | PlotTab::Phase) {
                        let sel_n = self
                            .selection
                            .iter()
                            .filter(|&i| matches!(self.entity(*i), Some(Entity::Neuron(_))))
                            .count();
                        // U8: say which neurons the cap dropped.
                        ui.weak(format!(
                            "{} of {} watched",
                            self.watch_sent.len(),
                            sel_n
                        ));
                    }
                    if self.plots.tab == PlotTab::Raster {
                        ui.label("height");
                        ui.add(
                            egui::DragValue::new(&mut self.plots.raster_h)
                                .range(40.0..=600.0)
                                .suffix("px"),
                        );
                    }
                });
                match self.plots.tab {
                    PlotTab::State(slot) => {
                        // Sliding window ending at "now" while following;
                        // the user's pan/zoom rules once follow is off.
                        let win = self.plots.win.bounds(self.plots.last_sim_t);
                        let auto_y = self.plots.win.auto_y;
                        if self.watch_sent.is_empty() {
                            // U8: readable weak text above the plot, not
                            // tiny in-plot text across the centre line.
                            ui.weak("no watched neurons — click a neuron to \
                                     watch it, or right-click -> \"Watch v(t)\"");
                        }
                        let y_range = self.plot_state_range(slot);
                        let mut p = egui_plot::Plot::new("splot")
                            .height(140.0)
                            .y_axis_formatter(|y, range| {
                                if crate::plot::y_label_hits_corner(&y, range) {
                                    String::new()
                                } else {
                                    format!("{:.0}", y.value)
                                }
                            });
                        if win.is_none() {
                            // Panned away: egui fits both axes.
                            p = p.auto_bounds(egui::Vec2b::new(true, auto_y));
                        } else {
                            // Following: x is the follow window, y we set
                            // ourselves below (Auto Y goes through
                            // fit_auto_y's min-span clamp).
                            p = p.auto_bounds(egui::Vec2b::new(false, false));
                        }
                        let pr = p.show(ui, |ui| {
                            // watch_sent order → stable per-neuron colors,
                            // shared with the phase tab.
                            let mut dmin = f64::MAX;
                            let mut dmax = f64::MIN;
                            for (ci, &id) in self.watch_sent.iter().enumerate() {
                                let Some(h) = self
                                    .plots
                                    .hist
                                    .slot(slot)
                                    .and_then(|s| s.get(&id))
                                else {
                                    continue;
                                };
                                let pts_vec: Vec<[f64; 2]> = match win {
                                    Some((lo, hi)) => h
                                        .iter()
                                        .filter(|(t, _)| (lo..=hi).contains(t))
                                        .map(|(t, v)| [*t, *v as f64])
                                        .collect(),
                                    None => h.iter().map(|(t, v)| [*t, *v as f64]).collect(),
                                };
                                for p in &pts_vec {
                                    dmin = dmin.min(p[1]);
                                    dmax = dmax.max(p[1]);
                                }
                                ui.line(
                                    Line::new(self.watch_label(id), PlotPoints::from(pts_vec))
                                        .color(crate::plot::watch_color(ci)),
                                );
                            }
                            if let Some((lo, hi)) = win {
                                ui.set_plot_bounds_x(lo..=hi);
                                let (y0, y1) = if auto_y && dmin <= dmax {
                                    crate::plot::fit_auto_y(dmin, dmax, y_range)
                                } else {
                                    // Descriptor plot_range: spikes can't
                                    // bounce the axis entering/leaving.
                                    y_range
                                };
                                ui.set_plot_bounds_y(y0..=y1);
                            }
                        });
                        self.plot_follow(&pr, win);
                    }
                    PlotTab::Phase => {
                        // v–u phase portrait per watched neuron over the
                        // same window; the original had a Poincaré canvas
                        // for 2-D trajectory projections — no nullclines
                        // there or here (we don't have per-neuron params
                        // on the wire).
                        let win = self.plots.win.bounds(self.plots.last_sim_t);
                        let p = egui_plot::Plot::new("phplot")
                            .height(140.0)
                            .y_axis_formatter(|y, range| {
                                if crate::plot::y_label_hits_corner(&y, range) {
                                    String::new()
                                } else {
                                    format!("{:.0}", y.value)
                                }
                            })
                            .x_axis_label(self.plot_state_unit(0))
                            .y_axis_label(self.plot_state_unit(1));
                        let _ = p.show(ui, |ui| {
                            if self.plots.hist.slot(1).is_none_or(|s| s.is_empty()) {
                                ui.text(egui_plot::Text::new(
                                    "hint",
                                    egui_plot::PlotPoint::new(-30.0, -10.0),
                                    "no watched neurons — watch a neuron to see\nits v–u trajectory here",
                                ));
                            }
                            for (ci, &id) in self.watch_sent.iter().enumerate() {
                                let (Some(vs), Some(us)) = (
                                    self.plots.hist.slot(0).and_then(|s| s.get(&id)),
                                    self.plots.hist.slot(1).and_then(|s| s.get(&id)),
                                ) else {
                                    continue;
                                };
                                let col = crate::plot::watch_color(ci);
                                let (pts, newest) = crate::plot::phase_points(vs, us, win);
                                if pts.is_empty() {
                                    continue;
                                }
                                ui.line(
                                    Line::new(self.watch_label(id), PlotPoints::from(pts))
                                        .color(col),
                                );
                                if let Some(p) = newest {
                                    ui.points(
                                        egui_plot::Points::new("now", vec![p])
                                            .radius(4.0_f32)
                                            .color(col),
                                    );
                                }
                            }
                        });
                    }
                    PlotTab::Raster => {
                        // Raster strip: x = sim milliseconds. With watches
                        // set, each row is a watched neuron (labeled);
                        // without, y is raw neuron id (sampled).
                        let watches = self.watch_sent.clone();
                        let n_neurons =
                            self.geo.as_ref().map(|g| g.pos.len() as f64).unwrap_or(1.0);
                        // Always pin y: toggling the flag between watch and
                        // scatter modes stranded stale bounds in PlotMemory
                        // and hid the dots.
                        let y_hi = if watches.is_empty() {
                            n_neurons - 0.5
                        } else {
                            watches.len() as f64 - 0.5
                        };
                        let win = self.plots.win.bounds(self.plots.last_sim_t);
                        let plot = egui_plot::Plot::new("raster")
                            .height(self.plots.raster_h.max(40.0))
                            .auto_bounds(egui::Vec2b::new(win.is_none(), false))
                            .include_y(-0.5)
                            .include_y(y_hi);
                        let pr = plot
                            .x_axis_formatter(|x, _| format!("{:.0}ms", x.value))
                            .y_axis_formatter(move |y, range| {
                                if crate::plot::y_label_hits_corner(&y, range) {
                                    return String::new();
                                }
                                let k = y.value.round() as usize;
                                if watches.is_empty() {
                                    format!("{}", k)
                                } else {
                                    watches
                                        .get(k)
                                        .map(|id| format!("n{id}"))
                                        .unwrap_or_default()
                                }
                            })
                            .show(ui, |ui| {
                                if self.plots.hist.raster.is_empty() {
                                    let cx = win.map(|(lo, hi)| (lo + hi) / 2.0)
                                        .unwrap_or(self.plots.last_sim_t);
                                    ui.text(egui_plot::Text::new(
                                        "hint",
                                        egui_plot::PlotPoint::new(cx, 0.0),
                                        "run the simulation — each spike lands here",
                                    ));
                                }
                                let mut pts = Vec::new();
                                let watches = &self.watch_sent;
                                for &(t, ref fired) in &self.plots.hist.raster {
                                    if let Some((lo, hi)) = win {
                                        if !(lo..=hi).contains(&t) {
                                            continue;
                                        }
                                    }
                                    if watches.is_empty() {
                                        let stride = (fired.len() / 200).max(1);
                                        for (k, &id) in fired.iter().enumerate() {
                                            if k % stride == 0 {
                                                pts.push([t, id as f64]);
                                            }
                                        }
                                    } else {
                                        for (k, &id) in watches.iter().enumerate() {
                                            if fired.contains(&id) {
                                                pts.push([t, k as f64]);
                                            }
                                        }
                                    }
                                }
                                ui.points(
                                    egui_plot::Points::new("spikes", PlotPoints::from(pts))
                                        .radius(1.0_f32)
                                        .color(egui::Color32::LIGHT_BLUE),
                                );
                                if let Some((lo, hi)) = win {
                                    ui.set_plot_bounds_x(lo..=hi);
                                }
                            });
                        self.plot_follow(&pr, win);
                    }
                    PlotTab::Readouts => self.plot_readouts(ui),
                    PlotTab::Samples => self.plot_samples(ui),
                    PlotTab::Jobs => self.jobs_ui(ui),
                }
            });
        self.win.plots = open;
    }

    /// Turn follow off when the user pans/zooms away from the imposed
    /// window (bounds no longer match what we set), and back on when
    /// they double-click — egui_plot's own "reset" gesture.
    pub(super) fn plot_follow(
        &mut self,
        pr: &egui_plot::PlotResponse<()>,
        imposed: Option<(f64, f64)>,
    ) {
        if pr.response.double_clicked() {
            self.plots.win.follow = true;
            return;
        }
        if let Some((lo, hi)) = imposed {
            let b = pr.transform.bounds();
            let eps = (self.plots.win.len_ms * 1e-6).max(1e-6);
            if pr.response.dragged()
                || (b.min()[0] - lo).abs() > eps
                || (b.max()[0] - hi).abs() > eps
            {
                self.plots.win.follow = false;
            }
        }
    }

    /// Bar chart of the selected readout's trained weights (original's
    /// ReadOut weights visualization).
    pub(super) fn plot_readouts(&mut self, ui: &mut egui::Ui) {
        let Some(g) = self.geo.clone() else { return };
        if g.readouts.is_empty() {
            ui.label("no readouts — net.add_readout(\"ro\", units) in a script");
            return;
        }
        self.plots.ro_sel = self.plots.ro_sel.min(g.readouts.len() - 1);
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("ro_sel")
                .selected_text(g.readouts[self.plots.ro_sel].0.clone())
                .show_ui(ui, |ui| {
                    for (i, (name, _, _)) in g.readouts.iter().enumerate() {
                        ui.selectable_value(&mut self.plots.ro_sel, i, name);
                    }
                });
            let (_, units, w) = &g.readouts[self.plots.ro_sel];
            ui.label(format!("{} units, {} weights", units.len(), w.len()));
        });
        let (_, _, w) = &g.readouts[self.plots.ro_sel];
        if w.is_empty() {
            ui.label("no weights — push them from a script: net.set_readout_weights(\"name\", ro.weights())");
            return;
        }
        egui_plot::Plot::new("row").height(80.0).show(ui, |ui| {
            let bars: Vec<egui_plot::Bar> = w
                .iter()
                .enumerate()
                .map(|(i, &v)| egui_plot::Bar::new(i as f64, v as f64).width(0.8))
                .collect();
            ui.bar_chart(egui_plot::BarChart::new("w", bars));
        });
    }

    /// Time-series plot of `net.samples` (the original's samples plot).
    pub(super) fn plot_samples(&mut self, ui: &mut egui::Ui) {
        let Some(g) = self.geo.clone() else { return };
        if g.samples.is_empty() {
            ui.label("no samples — net.samples_write(\"s\", times, values) in a script");
            return;
        }
        egui_plot::Plot::new("smp").height(100.0).show(ui, |ui| {
            for (name, t, v) in g.samples.iter().take(8) {
                let pts: PlotPoints = t
                    .iter()
                    .zip(v.iter())
                    .map(|(&a, &b)| [a as f64, b as f64])
                    .collect();
                ui.line(Line::new(name.clone(), pts));
            }
        });
    }
}

impl EditorApp {
    /// The model whose state descriptors drive the plot tabs: the first
    /// watched neuron's model, else the first type in the net, else the
    /// first registered model. Descriptor-driven — no model names here.
    fn plot_model(&self) -> Option<&proto::ModelDesc> {
        let by_id = |id: &str| self.models.iter().find(|m| m.id == id);
        let g = self.geo.as_deref();
        if let Some(&w) = self.watch_sent.first() {
            let t = g.and_then(|g| {
                g.ty.get(w as usize)
                    .and_then(|&ty| g.types.get(ty as usize))
            });
            if let Some(m) = t.and_then(|t| by_id(&t.model)) {
                return Some(m);
            }
        }
        g.and_then(|g| g.types.first())
            .and_then(|t| by_id(&t.model))
            .or_else(|| self.models.first())
    }

    /// State slots the tabs cover — at least one so slot 0 (v) always
    /// shows even before `Hello.models` lands.
    fn plot_n_states(&self) -> usize {
        self.plot_model().map(|m| m.state.len()).unwrap_or(0).max(1)
    }

    fn plot_state_name(&self, s: usize) -> String {
        self.plot_model()
            .and_then(|m| m.state.get(s))
            .map(|d| d.name.clone())
            .unwrap_or_else(|| format!("s{s}"))
    }

    /// `name (unit)` for axis labels.
    fn plot_state_unit(&self, s: usize) -> String {
        match self.plot_model().and_then(|m| m.state.get(s)) {
            Some(d) if !d.unit.is_empty() => format!("{} ({})", d.name, d.unit),
            Some(d) => d.name.clone(),
            None => format!("s{s}"),
        }
    }

    /// Y range from the descriptor's `plot_range` (the fixed Izhikevich
    /// span is just what izh's slot 0 declares).
    fn plot_state_range(&self, s: usize) -> (f64, f64) {
        self.plot_model()
            .and_then(|m| m.state.get(s))
            .map(|d| (d.plot_range.0 as f64, d.plot_range.1 as f64))
            .unwrap_or(crate::plot::V_Y_RANGE)
    }
}
