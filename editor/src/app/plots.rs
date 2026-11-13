//! Plots window: v(t), raster, v–u phase, readout weights, samples and
//! the jobs tab. Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;
use egui_plot::{Line, PlotPoints};

use crate::gpu::Entity;
use crate::icons;

use super::layout::{app_win, plots_default_rect, usable_rect};
use super::{EditorApp, PlotTab};

impl EditorApp {
    pub(super) fn plots(&mut self, ctx: &egui::Context) {
        if !self.show_plots {
            return;
        }
        let mut open = self.show_plots;
        let u = usable_rect(ctx);
        let dr = plots_default_rect(u);
        app_win(ctx, "Plots")
            .open(&mut open)
            .default_size(dr.size())
            .default_pos(dr.min)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(ic) = &self.icons {
                        let t = &mut self.plot_tab;
                        icons::selectable_value(ui, t, PlotTab::Voltage, &ic.voltage, "v(t)");
                        icons::selectable_value(ui, t, PlotTab::Phase, &ic.phase, "v–u");
                        icons::selectable_value(ui, t, PlotTab::Raster, &ic.raster, "raster");
                        icons::selectable_value(ui, t, PlotTab::Readouts, &ic.readout, "readouts");
                        icons::selectable_value(ui, t, PlotTab::Samples, &ic.samples, "samples");
                        icons::selectable_value(ui, t, PlotTab::Jobs, &ic.jobs, "jobs");
                    }
                    if self.plot_tab == PlotTab::Jobs && !self.jobs.is_empty() {
                        ui.separator();
                        ui.label(format!("{} job(s)", self.jobs.len()));
                    }
                    ui.separator();
                    if matches!(self.plot_tab, PlotTab::Voltage | PlotTab::Phase | PlotTab::Raster) {
                        ui.separator();
                        ui.label("window");
                        ui.add(
                            egui::DragValue::new(&mut self.plot_win.len_ms)
                                .range(20.0..=5000.0)
                                .suffix("ms"),
                        );
                        if !self.plot_win.follow {
                            if ui.button("Follow").clicked() {
                                self.plot_win.follow = true;
                            }
                        }
                        if matches!(self.plot_tab, PlotTab::Voltage | PlotTab::Phase) {
                            ui.checkbox(&mut self.plot_win.auto_y, "Auto Y");
                        }
                    }
                    if matches!(self.plot_tab, PlotTab::Voltage | PlotTab::Phase) {
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
                    if self.plot_tab == PlotTab::Raster {
                        ui.label("height");
                        ui.add(
                            egui::DragValue::new(&mut self.raster_h)
                                .range(40.0..=600.0)
                                .suffix("px"),
                        );
                    }
                });
                match self.plot_tab {
                    PlotTab::Voltage => {
                        // Sliding window ending at "now" while following;
                        // the user's pan/zoom rules once follow is off.
                        let win = self.plot_win.bounds(self.last_sim_t);
                        let auto_y = self.plot_win.auto_y;
                        if self.watch_sent.is_empty() {
                            // U8: readable weak text above the plot, not
                            // tiny in-plot text across the centre line.
                            ui.weak("no watched neurons — click a neuron to \
                                     watch it, or right-click -> \"Watch v(t)\"");
                        }
                        let mut p = egui_plot::Plot::new("vplot")
                            .height(140.0)
                            .y_axis_formatter(|y, range| {
                                if crate::plot::y_label_hits_corner(&y, range) {
                                    String::new()
                                } else {
                                    format!("{:.0}", y.value)
                                }
                            });
                        if win.is_none() || auto_y {
                            // x (or y too when following) from the data.
                            p = p.auto_bounds(egui::Vec2b::new(win.is_none(), auto_y));
                        } else {
                            p = p.auto_bounds(egui::Vec2b::new(false, false));
                        }
                        let pr = p.show(ui, |ui| {
                            if self.hist.v.is_empty() {
                                // Discoverable watching (Stage 13 E): the
                            }
                            // watch_sent order → stable per-neuron colors,
                            // shared with the phase tab.
                            for (ci, &id) in self.watch_sent.iter().enumerate() {
                                let Some(h) = self.hist.v.get(&id) else { continue };
                                let pts: PlotPoints = match win {
                                    Some((lo, hi)) => h
                                        .iter()
                                        .filter(|(t, _)| (lo..=hi).contains(t))
                                        .map(|(t, v)| [*t, *v as f64])
                                        .collect(),
                                    None => h.iter().map(|(t, v)| [*t, *v as f64]).collect(),
                                };
                                ui.line(
                                    Line::new(self.watch_label(id), pts)
                                        .color(crate::plot::watch_color(ci)),
                                );
                            }
                            if let Some((lo, hi)) = win {
                                ui.set_plot_bounds_x(lo..=hi);
                                if !auto_y {
                                    // Fixed Izhikevich span: spikes can't
                                    // bounce the axis entering/leaving.
                                    ui.set_plot_bounds_y(
                                        crate::plot::V_Y_RANGE.0..=crate::plot::V_Y_RANGE.1,
                                    );
                                }
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
                        let win = self.plot_win.bounds(self.last_sim_t);
                        let p = egui_plot::Plot::new("phplot")
                            .height(140.0)
                            .y_axis_formatter(|y, range| {
                                if crate::plot::y_label_hits_corner(&y, range) {
                                    String::new()
                                } else {
                                    format!("{:.0}", y.value)
                                }
                            })
                            .x_axis_label("v (mV)")
                            .y_axis_label("u");
                        let _ = p.show(ui, |ui| {
                            if self.hist.u.is_empty() {
                                ui.text(egui_plot::Text::new(
                                    "hint",
                                    egui_plot::PlotPoint::new(-30.0, -10.0),
                                    "no watched neurons — watch a neuron to see\nits v–u trajectory here",
                                ));
                            }
                            for (ci, &id) in self.watch_sent.iter().enumerate() {
                                let (Some(vs), Some(us)) =
                                    (self.hist.v.get(&id), self.hist.u.get(&id))
                                else {
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
                        let win = self.plot_win.bounds(self.last_sim_t);
                        let plot = egui_plot::Plot::new("raster")
                            .height(self.raster_h.max(40.0))
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
                                if self.hist.raster.is_empty() {
                                    let cx = win.map(|(lo, hi)| (lo + hi) / 2.0)
                                        .unwrap_or(self.last_sim_t);
                                    ui.text(egui_plot::Text::new(
                                        "hint",
                                        egui_plot::PlotPoint::new(cx, 0.0),
                                        "run the simulation — each spike lands here",
                                    ));
                                }
                                let mut pts = Vec::new();
                                let watches = &self.watch_sent;
                                for &(t, ref fired) in &self.hist.raster {
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
        self.show_plots = open;
    }

    /// Turn follow off when the user pans/zooms away from the imposed
    /// window (bounds no longer match what we set), and back on when
    /// they double-click — egui_plot's own "reset" gesture.
    pub(super) fn plot_follow(&mut self, pr: &egui_plot::PlotResponse<()>, imposed: Option<(f64, f64)>) {
        if pr.response.double_clicked() {
            self.plot_win.follow = true;
            return;
        }
        if let Some((lo, hi)) = imposed {
            let b = pr.transform.bounds();
            let eps = (self.plot_win.len_ms * 1e-6).max(1e-6);
            if pr.response.dragged()
                || (b.min()[0] - lo).abs() > eps
                || (b.max()[0] - hi).abs() > eps
            {
                self.plot_win.follow = false;
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
        self.ro_sel = self.ro_sel.min(g.readouts.len() - 1);
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("ro_sel")
                .selected_text(g.readouts[self.ro_sel].0.clone())
                .show_ui(ui, |ui| {
                    for (i, (name, _, _)) in g.readouts.iter().enumerate() {
                        ui.selectable_value(&mut self.ro_sel, i, name);
                    }
                });
            let (_, units, w) = &g.readouts[self.ro_sel];
            ui.label(format!("{} units, {} weights", units.len(), w.len()));
        });
        let (_, _, w) = &g.readouts[self.ro_sel];
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
