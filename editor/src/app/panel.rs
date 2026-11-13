//! Fixed left dock (Stage 22): Tools / Run / View sections — the
//! Photoshop-style tool palette that took over the crowded top toolbar.
//! Resizable (drag the right edge), collapsible (the « button; a slim
//! rail with » and the tool icons stays), and scrolls inside itself at
//! small window heights — it never overlaps the scene.

use eframe::egui;

use crate::icons::{self, Icons};
use crate::scene::Plane;
use proto::ClientMsg;

use super::{gate, gate_ui, Act, EditorApp, Mode};

/// Minimum interact height for every control in the panel (Stage 22
/// requires ≥ 28 px at 1440×900). `interact_size` is set on the panel's
/// ui so buttons/selectables/DragValues all pick it up.
pub(super) const PANEL_CONTROL_H: f32 = 28.0;

/// Width of the collapsed tool rail.
pub(super) const RAIL_W: f32 = 34.0;

/// The four edit modes + their icons/labels, in panel order.
fn modes(ic: &Icons) -> [(Mode, &egui::TextureHandle, &str); 4] {
    [
        (Mode::Navigate, &ic.navigate, "navigate"),
        (Mode::Insert, &ic.insert, "insert"),
        (Mode::Connect, &ic.connect, "connect"),
        (Mode::Move, &ic.mv, "move"),
    ]
}

impl EditorApp {
    /// The fixed left dock. `win.left_panel` on → resizable SidePanel;
    /// off → a slim rail (» to reopen, tool icons still reachable).
    /// Shown between the menu bar and the bottom strips, so it runs the
    /// full working height.
    pub(super) fn left_panel(&mut self, ctx: &egui::Context) {
        let Some(ic) = self.icons.clone() else { return };
        let narrow = super::layout::narrow_mode(ctx);
        // Stage 22B: at narrow widths the dock starts as the rail —
        // egui grows a resizable SidePanel to its content min-width
        // (~232 px of tool rows), which with an open inspector leaves
        // <45 % scene at 800 px. The rail keeps every tool icon
        // reachable; "»" reopens the dock (defaulted slimmer, 170 px,
        // still user-draggable to 400 px).
        if self.win.left_panel && (!narrow || self.win.left_narrow_open) {
            let def_w = if narrow { 170.0 } else { 216.0 };
            egui::SidePanel::left("tools_panel")
                .resizable(true)
                .default_width(def_w)
                .width_range(150.0..=400.0)
                .show(ctx, |ui| {
                    ui.spacing_mut().interact_size.y = PANEL_CONTROL_H;
                    // Probe for `panel_controls_are_at_least_28px` — the
                    // test reads back what the dock really enforced.
                    #[cfg(test)]
                    ui.ctx().data_mut(|d| {
                        d.insert_temp(egui::Id::new("test_panel_ih"), ui.spacing().interact_size.y);
                    });
                    ui.horizontal(|ui| {
                        ui.heading("Tools");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("«").on_hover_text("collapse panel").clicked() {
                                if narrow {
                                    self.win.left_narrow_open = false;
                                } else {
                                    self.win.left_panel = false;
                                }
                            }
                        });
                    });
                    ui.separator();
                    // Never taller than the window: content scrolls inside.
                    egui::ScrollArea::vertical()
                        .id_salt("tools_scroll")
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            ui.spacing_mut().interact_size.y = PANEL_CONTROL_H;
                            self.tools_section(ui, &ic);
                            self.run_section(ui, &ic);
                            self.view_section(ui, &ic);
                        });
                });
        } else {
            egui::SidePanel::left("tools_rail")
                .resizable(false)
                .exact_width(RAIL_W)
                .show(ctx, |ui| {
                    ui.spacing_mut().interact_size.y = PANEL_CONTROL_H;
                    if ui.button("»").on_hover_text("expand panel").clicked() {
                        self.win.left_panel = true;
                        self.win.left_narrow_open = true;
                    }
                    ui.separator();
                    let mut shown = self.eff_mode();
                    let why_edit = self.why(Act::Edit);
                    for (m, t, name) in modes(&ic) {
                        let why = if m == Mode::Navigate { None } else { why_edit };
                        let mut r = gate(ui, why, icons::selectable(shown == m, t, ""));
                        if r.clicked() && shown != m {
                            shown = m;
                            r.mark_changed();
                        }
                        r.on_hover_text(name);
                    }
                    if shown != self.mode {
                        self.mode = shown;
                    }
                    ui.separator();
                    // The run toggle is the one control that must stay
                    // reachable even from the rail.
                    let live = self.sim.running || self.sim.busy;
                    let t = if live { &ic.stop } else { &ic.run };
                    if gate(ui, None, icons::icon_button(t))
                        .on_hover_text(if live { "Stop" } else { "Run" })
                        .clicked()
                    {
                        if live {
                            self.ctl.abort_now();
                            self.send(ClientMsg::AutoRun(false));
                        } else {
                            self.send(ClientMsg::AutoRun(true));
                        }
                    }
                });
        }
    }

    fn tools_section(&mut self, ui: &mut egui::Ui, ic: &Icons) {
        ui.label(egui::RichText::new("TOOLS").weak().small());
        // 2×2 big buttons; the selected marker follows eff_mode (the
        // run/script fallback to Navigate), the stored mode only moves
        // on an enabled click.
        let mut shown = self.eff_mode();
        let why_edit = self.why(Act::Edit);
        let w = ((ui.available_width() - 8.0) / 2.0).max(48.0);
        egui::Grid::new("mode_grid")
            .num_columns(2)
            .spacing([4.0, 4.0])
            .show(ui, |ui| {
                for (k, (m, t, name)) in modes(ic).into_iter().enumerate() {
                    let why = if m == Mode::Navigate { None } else { why_edit };
                    let mut r = gate(
                        ui,
                        why,
                        icons::selectable(shown == m, t, name).min_size(egui::vec2(w, 32.0)),
                    );
                    if r.clicked() && shown != m {
                        shown = m;
                        r.mark_changed();
                    }
                    if k % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
        if shown != self.mode {
            self.mode = shown;
        }
        // Work plane: it steers Insert/Move — part of the toolbox.
        ui.horizontal(|ui| {
            ui.label("plane");
            let p = &mut self.plane;
            icons::selectable_value(ui, p, Plane::XZ, &ic.plane_xz, "x-z");
            icons::selectable_value(ui, p, Plane::XY, &ic.plane_xy, "x-y");
            icons::selectable_value(ui, p, Plane::YZ, &ic.plane_yz, "y-z");
        });
        ui.add_space(4.0);
        ui.separator();
    }

    fn run_section(&mut self, ui: &mut egui::Ui, ic: &Icons) {
        ui.label(egui::RichText::new("RUN").weak().small());
        // The state band — the "is it alive" readout: colour + sim time,
        // always visible above the buttons.
        let t_ms = self.ctl.shared.read().unwrap().snap.time_ms;
        let (band, label, dark) = if self.sim.busy {
            (
                egui::Color32::from_rgb(196, 148, 40),
                "script running…".to_string(),
                true,
            )
        } else if self.sim.running {
            (
                egui::Color32::from_rgb(46, 160, 67),
                format!("RUNNING   t = {:.1} s", t_ms / 1000.0),
                true,
            )
        } else if self.sim.sim_built {
            (
                egui::Color32::from_rgb(80, 90, 105),
                format!("paused   t = {:.1} s", t_ms / 1000.0),
                true,
            )
        } else {
            (
                ui.visuals().faint_bg_color,
                "build — no sim yet".to_string(),
                false,
            )
        };
        egui::Frame::new()
            .fill(band)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .corner_radius(4.0)
            .show(ui, |ui| {
                let w = ui.available_width();
                ui.set_width(w);
                let col = if dark {
                    egui::Color32::BLACK
                } else {
                    ui.visuals().weak_text_color()
                };
                ui.label(egui::RichText::new(label).color(col).strong().monospace());
            });

        // Run / Stop: the big one. Never gated — it's how you leave Run
        // and how you abort a busy script.
        let live = self.sim.running || self.sim.busy;
        let (t, label) = if live {
            (&ic.stop, "STOP")
        } else {
            (&ic.run, "RUN")
        };
        let w = ui.available_width();
        if ui
            .add(icons::button(t, label).min_size(egui::vec2(w, 36.0)))
            .clicked()
        {
            if live {
                self.ctl.abort_now();
                self.send(ClientMsg::AutoRun(false));
            } else {
                self.send(ClientMsg::AutoRun(true));
            }
        }

        let why_step = self.why(Act::Step);
        let why_reset = self.why(Act::Reset);
        ui.horizontal(|ui| {
            let w = (ui.available_width() - 2.0 * ui.spacing().item_spacing.x) / 3.0;
            if gate(
                ui,
                why_step,
                icons::button(&ic.step, "10 ms").min_size(egui::vec2(w, 28.0)),
            )
            .clicked()
            {
                self.send(ClientMsg::RunMs(10.0));
            }
            if gate(
                ui,
                why_step,
                icons::button(&ic.step_long, "100 ms").min_size(egui::vec2(w, 28.0)),
            )
            .clicked()
            {
                self.send(ClientMsg::RunMs(100.0));
            }
            if gate(
                ui,
                why_reset,
                icons::button(&ic.reset, "reset").min_size(egui::vec2(w, 28.0)),
            )
            .clicked()
            {
                self.send(ClientMsg::Reset);
                self.plots.hist.clear();
            }
        });

        ui.horizontal(|ui| {
            ui.add(icons::tinted(&ic.speed, ui.visuals().text_color()))
                .on_hover_text("speed (realtime factor)");
            let changed = gate_ui(ui, self.why(Act::Speed), |ui| {
                ui.add(
                    egui::Slider::new(&mut self.sim.speed, 0.05..=20.0)
                        .logarithmic(true)
                        .text("x"),
                )
                .changed()
            })
            .inner;
            if changed {
                self.send(ClientMsg::SetSpeed(self.sim.speed));
            }
        });
        let max_x = self.sim.speed_peak * self.sim.dt_ms / 1000.0;
        ui.weak(if max_x > 0.05 {
            format!("x{:.1} realtime (max x{:.0})", self.sim.speed, max_x)
        } else {
            format!("x{:.1} realtime", self.sim.speed)
        });

        ui.horizontal(|ui| {
            ui.label("backend");
            gate_ui(ui, self.why(Act::Backend), |ui| {
                egui::ComboBox::from_id_salt("p_backend")
                    .selected_text(match self.sim.backend {
                        proto::Backend::Cuda => "cuda",
                        proto::Backend::Cpu => "cpu",
                    })
                    .show_ui(ui, |ui| {
                        if ui.button("cuda").clicked() {
                            self.send(ClientMsg::SetBackend(proto::Backend::Cuda));
                        }
                        if ui.button("cpu").clicked() {
                            self.send(ClientMsg::SetBackend(proto::Backend::Cpu));
                        }
                    });
            });
        });
        ui.horizontal(|ui| {
            ui.label("dt");
            let changed = gate_ui(ui, self.why(Act::Dt), |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.sim.dt_ms)
                        .range(0.01..=1.0)
                        .speed(0.01)
                        .suffix(" ms"),
                )
                .changed()
            })
            .inner;
            if changed {
                self.send(ClientMsg::SetDtMs(self.sim.dt_ms));
            }
        });
        ui.horizontal(|ui| {
            ui.label("noise");
            let changed = gate_ui(ui, self.why(Act::Noise), |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.sim.noise_hz)
                        .range(0.0..=5000.0)
                        .speed(1.0)
                        .suffix(" Hz"),
                )
                .changed()
            })
            .inner;
            if changed {
                self.send(ClientMsg::SetNoise {
                    hz: self.sim.noise_hz,
                    kick: self.sim.noise_kick,
                });
            }
        });
        ui.add_space(4.0);
        ui.separator();
    }

    fn view_section(&mut self, ui: &mut egui::Ui, ic: &Icons) {
        ui.label(egui::RichText::new("VIEW").weak().small());
        let w = ((ui.available_width() - 8.0) / 2.0).max(48.0);
        egui::Grid::new("view_grid")
            .num_columns(2)
            .spacing([4.0, 4.0])
            .show(ui, |ui| {
                for (k, (flag, t, name)) in [
                    (&mut self.show_synapses, &ic.synapses, "synapses"),
                    (&mut self.show_input_links, &ic.connect, "input links"),
                    (&mut self.show_grid, &ic.grid, "grid"),
                    (&mut self.show_dots, &ic.raster, "spike dots"),
                ]
                .into_iter()
                .enumerate()
                {
                    let mut r =
                        ui.add(icons::selectable(*flag, t, name).min_size(egui::vec2(w, 26.0)));
                    if r.clicked() {
                        *flag = !*flag;
                        r.mark_changed();
                    }
                    if k % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
        ui.horizontal(|ui| {
            ui.label("grid view");
            egui::ComboBox::from_id_salt("p_grid_view")
                .selected_text(match self.grid_view {
                    crate::plane::GridView::Auto => "auto",
                    crate::plane::GridView::Cubes => "cubes",
                    crate::plane::GridView::Planes => "planes",
                })
                .show_ui(ui, |ui| {
                    use crate::plane::GridView;
                    let g = &mut self.grid_view;
                    ui.selectable_value(g, GridView::Auto, "auto");
                    ui.selectable_value(g, GridView::Cubes, "cubes");
                    ui.selectable_value(g, GridView::Planes, "planes");
                });
        });
        if ui
            .button("frame network (F)")
            .on_hover_text("fit the camera on the whole net")
            .clicked()
        {
            self.frame_all();
        }
        ui.add_space(4.0);
        ui.label(egui::RichText::new("PANELS").weak().small());
        // Window/dock toggles — the console's home is the View section.
        ui.horizontal_wrapped(|ui| {
            for (flag, name) in [
                (&mut self.win.console, "console"),
                (&mut self.win.inspector, "inspector"),
                (&mut self.win.plots, "plots"),
                (&mut self.win.inputs, "inputs"),
                (&mut self.win.script, "script"),
                (&mut self.win.files, "files"),
            ] {
                let mut r = ui.add(egui::Button::selectable(*flag, name));
                if r.clicked() {
                    *flag = !*flag;
                    r.mark_changed();
                }
            }
            let mut r = ui.add(egui::Button::selectable(self.show_fps, "fps"));
            if r.clicked() {
                self.show_fps = !self.show_fps;
                r.mark_changed();
            }
        });
        ui.add_space(4.0);
        ui.separator();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simctl::SimCtl;

    fn app(ctx: &egui::Context) -> EditorApp {
        let mut a = EditorApp::new(SimCtl::offline(), None);
        a.icons = Some(Icons::load(ctx));
        a
    }

    fn input(w: f32, h: f32) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(w, h),
            )),
            ..Default::default()
        }
    }

    /// The dock carves a bounded strip off the left edge at both required
    /// window sizes — the scene keeps almost the whole window. Two-sided
    /// assert: too narrow = the dock never showed, too wide = it ate the
    /// viewport. Stage 22B: at narrow widths the DEFAULT is the rail
    /// (~44 px, icons still reachable); the user-opened dock is bounded
    /// as before.
    /// Sabotage: `default_width(700.0)` or `width_range(150.0..=1200.0)`
    /// shrinks the scene below the bound; an empty `left_panel` fails the
    /// lower bound. Narrow sabotage (verified): drop the
    /// `!narrow || left_narrow_open` condition → 800 px takes ~232 px of
    /// dock and the rail bound fails.
    #[test]
    fn left_panel_leaves_viewport_room() {
        for (w, h) in [(800.0_f32, 700.0_f32), (1440.0, 900.0)] {
            // Default state first: wide → dock, narrow → rail.
            let ctx = egui::Context::default();
            let mut a = app(&ctx);
            let mut avail = egui::Rect::ZERO;
            for _ in 0..3 {
                let _ = ctx.run(input(w, h), |ctx| {
                    a.left_panel(ctx);
                    avail = ctx.available_rect();
                });
            }
            let taken = w - avail.width();
            if crate::app::layout::narrow_mode(&ctx) {
                assert!(
                    taken <= 60.0,
                    "narrow default should be the rail, took {taken:.0}px at {w}x{h}"
                );
            } else {
                assert!(
                    (150.0..=400.0).contains(&taken),
                    "left dock took {taken:.0}px at {w}x{h}"
                );
            }
            assert!(
                avail.height() >= h - 1.0,
                "left dock stole vertical space: {avail:?} at {w}x{h}"
            );
            // User-opened at narrow → bounded dock.
            let ctx = egui::Context::default();
            let mut a = app(&ctx);
            a.win.left_narrow_open = true;
            let mut avail = egui::Rect::ZERO;
            for _ in 0..3 {
                let _ = ctx.run(input(w, h), |ctx| {
                    a.left_panel(ctx);
                    avail = ctx.available_rect();
                });
            }
            let taken = w - avail.width();
            assert!(
                (150.0..=400.0).contains(&taken),
                "opened left dock took {taken:.0}px at {w}x{h}"
            );
        }
    }

    /// Collapsed, the dock is a slim rail — the scene gets everything
    /// back but the ~34 px strip.
    /// Sabotage: render the full panel regardless of `win.left_panel`
    /// → `taken` lands at the panel width (~216), over the bound.
    #[test]
    fn collapsed_dock_is_a_slim_rail() {
        let ctx = egui::Context::default();
        let mut a = app(&ctx);
        a.win.left_panel = false;
        let mut avail = egui::Rect::ZERO;
        for _ in 0..3 {
            let _ = ctx.run(input(800.0, 700.0), |ctx| {
                a.left_panel(ctx);
                avail = ctx.available_rect();
            });
        }
        // egui adds its panel frame margins to `exact_width` — the rail
        // lands at ~44 px, still an order of magnitude under the panel.
        assert!(
            avail.width() >= 800.0 - 60.0,
            "collapsed dock wider than the rail: {avail:?}"
        );
    }

    /// Every control inside the dock is ≥ 28 px interact height
    /// (Stage 22, 1440×900). The probe in `left_panel` reports the
    /// enforced `interact_size`, so this checks the real code path, not
    /// just the constant.
    /// Sabotage: drop the `interact_size.y` override (or set it to 20)
    /// → the probe reads egui's default (~20 px) and this fails.
    #[test]
    fn panel_controls_are_at_least_28px() {
        let ctx = egui::Context::default();
        let mut a = app(&ctx);
        for _ in 0..3 {
            let _ = ctx.run(input(1440.0, 900.0), |ctx| a.left_panel(ctx));
        }
        let ih = ctx.data_mut(|d| d.get_temp::<f32>(egui::Id::new("test_panel_ih")));
        assert!(
            ih.unwrap_or(0.0) >= 28.0,
            "dock interact height = {ih:?} — controls too small"
        );
    }

    /// The top bar is menus + file name + status only — one row, under
    /// 40 px tall at 1440×900.
    /// Sabotage: re-adding the run-controls `horizontal_wrapped` row to
    /// `toolbar()` doubles the strip's height (~60 px) and this fails.
    #[test]
    fn top_bar_is_menus_only() {
        let ctx = egui::Context::default();
        let mut a = app(&ctx);
        let mut strip = 0.0f32;
        for _ in 0..3 {
            let _ = ctx.run(input(1440.0, 900.0), |ctx| {
                a.toolbar(ctx);
                strip = 900.0 - ctx.available_rect().height();
            });
        }
        assert!(strip > 0.0, "top bar never showed");
        assert!(strip < 40.0, "top bar grew back to {strip:.0}px");
    }
}
