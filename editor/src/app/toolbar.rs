//! Top toolbar, bottom status/hint bars, and the disconnect/reconnect
//! banner. Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;

use crate::gpu::Entity;
use crate::icons;
use crate::scene::Plane;
use crate::simctl::SimCtl;
use proto::ClientMsg;

use super::layout::usable_rect;
use super::{Dialog, EditorApp, Mode, PendingAction};

impl EditorApp {
    pub(super) fn toolbar(&mut self, ctx: &egui::Context) {
        let ic = match &self.icons {
            Some(i) => i.clone(),
            None => return,
        };
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            // Wrapped, not plain horizontal: a plain row that overflows
            // grows the panel's max_rect past the window, which makes the
            // sibling `horizontal_wrapped` mode row see an over-wide
            // boundary and never wrap — controls clip off-screen instead.
            ui.horizontal_wrapped(|ui| {
                ui.menu_button("File", |ui| {
                    if ui.add(icons::button(&ic.new, "New")).clicked() {
                        self.guarded(ctx, PendingAction::New);
                        ui.close();
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if self.login.saved_token.is_some() && ui.button("Log out").clicked() {
                        self.login.saved_token = None;
                        self.send(ClientMsg::Bye);
                        self.status = "logged out — saved session forgotten".into();
                        ui.close();
                    }
                    if ui.add(icons::button(&ic.open, "Load…")).clicked() {
                        self.dlg = Some(Dialog::Load {
                            path: self.net_path.clone().unwrap_or_default(),
                        });
                        ui.close();
                    }
                    ui.menu_image_text_button(
                        icons::tinted(&ic.recent, ui.visuals().text_color()),
                        "Recent files",
                        |ui| {
                            if self.files.recent.is_empty() {
                                ui.label("(empty)");
                            }
                            let rec: Vec<String> = self.files.recent.iter().cloned().collect();
                            for p in rec {
                                if ui.button(&p).clicked() {
                                    self.guarded(ctx, PendingAction::Load(p));
                                    ui.close();
                                }
                            }
                        },
                    );
                    if ui.add(icons::button(&ic.save, "Save")).clicked() {
                        match &self.net_path {
                            // Q6: ask for the file; saved_version/recent
                            // follow the server's Saved ack, not the ask.
                            Some(p) => {
                                let p = p.clone();
                                self.send(ClientMsg::SaveNet { path: p });
                            }
                            None => {
                                self.dlg = Some(Dialog::SaveAs {
                                    path: "net.ne2".into(),
                                });
                            }
                        }
                        ui.close();
                    }
                    if ui.add(icons::button(&ic.save_as, "Save as…")).clicked() {
                        self.dlg = Some(Dialog::SaveAs {
                            path: self.net_path.clone().unwrap_or("net.ne2".into()),
                        });
                        ui.close();
                    }
                    ui.separator();
                    if ui.add(icons::button(&ic.quit, "Quit")).clicked() {
                        self.guarded(ctx, PendingAction::Quit);
                    }
                });
                ui.menu_button("Edit", |ui| {
                    let sel = !self.selection.is_empty();
                    if ui
                        .add_enabled(sel, egui::Button::new("Copy (Ctrl+C)"))
                        .clicked()
                    {
                        self.copy_selection(false);
                        ui.close();
                    }
                    if ui
                        .add_enabled(sel, egui::Button::new("Cut (Ctrl+X)"))
                        .clicked()
                    {
                        self.copy_selection(true);
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            self.clipboard.is_some(),
                            egui::Button::new("Paste (Ctrl+V)"),
                        )
                        .clicked()
                    {
                        self.paste(None);
                        ui.close();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(sel, egui::Button::new("Delete selected (Del)"))
                        .clicked()
                    {
                        self.delete_selection();
                        ui.close();
                    }
                    if ui
                        .add_enabled(sel, egui::Button::new("Add selection to group…"))
                        .clicked()
                    {
                        self.dlg = Some(Dialog::AddToGroup {
                            name: "group".into(),
                        });
                        ui.close();
                    }
                });
                ui.menu_button("Insert", |ui| {
                    if ui.button("Neuron (I)").clicked() {
                        self.mode = Mode::Insert;
                        ui.close();
                    }
                    if ui.button("2D grid…").clicked() {
                        self.dlg = Some(Dialog::Grid2d {
                            group: "grid".into(),
                            w: 20,
                            h: 20,
                            d: 30.0,
                        });
                        ui.close();
                    }
                    if ui.button("3D grid…").clicked() {
                        self.dlg = Some(Dialog::Grid3d {
                            group: "grid".into(),
                            w: 10,
                            h: 10,
                            d2: 10,
                            spacing: 30.0,
                        });
                        ui.close();
                    }
                    if ui.button("Input…").clicked() {
                        self.dlg = Some(Dialog::NewInput {
                            name: "input".into(),
                            weight: 20.0,
                            value: 1.0,
                        });
                        ui.close();
                    }
                    if ui.button("Samples…").clicked() {
                        self.dlg = Some(Dialog::NewSamples {
                            name: "samples".into(),
                        });
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            !self.selection.is_empty(),
                            egui::Button::new("Readout from selection…"),
                        )
                        .clicked()
                    {
                        self.dlg = Some(Dialog::NewReadout { name: "ro".into() });
                        ui.close();
                    }
                });
                ui.menu_button("Group", |ui| {
                    if ui.button("Create connections…").clicked() {
                        self.dlg = Some(Dialog::Connections {
                            group: "grid".into(),
                            scaling: 0.35,
                            lambda: 120.0,
                            w: 6.0,
                        });
                        ui.close();
                    }
                    if ui.button("Set synapse weights…").clicked() {
                        self.dlg = Some(Dialog::Weights {
                            group: "grid".into(),
                            mean: 6.0,
                            sd: 1.0,
                        });
                        ui.close();
                    }
                    if ui.button("Set synapse tau…").clicked() {
                        self.dlg = Some(Dialog::SynTau {
                            src: "grid".into(),
                            dst: "grid".into(),
                            mean: 2.0,
                            sd: 0.0,
                        });
                        ui.close();
                    }
                    if ui.button("Set exc/inh…").clicked() {
                        self.dlg = Some(Dialog::ExcInh {
                            group: "grid".into(),
                            frac: 0.2,
                        });
                        ui.close();
                    }
                });
                ui.menu_button("View", |ui| {
                    // toggles stay open so several windows can flip at once
                    let err_seq = self.ctl.shared.read().unwrap().err_seq;
                    let unread_err = err_seq > self.err_seq_seen && !self.show_console;
                    for (flag, icon, label) in [
                        (&mut self.show_inspector, &ic.inspector, "Inspector"),
                        (&mut self.show_plots, &ic.plots, "Plots"),
                        (&mut self.show_console, &ic.console, "Console"),
                        (&mut self.show_script, &ic.script, "Script editor"),
                        (&mut self.show_inputs, &ic.connect, "Inputs"),
                        (&mut self.show_files, &ic.save, "Files"),
                        (&mut self.show_fps, &ic.speed, "frame stats"),
                    ] {
                        let dot = unread_err && label == "Console";
                        let btn = if dot {
                            icons::button(
                                icon,
                                egui::RichText::new(format!("{label} !"))
                                    .color(egui::Color32::LIGHT_RED),
                            )
                            .selected(*flag)
                        } else {
                            icons::button(icon, label).selected(*flag)
                        };
                        if ui.add(btn).clicked() {
                            *flag = !*flag;
                        }
                    }
                    ui.separator();
                    if ui.button("Reset layout").clicked() {
                        // windows return to their default positions
                        ctx.memory_mut(|m| m.reset_areas());
                        ui.close();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("Manual  (?)").clicked() {
                        self.show_manual = true;
                        ui.close();
                    }
                    if ui.button("Shortcuts").clicked() {
                        self.show_keys = true;
                        ui.close();
                    }
                    ui.menu_button("Examples", |ui| {
                        for ex in crate::examples::EXAMPLES {
                            if ui.button(ex.name).clicked() {
                                self.load_example(ex);
                                ui.close();
                            }
                        }
                    });
                    if ui.add(icons::button(&ic.about, "About")).clicked() {
                        self.dlg = Some(Dialog::About);
                        ui.close();
                    }
                });
                ui.separator();
                // sim controls — "Stop" (not "Pause"): abort_now flips the
                // flag on_progress/run chunks/builders watch. A busy script
                // can't resume anyway, so Pause was a lie.
                let live = self.sim.running || self.sim.busy;
                let (run_icon, run_label) = if live {
                    (&ic.stop, "Stop")
                } else {
                    (&ic.run, "Run")
                };
                // Stage 15 B: each control group (icon+widget+label) is a
                // nested horizontal — wrapping moves a group whole instead
                // of splitting e.g. the speed slider from its "realtime".
                toolbar_group(ui, 300.0, |ui| {
                    if ui.add(icons::button(run_icon, run_label)).clicked() {
                        if live {
                            self.ctl.abort_now();
                            self.send(ClientMsg::AutoRun(false));
                        } else {
                            self.send(ClientMsg::AutoRun(true));
                        }
                    }
                    if ui
                        .add(icons::button(&ic.step, "10ms"))
                        .on_hover_text("Step 10 ms")
                        .clicked()
                    {
                        self.send(ClientMsg::RunMs(10.0));
                    }
                    if ui
                        .add(icons::button(&ic.step_long, "100ms"))
                        .on_hover_text("Step 100 ms")
                        .clicked()
                    {
                        self.send(ClientMsg::RunMs(100.0));
                    }
                    if ui.add(icons::button(&ic.reset, "Reset")).clicked() {
                        self.send(ClientMsg::Reset);
                        self.plots.hist.clear();
                    }
                });
                toolbar_group(ui, 430.0, |ui| {
                    let fg = ui.visuals().text_color();
                    ui.add(icons::tinted(&ic.speed, fg))
                        .on_hover_text("Speed (realtime factor)");
                    if ui
                        .add(egui::Slider::new(&mut self.sim.speed, 0.05..=20.0).logarithmic(true))
                        .changed()
                    {
                        self.send(ClientMsg::SetSpeed(self.sim.speed));
                    }
                    // One speed display: requested realtime factor + the best
                    // this machine has reached (peak steps/s → realtime).
                    let max_x = self.sim.speed_peak * self.sim.dt_ms / 1000.0;
                    ui.label(if max_x > 0.05 {
                        format!("x{:.1} realtime (max x{:.0})", self.sim.speed, max_x)
                    } else {
                        format!("x{:.1} realtime", self.sim.speed)
                    });
                });
                ui.separator();
                toolbar_group(ui, 120.0, |ui| {
                    let backend_icon = match self.sim.backend {
                        proto::Backend::Cuda => &ic.backend_gpu,
                        proto::Backend::Cpu => &ic.backend_cpu,
                    };
                    ui.add(icons::tinted(backend_icon, ui.visuals().text_color()))
                        .on_hover_text("Backend");
                    egui::ComboBox::from_id_salt("backend")
                        .selected_text(match self.sim.backend {
                            proto::Backend::Cuda => "cuda",
                            proto::Backend::Cpu => "cpu",
                        })
                        .show_ui(ui, |ui| {
                            if ui.add(icons::button(&ic.backend_gpu, "cuda")).clicked() {
                                self.send(ClientMsg::SetBackend(proto::Backend::Cuda));
                            }
                            if ui.add(icons::button(&ic.backend_cpu, "cpu")).clicked() {
                                self.send(ClientMsg::SetBackend(proto::Backend::Cpu));
                            }
                        });
                });
                toolbar_group(ui, 140.0, |ui| {
                    ui.add(icons::tinted(&ic.timestep, ui.visuals().text_color()))
                        .on_hover_text("Time step dt");
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.sim.dt_ms)
                                .range(0.01..=1.0)
                                .speed(0.01)
                                .suffix(" ms"),
                        )
                        .changed()
                    {
                        self.send(ClientMsg::SetDtMs(self.sim.dt_ms));
                    }
                });
                toolbar_group(ui, 160.0, |ui| {
                    ui.add(icons::tinted(&ic.noise, ui.visuals().text_color()))
                        .on_hover_text("Background noise rate");
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.sim.noise_hz)
                                .range(0.0..=5000.0)
                                .speed(1.0)
                                .suffix(" Hz"),
                        )
                        .changed()
                    {
                        self.send(ClientMsg::SetNoise {
                            hz: self.sim.noise_hz,
                            kick: self.sim.noise_kick,
                        });
                    }
                });
            });
            ui.horizontal_wrapped(|ui| {
                // Same wrapping rule as row 1: groups move whole.
                toolbar_group(ui, 380.0, |ui| {
                    let m = &mut self.mode;
                    icons::selectable_value(ui, m, Mode::Navigate, &ic.navigate, "Navigate");
                    icons::selectable_value(ui, m, Mode::Insert, &ic.insert, "Insert");
                    icons::selectable_value(ui, m, Mode::Connect, &ic.connect, "Connect");
                    icons::selectable_value(ui, m, Mode::Move, &ic.mv, "Move");
                });
                ui.separator();
                toolbar_group(ui, 240.0, |ui| {
                    ui.label("plane");
                    let p = &mut self.plane;
                    icons::selectable_value(ui, p, Plane::XZ, &ic.plane_xz, "x-z");
                    icons::selectable_value(ui, p, Plane::XY, &ic.plane_xy, "x-y");
                    icons::selectable_value(ui, p, Plane::YZ, &ic.plane_yz, "y-z");
                });
                ui.separator();
                toolbar_group(ui, 340.0, |ui| {
                    icons::toggle(ui, &mut self.show_synapses, &ic.synapses, "synapses");
                    icons::toggle(ui, &mut self.show_input_links, &ic.connect, "input links");
                    icons::toggle(ui, &mut self.show_grid, &ic.grid, "grid");
                    icons::toggle(ui, &mut self.show_dots, &ic.raster, "spike dots");
                });
                // Grid-group render mode: the plane LOD keeps big grids
                // interactive; cubes stay available for editing up close.
                toolbar_group(ui, 150.0, |ui| {
                    let view_icon = match self.grid_view {
                        crate::plane::GridView::Auto => &ic.view_auto,
                        crate::plane::GridView::Cubes => &ic.view_cubes,
                        crate::plane::GridView::Planes => &ic.view_planes,
                    };
                    ui.add(icons::tinted(view_icon, ui.visuals().text_color()))
                        .on_hover_text("Grid view");
                    egui::ComboBox::from_id_salt("grid_view")
                        .selected_text(match self.grid_view {
                            crate::plane::GridView::Auto => "auto",
                            crate::plane::GridView::Cubes => "cubes",
                            crate::plane::GridView::Planes => "planes",
                        })
                        .show_ui(ui, |ui| {
                            use crate::plane::GridView;
                            let g = &mut self.grid_view;
                            icons::selectable_value(ui, g, GridView::Auto, &ic.view_auto, "auto");
                            icons::selectable_value(
                                ui,
                                g,
                                GridView::Cubes,
                                &ic.view_cubes,
                                "cubes",
                            );
                            icons::selectable_value(
                                ui,
                                g,
                                GridView::Planes,
                                &ic.view_planes,
                                "planes",
                            );
                        });
                });
                if ui
                    .add(icons::icon_button(&ic.frame))
                    .on_hover_text("Frame network (F)")
                    .clicked()
                {
                    self.frame_all();
                }
                ui.separator();
                // U1: while the status line still holds the latest
                // error, draw it in warning colour.
                if self.last_error.as_deref() == Some(self.status.as_str()) {
                    ui.label(egui::RichText::new(&self.status).color(egui::Color32::LIGHT_RED));
                } else {
                    ui.label(&self.status);
                }
            });
        });
    }

    pub(super) fn status_bar(&mut self, ctx: &egui::Context) {
        let (t, rate, spikes) = {
            let sh = self.ctl.shared.read().unwrap();
            (sh.snap.time_ms, sh.snap.rate_hz, sh.snap.spikes_total)
        };
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("t = {t:.1} ms"));
                ui.separator();
                ui.label(format!("rate = {rate:.1} Hz"));
                ui.separator();
                ui.label(format!("spikes = {spikes}"));
                if let Some(g) = &self.geo {
                    let (mut lo, mut hi) = (u16::MAX, 0u16);
                    for &d in &g.syn_delay_steps {
                        lo = lo.min(d);
                        hi = hi.max(d);
                    }
                    if hi > 0 {
                        ui.separator();
                        ui.label(format!("delays {lo}..{hi} steps"));
                    }
                }
                if let (Some(g), Some(inst)) = (&self.geo, self.hover) {
                    ui.separator();
                    let label = match crate::view::entity_of(g, inst) {
                        Some(Entity::Neuron(n)) => g
                            .names
                            .get(n as usize)
                            .cloned()
                            .unwrap_or_else(|| format!("neuron {n}")),
                        Some(Entity::Input(k)) => format!(
                            "input {}",
                            g.inputs
                                .get(k as usize)
                                .map(|i| i.name.as_str())
                                .unwrap_or("?")
                        ),
                        Some(Entity::Readout(k)) => format!(
                            "readout {}",
                            g.readouts
                                .get(k as usize)
                                .map(|r| r.0.as_str())
                                .unwrap_or("?")
                        ),
                        None => String::new(),
                    };
                    if !label.is_empty() {
                        ui.label(label);
                    }
                }
                if self.sim.busy {
                    ui.separator();
                    ui.spinner();
                    ui.label("script running");
                }
            });
        });
    }

    /// Reserved strip for the mode/drag hint (Stage 15 C): as a panel it
    /// shrinks `usable_rect`, so floating windows can never cover it.
    pub(super) fn hints_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("hints").show(ctx, |ui| {
            if let Some(md) = &self.move_drag {
                ui.label(format!(
                    "plane: {} · Shift: vertical · X/Y/Z: axis lock · Esc: cancel",
                    crate::view::drag_plane_name(md.plane)
                ));
            } else {
                // U8: under ~1100 px the long hint eats a second row —
                // use the terse variant there.
                let narrow = usable_rect(ctx).width() < 1100.0;
                let hint = if narrow {
                    crate::view::mode_hint_short(self.mode)
                } else {
                    crate::view::mode_hint(self.mode)
                };
                ui.add(egui::Label::new(hint).wrap());
            }
        });
    }

    /// U5: transport dropped — an explicit banner, not a silent freeze.
    /// Native (login-mode) offers a reconnect using the stored session;
    /// browser reloads (the cookie session survives).
    pub(super) fn disconnect_banner(&mut self, ctx: &egui::Context) {
        if !show_disconnect_banner(
            self.ctl.shared.read().unwrap().connected,
            self.login.dlg.is_some(),
        ) {
            return;
        }
        // U5b: schedule the first auto-retry once, then countdown in
        // the banner. egui only repaints on dirty — nudge it so the
        // countdown (and the due attempt) actually ticks.
        // U5b: arm the next attempt from the *persisted* delay —
        // a fresh login resets it to 2 s, but a retry that fails (or,
        // on wasm, a pending socket that closes right after) keeps
        // backing off instead of looping at the initial 2 s.
        let now = ctx.input(|i| i.time);
        if self.login.reconnect_at.is_none() {
            self.login.reconnect_at = Some(now + self.login.reconnect_delay);
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(500));
        let can_retry = self.can_reconnect();
        let due = can_retry && self.login.reconnect_at.is_some_and(|t| now >= t);
        let mut manual = false;
        let top = usable_rect(ctx).min.y + 8.0;
        egui::Area::new("disconnect-banner".into())
            .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, top))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("connection to server lost")
                                .color(egui::Color32::LIGHT_RED),
                        );
                        if can_retry {
                            let left = self
                                .login
                                .reconnect_at
                                .map(|t| (t - now).ceil().max(0.0) as u32)
                                .unwrap_or(0);
                            ui.weak(format!("retrying in {left}s"));
                            manual = ui.button("Retry now").clicked();
                        }
                    });
                });
            });
        if due || manual {
            if self.try_reconnect() {
                // Attempt launched. Native connect_secure only returns
                // Ok for a live socket; the wasm socket is still
                // pending — either way, clear the timer: if it closes
                // again the banner re-arms at the (already grown)
                // delay, so attempts march 2,4,8…60 and never tighten.
                self.login.reconnect_at = None;
                self.status = "reconnecting…".into();
            } else {
                self.login.reconnect_at = Some(now + self.login.reconnect_delay);
            }
            self.login.reconnect_delay = next_backoff(self.login.reconnect_delay);
        }
    }

    /// Whether a retry is possible at all (native local mode has no
    /// socket to lose; the flag can never go false there anyway).
    fn can_reconnect(&self) -> bool {
        if cfg!(target_arch = "wasm32") {
            true // ws url recomputed from location each attempt
        } else {
            !self.login.connect_url.is_empty()
        }
    }

    /// One reconnect attempt, in place — the dead SimCtl is replaced
    /// wholesale (native spawns fresh pump/writer threads; wasm opens
    /// a new WebSocket, whose server re-sends Hello + Geo).
    fn try_reconnect(&mut self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            let host = web_sys::window()
                .and_then(|w| w.location().host().ok())
                .unwrap_or_default();
            let secure = web_sys::window()
                .and_then(|w| w.location().protocol().ok())
                .as_deref()
                == Some("https:");
            let url = format!("{}://{}/ws", if secure { "wss" } else { "ws" }, host);
            match SimCtl::connect(&url) {
                Ok(ctl) => {
                    self.ctl = ctl;
                    true
                }
                Err(e) => {
                    self.status = format!("reconnect: {e}");
                    false
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        match SimCtl::connect_secure(&self.login.connect_url, &self.login.connect_tok) {
            Ok(ctl) => {
                self.ctl = ctl;
                true
            }
            Err(e) => {
                self.status = format!("reconnect: {e}");
                false
            }
        }
    }
}

/// One toolbar control group laid out atomically (Stage 15 B). Inside a
/// `horizontal_wrapped` row, plain `ui.horizontal` claims the whole
/// remaining row as its desired size, so egui's wrap check
/// (`available < desired`) never fires and groups split mid-control.
/// Claiming an explicit upper-bound width instead makes the wrap check
/// see the group's real footprint — it moves to the next row whole.
/// `max_w` must overestimate, never undershoot: overshoot only packs a
/// little early; undershoot lets content overflow the row edge.
pub(super) fn toolbar_group(
    ui: &mut egui::Ui,
    max_w: f32,
    add: impl FnOnce(&mut egui::Ui),
) -> egui::InnerResponse<()> {
    ui.allocate_ui_with_layout(
        egui::vec2(max_w, ui.spacing().interact_size.y),
        egui::Layout::left_to_right(egui::Align::Center),
        add,
    )
}

/// U5b: next auto-retry delay — 2 s first, doubling, capped at 60 s.
/// A manual Retry keeps the same progression (a click is not a loop).
pub(super) fn next_backoff(prev: f64) -> f64 {
    if prev <= 0.0 {
        2.0
    } else {
        (prev * 2.0).min(60.0)
    }
}

/// U5: the banner shows only on a real mid-session disconnect — not
/// while the login dialog is up (there is no session to lose yet).
pub(super) fn show_disconnect_banner(connected: bool, login_open: bool) -> bool {
    !connected && !login_open
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Stage 15 B: `toolbar_group` units inside `horizontal_wrapped` wrap
    /// whole — a group never splits its icon/control across two rows, and
    /// a group that doesn't fit moves to the next row instead of
    /// overflowing the panel edge.
    #[test]
    fn wrapped_toolbar_groups_never_split() {
        let ctx = egui::Context::default();
        let mut same_row = true;
        let mut fits = true;
        let mut h = 0.0f32;
        for _ in 0..3 {
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::pos2(0.0, 0.0),
                        egui::vec2(300.0, 500.0),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    egui::TopBottomPanel::top("t").show(ctx, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            let max_x = ui.max_rect().max.x;
                            for g in 0..4 {
                                let r = toolbar_group(ui, 260.0, |ui| {
                                    let a = ui.label(format!("group-{g}")).rect;
                                    let b = ui.add(egui::Slider::new(&mut 0.5f32, 0.0..=1.0)).rect;
                                    same_row &= a.center().y == b.center().y;
                                })
                                .response
                                .rect;
                                fits &= r.max.x <= max_x + 0.5;
                            }
                        });
                        h = ui.min_rect().height();
                    });
                },
            );
        }
        assert!(same_row, "a control group split across wrap rows");
        assert!(fits, "a group overflowed the panel edge");
        assert!(h > 40.0, "groups did not wrap (panel {h}px)");
    }

    #[test]
    #[test]
    fn reconnect_backoff_doubles_to_60() {
        let mut d = 0.0;
        let seq: Vec<u64> = (0..8)
            .map(|_| {
                d = next_backoff(d);
                d as u64
            })
            .collect();
        assert_eq!(seq, vec![2, 4, 8, 16, 32, 60, 60, 60]);
    }

    #[test]
    #[test]
    fn disconnect_banner_gating() {
        assert!(show_disconnect_banner(false, false));
        assert!(!show_disconnect_banner(true, false));
        assert!(!show_disconnect_banner(false, true));
    }
}
