//! `?` shortcuts overlay and the built-in manual window. Moved
//! verbatim out of `app.rs` (Stage 19).

use eframe::egui;

use super::layout::{app_win, clamp_window_rect, usable_rect};
use super::EditorApp;

impl EditorApp {
    /// `?` — every mode key and mouse gesture in one small overlay.
    /// (WASD/QE pan the camera, so tools use other letters.)
    pub(super) fn keys_window(&mut self, ctx: &egui::Context) {
        if !self.win.keys {
            return;
        }
        let mut open = self.win.keys;
        app_win(ctx, "Shortcuts")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                egui::Grid::new("keys_grid").num_columns(2).show(ui, |ui| {
                    let row = |ui: &mut egui::Ui, k: &str, what: &str| {
                        ui.monospace(k);
                        ui.label(what);
                        ui.end_row();
                    };
                    ui.strong("modes");
                    ui.end_row();
                    row(ui, "V / Esc", "navigate");
                    row(ui, "I", "insert neuron");
                    row(ui, "C", "connect");
                    row(ui, "M", "move");
                    row(ui, "Esc", "cancel drag/connect, else navigate");
                    ui.end_row();
                    ui.strong("camera");
                    ui.end_row();
                    row(ui, "right-drag", "orbit");
                    row(ui, "wheel", "zoom");
                    row(ui, "W A S D / Q E", "pan");
                    row(ui, "double-click", "focus");
                    row(ui, "F", "frame network");
                    ui.end_row();
                    ui.strong("mouse (navigate)");
                    ui.end_row();
                    row(ui, "click", "select (Shift adds)");
                    row(ui, "drag on object", "move it");
                    row(ui, "drag on empty", "box select");
                    row(ui, "right-click", "context menu");
                    ui.end_row();
                    ui.strong("while moving");
                    ui.end_row();
                    row(ui, "Shift (hold)", "vertical camera plane");
                    row(ui, "X / Y / Z", "axis lock (repeat releases)");
                    ui.end_row();
                    ui.strong("edit");
                    ui.end_row();
                    row(ui, "Ctrl+C / X / V", "copy / cut / paste");
                    row(ui, "Del", "delete selection");
                    ui.end_row();
                    ui.strong("windows");
                    ui.end_row();
                    row(
                        ui,
                        "Ctrl+1..6",
                        "inspector · plots · console · script · inputs · files",
                    );
                    row(ui, "?", "manual");
                });
            });
        self.win.keys = open;
    }

    /// Manual window — `docs/manual/*.md` baked into the binary.
    pub(super) fn manual_window(&mut self, ctx: &egui::Context) {
        if !self.win.manual {
            return;
        }
        let mut open = self.win.manual;
        // Stage 15 A: narrow viewports get a shrunken manual — cap the
        // default at 70% of the usable strip instead of covering it.
        let u = usable_rect(ctx);
        let dr = clamp_window_rect(
            egui::Rect::from_min_size(
                u.min + egui::vec2(24.0, 8.0),
                egui::vec2(640.0, 520.0).min(u.size() * 0.7),
            ),
            u,
        );
        app_win(ctx, "Manual")
            .open(&mut open)
            .default_size(dr.size())
            .default_pos(dr.min)
            .show(ctx, |ui| {
                // horizontal children get an unbounded max_rect in x, so the
                // body column must be ALLOCATED an explicit rect — otherwise
                // ScrollArea grabs the infinite row width, the window grows
                // every frame, and wrapped labels still never wrap.
                let body_w = (ui.clip_rect().width() - 110.0 - 40.0).clamp(80.0, 480.0);
                let body_h = ui.available_height().max(60.0);
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.set_width(110.0);
                        for (i, p) in crate::manual::PAGES.iter().enumerate() {
                            ui.selectable_value(&mut self.win.manual_page, i, p.title);
                        }
                    });
                    ui.separator();
                    // allocate_ui_with_layout (not allocate_ui): plain
                    // allocate_ui inherits this row's horizontal layout and
                    // the scroll column would lay its widgets out sideways.
                    ui.allocate_ui_with_layout(
                        egui::vec2(body_w, body_h),
                        egui::Layout::top_down(egui::Align::LEFT),
                        |ui| {
                            crate::manual::body_scroll(ui, body_w, self.win.manual_page);
                        },
                    );
                });
            });
        self.win.manual = open;
    }
}
