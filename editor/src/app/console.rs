//! Console dock (log lines + eval input) — a bottom panel like a
//! terminal, not a floating window (Stage 22). Toggle: View section,
//! `Ctrl+3` or `` ` ``. Errors still reach the status bar (U1) and an
//! unseen error marks the View-menu item while the dock is closed.

use eframe::egui;

use proto::ClientMsg;

use super::{gate, Act, EditorApp};

impl EditorApp {
    pub(super) fn console_lines(&self) -> Vec<String> {
        self.ctl
            .shared
            .read()
            .unwrap()
            .console
            .iter()
            .cloned()
            .collect()
    }

    /// The docked Rhai console: a resizable-height bottom panel between
    /// the side docks (update() shows it after `left_panel`, so it
    /// never slides under them). `win.console` off → hidden entirely.
    pub(super) fn console(&mut self, ctx: &egui::Context) {
        if !self.win.console {
            return;
        }
        // U1: an open console has seen every error so far.
        self.console.err_seq_seen = self.ctl.shared.read().unwrap().err_seq;
        egui::TopBottomPanel::bottom("console_dock")
            .resizable(true)
            .default_height(180.0)
            .height_range(60.0..=400.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("console").monospace().weak());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button("▁")
                            .on_hover_text("hide console ( ` / Ctrl+3 to reopen)")
                            .clicked()
                        {
                            self.win.console = false;
                        }
                        if ui.button("clear").clicked() {
                            if let Ok(mut w) = self.ctl.shared.write() {
                                w.console.clear();
                            }
                        }
                    });
                });
                let lines = self.console_lines();
                // Output fills everything above the input row + hint.
                let out_h = (ui.available_height() - 72.0).max(24.0);
                egui::ScrollArea::vertical()
                    .id_salt("conout")
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .max_height(out_h)
                    .show(ui, |ui| {
                        for l in &lines {
                            ui.label(egui::RichText::new(l).monospace().size(11.0));
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("›").monospace());
                    let mut lay = |ui: &egui::Ui, s: &dyn egui::TextBuffer, wrap: f32| {
                        crate::syntax::highlight(ui, s.as_str(), wrap)
                    };
                    let te = egui::TextEdit::multiline(&mut self.console.input)
                        .desired_rows(2)
                        .desired_width(ui.available_width() - 70.0)
                        .font(egui::TextStyle::Monospace)
                        .layouter(&mut lay);
                    let r = ui.add(te);
                    super::hl(ui, "console_input", r.rect);
                    // history
                    if r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                        let idx = self
                            .console
                            .hist_idx
                            .map(|i| i.saturating_sub(1))
                            .unwrap_or(self.console.history.len().saturating_sub(1));
                        if let Some(h) = self.console.history.get(idx) {
                            self.console.input = h.clone();
                            self.console.hist_idx = Some(idx);
                        }
                    }
                    if r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                        if let Some(i) = self.console.hist_idx {
                            let idx = i + 1;
                            if idx < self.console.history.len() {
                                self.console.input = self.console.history[idx].clone();
                                self.console.hist_idx = Some(idx);
                            } else {
                                self.console.input.clear();
                                self.console.hist_idx = None;
                            }
                        }
                    }
                    let why = self.why(Act::Eval);
                    let run = gate(ui, why, egui::Button::new("Run"));
                    super::hl(ui, "console_run", run.rect);
                    let enter = why.is_none()
                        && r.has_focus()
                        && ui.input(|i| {
                            i.key_pressed(egui::Key::Enter)
                                && (i.modifiers.ctrl || i.modifiers.command)
                        });
                    if (run.clicked() || enter) && !self.console.input.trim().is_empty() {
                        let code = self.console.input.clone();
                        self.console.history.push(code.clone());
                        self.console.hist_idx = None;
                        self.console.input.clear();
                        self.send(ClientMsg::Eval { code });
                    }
                });
                ui.small(
                    "Ctrl+Enter runs · up/down = history · `net`, `sim`, `rng` in scope · ` hides",
                );
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::icons::Icons;
    use crate::simctl::SimCtl;

    fn app(ctx: &egui::Context) -> EditorApp {
        let mut a = EditorApp::new(SimCtl::offline(), None);
        a.icons = Some(Icons::load(ctx));
        a
    }

    /// The console is a real dock: a bounded strip off the bottom of the
    /// remaining space, at both required sizes and beside the left dock.
    /// Two-sided: 0 px = it never docked (a floating `egui::Window`
    /// doesn't shrink `available_rect` — the classic regression);
    /// > 400 px = it swallowed the scene.
    #[test]
    fn console_dock_carves_bottom_space() {
        for (w, h) in [(800.0_f32, 700.0_f32), (1440.0, 900.0)] {
            let ctx = egui::Context::default();
            let mut a = app(&ctx);
            a.win.console = true;
            let mut avail = egui::Rect::ZERO;
            for _ in 0..3 {
                let _ = ctx.run(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::pos2(0.0, 0.0),
                            egui::vec2(w, h),
                        )),
                        ..Default::default()
                    },
                    |ctx| {
                        a.left_panel(ctx);
                        a.console(ctx);
                        avail = ctx.available_rect();
                    },
                );
            }
            let taken = h - avail.height();
            assert!(
                (60.0..=400.0).contains(&taken),
                "console dock took {taken:.0}px at {w}x{h} — docked?"
            );
            assert!(avail.height() >= 200.0, "console ate the scene: {avail:?}");
            // The dock must not span under the left panel either
            // (Stage 22B: at narrow widths that's the ~44 px rail).
            let left_min = if crate::app::layout::narrow_mode(&ctx) {
                28.0
            } else {
                150.0
            };
            assert!(avail.min.x >= left_min, "console slid under the left dock");
        }
    }

    /// `win.console` off → the dock releases the whole strip.
    /// Sabotage: render the panel unconditionally → height taken ≠ 0.
    #[test]
    fn hidden_console_leaves_full_height() {
        let ctx = egui::Context::default();
        let mut a = app(&ctx);
        a.win.console = false;
        let mut avail = egui::Rect::ZERO;
        for _ in 0..3 {
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::pos2(0.0, 0.0),
                        egui::vec2(800.0, 700.0),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    a.console(ctx);
                    avail = ctx.available_rect();
                },
            );
        }
        assert_eq!(avail.height(), 700.0, "hidden console still takes space");
    }
}
