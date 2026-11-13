//! Console window (log lines + eval input) and its reader helper.
//! Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;

use proto::ClientMsg;

use super::layout::{app_win, console_default_rect, usable_rect};
use super::EditorApp;

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

    pub(super) fn console(&mut self, ctx: &egui::Context) {
        if !self.win.console {
            return;
        }
        // U1: an open console has seen every error so far.
        self.console.err_seq_seen = self.ctl.shared.read().unwrap().err_seq;
        let mut open = self.win.console;
        let u = usable_rect(ctx);
        let dr = console_default_rect(u);
        app_win(ctx, "Console (Rhai)")
            .open(&mut open)
            .default_size(dr.size())
            .default_pos(dr.min)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Clear").clicked() {
                        if let Ok(mut w) = self.ctl.shared.write() {
                            w.console.clear();
                        }
                    }
                });
                let lines = self.console_lines();
                // Fill the panel minus the input row (~70px): the old fixed
                // 90px cap left most of a tall console blank.
                let out_h = (ui.available_height() - 70.0).max(40.0);
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
                    let mut lay = |ui: &egui::Ui, s: &dyn egui::TextBuffer, wrap: f32| {
                        crate::syntax::highlight(ui, s.as_str(), wrap)
                    };
                    let te = egui::TextEdit::multiline(&mut self.console.input)
                        .desired_rows(2)
                        .desired_width(ui.available_width() - 60.0)
                        .font(egui::TextStyle::Monospace)
                        .layouter(&mut lay);
                    let r = ui.add(te);
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
                    let run = ui.button("Run");
                    let enter = r.has_focus()
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
                ui.small("Ctrl+Enter runs · up/down = history · `net`, `sim`, `rng` in scope");
            });
        self.win.console = open;
    }
}
