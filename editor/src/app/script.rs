//! Script editor window (Rhai text, run/save/load, examples) and the
//! script actions. Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;

use proto::{ClientMsg, NetEdit};

use super::files::dir_of;
use super::layout::{app_win, default_rect, usable_rect, DefWin};
use super::{gate, Act, EditorApp, PlotTab};

impl EditorApp {
    /// Load a bundled example into the script editor (Stage 13 H).
    pub(super) fn load_example(&mut self, ex: &'static crate::examples::Example) {
        self.script.name = ex.file.trim_end_matches(".rhai").to_string();
        self.script.text = ex.src.to_string();
        self.script.example_watch = Some(ex.watch);
        self.script.example_base_ver = None;
        self.script.example_deadline_ms = f64::MAX;
        self.script.example_quiet = 0;
        self.win.script = true;
        self.status = format!("example {:?} loaded — press Run", ex.name);
    }

    /// Run the console script (button + Ctrl+Enter share this). A loaded
    /// example's `watch` list stays pending until the post-Eval Geo
    /// covers the ids — sync_shared applies it via the selection.
    pub(super) fn run_script(&mut self) {
        let code = self.script.text.clone();
        self.send(ClientMsg::Eval { code });
        // Auto-watch arms on Run: only a Geo newer than this version is
        // the eval's result — the pre-run net may already cover the ids.
        if self.script.example_watch.is_some() {
            self.script.example_base_ver = Some(self.geo_version);
            self.script.example_deadline_ms = self.wall_ms + 45_000.0;
            self.script.example_quiet = 0;
        }
    }

    /// Multiline script editor (the original's QScintilla script window):
    /// scripts saved in the .ne2, load/save to .rhai files, Run via worker.
    pub(super) fn script_window(&mut self, ctx: &egui::Context) {
        if !self.win.script {
            return;
        }
        let mut open = self.win.script;
        let u = usable_rect(ctx);
        let dr = default_rect(DefWin::Script, u);
        app_win(ctx, "Script editor")
            .open(&mut open)
            .default_pos(dr.min)
            .default_size(dr.size())
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("name");
                    ui.text_edit_singleline(&mut self.script.name);
                    if gate(ui, self.why(Act::Eval), egui::Button::new("Run")).clicked() {
                        self.run_script();
                    }
                    // Jobs run on their own env — not gated on busy.
                    if ui
                        .button("Run as job")
                        .on_hover_text("background job on the server — survives disconnect; progress in the jobs tab")
                        .clicked()
                    {
                        self.send(ClientMsg::StartJob {
                            name: self.script.name.clone(),
                            script: self.script.text.clone(),
                        });
                        self.plots.tab = PlotTab::Jobs;
                        self.win.plots = true;
                    }
                    if gate(ui, self.why(Act::Edit), egui::Button::new("Save to net")).clicked() {
                        self.edit_net(NetEdit::ScriptAdd {
                            name: self.script.name.clone(),
                            src: self.script.text.clone(),
                        });
                        self.status = format!("script {:?} saved to net", self.script.name);
                    }
                    ui.menu_button("Examples", |ui| {
                        for ex in crate::examples::EXAMPLES {
                            if ui.button(ex.name).clicked() {
                                self.load_example(ex);
                                ui.close();
                            }
                        }
                    });
                });
                ui.horizontal(|ui| {
                    ui.label("file");
                    ui.text_edit_singleline(&mut self.script.file);
                    if ui.button("Open").clicked() {
                        self.send(ClientMsg::ReadFile {
                            path: self.script.file.clone(),
                        });
                    }
                    if ui.button("Save").clicked() {
                        self.send(ClientMsg::WriteFile {
                            path: self.script.file.clone(),
                            data: self.script.text.clone().into_bytes(),
                        });
                    }
                    if ui.button("List").clicked() {
                        self.send(ClientMsg::ListFiles {
                            path: dir_of(&self.script.file),
                        });
                    }
                });
                if let Some((p, is_dir)) = self.file_list_ui(ui) {
                    if is_dir {
                        self.send(ClientMsg::ListFiles { path: p });
                    } else {
                        self.script.file = p.clone();
                        self.send(ClientMsg::ReadFile { path: p });
                    }
                }
                // scripts stored in the loaded .ne2
                if let Some(g) = self.geo.clone() {
                    if !g.scripts.is_empty() {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("in file:");
                            for (name, code) in &g.scripts {
                                if ui.button(name).clicked() {
                                    self.script.name = name.clone();
                                    self.script.text = code.clone();
                                }
                            }
                        });
                    }
                }
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut lay = |ui: &egui::Ui, s: &dyn egui::TextBuffer, wrap: f32| {
                        crate::syntax::highlight(ui, s.as_str(), wrap)
                    };
                    let r = ui.add(
                        egui::TextEdit::multiline(&mut self.script.text)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(16)
                            .layouter(&mut lay),
                    );
                    let run = self.can(Act::Eval)
                        && r.has_focus()
                        && ui.input(|i| {
                            i.key_pressed(egui::Key::Enter)
                                && (i.modifiers.ctrl || i.modifiers.command)
                        });
                    if run && !self.script.text.trim().is_empty() {
                        self.run_script();
                    }
                });
            });
        self.win.script = open;
    }
}
