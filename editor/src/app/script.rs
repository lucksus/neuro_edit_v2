//! Script editor window (Rhai text, run/save/load, examples) and the
//! script actions. Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;

use proto::ClientMsg;

use super::files::dir_of;
use super::layout::{app_win, clamp_window_rect, usable_rect};
use super::{EditorApp, PlotTab};

impl EditorApp {
    /// Load a bundled example into the script editor (Stage 13 H).
    pub(super) fn load_example(&mut self, ex: &'static crate::examples::Example) {
        self.script_name = ex.file.trim_end_matches(".rhai").to_string();
        self.script_text = ex.src.to_string();
        self.example_watch = Some(ex.watch);
        self.example_base_ver = None;
        self.example_deadline_ms = f64::MAX;
        self.example_quiet = 0;
        self.show_script = true;
        self.status = format!("example {:?} loaded — press Run", ex.name);
    }

    /// Run the console script (button + Ctrl+Enter share this). A loaded
    /// example's `watch` list stays pending until the post-Eval Geo
    /// covers the ids — sync_shared applies it via the selection.
    pub(super) fn run_script(&mut self) {
        let code = self.script_text.clone();
        self.send(ClientMsg::Eval { code });
        // Auto-watch arms on Run: only a Geo newer than this version is
        // the eval's result — the pre-run net may already cover the ids.
        if self.example_watch.is_some() {
            self.example_base_ver = Some(self.geo_version);
            self.example_deadline_ms = self.wall_ms + 45_000.0;
            self.example_quiet = 0;
        }
    }

    /// Multiline script editor (the original's QScintilla script window):
    /// scripts saved in the .ne2, load/save to .rhai files, Run via worker.
    pub(super) fn script_window(&mut self, ctx: &egui::Context) {
        if !self.show_script {
            return;
        }
        let mut open = self.show_script;
        let u = usable_rect(ctx);
        let dr = clamp_window_rect(
            egui::Rect::from_min_size(u.min + egui::vec2(12.0, 8.0), egui::vec2(480.0, 360.0)),
            u,
        );
        app_win(ctx, "Script editor")
            .open(&mut open)
            .default_pos(dr.min)
            .default_size(dr.size())
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("name");
                    ui.text_edit_singleline(&mut self.script_name);
                    if ui.button("Run").clicked() {
                        self.run_script();
                    }
                    if ui
                        .button("Run as job")
                        .on_hover_text("background job on the server — survives disconnect; progress in the jobs tab")
                        .clicked()
                    {
                        self.send(ClientMsg::StartJob {
                            name: self.script_name.clone(),
                            script: self.script_text.clone(),
                        });
                        self.plot_tab = PlotTab::Jobs;
                        self.show_plots = true;
                    }
                    if ui.button("Save to net").clicked() {
                        self.send(ClientMsg::Eval {
                            code: format!(
                                "net.script_add({:?}, {:?})",
                                self.script_name, self.script_text
                            ),
                        });
                        self.status = format!("script {:?} saved to net", self.script_name);
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
                    ui.text_edit_singleline(&mut self.script_file);
                    if ui.button("Open").clicked() {
                        self.send(ClientMsg::ReadFile {
                            path: self.script_file.clone(),
                        });
                    }
                    if ui.button("Save").clicked() {
                        self.send(ClientMsg::WriteFile {
                            path: self.script_file.clone(),
                            data: self.script_text.clone().into_bytes(),
                        });
                    }
                    if ui.button("List").clicked() {
                        self.send(ClientMsg::ListFiles {
                            path: dir_of(&self.script_file),
                        });
                    }
                });
                if let Some((p, is_dir)) = self.file_list_ui(ui) {
                    if is_dir {
                        self.send(ClientMsg::ListFiles { path: p });
                    } else {
                        self.script_file = p.clone();
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
                                    self.script_name = name.clone();
                                    self.script_text = code.clone();
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
                        egui::TextEdit::multiline(&mut self.script_text)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(16)
                            .layouter(&mut lay),
                    );
                    let run = r.has_focus()
                        && ui.input(|i| {
                            i.key_pressed(egui::Key::Enter)
                                && (i.modifiers.ctrl || i.modifiers.command)
                        });
                    if run && !self.script_text.trim().is_empty() {
                        self.run_script();
                    }
                });
            });
        self.show_script = open;
    }
}
