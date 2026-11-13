//! Jobs tab inside the plots window: server job list, progress plot,
//! open/cancel. Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;
use egui_plot::{Line, PlotPoints};

use proto::ClientMsg;

use super::EditorApp;

impl EditorApp {
    /// Jobs tab: server job list, live `JobProgress` plot for the
    /// selected job, cancel/re-open controls, and the opened detail
    /// (status + result.json + script).
    pub(super) fn jobs_ui(&mut self, ui: &mut egui::Ui) {
        // U8: one fetch when the tab opens — JobStarted/JobDone/
        // JobProgress pushes keep the list live after that, so the
        // 2 s poll went away. Cancel clicks refetch on JobDone.
        if !self.jobs.fetched {
            self.jobs.fetched = true;
            self.send(ClientMsg::ListJobs);
        }
        if self.jobs.list.is_empty() {
            ui.label("no jobs — \"Run as job\" in the script editor starts one");
        }
        let jobs = self.jobs.list.clone();
        let rows = self
            .jobs
            .progress
            .get(&self.jobs.sel)
            .cloned()
            .unwrap_or_default();
        let detail = self.jobs.detail.as_ref().and_then(|d| {
            (d.id == self.jobs.sel).then(|| {
                (
                    d.name.clone(),
                    d.status.clone(),
                    d.result_json.clone(),
                    d.script.clone(),
                )
            })
        });
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width(260.0);
                for j in &jobs {
                    ui.horizontal(|ui| {
                        let label = format!("#{} {} — {}", j.id, j.name, j.status);
                        if ui.selectable_label(self.jobs.sel == j.id, label).clicked() {
                            self.jobs.sel = j.id;
                        }
                        if matches!(j.status.as_str(), "running" | "cancelling")
                            && ui.small_button("cancel").clicked()
                        {
                            self.send(ClientMsg::CancelJob { id: j.id });
                        }
                        if ui.small_button("open").clicked() {
                            self.send(ClientMsg::OpenJob { id: j.id });
                        }
                    });
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                match rows.is_empty() {
                    false => {
                        // field names in first-appearance order
                        let mut names: Vec<String> = Vec::new();
                        for r in &rows {
                            for (k, _) in r {
                                if !names.contains(k) {
                                    names.push(k.clone());
                                }
                            }
                        }
                        egui_plot::Plot::new("jobprog")
                            .height(90.0)
                            .legend(egui_plot::Legend::default())
                            .show(ui, |ui| {
                                for name in &names {
                                    let pts: PlotPoints = rows
                                        .iter()
                                        .enumerate()
                                        .filter_map(|(i, r)| {
                                            r.iter()
                                                .find(|(k, _)| k == name)
                                                .map(|(_, v)| [i as f64, *v])
                                        })
                                        .collect();
                                    ui.line(Line::new(name.clone(), pts));
                                }
                            });
                        ui.label(format!(
                            "job #{} — {} progress row(s)",
                            self.jobs.sel,
                            rows.len()
                        ));
                    }
                    _ => {
                        ui.label(if self.jobs.sel == 0 {
                            "select a job to plot its progress".to_string()
                        } else {
                            format!("job #{} — no progress rows yet", self.jobs.sel)
                        });
                    }
                }
                if let Some((name, status, result_json, script)) = detail {
                    ui.collapsing("opened job", |ui| {
                        ui.label(format!("#{} {} — {}", self.jobs.sel, name, status));
                        if let Some(r) = &result_json {
                            ui.label("result.json:");
                            ui.monospace(r);
                        }
                        egui::ScrollArea::vertical()
                            .max_height(80.0)
                            .show(ui, |ui| {
                                ui.monospace(&script);
                            });
                    });
                }
            });
        });
    }
}
