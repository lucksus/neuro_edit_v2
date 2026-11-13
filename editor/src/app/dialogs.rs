//! One drawing function per `Dialog` variant plus the shared
//! `group_combo` picker. Moved verbatim out of `app.rs` (Stage 19):
//! the old `dialog()` match is now a dispatch to these methods.

use eframe::egui;

use crate::gpu::Entity;
use proto::{f64_of, ClientMsg, NetEdit};

use super::files::dir_of;
use super::layout::app_win;
use super::{Dialog, EditorApp, PendingAction};

impl EditorApp {
    pub(super) fn dialog(&mut self, ctx: &egui::Context) {
        let Some(mut d) = self.dlg.take() else { return };
        let mut open = true;
        let mut keep = true;
        match &mut d {
            Dialog::Grid2d {
                group,
                w,
                h,
                d: dist,
            } => {
                self.dlg_grid2d(ctx, &mut open, &mut keep, group, w, h, dist);
            }
            Dialog::Grid3d {
                group,
                w,
                h,
                d2,
                spacing,
            } => {
                self.dlg_grid3d(ctx, &mut open, &mut keep, group, w, h, d2, spacing);
            }
            Dialog::Connections {
                group,
                scaling,
                lambda,
                w,
            } => {
                self.dlg_connections(ctx, &mut open, &mut keep, group, scaling, lambda, w);
            }
            Dialog::Weights { group, mean, sd } => {
                self.dlg_weights(ctx, &mut open, &mut keep, group, mean, sd);
            }
            Dialog::SynTau { src, dst, mean, sd } => {
                self.dlg_syn_tau(ctx, &mut open, &mut keep, src, dst, mean, sd);
            }
            Dialog::ExcInh { group, frac } => {
                self.dlg_exc_inh(ctx, &mut open, &mut keep, group, frac);
            }
            Dialog::NewInput {
                name,
                weight,
                value,
            } => {
                self.dlg_new_input(ctx, &mut open, &mut keep, name, weight, value);
            }
            Dialog::NewSamples { name } => {
                self.dlg_new_samples(ctx, &mut open, &mut keep, name);
            }
            Dialog::NewReadout { name } => {
                self.dlg_new_readout(ctx, &mut open, &mut keep, name);
            }
            Dialog::AddToGroup { name } => {
                self.dlg_add_to_group(ctx, &mut open, &mut keep, name);
            }
            Dialog::Discard { then } => {
                self.dlg_discard(ctx, &mut open, &mut keep, then);
            }
            Dialog::SaveAs { path } => {
                self.dlg_save_as(ctx, &mut open, &mut keep, path);
            }
            Dialog::Load { path } => {
                self.dlg_load(ctx, &mut open, &mut keep, path);
            }
            Dialog::About => {
                self.dlg_about(ctx, &mut open, &mut keep);
            }
        }
        if open && keep {
            self.dlg = Some(d);
        }
    }

    /// Group-name picker for dialogs: combo of the live groups instead of
    /// a free-text field that happily accepted typos.
    pub(super) fn group_combo(&self, ui: &mut egui::Ui, id: &str, group: &mut String) {
        let names: Vec<String> = self
            .geo
            .as_ref()
            .map(|g| g.groups.iter().map(|(n, _)| n.clone()).collect())
            .unwrap_or_default();
        let label = if group.is_empty() {
            "(group)".to_string()
        } else {
            group.clone()
        };
        egui::ComboBox::from_id_salt(id)
            .width(140.0)
            .selected_text(label)
            .show_ui(ui, |ui| {
                for n in &names {
                    ui.selectable_value(group, n.clone(), n);
                }
            });
    }

    fn dlg_grid2d(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, group: &mut String, w: &mut i32, h: &mut i32, dist: &mut f32) {
                app_win(ctx, "Create 2D grid")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            ui.text_edit_singleline(group);
                        });
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(w).prefix("w ").range(1..=512));
                            ui.add(egui::DragValue::new(h).prefix("h ").range(1..=512));
                            ui.add(
                                egui::DragValue::new(dist)
                                    .prefix("dist ")
                                    .range(0.1..=1000.0),
                            );
                        });
                        // U8: show the size before Create — a stray
                        // 1000×1000 used to go straight to the wire.
                        ui.label(format!("= {} neurons", (*w).max(0) * (*h).max(0)));
                        if ui.button("Create").clicked() {
                            let (g, w2, h2, d2) = (group.clone(), *w, *h, *dist);
                            self.edit_net(NetEdit::CreateGrid2d {
                                group: g,
                                w: w2.max(1) as u32,
                                h: h2.max(1) as u32,
                                spacing: d2,
                                origin: [0.0; 3],
                            });
                            self.frame_pending = true;
                            *keep = false;
                        }
                    });
    }

    fn dlg_grid3d(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, group: &mut String, w: &mut i32, h: &mut i32, d2: &mut i32, spacing: &mut f32) {
                app_win(ctx, "Create 3D grid")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            ui.text_edit_singleline(group);
                        });
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(w).prefix("w ").range(1..=256));
                            ui.add(egui::DragValue::new(h).prefix("h ").range(1..=256));
                            ui.add(egui::DragValue::new(d2).prefix("d ").range(1..=256));
                        });
                        if ui.button("Create").clicked() {
                            let (g, w2, h2, dd, sp) = (group.clone(), *w, *h, *d2, *spacing);
                            self.edit_net(NetEdit::CreateGrid3d {
                                group: g,
                                w: w2.max(1) as u32,
                                h: h2.max(1) as u32,
                                d: dd.max(1) as u32,
                                spacing: sp,
                                origin: [0.0; 3],
                            });
                            self.frame_pending = true;
                            *keep = false;
                        }
                    });
    }

    fn dlg_connections(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, group: &mut String, scaling: &mut f32, lambda: &mut f32, w: &mut f32) {
                app_win(ctx, "Create connections")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            self.group_combo(ui, "conn_group", group);
                        });
                        ui.add(egui::DragValue::new(scaling).prefix("scaling ").speed(0.01));
                        ui.add(egui::DragValue::new(lambda).prefix("lambda "));
                        ui.add(egui::DragValue::new(w).prefix("w "));
                        if ui.button("Create").clicked() {
                            self.edit_net(NetEdit::CreateConnections {
                                group: group.clone(),
                                scaling: f64_of(*scaling),
                                lambda: f64_of(*lambda),
                                w: *w,
                            });
                            *keep = false;
                        }
                    });
    }

    fn dlg_weights(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, group: &mut String, mean: &mut f32, sd: &mut f32) {
                app_win(ctx, "Set synapse weights")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            self.group_combo(ui, "w_group", group);
                        });
                        ui.add(egui::DragValue::new(mean).prefix("mean "));
                        ui.add(egui::DragValue::new(sd).prefix("sd "));
                        if ui.button("Apply").clicked() {
                            self.edit_net(NetEdit::SetSynapseWeights {
                                group: group.clone(),
                                mean: f64_of(*mean),
                                sd: f64_of(*sd),
                            });
                            *keep = false;
                        }
                    });
    }

    fn dlg_syn_tau(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, src: &mut String, dst: &mut String, mean: &mut f32, sd: &mut f32) {
                app_win(ctx, "Set synapse tau (ms)")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.label("0 = merged global tau; >0 = own current trace");
                        ui.horizontal(|ui| {
                            ui.label("src group");
                            self.group_combo(ui, "tau_src", src);
                        });
                        ui.horizontal(|ui| {
                            ui.label("dst group");
                            self.group_combo(ui, "tau_dst", dst);
                        });
                        ui.add(egui::DragValue::new(mean).prefix("mean ms "));
                        ui.add(egui::DragValue::new(sd).prefix("sd "));
                        if ui.button("Apply").clicked() {
                            self.edit_net(NetEdit::SetSynTauDist {
                                src: src.clone(),
                                dst: dst.clone(),
                                mean: f64_of(*mean),
                                sd: f64_of(*sd),
                            });
                            *keep = false;
                        }
                    });
    }

    fn dlg_exc_inh(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, group: &mut String, frac: &mut f32) {
                app_win(ctx, "Set excitatory/inhibitory")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            self.group_combo(ui, "ei_group", group);
                        });
                        ui.add(egui::Slider::new(frac, 0.0..=1.0).text("inh frac"));
                        if ui.button("Apply").clicked() {
                            self.edit_net(NetEdit::SetExcInh {
                                group: group.clone(),
                                frac: f64_of(*frac),
                            });
                            *keep = false;
                        }
                    });
    }

    fn dlg_new_input(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, name: &mut String, weight: &mut f32, value: &mut f32) {
                app_win(ctx, "New input on selection")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("name");
                            ui.text_edit_singleline(name);
                        });
                        ui.add(egui::DragValue::new(weight).prefix("w "));
                        ui.add(egui::DragValue::new(value).prefix("value ").speed(0.1));
                        let targets: Vec<u32> = self
                            .selection
                            .iter()
                            .filter_map(|&i| match self.entity(i) {
                                Some(Entity::Neuron(n)) => Some(n),
                                _ => None,
                            })
                            .collect();
                        ui.label(format!("{} targets (from selection)", targets.len()));
                        if ui.button("Create").clicked() && !targets.is_empty() {
                            let n2 = name.clone();
                            let wt = *weight;
                            let tw: Vec<(u32, f32)> = targets.iter().map(|&t| (t, wt)).collect();
                            self.edit_net(NetEdit::AddInput {
                                name: n2,
                                targets: tw,
                                pos: [0.0; 3],
                            });
                            // a new input at value 0 drives nothing —
                            // apply the dialog's value right after create
                            self.send(ClientMsg::SetInputConst {
                                name: name.clone(),
                                value: *value,
                            });
                            *keep = false;
                        }
                    });
    }

    fn dlg_new_samples(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, name: &mut String) {
                app_win(ctx, "New samples node")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("name");
                            ui.text_edit_singleline(name);
                        });
                        ui.label("empty (t,value) series — fill via net.samples_write");
                        if ui.button("Create").clicked() {
                            let n2 = name.clone();
                            self.edit_net(NetEdit::AddSamples { name: n2 });
                            *keep = false;
                        }
                    });
    }

    fn dlg_new_readout(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, name: &mut String) {
                app_win(ctx, "New readout on selection")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("name");
                            ui.text_edit_singleline(name);
                        });
                        let units: Vec<u32> = self
                            .selection
                            .iter()
                            .filter_map(|&i| match self.entity(i) {
                                Some(Entity::Neuron(n)) => Some(n),
                                _ => None,
                            })
                            .collect();
                        ui.label(format!("{} units (from selection)", units.len()));
                        if ui.button("Create").clicked() && !units.is_empty() {
                            let n2 = name.clone();
                            self.edit_net(NetEdit::AddReadout { name: n2, units });
                            *keep = false;
                        }
                    });
    }

    fn dlg_add_to_group(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, name: &mut String) {
                app_win(ctx, "Add selection to group")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            ui.text_edit_singleline(name);
                        });
                        let ids: Vec<u32> = self
                            .selection
                            .iter()
                            .filter_map(|&i| match self.entity(i) {
                                Some(Entity::Neuron(n)) => Some(n),
                                _ => None,
                            })
                            .collect();
                        ui.label(format!("{} neurons", ids.len()));
                        if ui.button("Add").clicked() && !ids.is_empty() {
                            let nm = name.clone();
                            self.edit_net(NetEdit::AddToGroup { name: nm, ids });
                            *keep = false;
                        }
                    });
    }

    fn dlg_discard(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, then: &mut PendingAction) {
                app_win(ctx, "Unsaved changes")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.label("The network has unsaved changes — discard them?");
                        ui.horizontal(|ui| {
                            if ui.button("Discard").clicked() {
                                let act = then.clone();
                                self.do_pending(ctx, act);
                                *keep = false;
                            }
                            if ui.button("Cancel").clicked() {
                                *keep = false;
                            }
                        });
                    });
    }

    fn dlg_save_as(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, path: &mut String) {
                app_win(ctx, "Save as").open(open).show(ctx, |ui| {
                    ui.text_edit_singleline(path);
                    if ui.button("Save").clicked() {
                        let p = path.clone();
                        self.send(ClientMsg::SaveNet { path: p });
                        *keep = false;
                    }
                });
    }

    fn dlg_load(&mut self, ctx: &egui::Context, open: &mut bool, keep: &mut bool, path: &mut String) {
                app_win(ctx, "Load network")
                    .open(open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(path);
                            if ui.button("list").clicked() {
                                self.send(ClientMsg::ListFiles { path: dir_of(path) });
                            }
                            if ui.button("Load").clicked() {
                                let p = path.clone();
                                self.guarded(ctx, PendingAction::Load(p));
                                *keep = false;
                            }
                        });
                        if let Some((p, is_dir)) = self.file_list_ui(ui) {
                            *path = p.clone();
                            if is_dir {
                                self.send(ClientMsg::ListFiles { path: p });
                            }
                        }
                    });
    }

    fn dlg_about(&mut self, ctx: &egui::Context, open: &mut bool, _keep: &mut bool) {
                app_win(ctx, "About neuro_edit v2")
                    .open(open)
                    .show(ctx, |ui| {
                        if let Some(i) = &self.icons {
                            // 740×246 art, drawn at half size so it stays
                            // sharp on 2x screens
                            ui.add(
                                egui::Image::new(&i.banner)
                                    .fit_to_exact_size(egui::vec2(370.0, 123.0)),
                            );
                        }
                        ui.heading("neuro_edit v2");
                        ui.label("Izhikevich spiking-network simulator and 3D editor.");
                        ui.label("Rust + CUDA (cudarc/NVRTC) + egui/wgpu + Rhai.");
                        ui.label("Rewrite of the C++/Qt4 neuro_edit (2011–2012).");
                        ui.label("License: GPL-3.0");
                    });
    }
}
