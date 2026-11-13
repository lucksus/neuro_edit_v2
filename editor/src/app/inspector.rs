//! Inspector window (selected entity props + edit sends) and the
//! Inputs window. Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;

use crate::gpu::Entity;
use proto::{ClientMsg, NetEdit};

use super::layout::{app_win, clamp_window_rect, usable_rect};
use super::{EditorApp, PlotTab};

impl EditorApp {
    pub(super) fn inspector(&mut self, ctx: &egui::Context) {
        if !self.show_inspector {
            return;
        }
        let mut open = self.show_inspector;
        let u = usable_rect(ctx);
        let dr = clamp_window_rect(
            egui::Rect::from_min_size(
                egui::pos2(u.max.x - 248.0, u.min.y + 8.0),
                egui::vec2(240.0, 200.0),
            ),
            u,
        );
        app_win(ctx, "Inspector")
            .open(&mut open)
            .default_width(240.0)
            .default_pos(dr.min)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let sel: Vec<u32> = self.selection.iter().copied().collect();
                    if sel.len() == 1 {
                        self.inspect_one(ui, sel[0]);
                    } else if !sel.is_empty() {
                        ui.label(format!("{} objects selected", sel.len()));
                        if ui.button("Delete").clicked() {
                            self.delete_selection();
                        }
                    } else {
                        ui.label("nothing selected");
                    }
                    ui.separator();
                    ui.heading("Groups");
                    if let Some(g) = self.geo.clone() {
                        for (gi, (name, members)) in g.groups.iter().enumerate() {
                            ui.label(format!("{name}: {} neurons", members.len()));
                            // 3D lattices get a slice/projection choice for the
                            // activity plane (2D grids show their single sheet).
                            if let Some(gs) = g.grids.iter().find(|x| x.group == gi) {
                                if gs.dims[2] > 1 {
                                    let cfg = self.plane_cfgs.entry(gi as u32).or_default();
                                    ui.horizontal(|ui| {
                                        ui.label("  collapse");
                                        for a in 0..3u8 {
                                            ui.selectable_value(&mut cfg.axis, a, format!("d{a}"));
                                        }
                                        ui.separator();
                                        let cur = cfg.proj;
                                        if ui
                                            .selectable_label(cur == crate::plane::Proj::Max, "max")
                                            .clicked()
                                        {
                                            cfg.proj = crate::plane::Proj::Max;
                                        }
                                        if ui
                                            .selectable_label(
                                                cur == crate::plane::Proj::Mean,
                                                "mean",
                                            )
                                            .clicked()
                                        {
                                            cfg.proj = crate::plane::Proj::Mean;
                                        }
                                        if ui
                                            .selectable_label(
                                                matches!(cur, crate::plane::Proj::Slice(_)),
                                                "slice",
                                            )
                                            .clicked()
                                        {
                                            cfg.proj = crate::plane::Proj::Slice(0);
                                        }
                                    });
                                    if let crate::plane::Proj::Slice(k) = &mut cfg.proj {
                                        let max =
                                            gs.dims[cfg.axis.min(2) as usize].saturating_sub(1);
                                        *k = (*k).min(max);
                                        ui.horizontal(|ui| {
                                            ui.label("  slice");
                                            ui.add(
                                                egui::DragValue::new(k).range(0..=max).speed(0.1),
                                            );
                                            ui.label(format!("/ {max}"));
                                        });
                                    }
                                }
                            }
                        }
                        ui.separator();
                        ui.heading("Readouts");
                        for (name, units, _) in &g.readouts {
                            ui.label(format!("{name}: {} units", units.len()));
                        }
                        if g.readouts.is_empty() {
                            ui.label("(none)");
                        }
                    }
                });
            });
        self.show_inspector = open;
    }

    /// Constant-drive inputs, in their own floating window.
    pub(super) fn inputs_window(&mut self, ctx: &egui::Context) {
        if !self.show_inputs {
            return;
        }
        let mut open = self.show_inputs;
        let u = usable_rect(ctx);
        let dr = clamp_window_rect(
            egui::Rect::from_min_size(u.min + egui::vec2(260.0, 8.0), egui::vec2(220.0, 140.0)),
            u,
        );
        app_win(ctx, "Inputs")
            .open(&mut open)
            .default_size(dr.size())
            .default_pos(dr.min)
            .show(ctx, |ui| {
                if let Some(g) = self.geo.clone() {
                    for (k, inp) in g.inputs.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(&inp.name);
                            // seeded from the live sim value, not 0 — the
                            // field must show what the sim is doing
                            let mut v = self.input_values.get(k).copied().unwrap_or(0.0);
                            if ui.add(egui::DragValue::new(&mut v).speed(0.5)).changed() {
                                self.send(ClientMsg::SetInputConst {
                                    name: inp.name.clone(),
                                    value: v,
                                });
                            }
                        });
                    }
                    if g.inputs.is_empty() {
                        ui.label("(none)");
                    }
                }
            });
        self.show_inputs = open;
    }

    pub(super) fn inspect_one(&mut self, ui: &mut egui::Ui, inst_id: u32) {
        let Some(g) = self.geo.clone() else { return };
        match self.entity(inst_id) {
            Some(Entity::Neuron(id)) => {
                let i = id as usize;
                if i >= g.pos.len() {
                    return;
                }
                ui.label(format!("neuron #{id}"));
                let mut name = g.names[i].clone();
                if ui.text_edit_singleline(&mut name).changed() {
                    self.edit_net(NetEdit::SetNeuronName { id, name });
                }
                let mut p = g.pos[i];
                ui.horizontal(|ui| {
                    ui.label("pos");
                    let mut changed = false;
                    for pv in &mut p {
                        changed |= ui.add(egui::DragValue::new(pv).speed(1.0)).changed();
                    }
                    if changed {
                        self.edit_net(NetEdit::SetNeuronPos { id, pos: p });
                    }
                });
                // U2: the kind is editable — Dale's law re-signs its
                // outgoing synapses at the next sim build.
                ui.horizontal(|ui| {
                    ui.label("kind:");
                    let mut inh = g.kind[i] != 0;
                    for (v, label) in [(false, "exc"), (true, "inh")] {
                        if ui.selectable_label(inh == v, label).clicked() && inh != v {
                            self.edit_net(NetEdit::SetNeuronKind {
                                id,
                                kind: if v {
                                    proto::NeuronKind::Inhibitory
                                } else {
                                    proto::NeuronKind::Excitatory
                                },
                            });
                            inh = v;
                        }
                    }
                });
                // Discoverable watching: one click to the plots window.
                if ui.button("Plot v(t)").clicked() {
                    self.watch(vec![id]);
                    self.show_plots = true;
                    self.plot_tab = PlotTab::Voltage;
                }
                ui.label("group memberships:".to_string());
                for (gname, members) in &g.groups {
                    if members.contains(&id) {
                        ui.label(format!("  {gname}"));
                    }
                }
            }
            Some(Entity::Input(k)) => {
                let inp = &g.inputs[k as usize];
                ui.label(format!("input: {}", inp.name));
                ui.label(format!("{} targets", inp.targets.len()));
                let mut v = self.input_values.get(k as usize).copied().unwrap_or(0.0);
                if ui
                    .add(egui::Slider::new(&mut v, -50.0..=50.0).text("value"))
                    .changed()
                {
                    self.send(ClientMsg::SetInputConst {
                        name: inp.name.clone(),
                        value: v,
                    });
                }
            }
            Some(Entity::Readout(k)) => {
                let (name, units, w) = &g.readouts[k as usize];
                ui.label(format!("readout: {name}"));
                ui.label(format!("{} units", units.len()));
                if w.is_empty() {
                    ui.label("weights: —");
                } else {
                    ui.label(format!("weights: {} (plot in Plots)", w.len()));
                }
            }
            None => {
                ui.label("?");
            }
        }
    }
}
