//! Inspector window (selected entity props + edit sends) and the
//! Inputs window. Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;

use crate::gpu::Entity;
use proto::{ClientMsg, NetEdit};

use super::layout::{app_win, default_rect, usable_rect, DefWin};
use super::{gate, gate_ui, Act, EditorApp, PlotTab};

impl EditorApp {
    pub(super) fn inspector(&mut self, ctx: &egui::Context) {
        if !self.win.inspector {
            return;
        }
        let mut open = self.win.inspector;
        let u = usable_rect(ctx);
        let dr = default_rect(DefWin::Inspector, u);
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
                        let why_edit = self.why(Act::Edit);
                        // Retype every selected neuron in one edit.
                        if let Some(g) = self.geo.clone() {
                            let ids: Vec<u32> = sel
                                .iter()
                                .filter_map(|&i| match self.entity(i) {
                                    Some(Entity::Neuron(id)) => Some(id),
                                    _ => None,
                                })
                                .collect();
                            if !ids.is_empty() && !g.types.is_empty() {
                                ui.horizontal(|ui| {
                                    ui.label("type:");
                                    gate_ui(ui, why_edit, |ui| {
                                        egui::ComboBox::from_id_salt("multi_type")
                                            .selected_text("set…")
                                            .show_ui(ui, |ui| {
                                                for t in &g.types {
                                                    if ui.selectable_label(false, &t.name).clicked()
                                                    {
                                                        self.edit_net(NetEdit::SetNeuronType {
                                                            ids: ids.clone(),
                                                            ty: t.name.clone(),
                                                        });
                                                    }
                                                }
                                            });
                                    });
                                });
                            }
                        }
                        if gate(ui, why_edit, egui::Button::new("Delete")).clicked() {
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
        self.win.inspector = open;
    }

    /// Constant-drive inputs, in their own floating window.
    pub(super) fn inputs_window(&mut self, ctx: &egui::Context) {
        if !self.win.inputs {
            return;
        }
        let mut open = self.win.inputs;
        let u = usable_rect(ctx);
        let dr = default_rect(DefWin::Inputs, u);
        app_win(ctx, "Inputs")
            .open(&mut open)
            .default_size(dr.size())
            .default_pos(dr.min)
            .show(ctx, |ui| {
                if let Some(g) = self.geo.clone() {
                    let why = self.why(Act::SetInput);
                    for (k, inp) in g.inputs.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(&inp.name);
                            // seeded from the live sim value, not 0 — the
                            // field must show what the sim is doing
                            let mut v = self.input_values.get(k).copied().unwrap_or(0.0);
                            let changed = gate_ui(ui, why, |ui| {
                                ui.add(egui::DragValue::new(&mut v).speed(0.5)).changed()
                            })
                            .inner;
                            if changed {
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
        self.win.inputs = open;
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
                let why_edit = self.why(Act::Edit);
                gate_ui(ui, why_edit, |ui| {
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
                });
                // ---- type / model / params (protocol 2 descriptors) ----
                // Everything below is built from `g.types` + `self.models`
                // — a new registered model's fields appear with no editor
                // change.
                let ty_i = g.ty.get(i).copied().unwrap_or(0) as usize;
                let trow = g.types.get(ty_i);
                let mdesc = trow
                    .and_then(|t| self.models.iter().find(|m| m.id == t.model))
                    .cloned();
                gate_ui(ui, why_edit, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("type:");
                        let cur = trow.map(|t| t.name.clone()).unwrap_or_else(|| "—".into());
                        egui::ComboBox::from_id_salt(("ntype", id))
                            .selected_text(cur)
                            .show_ui(ui, |ui| {
                                for t in &g.types {
                                    let is = trow.is_some_and(|c| c.name == t.name);
                                    if ui.selectable_label(is, &t.name).clicked() && !is {
                                        self.edit_net(NetEdit::SetNeuronType {
                                            ids: vec![id],
                                            ty: t.name.clone(),
                                        });
                                    }
                                }
                            });
                        // U2: the kind is editable — Dale's law re-signs its
                        // outgoing synapses at the next sim build.
                        ui.separator();
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
                });
                if let (Some(t), Some(m)) = (trow, mdesc) {
                    gate_ui(ui, why_edit, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("model: {}", m.label));
                            if !m.presets.is_empty() {
                                egui::ComboBox::from_id_salt(("preset", id))
                                    .selected_text("preset…")
                                    .show_ui(ui, |ui| {
                                        for (pname, row) in &m.presets {
                                            if ui.selectable_label(false, pname).clicked() {
                                                self.edit_net(NetEdit::SetNeuronParams {
                                                    ids: vec![id],
                                                    params: Some(row.clone()),
                                                });
                                            }
                                        }
                                    });
                            }
                        });
                        // One field per ParamDesc — unit + range ride the
                        // descriptor, doc rides the tooltip.
                        let eff: Vec<f32> = g
                            .params
                            .get(i)
                            .and_then(|p| p.clone())
                            .unwrap_or_else(|| t.params.clone());
                        let over = g.params.get(i).is_some_and(|p| p.is_some());
                        for (pi, pd) in m.params.iter().enumerate() {
                            ui.horizontal(|ui| {
                                let mut v = eff.get(pi).copied().unwrap_or(pd.default);
                                let speed = ((pd.max - pd.min) / 100.0).abs().max(0.001);
                                let dv = egui::DragValue::new(&mut v)
                                    .speed(speed)
                                    .range(pd.min..=pd.max);
                                let r = ui.add(dv).on_hover_text(&pd.doc);
                                ui.label(format!("{} [{}]", pd.name, pd.unit));
                                if r.changed() {
                                    let mut row = eff.clone();
                                    row.resize(m.params.len(), pd.default);
                                    row[pi] = v;
                                    self.edit_net(NetEdit::SetNeuronParams {
                                        ids: vec![id],
                                        params: Some(row),
                                    });
                                }
                            });
                        }
                        if over {
                            ui.horizontal(|ui| {
                                ui.weak("params: override");
                                if ui.button("reset to type").clicked() {
                                    self.edit_net(NetEdit::SetNeuronParams {
                                        ids: vec![id],
                                        params: None,
                                    });
                                }
                            });
                        }
                    });
                }
                // Discoverable watching: one click to the plots window.
                if ui.button("Plot state").clicked() {
                    self.watch(vec![id]);
                    self.win.plots = true;
                    self.plots.tab = PlotTab::State(0);
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
                // Live input drive — safe during a run, gated only on busy.
                let why = self.why(Act::SetInput);
                let changed = gate_ui(ui, why, |ui| {
                    let mut v = self.input_values.get(k as usize).copied().unwrap_or(0.0);
                    let changed = ui
                        .add(egui::Slider::new(&mut v, -50.0..=50.0).text("value"))
                        .changed();
                    (v, changed)
                })
                .inner;
                if changed.1 {
                    self.send(ClientMsg::SetInputConst {
                        name: inp.name.clone(),
                        value: changed.0,
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
