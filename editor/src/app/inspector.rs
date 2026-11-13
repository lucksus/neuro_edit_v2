//! Inspector: a resizable right-hand dock (Stage 22) with the selected
//! entity's props, the auto-watched V(t) trace, and the model's
//! parameter design map (izh a–b/c–d) or one trace per state slot.
//! The Inputs window (floating) also lives here.

use eframe::egui;
use egui_plot::{Line, PlotPoints};

use crate::gpu::Entity;
use proto::{ClientMsg, NetEdit};

use super::layout::{app_win, default_rect, usable_rect, DefWin};
use super::{gate, gate_ui, Act, EditorApp, PlotTab};

/// `name [unit]` — but an empty unit is just the name (Stage 21's
/// `b []` defect: AdEx's unitless params rendered a literal `[]`).
pub(super) fn param_label(pd: &proto::ParamDesc) -> String {
    if pd.unit.is_empty() {
        pd.name.clone()
    } else {
        format!("{} [{}]", pd.name, pd.unit)
    }
}

/// The map spot nearest to plot point `p`, inside a 5%-of-range ellipse
/// — the "click a region" hit test. Distances normalize per axis so a
/// click anywhere in a spot's region applies it.
pub(super) fn spot_at(plane: &proto::MapPlaneDesc, p: [f64; 2]) -> Option<usize> {
    let sx = (plane.x_range.1 - plane.x_range.0) as f64;
    let sy = (plane.y_range.1 - plane.y_range.0) as f64;
    if sx <= 0.0 || sy <= 0.0 {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for (i, s) in plane.spots.iter().enumerate() {
        let (Some(&px), Some(&py)) = (s.params.get(plane.x), s.params.get(plane.y)) else {
            continue;
        };
        let dx = (p[0] - px as f64) / (sx * 0.05);
        let dy = (p[1] - py as f64) / (sy * 0.05);
        let d = dx * dx + dy * dy;
        if d < 1.0 && best.is_none_or(|(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

impl EditorApp {
    /// The right dock: identity + editable props on top, live traces and
    /// the parameter map below — a taller inspector than the old window.
    pub(super) fn inspector(&mut self, ctx: &egui::Context) {
        if !self.win.inspector {
            return;
        }
        egui::SidePanel::right("inspector_dock")
            .resizable(true)
            .default_width(300.0)
            .width_range(220.0..=520.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Inspector");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button("»")
                            .on_hover_text("hide inspector (Ctrl+1 to reopen)")
                            .clicked()
                        {
                            self.win.inspector = false;
                        }
                    });
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .id_salt("inspector_scroll")
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
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
                                                        if ui
                                                            .selectable_label(false, &t.name)
                                                            .clicked()
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
                            ui.weak("nothing selected — click a neuron, input or readout");
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
                                                ui.selectable_value(
                                                    &mut cfg.axis,
                                                    a,
                                                    format!("d{a}"),
                                                );
                                            }
                                            ui.separator();
                                            let cur = cfg.proj;
                                            if ui
                                                .selectable_label(
                                                    cur == crate::plane::Proj::Max,
                                                    "max",
                                                )
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
                                                    egui::DragValue::new(k)
                                                        .range(0..=max)
                                                        .speed(0.1),
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

    /// The trailing 300 ms trace of one state slot of neuron `id` —
    /// descriptor-named ("v (mV)"), descriptor-ranged, always following.
    /// Watching is automatic: selection → `update_watch` → `Watch(ids)`,
    /// so this needs no button (Stage 22).
    fn neuron_trace(&mut self, ui: &mut egui::Ui, id: u32, s: usize, m: &proto::ModelDesc) {
        let d = m.state.get(s);
        let name = d.map(|d| d.name.as_str()).unwrap_or("s?");
        let unit = d.map(|d| d.unit.as_str()).unwrap_or("");
        let range = d.map(|d| d.plot_range).unwrap_or((-90.0, 40.0));
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(if unit.is_empty() {
                    name.to_string()
                } else {
                    format!("{name} ({unit})")
                })
                .small()
                .strong(),
            );
            if s == 0 {
                ui.weak("· auto-watch");
            }
        });
        let Some(h) = self.plots.hist.slot(s).and_then(|sl| sl.get(&id)) else {
            ui.weak("watching — the trace appears once the sim publishes a step");
            return;
        };
        if h.is_empty() {
            ui.weak("watching — the trace appears once the sim publishes a step");
            return;
        }
        let t = self.plots.last_sim_t;
        let (lo, hi) = ((t - 300.0).max(0.0), t.max(300.0));
        let pts: PlotPoints = h
            .iter()
            .filter(|(tt, _)| (lo..=hi).contains(tt))
            .map(|(tt, v)| [*tt, *v as f64])
            .collect();
        let name = name.to_string();
        let _ = egui_plot::Plot::new(("insp_trace", id, s))
            .height(110.0)
            .show(ui, |ui| {
                ui.line(Line::new(name, pts).color(crate::plot::watch_color(0)));
                ui.set_plot_bounds_x(lo..=hi);
                ui.set_plot_bounds_y(range.0 as f64..=range.1 as f64);
            });
    }

    /// The model's parameter design map: one panel per plane (izh draws
    /// a–b and c–d), region spots as labels, a cross at the neuron's
    /// current parameter pair, click a region to apply its row. Gated
    /// like the param fields — a click is a `SetNeuronParams` edit.
    fn neuron_map(
        &mut self,
        ui: &mut egui::Ui,
        id: u32,
        m: &proto::ModelDesc,
        eff: &[f32],
        why_edit: Option<&'static str>,
    ) {
        let Some(map) = &m.map else { return };
        for (pi, plane) in map.planes.iter().enumerate() {
            let xname = m
                .params
                .get(plane.x)
                .map(param_label)
                .unwrap_or_else(|| format!("p{}", plane.x));
            let yname = m
                .params
                .get(plane.y)
                .map(param_label)
                .unwrap_or_else(|| format!("p{}", plane.y));
            let inner = gate_ui(ui, why_edit, |ui| {
                egui_plot::Plot::new(("insp_map", id, pi))
                    .height(140.0)
                    .x_axis_label(xname)
                    .y_axis_label(yname)
                    .include_x(plane.x_range.0 as f64)
                    .include_x(plane.x_range.1 as f64)
                    .include_y(plane.y_range.0 as f64)
                    .include_y(plane.y_range.1 as f64)
                    .label_formatter(|_, _| String::new())
                    .show(ui, |ui| {
                        for s in &plane.spots {
                            if let (Some(&px), Some(&py)) =
                                (s.params.get(plane.x), s.params.get(plane.y))
                            {
                                ui.text(egui_plot::Text::new(
                                    "",
                                    egui_plot::PlotPoint::new(px as f64, py as f64),
                                    egui::RichText::new(&s.label).strong().size(10.0),
                                ));
                            }
                        }
                        if let (Some(&cx), Some(&cy)) = (eff.get(plane.x), eff.get(plane.y)) {
                            ui.points(
                                egui_plot::Points::new("", vec![[cx as f64, cy as f64]])
                                    .shape(egui_plot::MarkerShape::Cross)
                                    .radius(6.0_f32)
                                    .color(egui::Color32::WHITE),
                            );
                        }
                    })
            })
            .inner;
            // Click a region → apply its param row (an edit; the gate
            // above swallows clicks while running/script).
            let pt = inner
                .response
                .interact_pointer_pos()
                .or_else(|| inner.response.hover_pos())
                .map(|pos| inner.transform.value_from_position(pos));
            if let Some(pt) = pt {
                if let Some(i) = spot_at(plane, [pt.x, pt.y]) {
                    let spot = &plane.spots[i];
                    if inner.response.clicked() {
                        self.edit_net(NetEdit::SetNeuronParams {
                            ids: vec![id],
                            params: Some(spot.params.clone()),
                        });
                        self.status = format!("neuron {id} ← {}", spot.name);
                    } else {
                        inner
                            .response
                            .clone()
                            .on_hover_text(format!("{} — click to apply", spot.name));
                    }
                }
            }
        }
        if let Some(w) = why_edit {
            ui.weak(w);
        }
    }

    pub(super) fn inspect_one(&mut self, ui: &mut egui::Ui, inst_id: u32) {
        let Some(g) = self.geo.clone() else { return };
        match self.entity(inst_id) {
            Some(Entity::Neuron(id)) => {
                let i = id as usize;
                if i >= g.pos.len() {
                    return;
                }
                ui.label(format!("neuron {}", self.watch_label(id)));
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
                let mut eff = Vec::new();
                if let (Some(t), Some(m)) = (trow, mdesc.as_ref()) {
                    // The row the map marker sits on: the neuron's own
                    // params when overridden, else the type's.
                    eff = g
                        .params
                        .get(i)
                        .and_then(|p| p.clone())
                        .unwrap_or_else(|| t.params.clone());
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
                        let over = g.params.get(i).is_some_and(|p| p.is_some());
                        for (pi, pd) in m.params.iter().enumerate() {
                            ui.horizontal(|ui| {
                                let mut v = eff.get(pi).copied().unwrap_or(pd.default);
                                let speed = ((pd.max - pd.min) / 100.0).abs().max(0.001);
                                let dv = egui::DragValue::new(&mut v)
                                    .speed(speed)
                                    .range(pd.min..=pd.max);
                                let r = ui.add(dv).on_hover_text(&pd.doc);
                                ui.label(param_label(pd));
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
                // ---- live traces + the model's design map (Stage 22) ----
                // Slot 0 is the membrane trace; watching is automatic via
                // the selection (update_watch sends Watch for selected
                // neurons — no manual watch step).
                if let Some(m) = mdesc.as_ref() {
                    ui.separator();
                    self.neuron_trace(ui, id, 0, m);
                    if m.map.is_some() {
                        self.neuron_map(ui, id, m, &eff, why_edit);
                    } else {
                        for s in 1..m.state.len() {
                            self.neuron_trace(ui, id, s, m);
                        }
                    }
                }
                if ui
                    .small_button("raster/phase → Plots")
                    .on_hover_text("open the Plots window on this neuron's traces")
                    .clicked()
                {
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

    /// Stage 21 defect: a param with an empty unit must render its bare
    /// name — `b`, not `b []`.
    /// Sabotage: `format!("{} [{}]", name, unit)` unconditionally → the
    /// empty-unit assert fails with "b []".
    #[test]
    fn empty_unit_shows_name_only() {
        let mk = |name: &str, unit: &str| proto::ParamDesc {
            name: name.into(),
            unit: unit.into(),
            default: 0.0,
            min: 0.0,
            max: 1.0,
            doc: String::new(),
        };
        assert_eq!(param_label(&mk("b", "")), "b");
        assert_eq!(param_label(&mk("a", "1/ms")), "a [1/ms]");
    }

    /// Click-a-region: a point inside a spot's 5% ellipse selects it,
    /// the nearest spot wins on a borderline, and empty space misses.
    /// Sabotage: `Some(0)` for every point → the miss assert fails.
    #[test]
    fn spot_at_hits_region_then_nearest() {
        let plane = proto::MapPlaneDesc {
            x: 0,
            y: 1,
            x_range: (0.0, 0.15),
            y_range: (0.0, 0.3),
            spots: vec![
                proto::MapSpot {
                    label: "RS".into(),
                    name: "regular spiking".into(),
                    params: vec![0.02, 0.2, -65.0, 8.0],
                },
                proto::MapSpot {
                    label: "FS".into(),
                    name: "fast spiking".into(),
                    params: vec![0.1, 0.2, -65.0, 2.0],
                },
            ],
        };
        // Dead centre on RS.
        assert_eq!(spot_at(&plane, [0.02, 0.2]), Some(0));
        // Inside the 5% ellipse of FS (x edge).
        assert_eq!(spot_at(&plane, [0.106, 0.2]), Some(1));
        // Just outside it → nothing (spots own a neighbourhood, not a
        // Voronoi cell — empty space misses).
        assert_eq!(spot_at(&plane, [0.07, 0.2]), None);
        // Far corner → nothing.
        assert_eq!(spot_at(&plane, [0.05, 0.29]), None);
        // Two spots close enough to overlap: the nearer wins.
        let mut close = plane.clone();
        close.spots.push(proto::MapSpot {
            label: "X".into(),
            name: "near-RS".into(),
            params: vec![0.023, 0.2, -65.0, 8.0],
        });
        assert_eq!(spot_at(&close, [0.0215, 0.2]), Some(2));
        // A spot with a truncated row can't be hit (defensive).
        let mut broken = plane.clone();
        broken.spots[0].params.truncate(1);
        assert_eq!(spot_at(&broken, [0.02, 0.2]), None);
    }

    /// The inspector is a real dock: it carves a bounded strip off the
    /// right edge (beside the left dock and above the console) at both
    /// required sizes, and the scene keeps a usable centre.
    /// Sabotage: an `egui::Window` instead of the SidePanel leaves
    /// `available_rect` untouched → `right_taken` = 0 → fails.
    #[test]
    fn inspector_dock_carves_right_space() {
        for (w, h) in [(800.0_f32, 700.0_f32), (1440.0, 900.0)] {
            let ctx = egui::Context::default();
            let mut a = app(&ctx);
            a.win.inspector = true;
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
                        a.inspector(ctx);
                        a.console(ctx);
                        avail = ctx.available_rect();
                    },
                );
            }
            let right_taken = w - avail.max.x;
            assert!(
                (220.0..=520.0).contains(&right_taken),
                "inspector dock took {right_taken:.0}px at {w}x{h} — docked?"
            );
            assert!(
                avail.min.x >= 150.0 && avail.max.y <= h - 60.0,
                "docks crossed: {avail:?} at {w}x{h}"
            );
            assert!(
                avail.width() >= 200.0 && avail.height() >= 300.0,
                "docks ate the scene: {avail:?} at {w}x{h}"
            );
        }
    }

    /// The rule "the Plots window can never cover the inspector" is
    /// structural: floating windows clamp to `usable_rect`, which is the
    /// space left after the docks are shown. Assert that rect at both
    /// required sizes can't touch either dock's strip — any window in it
    /// is provably clear.
    /// Sabotage: show the inspector after `plots()` (or as a window) →
    /// `avail` reaches the right edge → fails.
    #[test]
    fn usable_rect_excludes_all_docks() {
        for (w, h) in [(800.0_f32, 700.0_f32), (1440.0, 900.0)] {
            let ctx = egui::Context::default();
            let mut a = app(&ctx);
            a.win.inspector = true;
            a.win.console = true;
            let mut u = egui::Rect::ZERO;
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
                        a.inspector(ctx);
                        a.console(ctx);
                        u = usable_rect(ctx);
                    },
                );
            }
            assert!(u.min.x >= 150.0, "usable reaches under the left dock");
            assert!(u.max.x <= w - 220.0, "usable reaches under the inspector");
            assert!(u.max.y <= h - 60.0, "usable reaches under the console");
        }
    }
}
