//! Inspector: a resizable right-hand dock (Stage 22) with the selected
//! entity's props, the auto-watched V(t) trace, and the model's
//! parameter design map (izh a–b/c–d) or one trace per state slot.
//! The Inputs window (floating) also lives here.

use eframe::egui;
use egui_plot::{Line, PlotPoints};

use crate::gpu::Entity;
use proto::{ClientMsg, NetEdit};

use super::layout::{app_win, default_rect, narrow_mode, usable_rect, DefWin};
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

/// Fraction of the axis range one label character is assumed to occupy
/// (≈10 px of text on a ~230 px map) — deliberately generous so the
/// collision resolver errs toward separating labels.
const MAP_LABEL_CHAR_W: f64 = 0.034;
/// Fraction of the y range per stacked label line.
const MAP_LABEL_LINE_H: f64 = 0.09;
/// A spot's shaded region is exactly `spot_at`'s 5%-of-range ellipse —
/// what you see is what you can click.
const MAP_REGION_FRAC: f64 = 0.05;

/// One laid-out map label: the `Text` anchor point, which corner of the
/// text sits there, and the data-space rect the text is estimated to
/// cover (the collision resolver's currency — also what tests assert).
pub(super) struct MapLabel {
    pub spot: usize,
    pub pos: [f64; 2],
    pub align: egui::Align2,
    /// The collision resolver's currency — asserted in tests, not
    /// needed by the draw itself (Text anchors at `pos`).
    #[allow(dead_code)]
    pub rect: [f64; 4],
}

/// (center, member spot indices) — spots sharing a centre (within
/// ~0.1% of a range) draw concentric ellipses and share one label
/// column. Order = first-seen spot order, stable.
pub(super) fn map_clusters(plane: &proto::MapPlaneDesc) -> Vec<([f64; 2], Vec<usize>)> {
    let xr = plane.x_range.1 - plane.x_range.0;
    let yr = plane.y_range.1 - plane.y_range.0;
    let mut out: Vec<([f64; 2], Vec<usize>)> = Vec::new();
    for (i, s) in plane.spots.iter().enumerate() {
        let (Some(&px), Some(&py)) = (s.params.get(plane.x), s.params.get(plane.y)) else {
            continue;
        };
        let c = [px as f64, py as f64];
        if let Some(cl) = out.iter_mut().find(|(cc, _)| {
            (cc[0] - c[0]).abs() < xr as f64 * 1e-3 && (cc[1] - c[1]).abs() < yr as f64 * 1e-3
        }) {
            cl.1.push(i);
        } else {
            out.push((c, vec![i]));
        }
    }
    out
}

/// The spot's shaded click region, centred on its param pair — the
/// same 5%-of-range ellipse [`spot_at`] hit-tests.
pub(super) fn spot_ellipse(
    plane: &proto::MapPlaneDesc,
    s: &proto::MapSpot,
) -> Option<([f64; 2], [f64; 2])> {
    let xr = (plane.x_range.1 - plane.x_range.0) as f64;
    let yr = (plane.y_range.1 - plane.y_range.0) as f64;
    let (Some(&px), Some(&py)) = (s.params.get(plane.x), s.params.get(plane.y)) else {
        return None;
    };
    Some((
        [px as f64, py as f64],
        [xr * MAP_REGION_FRAC, yr * MAP_REGION_FRAC],
    ))
}

/// Position every spot's label so none overlap each other or the
/// current-params marker:
/// - coincident spots share one column whose labels fan vertically;
/// - the column sits just right of the cluster's ellipse (left when
///   the cluster is in the right ~40 % of the plane);
/// - columns that still collide get pushed apart vertically;
/// - a column covering the marker flips to the other side.
pub(super) fn map_labels(plane: &proto::MapPlaneDesc, marker: Option<[f64; 2]>) -> Vec<MapLabel> {
    let xr = (plane.x_range.1 - plane.x_range.0) as f64;
    let yr = (plane.y_range.1 - plane.y_range.0) as f64;
    if xr <= 0.0 || yr <= 0.0 {
        return Vec::new();
    }
    let char_w = MAP_LABEL_CHAR_W * xr;
    let line_h = MAP_LABEL_LINE_H * yr;
    let rx = xr * MAP_REGION_FRAC;
    let clusters = map_clusters(plane);
    // column side + per-line centres before collision resolution
    struct Col {
        members: Vec<usize>,
        c: [f64; 2],
        right: bool,
        ys: Vec<f64>,
    }
    let mut cols: Vec<Col> = clusters
        .iter()
        .map(|(c, members)| {
            let n = members.len() as f64;
            let ys = members
                .iter()
                .enumerate()
                .map(|(k, _)| c[1] + (k as f64 - (n - 1.0) / 2.0) * line_h)
                .collect();
            Col {
                members: members.clone(),
                c: *c,
                right: c[0] <= plane.x_range.0 as f64 + xr * 0.6,
                ys,
            }
        })
        .collect();
    let rect = |col: &Col, k: usize| -> [f64; 4] {
        let s = &plane.spots[col.members[k]];
        let w = s.label.chars().count().max(1) as f64 * char_w;
        let ax = if col.right {
            col.c[0] + rx * 1.5
        } else {
            col.c[0] - rx * 1.5
        };
        let cy = col.ys[k];
        if col.right {
            [ax, ax + w, cy - line_h / 2.0, cy + line_h / 2.0]
        } else {
            [ax - w, ax, cy - line_h / 2.0, cy + line_h / 2.0]
        }
    };
    let overlap = |a: [f64; 4], b: [f64; 4]| -> f64 {
        let ox = a[1].min(b[1]) - a[0].max(b[0]);
        let oy = a[3].min(b[3]) - a[2].max(b[2]);
        if ox > 0.0 && oy > 0.0 {
            oy
        } else {
            0.0
        }
    };
    // Push colliding columns apart vertically (the later one moves
    // toward its own side of the plane). Four passes suffice for the
    // handful of clusters a real map has.
    for _ in 0..4 {
        let mut moved = false;
        for j in 0..cols.len() {
            for i in 0..j {
                let shift = (0..cols[j].members.len())
                    .flat_map(|kj| (0..cols[i].members.len()).map(move |ki| (kj, ki)))
                    .map(|(kj, ki)| overlap(rect(&cols[j], kj), rect(&cols[i], ki)))
                    .fold(0.0_f64, f64::max);
                if shift > 0.0 {
                    let down = cols[j].c[1] < cols[i].c[1];
                    let d = (shift + line_h * 0.25) * if down { -1.0 } else { 1.0 };
                    for y in &mut cols[j].ys {
                        *y += d;
                    }
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
    }
    // A column covering the marker flips to the other side.
    if let Some(m) = marker {
        for col in &mut cols {
            if (0..col.members.len()).any(|k| {
                let r = rect(col, k);
                m[0] >= r[0] && m[0] <= r[1] && m[1] >= r[2] && m[1] <= r[3]
            }) {
                col.right = !col.right;
            }
        }
    }
    let mut out = Vec::new();
    for col in &cols {
        for (k, &spot) in col.members.iter().enumerate() {
            let r = rect(col, k);
            out.push(MapLabel {
                spot,
                pos: if col.right {
                    [r[0], col.ys[k]]
                } else {
                    [r[1], col.ys[k]]
                },
                align: if col.right {
                    egui::Align2::LEFT_CENTER
                } else {
                    egui::Align2::RIGHT_CENTER
                },
                rect: r,
            });
        }
    }
    out
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
    ///
    /// Stage 22B narrow rule (`NARROW_W`, ~1100 px): the dock becomes a
    /// slim rail that auto-expands the moment a selection exists, and
    /// returns to the rail when the selection empties. The » control
    /// collapses back to the rail; Ctrl+1 / the PANELS toggle still
    /// hide the inspector entirely.
    pub(super) fn inspector(&mut self, ctx: &egui::Context) {
        if !self.win.inspector {
            return;
        }
        let narrow = narrow_mode(ctx);
        if narrow {
            let has_sel = !self.selection.is_empty();
            if has_sel && !self.win.insp_narrow_sel_seen {
                self.win.insp_narrow_open = true;
            }
            self.win.insp_narrow_sel_seen = has_sel;
            if !self.win.insp_narrow_open {
                egui::SidePanel::right("inspector_rail")
                    .resizable(false)
                    .exact_width(super::panel::RAIL_W)
                    .show(ctx, |ui| {
                        ui.spacing_mut().interact_size.y = super::panel::PANEL_CONTROL_H;
                        if ui.button("«").on_hover_text("open inspector").clicked() {
                            self.win.insp_narrow_open = true;
                        }
                        ui.separator();
                        ui.weak("…");
                    });
                return;
            }
        }
        let (def_w, lo_w, hi_w) = if narrow {
            (260.0, 180.0, 360.0)
        } else {
            (300.0, 220.0, 520.0)
        };
        egui::SidePanel::right("inspector_dock")
            .resizable(true)
            .default_width(def_w)
            .width_range(lo_w..=hi_w)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Inspector");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button("»")
                            .on_hover_text(if narrow {
                                "collapse to the rail (reopens on selection)"
                            } else {
                                "hide inspector (Ctrl+1 to reopen)"
                            })
                            .clicked()
                        {
                            if narrow {
                                self.win.insp_narrow_open = false;
                            } else {
                                self.win.inspector = false;
                            }
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
                            if !g.groups.is_empty() {
                                // Stage 23: one control family — the
                                // group's view IS its interaction mode.
                                // A collapsed block is always whole.
                                ui.horizontal(|ui| {
                                    ui.weak("default:");
                                    for (v, l) in [
                                        (super::GroupMode::Whole, "whole"),
                                        (super::GroupMode::Inside, "inside"),
                                    ] {
                                        ui.selectable_value(&mut self.group_mode_default, v, l);
                                    }
                                });
                            }
                            for (gi, (name, members)) in g.groups.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    ui.label(format!("{name}: {} neurons", members.len()));
                                    let mut mode = self.group_mode(name);
                                    for (v, l, tip) in [
                                        (
                                            super::GroupMode::Whole,
                                            "whole",
                                            "clicks select the group, drags move every member",
                                        ),
                                        (
                                            super::GroupMode::Inside,
                                            "inside",
                                            "clicks and drags act on single members",
                                        ),
                                        (
                                            super::GroupMode::Block,
                                            "block",
                                            "collapse to a labelled block — always whole-group",
                                        ),
                                    ] {
                                        if ui
                                            .selectable_label(mode == v, l)
                                            .on_hover_text(tip)
                                            .clicked()
                                        {
                                            mode = v;
                                            self.group_modes.insert(name.clone(), v);
                                        }
                                    }
                                });
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
            // The marker sits on the neuron's own param row — labels
            // are laid out around it (map_labels flips a covering
            // column to the other side of its cluster).
            let marker = match (eff.get(plane.x), eff.get(plane.y)) {
                (Some(&cx), Some(&cy)) => Some([cx as f64, cy as f64]),
                _ => None,
            };
            let labels = map_labels(plane, marker);
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
                        // Shaded click regions first — each is the same
                        // 5%-of-range ellipse `spot_at` hit-tests, so the
                        // visible region IS the clickable one (Stage 22B:
                        // the map reads as a design map, not dots+text).
                        for (si, s) in plane.spots.iter().enumerate() {
                            if let Some((c, r)) = spot_ellipse(plane, s) {
                                let col = crate::plot::watch_color(si);
                                let pts: Vec<[f64; 2]> = (0..32)
                                    .map(|k| {
                                        let a = k as f64 / 32.0 * std::f64::consts::TAU;
                                        [c[0] + r[0] * a.cos(), c[1] + r[1] * a.sin()]
                                    })
                                    .collect();
                                ui.polygon(
                                    egui_plot::Polygon::new("", PlotPoints::from(pts))
                                        .fill_color(col.gamma_multiply(0.18))
                                        .stroke(egui::Stroke::new(1.0_f32, col)),
                                );
                            }
                        }
                        // Labels — clustered, fanned, collision-free
                        // (map_labels); each tinted its region's colour.
                        for l in &labels {
                            let s = &plane.spots[l.spot];
                            ui.text(
                                egui_plot::Text::new(
                                    "",
                                    egui_plot::PlotPoint::new(l.pos[0], l.pos[1]),
                                    egui::RichText::new(&s.label)
                                        .strong()
                                        .size(10.0)
                                        .color(crate::plot::watch_color(l.spot)),
                                )
                                .anchor(l.align),
                            );
                        }
                        // The current-params marker last — always on top.
                        if let Some(m) = marker {
                            ui.points(
                                egui_plot::Points::new("", vec![m])
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
                // Stage 22B: this row read as a stray widget — make it
                // an explicit action.
                if ui
                    .button("Open in Plots window")
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
    /// required sizes, and the scene keeps a usable centre. Stage 22B:
    /// at narrow widths (< NARROW_W) with nothing selected it is only
    /// the slim rail; with a selection it opens as a bounded dock.
    /// Sabotage: an `egui::Window` instead of the SidePanel leaves
    /// `available_rect` untouched → `right_taken` = 0 → fails.
    #[test]
    fn inspector_dock_carves_right_space() {
        for (w, h) in [(800.0_f32, 700.0_f32), (1440.0, 900.0)] {
            // Nothing selected: wide stays a dock, narrow falls to the
            // rail (exact narrow rule — see NOTES Stage 22B).
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
            if narrow_mode(&ctx) {
                assert!(
                    (28.0..=60.0).contains(&right_taken),
                    "no selection at {w}: inspector should be a rail, took {right_taken:.0}px"
                );
            } else {
                assert!(
                    (220.0..=520.0).contains(&right_taken),
                    "inspector dock took {right_taken:.0}px at {w}x{h} — docked?"
                );
            }
            // Selection: the dock opens at BOTH sizes, bounded, and the
            // narrow scene keeps ≥45% of the width.
            let ctx = egui::Context::default();
            let mut a = app(&ctx);
            a.win.inspector = true;
            a.win.console = true;
            a.selection.insert(0);
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
            let (lo, hi) = if narrow_mode(&ctx) {
                (180.0, 360.0)
            } else {
                (220.0, 520.0)
            };
            assert!(
                (lo..=hi).contains(&right_taken),
                "selected at {w}: inspector took {right_taken:.0}px, want {lo}..={hi}"
            );
            assert!(
                avail.width() >= w * 0.45,
                "docks ate the scene at {w}x{h}: {:.0}px of {w}",
                avail.width()
            );
        }
    }

    /// The rule "the Plots window can never cover the inspector" is
    /// structural: floating windows clamp to `usable_rect`, which is the
    /// space left after the docks are shown. Assert that rect at both
    /// required sizes can't touch either dock's strip — any window in it
    /// is provably clear. (A selection is held so the narrow inspector
    /// is a dock, not the rail — the stricter case.)
    /// Sabotage: show the inspector after `plots()` (or as a window) →
    /// `avail` reaches the right edge → fails.
    #[test]
    fn usable_rect_excludes_all_docks() {
        for (w, h) in [(800.0_f32, 700.0_f32), (1440.0, 900.0)] {
            let ctx = egui::Context::default();
            let mut a = app(&ctx);
            a.win.inspector = true;
            a.win.console = true;
            a.selection.insert(0);
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
            let (left_min, insp_min) = if narrow_mode(&ctx) {
                (28.0, 180.0) // tools rail, inspector dock min
            } else {
                (150.0, 220.0)
            };
            assert!(
                u.min.x >= left_min,
                "usable reaches under the left dock/rail"
            );
            assert!(
                u.max.x <= w - insp_min,
                "usable reaches under the inspector"
            );
            assert!(u.max.y <= h - 60.0, "usable reaches under the console");
        }
    }

    /// izh fixture with the real a–b spot coordinates (stage-20 map).
    fn izh_ab_plane() -> proto::MapPlaneDesc {
        let mk = |label: &str, a: f32, b: f32| proto::MapSpot {
            label: label.into(),
            name: label.into(),
            params: vec![a, b, -65.0, 8.0],
        };
        proto::MapPlaneDesc {
            x: 0,
            y: 1,
            x_range: (0.0, 0.2),
            y_range: (0.0, 0.3),
            spots: vec![
                mk("RS", 0.02, 0.2),
                mk("IB", 0.02, 0.2),
                mk("CH", 0.02, 0.2),
                mk("FS", 0.1, 0.2),
                mk("TH", 0.02, 0.25),
                mk("LTS", 0.02, 0.25),
            ],
        }
    }

    /// Stage 22B defect: a–b labels sat on top of each other — three
    /// coincident spots (RS/IB/CH), a second coincident pair (TH/LTS)
    /// and a marker on the shared pile all drew at the same point.
    /// Every label rect must be pairwise disjoint and none may cover
    /// the marker.
    /// Sabotage: `map_labels` returning `pos = spot centre` for all →
    /// the coincident RS/IB rects overlap → fails.
    #[test]
    fn map_labels_never_overlap() {
        for plane in [izh_ab_plane(), {
            let mut p = izh_ab_plane();
            p.x = 2;
            p.y = 3;
            p.x_range = (-100.0, -30.0);
            p.y_range = (0.0, 10.0);
            p.spots = vec![
                proto::MapSpot {
                    label: "RS".into(),
                    name: "RS".into(),
                    params: vec![0.02, 0.2, -65.0, 8.0],
                },
                proto::MapSpot {
                    label: "IB".into(),
                    name: "IB".into(),
                    params: vec![0.02, 0.2, -55.0, 4.0],
                },
                proto::MapSpot {
                    label: "CH".into(),
                    name: "CH".into(),
                    params: vec![0.02, 0.2, -50.0, 2.0],
                },
                proto::MapSpot {
                    label: "FS".into(),
                    name: "FS".into(),
                    params: vec![0.1, 0.2, -65.0, 2.0],
                },
                proto::MapSpot {
                    label: "LTS".into(),
                    name: "LTS".into(),
                    params: vec![0.02, 0.25, -65.0, 2.0],
                },
                proto::MapSpot {
                    label: "RZ".into(),
                    name: "RZ".into(),
                    params: vec![0.1, 0.16, -65.0, 2.0],
                },
                proto::MapSpot {
                    label: "TC".into(),
                    name: "TC".into(),
                    params: vec![0.02, 0.25, -80.0, 4.0],
                },
            ];
            p
        }] {
            // Marker parked ON a spot — the worst case.
            let mrow = plane.spots[0].params.clone();
            let marker = [mrow[plane.x] as f64, mrow[plane.y] as f64];
            let labels = map_labels(&plane, Some(marker));
            assert_eq!(labels.len(), plane.spots.len());
            for (i, a) in labels.iter().enumerate() {
                for b in &labels[i + 1..] {
                    let ox = a.rect[1].min(b.rect[1]) - a.rect[0].max(b.rect[0]);
                    let oy = a.rect[3].min(b.rect[3]) - a.rect[2].max(b.rect[2]);
                    assert!(
                        ox <= 0.0 || oy <= 0.0,
                        "labels {:?} and {:?} overlap",
                        a.rect,
                        b.rect
                    );
                }
                assert!(
                    marker[0] < a.rect[0]
                        || marker[0] > a.rect[1]
                        || marker[1] < a.rect[2]
                        || marker[1] > a.rect[3],
                    "label {:?} covers marker {marker:?}",
                    a.rect
                );
            }
        }
    }

    /// Every spot owns a visible ellipse centred on its params, exactly
    /// `spot_at`'s hit region — the map's "what you see is what you
    /// click" invariant.
    /// Sabotage: centre the ellipse at (0,0) → the centre assert fails.
    #[test]
    fn map_ellipses_match_hit_regions() {
        let plane = izh_ab_plane();
        let xr = (plane.x_range.1 - plane.x_range.0) as f64;
        for s in &plane.spots {
            let (c, r) = spot_ellipse(&plane, s).unwrap();
            assert_eq!(c[0] as f32, s.params[0]);
            assert_eq!(c[1] as f32, s.params[1]);
            assert!((r[0] - xr * MAP_REGION_FRAC).abs() < 1e-6);
            // And the ellipse centre must hit-test to this region.
            assert!(spot_at(&plane, c).is_some());
        }
    }
}
