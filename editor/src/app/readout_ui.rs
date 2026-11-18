//! Stage 26: the readout inspector — the MLP's layer list (input width,
//! each hidden layer with size + activation, output head), an editable
//! draft of the readout's wanted hidden layers, and a Train button that
//! re-runs the script which built the net (examples read the readout's
//! layers before `net.clear()` and train with them).

use eframe::egui;

use proto::{ClientMsg, NetEdit, ReadoutMlp};

use super::{gate, gate_ui, Act, EditorApp};

/// The inspector's editable copy of one readout's wanted layers.
/// Re-seeded whenever the selected readout or its published config
/// changes, so an Undo or a script edit shows up immediately.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RoDraft {
    pub name: String,
    /// Published config + trained hidden layers this draft was seeded
    /// from.
    pub seeded_from: (Vec<u32>, Vec<u32>),
    pub layers: Vec<u32>,
}

/// Default hidden width for a newly added layer row.
const NEW_LAYER: u32 = 64;

/// "4096 → 128 → 64 → 4" — compact shape text.
pub(crate) fn shape_text(sizes: &[u32]) -> String {
    sizes
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(" → ")
}

/// One line per MLP layer as the scene stack labels them:
/// ("4096", "in"), ("128", "tanh"), …, ("4", "softmax"|"score").
pub(crate) fn layer_labels(m: &ReadoutMlp) -> Vec<(u32, String)> {
    let n = m.sizes.len();
    m.sizes
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let tag = if i == 0 {
                "in".to_string()
            } else if i + 1 == n {
                if m.softmax { "softmax" } else { "score" }.to_string()
            } else {
                m.act.clone()
            };
            (s, tag)
        })
        .collect()
}

impl EditorApp {
    /// Readout section of the inspector (selected readout `k`).
    pub(super) fn inspect_readout(&mut self, ui: &mut egui::Ui, k: usize) {
        let Some(g) = self.geo.clone() else { return };
        let Some((name, units, w)) = g.readouts.get(k) else {
            return;
        };
        let m = g.readout_mlps.get(k).cloned().unwrap_or_default();
        ui.label(format!("readout: {name}"));
        ui.label(format!("{} units", units.len()));
        if w.is_empty() {
            ui.label("linear weights: —");
        } else {
            ui.label(format!("linear weights: {} (plot in Plots)", w.len()));
        }

        // Seed / re-seed the draft from the published state.
        let seed = (m.layers.clone(), m.trained_hidden().to_vec());
        let reseed = match &self.ro_draft {
            Some(d) => d.name != *name || d.seeded_from != seed,
            None => true,
        };
        if reseed {
            let layers = if !m.layers.is_empty() {
                m.layers.clone()
            } else {
                m.trained_hidden().to_vec()
            };
            self.ro_draft = Some(RoDraft {
                name: name.clone(),
                seeded_from: seed,
                layers,
            });
        }

        ui.separator();
        ui.strong("MLP readout (non-spiking, backprop)");
        let act = if m.act.is_empty() {
            "tanh".to_string()
        } else {
            m.act.clone()
        };
        let why_edit = self.why(Act::Edit);
        let mut draft = self.ro_draft.clone().expect("seeded above");
        let mut remove: Option<usize> = None;
        egui::Grid::new(("ro_layers", k))
            .num_columns(3)
            .striped(true)
            .show(ui, |ui| {
                ui.label("input");
                ui.label(format!("{}", units.len()));
                ui.weak("features");
                ui.end_row();
                for (i, s) in draft.layers.iter_mut().enumerate() {
                    ui.label(format!("hidden {}", i + 1));
                    gate_ui(ui, why_edit, |ui| {
                        ui.add(
                            egui::DragValue::new(s)
                                .range(1..=net_format::MAX_MLP_LAYER)
                                .speed(1.0),
                        )
                        .on_hover_text("units in this layer — drag or type");
                    });
                    ui.horizontal(|ui| {
                        ui.weak(&act);
                        if gate(ui, why_edit, egui::Button::new("✖").small())
                            .on_hover_text("remove this layer")
                            .clicked()
                        {
                            remove = Some(i);
                        }
                    });
                    ui.end_row();
                }
                ui.label("output");
                if m.sizes.is_empty() {
                    ui.weak("—");
                    ui.weak("set by the script");
                } else {
                    let out = *m.sizes.last().unwrap();
                    ui.label(format!("{out}"));
                    ui.weak(if m.softmax { "softmax" } else { "score" });
                }
                ui.end_row();
            });
        if let Some(i) = remove {
            draft.layers.remove(i);
        }
        ui.horizontal(|ui| {
            if gate(
                ui,
                why_edit.or_else(|| {
                    (draft.layers.len() >= net_format::MAX_MLP_DEPTH).then_some("at most 8 layers")
                }),
                egui::Button::new("+ layer"),
            )
            .on_hover_text("append a hidden layer")
            .clicked()
            {
                let s = draft.layers.last().map_or(NEW_LAYER, |&l| (l / 2).max(2));
                draft.layers.push(s);
            }
            let dirty = draft.layers != m.layers;
            if gate(
                ui,
                why_edit.or_else(|| (!dirty).then_some("no change to apply")),
                egui::Button::new("Apply"),
            )
            .on_hover_text("store these layers on the readout (undoable); Train uses them")
            .clicked()
            {
                self.edit_net(NetEdit::SetReadoutLayers {
                    name: name.clone(),
                    layers: draft.layers.clone(),
                });
            }
            if !m.layers.is_empty()
                && gate(ui, why_edit, egui::Button::new("Script default"))
                    .on_hover_text("clear the readout's layer config — the script picks")
                    .clicked()
            {
                self.edit_net(NetEdit::SetReadoutLayers {
                    name: name.clone(),
                    layers: Vec::new(),
                });
                draft.layers = m.trained_hidden().to_vec();
            }
        });
        if m.layers.is_empty() {
            ui.weak("config: script default");
        } else {
            ui.weak(format!("config: {:?}", m.layers));
        }
        let pending = draft.layers != m.layers;
        self.ro_draft = Some(draft);

        // Trained-state line.
        if m.sizes.is_empty() {
            ui.label("trained: none attached");
        } else {
            let head = if m.softmax { "softmax" } else { "score" };
            ui.label(format!("trained: {} ({act}, {head})", shape_text(&m.sizes)));
            if m.stale() {
                ui.colored_label(
                    egui::Color32::from_rgb(230, 160, 40),
                    "⚠ needs retraining — config differs from the trained net",
                );
            } else {
                ui.colored_label(egui::Color32::from_rgb(110, 190, 110), "✔ matches config");
            }
        }
        if pending {
            ui.weak("Apply first — Train uses the stored config.");
        }

        // Train = re-run the script that built this net.
        let why_run = self.why(Act::Eval).or_else(|| {
            self.last_build
                .is_none()
                .then_some("no script built this net — run an example or script first")
        });
        let label = match &self.last_build {
            Some((n, _)) => format!("Train: re-run {n}"),
            None => "Train: re-run script".to_string(),
        };
        if gate(ui, why_run, egui::Button::new(label))
            .on_hover_text(
                "re-runs the script that built this net; the examples read this \
                 readout's layers before net.clear() and train with them — held-out \
                 numbers print in the console",
            )
            .clicked()
        {
            self.rerun_build_script();
        }
    }

    /// Re-run the script that built the current net (Train button).
    pub(super) fn rerun_build_script(&mut self) {
        let Some((name, code)) = self.last_build.clone() else {
            return;
        };
        self.send(ClientMsg::Eval { code });
        self.win.console = true;
        self.status = format!("training: re-running {name} — results in the console");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_name_every_layer() {
        let m = ReadoutMlp {
            layers: vec![128, 64],
            sizes: vec![4096, 128, 64, 4],
            act: "tanh".into(),
            softmax: true,
        };
        let l = layer_labels(&m);
        assert_eq!(l[0], (4096, "in".to_string()));
        assert_eq!(l[1], (128, "tanh".to_string()));
        assert_eq!(l[3], (4, "softmax".to_string()));
        assert!(!m.stale());
        let s = ReadoutMlp {
            layers: vec![128, 64, 32],
            ..m.clone()
        };
        assert!(s.stale());
        assert_eq!(shape_text(&m.sizes), "4096 → 128 → 64 → 4");
    }
}
