//! Editor actions: selection/watch, insert/connect/move, load/save
//! guards, copy/paste/delete. Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;

use crate::gpu::Entity;
use crate::scene::Particles;
use proto::{ClientMsg, NetEdit};

use super::{Dialog, EditorApp, MoveDrag, PendingAction, PlotTab};

impl EditorApp {
    /// Context-menu helper: watch a neuron's v(t) by selecting it (the
    /// plots derive their watch list from the selection).
    pub fn watch(&mut self, ids: Vec<u32>) {
        self.selection = ids.into_iter().collect();
    }

    /// Plot/legend label for a watched neuron: its name when set, the
    /// id always — "n7" style falls back to the bare id.
    pub(super) fn watch_label(&self, id: u32) -> String {
        let name = self
            .geo
            .as_ref()
            .and_then(|g| g.names.get(id as usize))
            .map(String::as_str)
            .unwrap_or("");
        if name.is_empty() {
            format!("n{id}")
        } else {
            format!("{name} (#{id})")
        }
    }

    /// Context-menu helper: delete one neuron.
    pub fn delete_entity(&mut self, id: u32) {
        self.edit_net(NetEdit::DeleteNeurons { ids: vec![id] });
    }

    /// Context-menu helper: select an input so its value slider shows in
    /// the inspector.
    pub fn open_input_dialog(&mut self, k: u32) {
        if let Some(g) = &self.geo {
            let inst = g.pos.len() as u32 + k;
            self.selection = [inst].into_iter().collect();
            self.show_inspector = true;
        }
    }

    /// Context-menu helper: jump the plots panel to this readout's weights.
    pub fn plot_readout_weights(&mut self, k: u32) {
        self.plots.ro_sel = k as usize;
        self.plots.tab = PlotTab::Readouts;
        self.show_plots = true;
    }

    /// Fit the camera on the whole network.
    pub fn frame_all(&mut self) {
        let Some(g) = &self.geo else { return };
        let (mut mn, mut mx) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
        for p in &g.pos {
            mn = mn.min((*p).into());
            mx = mx.max((*p).into());
        }
        if mn.x <= mx.x {
            self.camera.frame(mn, mx);
        }
    }

    /// Load a file through the worker. `net_path`/recents update on the
    /// server's `NetLoaded` ack (Q7) — a failed load leaves the document
    /// named as it was.
    pub fn load_net(&mut self, path: String) {
        self.send(ClientMsg::LoadNet { path });
    }

    pub fn push_recent(&mut self, path: String) {
        self.files.recent.retain(|p| *p != path);
        self.files.recent.push_front(path);
        self.files.recent.truncate(8);
    }

    /// Selection → watched neurons for v plots (8 lowest selected ids —
    /// a HashSet's iteration order is effectively random, U8).
    pub(super) fn update_watch(&mut self) {
        let mut ids: Vec<u32> = self
            .selection
            .iter()
            .filter_map(|&i| match self.entity(i) {
                Some(Entity::Neuron(n)) => Some(n),
                _ => None,
            })
            .collect();
        ids.sort_unstable();
        ids.truncate(8);
        if ids != self.watch_sent {
            self.watch_sent = ids.clone();
            self.send(ClientMsg::Watch(ids));
            self.plots.hist.v.retain(|k, _| self.watch_sent.contains(k));
            self.plots.hist.u.retain(|k, _| self.watch_sent.contains(k));
        }
    }

    pub fn entity(&self, inst_id: u32) -> Option<Entity> {
        self.geo
            .as_ref()
            .and_then(|g| crate::view::entity_of(g, inst_id))
    }

    /// Place a neuron on the work plane (insert mode).
    pub fn insert_neuron(&mut self, p: glam::Vec3) {
        let kind = if self.insert_inh {
            proto::NeuronKind::Inhibitory
        } else {
            proto::NeuronKind::Excitatory
        };
        self.edit_net(NetEdit::AddNeuron {
            pos: [p.x, p.y, p.z],
            kind,
        });
        self.status = format!("placed neuron at ({:.0},{:.0},{:.0})", p.x, p.y, p.z);
    }

    /// Finish a connect drag: synapse a→b with default weight by kind.
    pub fn connect_neurons(&mut self, a: u32, b: u32) {
        if a == b {
            self.status = format!("can't connect neuron {a} to itself");
            return;
        }
        let inh = self
            .geo
            .as_ref()
            .and_then(|g| g.kind.get(a as usize))
            .is_some_and(|&k| k != 0);
        self.edit_net(NetEdit::Connect {
            from: a,
            to: b,
            w: if inh { 8.0 } else { 6.0 },
        });
        self.status = format!("connected {a} -> {b}");
    }

    /// Connect mode: input marker `k` (instance index n+k) → neuron `t`.
    /// Appends to the input's target list — existing targets survive.
    pub fn connect_input(&mut self, k: u32, t: u32) {
        let name = self
            .geo
            .as_ref()
            .and_then(|g| g.inputs.get(k as usize))
            .map(|i| i.name.clone());
        let Some(name) = name else { return };
        self.edit_net(NetEdit::AddInputTarget {
            name: name.clone(),
            target: t,
            w: 6.0,
        });
        self.status = format!("input {name} -> {t}");
    }

    /// Apply a finished move drag (delta on the work plane).
    pub fn apply_move(&mut self, md: &MoveDrag, delta: glam::Vec3) {
        let mut neurons = Vec::new();
        let mut inputs = Vec::new();
        // Readouts have no stored position (they auto-float above their
        // units) — they can't move; inputs and neurons can.
        for (&inst, &p) in md.ids.iter().zip(md.start_pos.iter()) {
            match self.entity(inst) {
                Some(Entity::Neuron(n)) => neurons.push((n, (p + delta).into())),
                Some(Entity::Input(k)) => inputs.push((k, (p + delta).into())),
                _ => {}
            }
        }
        // Two edits back-to-back: send_edit bases them v+1, v+2 so both
        // apply in order. An input dragged somewhere stores its dropped
        // pos (non-zero → no auto-place snap-back).
        if !neurons.is_empty() {
            self.edit_net(NetEdit::MoveNeurons { items: neurons });
        }
        if !inputs.is_empty() {
            self.edit_net(NetEdit::MoveInputs { items: inputs });
        }
    }

    /// True when the published net has edits since the last save.
    pub(super) fn dirty(&self) -> bool {
        let sh = self.ctl.shared.read().unwrap();
        sh.geo.as_ref().map(|g| g.version).unwrap_or(0) != self.saved_version
    }

    /// Run `act` immediately, or behind the unsaved-changes dialog.
    pub(super) fn guarded(&mut self, ctx: &egui::Context, act: PendingAction) {
        if self.dirty() {
            self.dlg = Some(Dialog::Discard { then: act });
        } else {
            self.do_pending(ctx, act);
        }
    }

    pub(super) fn do_pending(&mut self, ctx: &egui::Context, act: PendingAction) {
        match act {
            PendingAction::New => {
                self.send(ClientMsg::NewNet);
                self.net_path = None;
            }
            PendingAction::Load(p) => self.load_net(p),
            PendingAction::Quit => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    /// New/load/`net.clear()` — drop every piece of UI state that refers
    /// to the old net's ids, and ask for a reframe on the new geometry.
    pub(super) fn reset_net_state(&mut self) {
        self.selection.clear();
        self.connect_from = None;
        self.move_drag = None;
        self.preview_delta = None;
        self.box_sel = None;
        self.hover = None;
        self.pick_click = None;
        if !self.watch_sent.is_empty() {
            self.watch_sent.clear();
            self.send(ClientMsg::Watch(Vec::new()));
        }
        self.plots.hist.clear();
        self.particles = Particles::new();
        self.activity.clear();
        self.grid_traces.clear();
        self.input_values = Default::default();
        self.plane_cfgs.clear();
        self.frame_pending = true;
        // The replaced net starts clean — its version is the saved marker.
        let sh = self.ctl.shared.read().unwrap();
        self.saved_version = sh.geo.as_ref().map(|g| g.version).unwrap_or(0);
    }

    pub fn delete_selection(&mut self) {
        let ids: Vec<u32> = self
            .selection
            .iter()
            .filter_map(|&i| match self.entity(i) {
                Some(Entity::Neuron(n)) => Some(n),
                _ => None,
            })
            .collect();
        if ids.is_empty() {
            return;
        }
        self.edit_net(NetEdit::DeleteNeurons { ids });
        self.selection.clear();
    }

    /// Serialize the selected neurons (+ internal synapses/groups) to JSON on
    /// the worker; the UI picks it up via `Shared::clipboard` next frame.
    pub fn copy_selection(&mut self, cut: bool) {
        let ids: std::collections::HashSet<u32> = self
            .selection
            .iter()
            .filter_map(|&i| match self.entity(i) {
                Some(Entity::Neuron(n)) => Some(n),
                _ => None,
            })
            .collect();
        if ids.is_empty() {
            return;
        }
        self.send(ClientMsg::CopySubnet {
            ids: ids.into_iter().collect(),
        });
        if cut {
            self.delete_selection();
        }
    }

    /// Paste a subnet JSON (`text` wins over the internal clipboard).
    pub fn paste(&mut self, text: Option<String>) {
        let Some(txt) = text.or_else(|| self.clipboard.clone()) else {
            return;
        };
        self.edit_net(NetEdit::InsertSubnetJson {
            json: txt,
            offset: [30.0, 0.0, 30.0],
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[test]
    fn watch_takes_lowest_ids_not_hash_order() {
        // U8: selection is a HashSet — iterating it gave an arbitrary
        // 8 of the selected neurons. Now the 8 lowest ids win.
        let mut app = EditorApp::new(crate::simctl::SimCtl::offline(), None);
        let mut geo = crate::snap::GeoSnap::default();
        geo.pos = vec![[0.0; 3]; 20];
        app.geo = Some(std::sync::Arc::new(geo));
        app.selection = [15, 3, 9, 1, 12, 7, 19, 2, 11, 5].into_iter().collect();
        app.update_watch();
        assert_eq!(app.watch_sent, vec![1, 2, 3, 5, 7, 9, 11, 12]);
    }
}
