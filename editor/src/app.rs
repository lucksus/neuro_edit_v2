//! eframe App: toolbar, 3D viewport (egui-wgpu callback), inspector,
//! console, plots. Communicates with the sim worker via `SimCtl`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use eframe::egui;
use egui_plot::{Line, PlotPoints};

use crate::camera::Camera;
use crate::gpu::{Entity, Gpu};
use crate::scene::{Particles, Plane};
use crate::simctl::{Cmd, SimCtl};
use crate::snap::GeoSnap;

#[allow(dead_code)]
pub struct EditorApp {
    pub ctl: SimCtl,
    pub camera: Camera,
    pub geo: Option<Arc<GeoSnap>>,
    geo_version: u64,
    /// Frame the camera on the next geometry publish (grid dialogs, load).
    frame_pending: bool,
    last_snap_seq: u64,
    pub activity: Vec<f32>,
    pub particles: Particles,
    /// (time_ms, fired-ids) per publish — x axis is real sim milliseconds.
    raster: VecDeque<(f64, Vec<u32>)>,
    v_hist: HashMap<u32, VecDeque<(f64, f32)>>,

    pub selection: HashSet<u32>, // instance ids
    pub mode: Mode,
    pub plane: Plane,
    pub insert: InsertKind,
    pub connect_from: Option<u32>,
    pub box_sel: Option<(egui::Pos2, egui::Pos2)>,
    pub move_drag: Option<MoveDrag>,
    pub preview_delta: Option<glam::Vec3>,
    pub last_pick: Option<u32>,
    /// Instance under the cursor (hover pick, one frame behind).
    pub hover: Option<u32>,
    /// Click waiting for a free pick slot (uv in the viewport). Clicks queue
    /// here so an in-flight hover pick can't drop them.
    pub pick_click: Option<glam::Vec2>,
    pub show_grid: bool,
    pub show_synapses: bool,
    /// Grid-group render mode (activity plane vs per-neuron cubes).
    pub grid_view: crate::plane::GridView,
    /// Per-group 3D display config, keyed by `GeoSnap::grids[].group`.
    pub plane_cfgs: HashMap<u32, crate::plane::PlaneCfg>,
    /// Last snap's per-grid spike traces (fed to the plane textures).
    grid_traces: Vec<std::sync::Arc<Vec<f32>>>,
    /// Last snap's per-input constant values (drives the Inputs panel).
    input_values: std::sync::Arc<Vec<f32>>,
    /// Upload caches for `plane_frame` — textures only re-upload on change.
    plane_snap: u64,
    plane_geo: u64,
    plane_cfgs_last: Vec<crate::plane::PlaneCfg>,

    console_input: String,
    history: Vec<String>,
    hist_idx: Option<usize>,
    show_console: bool,
    show_plots: bool,
    show_inspector: bool,
    show_script: bool,
    plot_tab: PlotTab,
    /// Raster strip height in px — draggable in the plots header.
    raster_h: f32,
    /// Readout index shown in the weights tab.
    ro_sel: usize,
    /// Script editor buffer + the name it is saved under in the file.
    script_text: String,
    script_name: String,
    script_file: String,
    /// Recently opened/saved .ne2 paths (persisted across runs).
    pub recent: VecDeque<String>,

    speed: f32,
    speed_actual: f32,
    running: bool,
    sim_built: bool,
    busy: bool,
    dt_ms: f32,
    noise_hz: f32,
    noise_kick: f32,
    pub net_path: Option<String>,

    dlg: Option<Dialog>,
    /// Last net-replacement generation seen — a change means New/load/
    /// `net.clear()` happened on the worker and UI state must reset.
    last_net_gen: u64,
    /// `GeoSnap::version` at the last save — anything newer = unsaved edits.
    saved_version: u64,
    watch_sent: Vec<u32>,
    pub status: String,
    /// Last subnet JSON we copied (for the Paste menu without OS access).
    clipboard: Option<String>,
    /// Original neuro_edit artwork, loaded on first frame.
    icons: Option<crate::icons::Icons>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Navigate,
    Insert,
    Connect,
    Move,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum InsertKind {
    Neuron,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PlotTab {
    Voltage,
    Raster,
    Readouts,
    Samples,
}

#[derive(Clone)]
pub struct MoveDrag {
    pub plane_pt: glam::Vec3,
    pub start_pos: Vec<glam::Vec3>,
    pub ids: Vec<u32>,
}

/// Action queued behind the unsaved-changes confirm dialog.
#[derive(Clone)]
pub enum PendingAction {
    New,
    Load(String),
    Quit,
}

#[derive(Clone)]
pub enum Dialog {
    /// Unsaved edits — confirm before replacing/abandoning the net.
    Discard {
        then: PendingAction,
    },
    Grid2d {
        group: String,
        w: i32,
        h: i32,
        d: f32,
    },
    Grid3d {
        group: String,
        w: i32,
        h: i32,
        d2: i32,
        spacing: f32,
    },
    Connections {
        group: String,
        scaling: f32,
        lambda: f32,
        w: f32,
    },
    Weights {
        group: String,
        mean: f32,
        sd: f32,
    },
    ExcInh {
        group: String,
        frac: f32,
    },
    NewInput {
        name: String,
        weight: f32,
        value: f32,
    },
    NewSamples {
        name: String,
    },
    NewReadout {
        name: String,
    },
    AddToGroup {
        name: String,
    },
    SaveAs {
        path: String,
    },
    Load {
        path: String,
    },
    About,
}

/// Persisted UI state (recent files) across sessions.
#[derive(serde::Serialize, serde::Deserialize, Default)]
pub struct Persisted {
    pub recent: Vec<String>,
}

impl EditorApp {
    pub fn new(ctl: SimCtl, storage: Option<&dyn eframe::Storage>) -> Self {
        let recent: VecDeque<String> = storage
            .and_then(|s| eframe::get_value::<Persisted>(s, "state"))
            .map(|p| p.recent.into_iter().collect())
            .unwrap_or_default();
        Self {
            recent,
            ctl,
            camera: Camera::default(),
            geo: None,
            geo_version: 0,
            frame_pending: false,
            last_snap_seq: 0,
            activity: Vec::new(),
            particles: Particles::new(),
            raster: VecDeque::with_capacity(600),
            v_hist: HashMap::new(),
            selection: HashSet::new(),
            mode: Mode::Navigate,
            plane: Plane::XZ,
            insert: InsertKind::Neuron,
            connect_from: None,
            box_sel: None,
            move_drag: None,
            preview_delta: None,
            last_pick: None,
            hover: None,
            pick_click: None,
            console_input: String::new(),
            history: Vec::new(),
            hist_idx: None,
            show_console: true,
            show_plots: true,
            show_inspector: true,
            show_script: false,
            plot_tab: PlotTab::Voltage,
            raster_h: 100.0,
            ro_sel: 0,
            script_text: String::new(),
            script_name: "main".into(),
            script_file: "script.rhai".into(),
            show_grid: true,
            show_synapses: true,
            grid_view: crate::plane::GridView::Auto,
            plane_cfgs: HashMap::new(),
            grid_traces: Vec::new(),
            input_values: std::sync::Arc::new(Vec::new()),
            plane_snap: 0,
            plane_geo: 0,
            plane_cfgs_last: Vec::new(),
            speed: 1.0,
            speed_actual: 0.0,
            running: false,
            sim_built: false,
            busy: false,
            dt_ms: 0.1,
            noise_hz: 0.0,
            noise_kick: 15.0,
            net_path: None,
            dlg: None,
            last_net_gen: 0,
            saved_version: 0,
            watch_sent: Vec::new(),
            status: String::new(),
            clipboard: None,
            icons: None,
        }
    }

    fn send(&self, c: Cmd) {
        self.ctl.send(c);
    }

    fn edit_net(&self, f: impl FnOnce(&mut net_format::Network) + Send + 'static) {
        self.send(Cmd::EditNet(Box::new(f)));
    }

    /// Context-menu helper: watch a neuron's v(t) by selecting it (the
    /// plots derive their watch list from the selection).
    pub fn watch(&mut self, ids: Vec<u32>) {
        self.selection = ids.into_iter().collect();
    }

    /// Context-menu helper: delete one neuron.
    pub fn delete_entity(&mut self, id: u32) {
        self.edit_net(move |n| n.delete_neurons(vec![id]));
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
        self.ro_sel = k as usize;
        self.plot_tab = PlotTab::Readouts;
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

    /// Load a file through the worker and remember it in recents.
    pub fn load_net(&mut self, path: String) {
        self.send(Cmd::LoadNet(path.clone()));
        self.net_path = Some(path.clone());
        self.push_recent(path);
        self.selection.clear();
    }

    pub fn push_recent(&mut self, path: String) {
        self.recent.retain(|p| *p != path);
        self.recent.push_front(path);
        self.recent.truncate(8);
    }

    /// Selection → watched neurons for v plots (first 8 selected neurons).
    fn update_watch(&mut self) {
        let ids: Vec<u32> = self
            .selection
            .iter()
            .filter_map(|&i| match self.entity(i) {
                Some(Entity::Neuron(n)) => Some(n),
                _ => None,
            })
            .take(8)
            .collect();
        if ids != self.watch_sent {
            self.watch_sent = ids.clone();
            self.send(Cmd::Watch(ids));
            self.v_hist.retain(|k, _| self.watch_sent.contains(k));
        }
    }

    pub fn entity(&self, inst_id: u32) -> Option<Entity> {
        self.geo
            .as_ref()
            .and_then(|g| crate::view::entity_of(g, inst_id))
    }

    /// Place a neuron on the work plane (insert mode).
    pub fn insert_neuron(&mut self, p: glam::Vec3) {
        self.edit_net(move |n| {
            n.add_neuron([p.x, p.y, p.z], net_format::NeuronKind::Excitatory);
        });
        self.status = format!("placed neuron at ({:.0},{:.0},{:.0})", p.x, p.y, p.z);
    }

    /// Finish a connect drag: synapse a→b with default weight by kind.
    pub fn connect_neurons(&mut self, a: u32, b: u32) {
        if a == b {
            self.status = format!("can't connect neuron {a} to itself");
            return;
        }
        self.edit_net(move |n| {
            n.connect(a, b, 6.0);
        });
        self.status = format!("connected {a} → {b}");
    }

    /// Apply a finished move drag (delta on the work plane).
    pub fn apply_move(&mut self, md: &MoveDrag, delta: glam::Vec3) {
        let moves: Vec<(u32, [f32; 3])> = md
            .ids
            .iter()
            .zip(md.start_pos.iter())
            .filter_map(|(&inst, &p)| match self.entity(inst) {
                Some(Entity::Neuron(n)) => Some((n, (p + delta).into())),
                _ => None,
            })
            .collect();
        self.edit_net(move |net| {
            for (id, pos) in &moves {
                net.neuron_mut(*id).pos = *pos;
            }
            for (id, _) in &moves {
                net.refresh_delays_of(*id);
            }
            net.invalidate_grids();
        });
    }

    /// True when the published net has edits since the last save.
    fn dirty(&self) -> bool {
        let sh = self.ctl.shared.read().unwrap();
        sh.geo.as_ref().map(|g| g.version).unwrap_or(0) != self.saved_version
    }

    /// Run `act` immediately, or behind the unsaved-changes dialog.
    fn guarded(&mut self, ctx: &egui::Context, act: PendingAction) {
        if self.dirty() {
            self.dlg = Some(Dialog::Discard { then: act });
        } else {
            self.do_pending(ctx, act);
        }
    }

    fn do_pending(&mut self, ctx: &egui::Context, act: PendingAction) {
        match act {
            PendingAction::New => {
                self.send(Cmd::NewNet);
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
    fn reset_net_state(&mut self) {
        self.selection.clear();
        self.connect_from = None;
        self.move_drag = None;
        self.preview_delta = None;
        self.box_sel = None;
        self.hover = None;
        self.pick_click = None;
        if !self.watch_sent.is_empty() {
            self.watch_sent.clear();
            self.send(Cmd::Watch(Vec::new()));
        }
        self.v_hist.clear();
        self.raster.clear();
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

    /// Pull fresh shared state (geometry + snapshot) into the UI.
    fn sync_shared(&mut self, gpu: &mut Gpu) {
        let (geo_new, snap, running, sim_built, busy, net_gen) = {
            let sh = self.ctl.shared.read().unwrap();
            (
                sh.geo.clone(),
                sh.snap.clone(),
                sh.running,
                sh.sim_built,
                sh.busy,
                sh.net_gen,
            )
        };
        if net_gen != self.last_net_gen {
            self.last_net_gen = net_gen;
            self.reset_net_state();
        }
        if let Some(g) = geo_new {
            if g.version != self.geo_version {
                gpu.set_scene(&g);
                self.geo_version = g.version;
                // Auto-frame on empty → non-empty (first net, script-built
                // grid on a blank canvas, rebuild after New) and whenever a
                // dialog asked for it (frame_pending).
                let was_empty = self.geo.as_ref().map_or(true, |old| old.pos.is_empty());
                if !g.pos.is_empty() && (was_empty || self.frame_pending) {
                    let (mn, mx) = gpu.scene.as_ref().unwrap().bounds;
                    self.camera.frame(mn, mx);
                }
                self.frame_pending = false;
                self.geo = Some(g);
            }
        }
        if snap.seq != self.last_snap_seq {
            self.last_snap_seq = snap.seq;
            self.speed_actual = snap.steps_per_sec;
            self.activity = snap.activity.to_vec();
            self.grid_traces = snap.grid_traces.clone();
            self.input_values = snap.input_values.clone();
            if let Some(g) = &self.geo {
                self.particles.feed(g, &snap.counts, snap.time_ms);
            }
            // raster bin: ids with count>0
            let fired: Vec<u32> = snap
                .counts
                .iter()
                .enumerate()
                .filter(|(_, c)| **c > 0)
                .map(|(i, _)| i as u32)
                .collect();
            self.raster.push_back((snap.time_ms, fired));
            if self.raster.len() > 600 {
                self.raster.pop_front();
            }
            for (id, vals) in &snap.watched_v {
                let h = self.v_hist.entry(*id).or_default();
                let n = vals.len();
                for (i, &v) in vals.iter().enumerate() {
                    // per-step samples end at snap.time_ms — walk back by dt
                    let t = snap.time_ms - (n - 1 - i) as f64 * self.dt_ms as f64;
                    h.push_back((t, v));
                }
                while h.len() > 4096 {
                    h.pop_front();
                }
            }
        }
        self.running = running;
        self.sim_built = sim_built;
        self.busy = busy;
    }

    /// Per-frame grid-plane bookkeeping for the renderer: quad params and
    /// hide bits (cheap, every call), and the reduced activity textures
    /// only when the trace data or display config actually changed (a
    /// 1M-texel plane is a ~4MB padded upload — not per-frame).
    ///
    /// Returns the per-plane draw flags for `DrawParams::planes_on`.
    pub(crate) fn plane_frame(&mut self, gpu: &mut Gpu, vp_h_px: f32) -> Option<Vec<bool>> {
        let g = self.geo.clone()?;
        if g.grids.is_empty() {
            return None;
        }
        let eye = self.camera.eye();
        let mut on = Vec::with_capacity(g.grids.len());
        let mut params = Vec::with_capacity(g.grids.len());
        let mut cfgs = Vec::with_capacity(g.grids.len());
        let mut bits = vec![0u32; g.pos.len() / 32 + 1];
        for gs in &g.grids {
            let cfg = self
                .plane_cfgs
                .get(&(gs.group as u32))
                .copied()
                .unwrap_or_default();
            cfgs.push(cfg);
            let dist = (glam::Vec3::from(crate::plane::center(gs)) - eye).length();
            let show = match self.grid_view {
                crate::plane::GridView::Planes => true,
                crate::plane::GridView::Cubes => false,
                crate::plane::GridView::Auto => {
                    // A small selection touching this grid flips it back
                    // to cubes so its neurons stay editable; big selects
                    // keep the plane. Members are neuron ids == instance
                    // ids for neurons.
                    let sel_touches = self.selection.len() <= crate::plane::SEL_CUBE_MAX
                        && !self.selection.is_empty()
                        && gs.members.iter().any(|m| self.selection.contains(m));
                    !sel_touches && crate::plane::texel_px(gs, dist, vp_h_px) < crate::plane::LOD_PX
                }
            };
            on.push(show);
            let (ua, va) = crate::plane::plane_dims(gs, &cfg);
            let (origin, du, dv) = crate::plane::quad(gs, &cfg);
            params.push(crate::gpu::PlaneParamsGpu {
                origin,
                du,
                dv,
                wh: [gs.dims[ua], gs.dims[va], 0, 0],
            });
            if show {
                for &m in &gs.members {
                    bits[m as usize >> 5] |= 1 << (m & 31);
                }
            }
        }
        gpu.write_plane_params(&params);
        gpu.write_hide_bits(&bits);

        let need_upload = self.plane_snap != self.last_snap_seq
            || self.plane_geo != self.geo_version
            || self.plane_cfgs_last != cfgs;
        if need_upload {
            for (i, gs) in g.grids.iter().enumerate() {
                let tr = self.grid_traces.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
                let (w, h, pix, pick) = crate::plane::reduce(gs, tr, &cfgs[i]);
                gpu.ensure_plane_tex(i, w, h);
                gpu.write_plane(i, &pix, &pick);
            }
            self.plane_snap = self.last_snap_seq;
            self.plane_geo = self.geo_version;
            self.plane_cfgs_last = cfgs;
        }
        Some(on)
    }

    fn console_lines(&self) -> Vec<String> {
        self.ctl
            .shared
            .read()
            .unwrap()
            .console
            .iter()
            .cloned()
            .collect()
    }

    // ---------------- panels ----------------

    fn toolbar(&mut self, ctx: &egui::Context) {
        let (ic_play, ic_pause, ic_add, ic_conn, ic_load, ic_save) = match &self.icons {
            Some(i) => (
                i.play.clone(),
                i.pause.clone(),
                i.add_neuron.clone(),
                i.connect.clone(),
                i.load.clone(),
                i.save.clone(),
            ),
            None => return,
        };
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        self.guarded(ctx, PendingAction::New);
                        ui.close();
                    }
                    if ui
                        .add(egui::Button::image_and_text(&ic_load, "Load…"))
                        .clicked()
                    {
                        self.dlg = Some(Dialog::Load {
                            path: self.net_path.clone().unwrap_or_default(),
                        });
                        ui.close();
                    }
                    ui.menu_button("Recent files", |ui| {
                        if self.recent.is_empty() {
                            ui.label("(empty)");
                        }
                        let rec: Vec<String> = self.recent.iter().cloned().collect();
                        for p in rec {
                            if ui.button(&p).clicked() {
                                self.guarded(ctx, PendingAction::Load(p));
                                ui.close();
                            }
                        }
                    });
                    if ui
                        .add(egui::Button::image_and_text(&ic_save, "Save"))
                        .clicked()
                    {
                        match &self.net_path {
                            Some(p) => {
                                self.send(Cmd::Eval(format!("net.save({p:?})")));
                                self.saved_version = self
                                    .ctl
                                    .shared
                                    .read()
                                    .unwrap()
                                    .geo
                                    .as_ref()
                                    .map(|g| g.version)
                                    .unwrap_or(0);
                                let p = p.clone();
                                self.push_recent(p);
                            }
                            None => {
                                self.dlg = Some(Dialog::SaveAs {
                                    path: "net.ne2".into(),
                                });
                            }
                        }
                        ui.close();
                    }
                    if ui
                        .add(egui::Button::image_and_text(&ic_save, "Save as…"))
                        .clicked()
                    {
                        self.dlg = Some(Dialog::SaveAs {
                            path: self.net_path.clone().unwrap_or("net.ne2".into()),
                        });
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        self.guarded(ctx, PendingAction::Quit);
                    }
                });
                ui.menu_button("Edit", |ui| {
                    let sel = !self.selection.is_empty();
                    if ui
                        .add_enabled(sel, egui::Button::new("Copy (Ctrl+C)"))
                        .clicked()
                    {
                        self.copy_selection(false);
                        ui.close();
                    }
                    if ui
                        .add_enabled(sel, egui::Button::new("Cut (Ctrl+X)"))
                        .clicked()
                    {
                        self.copy_selection(true);
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            self.clipboard.is_some(),
                            egui::Button::new("Paste (Ctrl+V)"),
                        )
                        .clicked()
                    {
                        self.paste(None);
                        ui.close();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(sel, egui::Button::new("Delete selected (Del)"))
                        .clicked()
                    {
                        self.delete_selection();
                        ui.close();
                    }
                    if ui
                        .add_enabled(sel, egui::Button::new("Add selection to group…"))
                        .clicked()
                    {
                        self.dlg = Some(Dialog::AddToGroup {
                            name: "group".into(),
                        });
                        ui.close();
                    }
                });
                ui.menu_button("Insert", |ui| {
                    if ui.button("Neuron (I)").clicked() {
                        self.mode = Mode::Insert;
                        ui.close();
                    }
                    if ui.button("2D grid…").clicked() {
                        self.dlg = Some(Dialog::Grid2d {
                            group: "grid".into(),
                            w: 20,
                            h: 20,
                            d: 30.0,
                        });
                        ui.close();
                    }
                    if ui.button("3D grid…").clicked() {
                        self.dlg = Some(Dialog::Grid3d {
                            group: "grid".into(),
                            w: 10,
                            h: 10,
                            d2: 10,
                            spacing: 30.0,
                        });
                        ui.close();
                    }
                    if ui.button("Input…").clicked() {
                        self.dlg = Some(Dialog::NewInput {
                            name: "input".into(),
                            weight: 20.0,
                            value: 1.0,
                        });
                        ui.close();
                    }
                    if ui.button("Samples…").clicked() {
                        self.dlg = Some(Dialog::NewSamples {
                            name: "samples".into(),
                        });
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            !self.selection.is_empty(),
                            egui::Button::new("Readout from selection…"),
                        )
                        .clicked()
                    {
                        self.dlg = Some(Dialog::NewReadout { name: "ro".into() });
                        ui.close();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("About").clicked() {
                        self.dlg = Some(Dialog::About);
                        ui.close();
                    }
                });
                ui.menu_button("Group", |ui| {
                    if ui.button("Create connections…").clicked() {
                        self.dlg = Some(Dialog::Connections {
                            group: "grid".into(),
                            scaling: 0.35,
                            lambda: 120.0,
                            w: 6.0,
                        });
                        ui.close();
                    }
                    if ui.button("Set synapse weights…").clicked() {
                        self.dlg = Some(Dialog::Weights {
                            group: "grid".into(),
                            mean: 6.0,
                            sd: 1.0,
                        });
                        ui.close();
                    }
                    if ui.button("Set exc/inh…").clicked() {
                        self.dlg = Some(Dialog::ExcInh {
                            group: "grid".into(),
                            frac: 0.2,
                        });
                        ui.close();
                    }
                });
                ui.separator();
                // sim controls — "Stop" (not "Pause"): abort_now flips the
                // flag on_progress/run chunks/builders watch. A busy script
                // can't resume anyway, so Pause was a lie.
                let live = self.running || self.busy;
                let (run_icon, run_label) = if live {
                    (ic_pause, " Stop")
                } else {
                    (ic_play, " Run")
                };
                if ui
                    .add(egui::Button::image_and_text(&run_icon, run_label))
                    .clicked()
                {
                    if live {
                        self.ctl.abort_now();
                        self.send(Cmd::AutoRun(false));
                    } else {
                        self.send(Cmd::AutoRun(true));
                    }
                }
                if ui.button("Step 10ms").clicked() {
                    self.send(Cmd::RunMs(10.0));
                }
                if ui.button("Step 100ms").clicked() {
                    self.send(Cmd::RunMs(100.0));
                }
                if ui.button("Reset").clicked() {
                    self.send(Cmd::Reset);
                    self.raster.clear();
                    self.v_hist.clear();
                }
                ui.label("speed");
                if ui
                    .add(egui::Slider::new(&mut self.speed, 0.05..=20.0).logarithmic(true))
                    .changed()
                {
                    self.send(Cmd::SetSpeed(self.speed));
                }
                ui.separator();
                ui.label("backend");
                egui::ComboBox::from_id_salt("backend")
                    .selected_text("cuda/cpu")
                    .show_ui(ui, |ui| {
                        if ui.button("cuda").clicked() {
                            self.send(Cmd::SetBackend(sim::BackendKind::Cuda));
                        }
                        if ui.button("cpu").clicked() {
                            self.send(Cmd::SetBackend(sim::BackendKind::Cpu));
                        }
                    });
                ui.label("dt");
                if ui
                    .add(
                        egui::DragValue::new(&mut self.dt_ms)
                            .range(0.01..=1.0)
                            .speed(0.01),
                    )
                    .changed()
                {
                    self.send(Cmd::SetDtMs(self.dt_ms));
                }
                ui.label("noise Hz");
                if ui
                    .add(
                        egui::DragValue::new(&mut self.noise_hz)
                            .range(0.0..=5000.0)
                            .speed(1.0),
                    )
                    .changed()
                {
                    self.send(Cmd::SetNoise(self.noise_hz, self.noise_kick));
                }
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.mode, Mode::Navigate, "Navigate");
                if ui
                    .add(
                        egui::Button::image_and_text(&ic_add, "Insert")
                            .selected(self.mode == Mode::Insert),
                    )
                    .clicked()
                {
                    self.mode = Mode::Insert;
                }
                if ui
                    .add(
                        egui::Button::image_and_text(&ic_conn, "Connect")
                            .selected(self.mode == Mode::Connect),
                    )
                    .clicked()
                {
                    self.mode = Mode::Connect;
                }
                ui.selectable_value(&mut self.mode, Mode::Move, "Move");
                ui.separator();
                ui.label("plane");
                ui.selectable_value(&mut self.plane, Plane::XZ, "x-z");
                ui.selectable_value(&mut self.plane, Plane::XY, "x-y");
                ui.selectable_value(&mut self.plane, Plane::YZ, "y-z");
                ui.separator();
                ui.checkbox(&mut self.show_synapses, "synapses");
                ui.checkbox(&mut self.show_grid, "grid");
                // Grid-group render mode: the plane LOD keeps big grids
                // interactive; cubes stay available for editing up close.
                egui::ComboBox::from_id_salt("grid_view")
                    .selected_text(match self.grid_view {
                        crate::plane::GridView::Auto => "auto",
                        crate::plane::GridView::Cubes => "cubes",
                        crate::plane::GridView::Planes => "planes",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.grid_view,
                            crate::plane::GridView::Auto,
                            "auto",
                        );
                        ui.selectable_value(
                            &mut self.grid_view,
                            crate::plane::GridView::Cubes,
                            "cubes",
                        );
                        ui.selectable_value(
                            &mut self.grid_view,
                            crate::plane::GridView::Planes,
                            "planes",
                        );
                    });
                if ui
                    .button("Frame")
                    .on_hover_text("Frame network (F)")
                    .clicked()
                {
                    self.frame_all();
                }
                ui.separator();
                ui.checkbox(&mut self.show_console, "console");
                ui.checkbox(&mut self.show_plots, "plots");
                ui.checkbox(&mut self.show_inspector, "inspector");
                ui.checkbox(&mut self.show_script, "script");
                ui.separator();
                ui.label(&self.status);
            });
        });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        let (t, rate, spikes, sps) = {
            let sh = self.ctl.shared.read().unwrap();
            (
                sh.snap.time_ms,
                sh.snap.rate_hz,
                sh.snap.spikes_total,
                sh.snap.steps_per_sec,
            )
        };
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("t = {t:.1} ms"));
                ui.separator();
                ui.label(format!("rate = {rate:.1} Hz"));
                ui.separator();
                ui.label(format!("spikes = {spikes}"));
                ui.separator();
                ui.label(format!("sim speed = {sps:.0} steps/s"));
                if let (Some(g), Some(inst)) = (&self.geo, self.hover) {
                    ui.separator();
                    let label = match crate::view::entity_of(g, inst) {
                        Some(Entity::Neuron(n)) => g
                            .names
                            .get(n as usize)
                            .cloned()
                            .unwrap_or_else(|| format!("neuron {n}")),
                        Some(Entity::Input(k)) => format!(
                            "input {}",
                            g.inputs
                                .get(k as usize)
                                .map(|i| i.name.as_str())
                                .unwrap_or("?")
                        ),
                        Some(Entity::Readout(k)) => format!(
                            "readout {}",
                            g.readouts
                                .get(k as usize)
                                .map(|r| r.0.as_str())
                                .unwrap_or("?")
                        ),
                        None => String::new(),
                    };
                    if !label.is_empty() {
                        ui.label(format!("⦿ {label}"));
                    }
                }
                if self.busy {
                    ui.separator();
                    ui.spinner();
                    ui.label("script running");
                }
            });
        });
    }

    fn inspector(&mut self, ctx: &egui::Context) {
        if !self.show_inspector {
            return;
        }
        egui::SidePanel::right("inspector")
            .default_width(240.0)
            .show(ctx, |ui| {
                ui.heading("Inspector");
                ui.separator();
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
                                        .selectable_label(cur == crate::plane::Proj::Mean, "mean")
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
                                    let max = gs.dims[cfg.axis.min(2) as usize] - 1;
                                    *k = (*k).min(max);
                                    ui.horizontal(|ui| {
                                        ui.label("  slice");
                                        ui.add(egui::DragValue::new(k).range(0..=max).speed(0.1));
                                        ui.label(format!("/ {max}"));
                                    });
                                }
                            }
                        }
                    }
                    ui.separator();
                    ui.heading("Inputs");
                    for (k, inp) in g.inputs.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(&inp.name);
                            // seeded from the live sim value, not 0 — the
                            // field must show what the sim is doing
                            let mut v = self.input_values.get(k).copied().unwrap_or(0.0);
                            if ui.add(egui::DragValue::new(&mut v).speed(0.5)).changed() {
                                self.send(Cmd::SetInputConst {
                                    name: inp.name.clone(),
                                    value: v,
                                });
                            }
                        });
                    }
                    if g.inputs.is_empty() {
                        ui.label("(none)");
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
    }

    fn inspect_one(&mut self, ui: &mut egui::Ui, inst_id: u32) {
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
                    self.edit_net(move |n| n.neuron_mut(id).name = name.clone());
                }
                let mut p = g.pos[i];
                ui.horizontal(|ui| {
                    ui.label("pos");
                    let mut changed = false;
                    for pv in &mut p {
                        changed |= ui.add(egui::DragValue::new(pv).speed(1.0)).changed();
                    }
                    if changed {
                        self.edit_net(move |n| {
                            n.neuron_mut(id).pos = p;
                            n.refresh_delays_of(id);
                            n.invalidate_grids();
                        });
                    }
                });
                ui.label(format!(
                    "kind: {}",
                    if g.kind[i] == 0 { "exc" } else { "inh" }
                ));
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
                    self.send(Cmd::SetInputConst {
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
        self.edit_net(move |n| n.delete_neurons(ids.clone()));
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
        let sh = self.ctl.shared.clone();
        let ids2 = ids.clone();
        self.edit_net(move |n| {
            let sub = n.subnet(&ids2);
            if let Ok(txt) = net_format::io::to_json(&sub) {
                if let Ok(mut w) = sh.write() {
                    w.clipboard = Some(txt);
                }
            }
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
        match net_format::io::from_json(&txt) {
            Ok(sub) => {
                self.edit_net(move |n| {
                    n.insert_subnet(&sub, [30.0, 0.0, 30.0]);
                });
            }
            Err(e) => self.status = format!("paste: {e}"),
        }
    }

    fn console(&mut self, ctx: &egui::Context) {
        if !self.show_console {
            return;
        }
        egui::TopBottomPanel::bottom("console")
            .resizable(true)
            .default_height(160.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Script console (Rhai)");
                    if ui.button("Clear").clicked() {
                        if let Ok(mut w) = self.ctl.shared.write() {
                            w.console.clear();
                        }
                    }
                });
                let lines = self.console_lines();
                egui::ScrollArea::vertical()
                    .id_salt("conout")
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .max_height(90.0)
                    .show(ui, |ui| {
                        for l in &lines {
                            ui.label(egui::RichText::new(l).monospace().size(11.0));
                        }
                    });
                ui.horizontal(|ui| {
                    let mut lay = |ui: &egui::Ui, s: &dyn egui::TextBuffer, wrap: f32| {
                        crate::syntax::highlight(ui, s.as_str(), wrap)
                    };
                    let te = egui::TextEdit::multiline(&mut self.console_input)
                        .desired_rows(2)
                        .desired_width(ui.available_width() - 60.0)
                        .font(egui::TextStyle::Monospace)
                        .layouter(&mut lay);
                    let r = ui.add(te);
                    // history
                    if r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                        let idx = self
                            .hist_idx
                            .map(|i| i.saturating_sub(1))
                            .unwrap_or(self.history.len().saturating_sub(1));
                        if let Some(h) = self.history.get(idx) {
                            self.console_input = h.clone();
                            self.hist_idx = Some(idx);
                        }
                    }
                    if r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                        if let Some(i) = self.hist_idx {
                            let idx = i + 1;
                            if idx < self.history.len() {
                                self.console_input = self.history[idx].clone();
                                self.hist_idx = Some(idx);
                            } else {
                                self.console_input.clear();
                                self.hist_idx = None;
                            }
                        }
                    }
                    let run = ui.button("Run");
                    let enter = r.has_focus()
                        && ui.input(|i| {
                            i.key_pressed(egui::Key::Enter)
                                && (i.modifiers.ctrl || i.modifiers.command)
                        });
                    if (run.clicked() || enter) && !self.console_input.trim().is_empty() {
                        let code = self.console_input.clone();
                        self.history.push(code.clone());
                        self.hist_idx = None;
                        self.console_input.clear();
                        self.send(Cmd::Eval(code));
                    }
                });
                ui.small("Ctrl+Enter runs · ↑/↓ history · `net`, `sim`, `rng` in scope");
            });
    }

    /// Multiline script editor (the original's QScintilla script window):
    /// scripts saved in the .ne2, load/save to .rhai files, Run via worker.
    fn script_window(&mut self, ctx: &egui::Context) {
        if !self.show_script {
            return;
        }
        let mut open = self.show_script;
        egui::Window::new("Script editor")
            .open(&mut open)
            .default_size([480.0, 360.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("name");
                    ui.text_edit_singleline(&mut self.script_name);
                    if ui.button("Run").clicked() {
                        let code = self.script_text.clone();
                        self.send(Cmd::Eval(code));
                    }
                    if ui.button("Save to net").clicked() {
                        self.send(Cmd::Eval(format!(
                            "net.script_add({:?}, {:?})",
                            self.script_name, self.script_text
                        )));
                        self.status = format!("script {:?} saved to net", self.script_name);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("file");
                    ui.text_edit_singleline(&mut self.script_file);
                    if ui.button("Open").clicked() {
                        match std::fs::read_to_string(&self.script_file) {
                            Ok(s) => self.script_text = s,
                            Err(e) => self.status = format!("open: {e}"),
                        }
                    }
                    if ui.button("Save").clicked() {
                        if let Err(e) = std::fs::write(&self.script_file, &self.script_text) {
                            self.status = format!("save: {e}");
                        }
                    }
                });
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
                        let code = self.script_text.clone();
                        self.send(Cmd::Eval(code));
                    }
                });
            });
        self.show_script = open;
    }

    fn plots(&mut self, ctx: &egui::Context) {
        if !self.show_plots {
            return;
        }
        egui::TopBottomPanel::bottom("plots")
            .resizable(true)
            .default_height(140.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Plots");
                    ui.separator();
                    ui.selectable_value(&mut self.plot_tab, PlotTab::Voltage, "v(t)");
                    ui.selectable_value(&mut self.plot_tab, PlotTab::Raster, "raster");
                    ui.selectable_value(&mut self.plot_tab, PlotTab::Readouts, "readouts");
                    ui.selectable_value(&mut self.plot_tab, PlotTab::Samples, "samples");
                    ui.separator();
                    if self.plot_tab == PlotTab::Voltage {
                        ui.label(format!(
                            "watching {} neuron(s) — select neurons to plot v(t)",
                            self.watch_sent.len()
                        ));
                    }
                    if self.plot_tab == PlotTab::Raster {
                        ui.label("height");
                        ui.add(
                            egui::DragValue::new(&mut self.raster_h)
                                .range(40.0..=600.0)
                                .suffix("px"),
                        );
                    }
                });
                match self.plot_tab {
                    PlotTab::Voltage => {
                        egui_plot::Plot::new("vplot").height(100.0).show(ui, |ui| {
                            for (&id, h) in &self.v_hist {
                                let pts: PlotPoints =
                                    h.iter().map(|(t, v)| [*t, *v as f64]).collect();
                                ui.line(Line::new(format!("n{id}"), pts));
                            }
                        });
                    }
                    PlotTab::Raster => {
                        // Raster strip: x = sim milliseconds. With watches
                        // set, each row is a watched neuron (labeled);
                        // without, y is raw neuron id (sampled).
                        let watches = self.watch_sent.clone();
                        let n_neurons =
                            self.geo.as_ref().map(|g| g.pos.len() as f64).unwrap_or(1.0);
                        // Always pin y: toggling the flag between watch and
                        // scatter modes stranded stale bounds in PlotMemory
                        // and hid the dots.
                        let y_hi = if watches.is_empty() {
                            n_neurons - 0.5
                        } else {
                            watches.len() as f64 - 0.5
                        };
                        let plot = egui_plot::Plot::new("raster")
                            .height(self.raster_h.max(40.0))
                            .auto_bounds(egui::Vec2b::new(true, false))
                            .include_y(-0.5)
                            .include_y(y_hi);
                        plot.x_axis_formatter(|x, _| format!("{:.0}ms", x.value))
                            .y_axis_formatter(move |y, _| {
                                let k = y.value.round() as usize;
                                if watches.is_empty() {
                                    format!("{}", k)
                                } else {
                                    watches
                                        .get(k)
                                        .map(|id| format!("n{id}"))
                                        .unwrap_or_default()
                                }
                            })
                            .show(ui, |ui| {
                                let mut pts = Vec::new();
                                let watches = &self.watch_sent;
                                for &(t, ref fired) in &self.raster {
                                    if watches.is_empty() {
                                        let stride = (fired.len() / 200).max(1);
                                        for (k, &id) in fired.iter().enumerate() {
                                            if k % stride == 0 {
                                                pts.push([t, id as f64]);
                                            }
                                        }
                                    } else {
                                        for (k, &id) in watches.iter().enumerate() {
                                            if fired.contains(&id) {
                                                pts.push([t, k as f64]);
                                            }
                                        }
                                    }
                                }
                                ui.points(
                                    egui_plot::Points::new("spikes", PlotPoints::from(pts))
                                        .radius(1.0_f32)
                                        .color(egui::Color32::LIGHT_BLUE),
                                );
                            });
                    }
                    PlotTab::Readouts => self.plot_readouts(ui),
                    PlotTab::Samples => self.plot_samples(ui),
                }
            });
    }

    /// Bar chart of the selected readout's trained weights (original's
    /// ReadOut weights visualization).
    fn plot_readouts(&mut self, ui: &mut egui::Ui) {
        let Some(g) = self.geo.clone() else { return };
        if g.readouts.is_empty() {
            ui.label("no readouts — net.add_readout(\"ro\", units) in a script");
            return;
        }
        self.ro_sel = self.ro_sel.min(g.readouts.len() - 1);
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("ro_sel")
                .selected_text(g.readouts[self.ro_sel].0.clone())
                .show_ui(ui, |ui| {
                    for (i, (name, _, _)) in g.readouts.iter().enumerate() {
                        ui.selectable_value(&mut self.ro_sel, i, name);
                    }
                });
            let (_, units, w) = &g.readouts[self.ro_sel];
            ui.label(format!("{} units, {} weights", units.len(), w.len()));
        });
        let (_, _, w) = &g.readouts[self.ro_sel];
        if w.is_empty() {
            ui.label("no weights — push them from a script: net.set_readout_weights(\"name\", ro.weights())");
            return;
        }
        egui_plot::Plot::new("row").height(80.0).show(ui, |ui| {
            let bars: Vec<egui_plot::Bar> = w
                .iter()
                .enumerate()
                .map(|(i, &v)| egui_plot::Bar::new(i as f64, v as f64).width(0.8))
                .collect();
            ui.bar_chart(egui_plot::BarChart::new("w", bars));
        });
    }

    /// Time-series plot of `net.samples` (the original's samples plot).
    fn plot_samples(&mut self, ui: &mut egui::Ui) {
        let Some(g) = self.geo.clone() else { return };
        if g.samples.is_empty() {
            ui.label("no samples — net.samples_write(\"s\", times, values) in a script");
            return;
        }
        egui_plot::Plot::new("smp").height(100.0).show(ui, |ui| {
            for (name, t, v) in g.samples.iter().take(8) {
                let pts: PlotPoints = t
                    .iter()
                    .zip(v.iter())
                    .map(|(&a, &b)| [a as f64, b as f64])
                    .collect();
                ui.line(Line::new(name.clone(), pts));
            }
        });
    }

    /// Group-name picker for dialogs: combo of the live groups instead of
    /// a free-text field that happily accepted typos.
    fn group_combo(&self, ui: &mut egui::Ui, id: &str, group: &mut String) {
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

    fn dialog(&mut self, ctx: &egui::Context) {
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
                egui::Window::new("Create 2D grid")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            ui.text_edit_singleline(group);
                        });
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(w).prefix("w "));
                            ui.add(egui::DragValue::new(h).prefix("h "));
                            ui.add(egui::DragValue::new(dist).prefix("dist "));
                        });
                        if ui.button("Create").clicked() {
                            let (g, w2, h2, d2) = (group.clone(), *w, *h, *dist);
                            self.edit_net(move |n| {
                                net_format::builders::create_2d_grid(
                                    n,
                                    &g,
                                    w2.max(1) as usize,
                                    h2.max(1) as usize,
                                    d2,
                                    [0.0; 3],
                                );
                            });
                            self.frame_pending = true;
                            keep = false;
                        }
                    });
            }
            Dialog::Grid3d {
                group,
                w,
                h,
                d2,
                spacing,
            } => {
                egui::Window::new("Create 3D grid")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            ui.text_edit_singleline(group);
                        });
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(w).prefix("w "));
                            ui.add(egui::DragValue::new(h).prefix("h "));
                            ui.add(egui::DragValue::new(d2).prefix("d "));
                            ui.add(egui::DragValue::new(spacing).prefix("spacing "));
                        });
                        if ui.button("Create").clicked() {
                            let (g, w2, h2, dd, sp) = (group.clone(), *w, *h, *d2, *spacing);
                            self.edit_net(move |n| {
                                net_format::builders::create_3d_grid(
                                    n,
                                    &g,
                                    w2.max(1) as usize,
                                    h2.max(1) as usize,
                                    dd.max(1) as usize,
                                    sp,
                                    [0.0; 3],
                                );
                            });
                            self.frame_pending = true;
                            keep = false;
                        }
                    });
            }
            Dialog::Connections {
                group,
                scaling,
                lambda,
                w,
            } => {
                egui::Window::new("Create connections")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            self.group_combo(ui, "conn_group", group);
                        });
                        ui.add(egui::DragValue::new(scaling).prefix("scaling ").speed(0.01));
                        ui.add(egui::DragValue::new(lambda).prefix("lambda "));
                        ui.add(egui::DragValue::new(w).prefix("w "));
                        if ui.button("Create").clicked() {
                            let (g, sc, la, ww) = (group.clone(), *scaling, *lambda, *w);
                            self.send(Cmd::Eval(format!(
                                "net.create_connections({g:?}, {sc:?}, {la:?}, {ww:?})"
                            )));
                            keep = false;
                        }
                    });
            }
            Dialog::Weights { group, mean, sd } => {
                egui::Window::new("Set synapse weights")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            self.group_combo(ui, "w_group", group);
                        });
                        ui.add(egui::DragValue::new(mean).prefix("mean "));
                        ui.add(egui::DragValue::new(sd).prefix("sd "));
                        if ui.button("Apply").clicked() {
                            let (g, m, s) = (group.clone(), *mean, *sd);
                            self.send(Cmd::Eval(format!(
                                "net.set_synapse_weights({g:?}, {m:?}, {s:?})"
                            )));
                            keep = false;
                        }
                    });
            }
            Dialog::ExcInh { group, frac } => {
                egui::Window::new("Set excitatory/inhibitory")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            self.group_combo(ui, "ei_group", group);
                        });
                        ui.add(egui::Slider::new(frac, 0.0..=1.0).text("inh frac"));
                        if ui.button("Apply").clicked() {
                            let (g, f) = (group.clone(), *frac);
                            self.send(Cmd::Eval(format!("net.set_exc_inh({g:?}, {f:?})")));
                            keep = false;
                        }
                    });
            }
            Dialog::NewInput {
                name,
                weight,
                value,
            } => {
                egui::Window::new("New input on selection")
                    .open(&mut open)
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
                            self.edit_net(move |net| {
                                net.add_input(&n2, tw, [0.0; 3]);
                            });
                            // a new input at value 0 drives nothing —
                            // apply the dialog's value right after create
                            self.send(Cmd::SetInputConst {
                                name: name.clone(),
                                value: *value,
                            });
                            keep = false;
                        }
                    });
            }
            Dialog::NewSamples { name } => {
                egui::Window::new("New samples node")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("name");
                            ui.text_edit_singleline(name);
                        });
                        ui.label("empty (t,value) series — fill via net.samples_write");
                        if ui.button("Create").clicked() {
                            let n2 = name.clone();
                            self.edit_net(move |net| {
                                net.samples_mut(&n2);
                            });
                            keep = false;
                        }
                    });
            }
            Dialog::NewReadout { name } => {
                egui::Window::new("New readout on selection")
                    .open(&mut open)
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
                            self.edit_net(move |net| {
                                net.add_readout(&n2, units.clone());
                            });
                            keep = false;
                        }
                    });
            }
            Dialog::AddToGroup { name } => {
                egui::Window::new("Add selection to group")
                    .open(&mut open)
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
                            self.edit_net(move |net| {
                                let gi = net.add_group(&nm);
                                for id in &ids {
                                    if !net.groups[gi].members.contains(id) {
                                        net.groups[gi].members.push(*id);
                                    }
                                }
                                net.invalidate_grids();
                            });
                            keep = false;
                        }
                    });
            }
            Dialog::Discard { then } => {
                egui::Window::new("Unsaved changes")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.label("The network has unsaved changes — discard them?");
                        ui.horizontal(|ui| {
                            if ui.button("Discard").clicked() {
                                let act = then.clone();
                                self.do_pending(ctx, act);
                                keep = false;
                            }
                            if ui.button("Cancel").clicked() {
                                keep = false;
                            }
                        });
                    });
            }
            Dialog::SaveAs { path } => {
                egui::Window::new("Save as")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.text_edit_singleline(path);
                        if ui.button("Save").clicked() {
                            let p = path.clone();
                            self.send(Cmd::Eval(format!("net.save({p:?})")));
                            self.saved_version = self
                                .ctl
                                .shared
                                .read()
                                .unwrap()
                                .geo
                                .as_ref()
                                .map(|g| g.version)
                                .unwrap_or(0);
                            self.net_path = Some(p.clone());
                            self.push_recent(p);
                            keep = false;
                        }
                    });
            }
            Dialog::Load { path } => {
                egui::Window::new("Load network")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.text_edit_singleline(path);
                        if ui.button("Load").clicked() {
                            let p = path.clone();
                            self.guarded(ctx, PendingAction::Load(p));
                            keep = false;
                        }
                    });
            }
            Dialog::About => {
                egui::Window::new("About neuro_edit v2")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        if let Some(i) = &self.icons {
                            ui.image(&i.about);
                        }
                        ui.heading("neuro_edit v2");
                        ui.label("Izhikevich spiking-network simulator and 3D editor.");
                        ui.label("Rust + CUDA (cudarc/NVRTC) + egui/wgpu + Rhai.");
                        ui.label("Rewrite of the C++/Qt4 neuro_edit (2011–2012).");
                        ui.label("License: GPL-3.0");
                    });
            }
        }
        if open && keep {
            self.dlg = Some(d);
        }
    }
}

impl eframe::App for EditorApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let p = Persisted {
            recent: self.recent.iter().cloned().collect(),
        };
        eframe::set_value(storage, "state", &p);
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.icons
            .get_or_insert_with(|| crate::icons::Icons::load(ctx));
        // GPU lives in egui-wgpu callback resources; we need it here for
        // scene updates + picking. Get it via render_state.
        let rs = frame.wgpu_render_state().expect("wgpu render state");
        {
            let mut res = rs.renderer.write();
            if res.callback_resources.get::<Gpu>().is_none() {
                res.callback_resources.insert(Gpu::new(
                    rs.device.clone(),
                    rs.queue.clone(),
                    rs.target_format,
                ));
            }
        }
        crate::view::with_gpu(rs, |gpu| self.sync_shared(gpu));
        self.update_watch();

        // clipboard staged by the worker → OS clipboard + internal copy
        let staged = self
            .ctl
            .shared
            .write()
            .ok()
            .and_then(|mut w| w.clipboard.take());
        if let Some(txt) = staged {
            ctx.copy_text(txt.clone());
            self.clipboard = Some(txt);
        }
        // system paste events (Ctrl+V when no text field owns it)
        for ev in ctx.input(|i| i.events.clone()) {
            if let egui::Event::Paste(txt) = ev {
                self.paste(Some(txt.clone()));
            }
        }

        // OS close button: with unsaved edits, hold the window open behind
        // the discard dialog instead of exiting silently.
        if ctx.input(|i| i.viewport().close_requested()) && self.dirty() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            if !matches!(self.dlg, Some(Dialog::Discard { .. })) {
                self.dlg = Some(Dialog::Discard {
                    then: PendingAction::Quit,
                });
            }
        }

        self.toolbar(ctx);
        self.status_bar(ctx);
        self.inspector(ctx);
        self.console(ctx);
        self.script_window(ctx);
        self.plots(ctx);
        self.dialog(ctx);

        crate::view::viewport(self, ctx, rs);
        ctx.request_repaint();
    }
}
