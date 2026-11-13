//! eframe App: toolbar, 3D viewport (egui-wgpu callback), inspector,
//! console, plots. Communicates with the sim worker via `SimCtl`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use eframe::egui;
use egui_plot::{Line, PlotPoints};

use crate::camera::Camera;
use crate::gpu::{Entity, Gpu};
use crate::icons;
use crate::scene::{Particles, Plane};
use crate::simctl::SimCtl;
use crate::snap::GeoSnap;
use proto::{ClientMsg, NetEdit};

mod actions;
mod files;
mod help;
mod inspector;
mod layout;
#[cfg(not(target_arch = "wasm32"))]
mod login;
mod sync;
mod toolbar;

use files::*;
use layout::*;

#[allow(dead_code)]
pub struct EditorApp {
    pub ctl: SimCtl,
    pub camera: Camera,
    pub geo: Option<Arc<GeoSnap>>,
    /// Bumped on every geo assign — SceneKey input (view.rs reads it).
    pub(crate) geo_version: u64,
    /// Frame the camera on the next geometry publish (grid dialogs, load).
    frame_pending: bool,
    last_snap_seq: u64,
    /// Smoothed per-frame cost (egui+GPU submit), drawn when `show_fps`.
    pub frame_ms: f32,
    last_frame_ms: f64,
    pub show_fps: bool,
    /// Set this frame when any ServerMsg landed (dirty_seq moved) —
    /// drives the conditional repaint at the end of `update`.
    pub net_touched: bool,
    last_dirty_seq: u64,
    /// Fingerprint contribution of `plane_frame` (params+cfgs+hide bits+
    /// on-flags+trace uploads) for the SceneKey.
    pub plane_key: u64,
    /// Set by viewport each frame: spike-dot particles exist → keep
    /// repainting until they finish their wall-clock flight.
    pub parts_active: bool,
    /// A pick readback is queued on the GPU — one more frame must run to
    /// consume it even if nothing else animates.
    pub pick_in_flight: bool,
    /// Last hover-pick target (pixel + camera + geometry): a new pick is
    /// only queued when this changes — otherwise pick_in_flight would
    /// keep the app repainting forever under a still cursor.
    pub last_hover_pick: Option<(u32, u32, u64, u64)>,
    pub activity: Vec<f32>,
    pub particles: Particles,
    /// v(t) histories + raster, cleared when fed sim time goes backwards.
    hist: crate::plot::PlotHist,
    /// Sliding-window/follow state shared by the v(t), raster and
    /// phase tabs.
    plot_win: crate::plot::PlotWindow,
    /// Newest sim time seen in a snapshot — the window's right edge.
    last_sim_t: f64,
    /// `hist.resets` as of the last sync — a bump re-enables follow.
    hist_resets_seen: usize,

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
    /// Click waiting for a free pick slot (uv in the viewport) + queue
    /// time (`wall_ms`). Clicks queue so an in-flight hover pick can't
    /// drop them; a click still queued after ~800ms is stale — the user
    /// has moved on — so view.rs drops it instead of applying a hit/miss
    /// to newer state (e.g. an example auto-watch applied meanwhile).
    pub pick_click: Option<(glam::Vec2, f64)>,
    /// Per-instance cube half sizes (copied from `Scene` on geo change) —
    /// the selection brackets size to the projected object box.
    pub inst_half: Vec<f32>,
    pub show_grid: bool,
    pub show_synapses: bool,
    /// U8: Alt held during insert places an inhibitory neuron —
    /// otherwise new neurons are always excitatory.
    pub insert_inh: bool,
    /// U7: gold input→target links, toggled independently of synapses.
    pub show_input_links: bool,
    /// Spike dots travelling display synapses (View toggle).
    pub show_dots: bool,
    /// Wall-clock ms (egui `i.time`) — particle flights ride this clock
    /// so their duration is frame-cadence/sim-rate independent.
    pub wall_ms: f64,
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
    /// U1: latest server error — the status line shows it in warning
    /// colour while `self.status` still equals it.
    last_error: Option<String>,
    /// U1: `Shared::err_seq` at the last console-open — the Console
    /// button gets an unread dot while unseen errors exist.
    err_seq_seen: u64,
    /// U5: last successful connect (url, session token) — the
    /// disconnect banner's reconnect button.
    connect_url: String,
    connect_tok: String,
    /// U5b: auto-retry schedule — egui time of the next attempt and
    /// the current backoff step (2,4,8…60 s).
    reconnect_at: Option<f64>,
    reconnect_delay: f64,
    pub(crate) show_console: bool,
    pub(crate) show_plots: bool,
    pub(crate) show_inspector: bool,
    pub(crate) show_script: bool,
    pub(crate) show_inputs: bool,
    /// `?` shortcut overlay.
    pub(crate) show_keys: bool,
    /// Manual window (docs/manual pages baked in).
    pub(crate) show_manual: bool,
    manual_page: usize,
    /// Files window (per-user space when remote, app-data dir locally).
    pub(crate) show_files: bool,
    files: FilesUi,
    /// Multi-user login (`--connect http(s)://…`) — `Some` until a
    /// session exists.
    login: Option<LoginDlg>,
    /// Remembered (url, session-token) for `remember me` — persisted.
    pub saved_token: Option<(String, String)>,
    plot_tab: PlotTab,
    /// Raster strip height in px — draggable in the plots header.
    raster_h: f32,
    /// Readout index shown in the weights tab.
    ro_sel: usize,
    /// Script editor buffer + the name it is saved under in the file.
    script_text: String,
    script_name: String,
    script_file: String,
    /// Server-side dir listing for the file pickers (from `ListFiles`).
    file_list: Option<(String, Vec<proto::FileEntry>)>,
    /// Server jobs (list + live progress rows + last opened detail).
    jobs: Vec<proto::JobInfo>,
    job_progress: std::collections::BTreeMap<u64, Vec<Vec<(String, f64)>>>,
    job_detail: Option<crate::snap::JobDetailSnap>,
    /// Job selected in the jobs tab (id; 0 = none yet).
    job_sel: u64,
    /// U8: the Jobs tab fetched the list since it was opened —
    /// pushes update it live, no 2 s poll.
    jobs_fetched: bool,
    /// Recently opened/saved .ne2 paths (persisted across runs).
    pub recent: VecDeque<String>,

    speed: f32,
    /// Best steps/s the machine has reached — shown as `(max xN)` next to
    /// the requested speed so "how fast can this go" is one display.
    speed_peak: f32,
    /// Backend the worker reports (request combo selected text).
    backend: proto::Backend,
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
    /// Neuron ids an example asked to watch on Run (see examples.rs).
    example_watch: Option<&'static [u32]>,
    /// `geo_version` when the example's Run was sent — the watch applies
    /// only to geometry newer than that (the pre-run net may already
    /// contain the ids; watching them early just gets wiped by the
    /// eval's net.clear → reset).
    example_base_ver: Option<u64>,
    /// `wall_ms` deadline for the post-Run Geo — a script that never
    /// builds a covering net must not fire the watch on a later,
    /// unrelated net change.
    example_deadline_ms: f64,
    example_quiet: u8,
    pub status: String,
    /// Mode as of last frame — a change clears `status` (stale
    /// mode-scoped messages like "connect: missed" shouldn't stick).
    last_mode: Mode,
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
    Phase,
    Raster,
    Readouts,
    Samples,
    Jobs,
}

#[derive(Clone)]
pub struct MoveDrag {
    /// Work-plane anchor: the grabbed object's world position. The drag
    /// plane passes through it (not the camera target) so the object
    /// stays under the cursor for the whole drag.
    pub anchor: glam::Vec3,
    /// Constraint-space zero point: `delta = hit - plane_pt`. Re-anchored
    /// on plane/axis switches so the object doesn't jump.
    pub plane_pt: glam::Vec3,
    /// The live constraint (work plane / Shift vertical / axis lock).
    pub plane: crate::view::DragPlane,
    /// X/Y/Z toggle state: `Some(0|1|2)` = locked to that axis.
    pub axis: Option<u8>,
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
    /// Per-synapse current trace tau on src→dst edges (sd=0 → uniform).
    SynTau {
        src: String,
        dst: String,
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

/// Files-window state: a listing of the user's space (remote per-user
/// root over the file protocol, or the local app-data dir when running
/// without --connect). One window, two backends.
#[derive(Default)]
pub struct FilesUi {
    /// Remote: path inside the user's root ("" = root). Local: absolute
    /// path under `local_root`.
    pub cwd: String,
    /// Entries of `cwd` (remote: last `FileList` reply; local: read_dir).
    pub entries: Vec<proto::FileEntry>,
    /// Native, no --connect: files live in `dirs::data_dir()/neuro_edit`
    /// through std::fs instead of the wire protocol.
    pub local_root: Option<std::path::PathBuf>,
    /// (used, quota) from `Usage` / local dir_size.
    pub usage: Option<(u64, u64)>,
    pub sel: Option<String>,
    pub mkdir: String,
    pub rename_to: String,
    /// U8: the "rename…" button opens the rename row (it used to be
    /// a no-op that only relied on rename_to being seeded).
    pub rename_open: bool,
    pub confirm_delete: Option<String>,
    /// Uploads in flight: (rel-path, bytes, offset). One at a time —
    /// `UploadStart` waits on `UploadReady` before chunks go out.
    pub upload: Option<(String, Vec<u8>, usize)>,
    pub upload_q: std::collections::VecDeque<(String, Vec<u8>)>,
    /// First-open listing requested?
    pub opened: bool,
    /// egui drag-drop payloads stashed by `files_window` for `files_pump`.
    pub ctx_dropped: Option<Vec<egui::DroppedFile>>,
    pub status: String,
}


/// Login dialog state for `--connect http(s)://…` (multi-user server):
/// the app boots "offline" and swaps in the real `SimCtl` on success.
pub struct LoginDlg {
    pub url: String,
    pub email: String,
    pub password: String,
    pub remember: bool,
    pub busy: bool,
    pub error: String,
    /// Result of the background login+connect attempt.
    pub rx: Option<std::sync::mpsc::Receiver<Result<(crate::simctl::SimCtl, String), String>>>,
}

/// Persisted UI state (recent files + window visibility) across sessions.
/// Window positions/sizes persist separately through egui's own memory
/// (eframe stores it under the same storage slot).
#[derive(serde::Serialize, serde::Deserialize, Default)]
pub struct Persisted {
    pub recent: Vec<String>,
    #[serde(default)]
    pub windows: WindowVis,
    /// `remember me`: (server base-url, session token).
    #[serde(default)]
    pub token: Option<(String, String)>,
}

/// Which floating windows are open.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct WindowVis {
    pub inspector: bool,
    pub plots: bool,
    pub console: bool,
    pub script: bool,
    pub inputs: bool,
    #[serde(default)]
    pub files: bool,
}

impl Default for WindowVis {
    fn default() -> Self {
        Self {
            inspector: true,
            plots: true,
            console: true,
            script: false,
            inputs: false,
            files: false,
        }
    }
}

impl EditorApp {
    pub fn new(ctl: SimCtl, storage: Option<&dyn eframe::Storage>) -> Self {
        let persisted = storage.and_then(|s| eframe::get_value::<Persisted>(s, "state"));
        let wins = persisted.as_ref().map(|p| p.windows).unwrap_or_default();
        let persisted_token = persisted.as_ref().and_then(|p| p.token.clone());
        let recent: VecDeque<String> = persisted
            .map(|p| p.recent.into_iter().collect())
            .unwrap_or_default();
        #[allow(unused_variables)]
        let is_remote = ctl.remote;
        Self {
            recent,
            ctl,
            camera: Camera::default(),
            geo: None,
            geo_version: 0,
            frame_pending: false,
            last_snap_seq: 0,
            frame_ms: 0.0,
            last_frame_ms: 0.0,
            show_fps: false,
            net_touched: false,
            last_dirty_seq: 0,
            plane_key: 0,
            parts_active: false,
            pick_in_flight: false,
            last_hover_pick: None,
            activity: Vec::new(),
            particles: Particles::new(),
            hist: crate::plot::PlotHist::default(),
            plot_win: crate::plot::PlotWindow::default(),
            last_sim_t: 0.0,
            hist_resets_seen: 0,
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
            inst_half: Vec::new(),
            console_input: String::new(),
            history: Vec::new(),
            hist_idx: None,
            last_error: None,
            err_seq_seen: 0,
            connect_url: String::new(),
            connect_tok: String::new(),
            reconnect_at: None,
            reconnect_delay: 2.0,
            show_console: wins.console,
            show_plots: wins.plots,
            show_inspector: wins.inspector,
            show_script: wins.script,
            show_inputs: wins.inputs,
            show_keys: false,
            show_manual: false,
            manual_page: 0,
            show_files: wins.files,
            files: {
                #[allow(unused_mut)]
                let mut f = FilesUi::default();
                #[cfg(not(target_arch = "wasm32"))]
                if !is_remote {
                    f.local_root = Some(local_files_root());
                    f.cwd = f
                        .local_root
                        .as_ref()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
                }
                f
            },
            login: None,
            saved_token: persisted_token,
            plot_tab: PlotTab::Voltage,
            raster_h: 100.0,
            ro_sel: 0,
            script_text: String::new(),
            script_name: "main".into(),
            script_file: "script.rhai".into(),
            file_list: None,
            jobs: Vec::new(),
            job_progress: std::collections::BTreeMap::new(),
            job_detail: None,
            job_sel: 0,
            jobs_fetched: false,
            show_grid: true,
            show_synapses: true,
            insert_inh: false,
            show_input_links: true,
            show_dots: true,
            wall_ms: 0.0,
            grid_view: crate::plane::GridView::Auto,
            plane_cfgs: HashMap::new(),
            grid_traces: Vec::new(),
            input_values: std::sync::Arc::new(Vec::new()),
            plane_snap: 0,
            plane_geo: 0,
            plane_cfgs_last: Vec::new(),
            speed: 1.0,
            speed_peak: 0.0,
            backend: proto::Backend::Cpu,
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
            example_watch: None,
            example_base_ver: None,
            example_deadline_ms: f64::MAX,
            example_quiet: 0,
            status: String::new(),
            last_mode: Mode::Navigate,
            clipboard: None,
            icons: None,
        }
    }

    fn send(&self, c: ClientMsg) {
        self.ctl.send(c);
    }


    /// Apply a structured edit — versioned, delta-able, and expressible
    /// on the wire (unlike the old closure path).
    fn edit_net(&self, e: NetEdit) {
        self.ctl.send_edit(e);
    }









    /// Load a bundled example into the script editor (Stage 13 H).
    fn load_example(&mut self, ex: &'static crate::examples::Example) {
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
    fn run_script(&mut self) {
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









    // ------------------------------------------------------------------
    // Files window — one UI, two backends: the file protocol (remote,
    // per-user root) or std::fs on the local app-data dir.
    // ------------------------------------------------------------------












    // ------------------------------------------------------------------
    // Login dialog — `--connect http(s)://host` multi-user servers.
    // ------------------------------------------------------------------










    fn console(&mut self, ctx: &egui::Context) {
        if !self.show_console {
            return;
        }
        // U1: an open console has seen every error so far.
        self.err_seq_seen = self.ctl.shared.read().unwrap().err_seq;
        let mut open = self.show_console;
        let u = usable_rect(ctx);
        let dr = console_default_rect(u);
        app_win(ctx, "Console (Rhai)")
            .open(&mut open)
            .default_size(dr.size())
            .default_pos(dr.min)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Clear").clicked() {
                        if let Ok(mut w) = self.ctl.shared.write() {
                            w.console.clear();
                        }
                    }
                });
                let lines = self.console_lines();
                // Fill the panel minus the input row (~70px): the old fixed
                // 90px cap left most of a tall console blank.
                let out_h = (ui.available_height() - 70.0).max(40.0);
                egui::ScrollArea::vertical()
                    .id_salt("conout")
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .max_height(out_h)
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
                        self.send(ClientMsg::Eval { code });
                    }
                });
                ui.small("Ctrl+Enter runs · up/down = history · `net`, `sim`, `rng` in scope");
            });
        self.show_console = open;
    }

    /// Multiline script editor (the original's QScintilla script window):
    /// scripts saved in the .ne2, load/save to .rhai files, Run via worker.
    fn script_window(&mut self, ctx: &egui::Context) {
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

    fn plots(&mut self, ctx: &egui::Context) {
        if !self.show_plots {
            return;
        }
        let mut open = self.show_plots;
        let u = usable_rect(ctx);
        let dr = plots_default_rect(u);
        app_win(ctx, "Plots")
            .open(&mut open)
            .default_size(dr.size())
            .default_pos(dr.min)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(ic) = &self.icons {
                        let t = &mut self.plot_tab;
                        icons::selectable_value(ui, t, PlotTab::Voltage, &ic.voltage, "v(t)");
                        icons::selectable_value(ui, t, PlotTab::Phase, &ic.phase, "v–u");
                        icons::selectable_value(ui, t, PlotTab::Raster, &ic.raster, "raster");
                        icons::selectable_value(ui, t, PlotTab::Readouts, &ic.readout, "readouts");
                        icons::selectable_value(ui, t, PlotTab::Samples, &ic.samples, "samples");
                        icons::selectable_value(ui, t, PlotTab::Jobs, &ic.jobs, "jobs");
                    }
                    if self.plot_tab == PlotTab::Jobs && !self.jobs.is_empty() {
                        ui.separator();
                        ui.label(format!("{} job(s)", self.jobs.len()));
                    }
                    ui.separator();
                    if matches!(self.plot_tab, PlotTab::Voltage | PlotTab::Phase | PlotTab::Raster) {
                        ui.separator();
                        ui.label("window");
                        ui.add(
                            egui::DragValue::new(&mut self.plot_win.len_ms)
                                .range(20.0..=5000.0)
                                .suffix("ms"),
                        );
                        if !self.plot_win.follow {
                            if ui.button("Follow").clicked() {
                                self.plot_win.follow = true;
                            }
                        }
                        if matches!(self.plot_tab, PlotTab::Voltage | PlotTab::Phase) {
                            ui.checkbox(&mut self.plot_win.auto_y, "Auto Y");
                        }
                    }
                    if matches!(self.plot_tab, PlotTab::Voltage | PlotTab::Phase) {
                        let sel_n = self
                            .selection
                            .iter()
                            .filter(|&i| matches!(self.entity(*i), Some(Entity::Neuron(_))))
                            .count();
                        // U8: say which neurons the cap dropped.
                        ui.weak(format!(
                            "{} of {} watched",
                            self.watch_sent.len(),
                            sel_n
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
                        // Sliding window ending at "now" while following;
                        // the user's pan/zoom rules once follow is off.
                        let win = self.plot_win.bounds(self.last_sim_t);
                        let auto_y = self.plot_win.auto_y;
                        if self.watch_sent.is_empty() {
                            // U8: readable weak text above the plot, not
                            // tiny in-plot text across the centre line.
                            ui.weak("no watched neurons — click a neuron to \
                                     watch it, or right-click -> \"Watch v(t)\"");
                        }
                        let mut p = egui_plot::Plot::new("vplot")
                            .height(140.0)
                            .y_axis_formatter(|y, range| {
                                if crate::plot::y_label_hits_corner(&y, range) {
                                    String::new()
                                } else {
                                    format!("{:.0}", y.value)
                                }
                            });
                        if win.is_none() || auto_y {
                            // x (or y too when following) from the data.
                            p = p.auto_bounds(egui::Vec2b::new(win.is_none(), auto_y));
                        } else {
                            p = p.auto_bounds(egui::Vec2b::new(false, false));
                        }
                        let pr = p.show(ui, |ui| {
                            if self.hist.v.is_empty() {
                                // Discoverable watching (Stage 13 E): the
                            }
                            // watch_sent order → stable per-neuron colors,
                            // shared with the phase tab.
                            for (ci, &id) in self.watch_sent.iter().enumerate() {
                                let Some(h) = self.hist.v.get(&id) else { continue };
                                let pts: PlotPoints = match win {
                                    Some((lo, hi)) => h
                                        .iter()
                                        .filter(|(t, _)| (lo..=hi).contains(t))
                                        .map(|(t, v)| [*t, *v as f64])
                                        .collect(),
                                    None => h.iter().map(|(t, v)| [*t, *v as f64]).collect(),
                                };
                                ui.line(
                                    Line::new(self.watch_label(id), pts)
                                        .color(crate::plot::watch_color(ci)),
                                );
                            }
                            if let Some((lo, hi)) = win {
                                ui.set_plot_bounds_x(lo..=hi);
                                if !auto_y {
                                    // Fixed Izhikevich span: spikes can't
                                    // bounce the axis entering/leaving.
                                    ui.set_plot_bounds_y(
                                        crate::plot::V_Y_RANGE.0..=crate::plot::V_Y_RANGE.1,
                                    );
                                }
                            }
                        });
                        self.plot_follow(&pr, win);
                    }
                    PlotTab::Phase => {
                        // v–u phase portrait per watched neuron over the
                        // same window; the original had a Poincaré canvas
                        // for 2-D trajectory projections — no nullclines
                        // there or here (we don't have per-neuron params
                        // on the wire).
                        let win = self.plot_win.bounds(self.last_sim_t);
                        let p = egui_plot::Plot::new("phplot")
                            .height(140.0)
                            .y_axis_formatter(|y, range| {
                                if crate::plot::y_label_hits_corner(&y, range) {
                                    String::new()
                                } else {
                                    format!("{:.0}", y.value)
                                }
                            })
                            .x_axis_label("v (mV)")
                            .y_axis_label("u");
                        let _ = p.show(ui, |ui| {
                            if self.hist.u.is_empty() {
                                ui.text(egui_plot::Text::new(
                                    "hint",
                                    egui_plot::PlotPoint::new(-30.0, -10.0),
                                    "no watched neurons — watch a neuron to see\nits v–u trajectory here",
                                ));
                            }
                            for (ci, &id) in self.watch_sent.iter().enumerate() {
                                let (Some(vs), Some(us)) =
                                    (self.hist.v.get(&id), self.hist.u.get(&id))
                                else {
                                    continue;
                                };
                                let col = crate::plot::watch_color(ci);
                                let (pts, newest) = crate::plot::phase_points(vs, us, win);
                                if pts.is_empty() {
                                    continue;
                                }
                                ui.line(
                                    Line::new(self.watch_label(id), PlotPoints::from(pts))
                                        .color(col),
                                );
                                if let Some(p) = newest {
                                    ui.points(
                                        egui_plot::Points::new("now", vec![p])
                                            .radius(4.0_f32)
                                            .color(col),
                                    );
                                }
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
                        let win = self.plot_win.bounds(self.last_sim_t);
                        let plot = egui_plot::Plot::new("raster")
                            .height(self.raster_h.max(40.0))
                            .auto_bounds(egui::Vec2b::new(win.is_none(), false))
                            .include_y(-0.5)
                            .include_y(y_hi);
                        let pr = plot
                            .x_axis_formatter(|x, _| format!("{:.0}ms", x.value))
                            .y_axis_formatter(move |y, range| {
                                if crate::plot::y_label_hits_corner(&y, range) {
                                    return String::new();
                                }
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
                                if self.hist.raster.is_empty() {
                                    let cx = win.map(|(lo, hi)| (lo + hi) / 2.0)
                                        .unwrap_or(self.last_sim_t);
                                    ui.text(egui_plot::Text::new(
                                        "hint",
                                        egui_plot::PlotPoint::new(cx, 0.0),
                                        "run the simulation — each spike lands here",
                                    ));
                                }
                                let mut pts = Vec::new();
                                let watches = &self.watch_sent;
                                for &(t, ref fired) in &self.hist.raster {
                                    if let Some((lo, hi)) = win {
                                        if !(lo..=hi).contains(&t) {
                                            continue;
                                        }
                                    }
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
                                if let Some((lo, hi)) = win {
                                    ui.set_plot_bounds_x(lo..=hi);
                                }
                            });
                        self.plot_follow(&pr, win);
                    }
                    PlotTab::Readouts => self.plot_readouts(ui),
                    PlotTab::Samples => self.plot_samples(ui),
                    PlotTab::Jobs => self.jobs_ui(ui),
                }
            });
        self.show_plots = open;
    }

    /// Turn follow off when the user pans/zooms away from the imposed
    /// window (bounds no longer match what we set), and back on when
    /// they double-click — egui_plot's own "reset" gesture.
    fn plot_follow(&mut self, pr: &egui_plot::PlotResponse<()>, imposed: Option<(f64, f64)>) {
        if pr.response.double_clicked() {
            self.plot_win.follow = true;
            return;
        }
        if let Some((lo, hi)) = imposed {
            let b = pr.transform.bounds();
            let eps = (self.plot_win.len_ms * 1e-6).max(1e-6);
            if pr.response.dragged()
                || (b.min()[0] - lo).abs() > eps
                || (b.max()[0] - hi).abs() > eps
            {
                self.plot_win.follow = false;
            }
        }
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

    /// Jobs tab: server job list, live `JobProgress` plot for the
    /// selected job, cancel/re-open controls, and the opened detail
    /// (status + result.json + script).
    fn jobs_ui(&mut self, ui: &mut egui::Ui) {
        // U8: one fetch when the tab opens — JobStarted/JobDone/
        // JobProgress pushes keep the list live after that, so the
        // 2 s poll went away. Cancel clicks refetch on JobDone.
        if !self.jobs_fetched {
            self.jobs_fetched = true;
            self.send(ClientMsg::ListJobs);
        }
        if self.jobs.is_empty() {
            ui.label("no jobs — \"Run as job\" in the script editor starts one");
        }
        let jobs = self.jobs.clone();
        let rows = self
            .job_progress
            .get(&self.job_sel)
            .cloned()
            .unwrap_or_default();
        let detail = self.job_detail.as_ref().and_then(|d| {
            (d.id == self.job_sel).then(|| {
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
                        if ui.selectable_label(self.job_sel == j.id, label).clicked() {
                            self.job_sel = j.id;
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
                            self.job_sel,
                            rows.len()
                        ));
                    }
                    _ => {
                        ui.label(if self.job_sel == 0 {
                            "select a job to plot its progress".to_string()
                        } else {
                            format!("job #{} — no progress rows yet", self.job_sel)
                        });
                    }
                }
                if let Some((name, status, result_json, script)) = detail {
                    ui.collapsing("opened job", |ui| {
                        ui.label(format!("#{} {} — {}", self.job_sel, name, status));
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
                app_win(ctx, "Create 2D grid")
                    .open(&mut open)
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
                app_win(ctx, "Create 3D grid")
                    .open(&mut open)
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
                app_win(ctx, "Create connections")
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
                            self.send(ClientMsg::Eval {
                                code: format!(
                                    "net.create_connections({g:?}, {sc:?}, {la:?}, {ww:?})"
                                ),
                            });
                            keep = false;
                        }
                    });
            }
            Dialog::Weights { group, mean, sd } => {
                app_win(ctx, "Set synapse weights")
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
                            self.send(ClientMsg::Eval {
                                code: format!("net.set_synapse_weights({g:?}, {m:?}, {s:?})"),
                            });
                            keep = false;
                        }
                    });
            }
            Dialog::SynTau { src, dst, mean, sd } => {
                app_win(ctx, "Set synapse tau (ms)")
                    .open(&mut open)
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
                            let (a, b, m, s) = (src.clone(), dst.clone(), *mean, *sd);
                            self.send(ClientMsg::Eval {
                                code: format!("net.set_syn_tau_dist({a:?}, {b:?}, {m:?}, {s:?})"),
                            });
                            keep = false;
                        }
                    });
            }
            Dialog::ExcInh { group, frac } => {
                app_win(ctx, "Set excitatory/inhibitory")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("group");
                            self.group_combo(ui, "ei_group", group);
                        });
                        ui.add(egui::Slider::new(frac, 0.0..=1.0).text("inh frac"));
                        if ui.button("Apply").clicked() {
                            let (g, f) = (group.clone(), *frac);
                            self.send(ClientMsg::Eval {
                                code: format!("net.set_exc_inh({g:?}, {f:?})"),
                            });
                            keep = false;
                        }
                    });
            }
            Dialog::NewInput {
                name,
                weight,
                value,
            } => {
                app_win(ctx, "New input on selection")
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
                            keep = false;
                        }
                    });
            }
            Dialog::NewSamples { name } => {
                app_win(ctx, "New samples node")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("name");
                            ui.text_edit_singleline(name);
                        });
                        ui.label("empty (t,value) series — fill via net.samples_write");
                        if ui.button("Create").clicked() {
                            let n2 = name.clone();
                            self.edit_net(NetEdit::AddSamples { name: n2 });
                            keep = false;
                        }
                    });
            }
            Dialog::NewReadout { name } => {
                app_win(ctx, "New readout on selection")
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
                            self.edit_net(NetEdit::AddReadout { name: n2, units });
                            keep = false;
                        }
                    });
            }
            Dialog::AddToGroup { name } => {
                app_win(ctx, "Add selection to group")
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
                            self.edit_net(NetEdit::AddToGroup { name: nm, ids });
                            keep = false;
                        }
                    });
            }
            Dialog::Discard { then } => {
                app_win(ctx, "Unsaved changes")
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
                app_win(ctx, "Save as").open(&mut open).show(ctx, |ui| {
                    ui.text_edit_singleline(path);
                    if ui.button("Save").clicked() {
                        let p = path.clone();
                        self.send(ClientMsg::SaveNet { path: p });
                        keep = false;
                    }
                });
            }
            Dialog::Load { path } => {
                app_win(ctx, "Load network")
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(path);
                            if ui.button("list").clicked() {
                                self.send(ClientMsg::ListFiles { path: dir_of(path) });
                            }
                            if ui.button("Load").clicked() {
                                let p = path.clone();
                                self.guarded(ctx, PendingAction::Load(p));
                                keep = false;
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
            Dialog::About => {
                app_win(ctx, "About neuro_edit v2")
                    .open(&mut open)
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
        if open && keep {
            self.dlg = Some(d);
        }
    }
}

impl eframe::App for EditorApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let p = Persisted {
            recent: self.recent.iter().cloned().collect(),
            windows: WindowVis {
                inspector: self.show_inspector,
                plots: self.show_plots,
                console: self.show_console,
                script: self.show_script,
                inputs: self.show_inputs,
                files: self.show_files,
            },
            token: self.saved_token.clone(),
        };
        eframe::set_value(storage, "state", &p);
        secure_storage_file();
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        crate::icons::Icons::ensure(&mut self.icons, ctx);
        self.insert_inh = ctx.input(|i| i.modifiers.alt);
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
        self.wall_ms = ctx.input(|i| i.time) * 1000.0;
        // Smoothed frame cost (10% EMA) for the fps readout. Gaps over
        // 200 ms are idle sleeps, not slow frames — skip them.
        let dt_ms = (self.wall_ms - self.last_frame_ms) as f32;
        self.last_frame_ms = self.wall_ms;
        if dt_ms > 0.0 && dt_ms < 200.0 {
            self.frame_ms = if self.frame_ms <= 0.0 {
                dt_ms
            } else {
                self.frame_ms * 0.9 + dt_ms * 0.1
            };
        }
        crate::view::with_gpu(rs, |gpu| self.sync_shared(gpu));
        self.update_watch();
        if self.mode != self.last_mode {
            self.status.clear();
            self.last_mode = self.mode;
        }

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
        // system paste events (Ctrl+V when no text field owns it) —
        // Q9: a Paste event fired while a text field has keyboard
        // focus belongs to the field, not the network.
        for txt in paste_events(&ctx.input(|i| i.events.clone()), ctx.wants_keyboard_input()) {
            self.paste(Some(txt));
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
        self.hints_bar(ctx);
        self.disconnect_banner(ctx);
        self.inspector(ctx);
        self.console(ctx);
        self.script_window(ctx);
        self.plots(ctx);
        self.inputs_window(ctx);
        self.files_window(ctx);
        #[cfg(not(target_arch = "wasm32"))]
        self.login_window(ctx);
        self.keys_window(ctx);
        self.manual_window(ctx);
        self.dialog(ctx);

        crate::view::viewport(self, ctx, rs);
        // Stage 14 E: repaint on demand, not unconditionally — a quiet app
        // (paused sim, still camera, no drags/picks/uploads) sleeps between
        // input events. Server traffic flags itself via net_touched.
        let animating = self.running
            || self.busy
            || self.net_touched
            || self.camera.animating()
            || self.move_drag.is_some()
            || self.box_sel.is_some()
            || self.parts_active
            || self.pick_click.is_some()
            || self.pick_in_flight
            || !self.files.upload_q.is_empty();
        if animating {
            ctx.request_repaint();
        }
        // U5b: one quiet heartbeat per second even when fully idle —
        // transport writes (a closed socket, a late ServerMsg) bump
        // dirty_seq inside Shared, but nothing else wakes a frame to
        // notice, so an idle page would sit frozen forever. 1 fps is
        // effectively free and bounds the disconnect notice to ~1 s.
        ctx.request_repaint_after(std::time::Duration::from_secs(1));
        self.net_touched = false;
    }
}

