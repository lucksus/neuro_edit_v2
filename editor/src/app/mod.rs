//! eframe App: toolbar, 3D viewport (egui-wgpu callback), inspector,
//! console, plots. Communicates with the sim worker via `SimCtl`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use eframe::egui;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::scene::{Particles, Plane};
use crate::simctl::SimCtl;
use crate::snap::GeoSnap;
use proto::{ClientMsg, NetEdit};

mod actions;
mod console;
mod dialogs;
mod files;
mod help;
mod inspector;
mod jobs;
mod layout;
mod script;
#[cfg(not(target_arch = "wasm32"))]
mod login;
mod plots;
mod sync;
mod toolbar;

#[cfg(not(target_arch = "wasm32"))]
use files::local_files_root;
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
    /// Files-window + picker + recent-files state.
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
    /// Server jobs (list + live progress rows + last opened detail).
    jobs: Vec<proto::JobInfo>,
    job_progress: std::collections::BTreeMap<u64, Vec<Vec<(String, f64)>>>,
    job_detail: Option<crate::snap::JobDetailSnap>,
    /// Job selected in the jobs tab (id; 0 = none yet).
    job_sel: u64,
    /// U8: the Jobs tab fetched the list since it was opened —
    /// pushes update it live, no 2 s poll.
    jobs_fetched: bool,

    /// Run-controls state (speed/backend/busy/dt/noise) — toolbar run
    /// cluster, fed by `Shared` snapshots.
    sim: SimControls,
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
    /// Last `FileList` reply the open/save picker renders.
    pub file_list: Option<(String, Vec<proto::FileEntry>)>,
    /// Recently opened nets (File → Open recent).
    pub recent: VecDeque<String>,
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

/// Run-controls state shown in the toolbar's sim cluster: requested
/// speed, worker backend, run/build/busy flags, dt and noise params.
pub struct SimControls {
    pub speed: f32,
    /// Best steps/s the machine has reached — shown as `(max xN)` next to
    /// the requested speed so "how fast can this go" is one display.
    pub speed_peak: f32,
    /// Backend the worker reports (request combo selected text).
    pub backend: proto::Backend,
    pub running: bool,
    pub sim_built: bool,
    pub busy: bool,
    pub dt_ms: f32,
    pub noise_hz: f32,
    pub noise_kick: f32,
}

impl Default for SimControls {
    fn default() -> Self {
        Self {
            speed: 1.0,
            speed_peak: 0.0,
            backend: proto::Backend::Cpu,
            running: false,
            sim_built: false,
            busy: false,
            dt_ms: 0.1,
            noise_hz: 0.0,
            noise_kick: 15.0,
        }
    }
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
                f.recent = recent;
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
            sim: SimControls::default(),
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
























    // ---------------- panels ----------------









    // ------------------------------------------------------------------
    // Files window — one UI, two backends: the file protocol (remote,
    // per-user root) or std::fs on the local app-data dir.
    // ------------------------------------------------------------------












    // ------------------------------------------------------------------
    // Login dialog — `--connect http(s)://host` multi-user servers.
    // ------------------------------------------------------------------


















}

impl eframe::App for EditorApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let p = Persisted {
            recent: self.files.recent.iter().cloned().collect(),
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
        let animating = self.sim.running
            || self.sim.busy
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

