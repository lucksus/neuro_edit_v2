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
    /// `ListJobs` poll timestamp (egui time, seconds) while the tab is open.
    last_jobs_poll: f64,
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

fn fmt_size(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.1} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

fn fmt_date(ms: u64) -> String {
    if ms == 0 {
        return "—".into();
    }
    // day-precision UTC, civil-from-days
    let z = (ms / 86_400_000) as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{:02}-{:02}", if m <= 2 { y + 1 } else { y }, m, d)
}

/// Browser download: bytes → Blob → object URL → anchor click.
#[cfg(target_arch = "wasm32")]
fn download_blob(name: &str, bytes: &[u8]) {
    use wasm_bindgen::JsCast;
    let Some(w) = web_sys::window() else { return };
    let Some(doc) = w.document() else { return };
    let arr = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::new();
    parts.push(&arr);
    let Ok(blob) = web_sys::Blob::new_with_u8_array_sequence(&parts) else {
        return;
    };
    let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) else {
        return;
    };
    if let Ok(a) = doc.create_element("a") {
        if let Ok(a) = a.dyn_into::<web_sys::HtmlAnchorElement>() {
            a.set_href(&url);
            a.set_download(name);
            a.click();
        }
    }
    let _ = web_sys::Url::revoke_object_url(&url);
}

// Browser file picker: a transient `<input type=file multiple>` whose
// `change` handler FileReader-reads each File into PICKED, drained by
// `files_pump` on the next frames.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PICKED: std::cell::RefCell<Vec<(String, Vec<u8>)>> =
        std::cell::RefCell::new(Vec::new());
}

#[cfg(target_arch = "wasm32")]
fn web_file_picker() {
    use wasm_bindgen::{closure::Closure, JsCast};
    let Some(w) = web_sys::window() else { return };
    let Some(doc) = w.document() else { return };
    let Ok(el) = doc.create_element("input") else {
        return;
    };
    let Ok(input) = el.dyn_into::<web_sys::HtmlInputElement>() else {
        return;
    };
    input.set_type("file");
    input.set_multiple(true);
    // the input element stays alive via this closure's capture
    let inp = input.clone();
    let onchange = Closure::wrap(Box::new(move |_: web_sys::Event| {
        let Some(list) = inp.files() else { return };
        for j in 0..list.length() {
            let Some(file) = list.item(j) else { continue };
            let name = file.name();
            let Ok(reader) = web_sys::FileReader::new() else {
                continue;
            };
            let r2 = reader.clone();
            let onload = Closure::wrap(Box::new(move |_: web_sys::Event| {
                if let Ok(buf) = r2.result() {
                    let arr = js_sys::Uint8Array::new(&buf);
                    PICKED.with(|p| p.borrow_mut().push((name.clone(), arr.to_vec())));
                }
            }) as Box<dyn FnMut(web_sys::Event)>);
            reader.set_onload(Some(onload.as_ref().unchecked_ref()));
            let _ = reader.read_as_array_buffer(&file);
            onload.forget();
        }
    }) as Box<dyn FnMut(web_sys::Event)>);
    input.set_onchange(Some(onchange.as_ref().unchecked_ref()));
    input.click();
    onchange.forget();
}

#[cfg(not(target_arch = "wasm32"))]
fn local_dir_size(root: &std::path::Path) -> u64 {
    std::fs::read_dir(root)
        .map(|rd| {
            rd.flatten()
                .map(|e| {
                    let p = e.path();
                    if p.is_dir() {
                        local_dir_size(&p)
                    } else {
                        e.metadata().map(|m| m.len()).unwrap_or(0)
                    }
                })
                .sum()
        })
        .unwrap_or(0)
}

/// `dirs::data_dir()/neuro_edit` — the native no-`--connect` file space,
/// created on first use.
#[cfg(not(target_arch = "wasm32"))]
pub fn local_files_root() -> std::path::PathBuf {
    let d = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("neuro_edit");
    let _ = std::fs::create_dir_all(&d);
    d
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
            last_jobs_poll: -10.0,
            show_grid: true,
            show_synapses: true,
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

    /// Render the latest `ListFiles` reply as clickable entries under a
    /// file picker; returns `(path, is_dir)` of the picked entry.
    fn file_list_ui(&self, ui: &mut egui::Ui) -> Option<(String, bool)> {
        let (dir, entries) = self.file_list.as_ref()?;
        let mut picked = None;
        egui::ScrollArea::vertical()
            .max_height(120.0)
            .show(ui, |ui| {
                for e in entries {
                    let label = if e.is_dir {
                        format!("{}/", e.name)
                    } else {
                        format!("{} ({} B)", e.name, e.size)
                    };
                    if ui.small_button(label).clicked() {
                        picked = Some((
                            if dir == "." {
                                e.name.clone()
                            } else {
                                format!("{dir}/{}", e.name)
                            },
                            e.is_dir,
                        ));
                    }
                }
            });
        picked
    }

    /// Apply a structured edit — versioned, delta-able, and expressible
    /// on the wire (unlike the old closure path).
    fn edit_net(&self, e: NetEdit) {
        self.ctl.send_edit(e);
    }

    /// Context-menu helper: watch a neuron's v(t) by selecting it (the
    /// plots derive their watch list from the selection).
    pub fn watch(&mut self, ids: Vec<u32>) {
        self.selection = ids.into_iter().collect();
    }

    /// Plot/legend label for a watched neuron: its name when set, the
    /// id always — "n7" style falls back to the bare id.
    fn watch_label(&self, id: u32) -> String {
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
        self.send(ClientMsg::LoadNet { path: path.clone() });
        self.net_path = Some(path.clone());
        self.push_recent(path);
        self.selection.clear();
    }

    pub fn push_recent(&mut self, path: String) {
        self.recent.retain(|p| *p != path);
        self.recent.push_front(path);
        self.recent.truncate(8);
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
            self.send(ClientMsg::Watch(ids));
            self.hist.v.retain(|k, _| self.watch_sent.contains(k));
            self.hist.u.retain(|k, _| self.watch_sent.contains(k));
        }
    }

    pub fn entity(&self, inst_id: u32) -> Option<Entity> {
        self.geo
            .as_ref()
            .and_then(|g| crate::view::entity_of(g, inst_id))
    }

    /// Place a neuron on the work plane (insert mode).
    pub fn insert_neuron(&mut self, p: glam::Vec3) {
        self.edit_net(NetEdit::AddNeuron {
            pos: [p.x, p.y, p.z],
            kind: proto::NeuronKind::Excitatory,
        });
        self.status = format!("placed neuron at ({:.0},{:.0},{:.0})", p.x, p.y, p.z);
    }

    /// Finish a connect drag: synapse a→b with default weight by kind.
    pub fn connect_neurons(&mut self, a: u32, b: u32) {
        if a == b {
            self.status = format!("can't connect neuron {a} to itself");
            return;
        }
        self.edit_net(NetEdit::Connect {
            from: a,
            to: b,
            w: 6.0,
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
            self.send(ClientMsg::Watch(Vec::new()));
        }
        self.hist.clear();
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
        // Any ServerMsg that landed since the last frame bumps dirty_seq —
        // that's our "repaint on transport" signal for Stage 14 E.
        {
            let sh = self.ctl.shared.read().unwrap();
            if sh.dirty_seq != self.last_dirty_seq {
                self.net_touched = true;
                self.last_dirty_seq = sh.dirty_seq;
            }
        }
        // One-shot consumables the worker answered for us.
        {
            let mut sh = self.ctl.shared.write().unwrap();
            if let Some((path, text)) = sh.file_data.take() {
                self.script_file = path;
                self.script_text = text;
                self.status = "loaded".into();
            }
            if let Some(p) = sh.file_written.take() {
                self.status = format!("wrote {p}");
            }
            if let Some((dir, entries)) = sh.files.take() {
                if self.files.local_root.is_none() && dir == self.files.cwd {
                    self.files.entries = entries.clone();
                }
                self.file_list = Some((dir, entries));
            }
            if let Some((used, quota)) = sh.usage.take() {
                self.files.usage = Some((used, quota));
            }
            if let Some(op) = sh.file_op.take() {
                // a mkdir/delete/rename/upload landed — relist
                if let Some(p) = op.strip_prefix("upload ready ") {
                    // drop the lock: upload_ready sends client messages
                    drop(sh);
                    self.upload_ready(p);
                    sh = self.ctl.shared.write().unwrap();
                } else {
                    drop(sh);
                    self.files_refresh();
                    self.status = op;
                    sh = self.ctl.shared.write().unwrap();
                }
            }
            let dl = sh.download.take();
            if let Some((path, bytes, done)) = dl {
                if done {
                    drop(sh);
                    self.save_download(&path, &bytes);
                    sh = self.ctl.shared.write().unwrap();
                } else {
                    // still streaming — put it back
                    sh.download = Some((path, bytes, done));
                }
            }
            self.jobs = sh.jobs.clone();
            self.job_progress = sh.job_progress.clone();
            if let Some(d) = sh.job_detail.take() {
                // a re-opened job's history becomes plottable too
                self.job_progress.insert(d.id, d.progress.clone());
                self.job_detail = Some(d);
            }
        }
        let (geo_new, snap, running, sim_built, busy, net_gen, cfg, backend) = {
            let sh = self.ctl.shared.read().unwrap();
            (
                sh.geo.clone(),
                sh.snap.clone(),
                sh.running,
                sh.sim_built,
                sh.busy,
                sh.net_gen,
                sh.cfg.clone(),
                sh.backend,
            )
        };
        // Toolbar widgets read the worker's live cfg so script-side
        // `sim.set_dt`/`sim.set_noise`/`sim.set_backend` calls show up.
        self.dt_ms = cfg.dt_ms;
        self.noise_hz = cfg.noise_rate_hz;
        self.noise_kick = cfg.noise_kick_mv;
        self.backend = backend;
        let gen_bumped = net_gen != self.last_net_gen;
        if gen_bumped {
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
                // Floor grid + camera clamp follow the net's own baseline.
                self.camera.floor_y = g.floor_y();
                self.inst_half = gpu
                    .scene
                    .as_ref()
                    .map(|s| s.inst_half.clone())
                    .unwrap_or_default();
                let was_empty = self.geo.as_ref().is_none_or(|old| old.pos.is_empty());
                if !g.pos.is_empty() && (was_empty || self.frame_pending) {
                    let (mn, mx) = gpu.scene.as_ref().unwrap().bounds;
                    self.camera.frame(mn, mx);
                }
                self.frame_pending = false;
                self.geo = Some(g);
            }
        }
        // A loaded example asked to auto-watch its neurons — apply it
        // once the new Geo covers the ids. The gen bump from a script's
        // net.clear may land a frame AFTER the covering Geo and clear the
        // selection again — keep re-applying until update_watch has sent
        // the watch (watch_sent matches), then stop.
        if let Some(ids) = self.example_watch {
            if ids.is_empty() || self.wall_ms > self.example_deadline_ms {
                self.example_watch = None;
            } else {
                // Only a Geo newer than the Run click counts — the pre-run
                // net can already cover the ids, and watching it just gets
                // wiped by the eval's net.clear → reset_net_state.
                let covered = example_geo_covers(self.example_base_ver, self.geo.as_deref(), ids);
                if covered {
                    // Rebuild the HashSet only when it differs — a fresh set
                    // gets a new random iteration order and update_watch would
                    // resend the same Watch list every frame.
                    if self.selection.len() != ids.len()
                        || ids.iter().any(|n| !self.selection.contains(n))
                    {
                        self.selection = ids.iter().copied().collect();
                    }
                    self.show_plots = true;
                    self.plot_tab = PlotTab::Voltage;
                }
                // Retire only after the watch has stayed applied across a
                // quiet stretch — a net-gen bump inside the eval (net.clear)
                // may land one frame before its Geo and clear the selection
                // again; this re-applies on the next frame.
                if covered
                    && self.watch_sent.len() == ids.len()
                    && ids.iter().all(|n| self.watch_sent.contains(n))
                    && !gen_bumped
                {
                    self.example_quiet += 1;
                    if self.example_quiet > 30 {
                        self.example_watch = None;
                        self.example_quiet = 0;
                    }
                } else {
                    self.example_quiet = 0;
                }
            }
        }
        if snap.seq != self.last_snap_seq {
            self.last_snap_seq = snap.seq;
            self.last_sim_t = snap.time_ms;
            self.speed_peak = self.speed_peak.max(snap.steps_per_sec);
            self.activity = snap.activity.to_vec();
            self.grid_traces = snap.grid_traces.clone();
            self.input_values = snap.input_values.clone();
            if let Some(g) = &self.geo {
                self.particles
                    .feed(g, &snap.counts, snap.time_ms, self.wall_ms);
            }
            // raster bin: ids with count>0 — folded into the histories
            // by PlotHist::feed, which also spots the sim clock going
            // backwards (Reset, sim.reset(), another client) and clears.
            let fired: Vec<u32> = snap
                .counts
                .iter()
                .enumerate()
                .filter(|(_, c)| **c > 0)
                .map(|(i, _)| i as u32)
                .collect();
            // watched_v/_u are parallel vectors on the wire — pair by id,
            // u missing ⇒ empty series (older servers).
            let watched: Vec<(u32, &[f32], &[f32])> = snap
                .watched_v
                .iter()
                .map(|(id, vs)| {
                    let us = snap
                        .watched_u
                        .iter()
                        .find(|(i, _)| i == id)
                        .map(|(_, us)| us.as_slice())
                        .unwrap_or(&[]);
                    (*id, vs.as_slice(), us)
                })
                .collect();
            self.hist
                .feed(snap.time_ms, fired, &watched, self.dt_ms as f64);
            // A reset while the user was panned away snaps the plots
            // back to live view — a rewind always wants "now".
            if self.hist.resets != self.hist_resets_seen {
                self.hist_resets_seen = self.hist.resets;
                self.plot_win.follow = true;
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
            self.plane_key = 0;
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
            self.plane_cfgs_last = cfgs.clone();
        }
        // SceneKey contribution: everything this call fed the GPU.
        let mut h = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        h.write(bytemuck::cast_slice(&params));
        for c in &cfgs {
            format!("{c:?}").hash(&mut h);
        }
        h.write(bytemuck::cast_slice(&bits));
        on.hash(&mut h);
        need_upload.hash(&mut h);
        self.plane_key = h.finish();
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
        let ic = match &self.icons {
            Some(i) => i.clone(),
            None => return,
        };
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            // Wrapped, not plain horizontal: a plain row that overflows
            // grows the panel's max_rect past the window, which makes the
            // sibling `horizontal_wrapped` mode row see an over-wide
            // boundary and never wrap — controls clip off-screen instead.
            ui.horizontal_wrapped(|ui| {
                ui.menu_button("File", |ui| {
                    if ui.add(icons::button(&ic.new, "New")).clicked() {
                        self.guarded(ctx, PendingAction::New);
                        ui.close();
                    }
                    if ui.add(icons::button(&ic.open, "Load…")).clicked() {
                        self.dlg = Some(Dialog::Load {
                            path: self.net_path.clone().unwrap_or_default(),
                        });
                        ui.close();
                    }
                    ui.menu_image_text_button(
                        icons::tinted(&ic.recent, ui.visuals().text_color()),
                        "Recent files",
                        |ui| {
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
                        },
                    );
                    if ui.add(icons::button(&ic.save, "Save")).clicked() {
                        match &self.net_path {
                            Some(p) => {
                                self.send(ClientMsg::Eval {
                                    code: format!("net.save({p:?})"),
                                });
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
                    if ui.add(icons::button(&ic.save_as, "Save as…")).clicked() {
                        self.dlg = Some(Dialog::SaveAs {
                            path: self.net_path.clone().unwrap_or("net.ne2".into()),
                        });
                        ui.close();
                    }
                    ui.separator();
                    if ui.add(icons::button(&ic.quit, "Quit")).clicked() {
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
                    if ui.button("Set synapse tau…").clicked() {
                        self.dlg = Some(Dialog::SynTau {
                            src: "grid".into(),
                            dst: "grid".into(),
                            mean: 2.0,
                            sd: 0.0,
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
                ui.menu_button("View", |ui| {
                    // toggles stay open so several windows can flip at once
                    for (flag, icon, label) in [
                        (&mut self.show_inspector, &ic.inspector, "Inspector"),
                        (&mut self.show_plots, &ic.plots, "Plots"),
                        (&mut self.show_console, &ic.console, "Console"),
                        (&mut self.show_script, &ic.script, "Script editor"),
                        (&mut self.show_inputs, &ic.connect, "Inputs"),
                        (&mut self.show_files, &ic.save, "Files"),
                        (&mut self.show_fps, &ic.speed, "frame stats"),
                    ] {
                        if ui.add(icons::button(icon, label).selected(*flag)).clicked() {
                            *flag = !*flag;
                        }
                    }
                    ui.separator();
                    if ui.button("Reset layout").clicked() {
                        // windows return to their default positions
                        ctx.memory_mut(|m| m.reset_areas());
                        ui.close();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("Manual  (?)").clicked() {
                        self.show_manual = true;
                        ui.close();
                    }
                    if ui.button("Shortcuts").clicked() {
                        self.show_keys = true;
                        ui.close();
                    }
                    ui.menu_button("Examples", |ui| {
                        for ex in crate::examples::EXAMPLES {
                            if ui.button(ex.name).clicked() {
                                self.load_example(ex);
                                ui.close();
                            }
                        }
                    });
                    if ui.add(icons::button(&ic.about, "About")).clicked() {
                        self.dlg = Some(Dialog::About);
                        ui.close();
                    }
                });
                // Always-visible help affordance.
                if ui.button("?").on_hover_text("Manual").clicked() {
                    self.show_manual = true;
                }
                ui.separator();
                // sim controls — "Stop" (not "Pause"): abort_now flips the
                // flag on_progress/run chunks/builders watch. A busy script
                // can't resume anyway, so Pause was a lie.
                let live = self.running || self.busy;
                let (run_icon, run_label) = if live {
                    (&ic.stop, "Stop")
                } else {
                    (&ic.run, "Run")
                };
                // Stage 15 B: each control group (icon+widget+label) is a
                // nested horizontal — wrapping moves a group whole instead
                // of splitting e.g. the speed slider from its "realtime".
                toolbar_group(ui, 300.0, |ui| {
                    if ui.add(icons::button(run_icon, run_label)).clicked() {
                        if live {
                            self.ctl.abort_now();
                            self.send(ClientMsg::AutoRun(false));
                        } else {
                            self.send(ClientMsg::AutoRun(true));
                        }
                    }
                    if ui
                        .add(icons::button(&ic.step, "10ms"))
                        .on_hover_text("Step 10 ms")
                        .clicked()
                    {
                        self.send(ClientMsg::RunMs(10.0));
                    }
                    if ui
                        .add(icons::button(&ic.step_long, "100ms"))
                        .on_hover_text("Step 100 ms")
                        .clicked()
                    {
                        self.send(ClientMsg::RunMs(100.0));
                    }
                    if ui.add(icons::button(&ic.reset, "Reset")).clicked() {
                        self.send(ClientMsg::Reset);
                        self.hist.clear();
                    }
                });
                toolbar_group(ui, 430.0, |ui| {
                    let fg = ui.visuals().text_color();
                    ui.add(icons::tinted(&ic.speed, fg))
                        .on_hover_text("Speed (realtime factor)");
                    if ui
                        .add(egui::Slider::new(&mut self.speed, 0.05..=20.0).logarithmic(true))
                        .changed()
                    {
                        self.send(ClientMsg::SetSpeed(self.speed));
                    }
                    // One speed display: requested realtime factor + the best
                    // this machine has reached (peak steps/s → realtime).
                    let max_x = self.speed_peak * self.dt_ms / 1000.0;
                    ui.label(if max_x > 0.05 {
                        format!("x{:.1} realtime (max x{:.0})", self.speed, max_x)
                    } else {
                        format!("x{:.1} realtime", self.speed)
                    });
                });
                ui.separator();
                toolbar_group(ui, 120.0, |ui| {
                    let backend_icon = match self.backend {
                        proto::Backend::Cuda => &ic.backend_gpu,
                        proto::Backend::Cpu => &ic.backend_cpu,
                    };
                    ui.add(icons::tinted(backend_icon, ui.visuals().text_color()))
                        .on_hover_text("Backend");
                    egui::ComboBox::from_id_salt("backend")
                        .selected_text(match self.backend {
                            proto::Backend::Cuda => "cuda",
                            proto::Backend::Cpu => "cpu",
                        })
                        .show_ui(ui, |ui| {
                            if ui.add(icons::button(&ic.backend_gpu, "cuda")).clicked() {
                                self.send(ClientMsg::SetBackend(proto::Backend::Cuda));
                            }
                            if ui.add(icons::button(&ic.backend_cpu, "cpu")).clicked() {
                                self.send(ClientMsg::SetBackend(proto::Backend::Cpu));
                            }
                        });
                });
                toolbar_group(ui, 140.0, |ui| {
                    ui.add(icons::tinted(&ic.timestep, ui.visuals().text_color()))
                        .on_hover_text("Time step dt");
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.dt_ms)
                                .range(0.01..=1.0)
                                .speed(0.01)
                                .suffix(" ms"),
                        )
                        .changed()
                    {
                        self.send(ClientMsg::SetDtMs(self.dt_ms));
                    }
                });
                toolbar_group(ui, 160.0, |ui| {
                    ui.add(icons::tinted(&ic.noise, ui.visuals().text_color()))
                        .on_hover_text("Background noise rate");
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.noise_hz)
                                .range(0.0..=5000.0)
                                .speed(1.0)
                                .suffix(" Hz"),
                        )
                        .changed()
                    {
                        self.send(ClientMsg::SetNoise {
                            hz: self.noise_hz,
                            kick: self.noise_kick,
                        });
                    }
                });
            });
            ui.horizontal_wrapped(|ui| {
                // Same wrapping rule as row 1: groups move whole.
                toolbar_group(ui, 380.0, |ui| {
                    let m = &mut self.mode;
                    icons::selectable_value(ui, m, Mode::Navigate, &ic.navigate, "Navigate");
                    icons::selectable_value(ui, m, Mode::Insert, &ic.insert, "Insert");
                    icons::selectable_value(ui, m, Mode::Connect, &ic.connect, "Connect");
                    icons::selectable_value(ui, m, Mode::Move, &ic.mv, "Move");
                });
                ui.separator();
                toolbar_group(ui, 240.0, |ui| {
                    ui.label("plane");
                    let p = &mut self.plane;
                    icons::selectable_value(ui, p, Plane::XZ, &ic.plane_xz, "x-z");
                    icons::selectable_value(ui, p, Plane::XY, &ic.plane_xy, "x-y");
                    icons::selectable_value(ui, p, Plane::YZ, &ic.plane_yz, "y-z");
                });
                ui.separator();
                toolbar_group(ui, 340.0, |ui| {
                    icons::toggle(ui, &mut self.show_synapses, &ic.synapses, "synapses");
                    icons::toggle(ui, &mut self.show_grid, &ic.grid, "grid");
                    icons::toggle(ui, &mut self.show_dots, &ic.raster, "spike dots");
                });
                // Grid-group render mode: the plane LOD keeps big grids
                // interactive; cubes stay available for editing up close.
                toolbar_group(ui, 150.0, |ui| {
                    let view_icon = match self.grid_view {
                        crate::plane::GridView::Auto => &ic.view_auto,
                        crate::plane::GridView::Cubes => &ic.view_cubes,
                        crate::plane::GridView::Planes => &ic.view_planes,
                    };
                    ui.add(icons::tinted(view_icon, ui.visuals().text_color()))
                        .on_hover_text("Grid view");
                    egui::ComboBox::from_id_salt("grid_view")
                        .selected_text(match self.grid_view {
                            crate::plane::GridView::Auto => "auto",
                            crate::plane::GridView::Cubes => "cubes",
                            crate::plane::GridView::Planes => "planes",
                        })
                        .show_ui(ui, |ui| {
                            use crate::plane::GridView;
                            let g = &mut self.grid_view;
                            icons::selectable_value(ui, g, GridView::Auto, &ic.view_auto, "auto");
                            icons::selectable_value(
                                ui,
                                g,
                                GridView::Cubes,
                                &ic.view_cubes,
                                "cubes",
                            );
                            icons::selectable_value(
                                ui,
                                g,
                                GridView::Planes,
                                &ic.view_planes,
                                "planes",
                            );
                        });
                });
                if ui
                    .add(icons::icon_button(&ic.frame))
                    .on_hover_text("Frame network (F)")
                    .clicked()
                {
                    self.frame_all();
                }
                ui.separator();
                ui.label(&self.status);
            });
        });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        let (t, rate, spikes) = {
            let sh = self.ctl.shared.read().unwrap();
            (sh.snap.time_ms, sh.snap.rate_hz, sh.snap.spikes_total)
        };
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("t = {t:.1} ms"));
                ui.separator();
                ui.label(format!("rate = {rate:.1} Hz"));
                ui.separator();
                ui.label(format!("spikes = {spikes}"));
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
                        ui.label(label);
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

    /// Reserved strip for the mode/drag hint (Stage 15 C): as a panel it
    /// shrinks `usable_rect`, so floating windows can never cover it.
    fn hints_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("hints").show(ctx, |ui| {
            if let Some(md) = &self.move_drag {
                ui.label(format!(
                    "plane: {} · Shift: vertical · X/Y/Z: axis lock · Esc: cancel",
                    crate::view::drag_plane_name(md.plane)
                ));
            } else {
                // Narrow viewports wrap to a second line rather than
                // clipping the hint tail.
                ui.add(egui::Label::new(crate::view::mode_hint(self.mode)).wrap());
            }
        });
    }

    fn inspector(&mut self, ctx: &egui::Context) {
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
                                        let max = gs.dims[cfg.axis.min(2) as usize] - 1;
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
    fn inputs_window(&mut self, ctx: &egui::Context) {
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

    // ------------------------------------------------------------------
    // Files window — one UI, two backends: the file protocol (remote,
    // per-user root) or std::fs on the local app-data dir.
    // ------------------------------------------------------------------

    /// Refresh `files.entries` for the current `cwd` (remote → ListFiles;
    /// local → read_dir) + the usage numbers.
    fn files_refresh(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(root) = self.files.local_root.clone() {
            let dir = root.join(&self.files.cwd);
            self.files.entries = std::fs::read_dir(&dir)
                .map(|rd| {
                    let mut v: Vec<proto::FileEntry> = rd
                        .flatten()
                        .filter_map(|e| {
                            let md = e.metadata().ok()?;
                            Some(proto::FileEntry {
                                name: e.file_name().to_string_lossy().into_owned(),
                                is_dir: md.is_dir(),
                                size: md.len(),
                                mtime_ms: md
                                    .modified()
                                    .ok()
                                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                    .map(|d| d.as_millis() as u64)
                                    .unwrap_or(0),
                            })
                        })
                        .collect();
                    v.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
                    v
                })
                .unwrap_or_default();
            self.files.usage = Some((local_dir_size(&root), 0));
            return;
        }
        self.ctl.send(ClientMsg::ListFiles {
            path: self.files.cwd.clone(),
        });
        self.ctl.send(ClientMsg::FileUsage);
    }

    /// `UploadReady` arrived — stream the queued chunks for this path.
    fn upload_ready(&mut self, path: &str) {
        let Some((p, data, off)) = &self.files.upload else {
            return;
        };
        if p != path {
            return;
        }
        // send the whole remaining tail; the writer thread queues them
        let mut off = *off;
        while off < data.len() {
            let end = (off + 256 * 1024).min(data.len());
            self.ctl.send(ClientMsg::UploadChunk {
                path: p.clone(),
                data: data[off..end].to_vec(),
                last: end == data.len(),
            });
            off = end;
        }
        if let Some(u) = self.files.upload.as_mut() {
            u.2 = off;
        }
    }

    /// Completed `DownloadFile` stream — hand the bytes to the user.
    fn save_download(&mut self, path: &str, bytes: &[u8]) {
        let name = std::path::Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "download".into());
        #[cfg(target_arch = "wasm32")]
        {
            download_blob(&name, bytes);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(dst) = rfd::FileDialog::new().set_file_name(&name).save_file() {
                match std::fs::write(&dst, bytes) {
                    Ok(()) => self.files.status = format!("saved {}", dst.display()),
                    Err(e) => self.files.status = format!("save: {e}"),
                }
            }
        }
    }

    /// Queue an upload (remote → chunked protocol; local → copy).
    fn queue_upload(&mut self, name: &str, bytes: Vec<u8>) {
        if let Some(root) = self.files.local_root.clone() {
            let dst = root.join(&self.files.cwd).join(name);
            match std::fs::write(&dst, &bytes) {
                Ok(()) => self.files.status = format!("uploaded {name}"),
                Err(e) => self.files.status = format!("upload {name}: {e}"),
            }
            self.files_refresh();
            return;
        }
        self.files.upload_q.push_back((
            if self.files.cwd.is_empty() {
                name.to_string()
            } else {
                format!("{}/{name}", self.files.cwd)
            },
            bytes,
        ));
    }

    /// Poll the upload queue: idle → `UploadStart` for the next item.
    fn files_pump(&mut self) {
        if self.files.local_root.is_none() && self.files.upload.is_none() {
            if let Some((p, data)) = self.files.upload_q.pop_front() {
                self.ctl.send(ClientMsg::UploadStart {
                    path: p.clone(),
                    size: data.len() as u64,
                });
                self.files.upload = Some((p, data, 0));
            }
        }
        // browser file-picker results (web only)
        #[cfg(target_arch = "wasm32")]
        PICKED.with(|p| {
            for (name, bytes) in p.borrow_mut().drain(..) {
                self.queue_upload(&name, bytes);
            }
        });
        // dropped files (drag & drop onto the app — works native + wasm)
        for f in self.files.ctx_dropped.take().unwrap_or_default() {
            if let Some(bytes) = f.bytes.as_deref() {
                self.queue_upload(&f.name, bytes.to_vec());
            }
            #[cfg(not(target_arch = "wasm32"))]
            if f.bytes.is_none() {
                if let Some(p) = &f.path {
                    if let Ok(b) = std::fs::read(p) {
                        self.queue_upload(&f.name, b);
                    }
                }
            }
        }
    }

    fn files_window(&mut self, ctx: &egui::Context) {
        // stash dropped files where files_pump can see them each frame
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            self.files.ctx_dropped = Some(dropped);
        }
        self.files_pump();
        if !self.show_files {
            return;
        }
        if !self.files.opened {
            self.files.opened = true;
            self.files_refresh();
        }
        let mut open = self.show_files;
        let u = usable_rect(ctx);
        let dr = clamp_window_rect(
            egui::Rect::from_min_size(
                egui::pos2(u.center().x - 210.0, u.min.y + 24.0),
                egui::vec2(420.0, 420.0),
            ),
            u,
        );
        app_win(ctx, "Files")
            .open(&mut open)
            .default_size(dr.size())
            .default_pos(dr.min)
            .show(ctx, |ui| {
                // path bar: root label + up + refresh
                ui.horizontal(|ui| {
                    if self.files.local_root.is_some() {
                        ui.label("app data:");
                    } else {
                        ui.label("my files:");
                    }
                    ui.monospace(if self.files.cwd.is_empty() {
                        "/".to_string()
                    } else {
                        self.files.cwd.clone()
                    });
                    if ui.button("^").clicked() && !self.files.cwd.is_empty() {
                        let cwd = std::path::Path::new(&self.files.cwd);
                        self.files.cwd = cwd
                            .parent()
                            .map(|p| p.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        self.files_refresh();
                    }
                    if ui.button("⟳").clicked() {
                        self.files_refresh();
                    }
                });
                // usage bar
                if let Some((used, quota)) = self.files.usage {
                    let (txt, frac) = if quota > 0 {
                        (
                            format!("{} of {} used", fmt_size(used), fmt_size(quota)),
                            used as f32 / quota as f32,
                        )
                    } else {
                        (format!("{} used", fmt_size(used)), 0.0)
                    };
                    ui.add(
                        egui::ProgressBar::new(frac.clamp(0.0, 1.0))
                            .text(txt)
                            .desired_height(10.0),
                    );
                }
                ui.separator();
                // listing
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let entries = self.files.entries.clone();
                    for e in &entries {
                        let label = if e.is_dir {
                            format!("📁 {}", e.name)
                        } else {
                            format!(
                                "{}   {}   {}",
                                e.name,
                                fmt_size(e.size),
                                fmt_date(e.mtime_ms)
                            )
                        };
                        let sel = self.files.sel.as_deref() == Some(e.name.as_str());
                        let r = ui.selectable_label(sel, label);
                        if r.clicked() {
                            self.files.sel = Some(e.name.clone());
                            self.files.rename_to = e.name.clone();
                        }
                        if r.double_clicked() {
                            self.files_open(e);
                        }
                    }
                    if entries.is_empty() {
                        ui.weak("(empty — drop files here to upload)");
                    }
                });
                ui.separator();
                // actions on the selection
                let sel = self.files.sel.clone();
                ui.horizontal_wrapped(|ui| {
                    if ui.button("new folder").clicked() {
                        self.files.mkdir = "new-folder".into();
                    }
                    if !self.files.mkdir.is_empty() {
                        ui.text_edit_singleline(&mut self.files.mkdir);
                        if ui.button("create").clicked() {
                            self.file_mkdir();
                        }
                    }
                    if sel.is_some() && ui.button("rename…").clicked() {
                        // rename_to already seeded on select
                    }
                    if sel.is_some() && ui.button("delete…").clicked() {
                        self.files.confirm_delete = sel.clone();
                    }
                    if sel.is_some() && ui.button("download").clicked() {
                        self.file_download();
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("upload…").clicked() {
                        if let Some(paths) = rfd::FileDialog::new().pick_files() {
                            for p in paths {
                                if let Ok(b) = std::fs::read(&p) {
                                    let n = p
                                        .file_name()
                                        .map(|x| x.to_string_lossy().into_owned())
                                        .unwrap_or_else(|| "file".into());
                                    self.queue_upload(&n, b);
                                }
                            }
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    if ui.button("upload…").clicked() {
                        web_file_picker();
                    }
                });
                if self.files.sel.is_some() && !self.files.rename_to.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label("rename to");
                        ui.text_edit_singleline(&mut self.files.rename_to);
                        if ui.button("ok").clicked() {
                            self.file_rename();
                        }
                    });
                }
                if let Some(d) = self.files.confirm_delete.clone() {
                    ui.horizontal(|ui| {
                        ui.colored_label(egui::Color32::LIGHT_RED, format!("delete {d}?"));
                        if ui.button("yes, delete").clicked() {
                            self.file_delete(&d);
                            self.files.confirm_delete = None;
                        }
                        if ui.button("cancel").clicked() {
                            self.files.confirm_delete = None;
                        }
                    });
                }
                if let Some((p, data, off)) = &self.files.upload {
                    ui.weak(format!(
                        "uploading {p}: {}/{}",
                        fmt_size(*off as u64),
                        fmt_size(data.len() as u64)
                    ));
                }
                if !self.files.status.is_empty() {
                    ui.weak(&self.files.status);
                }
            });
        self.show_files = open;
    }

    /// Double-click open: dirs descend; `.ne2` loads; `.rhai` opens in
    /// the script editor; anything else just stays selected.
    fn files_open(&mut self, e: &proto::FileEntry) {
        if e.is_dir {
            self.files.cwd = if self.files.cwd.is_empty() {
                e.name.clone()
            } else {
                format!("{}/{}", self.files.cwd, e.name)
            };
            self.files_refresh();
            return;
        }
        let path = if self.files.cwd.is_empty() {
            e.name.clone()
        } else {
            format!("{}/{}", self.files.cwd, e.name)
        };
        if e.name.ends_with(".ne2") {
            match self.files.local_root.clone() {
                Some(root) => {
                    let p = root.join(&self.files.cwd).join(&e.name);
                    self.ctl.send(ClientMsg::LoadNet {
                        path: p.to_string_lossy().into_owned(),
                    });
                }
                None => self.ctl.send(ClientMsg::LoadNet { path }),
            }
            self.status = format!("loading {e}", e = e.name);
        } else if e.name.ends_with(".rhai") {
            match self.files.local_root.clone() {
                Some(root) => {
                    let p = root.join(&self.files.cwd).join(&e.name);
                    if let Ok(t) = std::fs::read_to_string(&p) {
                        self.script_file = p.to_string_lossy().into_owned();
                        self.script_text = t;
                    }
                }
                None => self.ctl.send(ClientMsg::ReadFile { path }),
            }
            self.show_script = true;
        }
    }

    fn file_mkdir(&mut self) {
        let name = self.files.mkdir.clone();
        if let Some(root) = self.files.local_root.clone() {
            let _ = std::fs::create_dir_all(root.join(&self.files.cwd).join(&name));
            self.files.mkdir.clear();
            self.files_refresh();
        } else {
            let path = if self.files.cwd.is_empty() {
                name.clone()
            } else {
                format!("{}/{name}", self.files.cwd)
            };
            self.ctl.send(ClientMsg::Mkdir { path });
            self.files.mkdir.clear();
        }
    }

    fn file_rename(&mut self) {
        let Some(sel) = self.files.sel.clone() else {
            return;
        };
        let to_name = self.files.rename_to.clone();
        if let Some(root) = self.files.local_root.clone() {
            let dir = root.join(&self.files.cwd);
            let _ = std::fs::rename(dir.join(&sel), dir.join(&to_name));
            self.files_refresh();
        } else {
            let (a, b) = if self.files.cwd.is_empty() {
                (sel, to_name)
            } else {
                (
                    format!("{}/{sel}", self.files.cwd),
                    format!("{}/{to_name}", self.files.cwd),
                )
            };
            self.ctl.send(ClientMsg::RenameFile { from: a, to: b });
        }
        self.files.sel = None;
    }

    fn file_delete(&mut self, name: &str) {
        if let Some(root) = self.files.local_root.clone() {
            let p = root.join(&self.files.cwd).join(name);
            if p.is_dir() {
                let _ = std::fs::remove_dir_all(&p);
            } else {
                let _ = std::fs::remove_file(&p);
            }
            self.files_refresh();
        } else {
            let path = if self.files.cwd.is_empty() {
                name.to_string()
            } else {
                format!("{}/{name}", self.files.cwd)
            };
            self.ctl.send(ClientMsg::DeleteFile { path });
        }
        self.files.sel = None;
    }

    fn file_download(&mut self) {
        let Some(sel) = self.files.sel.clone() else {
            return;
        };
        if let Some(root) = self.files.local_root.clone() {
            let p = root.join(&self.files.cwd).join(&sel);
            if let Ok(bytes) = std::fs::read(&p) {
                self.save_download(&sel, &bytes);
            }
        } else {
            let path = if self.files.cwd.is_empty() {
                sel.clone()
            } else {
                format!("{}/{sel}", self.files.cwd)
            };
            self.files.status = format!("downloading {path}…");
            self.ctl.send(ClientMsg::DownloadFile { path });
        }
    }

    // ------------------------------------------------------------------
    // Login dialog — `--connect http(s)://host` multi-user servers.
    // ------------------------------------------------------------------

    /// `--connect http(s)://…` entry point: reuse the persisted token
    /// when it matches this server (falling back to the dialog when it
    /// expired or never existed).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_saved_login(&mut self, url: &str) {
        if let Some((u, tok)) = self.saved_token.clone() {
            if u == url {
                match SimCtl::connect_secure(url, &tok) {
                    Ok(ctl) => {
                        self.ctl = ctl;
                        self.status = "connected".into();
                        return;
                    }
                    Err(_) => self.saved_token = None, // expired — re-login
                }
            }
        }
        self.begin_login(url, "");
    }

    /// Ask for credentials before the real transport exists; called by
    /// main's `--connect http(s)://` path after `EditorApp::new`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn begin_login(&mut self, url: &str, email: &str) {
        self.login = Some(LoginDlg {
            url: url.to_string(),
            email: email.to_string(),
            password: String::new(),
            remember: true,
            busy: false,
            error: String::new(),
            rx: None,
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn login_window(&mut self, ctx: &egui::Context) {
        let Some(lg) = &mut self.login else {
            return;
        };
        // result of the background attempt
        let mut done: Option<Result<(SimCtl, String), String>> = None;
        if let Some(rx) = &lg.rx {
            if let Ok(r) = rx.try_recv() {
                done = Some(r);
            }
        }
        match done {
            Some(Ok((ctl, token))) => {
                self.ctl = ctl;
                if lg.remember {
                    self.saved_token = Some((lg.url.clone(), token));
                }
                self.login = None;
                self.status = "connected".into();
                return;
            }
            Some(Err(e)) => {
                lg.busy = false;
                lg.error = e;
                lg.rx = None;
            }
            None => {}
        }
        let mut keep = true;
        app_win(ctx, format!("Log in — {}", lg.url))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label("email");
                ui.text_edit_singleline(&mut lg.email);
                ui.label("password");
                let r = ui.add(egui::TextEdit::singleline(&mut lg.password).password(true));
                ui.checkbox(&mut lg.remember, "remember me on this machine");
                if !lg.error.is_empty() {
                    ui.colored_label(egui::Color32::LIGHT_RED, &lg.error);
                }
                ui.add_enabled_ui(!lg.busy, |ui| {
                    if ui.button("Log in").clicked()
                        || (r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                    {
                        lg.busy = true;
                        lg.error.clear();
                        let (tx, rx) = std::sync::mpsc::channel();
                        let (url, email, pw) =
                            (lg.url.clone(), lg.email.clone(), lg.password.clone());
                        std::thread::spawn(move || {
                            let r = SimCtl::remote_login(&url, &email, &pw).and_then(|tok| {
                                SimCtl::connect_secure(&url, &tok).map(|c| (c, tok))
                            });
                            let _ = tx.send(r.map_err(|e| e.to_string()));
                        });
                        lg.rx = Some(rx);
                    }
                });
                if ui.button("Cancel").clicked() {
                    keep = false;
                }
            });
        if !keep {
            self.login = None;
        }
    }

    /// `?` — every mode key and mouse gesture in one small overlay.
    /// (WASD/QE pan the camera, so tools use other letters.)
    fn keys_window(&mut self, ctx: &egui::Context) {
        if !self.show_keys {
            return;
        }
        let mut open = self.show_keys;
        app_win(ctx, "Shortcuts")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                egui::Grid::new("keys_grid").num_columns(2).show(ui, |ui| {
                    let row = |ui: &mut egui::Ui, k: &str, what: &str| {
                        ui.monospace(k);
                        ui.label(what);
                        ui.end_row();
                    };
                    ui.strong("modes");
                    ui.end_row();
                    row(ui, "V / Esc", "navigate");
                    row(ui, "I", "insert neuron");
                    row(ui, "C", "connect");
                    row(ui, "M", "move");
                    row(ui, "Esc", "cancel drag/connect, else navigate");
                    ui.end_row();
                    ui.strong("camera");
                    ui.end_row();
                    row(ui, "right-drag", "orbit");
                    row(ui, "wheel", "zoom");
                    row(ui, "W A S D / Q E", "pan");
                    row(ui, "double-click", "focus");
                    row(ui, "F", "frame network");
                    ui.end_row();
                    ui.strong("mouse (navigate)");
                    ui.end_row();
                    row(ui, "click", "select (Shift adds)");
                    row(ui, "drag on object", "move it");
                    row(ui, "drag on empty", "box select");
                    row(ui, "right-click", "context menu");
                    ui.end_row();
                    ui.strong("while moving");
                    ui.end_row();
                    row(ui, "Shift (hold)", "vertical camera plane");
                    row(ui, "X / Y / Z", "axis lock (repeat releases)");
                    ui.end_row();
                    ui.strong("edit");
                    ui.end_row();
                    row(ui, "Ctrl+C / X / V", "copy / cut / paste");
                    row(ui, "Del", "delete selection");
                    ui.end_row();
                    ui.strong("windows");
                    ui.end_row();
                    row(
                        ui,
                        "Ctrl+1..6",
                        "inspector · plots · console · script · inputs · files",
                    );
                    row(ui, "?", "manual");
                });
            });
        self.show_keys = open;
    }

    /// Manual window — `docs/manual/*.md` baked into the binary.
    fn manual_window(&mut self, ctx: &egui::Context) {
        if !self.show_manual {
            return;
        }
        let mut open = self.show_manual;
        // Stage 15 A: narrow viewports get a shrunken manual — cap the
        // default at 70% of the usable strip instead of covering it.
        let u = usable_rect(ctx);
        let dr = clamp_window_rect(
            egui::Rect::from_min_size(
                u.min + egui::vec2(24.0, 8.0),
                egui::vec2(640.0, 520.0).min(u.size() * 0.7),
            ),
            u,
        );
        app_win(ctx, "Manual")
            .open(&mut open)
            .default_size(dr.size())
            .default_pos(dr.min)
            .show(ctx, |ui| {
                // horizontal children get an unbounded max_rect in x, so the
                // body column must be ALLOCATED an explicit rect — otherwise
                // ScrollArea grabs the infinite row width, the window grows
                // every frame, and wrapped labels still never wrap.
                let body_w = (ui.clip_rect().width() - 110.0 - 40.0).clamp(80.0, 480.0);
                let body_h = ui.available_height().max(60.0);
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.set_width(110.0);
                        for (i, p) in crate::manual::PAGES.iter().enumerate() {
                            ui.selectable_value(&mut self.manual_page, i, p.title);
                        }
                    });
                    ui.separator();
                    // allocate_ui_with_layout (not allocate_ui): plain
                    // allocate_ui inherits this row's horizontal layout and
                    // the scroll column would lay its widgets out sideways.
                    ui.allocate_ui_with_layout(
                        egui::vec2(body_w, body_h),
                        egui::Layout::top_down(egui::Align::LEFT),
                        |ui| {
                            crate::manual::body_scroll(ui, body_w, self.manual_page);
                        },
                    );
                });
            });
        self.show_manual = open;
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
                ui.label(format!(
                    "kind: {}",
                    if g.kind[i] == 0 { "exc" } else { "inh" }
                ));
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

    fn console(&mut self, ctx: &egui::Context) {
        if !self.show_console {
            return;
        }
        let mut open = self.show_console;
        let u = usable_rect(ctx);
        let dr = clamp_window_rect(
            egui::Rect::from_min_size(
                egui::pos2(u.max.x - 430.0, u.max.y - 260.0),
                egui::vec2(420.0, 240.0),
            ),
            u,
        );
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
        let dr = clamp_window_rect(
            egui::Rect::from_min_size(
                egui::pos2(u.min.x + 12.0, u.max.y - 240.0),
                egui::vec2(480.0, 220.0),
            ),
            u,
        );
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
                        ui.label(format!("watching {}", self.watch_sent.len()));
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
                        let mut p = egui_plot::Plot::new("vplot").height(140.0);
                        if win.is_none() || auto_y {
                            // x (or y too when following) from the data.
                            p = p.auto_bounds(egui::Vec2b::new(win.is_none(), auto_y));
                        } else {
                            p = p.auto_bounds(egui::Vec2b::new(false, false));
                        }
                        let pr = p.show(ui, |ui| {
                            if self.hist.v.is_empty() {
                                // Discoverable watching (Stage 13 E): the
                                // empty plot itself explains how to fill it.
                                let cx = win.map(|(lo, hi)| (lo + hi) / 2.0)
                                    .unwrap_or(self.last_sim_t);
                                ui.text(egui_plot::Text::new(
                                    "hint",
                                    egui_plot::PlotPoint::new(cx, -25.0),
                                    "no watched neurons — click a neuron to watch it,\n\
                                     or right-click -> \"Watch v(t)\" · Inspector -> \"Plot v(t)\"",
                                ));
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
        // keep the list fresh while the tab is open
        let now = ui.ctx().input(|i| i.time);
        if now - self.last_jobs_poll > 2.0 {
            self.last_jobs_poll = now;
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
                            ui.add(egui::DragValue::new(w).prefix("w "));
                            ui.add(egui::DragValue::new(h).prefix("h "));
                            ui.add(egui::DragValue::new(dist).prefix("dist "));
                        });
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
                            ui.add(egui::DragValue::new(w).prefix("w "));
                            ui.add(egui::DragValue::new(h).prefix("h "));
                            ui.add(egui::DragValue::new(d2).prefix("d "));
                            ui.add(egui::DragValue::new(spacing).prefix("spacing "));
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
                        self.send(ClientMsg::Eval {
                            code: format!("net.save({p:?})"),
                        });
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
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        crate::icons::Icons::ensure(&mut self.icons, ctx);
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
        self.hints_bar(ctx);
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
        self.net_touched = false;
    }
}

/// The rect floating windows may occupy: below the toolbar, above the
/// status + hint strips (Stage 15 A/C). Only valid after those panels
/// have been shown this frame — update() calls windows last.
fn usable_rect(ctx: &egui::Context) -> egui::Rect {
    ctx.available_rect()
}

/// Clamp `want` fully inside `usable`, shrinking if too big. Used for
/// window *defaults* so first-run/narrow viewports start on-screen;
/// `constrain_to` then keeps dragged/resized windows inside too.
fn clamp_window_rect(want: egui::Rect, usable: egui::Rect) -> egui::Rect {
    let size = want.size().min(usable.size());
    let min = egui::pos2(
        want.min
            .x
            .clamp(usable.min.x, (usable.max.x - size.x).max(usable.min.x)),
        want.min
            .y
            .clamp(usable.min.y, (usable.max.y - size.y).max(usable.min.y)),
    );
    egui::Rect::from_min_size(min, size)
}

/// One toolbar control group laid out atomically (Stage 15 B). Inside a
/// `horizontal_wrapped` row, plain `ui.horizontal` claims the whole
/// remaining row as its desired size, so egui's wrap check
/// (`available < desired`) never fires and groups split mid-control.
/// Claiming an explicit upper-bound width instead makes the wrap check
/// see the group's real footprint — it moves to the next row whole.
/// `max_w` must overestimate, never undershoot: overshoot only packs a
/// little early; undershoot lets content overflow the row edge.
fn toolbar_group(
    ui: &mut egui::Ui,
    max_w: f32,
    add: impl FnOnce(&mut egui::Ui),
) -> egui::InnerResponse<()> {
    ui.allocate_ui_with_layout(
        egui::vec2(max_w, ui.spacing().interact_size.y),
        egui::Layout::left_to_right(egui::Align::Center),
        add,
    )
}

/// A floating window pinned to [`usable_rect`]. egui re-applies the
/// constraint every frame, so persisted off-screen positions and narrow
/// resizes get pulled back on-screen automatically.
fn app_win<'open>(ctx: &egui::Context, title: impl Into<egui::WidgetText>) -> egui::Window<'open> {
    egui::Window::new(title).constrain_to(usable_rect(ctx))
}

/// Parent dir of a path for a `ListFiles` request ("." at the top).
fn dir_of(p: &str) -> String {
    match p.rsplit_once('/') {
        Some(("", _)) => "/".into(),
        Some((d, _)) => d.to_string(),
        None => ".".into(),
    }
}

/// Example auto-watch gate (Stage 13 H): the Geo that may apply the
/// example's `watch` list must be newer than the version captured when
/// Run was sent (`base`) and contain every requested neuron id. The
/// pre-run net can already cover the ids — watching it just gets wiped
/// by the eval's `net.clear` → reset_net_state.
fn example_geo_covers(base: Option<u64>, geo: Option<&GeoSnap>, ids: &[u32]) -> bool {
    match (base, geo) {
        (Some(base), Some(g)) => {
            g.version != base && ids.iter().all(|&n| (n as usize) < g.pos.len())
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_clamp_keeps_rect_inside_usable() {
        let usable = egui::Rect::from_min_max(egui::pos2(0.0, 60.0), egui::pos2(800.0, 640.0));
        // Already inside → untouched.
        let inside = egui::Rect::from_min_size(egui::pos2(100.0, 100.0), egui::vec2(300.0, 200.0));
        assert_eq!(clamp_window_rect(inside, usable), inside);
        // Hanging off the right → pulled left so it fits.
        let off_right =
            egui::Rect::from_min_size(egui::pos2(700.0, 100.0), egui::vec2(300.0, 200.0));
        let got = clamp_window_rect(off_right, usable);
        assert!(got.max.x <= usable.max.x && got.min.x >= usable.min.x);
        // Hanging off the bottom → lifted above the usable strip.
        let off_bot = egui::Rect::from_min_size(egui::pos2(100.0, 600.0), egui::vec2(300.0, 200.0));
        let got = clamp_window_rect(off_bot, usable);
        assert!(got.max.y <= usable.max.y);
        // Off the left/top → pushed back in.
        let off_tl = egui::Rect::from_min_size(egui::pos2(-50.0, 10.0), egui::vec2(100.0, 100.0));
        let got = clamp_window_rect(off_tl, usable);
        assert!(got.min.x >= usable.min.x && got.min.y >= usable.min.y);
        // Bigger than the viewport → shrunk to fit, pinned top-left.
        let huge = egui::Rect::from_min_size(egui::pos2(50.0, 70.0), egui::vec2(2000.0, 2000.0));
        let got = clamp_window_rect(huge, usable);
        assert_eq!(got.size(), usable.size());
        assert_eq!(got.min, usable.min);
    }

    /// Stage 15 F: every app-state input to `render_offscreen` must move
    /// the SceneKey — driven through real EditorApp fields, not literals.
    #[test]
    fn scene_key_changes_with_app_state() {
        let app = EditorApp::new(crate::simctl::SimCtl::offline(), None);
        let sel: Vec<u32> = vec![3, 7];
        let act: Vec<f32> = vec![0.0; 8];
        let pvs: Vec<crate::gpu::Pv> = Vec::new();
        let parts: Vec<crate::gpu::Part> = Vec::new();
        let no_planes: Option<Vec<bool>> = None;
        let key = |a: &EditorApp| {
            crate::view::scene_key(
                a, 800, 600, 1.33, &sel, &act, &pvs, &parts, true, &no_planes,
            )
        };
        let base = key(&app);
        for f in [
            (|a: &mut EditorApp| a.show_grid = false) as fn(&mut EditorApp),
            |a| a.hover = Some(4),
            |a| a.preview_delta = Some(glam::Vec3::new(1.0, 0.0, 0.0)),
            |a| a.plane_key = 9,
            |a| a.geo_version = 2,
            |a| a.camera.orbit(10.0, 0.0),
        ] {
            let mut a2 = EditorApp::new(crate::simctl::SimCtl::offline(), None);
            f(&mut a2);
            assert_ne!(key(&a2), base, "SceneKey ignored an app-state input");
        }
        // Buffer/flag inputs (not on the app, but rendered too).
        assert_ne!(
            crate::view::scene_key(
                &app,
                800,
                600,
                1.33,
                &[1],
                &act,
                &pvs,
                &parts,
                true,
                &no_planes
            ),
            base
        );
        assert_ne!(
            crate::view::scene_key(
                &app, 801, 600, 1.33, &sel, &act, &pvs, &parts, true, &no_planes
            ),
            base
        );
        assert_ne!(
            crate::view::scene_key(
                &app, 800, 600, 1.33, &sel, &act, &pvs, &parts, false, &no_planes
            ),
            base
        );
        assert_ne!(
            crate::view::scene_key(
                &app,
                800,
                600,
                1.33,
                &sel,
                &act,
                &pvs,
                &parts,
                true,
                &Some(vec![true])
            ),
            base
        );
    }

    /// Stage 15 B: `toolbar_group` units inside `horizontal_wrapped` wrap
    /// whole — a group never splits its icon/control across two rows, and
    /// a group that doesn't fit moves to the next row instead of
    /// overflowing the panel edge.
    #[test]
    fn wrapped_toolbar_groups_never_split() {
        let ctx = egui::Context::default();
        let mut same_row = true;
        let mut fits = true;
        let mut h = 0.0f32;
        for _ in 0..3 {
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::pos2(0.0, 0.0),
                        egui::vec2(300.0, 500.0),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    egui::TopBottomPanel::top("t").show(ctx, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            let max_x = ui.max_rect().max.x;
                            for g in 0..4 {
                                let r = toolbar_group(ui, 260.0, |ui| {
                                    let a = ui.label(format!("group-{g}")).rect;
                                    let b = ui.add(egui::Slider::new(&mut 0.5f32, 0.0..=1.0)).rect;
                                    same_row &= a.center().y == b.center().y;
                                })
                                .response
                                .rect;
                                fits &= r.max.x <= max_x + 0.5;
                            }
                        });
                        h = ui.min_rect().height();
                    });
                },
            );
        }
        assert!(same_row, "a control group split across wrap rows");
        assert!(fits, "a group overflowed the panel edge");
        assert!(h > 40.0, "groups did not wrap (panel {h}px)");
    }

    #[test]
    fn example_watch_needs_post_run_geo() {
        let mut g = GeoSnap::default();
        g.version = 7;
        g.pos = vec![[0.0; 3]; 5];
        let ids = [0u32, 1, 2, 3, 4];
        // Not armed yet (loaded but never run) — pre-run geo must not apply.
        assert!(!example_geo_covers(None, Some(&g), &ids));
        // Same version as at Run — the eval hasn't landed yet.
        assert!(!example_geo_covers(Some(7), Some(&g), &ids));
        g.version = 8;
        assert!(example_geo_covers(Some(7), Some(&g), &ids));
        // Post-run geo that doesn't cover the ids — e.g. cleared but
        // not yet rebuilt — must not apply either.
        g.pos.truncate(3);
        assert!(!example_geo_covers(Some(7), Some(&g), &ids));
        assert!(!example_geo_covers(Some(7), None, &ids));
    }
}
