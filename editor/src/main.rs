//! neuro_edit v2 editor — egui/wgpu 3D editor + sim worker + Rhai console.
//!
//! GUI:      `editor [--net file.ne2] [--cpu|--cuda]`
//! Remote:   `editor --connect host:port`
//! Headless: `editor --screenshot out.png [--net file.ne2] [--script x.rhai]
//!           [--after-ms 500] [--size 1280x800]`
//! Browser:  `cargo build --target wasm32-unknown-unknown -p editor
//!           --no-default-features --features web` + wasm-bindgen, served
//!           by `server --web` — connects back to the serving host.

#[cfg(feature = "native")]
use clap::Parser;
#[cfg(feature = "native")]
use eframe::egui;
#[cfg(feature = "native")]
use sim::BackendKind;

#[cfg(feature = "native")]
#[derive(Parser)]
#[command(name = "editor", about = "neuro_edit v2 editor")]
struct Cli {
    /// Network file to open (.ne2).
    #[arg(long)]
    net: Option<String>,
    /// Attach to a running `server` over WebSocket instead of a local
    /// worker. `host:port`; scripts and files then run server-side.
    #[arg(long)]
    connect: Option<String>,
    /// Rhai script to run on the worker at startup.
    #[arg(long)]
    script: Option<String>,
    /// Headless screenshot: render the 3D scene to this PNG and exit.
    #[arg(long)]
    screenshot: Option<String>,
    /// Sim ms to run before the screenshot.
    #[arg(long, default_value_t = 0.0)]
    after_ms: f32,
    /// Screenshot width px.
    #[arg(long, default_value_t = 1280)]
    width: u32,
    /// Screenshot height px.
    #[arg(long, default_value_t = 800)]
    height: u32,
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    cuda: bool,
    /// Hide synapse lines in the screenshot.
    #[arg(long)]
    no_synapses: bool,
    /// CUDA: fixed-point synaptic ring → bitwise reproducible runs.
    #[arg(long)]
    int_weights: bool,
    /// Screenshot camera overrides (applied after auto-framing).
    #[arg(long)]
    cam_dist: Option<f32>,
    /// Screenshot camera yaw in degrees.
    #[arg(long)]
    cam_yaw: Option<f32>,
    /// Screenshot camera pitch in degrees.
    #[arg(long)]
    cam_pitch: Option<f32>,
    /// Render grid groups as activity planes in the screenshot.
    #[arg(long)]
    planes: bool,
    /// Screenshot background-noise kick rate (Hz).
    #[arg(long, default_value_t = 0.0)]
    noise_hz: f32,
    /// Screenshot background-noise kick amplitude (mV).
    #[arg(long, default_value_t = 0.0)]
    noise_kick: f32,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Review/screenshot driver: auto-select+watch these neuron ids once
    /// the script's Geo arrives (comma-separated, e.g. --watch 200).
    #[arg(long)]
    watch: Option<String>,
    /// Review/screenshot driver: start auto-run after the startup script
    /// (the sim keeps stepping — traces accumulate for shots).
    #[arg(long)]
    autorun: bool,
    /// Stage 23 review driver: set a group's mode `name:whole|inside|block`
    /// (repeatable, applied each frame).
    #[arg(long)]
    group_mode: Vec<String>,
    /// Stage 23 review driver: select a named group's members once Geo
    /// names them (whole-group selection shot).
    #[arg(long)]
    group_sel: Option<String>,
    /// Stage 23 review driver: select these instance ids once Geo covers
    /// them (inside-mode member shot).
    #[arg(long)]
    sel: Option<String>,
    /// Stage 23 review driver: keep this menubar menu open (e.g. "Edit").
    #[arg(long)]
    open_menu: Option<String>,
    /// Stage 23 review driver: after --group-sel/--sel lands, move the
    /// selection by x,y,z (gives the Edit menu a real undo label).
    #[arg(long)]
    edit_move: Option<String>,
    /// GUI window size WxH (default 1440x900) — deterministic size for
    /// review screenshots.
    #[arg(long)]
    window_size: Option<String>,
}

#[cfg(feature = "native")]
fn backend_of(cli: &Cli) -> BackendKind {
    if cli.cpu {
        BackendKind::Cpu
    } else {
        BackendKind::Cuda // default; --cpu overrides
    }
}

#[cfg(feature = "native")]
fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let backend = backend_of(&cli);

    if let Some(out) = &cli.screenshot {
        let opts = editor::shot::ShotOpts {
            net_path: cli.net.clone(),
            script: cli.script.clone(),
            after_ms: cli.after_ms,
            width: cli.width,
            height: cli.height,
            out: out.clone(),
            backend,
            seed: cli.seed,
            show_syn: !cli.no_synapses,
            int_weights: cli.int_weights,
            cam_dist: cli.cam_dist,
            cam_yaw: cli.cam_yaw,
            cam_pitch: cli.cam_pitch,
            grid_planes: cli.planes,
            plane_cfgs: Default::default(),
            noise_hz: cli.noise_hz,
            noise_kick: cli.noise_kick,
            group_sel: Default::default(),
        };
        return pollster::block_on(editor::shot::render_headless(opts));
    }

    // GUI mode. `--connect http(s)://…` is a multi-user server: the app
    // starts on a placeholder transport and logs in (remembered token or
    // the dialog); `host:port` is the unauthenticated wire as before.
    let mut login_url: Option<String> = None;
    let ctl = match &cli.connect {
        Some(addr) => {
            if cli.net.is_some() {
                eprintln!("--net ignored with --connect (the server owns the network)");
            }
            if cli.cpu || cli.cuda || cli.int_weights {
                eprintln!("--cpu/--cuda/--int-weights ignored with --connect (server's backend)");
            }
            if addr.starts_with("http://") || addr.starts_with("https://") {
                login_url = Some(addr.clone());
                editor::simctl::SimCtl::offline()
            } else {
                editor::simctl::SimCtl::connect(addr)?
            }
        }
        None => {
            let net = match &cli.net {
                Some(p) => net_format::io::load(p).unwrap_or_else(|e| {
                    eprintln!("load {p}: {e} — starting empty");
                    net_format::Network::new()
                }),
                None => net_format::Network::new(),
            };
            editor::simctl::SimCtl::start_cfg(
                net,
                backend,
                cli.seed,
                sim::SimConfig {
                    int_weights: cli.int_weights,
                    ..Default::default()
                },
            )
        }
    };
    if let Some(s) = &cli.script {
        let code = std::fs::read_to_string(s)?;
        ctl.send(proto::ClientMsg::Eval { code });
    }
    if cli.autorun {
        ctl.send(proto::ClientMsg::AutoRun(true));
    }
    // --watch ids → the same path a Run click on an example takes:
    // selection+watch applies when the eval's Geo covers the ids.
    let shot_watch: Option<&'static [u32]> = cli.watch.as_deref().map(|s| {
        let ids: Vec<u32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        &*Box::leak(ids.into_boxed_slice())
    });
    let shot_sel: Option<Vec<u32>> = cli
        .sel
        .as_deref()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect());
    let shot_group_modes = cli.group_mode.clone();
    let shot_group_sel = cli.group_sel.clone();
    let open_menu = cli.open_menu.clone();
    let shot_move: Option<glam::Vec3> = cli.edit_move.as_deref().map(|s| {
        let v: Vec<f32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        glam::vec3(
            v.first().copied().unwrap_or(0.0),
            v.get(1).copied().unwrap_or(0.0),
            v.get(2).copied().unwrap_or(0.0),
        )
    });
    let (win_w, win_h) = cli
        .window_size
        .as_deref()
        .and_then(|s| s.split_once('x'))
        .and_then(|(w, h)| w.parse::<f32>().ok().zip(h.parse::<f32>().ok()))
        .unwrap_or((1440.0, 900.0));
    let net_path = cli.net.clone();

    let icon_img = image::load_from_memory(include_bytes!("../assets/icon_256.png"))
        .expect("icon")
        .to_rgba8();
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([win_w, win_h])
            .with_title(match &cli.connect {
                Some(a) => format!("neuro_edit v2 — {a}"),
                None => "neuro_edit v2".to_string(),
            })
            .with_icon(egui::IconData {
                rgba: icon_img.to_vec(),
                width: icon_img.width(),
                height: icon_img.height(),
            }),
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            // Cap the GPU queue at one frame: drags read back the cursor
            // position a frame sooner (Stage 14 D).
            desired_maximum_frame_latency: Some(1),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "neuro_edit v2",
        native,
        Box::new(move |cc| {
            let mut app = editor::app::EditorApp::new(ctl, cc.storage);
            if let Some(url) = &login_url {
                app.try_saved_login(url);
            }
            app.net_path = net_path.clone();
            if let Some(p) = &net_path {
                app.push_recent(p.clone());
            }
            if let Some(ids) = shot_watch {
                app.arm_example_watch(ids);
            }
            app.shot_group_modes = shot_group_modes.clone();
            app.shot_group_sel = shot_group_sel.clone();
            app.shot_sel = shot_sel.clone();
            app.open_menu = open_menu.clone();
            app.shot_move = shot_move;
            // dark theme — set_theme, not set_visuals: eframe applies the
            // system light scheme AFTER the app creator, which would undo
            // a bare set_visuals(dark) on light desktops/browser OS themes
            cc.egui_ctx.set_theme(egui::Theme::Dark);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))
}

/// Stub for `cargo check -p editor --no-default-features` on native —
/// the real entries are `native` (local worker + CLI) or `web` (wasm).
#[cfg(all(not(feature = "native"), not(target_arch = "wasm32")))]
fn main() {
    eprintln!("editor built without the `native` feature — nothing to run");
}

/// Browser entry: the page is served by `server --web`; the editor
/// connects its wire transport back to the serving host. WebGPU only —
/// the scene pipelines need storage buffers, so a WebGL2 fallback would
/// fail later anyway; when no adapter exists we say so on the page.
#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast;

    let window = web_sys::window().expect("window");
    let document = window.document().expect("document");
    let canvas = document
        .get_element_by_id("ne_canvas")
        .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        .expect("<canvas id=ne_canvas>");
    let host = window
        .location()
        .host()
        .unwrap_or_else(|_| "127.0.0.1:3090".into());
    // https pages (the multi-user server) must speak wss and reach the
    // /ws endpoint; plain-http `--web` keeps ws://host (any path works —
    // the shared acceptor only checks Upgrade).
    let ws_url = if window.location().protocol().ok().as_deref() == Some("https:") {
        format!("wss://{host}/ws")
    } else {
        format!("ws://{host}/ws")
    };

    let mut opts = eframe::WebOptions::default();
    opts.wgpu_options.wgpu_setup =
        eframe::egui_wgpu::WgpuSetup::CreateNew(eframe::egui_wgpu::WgpuSetupCreateNew {
            instance_descriptor: wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU,
                ..Default::default()
            },
            ..Default::default()
        });
    // Same latency cap as native — harmless where unsupported (Stage 14 D).
    opts.wgpu_options.desired_maximum_frame_latency = Some(1);

    // Set when the app creator fails on the socket, so the error below
    // gets the server page instead of the WebGPU one (a mixed-content
    // `ws://` refusal used to read as "device request failed").
    let connect_failed = std::rc::Rc::new(std::cell::Cell::new(false));
    let connect_failed_in = connect_failed.clone();

    wasm_bindgen_futures::spawn_local(async move {
        let runner = eframe::WebRunner::new();
        let r = runner
            .start(
                canvas,
                opts,
                Box::new(move |cc| {
                    let ctl = editor::simctl::SimCtl::connect(&ws_url).map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                            connect_failed_in.set(true);
                            e.into()
                        },
                    )?;
                    cc.egui_ctx.set_theme(eframe::egui::Theme::Dark);
                    Ok(Box::new(editor::app::EditorApp::new(ctl, cc.storage)))
                }),
            )
            .await;
        if let Err(e) = r {
            // index.html's boot() already showed WebGPU help when the
            // adapter check failed; a failure HERE is either the socket
            // (connect_failed) or an adapter that was offered but whose
            // device request died. Render the matching page via
            // window.neWebgpuHelp, keeping the raw JsValue in the
            // details line.
            let (msg, show, fallback) = if connect_failed.get() {
                (
                    format!("Connection to the server failed: {e:?}"),
                    "showConnect",
                    "neuro_edit v2 could not connect to the server.",
                )
            } else {
                (
                    format!("Adapter/device request failed: {e:?}"),
                    "show",
                    "neuro_edit v2 needs WebGPU (the 3D view uses storage buffers).",
                )
            };
            if let Some(w) = web_sys::window() {
                let shown = js_sys::Reflect::get(&w, &"neWebgpuHelp".into())
                    .ok()
                    .and_then(|o| js_sys::Reflect::get(&o, &show.into()).ok())
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                    .map(|f| f.call1(&w, &msg.clone().into()));
                if shown.is_none() {
                    // help module not on the page (old bundle): last
                    // resort — the raw message on a dark page
                    let full = format!("{fallback} {msg}");
                    if let Some(b) = w.document().and_then(|d| d.body()) {
                        b.set_inner_html(&format!(
                            "<div style='color:#ddd;background:#181818;\
                             font:15px/1.6 monospace;padding:3em'>{full}</div>"
                        ));
                    }
                }
            }
        }
    });
}
