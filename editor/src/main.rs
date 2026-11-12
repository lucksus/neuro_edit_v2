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
        };
        return pollster::block_on(editor::shot::render_headless(opts));
    }

    // GUI mode
    let ctl = match &cli.connect {
        Some(addr) => {
            if cli.net.is_some() {
                eprintln!("--net ignored with --connect (the server owns the network)");
            }
            if cli.cpu || cli.cuda || cli.int_weights {
                eprintln!("--cpu/--cuda/--int-weights ignored with --connect (server's backend)");
            }
            editor::simctl::SimCtl::connect(addr)?
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
    let net_path = cli.net.clone();

    let icon_img = image::load_from_memory(include_bytes!("../assets/icon_256.png"))
        .expect("icon")
        .to_rgba8();
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_title(match &cli.connect {
                Some(a) => format!("neuro_edit v2 — {a}"),
                None => "neuro_edit v2".to_string(),
            })
            .with_icon(egui::IconData {
                rgba: icon_img.to_vec(),
                width: icon_img.width(),
                height: icon_img.height(),
            }),
        ..Default::default()
    };
    eframe::run_native(
        "neuro_edit v2",
        native,
        Box::new(move |cc| {
            let mut app = editor::app::EditorApp::new(ctl, cc.storage);
            app.net_path = net_path.clone();
            if let Some(p) = &net_path {
                app.push_recent(p.clone());
            }
            // dark theme
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
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

    let mut opts = eframe::WebOptions::default();
    opts.wgpu_options.wgpu_setup =
        eframe::egui_wgpu::WgpuSetup::CreateNew(eframe::egui_wgpu::WgpuSetupCreateNew {
            instance_descriptor: wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU,
                ..Default::default()
            },
            ..Default::default()
        });

    wasm_bindgen_futures::spawn_local(async move {
        let runner = eframe::WebRunner::new();
        let r = runner
            .start(
                canvas,
                opts,
                Box::new(move |cc| {
                    let ctl = editor::simctl::SimCtl::connect(&host)
                        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
                    cc.egui_ctx.set_visuals(eframe::egui::Visuals::dark());
                    Ok(Box::new(editor::app::EditorApp::new(ctl, cc.storage)))
                }),
            )
            .await;
        if let Err(e) = r {
            // index.html's boot() already showed WebGPU help when the
            // adapter check failed; a failure HERE means an adapter was
            // offered but the device request died — render the same
            // per-browser help via window.neWebgpuHelp.show(), keeping
            // the raw JsValue in the details line.
            let msg = format!("Adapter/device request failed: {e:?}");
            if let Some(w) = web_sys::window() {
                let shown = js_sys::Reflect::get(&w, &"neWebgpuHelp".into())
                    .ok()
                    .and_then(|o| js_sys::Reflect::get(&o, &"show".into()).ok())
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                    .map(|f| f.call1(&w, &msg.clone().into()));
                if shown.is_none() {
                    // help module not on the page (old bundle): last
                    // resort — the raw message on a dark page
                    let full = format!(
                        "neuro_edit v2 needs WebGPU (the 3D view uses storage \
                         buffers). {msg}"
                    );
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
