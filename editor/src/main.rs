//! neuro_edit v2 editor — egui/wgpu 3D editor + sim worker + Rhai console.
//!
//! GUI:      `editor [--net file.ne2] [--cpu|--cuda]`
//! Headless: `editor --screenshot out.png [--net file.ne2] [--script x.rhai]
//!           [--after-ms 500] [--size 1280x800]`

use clap::Parser;
use eframe::egui;
use sim::BackendKind;

#[derive(Parser)]
#[command(name = "editor", about = "neuro_edit v2 editor")]
struct Cli {
    /// Network file to open (.ne2).
    #[arg(long)]
    net: Option<String>,
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

fn backend_of(cli: &Cli) -> BackendKind {
    if cli.cpu {
        BackendKind::Cpu
    } else {
        BackendKind::Cuda // default; --cpu overrides
    }
}

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
    let net = match &cli.net {
        Some(p) => net_format::io::load(p).unwrap_or_else(|e| {
            eprintln!("load {p}: {e} — starting empty");
            net_format::Network::new()
        }),
        None => net_format::Network::new(),
    };
    let ctl = editor::simctl::SimCtl::start_cfg(
        net,
        backend,
        cli.seed,
        sim::SimConfig {
            int_weights: cli.int_weights,
            ..Default::default()
        },
    );
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
            .with_title("neuro_edit v2")
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
