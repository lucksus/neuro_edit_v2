//! neuro_edit v2 editor — egui/wgpu 3D editor + sim worker + Rhai console.
//!
//! GUI:      `editor [--net file.ne2] [--cpu|--cuda]`
//! Headless: `editor --screenshot out.png [--net file.ne2] [--script x.rhai]
//!           [--after-ms 500] [--size 1280x800]`

mod app;
mod camera;
mod gpu;
mod scene;
mod shot;
mod simctl;
mod snap;
mod view;

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
        let opts = shot::ShotOpts {
            net_path: cli.net.clone(),
            script: cli.script.clone(),
            after_ms: cli.after_ms,
            width: cli.width,
            height: cli.height,
            out: out.clone(),
            backend,
            seed: cli.seed,
            show_syn: !cli.no_synapses,
        };
        return pollster::block_on(shot::render_headless(opts));
    }

    // GUI mode
    let net = match &cli.net {
        Some(p) => net_format::io::load(p).unwrap_or_else(|e| {
            eprintln!("load {p}: {e} — starting empty");
            net_format::Network::new()
        }),
        None => net_format::Network::new(),
    };
    let ctl = simctl::SimCtl::start(net, backend, cli.seed);
    if let Some(s) = &cli.script {
        let code = std::fs::read_to_string(s)?;
        ctl.send(simctl::Cmd::Eval(code));
    }
    let net_path = cli.net.clone();

    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_title("neuro_edit v2"),
        ..Default::default()
    };
    eframe::run_native(
        "neuro_edit v2",
        native,
        Box::new(move |cc| {
            let mut app = app::EditorApp::new(ctl);
            app.net_path = net_path;
            // dark theme
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))
}
