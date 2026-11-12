//! neuro_edit server: the sim worker headless, serving the wire protocol
//! over WebSocket. Editors (`editor --connect`) and later browser UIs
//! attach to one shared run.
//!
//!   server [--bind 127.0.0.1:9337] [--data-dir DIR] [--net file.ne2]
//!          [--cpu|--cuda] [--seed N] [--script x.rhai]

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use sim::BackendKind;

#[derive(Parser)]
#[command(name = "server", about = "neuro_edit sim server (WebSocket)")]
struct Cli {
    /// Address to bind. localhost only by default — pass e.g.
    /// `0.0.0.0:9337` to expose the sim to the LAN (no auth!).
    #[arg(long, default_value = "127.0.0.1:9337")]
    bind: String,
    /// Confine every file path scripts/clients touch to this directory
    /// (.. and absolute paths are refused). REQUIRED for remote serving.
    #[arg(long)]
    data_dir: Option<std::path::PathBuf>,
    /// Network file to start with (path resolved in --data-dir when set).
    #[arg(long)]
    net: Option<String>,
    /// Rhai script to run at startup.
    #[arg(long)]
    script: Option<String>,
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    cuda: bool,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Fixed-point synaptic ring (CUDA): bitwise reproducible runs.
    #[arg(long)]
    int_weights: bool,
    /// Also serve the browser editor (index.html + wasm bundle) from
    /// this directory on the same port (`server --web editor/dist-web`).
    #[arg(long)]
    web: Option<std::path::PathBuf>,
    /// Cap on `UploadDataset` signal bytes (MiB). Guards the shared
    /// data_dir against huge pushes.
    #[arg(long, default_value_t = 64)]
    max_upload_mb: usize,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let backend = if cli.cpu {
        BackendKind::Cpu
    } else {
        BackendKind::Cuda
    };

    // A remote-reachable server must not hand the filesystem to clients.
    if cli.bind.starts_with("0.0.0.0") && cli.data_dir.is_none() {
        anyhow::bail!("--bind 0.0.0.0 without --data-dir would expose arbitrary paths — refusing");
    }

    // Resolve the start net through the same sandbox rules.
    let net = match (&cli.net, &cli.data_dir) {
        (None, _) => net_format::Network::new(),
        (Some(p), None) => net_format::io::load(p).with_context(|| format!("load {p}"))?,
        (Some(p), Some(dir)) => {
            let path = std::path::Path::new(p);
            if path.is_absolute()
                || path
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                anyhow::bail!("--net path {p:?} escapes --data-dir");
            }
            let p = dir.join(path);
            net_format::io::load(&p).with_context(|| format!("load {}", p.display()))?
        }
    };

    let srv = server::Server::start(
        net,
        sim::serve::ServeOpts {
            backend,
            seed: cli.seed,
            cfg: sim::SimConfig {
                int_weights: cli.int_weights,
                ..Default::default()
            },
            data_dir: cli.data_dir.clone(),
            max_upload_bytes: cli.max_upload_mb.saturating_mul(1024 * 1024),
            abort: Arc::new(AtomicBool::new(false)),
        },
    );

    if let Some(s) = &cli.script {
        // the script's own file fns are sandboxed; the server-side read
        // of the script file itself is the operator's choice
        let code = std::fs::read_to_string(s).with_context(|| format!("read {s}"))?;
        srv.send(proto::ClientMsg::Eval { code });
    }

    if let Some(d) = &cli.web {
        server::web_dir_or_err(d)?;
    }

    let listener =
        std::net::TcpListener::bind(&cli.bind).with_context(|| format!("bind {}", cli.bind))?;
    eprintln!(
        "listening on {}{}{}",
        cli.bind,
        cli.data_dir
            .as_ref()
            .map(|d| format!(" (data-dir {})", d.display()))
            .unwrap_or_default(),
        cli.web
            .as_ref()
            .map(|d| format!(" (web {})", d.display()))
            .unwrap_or_default()
    );
    match &cli.web {
        Some(_) => eprintln!("browser editor: open http://{}/", cli.bind),
        None => eprintln!("editor: editor --connect {}", cli.bind),
    }
    srv.accept_web(listener, cli.web)
}
