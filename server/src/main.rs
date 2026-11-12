//! neuro_edit server: the sim worker headless, serving the wire protocol
//! over WebSocket. Editors (`editor --connect`) and later browser UIs
//! attach to one shared run.
//!
//!   server [--bind 127.0.0.1:9337] [--data-dir DIR] [--net file.ne2]
//!          [--cpu|--cuda] [--seed N] [--script x.rhai]
//!   server --accounts STATE-DIR --data-dir ROOT --web editor/dist-web
//!          [--public-url https://…] [--trust-proxy ip[,ip]]
//!   server admin add|list|activate|disable --accounts STATE-DIR …
//!
//! Without `--accounts`: one shared run, loopback only, no auth — the
//! mode every test and local user has always had. With it: per-user
//! runs + the HTTP front door (landing, signup, back office, /app).

use std::io::Read;
use std::net::ToSocketAddrs;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Context;
use clap::{Parser, Subcommand};
use sim::BackendKind;

#[derive(Parser)]
#[command(
    name = "server",
    about = "neuro_edit sim server (WebSocket + multi-user HTTP)"
)]
struct Cli {
    /// Address to bind. Loopback only unless --accounts is given —
    /// multi-user production binds a LAN address behind nginx.
    #[arg(long, default_value = "127.0.0.1:9337")]
    bind: String,
    /// Confine every file path scripts/clients touch to this directory
    /// (.. and absolute paths are refused). REQUIRED for remote serving;
    /// with --accounts, user files live under DIR/users/<id>/.
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
    /// Serve the browser editor (index.html + wasm bundle) from this
    /// directory (`server --web editor/dist-web`); required with
    /// --accounts (it becomes /app).
    #[arg(long)]
    web: Option<std::path::PathBuf>,
    /// Cap on `UploadDataset`/chunked-upload bytes per file (MiB).
    #[arg(long, default_value_t = 64)]
    max_upload_mb: usize,
    /// Multi-user mode: accounts.db + user sessions live in STATE-DIR.
    /// Requires --data-dir (per-user roots) and --web (/app).
    #[arg(long)]
    accounts: Option<std::path::PathBuf>,
    /// The outward origin (https://neuroedit.ad4m.dev): sets the session
    /// cookie's Secure flag and the Origin POSTs must carry.
    #[arg(long)]
    public_url: Option<String>,
    /// Comma-separated proxy IPs whose X-Forwarded-For/Proto headers
    /// are trusted; spoofed headers from anyone else are ignored.
    #[arg(long, value_delimiter = ',')]
    trust_proxy: Vec<std::net::IpAddr>,
    /// Stop a user's run after this many client-less, job-less minutes.
    #[arg(long, default_value_t = 15)]
    idle_minutes: u64,
    /// Cap on simultaneously live per-user runs.
    #[arg(long, default_value_t = 4)]
    max_runs: usize,
    /// Default per-user storage quota (MiB).
    #[arg(long, default_value_t = 2048)]
    default_quota_mb: u64,
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Back-office operations on the accounts store.
    Admin {
        #[command(subcommand)]
        action: AdminAction,
    },
}

#[derive(Subcommand)]
enum AdminAction {
    /// Create (or re-activate) an admin account. The password is read
    /// from stdin — never argv, so it stays out of history/ps.
    Add {
        /// STATE-DIR holding accounts.db.
        #[arg(long)]
        accounts: std::path::PathBuf,
        #[arg(long)]
        email: String,
        #[arg(long)]
        name: String,
    },
    /// List all users.
    List {
        #[arg(long)]
        accounts: std::path::PathBuf,
    },
    /// Activate a pending/disabled account.
    Activate {
        #[arg(long)]
        accounts: std::path::PathBuf,
        email: String,
    },
    /// Disable an account (sessions die, its run stops).
    Disable {
        #[arg(long)]
        accounts: std::path::PathBuf,
        email: String,
    },
}

fn bind_is_loopback(bind: &str) -> bool {
    match bind.to_socket_addrs() {
        Ok(mut a) => a.all(|s| s.ip().is_loopback()),
        Err(_) => {
            bind.starts_with("127.") || bind.starts_with("[::1]") || bind.contains("localhost")
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if let Some(Cmd::Admin { action }) = &cli.command {
        return admin(action);
    }
    let backend = if cli.cpu {
        BackendKind::Cpu
    } else {
        BackendKind::Cuda
    };

    // A remote-reachable server must not hand the filesystem to clients:
    // loopback only without --accounts (any non-loopback bind refused).
    if cli.accounts.is_none() && !bind_is_loopback(&cli.bind) {
        anyhow::bail!(
            "--bind {} without --accounts would expose the shared run — \
             refusing (non-loopback binds need the accounts mode)",
            cli.bind
        );
    }
    if cli.accounts.is_some() {
        if cli.data_dir.is_none() {
            anyhow::bail!("--accounts requires --data-dir (per-user roots live under it)");
        }
        match &cli.web {
            Some(d) => server::web_dir_or_err(d)?,
            None => anyhow::bail!("--accounts requires --web (the /app editor bundle)"),
        }
    }

    if let Some(state_dir) = &cli.accounts {
        return serve_multi(&cli, state_dir, backend);
    }

    // -------- shared-run mode (unchanged since Stage 4) --------
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
            quota_bytes: 0,
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

/// Multi-user serve: accounts DB + HTTP front door + per-user runs.
fn serve_multi(cli: &Cli, state_dir: &std::path::Path, backend: BackendKind) -> anyhow::Result<()> {
    let accounts = Arc::new(server::accounts::Accounts::open(state_dir)?);
    let multi = Arc::new(server::multi::Multi::new(
        accounts.clone(),
        server::multi::MultiOpts {
            data_root: cli.data_dir.clone().unwrap(),
            web_dir: cli.web.clone().unwrap(),
            public_url: cli.public_url.clone(),
            trust_proxy: cli.trust_proxy.clone(),
            backend,
            seed: cli.seed,
            cfg: sim::SimConfig {
                int_weights: cli.int_weights,
                ..Default::default()
            },
            max_upload_bytes: cli.max_upload_mb.saturating_mul(1024 * 1024),
            max_runs: cli.max_runs,
            idle: std::time::Duration::from_secs(cli.idle_minutes * 60),
            default_quota_bytes: cli.default_quota_mb * (1 << 20),
        },
    ));
    let listener =
        std::net::TcpListener::bind(&cli.bind).with_context(|| format!("bind {}", cli.bind))?;
    let url = cli
        .public_url
        .clone()
        .unwrap_or_else(|| format!("http://{}", cli.bind));
    eprintln!(
        "listening on {} — {url} (accounts {}, data {}, {} users max, idle {}m)",
        cli.bind,
        state_dir.display(),
        cli.data_dir.as_ref().unwrap().display(),
        cli.max_runs,
        cli.idle_minutes,
    );
    eprintln!(
        "first admin: server admin add --accounts {} --email X --name Y",
        state_dir.display()
    );
    multi.serve(listener)
}

/// `server admin …` — same store the running server uses (SQLite is
/// fine with a CLI alongside the server; WAL keeps readers unblocked).
fn admin(action: &AdminAction) -> anyhow::Result<()> {
    match action {
        AdminAction::Add {
            accounts: dir,
            email,
            name,
        } => {
            eprint!("password for {email} (12+ chars): ");
            let mut pw = String::new();
            std::io::stdin().read_to_string(&mut pw)?;
            let pw = pw.trim_end_matches(['\n', '\r']);
            let u = server::accounts::Accounts::open(dir)?.create_admin(email, name, pw)?;
            eprintln!("admin {} ({}) active", u.email, u.id);
        }
        AdminAction::List { accounts: dir } => {
            for u in server::accounts::Accounts::open(dir)?.list() {
                println!(
                    "{}\t{}\t{}\t{}MB\t{}{}",
                    u.id,
                    u.email,
                    u.status,
                    u.quota_bytes / (1 << 20),
                    u.name,
                    if u.is_admin { " (admin)" } else { "" }
                );
            }
        }
        AdminAction::Activate {
            accounts: dir,
            email,
        } => {
            let accounts = server::accounts::Accounts::open(dir)?;
            let u = accounts
                .by_email(email)
                .with_context(|| format!("no user {email}"))?;
            accounts.set_status(u.id, "active")?;
            eprintln!("{email} active");
        }
        AdminAction::Disable {
            accounts: dir,
            email,
        } => {
            let accounts = server::accounts::Accounts::open(dir)?;
            let u = accounts
                .by_email(email)
                .with_context(|| format!("no user {email}"))?;
            accounts.set_status(u.id, "disabled")?;
            accounts.delete_user_sessions(u.id);
            eprintln!("{email} disabled (sessions cleared; a running server drops the run)");
        }
    }
    Ok(())
}
