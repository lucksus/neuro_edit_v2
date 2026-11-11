//! `sim` — headless runner for neuro_edit v2 networks.
//!
//! Examples:
//!   sim --net lsm.ne2 --script train.rhai
//!   sim --eval 'net.create_2d_grid("g",5,5,40.0); print(net)'
//!   sim --bench 100000 --bench-k 100 --dt 0.1 --ms 2000 --cuda
//!   sim                                  # interactive Rhai console

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use net_format::Network;

#[derive(Parser, Debug)]
#[command(name = "sim", about = "neuro_edit v2 headless simulator")]
struct Args {
    /// Network file (.ne2 or .json)
    #[arg(long)]
    net: Option<PathBuf>,

    /// Rhai script file to run
    #[arg(long)]
    script: Option<PathBuf>,

    /// Evaluate a Rhai snippet (may be repeated)
    #[arg(long)]
    eval: Vec<String>,

    /// Force CPU backend
    #[arg(long)]
    cpu: bool,

    /// Force CUDA backend
    #[arg(long)]
    cuda: bool,

    /// Benchmark: random net with N neurons
    #[arg(long)]
    bench: Option<usize>,

    /// Benchmark out-degree
    #[arg(long, default_value_t = 100)]
    bench_k: usize,

    /// Weight multiplier for the benchmark net
    #[arg(long, default_value_t = 1.0)]
    wscale: f32,

    /// Inhibitory weight magnitude for the benchmark net
    #[arg(long, default_value_t = 5.0)]
    w_inh: f32,

    /// Simulation step, ms
    #[arg(long, default_value_t = 0.1)]
    dt: f32,

    /// Run this many ms (for --bench or a bare --net)
    #[arg(long, default_value_t = 1000.0)]
    ms: f32,

    /// Warmup ms for --bench
    #[arg(long, default_value_t = 100.0)]
    warm: f32,

    /// Background noise rate, Hz (Poisson kicks)
    #[arg(long, default_value_t = 0.0)]
    noise: f32,

    /// Seed
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

fn load_net(path: &PathBuf) -> anyhow::Result<Network> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => Ok(net_format::io::load_json(path)?),
        _ => Ok(net_format::io::load(path)?),
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    let backend = if args.cpu {
        sim::BackendKind::Cpu
    } else {
        sim::BackendKind::Cuda
    };

    if let Some(n) = args.bench {
        let net =
            sim::bench::bench_net(n, args.bench_k, args.seed, args.wscale, args.w_inh, args.dt);
        let r = sim::bench::run_bench(
            &net, backend, args.dt, args.ms, args.warm, args.noise, args.seed,
        );
        println!(
            "{:?} N={} syn={} dt={} sim={}ms wall={:.3}s realtime_x={:.2} rate={:.1}Hz spikes={} gpu_mem={}MB",
            backend,
            n,
            n * args.bench_k,
            args.dt,
            args.ms,
            r.wall_s,
            r.realtime_x,
            r.rate_hz,
            r.spikes,
            r.gpu_mem_mb
        );
        return ExitCode::SUCCESS;
    }

    let net = match &args.net {
        Some(p) => match load_net(p) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("error loading {}: {e}", p.display());
                return ExitCode::FAILURE;
            }
        },
        None => Network::new(),
    };

    // Rhai environment over this net.
    let mut env = match sim::script::Env::new(net, backend, args.seed) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("sim init error: {e}");
            return ExitCode::FAILURE;
        }
    };
    env.cfg.borrow_mut().dt_ms = args.dt;
    env.cfg.borrow_mut().noise_rate_hz = args.noise;

    if let Some(path) = &args.script {
        let code = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error reading {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = env.eval_str(&code) {
            eprintln!("script error: {e}");
            return ExitCode::FAILURE;
        }
    }
    for code in &args.eval {
        if let Err(e) = env.eval_str(code) {
            eprintln!("eval error: {e}");
            return ExitCode::FAILURE;
        }
    }
    if args.script.is_none() && args.eval.is_empty() && args.net.is_none() {
        repl(&mut env);
    } else if args.net.is_some() && args.script.is_none() && args.eval.is_empty() {
        // bare --net: just run
        println!("running {}ms on {:?} backend...", args.ms, backend);
        env.run_ms(args.ms);
        println!("t={:.1}ms spikes={}", env.time_ms(), env.spikes_total());
    }
    ExitCode::SUCCESS
}

fn repl(env: &mut sim::script::Env) {
    println!("neuro_edit v2 sim — Rhai console. `help()` for the API, Ctrl-D to exit.");
    let stdin = std::io::stdin();
    print!("# ");
    std::io::stdout().flush().ok();
    for line in stdin.lock().lines() {
        match line {
            Ok(l) => {
                if l.trim() == "exit" {
                    break;
                }
                match env.eval_str(&l) {
                    Ok(v) => println!("{v}"),
                    Err(e) => println!("error: {e}"),
                }
            }
            Err(_) => break,
        }
        print!("# ");
        std::io::stdout().flush().ok();
    }
}
