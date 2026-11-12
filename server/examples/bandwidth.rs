//! Bandwidth probe: runs `sim::serve` on a large grid and sums the
//! encoded `Activity` frame sizes per wall-second — the number the
//! Stage-4 spec budgets at ≤ 2 MB/s at ~30 fps for a 1M-neuron grid.
//!
//!   cargo run -p server --release --example bandwidth -- \
//!        [--side 100] [--3d] [--cpu] [--secs 10] [--noise-hz 60]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use net_format::Network;
use proto::{ClientMsg, ServerMsg};
use sim::serve::ServeOpts;
use sim::{BackendKind, SimConfig};

fn main() {
    let mut side = 100usize;
    let mut d3 = false;
    let mut cpu = false;
    let mut secs = 10u64;
    let mut noise_hz = 60.0f64;
    let mut kick = 20.0f64;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--side" => {
                side = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--secs" => {
                secs = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--noise-hz" => {
                noise_hz = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--kick" => {
                kick = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--3d" => {
                d3 = true;
                i += 1;
            }
            "--cpu" => {
                cpu = true;
                i += 1;
            }
            a => panic!("unknown arg {a}"),
        }
    }

    let n = if d3 { side * side * side } else { side * side };
    eprintln!(
        "grid {n} neurons ({side}x{side}{})",
        if d3 {
            format!("x{side}")
        } else {
            String::new()
        }
    );

    // Build the grid through the script API so the geometry path matches
    // what a real session produces.
    let (tx, rx) = mpsc::channel::<(u64, ClientMsg)>();
    let bytes = Arc::new(AtomicU64::new(0));
    let frames = Arc::new(AtomicU64::new(0));
    let other = Arc::new(AtomicU64::new(0));
    let (b2, f2, o2) = (bytes.clone(), frames.clone(), other.clone());
    let (btx, brx) = mpsc::channel::<()>();
    std::thread::spawn(move || {
        let emit: sim::serve::Emit =
            std::rc::Rc::new(std::cell::RefCell::new(move |ev: proto::ServerEvent| {
                let sz = proto::encode(&ev.msg).len() as u64;
                if let ServerMsg::Activity(a) = &ev.msg {
                    b2.fetch_add(sz, Ordering::Relaxed);
                    f2.fetch_add(1, Ordering::Relaxed);
                    let mut lk = LAST.lock().unwrap();
                    lk.push((
                        a.time_ms,
                        a.spikes_total,
                        a.rate_hz,
                        a.spike_bits
                            .iter()
                            .map(|b| b.count_ones() as usize)
                            .sum::<usize>(),
                    ));
                    if lk.len() > 400 {
                        lk.remove(0);
                    }
                } else {
                    o2.fetch_add(sz, Ordering::Relaxed);
                    match &ev.msg {
                        ServerMsg::Console(s) | ServerMsg::Error(s) => {
                            eprintln!("[worker] {s}");
                            if s == "built" {
                                let _ = btx.send(());
                            }
                        }
                        _ => {}
                    }
                }
            }));
        sim::serve::run(
            Network::new(),
            ServeOpts {
                backend: if cpu {
                    BackendKind::Cpu
                } else {
                    BackendKind::Cuda
                },
                seed: 1,
                cfg: SimConfig::default(),
                data_dir: None,
                max_upload_bytes: usize::MAX,
                quota_bytes: 0,
                abort: Arc::new(AtomicBool::new(false)),
                shutdown: Arc::new(AtomicBool::new(false)),
                sim_limits: sim::SimLimits::default(),
            },
            rx,
            emit,
        );
    });
    std::thread::sleep(Duration::from_millis(300));

    let dim = if d3 {
        format!("net.create_3d_grid(\"g\", {side}, {side}, {side}, 3.0);")
    } else {
        format!("net.create_2d_grid(\"g\", {side}, {side}, 3.0);")
    };
    let script = format!("{dim} sim.set_noise({noise_hz}, {kick}); sim.run(50); print(\"built\");");
    tx.send((
        0,
        ClientMsg::Hello {
            proto: proto::PROTO_VERSION,
        },
    ))
    .unwrap();
    tx.send((0, ClientMsg::Eval { code: script })).unwrap();
    // wait for the build's sentinel before measuring
    if brx.recv_timeout(Duration::from_secs(300)).is_err() {
        panic!("build did not finish in 300s");
    }
    tx.send((0, ClientMsg::AutoRun(true))).unwrap();
    std::thread::sleep(Duration::from_secs(1));

    bytes.store(0, Ordering::Relaxed);
    frames.store(0, Ordering::Relaxed);
    let t0 = Instant::now();
    std::thread::sleep(Duration::from_secs(secs));
    let el = t0.elapsed().as_secs_f64();
    tx.send((0, ClientMsg::AutoRun(false))).unwrap();

    let b = bytes.load(Ordering::Relaxed) as f64;
    let f = frames.load(Ordering::Relaxed) as f64;
    println!(
        "n={} frames={} window={:.1}s  activity={:.0} B/s  ({:.2} MB/s)  avg-frame={:.0} B  fps={:.1}  other={} B total",
        n,
        f as u64,
        el,
        b / el,
        b / el / 1e6,
        b / f.max(1.0),
        f / el,
        other.load(Ordering::Relaxed),
    );
    let lk = LAST.lock().unwrap();
    if let (Some(first), Some(last)) = (lk.first(), lk.last()) {
        let dms = last.0 - first.0;
        let dsp = last.1 - first.1;
        println!(
            "sim covered {:.0}ms over the frames; spikes/window={} rate_hz={:.0}; set-bits/frame={}",
            dms, dsp, last.2, last.3
        );
    }
}

use std::sync::Mutex;
static LAST: Mutex<Vec<(f64, u64, f64, usize)>> = Mutex::new(Vec::new());
