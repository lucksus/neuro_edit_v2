//! End-to-end tests over the real WebSocket transport: server + worker
//! + two wire clients. Mirrors `sim/tests/serve.rs` but every frame goes
//! through bincode-encoded WS messages — the same path `editor
//! --connect` uses.

use std::net::TcpListener;
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use net_format::Network;
use proto::{ClientMsg, NetEdit, ServerMsg};
use sim::serve::ServeOpts;
use sim::{BackendKind, SimConfig};

struct Ws {
    srv: server::Server,
    addr: String,
}

fn start(net: Network, data_dir: Option<std::path::PathBuf>, seed: u64) -> Ws {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let srv = server::Server::start(
        net,
        ServeOpts {
            backend: BackendKind::Cpu,
            seed,
            cfg: SimConfig::default(),
            data_dir,
            max_upload_bytes: usize::MAX,
            abort: Arc::new(AtomicBool::new(false)),
        },
    );
    srv.spawn_acceptor(listener);
    Ws { srv, addr }
}

impl Ws {
    fn client(&self) -> server::ClientConn {
        server::ClientConn::connect(&self.addr).unwrap()
    }
}

/// Read until `f` matches; returns everything received.
fn until(c: &mut server::ClientConn, f: impl Fn(&ServerMsg) -> bool) -> Vec<ServerMsg> {
    let mut out = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        match c.recv_opt(Duration::from_millis(500)).unwrap() {
            Some(m) => {
                let hit = f(&m);
                out.push(m);
                if hit {
                    return out;
                }
            }
            None => assert!(Instant::now() < deadline, "timed out: {out:?}"),
        }
    }
}

/// Hydrate a client: Hello → targeted Hello + Geo + Status. The targeted
/// `Hello` is hydration's first frame — a *broadcast* `Status` from the
/// worker's startup snapshot may precede it, so wait for Hello first.
fn hydrate(c: &mut server::ClientConn) -> Vec<ServerMsg> {
    c.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })
    .unwrap();
    let mut out = until(c, |m| matches!(m, ServerMsg::Hello { .. }));
    out.extend(until(c, |m| matches!(m, ServerMsg::Status { .. })));
    out
}

fn base_version(msgs: &[ServerMsg]) -> u64 {
    msgs.iter()
        .find_map(|m| match m {
            ServerMsg::Geo(g) => Some(g.version),
            _ => None,
        })
        .expect("Geo")
}

#[test]
fn ws_hello_hydrates() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let w = start(net, None, 1);
    let mut c = w.client();
    let msgs = hydrate(&mut c);
    assert!(msgs
        .iter()
        .any(|m| matches!(m, ServerMsg::Hello { proto, .. } if *proto == proto::PROTO_VERSION)));
    assert!(msgs.iter().any(|m| matches!(
        m,
        ServerMsg::Geo(g) if g.pos.len() == 1
    )));
    assert!(msgs.iter().any(|m| matches!(m, ServerMsg::Status { .. })));
    w.srv.shutdown();
}

#[test]
fn ws_second_client_sees_edits() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    net.add_neuron([5.0; 3], net_format::NeuronKind::Excitatory);
    let w = start(net, None, 1);
    let mut c1 = w.client();
    let mut c2 = w.client();
    let base = base_version(&hydrate(&mut c1));
    hydrate(&mut c2);

    c1.send(&ClientMsg::Edit {
        base,
        edit: NetEdit::Connect {
            from: 0,
            to: 1,
            w: 1.0,
        },
    })
    .unwrap();
    // the delta is broadcast — client 2 sees client 1's edit
    let msgs = until(&mut c2, |m| {
        matches!(m, ServerMsg::GeoDelta(_)) || matches!(m, ServerMsg::Geo(_))
    });
    let d = msgs
        .iter()
        .find_map(|m| match m {
            ServerMsg::GeoDelta(d) => Some(&**d),
            _ => None,
        })
        .expect("GeoDelta on client 2");
    assert_eq!(d.syn_added.len(), 1);
    w.srv.shutdown();
}

/// A job script: builds a small noisy net, runs a few epochs streaming
/// `job_progress`, then burns script time so a client can disconnect
/// mid-job, then writes a marker into `job_dir()`.
fn job_script() -> String {
    r#"
        net.create_2d_grid("g", 8, 8, 10.0);
        net.create_connections("g", 1.0, 2.0, 1.0);
        net.set_synapse_weights("g", 0.6, 0.2);
        sim.set_noise(15.0, 8.0);
        for i in 0..4 {
            sim.run(150);
            job_progress(#{epoch: i, t_ms: sim.time()});
        }
        for j in 0..1500000 { let q = j % 7; }
        job_progress(#{epoch: 99});
        write_to_file(job_dir() + "/marker.txt", [1.0]);
    "#
    .to_string()
}

#[test]
fn ws_job_progress_survives_client() {
    let dir = std::env::temp_dir().join(format!("ws_job_{}_{}", std::process::id(), line!()));
    std::fs::create_dir_all(&dir).unwrap();
    let w = start(Network::new(), Some(dir.clone()), 7);
    let mut c1 = w.client();
    hydrate(&mut c1);

    c1.send(&ClientMsg::StartJob {
        name: "detector-smoke".into(),
        script: job_script(),
    })
    .unwrap();
    // JobStarted + the first JobProgress broadcast, then drop the
    // starter — the job must keep running without it.
    let msgs = until(&mut c1, |m| matches!(m, ServerMsg::JobProgress { .. }));
    let job_id = msgs
        .iter()
        .find_map(|m| match m {
            ServerMsg::JobStarted { id, .. } => Some(*id),
            _ => None,
        })
        .expect("JobStarted before progress");
    drop(c1);

    let mut c2 = w.client();
    hydrate(&mut c2);
    c2.send(&ClientMsg::ListJobs).unwrap();
    let msgs = until(&mut c2, |m| matches!(m, ServerMsg::JobList { .. }));
    let listed = msgs
        .iter()
        .find_map(|m| match m {
            ServerMsg::JobList { jobs } => Some(jobs.clone()),
            _ => None,
        })
        .unwrap();
    let entry = listed
        .iter()
        .find(|j| j.id == job_id)
        .expect("job listed after starter disconnected");
    assert_eq!(entry.name, "detector-smoke");

    // it finishes on its own and broadcasts the result
    let msgs = until(
        &mut c2,
        |m| matches!(m, ServerMsg::JobDone { id, .. } if *id == job_id),
    );
    let done = msgs
        .iter()
        .find_map(|m| match m {
            ServerMsg::JobDone { id, status, error } if *id == job_id => {
                Some((status.clone(), error.clone()))
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(done.0, "done", "job failed: {:?}", done.1);

    // runs/<id>/ got the script, metrics, result + the script's own file
    let run = dir.join("runs").join(job_id.to_string());
    assert!(run.join("script.rhai").is_file());
    assert!(run.join("metrics.json").is_file());
    assert!(run.join("result.json").is_file());
    assert!(run.join("marker.txt").is_file());
    let result: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["name"], "detector-smoke");

    // re-open: the detail carries script + accumulated progress
    c2.send(&ClientMsg::OpenJob { id: job_id }).unwrap();
    let msgs = until(
        &mut c2,
        |m| matches!(m, ServerMsg::JobDetail { id, .. } if *id == job_id),
    );
    let (status, script, progress, result_json) = msgs
        .iter()
        .find_map(|m| match m {
            ServerMsg::JobDetail {
                status,
                script,
                progress,
                result_json,
                ..
            } => Some((
                status.clone(),
                script.clone(),
                progress.len(),
                result_json.is_some(),
            )),
            _ => None,
        })
        .unwrap();
    assert_eq!(status, "done");
    assert!(script.contains("job_progress"));
    assert_eq!(progress, 5, "four epoch rows + the tail row");
    assert!(result_json);

    w.srv.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// `examples/detector.rhai` end-to-end as a background job: the shipped
/// script with its parameter block swapped for a tiny one (the
/// `[TEST-BLOCK]` markers delimit it), run server-side under a sandbox.
#[test]
fn ws_detector_example_small() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/detector.rhai"),
    )
    .unwrap();
    let b0 = src.find("// ---------- parameters").unwrap();
    let b1 = src.find("// ---------- /parameters").unwrap();
    let (head, tail) = (&src[..b0], &src[b1..]);
    let script = format!(
        "{head}\
let SEEDS = [1];
let DS_NAME = \"det_test\";
let NCH = 4;
let LEN_MS = 12000;
let DT_MS = 1.0;
let SPB = 10;
let RATE_HZ = 25.0;
let EV_EVERY = 400.0;
let EV_JITTER = 100.0;
let EV_WIN = 10.0;
let EV_STR = 180.0;
let DELAY_MS = 0.0;
let WIN_MS = 20;
let MARGIN = 60;
let NEG_PRE = 200.0;
let NEG_POST = 150.0;
let RIDGE_L = 1.0;
let NPERM = 5;
let LIQ_SIDE = 4;
let LIQ_SPACING = 60.0;
let CONN_SCALING = 0.35;
let CONN_LAMBDA = 90.0;
let W_REC = 3.0;
let W_IN = 12.0;
let TAU_REC = 0.0;
let TAU_INH = 0.0;
let PER_SRC = 8;
let TRAIN = [0.0, 6000.0];
let VAL = [6000.0, 9000.0];
let TEST = [9000.0, 12000.0];
{tail}"
    );

    let dir = std::env::temp_dir().join(format!("ws_det_{}_{}", std::process::id(), line!()));
    std::fs::create_dir_all(&dir).unwrap();
    let w = start(Network::new(), Some(dir.clone()), 7);
    let mut c = w.client();
    hydrate(&mut c);
    c.send(&ClientMsg::StartJob {
        name: "detector-small".into(),
        script,
    })
    .unwrap();
    let msgs = until(&mut c, |m| matches!(m, ServerMsg::JobDone { .. }));
    let (id, status, error) = msgs
        .iter()
        .find_map(|m| match m {
            ServerMsg::JobDone { id, status, error } => Some((*id, status.clone(), error.clone())),
            _ => None,
        })
        .unwrap();
    let console: Vec<String> = msgs
        .iter()
        .filter_map(|m| match m {
            ServerMsg::Console(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(status, "done", "job console: {console:?} err: {error:?}");
    assert!(
        console.iter().any(|s| s.contains("liquid auc=")),
        "no detector output: {console:?}"
    );

    // runs/<id>/ got the full record incl. the saved trained net
    let run = dir.join("runs").join(id.to_string());
    assert!(run.join("script.rhai").is_file());
    assert!(run.join("metrics.json").is_file());
    assert!(run.join("result.json").is_file());
    assert!(run.join("detector.ne2").is_file());
    // the generated dataset is cached under datasets/
    assert!(dir.join("datasets/det_test/signal.f32").is_file());
    w.srv.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ws_job_cancel() {
    let dir = std::env::temp_dir().join(format!("ws_jc_{}_{}", std::process::id(), line!()));
    std::fs::create_dir_all(&dir).unwrap();
    let w = start(Network::new(), Some(dir.clone()), 7);
    let mut c = w.client();
    hydrate(&mut c);

    // an infinite loop — only cancellation can end it
    c.send(&ClientMsg::StartJob {
        name: "forever".into(),
        script: r#"
            let i = 0;
            while true {
                i += 1;
                if i % 50000 == 0 { job_progress(#{i: i}); }
            }
        "#
        .into(),
    })
    .unwrap();
    let msgs = until(&mut c, |m| matches!(m, ServerMsg::JobProgress { .. }));
    let job_id = msgs
        .iter()
        .find_map(|m| match m {
            ServerMsg::JobStarted { id, .. } => Some(*id),
            _ => None,
        })
        .unwrap();
    c.send(&ClientMsg::CancelJob { id: job_id }).unwrap();
    let msgs = until(&mut c, |m| matches!(m, ServerMsg::JobDone { .. }));
    let status = msgs
        .iter()
        .find_map(|m| match m {
            ServerMsg::JobDone { status, .. } => Some(status.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(status, "cancelled");

    // the interactive worker never noticed — it's still answering
    c.send(&ClientMsg::Eval {
        code: "print(\"alive\")".into(),
    })
    .unwrap();
    let msgs = until(
        &mut c,
        |m| matches!(m, ServerMsg::Console(s) if s == "alive"),
    );
    assert!(msgs
        .iter()
        .any(|m| matches!(m, ServerMsg::Console(s) if s == "alive")));

    w.srv.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// A client saying Bye (editor quit) must not kill the shared worker —
/// the server swallows it; only `Server::shutdown` injects Bye into the
/// command channel.
#[test]
fn ws_bye_does_not_kill_other_clients() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let w = start(net, None, 1);
    let mut c1 = w.client();
    let mut c2 = w.client();
    hydrate(&mut c1);
    hydrate(&mut c2);

    c1.send(&ClientMsg::Bye).unwrap();
    drop(c1); // socket close = the real leave signal
    std::thread::sleep(Duration::from_millis(200));

    // client 2's eval still runs on the live worker
    c2.send(&ClientMsg::Eval {
        code: "print(\"still alive\")".into(),
    })
    .unwrap();
    let msgs = until(
        &mut c2,
        |m| matches!(m, ServerMsg::Console(s) if s.contains("still alive")),
    );
    assert!(!msgs.is_empty());
    w.srv.shutdown();
}

#[test]
fn ws_stale_edit_refused() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    net.add_neuron([5.0; 3], net_format::NeuronKind::Excitatory);
    let w = start(net, None, 1);
    let mut c = w.client();
    hydrate(&mut c);

    c.send(&ClientMsg::Edit {
        base: 999,
        edit: NetEdit::Connect {
            from: 0,
            to: 1,
            w: 1.0,
        },
    })
    .unwrap();
    let msgs = until(&mut c, |m| matches!(m, ServerMsg::GridTraces { .. }));
    assert!(
        msgs.iter()
            .any(|m| matches!(m, ServerMsg::Error(s) if s.contains("stale"))),
        "{msgs:?}"
    );
    assert!(msgs.iter().any(|m| matches!(m, ServerMsg::Geo(_))));
    w.srv.shutdown();
}

#[test]
fn ws_path_escape_refused() {
    let dir = std::env::temp_dir().join(format!("ws_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let w = start(net, Some(dir.clone()), 1);
    let mut c = w.client();
    hydrate(&mut c);

    c.send(&ClientMsg::Eval {
        code: "net.save(\"../escape.ne2\")".into(),
    })
    .unwrap();
    let msgs = until(
        &mut c,
        |m| matches!(m, ServerMsg::Console(s) if s.contains("!") || s.contains("path") || s.contains("..")),
    );
    assert!(msgs.iter().any(|m| matches!(
        m,
        ServerMsg::Console(s)
            if s.contains("path") || s.contains("..") || s.contains("refus") || s.contains("outside")
    )), "{msgs:?}");
    assert!(!dir.parent().unwrap().join("escape.ne2").exists());
    w.srv.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Rhai `import` must be confined to --data-dir like every other file
/// fn: `..` and absolute imports refused; a legit import inside still
/// works.
#[test]
fn ws_import_escape_refused() {
    let dir = std::env::temp_dir().join(format!("ws_imp_{}", std::process::id()));
    // next to server/, so "../ws_imp_out/..." resolves via the test cwd
    let outside = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ws_imp_out");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("evil.rhai"), "fn hi() { 99 }").unwrap();
    std::fs::write(dir.join("ok.rhai"), "fn hi() { 42 }").unwrap();
    let w = start(Network::new(), Some(dir.clone()), 1);
    let mut c = w.client();
    hydrate(&mut c);

    for code in [
        // up out of data_dir, then wherever the process cwd points
        r#"import "../ws_imp_out/evil" as e; print("BAD:"+e::hi());"#,
        // absolute import straight to the outside file
        &format!(
            "import \"{}/evil\" as e; print(\"BAD:\"+e::hi());",
            outside.display()
        ),
    ] {
        c.send(&ClientMsg::Eval {
            code: code.to_string(),
        })
        .unwrap();
        let msgs = until(
            &mut c,
            |m| matches!(m, ServerMsg::Console(s) if s.contains("escapes") || s.starts_with("BAD:")),
        );
        assert!(
            msgs.iter()
                .any(|m| matches!(m, ServerMsg::Console(s) if s.contains("escapes"))),
            "import not refused: {msgs:?}"
        );
    }

    // positive control: importing inside the dir works
    c.send(&ClientMsg::Eval {
        code: r#"import "ok" as o; print("imp="+o::hi());"#.into(),
    })
    .unwrap();
    let msgs = until(
        &mut c,
        |m| matches!(m, ServerMsg::Console(s) if s == "imp=42" || s.starts_with("!")),
    );
    assert!(
        msgs.iter()
            .any(|m| matches!(m, ServerMsg::Console(s) if s == "imp=42")),
        "inside import broke: {msgs:?}"
    );
    w.srv.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&outside);
}

/// A symlink planted inside --data-dir must not lead outside — for
/// net.save/net.load and for import.
#[cfg(unix)]
#[test]
fn ws_symlink_escape_refused() {
    let dir = std::env::temp_dir().join(format!("ws_sym_{}", std::process::id()));
    let outside = dir.parent().unwrap().join("ws_sym_out");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    let w = start(Network::new(), Some(dir.clone()), 1);
    let mut c = w.client();
    hydrate(&mut c);

    // a real .ne2 outside the dir + module file outside
    c.send(&ClientMsg::Eval {
        code: r#"net.add_neuron(0.0,0.0,0.0); net.save("ok.ne2"); print("saved");"#.into(),
    })
    .unwrap();
    until(
        &mut c,
        |m| matches!(m, ServerMsg::Console(s) if s == "saved"),
    );
    std::fs::copy(dir.join("ok.ne2"), outside.join("real.ne2")).unwrap();
    std::fs::write(outside.join("evil.rhai"), "fn hi() { 99 }").unwrap();

    // symlink dir inside → outside; symlink file inside → outside file
    std::os::unix::fs::symlink(&outside, dir.join("link")).unwrap();
    std::os::unix::fs::symlink(outside.join("real.ne2"), dir.join("evil.ne2")).unwrap();

    for code in [
        r#"net.save("link/x.ne2")"#,    // write through symlinked dir
        r#"net.load("link/real.ne2")"#, // read through symlinked dir
        r#"net.load("evil.ne2")"#,      // read through symlinked file
        r#"import "link/evil" as e;"#,  // module through symlinked dir
    ] {
        c.send(&ClientMsg::Eval { code: code.into() }).unwrap();
        let msgs = until(
            &mut c,
            |m| matches!(m, ServerMsg::Console(s) if s.contains("escapes") || s.contains("!")),
        );
        assert!(
            msgs.iter()
                .any(|m| matches!(m, ServerMsg::Console(s) if s.contains("escapes"))),
            "{code} not refused: {msgs:?}"
        );
    }
    assert!(!outside.join("x.ne2").exists());
    w.srv.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&outside);
}

/// The same scripted session gives identical results over the local
/// (channel) transport and the WS transport — the "local ≡ remote" test.
#[test]
fn ws_local_remote_equivalent() {
    let script = r#"
        net.create_2d_grid("g", 10, 10, 8.0);
        net.add_input("i", [[0, 5.0]]);
        sim.set_input("i", 5.0);
        sim.run(100);
        print("S=" + net.synapse_count());
        print("N=" + net.neuron_count());
        print("K=" + sim.spikes());
    "#;

    // --- local: sim::serve over a channel, seed 42
    let (ltx, lrx) = mpsc::channel();
    let (etx, erx) = mpsc::channel();
    std::thread::spawn(move || {
        let emit: sim::serve::Emit =
            std::rc::Rc::new(std::cell::RefCell::new(move |ev: proto::ServerEvent| {
                let _ = etx.send(ev.msg);
            }));
        sim::serve::run(
            Network::new(),
            ServeOpts {
                backend: BackendKind::Cpu,
                seed: 42,
                cfg: SimConfig::default(),
                data_dir: None,
                max_upload_bytes: usize::MAX,
                abort: Arc::new(AtomicBool::new(false)),
            },
            lrx,
            emit,
        );
    });
    ltx.send((
        0,
        ClientMsg::Hello {
            proto: proto::PROTO_VERSION,
        },
    ))
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        match erx.recv_timeout(Duration::from_secs(20)) {
            Ok(ServerMsg::Status { .. }) => break,
            Ok(_) => assert!(Instant::now() < deadline),
            Err(_) => panic!("no local hydration"),
        }
    }
    ltx.send((
        0,
        ClientMsg::Eval {
            code: script.into(),
        },
    ))
    .unwrap();
    let local = collect_k(&erx);
    let _ = ltx.send((0, ClientMsg::Bye));

    // --- remote: WS server, same seed
    let w = start(Network::new(), None, 42);
    let mut c = w.client();
    hydrate(&mut c);
    c.send(&ClientMsg::Eval {
        code: script.into(),
    })
    .unwrap();
    let remote = collect_k_ws(&mut c);
    w.srv.shutdown();

    assert_eq!(
        local, remote,
        "local vs remote mismatch: {local:?} vs {remote:?}"
    );
}

fn collect_k(rx: &mpsc::Receiver<ServerMsg>) -> Vec<String> {
    let mut out = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    while !out.iter().any(|s: &String| s.starts_with("K=")) {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(ServerMsg::Console(s))
                if s.starts_with("S=") || s.starts_with("N=") || s.starts_with("K=") =>
            {
                out.push(s)
            }
            Ok(_) => {}
            Err(_) => assert!(Instant::now() < deadline, "timed out: {out:?}"),
        }
    }
    out
}

fn collect_k_ws(c: &mut server::ClientConn) -> Vec<String> {
    let mut out = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    while !out.iter().any(|s: &String| s.starts_with("K=")) {
        match c.recv_opt(Duration::from_millis(500)).unwrap() {
            Some(ServerMsg::Console(s))
                if s.starts_with("S=") || s.starts_with("N=") || s.starts_with("K=") =>
            {
                out.push(s)
            }
            Some(_) => {}
            None => assert!(Instant::now() < deadline, "timed out: {out:?}"),
        }
    }
    out
}

fn http_get(addr: &str, target: &str) -> String {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    write!(
        s,
        "GET {target} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}

/// `server --web`: the same port answers plain GETs from `web_dir` and
/// upgrades WS connections — and refuses to read outside `web_dir`.
#[test]
fn ws_web_serves_http_and_ws_on_one_port() {
    let base = std::env::temp_dir().join(format!("ws_web_{}", std::process::id()));
    let dir = base.join("web");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("index.html"), "<html>ne-web</html>").unwrap();
    std::fs::write(dir.join("app.js"), "console.log(1)").unwrap();
    std::fs::write(base.join("secret.txt"), "nope").unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let srv = server::Server::start(
        Network::new(),
        ServeOpts {
            backend: BackendKind::Cpu,
            seed: 7,
            cfg: SimConfig::default(),
            data_dir: None,
            max_upload_bytes: usize::MAX,
            abort: Arc::new(AtomicBool::new(false)),
        },
    );
    srv.spawn_acceptor_web(listener, Some(dir.clone()));

    // plain HTTP: index at /, files by name, traversal refused
    let index = http_get(&addr, "/");
    assert!(index.starts_with("HTTP/1.1 200"), "{index:?}");
    assert!(index.contains("<html>ne-web</html>"), "{index:?}");
    let js = http_get(&addr, "/app.js");
    assert!(js.contains("Content-Type: text/javascript"), "{js:?}");
    assert!(js.contains("console.log(1)"), "{js:?}");
    let escape = http_get(&addr, "/../secret.txt");
    assert!(escape.starts_with("HTTP/1.1 404"), "{escape:?}");
    let missing = http_get(&addr, "/nope.wasm");
    assert!(missing.starts_with("HTTP/1.1 404"), "{missing:?}");

    // same port: the wire protocol still works
    let mut c = server::ClientConn::connect(&addr).unwrap();
    c.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })
    .unwrap();
    until(&mut c, |m| matches!(m, ServerMsg::Hello { .. }));
    c.send(&ClientMsg::Eval {
        code: "print(\"web-ok\")".into(),
    })
    .unwrap();
    until(
        &mut c,
        |m| matches!(m, ServerMsg::Console(s) if s == "web-ok"),
    );

    srv.shutdown();
    let _ = std::fs::remove_dir_all(&base);
}
