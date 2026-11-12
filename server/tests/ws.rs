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
