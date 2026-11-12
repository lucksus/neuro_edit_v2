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
