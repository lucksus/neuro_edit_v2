//! End-to-end tests for the message-driven worker (`sim::serve::run`):
//! drive it with `ClientMsg`s over a channel, capture `ServerEvent`s, and
//! assert on protocol behavior — hydration, edits, deltas, rate limiting,
//! sandboxing.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use net_format::Network;
use proto::{ClientMsg, NetEdit, ServerEvent, ServerMsg};
use sim::serve::{ServeOpts, MIN_ACTIVITY_MS};
use sim::{BackendKind, SimConfig};

const CLIENT: u64 = 7;

struct Rig {
    tx: mpsc::Sender<(u64, ClientMsg)>,
    rx: mpsc::Receiver<ServerEvent>,
    handle: std::thread::JoinHandle<()>,
}

fn start(net: Network, data_dir: Option<std::path::PathBuf>) -> Rig {
    let (tx, rx) = mpsc::channel::<(u64, ClientMsg)>();
    let (etx, erx) = mpsc::channel::<ServerEvent>();
    let handle = std::thread::spawn(move || {
        let emit: sim::serve::Emit = Rc::new(RefCell::new(move |ev| {
            let _ = etx.send(ev);
        }));
        sim::serve::run(
            net,
            ServeOpts {
                backend: BackendKind::Cpu,
                seed: 1,
                cfg: SimConfig::default(),
                data_dir,
                max_upload_bytes: usize::MAX,
                abort: Arc::new(AtomicBool::new(false)),
            },
            rx,
            emit,
        );
    });
    Rig {
        tx,
        rx: erx,
        handle,
    }
}

impl Rig {
    fn send(&self, m: ClientMsg) {
        self.tx.send((CLIENT, m)).unwrap();
    }

    /// Collect events until `f` matches (or the deadline hits).
    fn until(&self, f: impl Fn(&ServerMsg) -> bool) -> Vec<ServerEvent> {
        self.until_ev(|e| f(&e.msg))
    }

    /// Same, with the event's routing visible to the predicate.
    fn until_ev(&self, f: impl Fn(&ServerEvent) -> bool) -> Vec<ServerEvent> {
        let mut out = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            match self.rx.recv_timeout(Duration::from_secs(20)) {
                Ok(ev) => {
                    let hit = f(&ev);
                    out.push(ev);
                    if hit {
                        return out;
                    }
                }
                Err(_) => panic!("timed out waiting for event; got {out:?}"),
            }
            assert!(std::time::Instant::now() < deadline, "deadline exceeded");
        }
    }

    /// Drain whatever arrived within `ms`.
    fn drain(&self, ms: u64) -> Vec<ServerEvent> {
        let mut out = Vec::new();
        let end = std::time::Instant::now() + Duration::from_millis(ms);
        while let Ok(ev) = self.rx.recv_timeout(end - std::time::Instant::now()) {
            out.push(ev);
            if std::time::Instant::now() >= end {
                break;
            }
        }
        out
    }

    fn stop(self) {
        let _ = self.tx.send((CLIENT, ClientMsg::Bye));
        let _ = self.handle.join();
    }
}

fn first_geo(evs: &[ServerEvent]) -> &proto::Geo {
    evs.iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Geo(g) => Some(&**g),
            _ => None,
        })
        .expect("a Geo event")
}

#[test]
fn hello_hydrates_client() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let rig = start(net, None);
    rig.send(ClientMsg::Hello { proto: 1 });
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    let hello = evs
        .iter()
        .find(|e| matches!(e.msg, ServerMsg::Hello { .. }))
        .expect("Hello response");
    assert_eq!(hello.to, Some(CLIENT));
    let geo = evs
        .iter()
        .find(|e| matches!(e.msg, ServerMsg::Geo(_)) && e.to == Some(CLIENT))
        .expect("client-targeted Geo");
    match &geo.msg {
        ServerMsg::Geo(g) => assert_eq!(g.pos.len(), 1),
        _ => unreachable!(),
    }
    // hydration frames are client-targeted (the terminating event is the
    // targeted Status)
    assert!(matches!(evs.last().unwrap().msg, ServerMsg::Status { .. }));
    rig.stop();
}

#[test]
fn eval_script_change_broadcasts_geo() {
    let rig = start(Network::new(), None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::Eval {
        code: "net.create_2d_grid(\"g\", 4, 4, 8.0)".into(),
    });
    // GridTraces arrives after the Geo — wait for it, then check both.
    let evs = rig.until(|m| matches!(m, ServerMsg::GridTraces { .. }));
    // broadcast (not client-targeted) so every subscriber sees it
    let geo_ev = evs
        .iter()
        .find(|e| matches!(e.msg, ServerMsg::Geo(_)))
        .unwrap();
    assert_eq!(geo_ev.to, None);
    let geo = first_geo(&evs);
    assert_eq!(geo.pos.len(), 16);
    assert_eq!(geo.grids.len(), 1);
    assert_eq!(geo.grids[0].members.len(), 16);
    // GridTraces rides along sized to the new plane
    let gt = evs.iter().find_map(|e| match &e.msg {
        ServerMsg::GridTraces { grids, .. } => Some(grids),
        _ => None,
    });
    assert_eq!(gt.map(|g| g[0].len()), Some(16));
    rig.stop();
}

#[test]
fn edit_connect_yields_delta_not_geo() {
    let mut net = Network::new();
    let a = net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let b = net.add_neuron([5.0; 3], net_format::NeuronKind::Excitatory);
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    let base = first_geo(&evs).version;

    rig.send(ClientMsg::Edit {
        base,
        edit: NetEdit::Connect {
            from: a,
            to: b,
            w: 1.0,
        },
    });
    let evs = rig.until(|m| {
        matches!(m, ServerMsg::GeoDelta(_))
            || matches!(m, ServerMsg::Geo(_))
            || matches!(m, ServerMsg::Error(_))
    });
    let d = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => Some(&**d),
            _ => None,
        })
        .expect("a GeoDelta, not a full Geo");
    assert_eq!(d.syn_added.len(), 1);
    assert_eq!((d.syn_added[0].src, d.syn_added[0].tgt), (a, b));
    assert!(d.version > base);
    rig.stop();
}

#[test]
fn stale_base_refused_and_resynced() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    net.add_neuron([5.0; 3], net_format::NeuronKind::Excitatory);
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    rig.send(ClientMsg::Edit {
        base: 999,
        edit: NetEdit::Connect {
            from: 0,
            to: 1,
            w: 1.0,
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::GridTraces { .. }));
    assert!(
        evs.iter()
            .any(|e| matches!(&e.msg, ServerMsg::Error(s) if s.contains("stale"))),
        "stale edit must be refused: {evs:?}"
    );
    // resync: Error + Geo + GridTraces targeted at the offender
    assert!(evs.iter().all(|e| e.to == Some(CLIENT)));
    assert!(evs.iter().any(|e| matches!(e.msg, ServerMsg::Geo(_))));
    rig.stop();
}

#[test]
fn self_synapse_refused() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    let base = first_geo(&evs).version;
    rig.send(ClientMsg::Edit {
        base,
        edit: NetEdit::Connect {
            from: 0,
            to: 0,
            w: 1.0,
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(_)));
    assert!(evs
        .iter()
        .any(|e| matches!(&e.msg, ServerMsg::Error(s) if s.contains("self-synapse"))));
    rig.stop();
}

#[test]
fn run_ms_publishes_activity() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    net.add_neuron([5.0; 3], net_format::NeuronKind::Excitatory);
    net.connect(0, 1, 5.0);
    net.add_input("i", vec![(0, 1.0)], [0.0; 3]);
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::SetInputConst {
        name: "i".into(),
        value: 5.0,
    });
    rig.send(ClientMsg::RunMs(10.0));
    let evs = rig.until(|m| matches!(m, ServerMsg::Activity(a) if a.t_steps > 0));
    let a = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Activity(a) if a.t_steps > 0 => Some(&**a),
            _ => None,
        })
        .unwrap();
    assert!(a.t_steps > 0);
    assert_eq!(a.input_values.len(), 1);
    rig.stop();
}

#[test]
fn new_net_resets_gen_and_state() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    let gen0 = first_geo(&evs).gen;

    rig.send(ClientMsg::NewNet);
    let evs = rig.until(|m| matches!(m, ServerMsg::Geo(g) if g.gen > gen0));
    let geo = first_geo(&evs);
    assert!(geo.pos.is_empty());
    assert!(geo.gen > gen0);
    // a clean-slate Activity precedes/accompanies the new geo
    assert!(evs.iter().any(|e| matches!(
        &e.msg,
        ServerMsg::Activity(a) if a.t_steps == 0
    )));
    rig.stop();
}

#[test]
fn rate_limit_caps_activity_fps() {
    // A long synchronous run publishes many times internally; the wire
    // must see ≤ ~1/MIN_ACTIVITY_MS frames per second of wall time.
    let mut net = Network::new();
    for i in 0..50 {
        net.add_neuron([i as f32, 0.0, 0.0], net_format::NeuronKind::Excitatory);
    }
    net.add_input("i", vec![(0, 3.0)], [0.0; 3]);
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::SetInputConst {
        name: "i".into(),
        value: 5.0,
    });
    rig.until(|m| matches!(m, ServerMsg::Activity(_)));
    rig.send(ClientMsg::AutoRun(true));
    // collect ~700ms of activity frames
    let evs = rig.drain(700);
    rig.send(ClientMsg::AutoRun(false));
    let n_act = evs
        .iter()
        .filter(|e| matches!(e.msg, ServerMsg::Activity(_)))
        .count();
    assert!(n_act > 0, "expected activity while running");
    let cap = (700.0 / MIN_ACTIVITY_MS as f64).ceil() as usize + 2;
    assert!(
        n_act <= cap,
        "rate limit violated: {n_act} frames in ~700ms (cap {cap})"
    );
    rig.stop();
}

#[test]
fn data_dir_confines_paths() {
    let dir = std::env::temp_dir().join(format!("serve_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let rig = start(net, Some(dir.clone()));

    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    // escape attempts refused
    rig.send(ClientMsg::Eval {
        code: "net.save(\"../escape.ne2\")".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("!")));
    assert!(evs.iter().any(|e| matches!(
        &e.msg,
        ServerMsg::Console(s) if s.contains("path") || s.contains("..") || s.contains("refus") || s.contains("outside")
    )), "expected a path refusal: {evs:?}");
    assert!(!dir.parent().unwrap().join("escape.ne2").exists());

    // in-bounds writes work (print marks the end of the eval)
    rig.send(ClientMsg::Eval {
        code: "net.save(\"ok.ne2\"); print(\"wrote\")".into(),
    });
    rig.until(|m| matches!(m, ServerMsg::Console(s) if s == "wrote"));
    assert!(dir.join("ok.ne2").exists());

    // LoadNet through the sandbox
    rig.send(ClientMsg::LoadNet {
        path: "ok.ne2".into(),
    });
    rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("loaded ok.ne2")));

    // absolute paths refused
    rig.send(ClientMsg::LoadNet {
        path: "/etc/passwd".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("load")));
    assert!(evs.iter().any(|e| matches!(
        &e.msg,
        ServerMsg::Console(s) if s.contains("load /etc/passwd") && !s.contains("loaded")
    )));

    rig.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn second_client_sees_edits() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    net.add_neuron([5.0; 3], net_format::NeuronKind::Excitatory);
    let rig = start(net, None);
    // two logical clients share the one channel — ids multiplex events
    rig.send(ClientMsg::Subscribe);
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    let base = first_geo(&evs).version;
    rig.tx.send((99, ClientMsg::Subscribe)).unwrap();
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(99));

    // client A edits; broadcast delta reaches "both" (to: None)
    rig.send(ClientMsg::Edit {
        base,
        edit: NetEdit::Connect {
            from: 0,
            to: 1,
            w: 1.0,
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)));
    let d = evs
        .iter()
        .find(|e| matches!(e.msg, ServerMsg::GeoDelta(_)))
        .unwrap();
    assert_eq!(d.to, None, "deltas broadcast to all clients");
    rig.stop();
}
