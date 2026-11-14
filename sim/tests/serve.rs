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
    start_limit(net, data_dir, usize::MAX)
}

fn start_limit(net: Network, data_dir: Option<std::path::PathBuf>, max_upload_bytes: usize) -> Rig {
    start_opts(
        net,
        ServeOpts {
            backend: BackendKind::Cpu,
            seed: 1,
            cfg: SimConfig::default(),
            data_dir,
            max_upload_bytes,
            quota_bytes: 0,
            abort: Arc::new(AtomicBool::new(false)),
            shutdown: Arc::new(AtomicBool::new(false)),
            sim_limits: sim::SimLimits::default(),
        },
    )
}

fn start_opts(net: Network, opts: ServeOpts) -> Rig {
    let (tx, rx) = mpsc::channel::<(u64, ClientMsg)>();
    let (etx, erx) = mpsc::channel::<ServerEvent>();
    let handle = std::thread::spawn(move || {
        let emit: sim::serve::Emit = Rc::new(RefCell::new(move |ev| {
            let _ = etx.send(ev);
        }));
        sim::serve::run(net, opts, rx, emit);
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
    fn until(&self, mut f: impl FnMut(&ServerMsg) -> bool) -> Vec<ServerEvent> {
        self.until_ev(|e| f(&e.msg))
    }

    /// Same, with the event's routing visible to the predicate.
    fn until_ev(&self, mut f: impl FnMut(&ServerEvent) -> bool) -> Vec<ServerEvent> {
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
    rig.send(ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    });
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

/// Stage 13 J: MoveInputs stores the dropped position on the input,
/// emits an `inputs` delta, persists through save/load, and an inverse
/// edit restores the original spot (version-checked "undo").
#[test]
fn move_inputs_moves_persists_and_undoes() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    net.add_neuron([10.0; 3], net_format::NeuronKind::Excitatory);
    net.add_input("stim", vec![(0, 1.0)], [0.0; 3]); // auto-placed
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    let base = first_geo(&evs).version;

    // Drag the input somewhere.
    rig.send(ClientMsg::Edit {
        base,
        edit: NetEdit::MoveInputs {
            items: vec![(0, [40.0, 30.0, -12.0])],
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)));
    let d = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => Some(&**d),
            _ => None,
        })
        .expect("inputs delta");
    let inputs = d.inputs.as_ref().expect("inputs replacement in delta");
    assert_eq!(inputs[0].pos, [40.0, 30.0, -12.0]);
    let base2 = d.version;

    // "Undo" = the inverse version-checked edit puts it back.
    rig.send(ClientMsg::Edit {
        base: base2,
        edit: NetEdit::MoveInputs {
            items: vec![(0, [0.0; 3])],
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)));
    let d = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => Some(&**d),
            _ => None,
        })
        .unwrap();
    assert_eq!(d.inputs.as_ref().unwrap()[0].pos, [0.0; 3]);

    // Out-of-range input index is an error, not a panic.
    rig.send(ClientMsg::Edit {
        base: d.version,
        edit: NetEdit::MoveInputs {
            items: vec![(9, [1.0; 3])],
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(_)));
    assert!(evs
        .iter()
        .any(|e| matches!(&e.msg, ServerMsg::Error(s) if s.contains("no input"))));
    rig.stop();
}

/// The position MoveInputs writes is part of `net.inputs` — which the
/// .ne2 bincode already serializes — so a moved input survives a
/// save/load round-trip.
#[test]
fn moved_input_pos_survives_save_load() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    net.add_input("stim", vec![(0, 1.0)], [0.0; 3]);
    // What the server writes on MoveInputs:
    net.inputs[0].pos = [7.5, -3.0, 21.0];
    let dir = std::env::temp_dir().join(format!("ne13_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("in.ne2");
    net_format::io::save(&net, &p).unwrap();
    let back = net_format::io::load(&p).unwrap();
    assert_eq!(back.inputs[0].pos, [7.5, -3.0, 21.0]);
    std::fs::remove_dir_all(&dir).ok();
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

    // absolute paths refused — the typed Error (Q7 acks), not a
    // console line.
    rig.send(ClientMsg::LoadNet {
        path: "/etc/passwd".into(),
    });
    let evs = rig.until(|m| {
        matches!(&m, ServerMsg::Error(s) if s.contains("load"))
            || matches!(m, ServerMsg::Console(s) if s.contains("load"))
    });
    assert!(evs.iter().any(|e| matches!(
        &e.msg,
        ServerMsg::Error(s) | ServerMsg::Console(s)
            if s.contains("load /etc/passwd") && !s.contains("loaded")
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

#[test]
fn datasets_generate_load_and_stay_confined() {
    let dir = std::env::temp_dir().join(format!("serve_ds_{}_{}", std::process::id(), line!()));
    std::fs::create_dir_all(&dir).unwrap();
    let rig = start(Network::new(), Some(dir.clone()));
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    // generate: 2 s @ 0.5 ms = 4000 steps, 4 channels, a rate event
    // every 100 ms starting at t=100 → 19 event labels.
    rig.send(ClientMsg::Eval {
        code: r#"let d = dataset_gen(#{name:"ev", channels:4, len_ms:2000, dt_ms:0.5,
            dist:"poisson", rate_hz:40, seed:9,
            events:#{kind:"rate", every_ms:100, window_ms:10, strength:5}});
            print("gen n=" + d.events().len() + " ch=" + d.n_channels + " st=" + d.steps)"#
            .into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.starts_with("gen n=")));
    let line = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Console(s) if s.starts_with("gen n=") => Some(s.clone()),
            _ => None,
        })
        .unwrap();
    assert!(line.contains("ch=4 st=4000"), "{line}");
    assert!(line.contains("n=19"), "{line}");
    // cached on disk + reloadable + labels survive a fresh load
    assert!(dir.join("datasets/ev/signal.f32").is_file());
    assert!(dir.join("datasets/ev/labels.json").is_file());
    rig.send(ClientMsg::Eval {
        code: r#"let d = dataset("ev");
            print("again " + d.events().len() + " c0=" + d.channel(0).len())"#
            .into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.starts_with("again ")));
    assert!(
        evs.iter().any(|e| matches!(
            &e.msg,
            ServerMsg::Console(s) if s == "again 19 c0=4000"
        )),
        "{evs:?}"
    );

    // dataset names can't walk out of datasets/
    rig.send(ClientMsg::Eval {
        code: r#"dataset("../x")"#.into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("bad dataset name")));
    assert!(!evs.is_empty());
    // and neither can dataset_gen
    rig.send(ClientMsg::Eval {
        code: r#"dataset_gen(#{name:"../x", channels:1, len_ms:10, seed:1})"#.into(),
    });
    rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("bad dataset name")));

    rig.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn upload_dataset_writes_limits_and_loads() {
    let dir = std::env::temp_dir().join(format!("serve_ul_{}_{}", std::process::id(), line!()));
    std::fs::create_dir_all(&dir).unwrap();
    // 64 B cap — one 4-sample f32 channel fits, 100 B doesn't
    let rig = start_limit(Network::new(), Some(dir.clone()), 64);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    let meta = r#"{"channels":1,"steps":4,"dt_ms":0.5}"#.to_string();
    let labels = r#"{"events_ms":[1.0]}"#.to_string();
    let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

    // over-limit payload refused
    rig.send(ClientMsg::UploadDataset {
        name: "big".into(),
        signal_ext: "f32".into(),
        signal: vec![0u8; 100],
        meta_json: meta.clone(),
        labels_json: labels.clone(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(s) if s.contains("over the")));
    assert!(!evs.is_empty());
    assert!(!dir.join("datasets/big").exists());

    // traversal refused
    rig.send(ClientMsg::UploadDataset {
        name: "../evil".into(),
        signal_ext: "f32".into(),
        signal: f32s(&[1.0; 4]),
        meta_json: meta.clone(),
        labels_json: labels.clone(),
    });
    rig.until(|m| matches!(m, ServerMsg::Error(s) if s.contains("bad dataset name")));
    assert!(!dir.join("evil").exists());

    // legit upload lands and loads through `dataset()`
    rig.send(ClientMsg::UploadDataset {
        name: "tiny".into(),
        signal_ext: "f32".into(),
        signal: f32s(&[1.0, 2.0, 3.0, 4.0]),
        meta_json: meta,
        labels_json: labels,
    });
    rig.until(|m| matches!(m, ServerMsg::FileWritten { path } if path == "datasets/tiny"));
    assert!(dir.join("datasets/tiny/signal.f32").is_file());
    rig.send(ClientMsg::Eval {
        code: r#"let d = dataset("tiny");
            print("up " + d.n_channels + "x" + d.steps + " ev=" + d.events().len()
                  + " v=" + d.channel(0)[2])"#
            .into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.starts_with("up ")));
    assert!(
        evs.iter().any(|e| matches!(
            &e.msg,
            ServerMsg::Console(s) if s == "up 1x4 ev=1 v=3.0"
        )),
        "{evs:?}"
    );

    rig.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn per_synapse_tau_over_wire() {
    // script: 2 neurons, A->B w=30; set tau 50ms on the edge, fire A once,
    // B keeps firing for tens of ms — and syn_tau reads back 50.
    let rig = start(Network::new(), None);
    rig.send(ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    });
    rig.until(|m| matches!(m, ServerMsg::Status { .. }));
    rig.send(ClientMsg::Eval {
        code: r#"net.add_neuron(0,0,0);
            net.add_neuron(50,0,0);
            net.connect(0,1,25.0);
            net.set_syn_tau_one(0,1,50.0);
            net.add_to_group("all", [0,1]);
            net.set_syn_tau("all","all",0.0);   // group-pair setter resets it
            net.set_syn_tau_dist("all","all",60.0,0.0);
            print("tau=" + net.syn_tau(0,1));
            sim.set_external(0,25);
            let t0 = sim.time();
            while sim.spikes() == 0 { sim.run(1); }
            sim.set_external(0,0);
            sim.run(200);
            print("b_spikes=" + (sim.spikes() - 1));
            print("done")"#
            .into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s == "done"));
    let tau = evs.iter().find_map(|e| match &e.msg {
        ServerMsg::Console(s) if s.starts_with("tau=") => Some(s.clone()),
        _ => None,
    });
    assert_eq!(tau.as_deref(), Some("tau=60.0"), "{evs:?}");
    let sp = evs.iter().find_map(|e| match &e.msg {
        ServerMsg::Console(s) if s.starts_with("b_spikes=") => Some(s.clone()),
        _ => None,
    });
    // with a 60 ms trace a single kick fires B repeatedly (the merged
    // 2 ms default barely fires once — cpu_ref covers the comparison)
    let n: i64 = sp.unwrap().trim_start_matches("b_spikes=").parse().unwrap();
    assert!(n >= 2, "expected prolonged firing, got {n}");
    rig.stop();
}

// ---------- Stage 16: crash + security fixes (fable-review-1112) ----------

/// Q1/Q10: a >400-byte Eval whose echo truncation lands inside a
/// multi-byte char must not panic the worker — the line truncates on a
/// char boundary and the worker still answers the next command.
#[test]
fn eval_echo_truncates_on_char_boundary() {
    let rig = start(Network::new(), None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    // Pad to just over 400 bytes with the umlaut straddling byte 400:
    // 399 ASCII bytes + 'ü' (2 bytes, at index 399..401) + more text.
    let pad = "x".repeat(392); // `print("` = 7 bytes → umlaut starts at byte 399
    let code = format!("print(\"{}ü-abcdef\")", pad);
    assert_eq!(code.as_bytes()[399], 0xC3, "umlaut straddles the cut");
    rig.send(ClientMsg::Eval { code });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("truncated")));
    assert!(
        evs.iter().any(|e| matches!(
            &e.msg,
            ServerMsg::Console(s) if s.contains("truncated")
        )),
        "expected a truncated echo: {evs:?}"
    );

    // the worker lives on — the follow-up eval answers
    rig.send(ClientMsg::Eval {
        code: "print(\"still-here\")".into(),
    });
    rig.until(|m| matches!(m, ServerMsg::Console(s) if s == "still-here"));
    rig.stop();
}

/// Q2: InsertSubnetJson carrying a grid with axes [2,1,7] is refused
/// by validate() instead of panicking in `GridShape::world`.
#[test]
fn paste_grid_bad_axes_refused_not_panic() {
    let mut net = Network::new();
    net_format::builders::create_2d_grid(&mut net, "g", 2, 2, 10.0, [0.0; 3]);
    let mut json = net_format::io::to_json(&net).unwrap();
    json = json.replacen("[2,1,0]", "[2,1,7]", 1);
    assert!(json.contains("[2,1,7]"), "grid axes mutated in json");

    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    let base = first_geo(&evs).version;
    rig.send(ClientMsg::Edit {
        base,
        edit: NetEdit::InsertSubnetJson {
            json,
            offset: [0.0; 3],
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(_)));
    assert!(
        evs.iter().any(
            |e| matches!(&e.msg, ServerMsg::Error(s) if s.contains("refused") || s.contains("axes"))
        ),
        "expected an edit refusal, not a dead worker: {evs:?}"
    );
    // worker still alive
    rig.send(ClientMsg::Eval {
        code: "print(\"alive\")".into(),
    });
    rig.until(|m| matches!(m, ServerMsg::Console(s) if s == "alive"));
    rig.stop();
}

/// Q3: SetDtMs and script `sim.set_dt` refuse 0, negative, NaN,
/// infinite and absurd values with an Error — not a silent NaN state.
#[test]
fn set_dt_validated_both_paths() {
    let rig = start(Network::new(), None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    for bad in [0.0f32, -0.1, f32::NAN, f32::INFINITY, 10_000.0] {
        rig.send(ClientMsg::SetDtMs(bad));
        let evs = rig.until(|m| matches!(m, ServerMsg::Error(s) if s.contains("dt")));
        assert!(!evs.is_empty(), "SetDtMs({bad}) must error");
    }
    rig.send(ClientMsg::Eval {
        code: "sim.set_dt(-1)".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("!")));
    assert!(
        evs.iter().any(|e| matches!(
            &e.msg,
            ServerMsg::Console(s) if s.contains("dt")
        )),
        "script set_dt must error too: {evs:?}"
    );
    // a legal value still works
    rig.send(ClientMsg::SetDtMs(0.5));
    rig.send(ClientMsg::Eval {
        code: "print(\"dt-ok\")".into(),
    });
    rig.until(|m| matches!(m, ServerMsg::Console(s) if s == "dt-ok"));
    rig.stop();
}

/// S5: grid builders reject negative dims as errors (not giant usize
/// casts) and refuse texel counts past MAX_GRID_TEXELS.
#[test]
fn grid_dims_validated() {
    let rig = start(Network::new(), None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    for code in [
        "net.create_2d_grid(\"g\", -1, 4, 10.0)",
        "net.create_3d_grid(\"g\", 2, -4, 2, 10.0)",
        "net.create_2d_grid(\"g\", 10000, 10000, 1.0)", // 1e8 > 1<<26
    ] {
        rig.send(ClientMsg::Eval { code: code.into() });
        let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("!")));
        assert!(
            evs.iter().any(|e| matches!(
                &e.msg,
                ServerMsg::Console(s) if s.contains("dims") || s.contains("texels")
            )),
            "{code} must error: {evs:?}"
        );
    }
    rig.send(ClientMsg::Eval {
        code: "print(\"grid-done\")".into(),
    });
    rig.until(|m| matches!(m, ServerMsg::Console(s) if s == "grid-done"));
    rig.stop();
}

/// S5: a net over the server's SimLimits fails the pre-build check
/// with a clear error — no allocation is attempted.
#[test]
fn sim_limits_refuse_oversized_build() {
    let rig = start_opts(
        Network::new(),
        ServeOpts {
            backend: BackendKind::Cpu,
            seed: 1,
            cfg: SimConfig::default(),
            data_dir: None,
            max_upload_bytes: usize::MAX,
            quota_bytes: 0,
            abort: Arc::new(AtomicBool::new(false)),
            shutdown: Arc::new(AtomicBool::new(false)),
            sim_limits: sim::SimLimits {
                max_neurons: 16,
                max_synapses: 0,
                max_ring_bytes: 0,
            },
        },
    );
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::Eval {
        code: "net.create_2d_grid(\"g\", 8, 8, 10.0); sim.run(1)".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("!")));
    assert!(
        evs.iter().any(|e| matches!(
            &e.msg,
            ServerMsg::Console(s) if s.contains("server limit")
        )),
        "expected a size-limit error: {evs:?}"
    );
    rig.stop();
}

/// S7: every write path counts against the quota — WriteFile, script
/// `net.save`, and even Mkdir's directory entry.
#[test]
fn quota_covers_all_write_paths() {
    let dir = std::env::temp_dir().join(format!("serve_q16_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let rig = start_opts(
        Network::new(),
        ServeOpts {
            backend: BackendKind::Cpu,
            seed: 1,
            cfg: SimConfig::default(),
            data_dir: Some(dir.clone()),
            max_upload_bytes: usize::MAX,
            quota_bytes: 3000, // one small file fits; a 100-neuron .ne2 or a dir doesn't
            abort: Arc::new(AtomicBool::new(false)),
            shutdown: Arc::new(AtomicBool::new(false)),
            sim_limits: sim::SimLimits::default(),
        },
    );
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    // WriteFile over quota → refused.
    rig.send(ClientMsg::WriteFile {
        path: "big.txt".into(),
        data: vec![b'x'; 5000],
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(s) if s.contains("quota")));
    assert!(!evs.is_empty());
    assert!(!dir.join("big.txt").exists());

    // script save over quota → refused the same way (a 10x10 grid
    // serializes past 3 KB).
    rig.send(ClientMsg::Eval {
        code: "net.create_2d_grid(\"g\", 10, 10, 10.0); net.save(\"n.ne2\")".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("!")));
    assert!(
        evs.iter().any(|e| matches!(
            &e.msg,
            ServerMsg::Console(s) if s.contains("quota")
        )),
        "script save must hit the quota too: {evs:?}"
    );
    assert!(!dir.join("n.ne2").exists());

    // even an empty directory costs its entry.
    rig.send(ClientMsg::Mkdir {
        path: "a/b/c".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(s) if s.contains("quota")));
    assert!(!evs.is_empty());

    // a write that fits is still fine
    rig.send(ClientMsg::WriteFile {
        path: "small.txt".into(),
        data: b"hi".to_vec(),
    });
    rig.until(|m| matches!(m, ServerMsg::FileWritten { .. }));
    assert!(dir.join("small.txt").exists());
    rig.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

/// S21: `DeleteFile ""`/`"."` must refuse to remove the sandbox root,
/// and `RenameFile` of the root likewise. One bad client call must not
/// wipe the user's whole directory.
#[test]
fn sandbox_root_cannot_be_deleted_or_renamed() {
    let dir = std::env::temp_dir().join(format!("serve_root_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("keep.txt"), b"k").unwrap();
    let rig = start(Network::new(), Some(dir.clone()));
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    for p in ["", "."] {
        rig.send(ClientMsg::DeleteFile { path: p.into() });
        let evs = rig.until(|m| matches!(m, ServerMsg::Error(_)));
        assert!(
            evs.iter()
                .any(|e| matches!(&e.msg, ServerMsg::Error(s) if s.contains("root"))),
            "DeleteFile {p:?} must be refused: {evs:?}"
        );
    }
    rig.send(ClientMsg::RenameFile {
        from: "".into(),
        to: "moved".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(s) if s.contains("root")));
    assert!(!evs.is_empty());
    // root + contents intact, file ops still work
    assert!(dir.join("keep.txt").exists());
    rig.send(ClientMsg::DeleteFile {
        path: "keep.txt".into(),
    });
    rig.until(|m| matches!(m, ServerMsg::FileDeleted { .. }));
    assert!(!dir.join("keep.txt").exists());
    rig.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

/// S22: `UploadChunk` without a prior `UploadStart` is an error, and
/// chunks beyond the declared size are refused — the per-file cap
/// can't be bypassed by lying in `size`.
#[test]
fn upload_chunks_need_start_and_count_bytes() {
    let dir = std::env::temp_dir().join(format!("serve_up16_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let rig = start(Network::new(), Some(dir.clone()));
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    // chunk with no open upload
    rig.send(ClientMsg::UploadChunk {
        path: "sneaky.bin".into(),
        data: vec![1; 8],
        last: false,
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(s) if s.contains("UploadStart")));
    assert!(!evs.is_empty());
    assert!(!dir.join("sneaky.bin").exists());
    assert!(!dir.join("sneaky.bin.upload").exists());

    // declare 8 bytes, send 16 — the running count catches it
    rig.send(ClientMsg::UploadStart {
        path: "honest.bin".into(),
        size: 8,
    });
    rig.until(|m| matches!(m, ServerMsg::UploadReady { .. }));
    rig.send(ClientMsg::UploadChunk {
        path: "honest.bin".into(),
        data: vec![2; 16],
        last: false,
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(s) if s.contains("declared")));
    assert!(!evs.is_empty(), "{evs:?}");

    // a correct upload still lands
    rig.send(ClientMsg::UploadStart {
        path: "ok.bin".into(),
        size: 8,
    });
    rig.until(|m| matches!(m, ServerMsg::UploadReady { .. }));
    rig.send(ClientMsg::UploadChunk {
        path: "ok.bin".into(),
        data: vec![3; 8],
        last: true,
    });
    rig.until(|m| matches!(m, ServerMsg::FileWritten { .. }));
    assert!(dir.join("ok.bin").exists());
    rig.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

/// M3: sim settings persist through save→fresh-session→load. A tau set
/// in one run survives in the file and is adopted on load.
#[test]
fn sim_settings_persist_across_sessions() {
    let dir = std::env::temp_dir().join(format!("serve_m3_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let _ = std::fs::remove_file(dir.join("m3.ne2"));

    let rig = start(Network::new(), Some(dir.clone()));
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::Eval {
        code: "sim.set_synapse_tau(10); sim.set_dt(0.5); net.save(\"m3.ne2\");".into(),
    });
    rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("m3.ne2") || !s.is_empty()));
    rig.stop();

    // the file itself carries the settings
    let loaded = net_format::io::load(dir.join("m3.ne2").as_path()).unwrap();
    assert_eq!(loaded.meta.synapse_tau_ms, 10.0);
    assert_eq!(loaded.meta.dt_ms, 0.5);

    // a fresh session adopts them
    let rig = start(Network::new(), Some(dir.clone()));
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::Eval {
        code: "net.load(\"m3.ne2\"); print(sim.synapse_tau()); print(sim.dt());".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.trim() == "0.5"));
    assert!(
        evs.iter()
            .any(|e| matches!(&e.msg, ServerMsg::Console(s) if s.trim() == "10.0")),
        "loaded tau must become the live config: {evs:?}"
    );
    rig.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Stage 25: a fresh document gets fresh sim settings. Every bundled
/// example opens with `net.clear()`; before this fix the previous run's
/// `sim.set_noise`/`set_dt`/drive survived into it (Data's live check:
/// first_spikes gave 247 spikes after an LSM example instead of the
/// fresh 265). `net.clear()`/`ClientMsg::NewNet` now reset the
/// document-level cfg to defaults and drop the drive stash.
/// Sabotage: delete the `reset_doc_cfg` call in `net.clear()` → the
/// second first_spikes run keeps the leaked 60 Hz noise → counts differ
/// → this fails.
#[test]
fn examples_reset_sim_settings_on_clear() {
    std::env::set_var("NEURO_QUICK", "1");
    let rig = start(Network::new(), None);
    let spikes_of = |tag: &str| -> u64 {
        rig.send(ClientMsg::Eval {
            code: format!(
                "{}\nprint(\"SPK {tag} \" + sim.spikes());",
                include_str!("../../examples/first_spikes.rhai")
            ),
        });
        let evs = rig.until(|m| {
            matches!(m, ServerMsg::Console(l) if l.contains(&format!("SPK {tag}")))
        });
        evs.iter()
            .find_map(|e| match &e.msg {
                ServerMsg::Console(l) => l
                    .split(&format!("SPK {tag} "))
                    .nth(1)
                    .and_then(|t| t.trim().parse::<u64>().ok()),
                _ => None,
            })
            .expect("SPK marker line")
    };
    // first_spikes alone on a fresh worker — the baseline count.
    let fresh = spikes_of("fresh");
    // lsm_readout leaves 60 Hz × 8 mV background noise live as its
    // stay-alive tail — the exact leak the fix removes.
    rig.send(ClientMsg::Eval {
        code: format!(
            "{}\nprint(\"LSM_DONE\");",
            include_str!("../../examples/lsm_readout.rhai")
        ),
    });
    rig.until(|m| matches!(m, ServerMsg::Console(l) if l.contains("LSM_DONE")));
    let after = spikes_of("after");
    assert_eq!(
        fresh, after,
        "first_spikes after a noisy example must equal the fresh run"
    );
    assert!(fresh > 0, "first_spikes produced no spikes at all");
    rig.stop();
}

/// The same reset through the UI's New net: Status.cfg — which the
/// toolbar's noise/dt fields mirror — must read defaults after one.
/// Sabotage: delete the `reset_doc_cfg` in the NewNet arm → the Status
/// keeps noise_rate_hz=60 → fails.
#[test]
fn new_net_resets_sim_settings() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::SetNoise {
        hz: 60.0,
        kick: 8.0,
    });
    rig.send(ClientMsg::SetDtMs(0.5));
    rig.until(|m| matches!(
        m,
        ServerMsg::Status { cfg, .. } if cfg.noise_rate_hz == 60.0 && cfg.dt_ms == 0.5
    ));
    rig.send(ClientMsg::NewNet);
    rig.until(|m| {
        matches!(m, ServerMsg::Status { cfg, .. } if {
            cfg.noise_rate_hz == 0.0
                && cfg.noise_kick_mv == 20.0
                && cfg.dt_ms == net_format::DEFAULT_DT_MS
                && cfg.synapse_tau_ms == net_format::DEFAULT_SYNAPSE_TAU_MS
                && cfg.readout_tau_ms == net_format::DEFAULT_READOUT_TAU_MS
                && !cfg.int_weights
        })
    });
    rig.stop();
}

/// Q4: AddInput/AddReadout with dangling neuron ids are refused — and
/// the refusal doesn't wedge the edit stream (a valid edit after still
/// applies, Q5's server half).
#[test]
fn dangling_ids_refused_then_next_edit_applies() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let rig = start(net, None);
    rig.send(ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    });
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    rig.send(ClientMsg::Edit {
        base: 1,
        edit: NetEdit::AddInput {
            name: "bad".into(),
            targets: vec![(0, 1.0), (999, 1.0)],
            pos: [0.0; 3],
        },
    });
    rig.send(ClientMsg::Edit {
        base: 1,
        edit: NetEdit::AddReadout {
            name: "bad_ro".into(),
            units: vec![0, 12345],
        },
    });
    rig.send(ClientMsg::Edit {
        base: 1,
        edit: NetEdit::AddReadout {
            name: "ok_ro".into(),
            units: vec![0],
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)));
    let errors: Vec<String> = evs
        .iter()
        .filter_map(|e| match &e.msg {
            ServerMsg::Error(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(errors.len(), 2, "both dangling edits refused: {errors:?}");
    assert!(errors.iter().all(|e| e.starts_with("edit refused:")));
    let d = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => Some(d),
            _ => None,
        })
        .expect("the valid edit must still produce a delta");
    assert!(d.readouts.is_some(), "ok_ro landed: {d:?}");
    rig.stop();
}

/// Q6: SaveNet gets a typed ack carrying the written version; failures
/// come back as Errors (never a silent "saved").
/// Q7: LoadNet answers NetLoaded only on success.
#[test]
fn save_and_load_carry_typed_acks() {
    let dir = std::env::temp_dir().join(format!("serve_q67_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let _ = std::fs::remove_file(dir.join("q6.ne2"));
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let rig = start(net, Some(dir.clone()));
    rig.send(ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    });
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    rig.send(ClientMsg::SaveNet {
        path: "q6.ne2".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Saved { .. }));
    let saved = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Saved { path, version } => Some((path.clone(), *version)),
            _ => None,
        })
        .unwrap();
    assert_eq!(saved.0, "q6.ne2");
    assert_eq!(saved.1, 1, "net version 1 written");
    assert!(dir.join("q6.ne2").exists(), "file landed");

    // a save that can't resolve comes back as an Error, not a Saved
    rig.send(ClientMsg::SaveNet {
        path: "../escape/x.ne2".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(_)));
    assert!(evs.iter().any(|e| matches!(
        &e.msg,
        ServerMsg::Error(s) if s.contains("save")
    )));

    // load success → NetLoaded; bogus path → Error, no NetLoaded
    rig.send(ClientMsg::LoadNet {
        path: "q6.ne2".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::NetLoaded { .. }));
    assert!(evs.iter().any(|e| matches!(
        &e.msg,
        ServerMsg::NetLoaded { path, .. } if path == "q6.ne2"
    )));
    rig.send(ClientMsg::LoadNet {
        path: "nope.ne2".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(_)));
    assert!(!evs
        .iter()
        .any(|e| matches!(e.msg, ServerMsg::NetLoaded { .. })));
    rig.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

/// U2-kind: the inspector's kind selector round-trips — SetNeuronKind
/// applies Dale's law server-side and the delta carries the new kind.
#[test]
fn set_neuron_kind_edit_round_trips() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    net.add_neuron([10.0, 0.0, 0.0], net_format::NeuronKind::Excitatory);
    let rig = start(net, None);
    rig.send(ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    });
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    let base = first_geo(&evs).version;

    rig.send(ClientMsg::Edit {
        base,
        edit: NetEdit::SetNeuronKind {
            id: 0,
            kind: proto::NeuronKind::Inhibitory,
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)));
    let d = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => Some(d.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(d.neurons_changed.len(), 1);
    assert_eq!(d.neurons_changed[0].kind, 1, "delta carries inh kind");
    rig.stop();
}

/// 2026-11-13 (Nico, live site): `SetBackend` drops the whole sim — and
/// with it the stimulus program (`SetInputConst`, `PlayInput`,
/// `SetExternal` all live inside the built `Sim`, not the `Env`). After
/// CUDA→CPU no neuron showed excitation, and switching back didn't help
/// because the second rebuild dropped the drive again. The run flag
/// itself survives; the drive must too.
fn backend_switch_case(first: BackendKind, second: BackendKind) {
    let pb = |k: BackendKind| match k {
        BackendKind::Cpu => proto::Backend::Cpu,
        BackendKind::Cuda => proto::Backend::Cuda,
    };
    let mut net = Network::new();
    for i in 0..8 {
        net.add_neuron(
            [i as f32 * 10.0, 0.0, 0.0],
            net_format::NeuronKind::Excitatory,
        );
    }
    net.add_input("stim", vec![(0, 30.0), (4, 30.0)], [0.0; 3]);
    let rig = start_opts(
        net,
        ServeOpts {
            backend: first,
            seed: 1,
            cfg: SimConfig::default(),
            data_dir: None,
            max_upload_bytes: usize::MAX,
            quota_bytes: 0,
            abort: Arc::new(AtomicBool::new(false)),
            shutdown: Arc::new(AtomicBool::new(false)),
            sim_limits: sim::SimLimits::default(),
        },
    );
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::SetInputConst {
        name: "stim".into(),
        value: 1.0,
    });
    rig.send(ClientMsg::AutoRun(true));
    // Spikes on the first backend — sanity that the drive works at all.
    rig.until(|m| matches!(m, ServerMsg::Activity(a) if a.spikes_total > 0));

    // first -> second, then second -> first: after each hop the rebuilt
    // sim must keep spiking without re-setting the input.
    let mut prev = first;
    for to in [second, first] {
        rig.send(ClientMsg::SetBackend(pb(to)));
        // Status only re-emits on change — a same-kind flip produces none
        // — and `status!()` runs at the NEXT loop top, so it can land
        // after the Reset publish. Collect until both are in.
        rig.send(ClientMsg::Reset);
        let mut saw_status = to == prev;
        let mut saw_t0 = false;
        rig.until(|m| {
            match m {
                ServerMsg::Status { backend, .. } if *backend == pb(to) => {
                    saw_status = true;
                }
                ServerMsg::Activity(a) if a.t_steps == 0 => saw_t0 = true,
                _ => {}
            }
            saw_status && saw_t0
        });
        prev = to;
        // The rebuilt sim's counters restart at 0 — spikes now must come
        // from drive carried across the rebuild.
        rig.until(|m| matches!(m, ServerMsg::Activity(a) if a.spikes_total > 0));
    }
    rig.stop();
}

#[test]
fn backend_switch_keeps_drive() {
    // Same-kind flips exercise the same rebuild path even without a GPU.
    backend_switch_case(BackendKind::Cpu, BackendKind::Cpu);
    #[cfg(feature = "cuda")]
    if sim::cuda_available() {
        backend_switch_case(BackendKind::Cpu, BackendKind::Cuda);
        backend_switch_case(BackendKind::Cuda, BackendKind::Cpu);
    } else {
        eprintln!("no CUDA device — exercised the CPU->CPU rebuild path only");
    }
}

/// Same bug through the script API: `sim.set_input` / `sim.set_external`
/// must also survive a backend switch (scripts drive the same Sim).
#[test]
fn backend_switch_keeps_script_drive() {
    // Flip to CUDA when a device is reachable, else CPU (the bug is in
    // the rebuild, not the backend pair).
    #[allow(unused_mut)]
    let mut to = proto::Backend::Cpu;
    #[cfg(feature = "cuda")]
    if sim::cuda_available() {
        to = proto::Backend::Cuda;
    }
    let mut net = Network::new();
    for i in 0..8 {
        net.add_neuron(
            [i as f32 * 10.0, 0.0, 0.0],
            net_format::NeuronKind::Excitatory,
        );
    }
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::Eval {
        code: r#"net.add_input("s", [[0, 30.0]]); sim.set_input("s", 1.0);
                 sim.set_external(4, 30.0); print("driven")"#
            .into(),
    });
    rig.until(|m| matches!(m, ServerMsg::Console(s) if s == "driven"));
    rig.send(ClientMsg::AutoRun(true));
    rig.until(|m| matches!(m, ServerMsg::Activity(a) if a.spikes_total > 0));

    rig.send(ClientMsg::SetBackend(to));
    rig.send(ClientMsg::Reset);
    let mut saw_status = to == proto::Backend::Cpu;
    let mut saw_t0 = false;
    rig.until(|m| {
        match m {
            ServerMsg::Status { backend, .. } if *backend == to => saw_status = true,
            ServerMsg::Activity(a) if a.t_steps == 0 => saw_t0 = true,
            _ => {}
        }
        saw_status && saw_t0
    });
    rig.until(|m| matches!(m, ServerMsg::Activity(a) if a.spikes_total > 0));
    rig.stop();
}

// ---------------------------------------------------------------------------
// Stage 19: the editor's group ops used to arrive as `net.foo(...)` Rhai
// strings on `ClientMsg::Eval`; they are typed `NetEdit`s now. Each test
// sends the edit over the wire and verifies the network actually changed —
// reading weights/tau back via a CopySubnet JSON round-trip where the
// delta itself cannot carry the value.

/// Send an edit, wait for its GeoDelta, return it (fails on Error).
fn edit_delta(rig: &Rig, base: u64, edit: NetEdit) -> Box<proto::GeoDelta> {
    rig.send(ClientMsg::Edit { base, edit });
    let evs =
        rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)) || matches!(m, ServerMsg::Error(_)));
    evs.iter()
        .find_map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => Some(d.clone()),
            ServerMsg::Error(s) => panic!("edit refused: {s}"),
            _ => None,
        })
        .expect("a GeoDelta for the accepted edit")
}

/// Copy `ids` back out as a subnet and parse the JSON.
fn subnet_json(rig: &Rig, ids: &[u32]) -> serde_json::Value {
    serde_json::from_str(&subnet_txt(rig, ids)).expect("subnet json parses")
}

fn subscribe_base(rig: &Rig) -> u64 {
    rig.send(ClientMsg::Subscribe);
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    first_geo(&evs).version
}

#[test]
fn edit_create_connections_wires_group() {
    let mut net = Network::new();
    for i in 0..6 {
        net.add_neuron([i as f32, 0.0, 0.0], net_format::NeuronKind::Excitatory);
    }
    net.add_group("g");
    net.groups[0].members = (0..6).collect();
    let rig = start(net, None);
    let base = subscribe_base(&rig);

    // lambda >> every distance makes the probability saturates at the
    // scaling — 1.0 means every ordered pair connects, deterministically.
    let d = edit_delta(
        &rig,
        base,
        NetEdit::CreateConnections {
            group: "g".into(),
            scaling: 1.0,
            lambda: 1e9,
            w: 1.0,
        },
    );
    assert_eq!(d.syn_added.len(), 6 * 5, "all ordered pairs connected");
    assert_eq!(d.disp_changed.len(), 6);
    assert!(d.version > base);
    rig.stop();
}

#[test]
fn edit_set_synapse_weights_changes_net() {
    let mut net = Network::new();
    for i in 0..3 {
        net.add_neuron([i as f32, 0.0, 0.0], net_format::NeuronKind::Excitatory);
    }
    net.connect(0, 1, 1.0);
    net.connect(0, 2, 1.0);
    net.add_group("g");
    net.groups[0].members = vec![1, 2]; // targets of both synapses
    let rig = start(net, None);
    let base = subscribe_base(&rig);

    // sd 0 ⇒ every draw is exactly the mean.
    let d = edit_delta(
        &rig,
        base,
        NetEdit::SetSynapseWeights {
            group: "g".into(),
            mean: -5.0,
            sd: 0.0,
        },
    );
    assert!(d.version > base);
    let sub = subnet_json(&rig, &[0, 1, 2]);
    let ws: Vec<f64> = sub["synapses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["w"].as_f64().unwrap())
        .collect();
    assert_eq!(
        ws,
        vec![-5.0, -5.0],
        "weights did not reach the net: {ws:?}"
    );
    rig.stop();
}

#[test]
fn edit_set_syn_tau_dist_sets_tau() {
    let mut net = Network::new();
    for i in 0..3 {
        net.add_neuron([i as f32, 0.0, 0.0], net_format::NeuronKind::Excitatory);
    }
    net.connect(0, 1, 1.0);
    net.connect(0, 2, 1.0);
    net.connect(1, 2, 1.0); // untouched: src not in "a"
    net.add_group("a");
    net.groups[0].members = vec![0];
    net.add_group("b");
    net.groups[1].members = vec![1, 2];
    let rig = start(net, None);
    let base = subscribe_base(&rig);

    let d = edit_delta(
        &rig,
        base,
        NetEdit::SetSynTauDist {
            src: "a".into(),
            dst: "b".into(),
            mean: 33.0,
            sd: 0.0,
        },
    );
    assert!(d.version > base);
    let sub = subnet_json(&rig, &[0, 1, 2]);
    let taus: Vec<f64> = sub["synapses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["tau_ms"].as_f64().unwrap())
        .collect();
    assert_eq!(taus, vec![33.0, 33.0, 0.0], "tau did not land: {taus:?}");
    rig.stop();
}

#[test]
fn edit_set_exc_inh_relabels_group() {
    let mut net = Network::new();
    for i in 0..8 {
        net.add_neuron([i as f32, 0.0, 0.0], net_format::NeuronKind::Excitatory);
    }
    net.add_group("g");
    net.groups[0].members = (0..8).collect();
    let rig = start(net, None);
    let base = subscribe_base(&rig);

    // frac 1.0 ⇒ every member inhibitory, deterministically.
    let d = edit_delta(
        &rig,
        base,
        NetEdit::SetExcInh {
            group: "g".into(),
            frac: 1.0,
        },
    );
    assert_eq!(d.neurons_changed.len(), 8);
    assert!(
        d.neurons_changed.iter().all(|r| r.kind == 1),
        "all members must report inhibitory kind"
    );
    rig.stop();
}

#[test]
fn edit_script_add_stores_and_replaces_script() {
    let rig = start(Network::new(), None);
    let base = subscribe_base(&rig);

    let d = edit_delta(
        &rig,
        base,
        NetEdit::ScriptAdd {
            name: "drive".into(),
            src: "sim.run(10)".into(),
        },
    );
    assert_eq!(
        d.scripts.as_deref(),
        Some(&[("drive".to_string(), "sim.run(10)".to_string())][..])
    );

    // same name replaces in place — no duplicate entry.
    let d = edit_delta(
        &rig,
        d.version,
        NetEdit::ScriptAdd {
            name: "drive".into(),
            src: "sim.run(20)".into(),
        },
    );
    assert_eq!(
        d.scripts.as_deref(),
        Some(&[("drive".to_string(), "sim.run(20)".to_string())][..])
    );
    rig.stop();
}

/// Two overlapping groups over 9 neurons with a few synapses — the
/// starting net the group-edit tests share (deterministic, no rng).
fn group_net() -> Network {
    let mut net = Network::new();
    for i in 0..9 {
        net.add_neuron(
            [i as f32, (i % 3) as f32, 0.0],
            net_format::NeuronKind::Excitatory,
        );
    }
    net.add_group("g");
    net.groups[0].members = (0..7).collect();
    net.add_group("h");
    net.groups[1].members = (5..9).collect();
    net.connect(0, 5, 1.0);
    net.connect(0, 6, 1.0);
    net.connect(1, 6, 1.0);
    net
}

/// 19b: the five edits that replaced `format!`-built `Eval`s must
/// still echo the `> net.…` line the console used to show — it's how
/// a dialog teaches the script equivalent.
#[test]
fn typed_edits_echo_script_line() {
    let rig = start(group_net(), None);
    let mut base = subscribe_base(&rig);

    let cases: Vec<(NetEdit, &str)> = vec![
        (
            NetEdit::CreateConnections {
                group: "g".into(),
                scaling: 0.3,
                lambda: 0.1,
                w: 0.5,
            },
            "> net.create_connections(\"g\", 0.3, 0.1, 0.5)",
        ),
        (
            NetEdit::SetSynapseWeights {
                group: "g".into(),
                mean: 0.3,
                sd: 0.1,
            },
            "> net.set_synapse_weights(\"g\", 0.3, 0.1)",
        ),
        (
            NetEdit::SetSynTauDist {
                src: "g".into(),
                dst: "h".into(),
                mean: 0.3,
                sd: 0.1,
            },
            "> net.set_syn_tau_dist(\"g\", \"h\", 0.3, 0.1)",
        ),
        (
            NetEdit::SetExcInh {
                group: "g".into(),
                frac: 0.2,
            },
            "> net.set_exc_inh(\"g\", 0.2)",
        ),
        (
            NetEdit::ScriptAdd {
                name: "drive".into(),
                src: "sim.run(10)".into(),
            },
            "> net.script_add(\"drive\", \"sim.run(10)\")",
        ),
    ];
    for (edit, want) in cases {
        rig.send(ClientMsg::Edit { base, edit });
        let evs =
            rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)) || matches!(m, ServerMsg::Error(_)));
        let d = evs
            .iter()
            .find_map(|e| match &e.msg {
                ServerMsg::GeoDelta(d) => Some(&**d),
                _ => None,
            })
            .expect("edit refused");
        base = d.version;
        // broadcast, like the Eval echo was — every client sees it
        let hit = evs
            .iter()
            .find(|e| matches!(&e.msg, ServerMsg::Console(s) if s == want));
        assert!(hit.is_some(), "missing console echo {want:?}: {evs:?}");
        assert_eq!(hit.unwrap().to, None);
    }
    rig.stop();
}

/// 19b: the typed CreateConnections replaced an Eval — Stop must
/// still interrupt its O(m²) pair loop: busy flips, "stopped by user"
/// reaches the console, partial synapses stay AND reach the client.
#[test]
fn create_connections_edit_stops_on_abort() {
    use std::sync::atomic::Ordering;

    // 100×100 = 10k members → ~100M pair iterations; uncancelled the
    // build takes seconds even on this box.
    let mut net = Network::new();
    net_format::builders::create_2d_grid(&mut net, "g", 100, 100, 10.0, [0.0; 3]);
    let abort = Arc::new(AtomicBool::new(false));
    let rig = start_opts(
        net,
        ServeOpts {
            backend: BackendKind::Cpu,
            seed: 1,
            cfg: SimConfig::default(),
            data_dir: None,
            max_upload_bytes: usize::MAX,
            quota_bytes: 0,
            abort: abort.clone(),
            shutdown: Arc::new(AtomicBool::new(false)),
            sim_limits: sim::SimLimits::default(),
        },
    );
    let base = subscribe_base(&rig);

    // Small scaling: the O(m²) pair loop is the cost, not the kept
    // synapses — the partial delta stays small.
    rig.send(ClientMsg::Edit {
        base,
        edit: NetEdit::CreateConnections {
            group: "g".into(),
            scaling: 0.05,
            lambda: 1e9,
            w: 1.0,
        },
    });
    // Busy must mark the build while it runs (a GeoDelta first means
    // the whole O(m²) loop ran with no busy flag and no Stop).
    let evs = rig.until(|m| {
        matches!(m, ServerMsg::Status { busy: true, .. }) || matches!(m, ServerMsg::GeoDelta(_))
    });
    assert!(
        evs.iter()
            .any(|e| matches!(&e.msg, ServerMsg::Status { busy: true, .. })),
        "build must run under a busy status: {evs:?}"
    );

    // Let a few rows finish so partial work exists — the flag flip
    // must land mid-build, not before row 0.
    std::thread::sleep(Duration::from_millis(100));
    // SimCtl::abort_now(): the shared flag flips first (that's what
    // reaches a worker blocked inside the builder); the queued Abort
    // message cleans up `running` behind it.
    abort.store(true, Ordering::Relaxed);
    rig.send(ClientMsg::Abort);
    let t0 = std::time::Instant::now();
    // Terminal event of the stopped apply: the partial-work delta.
    let evs = rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)));
    let stop_elapsed = t0.elapsed();
    assert!(
        stop_elapsed < Duration::from_secs(10),
        "stop should land near-instantly, took {stop_elapsed:?}"
    );
    assert!(
        evs.iter().any(|e| matches!(
            &e.msg,
            ServerMsg::Console(s)
                if s == "! Runtime error: stopped by user (line 1, position 5)"
        )),
        "expected the Eval's 'stopped by user' line: {evs:?}"
    );
    assert!(
        evs.iter()
            .any(|e| matches!(&e.msg, ServerMsg::Status { busy: false, .. })),
        "busy must clear after the stopped build: {evs:?}"
    );

    let syn_added: usize = evs
        .iter()
        .map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => d.syn_added.len(),
            _ => 0,
        })
        .sum();
    // The worker kept the partial synapses; the client's delta must
    // cover every one of them — no silent divergence.
    rig.send(ClientMsg::Eval {
        code: "net.synapse_count()".into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.starts_with("= ")));
    let worker_syns: usize = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Console(s) => s.strip_prefix("= ")?.parse().ok(),
            _ => None,
        })
        .expect("synapse_count reply");
    assert_eq!(
        syn_added, worker_syns,
        "delta must cover every synapse the worker kept"
    );
    assert!(worker_syns > 0, "partial work kept");
    assert!(
        worker_syns < 100 * 100 * 99,
        "build must have stopped before all pairs were tried"
    );
    rig.stop();
}

/// Copy `ids` back out as a subnet — raw JSON text.
fn subnet_txt(rig: &Rig, ids: &[u32]) -> String {
    rig.send(ClientMsg::CopySubnet { ids: ids.to_vec() });
    let evs = rig.until(|m| matches!(m, ServerMsg::Clipboard(_)));
    evs.iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Clipboard(t) => Some(t.clone()),
            _ => None,
        })
        .expect("a Clipboard reply")
}

/// 19b: a typed group edit must land exactly like the `Eval` string
/// it replaced — same seed, same starting net, so the two sessions
/// end with byte-identical subnet JSON. This is what pins the
/// dialog's f32 → the f64 the Rhai parser produced (`proto::f64_of`)
/// and the identical rng draw order.
#[test]
fn typed_group_edits_match_eval() {
    let all: Vec<u32> = (0..9).collect();
    let cases: Vec<(&str, NetEdit)> = vec![
        (
            "net.create_connections(\"g\", 0.3, 3.0, 0.5)",
            NetEdit::CreateConnections {
                group: "g".into(),
                scaling: proto::f64_of(0.3),
                lambda: proto::f64_of(3.0),
                w: 0.5,
            },
        ),
        (
            "net.set_synapse_weights(\"g\", 0.3, 0.1)",
            NetEdit::SetSynapseWeights {
                group: "g".into(),
                mean: proto::f64_of(0.3),
                sd: proto::f64_of(0.1),
            },
        ),
        (
            "net.set_syn_tau_dist(\"g\", \"h\", 0.3, 0.1)",
            NetEdit::SetSynTauDist {
                src: "g".into(),
                dst: "h".into(),
                mean: proto::f64_of(0.3),
                sd: proto::f64_of(0.1),
            },
        ),
        (
            "net.set_exc_inh(\"g\", 0.2)",
            NetEdit::SetExcInh {
                group: "g".into(),
                frac: proto::f64_of(0.2),
            },
        ),
    ];
    for (code, edit) in cases {
        // CopySubnet is processed after the op either way — the
        // Clipboard reply is the settled net in both sessions.
        let rig_eval = start(group_net(), None);
        rig_eval.send(ClientMsg::Eval { code: code.into() });
        let eval_json = subnet_txt(&rig_eval, &all);

        let rig_edit = start(group_net(), None);
        let base = subscribe_base(&rig_edit);
        rig_edit.send(ClientMsg::Edit { base, edit });
        let edit_json = subnet_txt(&rig_edit, &all);

        assert_eq!(eval_json, edit_json, "`{code}` vs the typed edit diverged");
        rig_eval.stop();
        rig_edit.stop();
    }
}

// ---------------------------------------------------------------------------
// Stage 21 / protocol 2: model descriptors, types on the wire, generic
// watched state series, type/param edits, stale-bundle refusal.
// ---------------------------------------------------------------------------

/// `Hello` must carry the model registry: the editor builds its
/// inspector and plot tabs from these descriptors — a v1 Hello without
/// them is the mis-decode the bump exists to prevent.
#[test]
fn hello_carries_model_descriptors() {
    let rig = start(Network::new(), None);
    rig.send(ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    });
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    let hello = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Hello { models, .. } => Some(models),
            _ => None,
        })
        .expect("a Hello reply");
    assert!(!hello.is_empty(), "registry must not be empty");
    let izh = hello.iter().find(|m| m.id == "izh").expect("izh model");
    // the inspector fields: name/unit/range/default/doc + states + presets
    assert_eq!(izh.params.len(), 4);
    assert_eq!(izh.params[0].name, "a");
    assert!(!izh.state.is_empty(), "izh must expose state slots");
    assert_eq!(izh.state[0].name, "v");
    assert!(izh.presets.iter().any(|(n, _)| n.contains("spiking")));
    rig.stop();
}

/// Stage 22: `Hello` carries izh's parameter design map — the a–b and
/// c–d planes with every named cortical region — and the spot rows are
/// the same data the preset picker applies. Models without a canonical
/// map (AdEx) ship `map: None`.
#[test]
fn hello_carries_param_map() {
    let rig = start(Network::new(), None);
    rig.send(ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    });
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    let hello = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Hello { models, .. } => Some(models),
            _ => None,
        })
        .expect("a Hello reply");
    let izh = hello.iter().find(|m| m.id == "izh").expect("izh model");
    let map = izh.map.as_ref().expect("izh provides a design map");
    assert_eq!(map.planes.len(), 2, "a-b and c-d planes");
    assert_eq!((map.planes[0].x, map.planes[0].y), (0, 1));
    assert_eq!((map.planes[1].x, map.planes[1].y), (2, 3));
    for name in [
        "regular spiking",
        "intrinsically bursting",
        "chattering",
        "fast spiking",
        "low-threshold spiking",
    ] {
        assert!(
            map.planes
                .iter()
                .all(|p| p.spots.iter().any(|s| s.name == name)),
            "map is missing region {name}"
        );
    }
    // A spot's row and its coordinates must agree with the preset list:
    // the marker and the click data are one source of truth.
    let rs = map.planes[0]
        .spots
        .iter()
        .find(|s| s.name == "regular spiking")
        .unwrap();
    assert_eq!(
        izh.presets
            .iter()
            .find(|(n, _)| n == "regular spiking")
            .map(|(_, p)| p.as_slice()),
        Some(rs.params.as_slice()),
        "map row must be the preset row"
    );
    assert!((rs.params[0] - 0.02).abs() < 1e-6 && (rs.params[1] - 0.2).abs() < 1e-6);
    let adex = hello.iter().find(|m| m.id == "adex").expect("adex model");
    assert!(adex.map.is_none(), "AdEx has no canonical map");
    rig.stop();
}

/// `Geo` carries the type table, per-neuron `ty`, and overrides.
#[test]
fn geo_carries_types_and_overrides() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let b = net.add_neuron([5.0; 3], net_format::NeuronKind::Excitatory);
    net.set_izh(b, 0.03, 0.25, -60.0, 4.0); // per-neuron override
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    let geo = first_geo(&evs);
    assert_eq!(geo.ty.len(), 2);
    assert!(geo
        .types
        .iter()
        .any(|t| t.name == "exc" && t.model == "izh" && t.kind == proto::NeuronKind::Excitatory));
    assert_eq!(geo.params.len(), 2);
    assert!(geo.params[0].is_none(), "untyped neuron uses the type row");
    assert_eq!(
        geo.params[1].as_deref(),
        Some(&[0.03f32, 0.25, -60.0, 4.0][..]),
        "the override must ride Geo.params"
    );
    rig.stop();
}

/// `Activity.watched` carries one series per state slot per watched
/// neuron — slot 0 v, slot 1 u for izh (the old watched_v/watched_u).
#[test]
fn activity_watched_carries_state_slots() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    net.add_input("i", vec![(0, 1.0)], [0.0; 3]);
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::Watch(vec![0]));
    rig.send(ClientMsg::SetInputConst {
        name: "i".into(),
        value: 5.0,
    });
    rig.send(ClientMsg::RunMs(10.0));
    let evs = rig.until(|m| {
        matches!(m, ServerMsg::Activity(a) if a.t_steps > 0 && !a.watched.is_empty()
            && a.watched.iter().all(|(_, s)| !s.is_empty() && !s[0].is_empty()))
    });
    let a = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Activity(a) => Some(&**a),
            _ => None,
        })
        .unwrap();
    let (id, series) = &a.watched[0];
    assert_eq!(*id, 0);
    assert!(
        series.len() >= 2,
        "izh watches must expose >=2 state slots (v,u): {series:?}"
    );
    // the recorded v must actually be in the membrane working range —
    // proves the series carries state slot 0, not padding.
    assert!(series[0].iter().all(|&v| v > -100.0 && v < 100.0));
    rig.stop();
}

/// A wrong-version Hello gets an Error explaining the reload — and
/// nothing else. The old code sent a full hydration a stale client
/// would mis-decode; the client must also not be able to un-refuse
/// itself (a second Hello/Subscribe is dropped).
#[test]
fn stale_proto_hello_is_refused_then_dropped() {
    let rig = start(Network::new(), None);
    rig.send(ClientMsg::Hello { proto: 1 });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(_)));
    // Startup broadcasts (Status/Geo to `None`) may precede it — the
    // guarantee is: one Error to the client, and NO hydration frame
    // addressed to it.
    let to_client: Vec<_> = evs.iter().filter(|e| e.to == Some(CLIENT)).collect();
    assert_eq!(
        to_client.len(),
        1,
        "exactly the error must reach the refused client: {evs:?}"
    );
    let msg = match &to_client[0].msg {
        ServerMsg::Error(e) => e.clone(),
        _ => unreachable!(),
    };
    assert!(
        msg.contains("protocol mismatch") && msg.contains("reload the page"),
        "{msg}"
    );
    // The refused client stays refused: a follow-up Subscribe is dropped
    // (no Geo arrives within a quiet window).
    rig.send(ClientMsg::Subscribe);
    let later = rig.drain(300);
    assert!(
        later
            .iter()
            .all(|e| !matches!(e.msg, ServerMsg::Geo(_) | ServerMsg::Hello { .. })),
        "refused client must not be hydrated later: {later:?}"
    );
    // And a later net-version bump must stamp the refused id onto the
    // broadcast Geo's `except` — the socket layer skips those clients,
    // so the stale bundle can never be offered a frame it can't parse.
    let other = CLIENT + 1;
    rig.tx
        .send((
            other,
            ClientMsg::Hello {
                proto: proto::PROTO_VERSION,
            },
        ))
        .unwrap();
    rig.tx
        .send((
            other,
            ClientMsg::Eval {
                code: "net.add_neuron(0.0, 0.0, 0.0)".into(),
            },
        ))
        .unwrap();
    let evs = rig.until_ev(|e| e.to.is_none() && matches!(e.msg, ServerMsg::Geo(_)));
    let geo = evs
        .iter()
        .rev()
        .find(|e| e.to.is_none() && matches!(e.msg, ServerMsg::Geo(_)))
        .expect("the eval's republished Geo");
    assert_eq!(geo.to, None, "republish is a broadcast");
    assert!(
        geo.except.contains(&CLIENT),
        "refused client must be excluded from the broadcast: {geo:?}"
    );
    rig.stop();
}

/// UpsertType lands a type row in the delta (`types` = full table).
#[test]
fn upsert_type_edit_carries_types() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let rig = start(net, None);
    let base = subscribe_base(&rig);
    rig.send(ClientMsg::Edit {
        base,
        edit: NetEdit::UpsertType {
            name: "fs".into(),
            model: "izh".into(),
            params: vec![0.1, 0.2, -65.0, 2.0],
            kind: proto::NeuronKind::Inhibitory,
            color: Some([0.2, 0.9, 0.4]),
        },
    });
    let evs =
        rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)) || matches!(m, ServerMsg::Error(_)));
    let d = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => Some(&**d),
            _ => None,
        })
        .expect("a GeoDelta");
    let types = d.types.as_ref().expect("types table in the delta");
    let fs = types.iter().find(|t| t.name == "fs").expect("fs row");
    assert_eq!(fs.kind, proto::NeuronKind::Inhibitory);
    assert_eq!(fs.color, Some([0.2, 0.9, 0.4]));
    rig.stop();
}

/// SetNeuronType re-points neurons (clears overrides); SetNeuronParams
/// sets and clears the override row; SetSynapse edits w/tau/delay.
#[test]
fn type_and_param_edits_round_trip() {
    let mut net = Network::new();
    let a = net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let b = net.add_neuron([60.0; 3], net_format::NeuronKind::Excitatory);
    net.connect(a, b, 1.0);
    let rig = start(net, None);
    let base = subscribe_base(&rig);

    // define a type, then point neuron b at it — override cleared.
    rig.send(ClientMsg::Edit {
        base,
        edit: NetEdit::UpsertType {
            name: "fs".into(),
            model: "izh".into(),
            params: vec![0.1, 0.2, -65.0, 2.0],
            kind: proto::NeuronKind::Inhibitory,
            color: None,
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)));
    let v = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => Some(d.version),
            _ => None,
        })
        .unwrap();
    rig.send(ClientMsg::Edit {
        base: v,
        edit: NetEdit::SetNeuronType {
            ids: vec![b],
            ty: "fs".into(),
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)));
    let d = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => Some(&**d),
            _ => None,
        })
        .unwrap();
    let row = d
        .neurons_changed
        .iter()
        .find(|r| r.id == b)
        .expect("b's row in the delta");
    assert_eq!(row.kind, 1, "b must render inhibitory now");
    assert!(row.params.is_none(), "type change clears the override");

    // SetNeuronParams writes an override row of the type's width.
    rig.send(ClientMsg::Edit {
        base: d.version,
        edit: NetEdit::SetNeuronParams {
            ids: vec![b],
            params: Some(vec![0.1, 0.2, -60.0, 2.0]),
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::GeoDelta(_)));
    let d = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => Some(&**d),
            _ => None,
        })
        .unwrap();
    let row = d.neurons_changed.iter().find(|r| r.id == b).unwrap();
    assert_eq!(row.params.as_deref(), Some(&[0.1f32, 0.2, -60.0, 2.0][..]));

    // A wrong-width override is refused.
    rig.send(ClientMsg::Edit {
        base: d.version,
        edit: NetEdit::SetNeuronParams {
            ids: vec![b],
            params: Some(vec![1.0, 2.0]),
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Error(_)));
    assert!(evs
        .iter()
        .any(|e| matches!(&e.msg, ServerMsg::Error(s) if s.contains("params"))));

    // SetSynapse: w+tau land (delay is wire-visible via syn_delays).
    rig.send(ClientMsg::Edit {
        base: d.version,
        edit: NetEdit::SetSynapse {
            from: a,
            to: b,
            w: Some(3.0),
            tau_ms: Some(5.0),
            delay_ms: Some(4.0),
        },
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::GeoDelta(_) | ServerMsg::Error(_)));
    let d = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::GeoDelta(d) => Some(&**d),
            _ => None,
        })
        .expect("a GeoDelta");
    assert!(
        !d.syn_delays.is_empty(),
        "a delay_ms change must ride syn_delays"
    );
    rig.stop();
}

/// The review-1.3e script surface end-to-end through the wire: `models`,
/// `def_type`, `set_type`, `set_param`, `type_of` — and the type table
/// reaches the client's Geo.
#[test]
fn script_type_api_through_eval() {
    let mut net = Network::new();
    for i in 0..3 {
        net.add_neuron([i as f32, 0.0, 0.0], net_format::NeuronKind::Excitatory);
    }
    net.add_group("g");
    net.groups[0].members = vec![0, 1, 2];
    let rig = start(net, None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::Eval {
        code: r#"
        let ms = models();
        net.def_type("fs", "izh", #{a: 0.1, d: 2.0}, "inh");
        net.set_type("g", "fs");
        net.set_param(1, "d", 7.0);
        print("type api ok " + ms.len() + " " + net.type_name(0) + " " + ms[0].params.len())
        "#
        .into(),
    });
    let evs = rig.until(|m| matches!(m, ServerMsg::Geo(_)) || matches!(m, ServerMsg::Error(_)));
    assert!(
        evs.iter().any(|e| matches!(&e.msg,
            ServerMsg::Console(s) if s.contains("type api ok 2 fs 4"))),
        "script must have run (2 models, type_name==fs, 4 izh params): {evs:?}"
    );
    let geo = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Geo(g) => Some(&**g),
            _ => None,
        })
        .expect("script edits republish a Geo");
    let fs_i = geo
        .types
        .iter()
        .position(|t| t.name == "fs")
        .expect("fs in the wire table");
    assert_eq!(geo.types[fs_i].kind, proto::NeuronKind::Inhibitory);
    assert!(geo.ty.iter().all(|&t| t as usize == fs_i));
    assert_eq!(
        geo.params[1].as_deref().map(|p| p[3]),
        Some(7.0),
        "set_param must land as an override"
    );
    rig.stop();
}

// ---------------------------------------------------------------------------
// Stage 23: shared network undo/redo — worker-owned snapshot stacks, action
// labels in Status, one step per eval, one stack per run across clients.
// ---------------------------------------------------------------------------

/// Save the live net through the protocol and read the bytes back —
/// `data_dir` is None so the path is used verbatim.
fn saved_bytes(rig: &Rig, path: &str) -> Vec<u8> {
    rig.send(ClientMsg::SaveNet { path: path.into() });
    let evs = rig.until(|m| {
        matches!(m, ServerMsg::Saved { path: p, .. } if p == path)
            || matches!(m, ServerMsg::Error(_))
    });
    for e in &evs {
        if let ServerMsg::Error(err) = &e.msg {
            panic!("save {path}: {err}");
        }
    }
    std::fs::read(path).expect("saved .ne2 readable")
}

/// Next Geo or GeoDelta → the new net_version.
fn next_version(rig: &Rig) -> u64 {
    let evs = rig.until(|m| matches!(m, ServerMsg::Geo(_) | ServerMsg::GeoDelta(_)));
    evs.iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Geo(g) => Some(g.version),
            ServerMsg::GeoDelta(d) => Some(d.version),
            _ => None,
        })
        .expect("a publish")
}

/// Next full Geo (undo republishes whole geometry).
fn next_geo(rig: &Rig) -> proto::Geo {
    let evs = rig.until(|m| matches!(m, ServerMsg::Geo(_)));
    match &evs.last().unwrap().msg {
        ServerMsg::Geo(g) => (**g).clone(),
        _ => unreachable!(),
    }
}

fn tmp_path(tag: &str) -> String {
    let d = std::env::temp_dir().join(format!("ne_undo_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(tag).to_string_lossy().into_owned()
}

/// A net rich enough that every `NetEdit` variant has a valid target:
/// the group_net fixture plus an input, a readout, a samples node and a
/// stored script.
fn rich_net() -> Network {
    let mut net = group_net();
    net.upsert_type(net_format::NeuronType {
        name: "fs".into(),
        model: "izh".into(),
        params: vec![0.1, 0.25, -60.0, 4.0],
        kind: net_format::NeuronKind::Inhibitory,
        color: None,
    });
    net.add_input("kick", vec![(0, 1.0)], [0.0, 10.0, 0.0]);
    net.add_readout("ro", vec![0, 1]);
    net.samples_mut("sp").values = vec![0.5, 1.0];
    net.samples_mut("sp").times = vec![0.0, 1.0];
    net.scripts.push(("main".into(), "sim.run(1)".into()));
    net
}

fn subnet_json_for(ids: &[u32]) -> String {
    let set: std::collections::HashSet<u32> = ids.iter().copied().collect();
    let sub = rich_net().subnet(&set);
    net_format::io::to_json(&sub).unwrap()
}

/// Every `NetEdit` variant: apply → undo → byte-identical `.ne2` to the
/// pre-edit file; redo → byte-identical to the post-edit file. Snapshots
/// make the round trip exact even for rng-driven builders.
/// Sabotage: dropping `push_step` in the Edit arm (or pushing the
/// post-edit snapshot) leaves Undo with nothing/wrong to restore — the
/// `undone` bytes stop matching `before` for every case.
#[test]
fn undo_redo_round_trip_all_edit_kinds() {
    let k = proto::NeuronKind::Inhibitory;
    let cases: Vec<(&str, NetEdit)> = vec![
        (
            "add",
            NetEdit::AddNeuron {
                pos: [1.0; 3],
                kind: proto::NeuronKind::Excitatory,
            },
        ),
        ("del", NetEdit::DeleteNeurons { ids: vec![8] }),
        (
            "conn",
            NetEdit::Connect {
                from: 8,
                to: 0,
                w: 1.0,
            },
        ),
        ("disc", NetEdit::Disconnect { from: 0, to: 5 }),
        (
            "move",
            NetEdit::MoveNeurons {
                items: vec![(0, [9.0; 3]), (1, [8.0; 3])],
            },
        ),
        (
            "name",
            NetEdit::SetNeuronName {
                id: 0,
                name: "x".into(),
            },
        ),
        (
            "pos",
            NetEdit::SetNeuronPos {
                id: 0,
                pos: [3.0; 3],
            },
        ),
        (
            "kind",
            NetEdit::SetNeuronKind {
                id: 0,
                kind: proto::NeuronKind::Regular,
            },
        ),
        (
            "paste",
            NetEdit::InsertSubnetJson {
                json: subnet_json_for(&[0, 1]),
                offset: [30.0; 3],
            },
        ),
        (
            "grid2d",
            NetEdit::CreateGrid2d {
                group: "sheet".into(),
                w: 2,
                h: 2,
                spacing: 10.0,
                origin: [0.0; 3],
            },
        ),
        (
            "grid3d",
            NetEdit::CreateGrid3d {
                group: "vol".into(),
                w: 2,
                h: 2,
                d: 2,
                spacing: 10.0,
                origin: [0.0; 3],
            },
        ),
        (
            "input",
            NetEdit::AddInput {
                name: "in2".into(),
                targets: vec![(1, 1.0)],
                pos: [1.0, 2.0, 0.0],
            },
        ),
        ("samples", NetEdit::AddSamples { name: "sp2".into() }),
        (
            "readout",
            NetEdit::AddReadout {
                name: "ro2".into(),
                units: vec![0, 3],
            },
        ),
        (
            "movein",
            NetEdit::MoveInputs {
                items: vec![(0, [5.0, 9.0, 0.0])],
            },
        ),
        (
            "togroup",
            NetEdit::AddToGroup {
                name: "g".into(),
                ids: vec![7, 8],
            },
        ),
        (
            "intarget",
            NetEdit::AddInputTarget {
                name: "kick".into(),
                target: 4,
                w: 2.0,
            },
        ),
        (
            "conns",
            NetEdit::CreateConnections {
                group: "g".into(),
                scaling: proto::f64_of(0.3),
                lambda: proto::f64_of(3.0),
                w: 0.5,
            },
        ),
        (
            "weights",
            NetEdit::SetSynapseWeights {
                group: "g".into(),
                mean: proto::f64_of(0.3),
                sd: proto::f64_of(0.1),
            },
        ),
        (
            "tau",
            NetEdit::SetSynTauDist {
                src: "g".into(),
                dst: "h".into(),
                mean: proto::f64_of(0.3),
                sd: proto::f64_of(0.1),
            },
        ),
        (
            "excinh",
            NetEdit::SetExcInh {
                group: "g".into(),
                frac: proto::f64_of(0.2),
            },
        ),
        (
            "script",
            NetEdit::ScriptAdd {
                name: "s2".into(),
                src: "print(1)".into(),
            },
        ),
        (
            "type",
            NetEdit::UpsertType {
                name: "fs2".into(),
                model: "izh".into(),
                params: vec![0.1, 0.2, -65.0, 2.0],
                kind: k,
                color: None,
            },
        ),
        (
            "settype",
            NetEdit::SetNeuronType {
                ids: vec![0],
                ty: "fs".into(),
            },
        ),
        (
            "setparams",
            NetEdit::SetNeuronParams {
                ids: vec![0],
                params: Some(vec![0.02, 0.25, -55.0, 4.0]),
            },
        ),
        (
            "setsyn",
            NetEdit::SetSynapse {
                from: 0,
                to: 5,
                w: Some(2.5),
                tau_ms: Some(3.0),
                delay_ms: Some(4.0),
            },
        ),
    ];
    for (tag, edit) in cases {
        let rig = start(rich_net(), None);
        let base = subscribe_base(&rig);
        let before_p = tmp_path(&format!("{tag}_before.ne2"));
        let after_p = tmp_path(&format!("{tag}_after.ne2"));
        let undone_p = tmp_path(&format!("{tag}_undone.ne2"));
        let redone_p = tmp_path(&format!("{tag}_redone.ne2"));

        let before = saved_bytes(&rig, &before_p);
        rig.send(ClientMsg::Edit {
            base,
            edit: edit.clone(),
        });
        let v_edit = next_version(&rig);
        assert_eq!(v_edit, base + 1, "{tag}: edit must bump net_version");
        let after = saved_bytes(&rig, &after_p);
        assert_ne!(before, after, "{tag}: edit changed nothing?");

        rig.send(ClientMsg::Undo);
        let g = next_geo(&rig);
        assert_eq!(g.version, base + 2, "{tag}: undo bumps net_version");
        let undone = saved_bytes(&rig, &undone_p);
        assert_eq!(
            undone, before,
            "{tag}: undo must restore the pre-edit net byte-for-byte"
        );

        rig.send(ClientMsg::Redo);
        let g = next_geo(&rig);
        assert_eq!(g.version, base + 3, "{tag}: redo bumps net_version");
        let redone = saved_bytes(&rig, &redone_p);
        assert_eq!(
            redone, after,
            "{tag}: redo must restore the post-edit net byte-for-byte"
        );
        rig.stop();
    }
}

/// One eval that mutates the net several times is ONE undo step — the
/// Status label names the eval, and a single Undo reverts everything it
/// did.
/// Sabotage: recording a step per script mutation (or none) — either the
/// label isn't the eval's, or a second Undo still has steps left.
#[test]
fn eval_with_many_edits_is_one_undo_step() {
    let rig = start(Network::new(), None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));

    rig.send(ClientMsg::Eval {
        code: r#"
            net.create_2d_grid("g", 4, 4, 8.0);
            net.create_connections("g", 0.5, 10.0, 1.0);
            net.set_exc_inh("g", 0.2);
            net.add_input("kick", [[0, 1.0]]);
        "#
        .into(),
    });
    // The eval republishes a Geo — collecting through it also captures
    // the post-eval Status that names the undo step.
    let evs = rig.until(|m| matches!(m, ServerMsg::Geo(_)));
    let label = evs.iter().find_map(|e| match &e.msg {
        ServerMsg::Status { undo: Some(u), .. } => Some(u.clone()),
        _ => None,
    });
    assert!(
        label.as_deref().is_some_and(|l| l.starts_with("eval ")),
        "undo top must name the eval, got {label:?}"
    );
    let geo = evs
        .iter()
        .find_map(|e| match &e.msg {
            ServerMsg::Geo(g) => Some((**g).clone()),
            _ => None,
        })
        .expect("eval republished geometry");
    assert_eq!(geo.pos.len(), 16);
    let v = geo.version;

    rig.send(ClientMsg::Undo);
    let g = next_geo(&rig);
    assert_eq!(g.version, v + 1);
    assert_eq!(g.pos.len(), 0, "one undo must revert the whole eval");

    // The eval was one step — nothing else to undo.
    rig.send(ClientMsg::Undo);
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("nothing to undo")));
    assert!(!evs.is_empty());

    // And redo restores it all in one step too.
    rig.send(ClientMsg::Redo);
    let g = next_geo(&rig);
    assert_eq!(g.pos.len(), 16);
    rig.stop();
}

/// Two clients share ONE undo stack: the net is shared, so Ctrl+Z pops
/// the last net change whoever sent it. B's undo reverts A's edit when
/// A's edit is on top — and A's undo reverts B's edit when B's is.
/// Sabotage: a per-client stack would undo the wrong entry here — the
/// second undo asserts the GLOBAL top popped.
#[test]
fn undo_is_one_shared_stack_across_clients() {
    const A: u64 = 7;
    const B: u64 = 9;
    let rig = start(Network::new(), None);

    // Both clients hydrate.
    for id in [A, B] {
        rig.tx.send((id, ClientMsg::Subscribe)).unwrap();
        rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(id));
    }
    rig.drain(50);

    // A adds a neuron.
    rig.tx
        .send((
            A,
            ClientMsg::Edit {
                base: 1,
                edit: NetEdit::AddNeuron {
                    pos: [1.0; 3],
                    kind: proto::NeuronKind::Excitatory,
                },
            },
        ))
        .unwrap();
    let v1 = next_version(&rig);
    assert_eq!(v1, 2);

    // B moves it — B's edit lands on top of A's.
    rig.tx
        .send((
            B,
            ClientMsg::Edit {
                base: v1,
                edit: NetEdit::MoveNeurons {
                    items: vec![(0, [42.0; 3])],
                },
            },
        ))
        .unwrap();
    let v2 = next_version(&rig);
    assert_eq!(v2, 3);

    // A undoes — the GLOBAL top (B's move) pops, not A's add. The
    // broadcast Geo reaches every subscriber, not just A.
    rig.tx.send((A, ClientMsg::Undo)).unwrap();
    let evs = rig.until_ev(|e| matches!(e.msg, ServerMsg::Geo(_)));
    let geo_ev = evs
        .iter()
        .find(|e| matches!(e.msg, ServerMsg::Geo(_)))
        .unwrap();
    assert_eq!(geo_ev.to, None, "undo republishes to all clients");
    let g = first_geo(&evs);
    assert_eq!(g.pos.len(), 1, "A's neuron survived the undo of B's move");
    assert_eq!(g.pos[0], [1.0; 3], "B's move was the step that reverted");

    // B undoes next — now A's add (the only step left) pops.
    rig.tx.send((B, ClientMsg::Undo)).unwrap();
    let g = next_geo(&rig);
    assert_eq!(g.pos.len(), 0, "the second undo pops A's add");
    rig.stop();
}

/// Empty stacks answer the requester only — no broadcast noise, no panic.
#[test]
fn undo_redo_empty_stack_answers_caller() {
    let rig = start(Network::new(), None);
    rig.send(ClientMsg::Subscribe);
    rig.until_ev(|e| matches!(e.msg, ServerMsg::Status { .. }) && e.to == Some(CLIENT));
    rig.send(ClientMsg::Undo);
    let evs = rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("nothing to undo")));
    assert_eq!(
        evs.last().unwrap().to,
        Some(CLIENT),
        "empty-stack notice is for the requester"
    );
    rig.send(ClientMsg::Redo);
    rig.until(|m| matches!(m, ServerMsg::Console(s) if s.contains("nothing to redo")));
    rig.stop();
}
