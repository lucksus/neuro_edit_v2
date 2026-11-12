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
    rig.send(ClientMsg::Hello { proto: 1 });
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
