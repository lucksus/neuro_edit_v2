//! Stage 25: the tutorial's step machine driven headlessly through a
//! REAL worker — every advance must come from a real net/sim action
//! (NetEdit, AutoRun, eval, selection→watch), never a "Next" press.
//!
//! Sabotage contract: break one detector (e.g. gate step 2's
//! `inputs[i].targets`/`syn` check behind `false` in tutorial.rs) and
//! this test hangs on the step-3 wait — it must fail, not pass.

use editor::app::{EditorApp, P_SKIPPED};
use editor::simctl::SimCtl;
use eframe::App as _;
use net_format::Network;
use proto::{ClientMsg, NetEdit};
use sim::BackendKind;
use std::time::{Duration, Instant};

fn wait_for<T>(timeout: Duration, mut f: impl FnMut() -> Option<T>) -> T {
    let t0 = Instant::now();
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(t0.elapsed() < timeout, "timed out waiting for worker");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Wait until `app`'s shared Geo sees `n` neurons, then tick and assert
/// the machine advanced to `step`.
fn expect_step(app: &mut EditorApp, step: usize, timeout: Duration) {
    wait_for(timeout, || {
        app.tutorial_tick();
        (app.tut.step == step).then_some(())
    });
    assert_eq!(app.tut.step, step);
}

#[test]
fn tutorial_advances_on_real_actions_only() {
    let ctl = SimCtl::start(Network::new(), BackendKind::Cpu, 1);
    let mut app = EditorApp::new(ctl, None);

    // Boot path: unseen + hydrated empty geo → the tutorial auto-starts.
    wait_for(Duration::from_secs(10), || {
        app.tutorial_boot();
        app.tut.boot_done.then_some(())
    });
    assert!(app.tut_active(), "tutorial must auto-start on an empty net");
    assert_eq!(app.tut.step, 0);

    // step 0 → add a neuron (the same NetEdit the insert tool sends).
    app.ctl.send_edit(NetEdit::AddNeuron {
        pos: [0.0, 0.0, 0.0],
        kind: proto::NeuronKind::Excitatory,
    });
    expect_step(&mut app, 1, Duration::from_secs(10));

    // step 1 → add an input — untargeted, like a Create with an empty
    // selection, so step 2 still has its own lesson.
    app.ctl.send_edit(NetEdit::AddInput {
        name: "in0".into(),
        targets: vec![],
        pos: [30.0, 0.0, 0.0],
    });
    expect_step(&mut app, 2, Duration::from_secs(10));

    // step 2 → wire the input to the neuron (the connect-drag edit).
    app.ctl.send_edit(NetEdit::AddInputTarget {
        name: "in0".into(),
        target: 0,
        w: 20.0,
    });
    expect_step(&mut app, 3, Duration::from_secs(10));

    // step 3 → drive the input and press Run: the detector wants real
    // activity — a reported rate or counted spikes.
    app.ctl.send(ClientMsg::SetInputConst {
        name: "in0".into(),
        value: 1.0,
    });
    app.ctl.send(ClientMsg::AutoRun(true));
    expect_step(&mut app, 4, Duration::from_secs(15));

    // step 4 → grow to ≥5 neurons: four more inserts (a grid works too).
    for i in 1..5 {
        app.ctl.send_edit(NetEdit::AddNeuron {
            pos: [i as f32 * 20.0, 0.0, 0.0],
            kind: proto::NeuronKind::Excitatory,
        });
    }
    expect_step(&mut app, 5, Duration::from_secs(10));

    // step 5 → select a neuron: the selection→watch path is what puts
    // v(t) in the inspector, so the detector watches `watch_sent`.
    wait_for(Duration::from_secs(10), || {
        let g = app.ctl.shared.read().unwrap().geo.clone()?;
        (g.pos.len() >= 5).then_some(g)
    });
    app.geo = app.ctl.shared.read().unwrap().geo.clone();
    app.selection = [2u32].into_iter().collect();
    app.update_watch();
    expect_step(&mut app, 6, Duration::from_secs(10));
    assert!(app.win.console, "step 6 must open the docked console");
    assert!(app.tut.prefilled, "step 6 must prefill the console snippet");

    // step 6 → run the prefilled snippet: the detector is its marker
    // print in the console log — a real eval, not a click.
    let code = app.console.input.clone();
    assert!(
        code.contains("tutorial net built"),
        "prefill must carry the detector marker: {code}"
    );
    app.ctl.send(ClientMsg::Eval { code });
    expect_step(&mut app, 7, Duration::from_secs(15));
    assert!(!app.tut_active(), "step 7 is the done card, not a step");
    assert!(app.tut.show_done, "the done card must show");
    assert_eq!(app.tut.persist, 1, "done must persist");
}

/// Skip from the card: persist 2, full chrome restored, machine off.
#[test]
fn tutorial_skip_persists_and_restores() {
    let ctl = SimCtl::start(Network::new(), BackendKind::Cpu, 1);
    let mut app = EditorApp::new(ctl, None);
    app.tutorial_begin();
    assert!(app.tut_active());
    assert!(!app.win.plots && !app.win.console && !app.win.inspector);
    app.tutorial_end(P_SKIPPED);
    assert!(!app.tut_active());
    assert_eq!(app.tut.persist, 2);
    assert!(app.win.plots && app.win.console && app.win.inspector);
}

/// eframe storage round-trip: a saved `persist` byte keeps the tutorial
/// from auto-starting — done stays done, skipped stays skipped, and a
/// fresh Unseen still boots.
#[test]
fn tutorial_persist_round_trip() {
    struct Mem(std::sync::Mutex<std::collections::HashMap<String, String>>);
    impl eframe::Storage for Mem {
        fn get_string(&self, key: &str) -> Option<String> {
            self.0.lock().unwrap().get(key).cloned()
        }
        fn set_string(&mut self, key: &str, value: String) {
            self.0.lock().unwrap().insert(key.to_string(), value);
        }
        fn flush(&mut self) {}
    }
    let mut mem = Mem(std::sync::Mutex::new(std::collections::HashMap::new()));
    {
        let mut app = EditorApp::new(SimCtl::offline(), Some(&mem));
        app.tutorial_end(P_SKIPPED);
        app.save(&mut mem);
    }
    let mut app2 = EditorApp::new(SimCtl::offline(), Some(&mem));
    assert_eq!(app2.tut.persist, 2, "skipped must round-trip");
    app2.tutorial_boot();
    assert!(!app2.tut_active(), "a skipped tutorial never auto-starts");

    // Unseen still boots once Geo exists — same byte, default value.
    let mut app3 = EditorApp::new(SimCtl::start(Network::new(), BackendKind::Cpu, 1), None);
    wait_for(Duration::from_secs(10), || {
        app3.tutorial_boot();
        app3.tut.boot_done.then_some(())
    });
    assert!(app3.tut_active(), "unseen must auto-start on empty geo");
}
