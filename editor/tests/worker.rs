//! Headless integration test of the editor's sim worker:
//! commands in → immutable snapshots out. This is the same path the UI uses.

use editor::simctl::{Cmd, SimCtl};
use net_format::Network;
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

#[test]
fn worker_builds_runs_and_publishes() {
    let ctl = SimCtl::start(Network::new(), BackendKind::Cpu, 1);

    // Build a small liquid via the same Rhai API the console uses.
    ctl.send(Cmd::Eval(
        r#"
        net.create_2d_grid("g", 10, 10, 30.0);
        net.create_connections("g", 0.3, 100.0, 8.0);
        sim.set_noise(200.0, 5.0);
        sim.run(100.0);
        print("done t=" + sim.time());
        "#
        .to_string(),
    ));

    // Geometry snapshot should appear (net_version bumped by the edits).
    let geo = wait_for(Duration::from_secs(10), || {
        let g = ctl.shared.read().ok().and_then(|s| s.geo.clone())?;
        (g.pos.len() == 100).then_some(g)
    });
    assert!(!geo.syn.is_empty(), "expected synapses in geo snapshot");

    // The run must publish snapshots (time advances, spikes counted).
    wait_for(Duration::from_secs(10), || {
        let s = ctl.shared.read().ok()?;
        (s.snap.time_ms > 0.0).then_some(())
    });
    let s = ctl.shared.read().unwrap();
    assert!(
        s.snap.time_ms >= 99.9,
        "time_ms={} should reach ~100",
        s.snap.time_ms
    );
    assert!(
        s.console.iter().any(|l| l.contains("done t=")),
        "print output should reach the console log"
    );
    drop(s);

    // A run_ms command works too.
    ctl.send(Cmd::Eval("sim.run(50.0)".to_string()));
    wait_for(Duration::from_secs(10), || {
        let s = ctl.shared.read().ok()?;
        (s.snap.time_ms >= 149.9).then_some(())
    });

    // Pause/abort path: start a long run then abort — time must stop early.
    ctl.send(Cmd::Eval("sim.run(100000.0)".to_string()));
    std::thread::sleep(Duration::from_millis(80));
    ctl.send(Cmd::Abort);
    std::thread::sleep(Duration::from_millis(200));
    let t_stop = ctl.shared.read().unwrap().snap.time_ms;
    assert!(
        t_stop < 100_000.0,
        "abort should stop the chunked run, time_ms={t_stop}"
    );
}

/// Abort must interrupt even a pure-Rhai infinite loop — no `sim.run`
/// chunks, so only the engine's on_progress hook can see the flag.
#[test]
fn abort_stops_an_infinite_script() {
    use std::sync::atomic::Ordering;

    let ctl = SimCtl::start(Network::new(), BackendKind::Cpu, 1);
    ctl.send(Cmd::Eval("let x = 0; while true { x += 1; }".into()));
    // Give the eval a moment to spin up, then flip the flag exactly like
    // the pause button does via SimCtl::abort_now.
    std::thread::sleep(Duration::from_millis(100));
    ctl.abort.store(true, Ordering::Relaxed);
    ctl.send(Cmd::Abort);

    // busy clears once the eval terminates.
    wait_for(Duration::from_secs(10), || {
        let s = ctl.shared.read().ok()?;
        (!s.busy).then_some(())
    });
    // The console shows the user-facing "stopped by user at line N", not
    // rhai's generic "Script terminated".
    {
        let s = ctl.shared.read().unwrap();
        assert!(
            s.console
                .iter()
                .any(|l| l.contains("stopped by user at line")),
            "expected 'stopped by user at line N' in console, got: {:?}",
            s.console
        );
    }
    // The worker must still be alive: a follow-up eval answers.
    ctl.send(Cmd::Eval("sim.time()".into()));
    wait_for(Duration::from_secs(10), || {
        let s = ctl.shared.read().ok()?;
        s.console.iter().any(|l| l.starts_with("= ")).then_some(())
    });
}

/// Grid groups publish `geo.grids` metadata plus a per-texel spike trace
/// in `snap.grid_traces` — the data path the activity-plane renderer uses.
#[test]
fn worker_publishes_grid_traces() {
    let ctl = SimCtl::start(Network::new(), BackendKind::Cpu, 1);
    ctl.send(Cmd::Eval(
        r#"
        net.create_2d_grid("g", 8, 8, 30.0);
        net.create_connections("g", 0.3, 100.0, 8.0);
        sim.set_noise(500.0, 20.0);
        sim.run(80.0);
        "#
        .to_string(),
    ));
    let (geo, snap) = wait_for(Duration::from_secs(10), || {
        let s = ctl.shared.read().ok()?;
        let g = s.geo.clone()?;
        (s.snap.time_ms >= 79.9 && !s.snap.grid_traces.is_empty()).then_some((g, s.snap.clone()))
    });
    assert_eq!(geo.grids.len(), 1);
    let gs = &geo.grids[0];
    assert_eq!(gs.dims, [8, 8, 1]);
    assert_eq!(gs.members.len(), 64);
    // texel order: member k sits at lattice position k — every member
    // present exactly once.
    let mut sorted = gs.members.clone();
    sorted.sort();
    assert_eq!(sorted, (0..64).collect::<Vec<u32>>());

    let tr = &snap.grid_traces[0];
    assert_eq!(tr.len(), 64);
    assert!(
        tr.iter().any(|&v| v > 0.0),
        "500Hz noise should leave live texels"
    );
}

/// The Inputs panel binds to `snap.input_values` — a `set_input`/`SetInputConst`
/// value must be visible to the UI, not a hardcoded 0.
#[test]
fn worker_publishes_input_values() {
    let ctl = SimCtl::start(Network::new(), BackendKind::Cpu, 1);
    ctl.send(Cmd::Eval(
        r#"
        net.create_2d_grid("g", 4, 4, 30.0);
        net.add_input("src0", [[0, 12.0], [1, 12.0]]);
        sim.set_input("src0", 0.7);
        sim.run(20.0);
        "#
        .to_string(),
    ));
    let vals = wait_for(Duration::from_secs(10), || {
        let s = ctl.shared.read().ok()?;
        let v = s.snap.input_values.clone();
        (s.snap.time_ms >= 19.9 && !v.is_empty()).then_some(v)
    });
    assert_eq!(vals.len(), 1);
    assert!((vals[0] - 0.7).abs() < 1e-6, "input_values={vals:?}");

    // Same via the panel's command path.
    ctl.send(Cmd::SetInputConst {
        name: "src0".into(),
        value: 2.5,
    });
    wait_for(Duration::from_secs(10), || {
        let s = ctl.shared.read().ok()?;
        let v = s.snap.input_values.clone();
        ((v.first().copied().unwrap_or(0.0) - 2.5).abs() < 1e-6).then_some(())
    });
}

/// Same path on the CUDA backend (skipped gracefully if no GPU).
#[test]
fn worker_cuda_smoke() {
    if !sim::cuda_available() {
        eprintln!("no CUDA — skipping");
        return;
    }
    let ctl = SimCtl::start(Network::new(), BackendKind::Cuda, 1);
    ctl.send(Cmd::Eval(
        r#"
        net.create_2d_grid("g", 8, 8, 30.0);
        net.create_connections("g", 0.3, 100.0, 8.0);
        sim.set_noise(200.0, 5.0);
        sim.run(50.0);
        "#
        .to_string(),
    ));
    wait_for(Duration::from_secs(15), || {
        let s = ctl.shared.read().ok()?;
        (s.snap.time_ms >= 49.9 && s.geo.as_ref().is_some_and(|g| g.pos.len() == 64)).then_some(())
    });
}
