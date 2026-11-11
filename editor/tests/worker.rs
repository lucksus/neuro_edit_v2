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
