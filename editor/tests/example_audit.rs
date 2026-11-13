//! Stage 24 audit: every bundled example, run through the editor's real
//! worker path (ClientMsg::Eval → shared snapshots — the same path the
//! UI's Run button uses), must show activity without any input edits.
//! Each example's claim line is checked in the console where the
//! script prints one. The audit table lives in NOTES-for-data.md
//! (## Stage 24).
//!
//! The eval snapshot resets when the post-eval Geo lands, so the
//! activity check is what the researcher actually sees: send the same
//! RunMs the UI's Run button sends, then require the published Activity
//! to show time advancing and spikes arriving. Sabotage per case is on
//! the assertion itself.

use editor::simctl::SimCtl;
use net_format::Network;
use proto::ClientMsg as Cmd;
use sim::BackendKind;
use std::time::{Duration, Instant};

fn wait_for<T>(timeout: Duration, mut f: impl FnMut() -> Option<T>) -> T {
    let t0 = Instant::now();
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(t0.elapsed() < timeout, "timed out waiting for worker");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Run one example through the worker at quick scale; returns the
/// console once the script's end marker prints.
fn run_example(name: &str, src: &str) -> (SimCtl, Vec<String>) {
    std::env::set_var("NEURO_QUICK", "1");
    let ctl = SimCtl::start(Network::new(), BackendKind::Cpu, 1);
    ctl.send(Cmd::Eval {
        code: format!("// audit {name}\n{src}\nprint(\"AUDIT_DONE {name}\");"),
    });
    let out = wait_for(Duration::from_secs(90), || {
        let s = ctl.shared.read().ok()?;
        (s.console
            .iter()
            .any(|l| l.contains(&format!("AUDIT_DONE {name}"))))
        .then(|| s.console.iter().cloned().collect::<Vec<String>>())
    });
    (ctl, out)
}

/// The activity claim every example makes: after the script's eval, the
/// same RunMs the Run button sends must publish an Activity with time
/// advancing and new spikes — nothing needed editing.
/// Sabotage: strip the script's drive/noise tail → spikes_total stays 0.
fn assert_activity(ctl: &SimCtl, name: &str) {
    let t0 = ctl.shared.read().unwrap().snap.time_ms;
    ctl.send(Cmd::RunMs(300.0));
    let snap = wait_for(Duration::from_secs(15), || {
        let s = ctl.shared.read().ok()?;
        (s.snap.time_ms > t0 + 200.0 && s.snap.spikes_total > 0).then(|| s.snap.clone())
    });
    assert!(
        snap.rate_hz > 0.0,
        "{name}: spikes happened but mean rate is 0 — dead liquid?"
    );
}

#[test]
fn every_example_shows_activity_through_the_worker() {
    let claims: &[(&str, &str, &str)] = &[
        // (name, source, console line the claim lives on)
        (
            "first_spikes",
            include_str!("../../examples/first_spikes.rhai"),
            "spikes so far",
        ),
        (
            "grid_wave",
            include_str!("../../examples/grid_wave.rhai"),
            "mean rate",
        ),
        (
            "lsm_readout",
            include_str!("../../examples/lsm_readout.rhai"),
            "test mlp",
        ),
        (
            "lsm_big",
            include_str!("../../examples/lsm_big.rhai"),
            "test mlp",
        ),
        (
            "lsm_big_off",
            include_str!("../../examples/lsm_big_off.rhai"),
            "test mlp",
        ),
        (
            "lsm_adex",
            include_str!("../../examples/lsm_adex.rhai"),
            "test mlp",
        ),
        (
            "lsm_hidden",
            include_str!("../../examples/lsm_hidden.rhai"),
            "ridge_auc",
        ),
        (
            "detector",
            include_str!("../../examples/detector.rhai"),
            "liquid",
        ),
        (
            "memory_window",
            include_str!("../../examples/memory_window.rhai"),
            "horizon=",
        ),
    ];
    for (name, src, needle) in claims {
        let (ctl, out) = run_example(name, src);
        assert_activity(&ctl, name);
        assert!(
            out.iter().any(|l| l.contains(needle)),
            "{name}: claim line {needle:?} missing from console: {out:?}"
        );
        // the lsm demos print held-out numbers — mlp above chance for
        // the signal variants, at chance for the control.
        let auc = |k: &str| -> Option<f64> {
            out.iter()
                .find(|l| l.contains("test mlp"))
                .and_then(|l| l.split(k).nth(1))
                .and_then(|r| r.trim().split(' ').next()?.parse::<f64>().ok())
        };
        match *name {
            "lsm_readout" | "lsm_big" | "lsm_adex" => {
                let a = auc("auc=").unwrap_or(0.0);
                assert!(a > 0.6, "{name}: worker-run mlp auc at chance: {a}");
            }
            "lsm_big_off" => {
                let a = auc("auc=").unwrap_or(1.0);
                assert!(
                    (a - 0.5).abs() < 0.2,
                    "{name}: control mlp auc {a} — pipeline leaks the label"
                );
            }
            _ => {}
        }
        drop(ctl); // SimCtl::drop stops the worker thread
    }
}
