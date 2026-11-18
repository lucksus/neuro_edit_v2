//! Stage 24 audit: every bundled example, run through the editor's real
//! worker path (ClientMsg::Eval → shared snapshots — the same path the
//! UI's Run button uses), must show activity without any input edits.
//! Each example's claim line is checked in the console where the
//! script prints one. The audit table lives in NOTES-for-data.md
//! (## Stage 24/24d).
//!
//! The eval snapshot resets when the post-eval Geo lands, so the
//! activity check is what the researcher actually sees: send the same
//! RunMs the UI's Run button sends, then require the published Activity
//! to show time advancing and spikes arriving — in the LAST second of
//! a multi-second run, not just the first publish (Stage 24d). Sabotage
//! per case is on the assertion itself.

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
            // the printed line, not the console's echo of the code
            .any(|l| l.trim() == format!("AUDIT_DONE {name}")))
        .then(|| s.console.iter().cloned().collect::<Vec<String>>())
    });
    (ctl, out)
}

/// Wait until published sim time is ≥ `min_t` and unchanged for 300 ms;
/// returns (time, spikes_total) at rest.
fn settled(ctl: &SimCtl, min_t: f64) -> (f64, u64) {
    let mut last = (f64::NAN, 0u64);
    let mut since = Instant::now();
    wait_for(Duration::from_secs(60), || {
        let s = ctl.shared.read().ok()?;
        let now = (s.snap.time_ms, s.snap.spikes_total);
        if now.0 != last.0 {
            last = now;
            since = Instant::now();
        }
        (now.0 >= min_t && since.elapsed() > Duration::from_millis(300)).then_some(now)
    })
}

/// The activity claim every example makes: after the script's eval, the
/// same RunMs the Run button sends must publish an Activity with time
/// advancing and new spikes — and it must STILL be spiking in the last
/// second of a several-second run, not only inside the first 300 ms
/// (Stage 24d: the AdEx liquid used to die ~1 s after load while this
/// audit still passed on its early spikes).
/// Sabotage: strip the script's drive/noise tail (e.g. AdEx tail
/// `set_noise(60.0, 0.0)`) → spikes_total stalls and this fails.
fn assert_activity(ctl: &SimCtl, name: &str) {
    let t0 = ctl.shared.read().unwrap().snap.time_ms;
    // A few seconds of Run, the way a user watches it go.
    ctl.send(Cmd::RunMs(4000.0));
    // The post-eval snapshot reads t = 0 although the script itself ran
    // the sim (first_spikes: 2 s), so the first publish after RunMs can
    // already sit past t0 + 3900 — and the "last second" wait below was
    // satisfied by that same publish (Stage 26 found the audit failing
    // on main this way). Wait for the run to land AND settle, then
    // measure the last second from the settled time.
    let (t1, spikes_at_4s) = settled(ctl, t0 + 3900.0);
    // The last second is the real audit window — a liquid that died
    // inside the run republishes the same total.
    ctl.send(Cmd::RunMs(1000.0));
    let end = wait_for(Duration::from_secs(60), || {
        let s = ctl.shared.read().ok()?;
        (s.snap.time_ms >= t1 + 900.0).then(|| s.snap.clone())
    });
    assert!(
        end.rate_hz > 0.0,
        "{name}: spikes happened but mean rate is 0 — dead liquid?"
    );
    assert!(
        end.spikes_total > spikes_at_4s,
        "{name}: no new spikes in the last second of a 5 s run — dead liquid?"
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
            "lsm_deep",
            include_str!("../../examples/lsm_deep.rhai"),
            "test deep",
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
            "lsm_deep" => {
                // 4-way task: liquid readouts above chance (0.25) on
                // held-out windows, no-liquid baselines near it.
                let acc = |key: &str| -> f64 {
                    out.iter()
                        .find(|l| l.contains(key))
                        .and_then(|l| l.split("acc=").nth(1))
                        .and_then(|r| r.trim().split(' ').next()?.parse::<f64>().ok())
                        .unwrap_or(f64::NAN)
                };
                for k in ["test deep", "test 1hid", "test ridge"] {
                    let a = acc(k);
                    assert!(a > 0.45, "{name}: {k} acc {a} not above chance");
                }
                for k in ["test raw", "test bits"] {
                    let a = acc(k);
                    assert!(a < 0.4, "{name}: no-liquid {k} acc {a} — task leaks");
                }
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

/// Stage 26 — the inspector's Train flow through the real worker:
/// lsm_big runs, the readout's layers are set by the same
/// `NetEdit::SetReadoutLayers` the inspector sends, the same Eval the
/// Train button re-sends rebuilds the net — and the published Geo shows
/// the readout with the config kept and a trained MLP whose hidden
/// layers match it (no "needs retraining").
/// Sabotage: examples reading the layers after `net.clear()` (or the
/// SetReadoutLayers delta dropping `readout_mlps`) → sizes stay [24].
#[test]
fn readout_layers_set_then_rerun_trains_them() {
    let src = include_str!("../../examples/lsm_big.rhai");
    let (ctl, _) = run_example("lsm_big", src);
    let geo_mlp = |ctl: &SimCtl| -> Option<proto::ReadoutMlp> {
        let s = ctl.shared.read().ok()?;
        let g = s.geo.as_ref()?;
        let k = g.readouts.iter().position(|r| r.0 == "ro")?;
        g.readout_mlps.get(k).cloned()
    };
    let m0 = wait_for(Duration::from_secs(30), || {
        geo_mlp(&ctl).filter(|m| !m.sizes.is_empty())
    });
    assert_eq!(m0.trained_hidden(), &[24], "default lsm_big layers");
    assert!(m0.layers.is_empty() && !m0.stale());

    let base = ctl.shared.read().unwrap().geo.as_ref().unwrap().version;
    ctl.send(Cmd::Edit {
        base,
        edit: proto::NetEdit::SetReadoutLayers {
            name: "ro".into(),
            layers: vec![128, 64, 32],
        },
    });
    let m1 = wait_for(Duration::from_secs(30), || {
        geo_mlp(&ctl).filter(|m| !m.layers.is_empty())
    });
    assert!(m1.stale(), "config moved on, trained net is old: {m1:?}");

    ctl.send(Cmd::Eval {
        code: format!("{src}\nprint(\"AUDIT_DONE rerun\");"),
    });
    wait_for(Duration::from_secs(120), || {
        let s = ctl.shared.read().ok()?;
        s.console
            .iter()
            .any(|l| l.trim() == "AUDIT_DONE rerun")
            .then_some(())
    });
    let m2 = wait_for(Duration::from_secs(30), || {
        geo_mlp(&ctl).filter(|m| m.trained_hidden() == [128, 64, 32])
    });
    assert_eq!(m2.layers, vec![128, 64, 32], "config survived the rebuild");
    assert!(!m2.stale());
    let console = ctl.shared.read().unwrap().console.clone();
    assert!(
        console
            .iter()
            .any(|l| l.contains("[128, 64, 32] (from the readout")),
        "console names the layers used"
    );
}
