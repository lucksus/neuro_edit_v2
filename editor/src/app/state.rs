//! Editor state (Stage 22): Build / Run / Script — one enum derived from
//! the worker's `Status` flags, plus the per-action enable table every
//! gated control asks. `disabled()` returns the reason a control is off;
//! `None` means it's on. The reason doubles as the disabled tooltip.

use eframe::egui;

use super::{EditorApp, Mode};

/// What the editor as a whole is doing right now.
///
/// - `Build`: no live sim — the net is being edited.
/// - `Run`: a sim exists (`sim_built`) or is stepping (`running`). A
///   paused sim counts as Run: what sets it apart from Build is that
///   edits would tear the live sim down, and there's sim time to reset.
/// - `Script`: the worker is busy (an eval, a heavy edit, a load) — the
///   controls that would collide with the job are off until it ends.
///   Highest precedence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EditorState {
    Build,
    Run,
    Script,
}

/// The `Status` inputs the editor state derives from.
#[derive(Clone, Copy, Default)]
pub struct RunFlags {
    /// `Status.running` — auto-run is on (the run loop is stepping).
    pub running: bool,
    /// `Status.sim_built` — a sim has been built and published; it stays
    /// set while paused, cleared when the sim is dropped or the net is
    /// replaced.
    pub sim_built: bool,
    /// `Status.busy` — an eval or blocking op is in flight.
    pub busy: bool,
}

impl RunFlags {
    pub fn state(&self) -> EditorState {
        if self.busy {
            EditorState::Script
        } else if self.running || self.sim_built {
            EditorState::Run
        } else {
            EditorState::Build
        }
    }
}

/// An action a control can request — the enable table's key.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Act {
    /// Run / Pause (`AutoRun`). Always on — it IS how you leave Run.
    RunToggle,
    /// Step 10 / 100 ms (`RunMs`) — off while auto-running (pause to
    /// step) and while busy.
    Step,
    /// Reset to t=0 — off before a sim exists.
    Reset,
    /// Playback speed — live-safe during a run.
    Speed,
    /// Backend picker — invalidates the sim; off during a run.
    Backend,
    /// dt — invalidates the sim; off during a run.
    Dt,
    /// Background noise — applied to the live sim when built.
    Noise,
    /// Input-source value / play / external drive — live-safe.
    SetInput,
    /// Any structural `NetEdit` (insert/connect/move/delete/paste,
    /// kind/type/params, menus' builders). Edits invalidate the sim —
    /// mid-run that rebuilds it and silently loses state, so they're
    /// off until Pause.
    Edit,
    /// New / Load / recent — replaces the whole net.
    NewLoad,
    /// Save / download — reads the net; safe unless a script holds the
    /// worker.
    Save,
    /// Console/script eval. Always on while the worker is idle — evals
    /// that drive inputs during a run are a core workflow.
    Eval,
    /// Server-side job start — independent of the interactive env.
    Job,
}

/// `Some(reason)` when `act` is unavailable under `f`, else `None`.
pub fn disabled(f: RunFlags, a: Act) -> Option<&'static str> {
    const BUSY: &str = "worker busy — a script or load is running; wait, or press Stop";
    const RUNNING: &str = "the sim is running — pause it first";
    const REBUILD: &str = "restarts the sim from scratch — stop it first";
    const EDIT_RUNNING: &str = "edits rebuild the sim — pause the run first";
    const NO_SIM: &str = "no sim has been built yet — run first";
    match a {
        Act::RunToggle | Act::Job => None,
        Act::Step => {
            if f.busy {
                Some(BUSY)
            } else if f.running {
                Some("the sim is running — pause it to step")
            } else {
                None
            }
        }
        Act::Reset => {
            if f.busy {
                Some(BUSY)
            } else if !f.sim_built {
                Some(NO_SIM)
            } else {
                None
            }
        }
        Act::Speed | Act::Noise | Act::SetInput => {
            if f.busy {
                Some(BUSY)
            } else {
                None
            }
        }
        Act::Backend | Act::Dt => {
            if f.busy {
                Some(BUSY)
            } else if f.running {
                Some(REBUILD)
            } else {
                None
            }
        }
        Act::Edit => {
            if f.busy {
                Some(BUSY)
            } else if f.running {
                Some(EDIT_RUNNING)
            } else {
                None
            }
        }
        Act::NewLoad => {
            if f.busy {
                Some(BUSY)
            } else if f.running {
                Some("replaces the net — stop the sim first")
            } else {
                None
            }
        }
        Act::Save => {
            if f.busy {
                Some(BUSY)
            } else {
                None
            }
        }
        Act::Eval => {
            if f.busy {
                Some("a script is already running — wait, or press Stop")
            } else {
                None
            }
        }
    }
}

impl EditorApp {
    /// Status flags → the three-state enum.
    pub(crate) fn run_flags(&self) -> RunFlags {
        let sh = self.ctl.shared.read().unwrap();
        RunFlags {
            running: sh.running,
            sim_built: sh.sim_built,
            busy: sh.busy,
        }
    }

    pub(crate) fn editor_state(&self) -> EditorState {
        self.run_flags().state()
    }

    /// Disabled reason for `a` (the control's tooltip), `None` = on.
    pub(crate) fn why(&self, a: Act) -> Option<&'static str> {
        disabled(self.run_flags(), a)
    }

    pub(crate) fn can(&self, a: Act) -> bool {
        self.why(a).is_none()
    }

    /// The interaction mode the viewport acts on: while edits are off
    /// (running/script), edit modes collapse to Navigate — selection
    /// and watching still work. `self.mode` keeps its value, so pausing
    /// puts the user back in the mode they left.
    pub(crate) fn eff_mode(&self) -> Mode {
        if self.can(Act::Edit) {
            self.mode
        } else {
            Mode::Navigate
        }
    }
}

/// `add_enabled` + the disabled reason on the disabled hover — the one
/// call every gated control goes through.
pub(crate) fn gate(
    ui: &mut egui::Ui,
    why: Option<&'static str>,
    w: egui::Button,
) -> egui::Response {
    let r = ui.add_enabled(why.is_none(), w);
    match why {
        Some(why) => r.on_disabled_hover_text(why),
        None => r,
    }
}

/// The same gate for non-Button widgets (combo boxes, sliders, drag
/// values): wraps a whole `ui` scope. Returns the `InnerResponse` —
/// `.inner` is the closure's value, `.response` the disabled-hover
/// carrier.
pub(crate) fn gate_ui<R>(
    ui: &mut egui::Ui,
    why: Option<&'static str>,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    let inner = ui.add_enabled_ui(why.is_none(), add);
    if let Some(why) = why {
        inner.response.clone().on_disabled_hover_text(why);
    }
    inner
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(running: bool, sim_built: bool, busy: bool) -> RunFlags {
        RunFlags {
            running,
            sim_built,
            busy,
        }
    }

    #[test]
    fn state_derives_from_status() {
        assert_eq!(f(false, false, false).state(), EditorState::Build);
        assert_eq!(f(true, true, false).state(), EditorState::Run);
        // Paused with a built sim is still Run — the live state is what
        // makes edits dangerous, not the run flag.
        assert_eq!(f(false, true, false).state(), EditorState::Run);
        // Running implies a built sim, but derive defensively anyway.
        assert_eq!(f(true, false, false).state(), EditorState::Run);
        // busy wins over everything — a script during a run is Script.
        assert_eq!(f(true, true, true).state(), EditorState::Script);
        assert_eq!(f(false, false, true).state(), EditorState::Script);
    }

    #[test]
    fn transitions_build_run_pause_build() {
        // Idle → Run on AutoRun(true) → still Run when paused → Build
        // again when the sim is dropped (net replaced / reset).
        let mut flags = f(false, false, false);
        assert_eq!(flags.state(), EditorState::Build);
        flags.running = true;
        flags.sim_built = true;
        assert_eq!(flags.state(), EditorState::Run);
        flags.running = false; // paused — live sim state remains
        assert_eq!(flags.state(), EditorState::Run);
        flags.sim_built = false; // sim dropped (edit/new/dt change)
        assert_eq!(flags.state(), EditorState::Build);
        // busy is transient: on and off around any of the above.
        flags.busy = true;
        assert_eq!(flags.state(), EditorState::Script);
        flags.busy = false;
        assert_eq!(flags.state(), EditorState::Build);
    }

    #[test]
    fn enable_table() {
        let build = f(false, false, false);
        let run = f(true, true, false);
        let paused = f(false, true, false);
        let script = f(false, false, true);

        // RunToggle & jobs never gate.
        for fl in [build, run, paused, script] {
            assert!(disabled(fl, Act::RunToggle).is_none());
            assert!(disabled(fl, Act::Job).is_none());
        }
        // Live-safe controls: on in every state but Script.
        for a in [Act::Speed, Act::Noise, Act::SetInput, Act::Save] {
            assert!(disabled(build, a).is_none(), "{a:?} in Build");
            assert!(disabled(run, a).is_none(), "{a:?} in Run");
            assert!(disabled(paused, a).is_none(), "{a:?} paused");
            assert!(disabled(script, a).is_some(), "{a:?} in Script");
        }
        // Rebuild-cost controls: on in Build and paused-Run, off while
        // stepping or scripting.
        for a in [Act::Backend, Act::Dt, Act::Edit] {
            assert!(disabled(build, a).is_none(), "{a:?} in Build");
            assert!(disabled(run, a).is_some(), "{a:?} in Run");
            assert!(disabled(paused, a).is_none(), "{a:?} paused");
            assert!(disabled(script, a).is_some(), "{a:?} in Script");
        }
        // Step: on in Build and paused, off while auto-running.
        assert!(disabled(build, Act::Step).is_none());
        assert!(disabled(run, Act::Step).is_some());
        assert!(disabled(paused, Act::Step).is_none());
        // Reset needs a built sim; running doesn't block it.
        assert!(disabled(build, Act::Reset).is_some());
        assert!(disabled(run, Act::Reset).is_none());
        assert!(disabled(paused, Act::Reset).is_none());
        // Eval: one at a time — off only while busy.
        assert!(disabled(build, Act::Eval).is_none());
        assert!(disabled(run, Act::Eval).is_none());
        assert!(disabled(script, Act::Eval).is_some());
        // New/load: off while running or busy.
        assert!(disabled(build, Act::NewLoad).is_none());
        assert!(disabled(run, Act::NewLoad).is_some());
        assert!(disabled(paused, Act::NewLoad).is_none());
        assert!(disabled(script, Act::NewLoad).is_some());
    }
}
