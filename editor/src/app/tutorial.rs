//! Stage 25 first-open tutorial: a docked bottom bar walks a new user
//! through the real gestures — place a neuron, drive it, wire it, run
//! it, grow the net, watch v(t), run a script — and advances only when
//! the worker-published state says the thing actually happened (no
//! Next button). The control each step means gets a pulsing outline.
//!
//! Persistence is one byte in `Persisted.tutorial` (eframe storage —
//! native config dir, web localStorage): 0 unseen → auto-start on the
//! first hydrated net, 1 done, 2 skipped. Help > Tutorial reopens it;
//! a non-empty net asks through `Dialog::TutorialClear` first (the
//! clear is one `NewNet`, undoable like every net replacement).
//!
//! Fewer, bigger controls while it runs: the left dock keeps only the
//! section the step needs and the floating panels stay closed until
//! their step asks for them (inspector at "watch v(t)", console at
//! "script"). Done/skipped restores the WindowVis default.

use eframe::egui;

use proto::ClientMsg;

use super::{EditorApp, Mode};

/// Steps 0..=6 in order; `N_STEPS` is the done index.
pub const N_STEPS: usize = 7;

/// The line the prefilled console snippet prints — the step-7 detector
/// looks for it in `Shared::console`, so the advance can only come from
/// a real eval, not a click.
pub const TUT_MARKER: &str = "tutorial net built";

/// Step 7's prefilled console line — builds a small driven net through
/// the same script API the examples use. `net.clear()` first so a stale
/// doc config can't colour it.
pub const TUT_SNIPPET: &str = "net.clear(); net.create_2d_grid(\"net\", 3, 3, 40.0); \
     net.create_connections(\"net\", 0.4, 80.0, 7.0); print(\"tutorial net built\")";

/// Persisted values for `Persisted.tutorial`.
pub const P_UNSEEN: u8 = 0;
pub const P_DONE: u8 = 1;
pub const P_SKIPPED: u8 = 2;

/// Tutorial runtime state (the persisted byte lives in `persist`).
#[derive(Default)]
pub struct TutUi {
    /// Current step 0..N_STEPS — `N_STEPS` means the done card shows.
    pub step: usize,
    /// Card visible (an unfinished tutorial is running).
    pub active: bool,
    /// Boot decision consumed: auto-start / clear-ask / never.
    pub boot_done: bool,
    /// `--tutorial-step N` review driver — forces step N for shots.
    pub force_step: Option<usize>,
    /// Done card showing after the last detector fired.
    pub show_done: bool,
    /// Step 6 prefilled the console once — a second entry must not
    /// clobber a script the user edited.
    pub prefilled: bool,
    /// 0 unseen / 1 done / 2 skipped — mirrored into `Persisted` on save.
    pub persist: u8,
}

/// A widget the tutorial can outline — registers its screen rect under
/// `key` for one frame (insert_temp, so closed menus/dialogs stop
/// glowing by themselves). Cheap enough to call unconditionally.
pub(crate) fn hl(ui: &egui::Ui, key: &'static str, rect: egui::Rect) {
    ui.ctx().data_mut(|d| {
        d.insert_temp(egui::Id::new(("tut_hl", key)), rect);
    });
}

/// Highlight keys per step — unregistered keys paint nothing (a menu
/// that isn't open has no rect).
fn step_highlights(step: usize) -> &'static [&'static str] {
    match step {
        0 => &["tool_insert"],
        1 => &["menu_insert", "menu_input_item", "dlg_primary"],
        2 => &["tool_connect"],
        3 => &["run_btn"],
        4 => &["menu_insert", "menu_grid2d_item", "tool_insert"],
        5 => &["scene"],
        6 => &["console_input", "console_run"],
        _ => &[],
    }
}

/// (title, body) for the card — kept terse; the outline does the pointing.
fn step_text(step: usize) -> (&'static str, &'static str) {
    match step {
        0 => (
            "add a neuron",
            "pick “insert” in the tools on the left, then click anywhere in the scene",
        ),
        1 => (
            "add an input",
            "Insert > Input… > Create — with the neuron selected it attaches right away",
        ),
        2 => (
            "connect input -> neuron",
            "pick “connect” in the tools, then drag from the input's marker onto the neuron",
        ),
        3 => (
            "run it",
            "press RUN on the left — the status bar's rate should rise above 0",
        ),
        4 => (
            "grow the net",
            "the run paused so you can edit — Insert > 2D grid… (a 2×2 adds 4) or “insert” + clicks",
        ),
        5 => (
            "inspect a neuron",
            "click a neuron in the scene — the inspector on the right shows its live v(t)",
        ),
        6 => (
            "script it",
            "the console below is prefilled — press Ctrl+Enter (or its Run); Help > Examples has more",
        ),
        _ => ("done", "you're set — Help > Examples and Help > Manual are the next stops"),
    }
}

impl EditorApp {
    /// True while a step card is up — the left dock shows the reduced
    /// control set and the step machines below tick. `pub` like the
    /// other driver-surface methods — the headless tutorial test reads
    /// the machine through it.
    pub fn tut_active(&self) -> bool {
        self.tut.active && self.tut.step < N_STEPS
    }

    /// First-frame boot decision, once the worker's Geo has hydrated.
    /// Called every update; `boot_done` makes it a no-op after the first
    /// real check. Ordering: a `--tutorial-step` driver wins, then a
    /// persisted done/skip, then the net state — empty starts, non-empty
    /// asks through the clear dialog, `tut_never` (a `--script` launch
    /// isn't a first-open) never does. `pub` for the headless test.
    pub fn tutorial_boot(&mut self) {
        if self.tut.boot_done {
            return;
        }
        if let Some(s) = self.tut.force_step {
            self.tut.boot_done = true;
            // Replay the chrome the real machine would have reached at
            // this step — begin() gives the reduced set, then each
            // passed step's own opens.
            self.tutorial_begin();
            self.tut.step = s;
            if s >= 5 {
                self.win.inspector = true;
            }
            if s >= 6 {
                self.win.console = true;
                self.console.input = TUT_SNIPPET.into();
                self.tut.prefilled = true;
            }
            if s >= N_STEPS {
                self.tut.show_done = true;
                self.win.inspector = true;
                self.win.plots = true;
                self.win.console = true;
            }
            self.tut.active = s < N_STEPS;
            return;
        }
        if self.tut_never || self.tut.persist != P_UNSEEN {
            self.tut.boot_done = true;
            return;
        }
        let empty = {
            let sh = self.ctl.shared.read().unwrap();
            sh.geo
                .as_ref()
                .map(|g| g.pos.is_empty() && g.inputs.is_empty())
        };
        match empty {
            // Geo hasn't landed yet — check again next frame.
            None => {}
            Some(true) => {
                self.tut.boot_done = true;
                self.tutorial_begin();
            }
            Some(false) => {
                self.tut.boot_done = true;
                self.dlg = Some(super::Dialog::TutorialClear);
            }
        }
    }

    /// Begin on the (already empty) net: card on, chrome reduced.
    /// `pub` — Help > Tutorial and the headless test both enter here.
    pub fn tutorial_begin(&mut self) {
        self.tut.active = true;
        self.tut.step = 0;
        self.tut.show_done = false;
        self.tut.prefilled = false;
        // Fewer controls: floating panels stay out of the way until a
        // step asks for them; the dock is where the needed buttons live.
        self.win.plots = false;
        self.win.console = false;
        self.win.inspector = false;
        self.win.script = false;
        self.win.left_panel = true;
        self.win.left_narrow_open = true;
        self.mode = Mode::Navigate;
    }

    /// End of the road — done (persist 1) or skip (persist 2). Either
    /// way the full chrome comes back. `pub` for the persistence test.
    pub fn tutorial_end(&mut self, persist: u8) {
        self.tut.active = false;
        self.tut.persist = persist;
        self.win.inspector = true;
        self.win.plots = true;
        self.win.console = true;
    }

    /// Advance one step. Only real state moves the machine — detectors
    /// read the worker-published `Shared` (geometry, sim snapshot,
    /// console log), never a button press. At most one step per call.
    /// `pub` — update() and the headless step-machine test both drive it.
    pub fn tutorial_tick(&mut self) {
        if !self.tut_active() {
            return;
        }
        let sh = self.ctl.shared.read().unwrap();
        let Some(g) = sh.geo.clone() else {
            return;
        };
        let (rate, spikes) = (sh.snap.rate_hz, sh.snap.spikes_total);
        let marker = sh.console.iter().any(|l| l.contains(TUT_MARKER));
        drop(sh);
        let hit = match self.tut.step {
            0 => !g.pos.is_empty(),
            1 => !g.inputs.is_empty(),
            // input born targeted (dialog on a selection) or wired by
            // hand — a neuron↔neuron synapse counts too, the lesson is
            // the connect gesture.
            2 => g.inputs.iter().any(|i| !i.targets.is_empty()) || !g.syn.is_empty(),
            3 => rate > 0.0 || spikes > 0,
            4 => g.pos.len() >= 5,
            5 => !self.watch_sent.is_empty(),
            6 => marker,
            _ => false,
        };
        if hit {
            self.tut.step += 1;
            match self.tut.step {
                // Leaving Run: pause for the edits the next steps need
                // (the enable table would gate them anyway — do the
                // courteous thing and say so in the card).
                4 => self.send(ClientMsg::AutoRun(false)),
                5 => self.win.inspector = true,
                6 => {
                    self.win.console = true;
                    if !self.tut.prefilled {
                        self.console.input = TUT_SNIPPET.into();
                        self.tut.prefilled = true;
                    }
                }
                _ => {}
            }
            if self.tut.step >= N_STEPS {
                self.tut.show_done = true;
                self.tutorial_end(P_DONE);
            }
        }
    }

    /// The docked tutorial bar — a bottom strip above the hints line,
    /// spanning the full width so it can never cover the scene or sit
    /// under a dock. Title + body left, step dots center, skip right.
    pub(crate) fn tutorial_panel(&mut self, ctx: &egui::Context) {
        let done = self.tut.show_done;
        if !self.tut_active() && !done {
            return;
        }
        egui::TopBottomPanel::bottom("tutorial_bar")
            .frame(egui::Frame::NONE)
            .show_separator_line(false)
            .show(ctx, |ui| {
                ui.add_space(3.0);
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    let (title, body) = step_text(self.tut.step);
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("tutorial")
                                    .strong()
                                    .color(egui::Color32::from_rgb(120, 190, 255)),
                            );
                            ui.weak(if done {
                                "done".to_string()
                            } else {
                                format!("{} / {N_STEPS}", self.tut.step + 1)
                            });
                        });
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(title).strong());
                            ui.label("—");
                            ui.label(body);
                        });
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(10.0);
                        if done {
                            if ui.button("Finish").clicked() {
                                self.tut.show_done = false;
                            }
                        } else if ui.button("Skip tutorial").clicked() {
                            self.tutorial_end(P_SKIPPED);
                            self.tut.show_done = false;
                        }
                        // progress dots — the step machine's own index
                        for i in 0..N_STEPS {
                            let on = done || i < self.tut.step;
                            let cur = !done && i == self.tut.step;
                            let c = if on {
                                egui::Color32::from_rgb(70, 180, 90)
                            } else if cur {
                                egui::Color32::from_rgb(120, 190, 255)
                            } else {
                                ui.visuals().widgets.noninteractive.fg_stroke.color
                            };
                            ui.label(egui::RichText::new("●").color(c).small());
                        }
                    });
                });
                ui.add_space(3.0);
            });
    }

    /// Pulsing outlines on the controls the current step means — reads
    /// the rects widgets registered through [`hl`] this frame, so a
    /// closed menu or dialog simply has nothing to stroke.
    pub(crate) fn tutorial_overlay(&mut self, ctx: &egui::Context) {
        if !self.tut_active() {
            return;
        }
        let pulse = (0.5 + 0.5 * (self.wall_ms * 0.006).sin()) as f32;
        let col = egui::Color32::from_rgb(255, 200, 60).gamma_multiply(0.65 + 0.35 * pulse);
        for key in step_highlights(self.tut.step) {
            let Some(rect) = ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::new(("tut_hl", key))))
            else {
                continue;
            };
            egui::Area::new(egui::Id::new(("tut_hl_paint", key)))
                .order(egui::Order::Foreground)
                .interactable(false)
                .movable(false)
                .show(ctx, |ui| {
                    let r = rect.expand(3.0);
                    ui.painter().rect_stroke(
                        r,
                        4.0,
                        egui::Stroke::new(2.5_f32, col),
                        egui::StrokeKind::Outside,
                    );
                });
        }
    }
}
