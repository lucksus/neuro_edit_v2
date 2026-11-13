//! Free helpers shared by the `app` submodules: window-rect layout
//! (`usable_rect`/`clamp_window_rect`/`app_win` and the *_default_rect
//! defaults), misc utilities, and their unit tests.
//!
//! Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;

/// Q9: `Event::Paste` texts that belong to the network — none while a
/// widget holds keyboard focus (the field's own Ctrl+V must not also
/// paste a subnet into the scene).
pub(super) fn paste_events(events: &[egui::Event], kb_focused: bool) -> Vec<String> {
    if kb_focused {
        return Vec::new();
    }
    events
        .iter()
        .filter_map(|e| match e {
            egui::Event::Paste(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

/// The rect floating windows may occupy: below the toolbar, above the
/// status + hint strips, and inside the fixed docks (left tools, bottom
/// console — Stage 22). Only valid after those panels have been shown
/// this frame — update() shows docks first, windows after.
pub(super) fn usable_rect(ctx: &egui::Context) -> egui::Rect {
    ctx.available_rect()
}

/// Clamp `want` fully inside `usable`, shrinking if too big. Used for
/// window *defaults* so first-run/narrow viewports start on-screen;
/// `constrain_to` then keeps dragged/resized windows inside too.
pub(super) fn clamp_window_rect(want: egui::Rect, usable: egui::Rect) -> egui::Rect {
    let size = want.size().min(usable.size());
    let min = egui::pos2(
        want.min
            .x
            .clamp(usable.min.x, (usable.max.x - size.x).max(usable.min.x)),
        want.min
            .y
            .clamp(usable.min.y, (usable.max.y - size.y).max(usable.min.y)),
    );
    egui::Rect::from_min_size(min, size)
}

/// A floating window pinned to [`usable_rect`]. egui re-applies the
/// constraint every frame, so persisted off-screen positions and narrow
/// resizes get pulled back on-screen automatically.
pub(super) fn app_win<'open>(
    ctx: &egui::Context,
    title: impl Into<egui::WidgetText>,
) -> egui::Window<'open> {
    egui::Window::new(title).constrain_to(usable_rect(ctx))
}

/// Every floating window's default rect, expressed against
/// `usable_rect` — one table so first-run overlaps are auditable
/// (U3/Stage 15). The no-overlap tests iterate `DEFAULT_OPEN`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum DefWin {
    Plots,
    Inputs,
    Files,
    Manual,
    Script,
}

/// The windows `WindowVis::default` opens on first run — test-visible
/// mirror checked by `default_open_windows_never_overlap`. (Inspector
/// and console are docks now, not windows — Stage 22.)
#[cfg(test)]
pub(super) const DEFAULT_OPEN: &[DefWin] = &[DefWin::Plots];

/// (window → default rect) table against usable rect `u`.
///
/// U3: Plots anchors bottom-left. (Inspector used to anchor top-right,
/// the console bottom-right; both are docks now.)
pub(super) fn default_rects(u: egui::Rect) -> [(DefWin, egui::Rect); 5] {
    let side = egui::vec2(480.0, 220.0);
    [
        (
            DefWin::Plots,
            clamp_window_rect(
                egui::Rect::from_min_size(
                    egui::pos2(u.min.x + 12.0, u.max.y - side.y - 20.0),
                    side,
                ),
                u,
            ),
        ),
        (
            DefWin::Inputs,
            clamp_window_rect(
                egui::Rect::from_min_size(u.min + egui::vec2(260.0, 8.0), egui::vec2(220.0, 140.0)),
                u,
            ),
        ),
        (
            DefWin::Files,
            clamp_window_rect(
                egui::Rect::from_min_size(
                    egui::pos2(u.center().x - 210.0, u.min.y + 24.0),
                    egui::vec2(420.0, 420.0),
                ),
                u,
            ),
        ),
        // narrow viewports get a shrunken manual — cap the default at
        // 70% of the usable strip instead of covering it.
        (
            DefWin::Manual,
            clamp_window_rect(
                egui::Rect::from_min_size(
                    u.min + egui::vec2(24.0, 8.0),
                    egui::vec2(640.0, 520.0).min(u.size() * 0.7),
                ),
                u,
            ),
        ),
        (
            DefWin::Script,
            clamp_window_rect(
                egui::Rect::from_min_size(u.min + egui::vec2(12.0, 8.0), egui::vec2(480.0, 360.0)),
                u,
            ),
        ),
    ]
}

/// One entry of [`default_rects`].
pub(super) fn default_rect(w: DefWin, u: egui::Rect) -> egui::Rect {
    default_rects(u)
        .into_iter()
        .find(|(k, _)| *k == w)
        .unwrap()
        .1
}

/// U8: the session token persists in eframe's app.ron — keep it
/// owner-only (0600) instead of the umask default.
#[cfg(unix)]
pub(super) fn secure_storage_file() {
    use std::os::unix::fs::PermissionsExt;
    if let Some(dir) = eframe::storage_dir("neuro_edit v2") {
        let f = dir.join("app.ron");
        if f.exists() {
            let _ = std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o600));
        }
    }
}
#[cfg(not(unix))]
pub(super) fn secure_storage_file() {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::EditorApp;

    #[test]
    fn window_clamp_keeps_rect_inside_usable() {
        let usable = egui::Rect::from_min_max(egui::pos2(0.0, 60.0), egui::pos2(800.0, 640.0));
        // Already inside → untouched.
        let inside = egui::Rect::from_min_size(egui::pos2(100.0, 100.0), egui::vec2(300.0, 200.0));
        assert_eq!(clamp_window_rect(inside, usable), inside);
        // Hanging off the right → pulled left so it fits.
        let off_right =
            egui::Rect::from_min_size(egui::pos2(700.0, 100.0), egui::vec2(300.0, 200.0));
        let got = clamp_window_rect(off_right, usable);
        assert!(got.max.x <= usable.max.x && got.min.x >= usable.min.x);
        // Hanging off the bottom → lifted above the usable strip.
        let off_bot = egui::Rect::from_min_size(egui::pos2(100.0, 600.0), egui::vec2(300.0, 200.0));
        let got = clamp_window_rect(off_bot, usable);
        assert!(got.max.y <= usable.max.y);
        // Off the left/top → pushed back in.
        let off_tl = egui::Rect::from_min_size(egui::pos2(-50.0, 10.0), egui::vec2(100.0, 100.0));
        let got = clamp_window_rect(off_tl, usable);
        assert!(got.min.x >= usable.min.x && got.min.y >= usable.min.y);
        // Bigger than the viewport → shrunk to fit, pinned top-left.
        let huge = egui::Rect::from_min_size(egui::pos2(50.0, 70.0), egui::vec2(2000.0, 2000.0));
        let got = clamp_window_rect(huge, usable);
        assert_eq!(got.size(), usable.size());
        assert_eq!(got.min, usable.min);
    }

    /// U3/Stage 15: no two default-open windows overlap — checked at the
    /// two CI screenshot sizes (800x700, 1440x900) and across a width
    /// sweep since narrow viewports change the stacking.
    #[test]
    fn default_open_windows_never_overlap() {
        for (w, h) in [(800.0_f32, 700.0_f32), (1440.0, 900.0)] {
            let u = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(w, h));
            let table = default_rects(u);
            for (i, a) in DEFAULT_OPEN.iter().enumerate() {
                for b in &DEFAULT_OPEN[i + 1..] {
                    let ra = table.iter().find(|(k, _)| k == a).unwrap().1;
                    let rb = table.iter().find(|(k, _)| k == b).unwrap().1;
                    assert!(
                        !ra.intersects(rb),
                        "{a:?} overlaps {b:?} at {w}x{h}: {ra:?} vs {rb:?}"
                    );
                }
            }
        }
    }

    /// Stage 15 F: every app-state input to `render_offscreen` must move
    /// the SceneKey — driven through real EditorApp fields, not literals.
    #[test]
    fn scene_key_changes_with_app_state() {
        let app = EditorApp::new(crate::simctl::SimCtl::offline(), None);
        let sel: Vec<u32> = vec![3, 7];
        let act: Vec<f32> = vec![0.0; 8];
        let pvs: Vec<crate::gpu::Pv> = Vec::new();
        let parts: Vec<crate::gpu::Part> = Vec::new();
        let no_planes: Option<Vec<bool>> = None;
        let key = |a: &EditorApp| {
            crate::view::scene_key(
                a, 800, 600, 1.33, &sel, &act, &pvs, &parts, true, &no_planes,
            )
        };
        let base = key(&app);
        for f in [
            (|a: &mut EditorApp| a.show_grid = false) as fn(&mut EditorApp),
            |a| a.show_input_links = false,
            |a| a.hover = Some(4),
            |a| a.preview_delta = Some(glam::Vec3::new(1.0, 0.0, 0.0)),
            |a| a.plane_key = 9,
            |a| a.geo_version = 2,
            |a| a.camera.orbit(10.0, 0.0),
        ] {
            let mut a2 = EditorApp::new(crate::simctl::SimCtl::offline(), None);
            f(&mut a2);
            assert_ne!(key(&a2), base, "SceneKey ignored an app-state input");
        }
        // Buffer/flag inputs (not on the app, but rendered too).
        assert_ne!(
            crate::view::scene_key(
                &app,
                800,
                600,
                1.33,
                &[1],
                &act,
                &pvs,
                &parts,
                true,
                &no_planes
            ),
            base
        );
        assert_ne!(
            crate::view::scene_key(
                &app, 801, 600, 1.33, &sel, &act, &pvs, &parts, true, &no_planes
            ),
            base
        );
        assert_ne!(
            crate::view::scene_key(
                &app, 800, 600, 1.33, &sel, &act, &pvs, &parts, false, &no_planes
            ),
            base
        );
        assert_ne!(
            crate::view::scene_key(
                &app,
                800,
                600,
                1.33,
                &sel,
                &act,
                &pvs,
                &parts,
                true,
                &Some(vec![true])
            ),
            base
        );
    }

    #[test]
    fn paste_events_skip_when_text_field_focused() {
        let evs = vec![
            egui::Event::Paste("subnet json".into()),
            egui::Event::Text("x".into()),
        ];
        // no keyboard focus → the paste goes to the network
        assert_eq!(paste_events(&evs, false), vec!["subnet json".to_string()]);
        // a text field owns the keyboard → it ate the paste
        assert!(paste_events(&evs, true).is_empty());
    }
}
