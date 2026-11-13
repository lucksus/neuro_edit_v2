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
/// status + hint strips (Stage 15 A/C). Only valid after those panels
/// have been shown this frame — update() calls windows last.
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
pub(super) fn app_win<'open>(ctx: &egui::Context, title: impl Into<egui::WidgetText>) -> egui::Window<'open> {
    egui::Window::new(title).constrain_to(usable_rect(ctx))
}

/// U3: bottom-right Console default.
pub(super) fn console_default_rect(u: egui::Rect) -> egui::Rect {
    clamp_window_rect(
        egui::Rect::from_min_size(
            egui::pos2(u.max.x - 430.0, u.max.y - 260.0),
            egui::vec2(420.0, 240.0),
        ),
        u,
    )
}

/// U3: bottom-left Plots default — on narrow viewports the two
/// bottom-anchored windows collide, so Plots stacks above Console.
pub(super) fn plots_default_rect(u: egui::Rect) -> egui::Rect {
    let side = egui::vec2(480.0, 220.0);
    let c = console_default_rect(u);
    let stacked = c.min.x < u.min.x + 12.0 + side.x + 8.0;
    let y = if stacked {
        c.min.y - side.y - 12.0
    } else {
        u.max.y - side.y - 20.0
    };
    clamp_window_rect(
        egui::Rect::from_min_size(egui::pos2(u.min.x + 12.0, y), side),
        u,
    )
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
    use crate::simctl::SimCtl;

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
    fn plots_and_console_defaults_never_overlap() {
        // U3: at narrow widths the defaults stack; wide stays
        // side-by-side. Assert disjoint at every reasonable width.
        for w in [500.0_f32, 800.0, 930.0, 1440.0] {
            let u = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(w, 700.0));
            assert!(
                !plots_default_rect(u).intersects(console_default_rect(u)),
                "default windows overlap at {w} px"
            );
        }
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
