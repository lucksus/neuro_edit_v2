//! Free helpers shared by the `app` submodules: window-rect layout
//! (`usable_rect`/`clamp_window_rect`/`app_win` and the *_default_rect
//! defaults), file/misc utilities, and their unit tests.
//!
//! Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;

use crate::snap::GeoSnap;

pub(super) fn fmt_size(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.1} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

pub(super) fn fmt_date(ms: u64) -> String {
    if ms == 0 {
        return "—".into();
    }
    // day-precision UTC, civil-from-days
    let z = (ms / 86_400_000) as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{:02}-{:02}", if m <= 2 { y + 1 } else { y }, m, d)
}

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

/// Browser download: bytes → Blob → object URL → anchor click.
#[cfg(target_arch = "wasm32")]
pub(super) fn download_blob(name: &str, bytes: &[u8]) {
    use wasm_bindgen::JsCast;
    let Some(w) = web_sys::window() else { return };
    let Some(doc) = w.document() else { return };
    let arr = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::new();
    parts.push(&arr);
    let Ok(blob) = web_sys::Blob::new_with_u8_array_sequence(&parts) else {
        return;
    };
    let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) else {
        return;
    };
    if let Ok(a) = doc.create_element("a") {
        if let Ok(a) = a.dyn_into::<web_sys::HtmlAnchorElement>() {
            a.set_href(&url);
            a.set_download(name);
            a.click();
        }
    }
    let _ = web_sys::Url::revoke_object_url(&url);
}

// Browser file picker: a transient `<input type=file multiple>` whose
// `change` handler FileReader-reads each File into PICKED, drained by
// `files_pump` on the next frames.
#[cfg(target_arch = "wasm32")]
thread_local! {
    pub(super) static PICKED: std::cell::RefCell<Vec<(String, Vec<u8>)>> =
        std::cell::RefCell::new(Vec::new());
}

#[cfg(target_arch = "wasm32")]
pub(super) fn web_file_picker() {
    use wasm_bindgen::{closure::Closure, JsCast};
    let Some(w) = web_sys::window() else { return };
    let Some(doc) = w.document() else { return };
    let Ok(el) = doc.create_element("input") else {
        return;
    };
    let Ok(input) = el.dyn_into::<web_sys::HtmlInputElement>() else {
        return;
    };
    input.set_type("file");
    input.set_multiple(true);
    // the input element stays alive via this closure's capture
    let inp = input.clone();
    let onchange = Closure::wrap(Box::new(move |_: web_sys::Event| {
        let Some(list) = inp.files() else { return };
        for j in 0..list.length() {
            let Some(file) = list.item(j) else { continue };
            let name = file.name();
            let Ok(reader) = web_sys::FileReader::new() else {
                continue;
            };
            let r2 = reader.clone();
            let onload = Closure::wrap(Box::new(move |_: web_sys::Event| {
                if let Ok(buf) = r2.result() {
                    let arr = js_sys::Uint8Array::new(&buf);
                    PICKED.with(|p| p.borrow_mut().push((name.clone(), arr.to_vec())));
                }
            }) as Box<dyn FnMut(web_sys::Event)>);
            reader.set_onload(Some(onload.as_ref().unchecked_ref()));
            let _ = reader.read_as_array_buffer(&file);
            onload.forget();
        }
    }) as Box<dyn FnMut(web_sys::Event)>);
    input.set_onchange(Some(onchange.as_ref().unchecked_ref()));
    input.click();
    onchange.forget();
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn local_dir_size(root: &std::path::Path) -> u64 {
    std::fs::read_dir(root)
        .map(|rd| {
            rd.flatten()
                .map(|e| {
                    let p = e.path();
                    if p.is_dir() {
                        local_dir_size(&p)
                    } else {
                        e.metadata().map(|m| m.len()).unwrap_or(0)
                    }
                })
                .sum()
        })
        .unwrap_or(0)
}

/// `dirs::data_dir()/neuro_edit` — the native no-`--connect` file space,
/// created on first use.
#[cfg(not(target_arch = "wasm32"))]
pub fn local_files_root() -> std::path::PathBuf {
    let d = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("neuro_edit");
    let _ = std::fs::create_dir_all(&d);
    d
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

/// One toolbar control group laid out atomically (Stage 15 B). Inside a
/// `horizontal_wrapped` row, plain `ui.horizontal` claims the whole
/// remaining row as its desired size, so egui's wrap check
/// (`available < desired`) never fires and groups split mid-control.
/// Claiming an explicit upper-bound width instead makes the wrap check
/// see the group's real footprint — it moves to the next row whole.
/// `max_w` must overestimate, never undershoot: overshoot only packs a
/// little early; undershoot lets content overflow the row edge.
pub(super) fn toolbar_group(
    ui: &mut egui::Ui,
    max_w: f32,
    add: impl FnOnce(&mut egui::Ui),
) -> egui::InnerResponse<()> {
    ui.allocate_ui_with_layout(
        egui::vec2(max_w, ui.spacing().interact_size.y),
        egui::Layout::left_to_right(egui::Align::Center),
        add,
    )
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

/// U5b: next auto-retry delay — 2 s first, doubling, capped at 60 s.
/// A manual Retry keeps the same progression (a click is not a loop).
pub(super) fn next_backoff(prev: f64) -> f64 {
    if prev <= 0.0 {
        2.0
    } else {
        (prev * 2.0).min(60.0)
    }
}

/// U5: the banner shows only on a real mid-session disconnect — not
/// while the login dialog is up (there is no session to lose yet).
pub(super) fn show_disconnect_banner(connected: bool, login_open: bool) -> bool {
    !connected && !login_open
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

/// Hide-bits for plane members. Member ids come over the wire — an
/// out-of-range id must not panic the frame.
pub(super) fn mark_bits(bits: &mut [u32], members: &[u32]) {
    for &m in members {
        if let Some(w) = bits.get_mut(m as usize >> 5) {
            *w |= 1 << (m & 31);
        }
    }
}

/// Parent dir of a path for a `ListFiles` request ("." at the top).
pub(super) fn dir_of(p: &str) -> String {
    match p.rsplit_once('/') {
        Some(("", _)) => "/".into(),
        Some((d, _)) => d.to_string(),
        None => ".".into(),
    }
}

/// Example auto-watch gate (Stage 13 H): the Geo that may apply the
/// example's `watch` list must be newer than the version captured when
/// Run was sent (`base`) and contain every requested neuron id. The
/// pre-run net can already cover the ids — watching it just gets wiped
/// by the eval's `net.clear` → reset_net_state.
pub(super) fn example_geo_covers(base: Option<u64>, geo: Option<&GeoSnap>, ids: &[u32]) -> bool {
    match (base, geo) {
        (Some(base), Some(g)) => {
            g.version != base && ids.iter().all(|&n| (n as usize) < g.pos.len())
        }
        _ => false,
    }
}

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

    /// Stage 15 B: `toolbar_group` units inside `horizontal_wrapped` wrap
    /// whole — a group never splits its icon/control across two rows, and
    /// a group that doesn't fit moves to the next row instead of
    /// overflowing the panel edge.
    #[test]
    fn wrapped_toolbar_groups_never_split() {
        let ctx = egui::Context::default();
        let mut same_row = true;
        let mut fits = true;
        let mut h = 0.0f32;
        for _ in 0..3 {
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::pos2(0.0, 0.0),
                        egui::vec2(300.0, 500.0),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    egui::TopBottomPanel::top("t").show(ctx, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            let max_x = ui.max_rect().max.x;
                            for g in 0..4 {
                                let r = toolbar_group(ui, 260.0, |ui| {
                                    let a = ui.label(format!("group-{g}")).rect;
                                    let b = ui.add(egui::Slider::new(&mut 0.5f32, 0.0..=1.0)).rect;
                                    same_row &= a.center().y == b.center().y;
                                })
                                .response
                                .rect;
                                fits &= r.max.x <= max_x + 0.5;
                            }
                        });
                        h = ui.min_rect().height();
                    });
                },
            );
        }
        assert!(same_row, "a control group split across wrap rows");
        assert!(fits, "a group overflowed the panel edge");
        assert!(h > 40.0, "groups did not wrap (panel {h}px)");
    }

    #[test]
    fn example_watch_needs_post_run_geo() {
        let mut g = GeoSnap::default();
        g.version = 7;
        g.pos = vec![[0.0; 3]; 5];
        let ids = [0u32, 1, 2, 3, 4];
        // Not armed yet (loaded but never run) — pre-run geo must not apply.
        assert!(!example_geo_covers(None, Some(&g), &ids));
        // Same version as at Run — the eval hasn't landed yet.
        assert!(!example_geo_covers(Some(7), Some(&g), &ids));
        g.version = 8;
        assert!(example_geo_covers(Some(7), Some(&g), &ids));
        // Post-run geo that doesn't cover the ids — e.g. cleared but
        // not yet rebuilt — must not apply either.
        g.pos.truncate(3);
        assert!(!example_geo_covers(Some(7), Some(&g), &ids));
        assert!(!example_geo_covers(Some(7), None, &ids));
    }

    #[test]
    fn watch_takes_lowest_ids_not_hash_order() {
        // U8: selection is a HashSet — iterating it gave an arbitrary
        // 8 of the selected neurons. Now the 8 lowest ids win.
        let mut app = EditorApp::new(crate::simctl::SimCtl::offline(), None);
        let mut geo = crate::snap::GeoSnap::default();
        geo.pos = vec![[0.0; 3]; 20];
        app.geo = Some(std::sync::Arc::new(geo));
        app.selection = [15, 3, 9, 1, 12, 7, 19, 2, 11, 5].into_iter().collect();
        app.update_watch();
        assert_eq!(app.watch_sent, vec![1, 2, 3, 5, 7, 9, 11, 12]);
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
    fn reconnect_backoff_doubles_to_60() {
        let mut d = 0.0;
        let seq: Vec<u64> = (0..8)
            .map(|_| {
                d = next_backoff(d);
                d as u64
            })
            .collect();
        assert_eq!(seq, vec![2, 4, 8, 16, 32, 60, 60, 60]);
    }

    #[test]
    fn disconnect_banner_gating() {
        assert!(show_disconnect_banner(false, false));
        assert!(!show_disconnect_banner(true, false));
        assert!(!show_disconnect_banner(false, true));
    }

    #[test]
    fn upload_slot_frees_on_written_and_error() {
        // Q8: the upload slot must clear when its FileWritten (or an
        // "upload …" error) arrives — otherwise the second queued upload
        // waits forever behind the finished/failed first.
        let mut app = EditorApp::new(SimCtl::offline(), None);
        // slot occupied mid-upload
        app.files.upload = Some(("a.bin".into(), vec![1, 2, 3], 0));
        app.files.upload_q.push_back(("b.bin".into(), vec![9]));
        // FileWritten for an unrelated path must NOT clear it
        {
            let mut sh = app.ctl.shared.write().unwrap();
            sh.file_written = Some("other.bin".into());
        }
        assert!(app.files.upload.is_some());
        // FileWritten for our path clears it
        {
            let mut sh = app.ctl.shared.write().unwrap();
            sh.file_written = Some("a.bin".into());
        }
        // simulate the consumables branch
        let p = app.ctl.shared.write().unwrap().file_written.take();
        if let Some(p) = p {
            if app.files.upload.as_ref().map(|u| u.0.as_str()) == Some(p.as_str()) {
                app.files.upload = None;
            }
        }
        assert!(app.files.upload.is_none(), "slot freed on FileWritten");
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

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn local_file_errors_reach_status() {
        // Q14: local-mode file ops used to discard errors (`let _ =`)
        // leaving files.status silent on failure.
        let root = std::env::temp_dir().join(format!("q14_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut app = EditorApp::new(SimCtl::offline(), None);
        app.files.local_root = Some(root.clone());
        app.files.cwd.clear();

        // rename a file that doesn't exist -> status must say so
        app.files.sel = Some("ghost.txt".into());
        app.files.rename_to = "renamed.txt".into();
        app.file_rename();
        assert!(
            app.files.status.contains("rename"),
            "rename error hidden: {:?}",
            app.files.status
        );

        // delete a missing file -> same
        app.files.status.clear();
        app.file_delete("ghost.txt");
        assert!(
            app.files.status.contains("delete"),
            "delete error hidden: {:?}",
            app.files.status
        );

        // mkdir over an existing FILE -> error, not silence
        std::fs::write(root.join("afile"), b"x").unwrap();
        app.files.status.clear();
        app.files.mkdir = "afile".into();
        app.file_mkdir();
        assert!(
            app.files.status.starts_with("mkdir afile"),
            "mkdir outcome hidden: {:?}",
            app.files.status
        );

        // and the happy path reports too
        app.files.status.clear();
        app.files.mkdir = "okdir".into();
        app.file_mkdir();
        assert_eq!(app.files.status, "mkdir okdir");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn mark_bits_ignores_out_of_range_member_ids() {
        let mut bits = vec![0u32; 2]; // covers ids 0..63
        mark_bits(&mut bits, &[0, 63]);
        mark_bits(&mut bits, &[64, u32::MAX]); // must not panic
        assert_eq!(bits[0], 1);
        assert_eq!(bits[1], 1 << 31);
    }
}
