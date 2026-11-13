//! Central 3D viewport: the scene renders offscreen in `update`, and an
//! egui-wgpu paint callback blits it into the viewport rect. Input: camera
//! controls, GPU picking, box select, work-plane move/insert/connect.

use eframe::{egui, egui_wgpu};
use egui::epaint::PaintCallbackInfo;
use glam::{Mat4, Vec2, Vec3};

use crate::app::{Act, EditorApp, Mode, MoveDrag};
use crate::camera::Camera;
use crate::gpu::{DrawParams, Entity, Gpu};
use crate::scene::{self, Plane};
use crate::snap::GeoSnap;

/// Run `f` with the shared `Gpu` stored in egui-wgpu callback resources.
pub fn with_gpu(rs: &egui_wgpu::RenderState, f: impl FnOnce(&mut Gpu)) {
    let mut r = rs.renderer.write();
    if let Some(g) = r.callback_resources.get_mut::<Gpu>() {
        f(g);
    }
}

/// Map an instance index to its entity (inst order: neurons, inputs, readouts).
pub fn entity_of(geo: &GeoSnap, inst: u32) -> Option<Entity> {
    let i = inst as usize;
    let n = geo.pos.len();
    if i < n {
        Some(Entity::Neuron(inst))
    } else if i < n + geo.inputs.len() {
        Some(Entity::Input((i - n) as u32))
    } else if i < n + geo.inputs.len() + geo.readouts.len() {
        Some(Entity::Readout((i - n - geo.inputs.len()) as u32))
    } else {
        None
    }
}

/// World-space ray through a viewport-normalized pixel.
pub fn ray(cam: &Camera, aspect: f32, x: f32, y: f32) -> (Vec3, Vec3) {
    let inv: Mat4 = cam.view_proj(aspect).inverse();
    let p0 = inv.project_point3(Vec3::new(2.0 * x - 1.0, 1.0 - 2.0 * y, -1.0));
    let p1 = inv.project_point3(Vec3::new(2.0 * x - 1.0, 1.0 - 2.0 * y, 1.0));
    (p0, (p1 - p0).normalize())
}

/// Intersect a ray with the active work plane through `anchor`.
pub fn plane_hit(plane: Plane, anchor: Vec3, ro: Vec3, rd: Vec3) -> Option<Vec3> {
    let (nrm, off) = match plane {
        Plane::XZ => (Vec3::Y, anchor.y),
        Plane::XY => (Vec3::Z, anchor.z),
        Plane::YZ => (Vec3::X, anchor.x),
    };
    plane_hit_n(nrm, nrm * off, ro, rd)
}

/// Fingerprint the camera for the SceneKey (Stage 14 B): the view-proj
/// matrix plus the floor height — everything `write_cam` bakes in.
fn cam_bits(cam: &Camera, aspect: f32) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    use std::hash::Hasher;
    h.write(bytemuck::cast_slice(&cam.view_proj(aspect).to_cols_array()));
    h.write_u32(cam.floor_y.to_bits());
    h.finish()
}

/// Hash a Pod slice — the SceneKey contribution of a GPU-side data block.
fn hash_pod<T: bytemuck::NoUninit>(s: &[T]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytemuck::cast_slice(s));
    h.finish()
}

/// SceneKey contribution of the hide-bit field alone — `plane_frame`
/// uses it on the no-grids path so a block toggle still re-renders.
pub(crate) fn hash_bits(bits: &[u32]) -> u64 {
    hash_pod(bits)
}

/// Build the frame's SceneKey (Stage 14 B). Every input the scene pass
/// reads is fingerprinted — Stage 15 F audit:
///   w_px/h_px  scene-texture size (DPI is already in physical pixels)
///   cam        camera + aspect + vp-height (syn-LOD inputs)
///   preview    live move-drag delta uniform
///   hover      brightened instance
///   sel/act/pvs/parts/planes  per-frame buffer contents
///   params     draw toggles: pv/part counts, show_syn, show_grid,
///              show_input_links (U7), planes_on
///   geo        instance-buffer version
/// Deliberately unkeyed (compile-time or non-scene): clear colour, the
/// scene:: palette constants, misc.x/y (cube size/glow), misc.z (time —
/// unread by the shader), misc.w (pick-mode flag — the pick pass writes
/// its own uniform and never touches the scene texture).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scene_key(
    app: &EditorApp,
    w_px: u32,
    h_px: u32,
    aspect: f32,
    sel: &[u32],
    act: &[f32],
    pvs: &[crate::gpu::Pv],
    parts: &[crate::gpu::Part],
    show_syn: bool,
    planes_on: &Option<Vec<bool>>,
) -> crate::gpu::SceneKey {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    (pvs.len() as u32).hash(&mut h);
    (parts.len() as u32).hash(&mut h);
    show_syn.hash(&mut h);
    app.show_grid.hash(&mut h);
    app.show_input_links.hash(&mut h);
    planes_on.hash(&mut h);
    crate::gpu::SceneKey {
        w_px,
        h_px,
        cam: cam_bits(&app.camera, aspect),
        preview: app.preview_delta.map_or(0, |d| hash_pod(&[d])),
        hover: app.hover.map_or(0, |i| i as u64 + 1),
        sel: hash_pod(sel),
        act: hash_pod(act),
        pvs: hash_pod(pvs),
        parts: hash_pod(parts),
        planes: app.plane_key,
        params: h.finish(),
        geo: app.geo_version,
    }
}

/// Ray ↔ plane (normal, a point on it).
pub fn plane_hit_n(nrm: Vec3, pt: Vec3, ro: Vec3, rd: Vec3) -> Option<Vec3> {
    let d = rd.dot(nrm);
    if d.abs() < 1e-9 {
        return None;
    }
    let t = (pt - ro).dot(nrm) / d;
    (t > 0.0).then(|| ro + rd * t)
}

/// What a move drag is constrained to. Live in `MoveDrag`, resolved from
/// the work plane + modifiers every frame so Shift/X/Y/Z switch it
/// mid-drag without a jump (see `reanchor`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DragPlane {
    /// The toolbar work plane (XZ/XY/YZ) through the anchor.
    Work(Plane),
    /// Shift: the vertical plane through the anchor that faces the
    /// camera — normal is the camera forward's horizontal part.
    CamVertical(Vec3),
    /// X/Y/Z axis lock: the drag rides the axis line through the anchor.
    Axis(u8),
}

/// The world axes, indexed by `DragPlane::Axis`.
pub const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

/// Pick the constraint for this frame: an axis lock beats Shift, which
/// beats the work plane. A straight-down camera can't form a vertical
/// plane, so Shift falls back to the work plane.
pub fn drag_plane(work: Plane, shift: bool, axis: Option<u8>, cam_fwd: Vec3) -> DragPlane {
    if let Some(k) = axis {
        return DragPlane::Axis(k.min(2));
    }
    if shift {
        let h = Vec3::new(cam_fwd.x, 0.0, cam_fwd.z);
        if h.length_squared() > 1e-6 {
            return DragPlane::CamVertical(h.normalize());
        }
    }
    DragPlane::Work(work)
}

/// Closest point on the line `anchor + s·axis` to the cursor ray — the
/// hit "on" an axis lock.
pub fn axis_hit(anchor: Vec3, axis: Vec3, ro: Vec3, rd: Vec3) -> Option<Vec3> {
    let w0 = ro - anchor;
    let b = rd.dot(axis);
    let denom = 1.0 - b * b; // rd,axis unit ⇒ a=c=1
    if denom.abs() < 1e-6 {
        return None; // ray parallel to the axis
    }
    let s = (axis.dot(w0) - b * rd.dot(w0)) / denom;
    Some(anchor + axis * s)
}

/// Cursor-ray hit on the drag constraint through `anchor`.
pub fn drag_hit(dp: DragPlane, anchor: Vec3, ro: Vec3, rd: Vec3) -> Option<Vec3> {
    match dp {
        DragPlane::Work(p) => plane_hit(p, anchor, ro, rd),
        DragPlane::CamVertical(n) => plane_hit_n(n, anchor, ro, rd),
        DragPlane::Axis(k) => axis_hit(anchor, AXES[k.min(2) as usize], ro, rd),
    }
}

/// What a constraint switch does to the in-flight delta: a plane keeps
/// it whole (no jump), an axis lock drops the off-axis part.
pub fn kept_delta(dp: DragPlane, delta: Vec3) -> Vec3 {
    match dp {
        DragPlane::Axis(k) => AXES[k.min(2) as usize] * delta.dot(AXES[k.min(2) as usize]),
        _ => delta,
    }
}

/// Standing bottom-strip hint for the current mode. Stage 15 C moved it
/// from in-canvas paint to a reserved panel so windows can't cover it.
pub fn mode_hint(mode: Mode) -> &'static str {
    match mode {
        Mode::Navigate => "navigate: click select · drag = box select · right-drag orbit · wheel zoom · WASD/QE pan · dbl-click focus · I insert · C connect · M move · Del delete",
        Mode::Insert => "insert: click to place a neuron on the work plane · Alt = inhibitory · Esc to exit",
        Mode::Connect => "connect: click a neuron or input source, then a target neuron · Esc to abort",
        Mode::Move => "move: drag a selected object on the work plane · Esc to exit",
    }
}

/// U8: the long hint wraps to a second row on narrow windows —
/// a terse variant keeps the strip one row under ~1100 px.
pub fn mode_hint_short(mode: Mode) -> &'static str {
    match mode {
        Mode::Navigate => "click select · right-drag orbit · wheel zoom · WASD pan",
        Mode::Insert => "click to place a neuron · Alt = inhibitory · Esc exit",
        Mode::Connect => "click source, then target · Esc abort",
        Mode::Move => "drag a selected object · Esc exit",
    }
}

/// Name for the hint strip ("move plane: …").
pub fn drag_plane_name(dp: DragPlane) -> String {
    match dp {
        DragPlane::Work(Plane::XZ) => "x-z".into(),
        DragPlane::Work(Plane::XY) => "x-y".into(),
        DragPlane::Work(Plane::YZ) => "y-z".into(),
        DragPlane::CamVertical(_) => "vertical".into(),
        DragPlane::Axis(k) => format!("{} axis", ["x", "y", "z"][k.min(2) as usize]),
    }
}

/// Project a world point to viewport-normalized [0,1] coords.
pub fn project(cam: &Camera, aspect: f32, p: Vec3) -> Option<Vec2> {
    let clip = cam.view_proj(aspect) * p.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5))
}

/// egui-wgpu paint callback: blit the scene texture into the viewport.
struct BlitScene;

impl egui_wgpu::CallbackTrait for BlitScene {
    fn paint(
        &self,
        info: PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        res: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = res.get::<Gpu>() else { return };
        let vp = info.viewport_in_pixels();
        gpu.blit(
            pass,
            [
                vp.left_px.max(0) as u32,
                vp.top_px.max(0) as u32,
                vp.width_px.max(0) as u32,
                vp.height_px.max(0) as u32,
            ],
        );
    }
}

pub fn viewport(app: &mut EditorApp, ctx: &egui::Context, rs: &egui_wgpu::RenderState) {
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            let (rect, resp) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            let (w, h) = (rect.width(), rect.height());
            if w < 8.0 || h < 8.0 {
                return;
            }
            let aspect = w / h;
            let w_px = (w * ctx.pixels_per_point()) as u32;
            let h_px = (h * ctx.pixels_per_point()) as u32;

            camera_input(app, ctx, &resp);
            keys(app, ui);

            let mouse_vp = resp
                .interact_pointer_pos()
                .or(resp.hover_pos())
                .map(|p| Vec2::new((p.x - rect.min.x) / w, (p.y - rect.min.y) / h));

            // ---- pick result first: frees the pick slot so a click this
            // frame can always be queued (the every-frame hover pick would
            // otherwise starve click picks ~half the time) ----
            let mut picked: Option<(bool, Option<u32>)> = None;
            with_gpu(rs, |gpu| {
                picked = gpu.poll_pick();
            });
            if let Some((hover, res)) = picked {
                if hover {
                    app.hover = res;
                } else {
                    on_pick(app, res, ctx);
                }
            }

            // ---- pick request on click (and a cheap hover pick each frame
            // so the status bar + context menu know what's under the cursor).
            // Clicks queue in app.pick_click so an in-flight hover pick can't
            // drop them; the queue drains as soon as a pick slot frees.
            if resp.clicked()
                || (resp.drag_started_by(egui::PointerButton::Primary)
                    && app.eff_mode() == Mode::Move)
            {
                if let Some(mv) = mouse_vp {
                    app.pick_click = Some((mv, app.wall_ms));
                    // A collapsed block is a CPU-only glyph — remember it
                    // so a pick miss still lands on the group. A real
                    // entity hit beats it.
                    app.click_block = block_at(app, mv, rect, aspect);
                }
            }
            if let Some((mv, queued_ms)) = app.pick_click {
                if app.wall_ms - queued_ms > 800.0 {
                    app.pick_click = None;
                } else {
                    let px = (mv.x * w_px as f32).min(w_px as f32 - 1.0) as u32;
                    let py = (mv.y * h_px as f32).min(h_px as f32 - 1.0) as u32;
                    let mut sent = false;
                    with_gpu(rs, |gpu| {
                        sent = gpu.request_pick(px, py, w_px, h_px, false);
                    });
                    if sent {
                        app.pick_click = None;
                    }
                }
            } else if resp.hovered() && !resp.dragged() {
                if let Some(mv) = mouse_vp {
                    let px = (mv.x * w_px as f32).min(w_px as f32 - 1.0) as u32;
                    let py = (mv.y * h_px as f32).min(h_px as f32 - 1.0) as u32;
                    // Only re-pick when the target could differ — a still
                    // cursor under an unchanged scene needs no new pass.
                    let want = (px, py, cam_bits(&app.camera, aspect), app.geo_version);
                    if app.last_hover_pick != Some(want) {
                        let mut sent = false;
                        with_gpu(rs, |gpu| {
                            sent = gpu.request_pick(px, py, w_px, h_px, true);
                        });
                        if sent {
                            app.last_hover_pick = Some(want);
                        }
                    }
                }
            }

            context_menu(app, &resp);

            // ---- mode interactions ----
            mode_input(app, &resp, rect, mouse_vp, aspect);

            // Cursor feedback: grab/grabbing on selected objects, hand on
            // any object, crosshair in the placement modes.
            let icon = if app.move_drag.is_some() {
                egui::CursorIcon::Grabbing
            } else {
                match app.eff_mode() {
                    Mode::Insert | Mode::Connect => egui::CursorIcon::Crosshair,
                    Mode::Move => egui::CursorIcon::Move,
                    Mode::Navigate => match app.hover {
                        Some(i) if app.selection.contains(&i) => egui::CursorIcon::Grab,
                        Some(_) => egui::CursorIcon::PointingHand,
                        None => egui::CursorIcon::Default,
                    },
                }
            };
            ui.output_mut(|o| o.cursor_icon = icon);

            // ---- uploads + offscreen scene render ----
            let sel: Vec<u32> = app.selection.iter().copied().collect();
            let mut pvs = Vec::new();
            if let Some(md) = &app.move_drag {
                // The live drag constraint, through the object's current
                // (preview) position — brighter than the idle work plane.
                let cur = md.anchor + app.preview_delta.unwrap_or(Vec3::ZERO);
                let (axes, axis_line) = match md.plane {
                    DragPlane::Work(p) => (Some(scene::plane_axes(p)), None),
                    DragPlane::CamVertical(n) => {
                        (Some((n.cross(Vec3::Y).normalize(), Vec3::Y)), None)
                    }
                    DragPlane::Axis(k) => (None, Some(AXES[k.min(2) as usize])),
                };
                scene::drag_plane_vis(&mut pvs, cur, md.anchor, axes, axis_line);
            } else if app.eff_mode() == Mode::Move || app.eff_mode() == Mode::Insert {
                scene::work_plane(app.camera.target, app.plane, &mut pvs);
            }
            if let Some(g) = &app.geo {
                // Collapsed blocks draw their own glyph (overlay_2d) —
                // no member box; the rest highlight on full selection.
                let skip: std::collections::HashSet<usize> = app
                    .group_ids_mode(crate::app::GroupMode::Block)
                    .into_iter()
                    .collect();
                scene::group_boxes(g, &skip, &app.selection, &mut pvs);
                // input → target links: all of them on small nets, only the
                // selected input's on big ones (they'd double the line load).
                let show_all_inputs = g.syn.len() < 10_000 || g.inputs.len() <= 4;
                for (k, inp) in g.inputs.iter().enumerate() {
                    if !app.show_input_links {
                        break;
                    }
                    let inst = (g.pos.len() + k) as u32;
                    if !show_all_inputs && !app.selection.contains(&inst) {
                        continue;
                    }
                    if let Some(src) = inst_pos(g, inst) {
                        for &t in &inp.targets {
                            if let Some(&p) = g.pos.get(t as usize) {
                                pvs.push(scene::pv(src.into(), INPUT_LINK_COL));
                                pvs.push(scene::pv(p, INPUT_LINK_COL));
                            }
                        }
                    }
                }
                if app.eff_mode() == Mode::Connect {
                    if let (Some(from), Some(mv)) = (app.connect_from, mouse_vp) {
                        if let Some(fp) = inst_pos(g, from) {
                            let (ro, rd) = ray(&app.camera, aspect, mv.x, mv.y);
                            let to = plane_hit(app.plane, app.camera.target, ro, rd).unwrap_or(fp);
                            scene::connect_line(&mut pvs, fp, to);
                        }
                    }
                }
            }
            let act = app.activity.clone();
            // Flights live on the wall clock (MIN_FLIGHT_MS..MAX_FLIGHT_MS);
            // sim time only feeds the rewind detector.
            let parts = if app.show_dots {
                app.particles.instances(app.wall_ms)
            } else {
                Vec::new()
            };
            let pv_n = pvs.len() as u32;
            let show_syn = app.show_synapses;
            with_gpu(rs, |gpu| {
                gpu.write_cam(
                    &app.camera,
                    aspect,
                    h_px as f32,
                    ctx.input(|i| i.time) as f32,
                    app.preview_delta,
                    app.hover,
                );
                gpu.write_activity(&act);
                gpu.write_selection(&sel);
                gpu.write_pv(&pvs);
                // Grid planes: per-frame params/hide flags plus trace
                // uploads when the snap or the display config changed.
                let planes_on = app.plane_frame(gpu, h_px as f32);
                let np = gpu.write_parts(&parts) as u32;
                gpu.draw_params = Some(DrawParams {
                    pv_count: pv_n,
                    part_count: np,
                    show_syn,
                    show_grid: app.show_grid,
                    planes_on: planes_on.clone(),
                });
                let key = scene_key(
                    app, w_px, h_px, aspect, &sel, &act, &pvs, &parts, show_syn, &planes_on,
                );
                gpu.render_offscreen(w_px, h_px, &key);
                // Unfinished pick readbacks and in-flight particles keep the
                // repaint alive even when the sim is paused (Stage 14 E).
                app.pick_in_flight = gpu.pick_in_flight();
            });
            app.parts_active = !parts.is_empty();

            ui.painter()
                .add(egui_wgpu::Callback::new_paint_callback(rect, BlitScene));

            overlay_2d(app, ui, rect, aspect);

            if app.show_fps {
                ui.painter().text(
                    rect.right_top() + egui::vec2(-6.0, 6.0),
                    egui::Align2::RIGHT_TOP,
                    format!(
                        "{:5.1} ms  {:3.0} fps",
                        app.frame_ms,
                        1000.0 / app.frame_ms.max(0.01)
                    ),
                    egui::FontId::monospace(12.0),
                    egui::Color32::LIGHT_GRAY,
                );
            }

            if let Some((a, b)) = app.box_sel {
                ui.painter().rect_stroke(
                    egui::Rect::from_two_pos(a, b),
                    0.0,
                    egui::Stroke::new(1.0_f32, egui::Color32::LIGHT_BLUE),
                    egui::StrokeKind::Outside,
                );
            }
        });
}

/// Screen-drag → orbit delta. Stage 13 I: the horizontal sign is
/// flipped vs the naive mapping — a right drag sweeps the view the way
/// the original's camera did (scene swings left, so dragging right
/// feels like turning the scene, not the camera). Vertical stays as
/// before (drag down = look up).
fn orbit_delta(d: egui::Vec2) -> (f32, f32) {
    (-d.x, -d.y)
}

fn camera_input(app: &mut EditorApp, ctx: &egui::Context, resp: &egui::Response) {
    let dt = ctx.input(|i| i.stable_dt.min(0.05));
    app.camera.tick(dt);
    if resp.dragged_by(egui::PointerButton::Secondary) {
        let (dx, dy) = orbit_delta(resp.drag_delta());
        app.camera.orbit(dx, dy);
    }
    let wheel = ctx.input(|i| i.smooth_scroll_delta.y);
    if resp.hovered() && wheel != 0.0 {
        app.camera.zoom(wheel);
    }
    let (mut pr, mut pf, mut pu) = (0f32, 0f32, 0f32);
    let typing = ctx.wants_keyboard_input();
    ctx.input(|i| {
        if typing {
            return;
        }
        if i.key_down(egui::Key::W) {
            pf += 1.0;
        }
        if i.key_down(egui::Key::S) {
            pf -= 1.0;
        }
        if i.key_down(egui::Key::A) {
            pr -= 1.0;
        }
        if i.key_down(egui::Key::D) {
            pr += 1.0;
        }
        if i.key_down(egui::Key::Q) {
            pu -= 1.0;
        }
        if i.key_down(egui::Key::E) {
            pu += 1.0;
        }
    });
    if pr != 0.0 || pf != 0.0 || pu != 0.0 {
        app.camera.pan(pr, pf, pu, dt);
    }
}

fn keys(app: &mut EditorApp, ui: &mut egui::Ui) {
    if ui.ctx().wants_keyboard_input() {
        return; // a text field owns the keyboard — don't fire shortcuts
    }
    ui.input(|i| {
        if i.key_pressed(egui::Key::Escape) {
            if app.move_drag.is_some() {
                // Cancel: the drag only ever drove the shader preview, so
                // dropping it snaps everything back to its real position.
                app.move_drag = None;
                app.preview_delta = None;
                app.status.clear();
            } else if app.connect_from.is_some() {
                app.connect_from = None;
            } else {
                app.mode = Mode::Navigate;
            }
        }
        // Edit-mode keys are gated like their buttons (Stage 22): while
        // running/busy they don't fire — the reason shows on the tool
        // buttons' disabled tooltips.
        if app.can(Act::Edit) {
            if i.key_pressed(egui::Key::Delete) {
                app.delete_selection();
            }
            if i.key_pressed(egui::Key::I) {
                app.mode = Mode::Insert;
            }
            if i.key_pressed(egui::Key::C) {
                app.mode = Mode::Connect;
            }
            if i.key_pressed(egui::Key::M) {
                app.mode = Mode::Move;
            }
        }
        if i.key_pressed(egui::Key::V) && !i.modifiers.ctrl {
            app.mode = Mode::Navigate;
        }
        if i.key_pressed(egui::Key::F) {
            app.frame_all();
        }
        if i.key_pressed(egui::Key::Questionmark) {
            app.win.manual = true;
        }
        // The console dock gets a bare key too — ` is the terminal
        // convention, Ctrl+3 keeps working (Stage 22).
        if i.key_pressed(egui::Key::Backtick) {
            app.win.console = !app.win.console;
        }
        if i.modifiers.ctrl {
            for (key, flag) in [
                (egui::Key::Num1, &mut app.win.inspector),
                (egui::Key::Num2, &mut app.win.plots),
                (egui::Key::Num3, &mut app.win.console),
                (egui::Key::Num4, &mut app.win.script),
                (egui::Key::Num5, &mut app.win.inputs),
                (egui::Key::Num6, &mut app.win.files),
            ] {
                if i.key_pressed(key) {
                    *flag = !*flag;
                }
            }
        }
        if i.modifiers.ctrl && i.key_pressed(egui::Key::C) {
            app.copy_selection(false);
        }
        if i.modifiers.ctrl && i.key_pressed(egui::Key::X) && app.can(Act::Edit) {
            app.copy_selection(true);
        }
        if i.modifiers.ctrl && i.key_pressed(egui::Key::V) && app.can(Act::Edit) {
            app.paste(None);
        }
        // Stage 23: Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y → the worker's shared
        // stack. The methods re-check the edit gate themselves, so a
        // blocked undo lands in the status line instead of misfiring.
        if i.modifiers.ctrl && !i.modifiers.shift && i.key_pressed(egui::Key::Z) {
            app.undo();
        }
        if i.modifiers.ctrl
            && ((i.modifiers.shift && i.key_pressed(egui::Key::Z))
                || (!i.modifiers.shift && i.key_pressed(egui::Key::Y)))
        {
            app.redo();
        }
    });
}

/// Right-click context menu on the entity under the cursor (or empty space).
fn context_menu(app: &mut EditorApp, resp: &egui::Response) {
    resp.context_menu(|ui| {
        let ent = app
            .hover
            .and_then(|i| app.geo.as_ref().and_then(|g| entity_of(g, i)));
        let why_edit = app.why(Act::Edit);
        let gated = |ui: &mut egui::Ui, label: &str| {
            crate::app::gate(ui, why_edit, egui::Button::new(label))
        };
        match ent {
            Some(Entity::Neuron(n)) => {
                ui.label(format!("neuron {n}"));
                ui.separator();
                if ui.button("Watch v(t)").clicked() {
                    app.watch(vec![n]);
                    ui.close();
                }
                if gated(ui, "Connect from here").clicked() {
                    app.mode = Mode::Connect;
                    app.connect_from = app.hover;
                    ui.close();
                }
                if gated(ui, "Delete").clicked() {
                    app.delete_entity(n);
                    ui.close();
                }
            }
            Some(Entity::Input(k)) => {
                if let Some(g) = &app.geo {
                    ui.label(format!("input {}", g.inputs[k as usize].name));
                }
                ui.separator();
                // Live input values stay editable during a run
                // (SetInputConst is applied to the live sim).
                if crate::app::gate(ui, app.why(Act::SetInput), egui::Button::new("Set value…"))
                    .clicked()
                {
                    app.open_input_dialog(k);
                    ui.close();
                }
                if gated(ui, "Connect from here").clicked() {
                    app.mode = Mode::Connect;
                    app.connect_from = app.hover;
                    ui.close();
                }
            }
            Some(Entity::Readout(k)) => {
                if let Some(g) = &app.geo {
                    ui.label(format!("readout {}", g.readouts[k as usize].0));
                }
                ui.separator();
                if ui.button("Plot weights").clicked() {
                    app.plot_readout_weights(k);
                    ui.close();
                }
            }
            None => {
                ui.label("(empty space)");
                ui.separator();
                if gated(ui, "Insert neuron here").clicked() {
                    app.mode = Mode::Insert;
                    ui.close();
                }
                if gated(ui, "Paste").clicked() {
                    app.paste(None);
                    ui.close();
                }
            }
        }
        ui.separator();
        if ui.button("Frame network").clicked() {
            app.frame_all();
            ui.close();
        }
    });
}

fn on_pick(app: &mut EditorApp, res: Option<u32>, ctx: &egui::Context) {
    // The block candidate belongs to THIS pick only — taken once so a
    // stale flag can't apply to a later click.
    let block_members = app.click_block.take().and_then(|gi| {
        app.geo
            .as_ref()
            .and_then(|g| g.groups.get(gi))
            .map(|(_, m)| m.clone())
    });
    match app.eff_mode() {
        Mode::Connect => {
            // Only a hit on a neuron/input changes state: a missed click
            // keeps the pending source (dense scenes make small cubes easy
            // to miss), so Esc is the only way to abort a pending connection.
            if let Some(inst) = res {
                let ent = app.geo.as_ref().and_then(|g| entity_of(g, inst));
                match ent {
                    Some(Entity::Neuron(n)) => {
                        match app
                            .connect_from
                            .take()
                            .and_then(|fi| app.geo.as_ref().and_then(|g| entity_of(g, fi)))
                        {
                            Some(Entity::Neuron(a)) => app.connect_neurons(a, n),
                            Some(Entity::Input(k)) => app.connect_input(k, n),
                            _ => {
                                app.connect_from = Some(inst);
                                app.status = "connect: pick target".into();
                            }
                        }
                    }
                    Some(Entity::Input(_)) => {
                        // Inputs are sources, never targets.
                        app.connect_from = Some(inst);
                        app.status = "connect: pick a target neuron".into();
                    }
                    _ => {
                        app.status = "connect: that's not a neuron or input".into();
                    }
                }
            } else if app.connect_from.is_some() {
                app.status = "connect: missed — click a target neuron".into();
            }
        }
        Mode::Move => {
            // A click in Move mode selects what it hit (drags start in
            // mode_input; this pick is the click that launched them).
            if let Some(inst) = res {
                if !app.selection.contains(&inst) {
                    app.click_select(inst, false);
                }
                app.last_pick = Some(inst);
            } else if let Some(members) = block_members {
                // A collapsed block is always whole-group: clicking its
                // glyph in Move mode selects the members it stands for.
                app.selection = members.into_iter().collect();
            }
        }
        _ => {
            // Navigate: select/clear; dbl-click focus handled separately
            let shift = ctx.input(|i| i.modifiers.shift);
            match res {
                Some(inst) => {
                    // Whole-group mode: a member click takes its group.
                    app.click_select(inst, shift);
                    app.last_pick = Some(inst);
                }
                None => {
                    if !shift {
                        app.selection.clear();
                    }
                    // A pick miss on a collapsed block selects its group.
                    if let Some(members) = block_members {
                        app.selection.extend(members);
                    }
                    app.last_pick = None;
                }
            }
        }
    }
}

/// Input→target links use a dimmed input gold (Stage 15 D: these are the
/// gold lines seen when nothing is selected — the selection buffer is
/// zeroed each frame, so stale selection highlight cannot produce them).
/// U7: dimmed so dense input fans stop dominating the default view.
const INPUT_LINK_COL: [f32; 4] = crate::scene::rgba(crate::scene::INPUT_COL, 0.35);

fn mode_input(
    app: &mut EditorApp,
    resp: &egui::Response,
    rect: egui::Rect,
    mouse_vp: Option<Vec2>,
    aspect: f32,
) {
    match app.eff_mode() {
        Mode::Insert => {
            if resp.clicked() {
                if let Some(mv) = mouse_vp {
                    let (ro, rd) = ray(&app.camera, aspect, mv.x, mv.y);
                    if let Some(hit) = plane_hit(app.plane, app.camera.target, ro, rd) {
                        app.insert_neuron(hit);
                    }
                }
            }
        }
        Mode::Move | Mode::Navigate => {
            let nav = app.eff_mode() == Mode::Navigate;
            if resp.drag_started_by(egui::PointerButton::Primary) {
                if let Some(mv) = mouse_vp {
                    let (ro, rd) = ray(&app.camera, aspect, mv.x, mv.y);
                    // What did the press land on? The hover pick is a frame
                    // stale — fall back to a CPU ray test so the very first
                    // drag in a session still grabs what's under the cursor.
                    // Collapsed-block members are invisible: the ray test
                    // skips them. The glyph is their handle — a press on
                    // it counts as grabbing the group's first member, so
                    // Move and Navigate modes treat it identically to a
                    // visible member.
                    let hidden = app.block_members();
                    let mut under = app.hover.or_else(|| {
                        app.geo
                            .as_ref()
                            .and_then(|g| nearest_on_ray(g, ro, rd, 6.0, &hidden))
                    });
                    if under.is_none() {
                        under = block_at(app, mv, rect, aspect).and_then(|gi| {
                            app.geo
                                .as_ref()
                                .and_then(|g| g.groups.get(gi)?.1.first().copied())
                        });
                    }
                    match primary_drag_action(app.eff_mode(), under) {
                        DragAction::Move { grab } => {
                            if let Some(i) = grab {
                                if nav && !app.selection.contains(&i) {
                                    // Drag on an unselected object
                                    // selects it, then moves.
                                    if !resp.ctx.input(|i| i.modifiers.shift) {
                                        app.selection.clear();
                                    }
                                }
                                // Whole-group: a grab always moves the
                                // whole group, selected or not. In Move
                                // mode ungrouped entities keep the old
                                // "drag moves the selection" rule — only
                                // a real group expands the set.
                                let ex = app.group_expand(i);
                                if nav || ex.len() > 1 {
                                    app.selection.extend(ex);
                                }
                            }
                            begin_move(app, grab, mv, aspect);
                        }
                        DragAction::Box => {
                            if let Some(p) = resp.interact_pointer_pos() {
                                app.box_sel = Some((p, p));
                            }
                        }
                    }
                }
            }
            if resp.dragged_by(egui::PointerButton::Primary) {
                if let (Some(mv), Some(mut md)) = (mouse_vp, app.move_drag.clone()) {
                    let (ro, rd) = ray(&app.camera, aspect, mv.x, mv.y);
                    // X/Y/Z toggle an axis lock; Shift picks the vertical
                    // camera-facing plane. Switching re-anchors the drag
                    // zero so the object doesn't jump.
                    let (shift, kx, ky, kz) = resp.ctx.input(|i| {
                        let bare = !(i.modifiers.ctrl || i.modifiers.command || i.modifiers.alt);
                        (
                            i.modifiers.shift,
                            bare && i.key_pressed(egui::Key::X),
                            bare && i.key_pressed(egui::Key::Y),
                            bare && i.key_pressed(egui::Key::Z),
                        )
                    });
                    for (k, hit) in [(0u8, kx), (1, ky), (2, kz)] {
                        if hit {
                            md.axis = (md.axis != Some(k)).then_some(k);
                        }
                    }
                    let dp = drag_plane(
                        app.plane,
                        shift,
                        md.axis,
                        app.camera.target - app.camera.eye(),
                    );
                    if dp != md.plane {
                        if let Some(h) = drag_hit(dp, md.anchor, ro, rd) {
                            let d = kept_delta(dp, app.preview_delta.unwrap_or(Vec3::ZERO));
                            md.plane_pt = h - d;
                            app.preview_delta = Some(d);
                        }
                        md.plane = dp;
                    }
                    // Constraint through the grabbed object — it stays
                    // under the cursor for the whole drag.
                    if let Some(hit) = drag_hit(md.plane, md.anchor, ro, rd) {
                        let delta = hit - md.plane_pt;
                        app.preview_delta = Some(delta);
                        app.status =
                            format!("move d({:.0},{:.0},{:.0})", delta.x, delta.y, delta.z);
                    }
                    app.move_drag = Some(md);
                } else if let (Some((a, _)), Some(cur)) = (app.box_sel, resp.interact_pointer_pos())
                {
                    app.box_sel = Some((a, cur));
                }
            }
            if resp.drag_stopped_by(egui::PointerButton::Primary) {
                if let (Some(md), Some(d)) = (app.move_drag.take(), app.preview_delta.take()) {
                    app.apply_move(&md, d);
                    app.status.clear();
                } else if let Some((a, b)) = app.box_sel.take() {
                    let shift = resp.ctx.input(|i| i.modifiers.shift);
                    box_select(app, rect, a, b, aspect, shift);
                }
            }
            // double-click focus: CPU nearest on ray
            if resp.double_clicked() {
                if let (Some(g), Some(mv)) = (&app.geo, mouse_vp) {
                    let (ro, rd) = ray(&app.camera, aspect, mv.x, mv.y);
                    let hidden = app.block_members();
                    match nearest_on_ray(g, ro, rd, 25.0, &hidden) {
                        Some(inst) => {
                            if let Some(p) = inst_pos(g, inst) {
                                app.camera.focus(p);
                            }
                        }
                        None => {
                            // … or a collapsed block under the cursor —
                            // focus its centre.
                            if let Some(gi) = block_at(app, mv, rect, aspect) {
                                if let Some((mn, mx)) = EditorApp::group_aabb(g, &g.groups[gi].1) {
                                    app.camera.focus((mn + mx) * 0.5);
                                }
                            } else if let Some(hit) =
                                plane_hit(app.plane, app.camera.target, ro, rd)
                            {
                                app.camera.focus(hit);
                            }
                        }
                    }
                }
            }
        }
        Mode::Connect => {}
    }
}

/// What a primary-button drag means in this mode (pure — unit tested):
/// drag on an entity moves it (selecting first when unselected); drag on
/// empty space box-selects. Move mode is always-move, even on empty space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragAction {
    Move { grab: Option<u32> },
    Box,
}

pub fn primary_drag_action(mode: Mode, under: Option<u32>) -> DragAction {
    match (mode, under) {
        (Mode::Move, Some(i)) | (Mode::Navigate, Some(i)) => DragAction::Move { grab: Some(i) },
        (Mode::Move, None) => DragAction::Move { grab: None },
        _ => DragAction::Box,
    }
}

/// Start a move drag: the work plane is anchored at the grabbed object's
/// position (selection centroid for an empty-space Move-mode drag) so the
/// dragged objects track the cursor exactly.
fn begin_move(app: &mut EditorApp, grab: Option<u32>, mv: Vec2, aspect: f32) {
    let Some(g) = app.geo.clone() else { return };
    // (id, pos) pairs — keep them aligned (a positionless readout must not
    // shift the other ids' start positions).
    let pairs: Vec<(u32, Vec3)> = app
        .selection
        .iter()
        .filter_map(|&i| inst_pos(&g, i).map(|p| (i, p)))
        .collect();
    if pairs.is_empty() {
        return;
    }
    let anchor = grab
        .and_then(|i| inst_pos(&g, i))
        .or_else(|| {
            let c = pairs.iter().fold(Vec3::ZERO, |a, (_, p)| a + *p);
            (pairs.len() > 0).then(|| c / pairs.len() as f32)
        })
        .unwrap_or(app.camera.target);
    let (ro, rd) = ray(&app.camera, aspect, mv.x, mv.y);
    let dp = drag_plane(app.plane, false, None, app.camera.target - app.camera.eye());
    if let Some(hit) = drag_hit(dp, anchor, ro, rd) {
        app.move_drag = Some(MoveDrag {
            anchor,
            plane_pt: hit,
            plane: dp,
            axis: None,
            start_pos: pairs.iter().map(|(_, p)| *p).collect(),
            ids: pairs.iter().map(|(i, _)| *i).collect(),
        });
        app.preview_delta = Some(Vec3::ZERO);
    }
}

/// Nearest entity to a ray within `max_t` world units (double-click focus).
/// `skip` holds neuron ids that must not answer — collapsed-block members
/// are invisible, so only their glyph should catch the cursor.
fn nearest_on_ray(
    g: &GeoSnap,
    ro: Vec3,
    rd: Vec3,
    max_t: f32,
    skip: &std::collections::HashSet<u32>,
) -> Option<u32> {
    let mut best = None;
    let mut best_d = max_t;
    for (i, p) in g.pos.iter().enumerate() {
        if skip.contains(&(i as u32)) {
            continue;
        }
        let c = Vec3::from(*p) - ro;
        let t = c.dot(rd);
        if !(0.0..=4000.0).contains(&t) {
            continue;
        }
        let d = (c - rd * t).length();
        if d < best_d {
            best_d = d;
            best = Some(i as u32);
        }
    }
    best
}

fn box_select(
    app: &mut EditorApp,
    rect: egui::Rect,
    a: egui::Pos2,
    b: egui::Pos2,
    aspect: f32,
    additive: bool,
) {
    let Some(g) = app.geo.clone() else { return };
    let (min, max) = (a.min(b), a.max(b));
    if !additive {
        app.selection.clear();
    }
    let rw = rect.width();
    let rh = rect.height();
    let total = g.pos.len() + g.inputs.len() + g.readouts.len();
    for i in 0..total as u32 {
        if let Some(p) = inst_pos(&g, i) {
            if let Some(nd) = project(&app.camera, aspect, p) {
                let sx = rect.min.x + nd.x * rw;
                let sy = rect.min.y + nd.y * rh;
                if sx >= min.x && sx <= max.x && sy >= min.y && sy <= max.y {
                    app.selection.insert(i);
                }
            }
        }
    }
    // Stage 23: a box that catches any member of an atomic group
    // catches the whole group. (Field-disjoint form: `g` is a local Arc,
    // `&mut selection` must not trip the `&self` borrow.)
    EditorApp::expand_sel_groups_in(
        &app.group_modes,
        app.group_mode_default,
        &g,
        &mut app.selection,
    );
}

/// Project an AABB to its screen rect — the block glyph's drawn region
/// AND its click target, so hit-testing and paint can't drift apart.
pub(crate) fn aabb_screen_rect(
    cam: &Camera,
    aspect: f32,
    vp: egui::Rect,
    mn: Vec3,
    mx: Vec3,
) -> Option<egui::Rect> {
    let mut mn_s = egui::pos2(f32::MAX, f32::MAX);
    let mut mx_s = egui::pos2(f32::MIN, f32::MIN);
    let mut any = false;
    for i in 0..8 {
        let p = Vec3::new(
            if i & 1 == 0 { mn.x } else { mx.x },
            if i & 2 == 0 { mn.y } else { mx.y },
            if i & 4 == 0 { mn.z } else { mx.z },
        );
        if let Some(nd) = project(cam, aspect, p) {
            let s = egui::pos2(vp.min.x + nd.x * vp.width(), vp.min.y + nd.y * vp.height());
            mn_s = mn_s.min(s);
            mx_s = mx_s.max(s);
            any = true;
        }
    }
    any.then(|| egui::Rect::from_min_max(mn_s, mx_s))
}

/// Collapsed-block group under a viewport-normalized cursor: the
/// smallest containing glyph wins so overlapping blocks stay pickable.
fn block_at(app: &EditorApp, mv: Vec2, rect: egui::Rect, aspect: f32) -> Option<usize> {
    let g = app.geo.as_ref()?;
    let pt = egui::pos2(
        rect.min.x + mv.x * rect.width(),
        rect.min.y + mv.y * rect.height(),
    );
    let mut best: Option<(usize, f32)> = None;
    for gi in app.group_ids_mode(crate::app::GroupMode::Block) {
        let Some((mn, mx)) = EditorApp::group_aabb(g, &g.groups[gi].1) else {
            continue;
        };
        let Some(r) = aabb_screen_rect(&app.camera, aspect, rect, mn, mx) else {
            continue;
        };
        if r.contains(pt) {
            let area = r.width() * r.height();
            if best.is_none_or(|(_, a)| area < a) {
                best = Some((gi, area));
            }
        }
    }
    best.map(|(gi, _)| gi)
}

/// Entity world position (mirrors Gpu::set_scene's marker placement).
fn inst_pos(g: &GeoSnap, inst: u32) -> Option<Vec3> {
    match entity_of(g, inst) {
        Some(Entity::Neuron(i)) => g.pos.get(i as usize).map(|p| Vec3::from(*p)),
        Some(Entity::Input(k)) => {
            let inp = &g.inputs[k as usize];
            let mut p = inp.pos;
            if p == [0.0; 3] && !inp.targets.is_empty() {
                let mut c = Vec3::ZERO;
                for &t in &inp.targets {
                    c += Vec3::from(g.pos[t as usize]);
                }
                p = (c / inp.targets.len() as f32 + Vec3::Y * 60.0).into();
            }
            Some(Vec3::from(p))
        }
        Some(Entity::Readout(k)) => {
            let units = &g.readouts[k as usize].1;
            if units.is_empty() {
                return None;
            }
            let mut c = Vec3::ZERO;
            for &u in units {
                c += Vec3::from(g.pos[u as usize]);
            }
            Some(c / units.len() as f32 + Vec3::Y * 120.0)
        }
        None => None,
    }
}

/// Selection corner brackets + mode hint.
fn overlay_2d(app: &EditorApp, ui: &mut egui::Ui, rect: egui::Rect, aspect: f32) {
    let painter = ui.painter();
    const SEL_COL: egui::Color32 = egui::Color32::from_rgb(255, 120, 40);
    /// Past this many selected objects, per-object brackets turn to noise —
    /// draw one bracket around their union plus a count instead.
    const BIG_SEL: usize = 24;
    if let Some(g) = &app.geo {
        let big = app.selection.len() > BIG_SEL;
        let mut union: Option<egui::Rect> = None;
        for &id in &app.selection {
            let Some(p) = inst_pos(g, id) else { continue };
            let half = app.inst_half.get(id as usize).copied().unwrap_or(4.0) * 1.15;
            if let Some(r) = screen_box_rect(&app.camera, aspect, rect, p, half) {
                union = Some(union.map_or(r, |u: egui::Rect| u.union(r)));
                if !big {
                    draw_bracket(painter, r, SEL_COL);
                }
            }
        }
        if big {
            if let Some(r) = union {
                draw_bracket(painter, r, SEL_COL);
                painter.text(
                    egui::pos2(r.center().x, r.min.y - 4.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("{} selected", app.selection.len()),
                    egui::FontId::proportional(11.0),
                    SEL_COL,
                );
            }
        }
        // Stage 23: collapsed groups render as a labelled block at the
        // members' projected bounds — fill + name, brighter when the
        // whole group is selected.
        for gi in app.group_ids_mode(crate::app::GroupMode::Block) {
            let (name, members) = &g.groups[gi];
            let Some((mn, mx)) = EditorApp::group_aabb(g, members) else {
                continue;
            };
            let Some(r) = aabb_screen_rect(&app.camera, aspect, rect, mn, mx) else {
                continue;
            };
            let sel = app.group_fully_selected(members);
            let (fill, edge) = if sel {
                (
                    egui::Color32::from_rgba_unmultiplied(46, 46, 30, 200),
                    egui::Color32::from_rgb(255, 225, 80),
                )
            } else {
                (
                    egui::Color32::from_rgba_unmultiplied(38, 40, 54, 200),
                    egui::Color32::from_rgb(205, 190, 100),
                )
            };
            painter.rect_filled(r, 6.0, fill);
            painter.rect_stroke(
                r,
                6.0,
                egui::Stroke::new(if sel { 2.0 } else { 1.4_f32 }, edge),
                egui::StrokeKind::Inside,
            );
            painter.text(
                r.center(),
                egui::Align2::CENTER_CENTER,
                format!("{name}\n{} neurons", members.len()),
                egui::FontId::proportional(13.0),
                egui::Color32::from_rgb(235, 225, 160),
            );
        }
    }
    // Empty scene: a centered hint so a first launch isn't a blank void.
    let empty = app.geo.as_ref().map_or(true, |g| g.pos.is_empty());
    if empty {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "empty network\n\nFile > Load a .ne2, or build one in the console below:\nnet.create_2d_grid(\"g\", 100, 100, 5)",
            egui::FontId::monospace(14.0),
            egui::Color32::from_gray(150),
        );
    }
}

/// Project an instance's world-space cube (centre ± `half`) to its screen
/// bounding rect — the bracket hugs the object instead of orbiting its
/// centre at a fixed pixel radius (the original does the same with the
/// object radius in glscene.cpp's selection pass).
pub fn screen_box_rect(
    cam: &Camera,
    aspect: f32,
    vp: egui::Rect,
    center: Vec3,
    half: f32,
) -> Option<egui::Rect> {
    let mut mn = egui::pos2(f32::MAX, f32::MAX);
    let mut mx = egui::pos2(f32::MIN, f32::MIN);
    let mut any = false;
    for dx in [-half, half] {
        for dy in [-half, half] {
            for dz in [-half, half] {
                if let Some(nd) = project(cam, aspect, center + Vec3::new(dx, dy, dz)) {
                    let s = egui::pos2(vp.min.x + nd.x * vp.width(), vp.min.y + nd.y * vp.height());
                    mn = mn.min(s);
                    mx = mx.max(s);
                    any = true;
                }
            }
        }
    }
    any.then(|| egui::Rect::from_min_max(mn, mx))
}

/// Corner ticks on the rect's four corners.
fn draw_bracket(painter: &egui::Painter, r: egui::Rect, col: egui::Color32) {
    let t = (r.width().min(r.height()) * 0.3).clamp(4.0, 12.0);
    let st = egui::Stroke::new(1.5_f32, col);
    for (cx, sx) in [(r.min.x, 1f32), (r.max.x, -1f32)] {
        for (cy, sy) in [(r.min.y, 1f32), (r.max.y, -1f32)] {
            let c = egui::pos2(cx, cy);
            painter.line_segment([c, egui::pos2(cx + sx * t, cy)], st);
            painter.line_segment([c, egui::pos2(cx, cy + sy * t)], st);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn cam() -> Camera {
        let mut c = Camera::default();
        c.target = Vec3::new(0.0, 10.0, 0.0);
        c.dist = 200.0;
        c.yaw_deg = -35.0;
        c.pitch_deg = -25.0;
        c
    }

    #[test]
    fn drag_plane_keeps_object_under_cursor() {
        let c = cam();
        let aspect = 1.6;
        let obj = Vec3::new(30.0, -40.0, 15.0); // deliberately off camera.target
                                                // press ray through the object's centre on screen
        let nd = project(&c, aspect, obj).unwrap();
        let (ro, rd) = ray(&c, aspect, nd.x, nd.y);
        // plane anchored AT the object: the drag-start hit is the object
        // itself, so the initial delta is ~0 …
        let hit0 = plane_hit(Plane::XZ, obj, ro, rd).unwrap();
        assert!((hit0 - obj).length() < 1e-2);
        // … and after the cursor moves, obj + (hit1 - hit0) sits exactly on
        // the new cursor ray — the object tracks the pointer 1:1.
        let (ro1, rd1) = ray(&c, aspect, nd.x + 0.1, nd.y + 0.05);
        let hit1 = plane_hit(Plane::XZ, obj, ro1, rd1).unwrap();
        let obj_new = obj + (hit1 - hit0);
        assert!((obj_new - hit1).length() < 1e-2);
        // anchored at camera.target instead (the old behaviour): the start
        // hit is offset from the object — it would visibly jump on grab.
        let hit_bad = plane_hit(Plane::XZ, c.target, ro, rd);
        assert!(hit_bad.is_none() || (hit_bad.unwrap() - obj).length() > 1.0);
    }

    /// Stage 13 I: horizontal orbit is flipped — dragging RIGHT
    /// (d.x > 0) yields a negative yaw delta; vertical keeps its sign.
    #[test]
    fn orbit_delta_flips_horizontal_only() {
        let (dx, dy) = orbit_delta(egui::vec2(10.0, 4.0));
        assert_eq!(dx, -10.0);
        assert_eq!(dy, -4.0);
        // …and through the camera: right-drag decreases yaw; drag-down
        // keeps its old sign (decreases pitch).
        let mut c = cam();
        let yaw0 = c.yaw_deg;
        let pitch0 = c.pitch_deg;
        let (odx, ody) = orbit_delta(egui::vec2(10.0, 0.0));
        c.orbit(odx, ody);
        assert!(c.yaw_deg < yaw0);
        let (odx, ody) = orbit_delta(egui::vec2(0.0, 10.0));
        c.orbit(odx, ody);
        assert!(c.pitch_deg < pitch0);
    }

    #[test]
    fn drag_plane_choice_prefers_axis_then_shift() {
        let fwd = Vec3::new(0.5, -0.7, -0.5).normalize();
        assert_eq!(
            drag_plane(Plane::XZ, false, None, fwd),
            DragPlane::Work(Plane::XZ)
        );
        // Shift alone → vertical plane facing the camera …
        match drag_plane(Plane::XZ, true, None, fwd) {
            DragPlane::CamVertical(n) => {
                assert!((n.y).abs() < 1e-6, "vertical-plane normal is horizontal");
                let want = Vec3::new(fwd.x, 0.0, fwd.z).normalize();
                assert!((n - want).length() < 1e-5);
            }
            other => panic!("shift should give CamVertical, got {other:?}"),
        }
        // … axis lock wins over Shift …
        assert_eq!(
            drag_plane(Plane::XZ, true, Some(2), fwd),
            DragPlane::Axis(2)
        );
        // … and a straight-down camera falls back to the work plane.
        assert_eq!(
            drag_plane(Plane::XY, true, None, Vec3::NEG_Y),
            DragPlane::Work(Plane::XY)
        );
    }

    #[test]
    fn axis_hit_rides_the_locked_axis() {
        // Ray from origin along +x; locked-Y line through (5,0,0) — the
        // closest line point is the line's foot (5,0,0).
        let h = axis_hit(Vec3::new(5.0, 0.0, 0.0), Vec3::Y, Vec3::ZERO, Vec3::X).unwrap();
        assert!((h - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
        // Ray parallel to the axis → no hit.
        assert!(axis_hit(Vec3::ZERO, Vec3::Y, Vec3::ZERO, Vec3::Y).is_none());
    }

    /// Switching constraint mid-drag must not teleport the object: the
    /// re-anchor makes `hit - plane_pt` return the same delta. An axis
    /// lock additionally drops the off-axis component.
    #[test]
    fn reanchor_keeps_delta_continuous() {
        let anchor = Vec3::new(10.0, 5.0, 0.0);
        let ro = Vec3::new(10.0, 80.0, 120.0);
        let rd = (anchor - ro).normalize();
        // Mid-drag on the XZ plane, delta already nonzero.
        let mut md = (
            DragPlane::Work(Plane::XZ),
            plane_hit(Plane::XZ, anchor, ro, rd).unwrap(), // plane_pt
        );
        // pretend the cursor moved: new ray direction
        let rd2 = (anchor + Vec3::new(8.0, 0.0, 4.0) - ro).normalize();
        let h2 = drag_hit(md.0, anchor, ro, rd2).unwrap();
        let delta = h2 - md.1;
        assert!(delta.length() > 0.1);
        // Switch to the camera-facing vertical plane at this cursor pos:
        let dp = drag_plane(
            Plane::XZ,
            true,
            None,
            Vec3::new(0.0, -0.4, -1.0).normalize(),
        );
        let h_new = drag_hit(dp, anchor, ro, rd2).unwrap();
        // the exact re-anchor the drag loop performs:
        let d2 = kept_delta(dp, delta);
        md.1 = h_new - d2;
        md.0 = dp;
        // Same cursor → same delta afterwards (no jump on the frame of
        // the switch).
        let after = drag_hit(md.0, anchor, ro, rd2).unwrap() - md.1;
        assert!(
            (after - d2).length() < 1e-4,
            "plane switch jumped: {after:?} vs {d2:?}"
        );
        // Switch to a Y axis lock: only y survives.
        let dp = DragPlane::Axis(1);
        let h_new = drag_hit(dp, anchor, ro, rd2).unwrap();
        let d3 = kept_delta(dp, after);
        assert!(
            d3.x.abs() < 1e-5 && d3.z.abs() < 1e-5,
            "axis lock keeps y only: {d3:?}"
        );
        md.1 = h_new - d3;
        md.0 = dp;
        let after = drag_hit(md.0, anchor, ro, rd2).unwrap() - md.1;
        assert!((after - d3).length() < 1e-4);
        // An axis hit lands ON the line through the anchor.
        let h = axis_hit(anchor, Vec3::Y, ro, rd2).unwrap();
        assert!((h - anchor).dot(Vec3::X).abs() < 1e-4);
        assert!((h - anchor).dot(Vec3::Z).abs() < 1e-4);
    }

    #[test]
    fn drag_gesture_mapping() {
        assert_eq!(
            primary_drag_action(Mode::Navigate, Some(7)),
            DragAction::Move { grab: Some(7) }
        );
        assert_eq!(primary_drag_action(Mode::Navigate, None), DragAction::Box);
        assert_eq!(
            primary_drag_action(Mode::Move, Some(3)),
            DragAction::Move { grab: Some(3) }
        );
        assert_eq!(
            primary_drag_action(Mode::Move, None),
            DragAction::Move { grab: None }
        );
        assert_eq!(primary_drag_action(Mode::Connect, None), DragAction::Box);
    }

    #[test]
    fn bracket_rect_follows_projected_box() {
        let c = cam();
        let vp = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 500.0));
        let aspect = 800.0 / 500.0;
        let near = Vec3::new(0.0, 10.0, 0.0);
        let r = screen_box_rect(&c, aspect, vp, near, 4.0).unwrap();
        // the bracket contains the projected centre and has real extent
        let nd = project(&c, aspect, near).unwrap();
        let s = egui::pos2(nd.x * 800.0, nd.y * 500.0);
        assert!(r.contains(s));
        assert!(r.width() > 1.0 && r.height() > 1.0);
        // a bigger object gets a bigger bracket — screen-sized, not 7px
        let r_big = screen_box_rect(&c, aspect, vp, near, 12.0).unwrap();
        assert!(r_big.width() > r.width() * 1.5);
        // same box further away shrinks the bracket
        let far = near - (c.eye() - near).normalize() * 150.0;
        let r_far = screen_box_rect(&c, aspect, vp, far, 4.0).unwrap();
        assert!(r_far.width() < r.width());
    }

    #[test]
    fn hover_not_selected_still_maps_to_move() {
        // Navigate drag on an UNselected object → Move (the caller selects
        // it first); drag on empty space → box select.
        let sel: HashSet<u32> = [1, 2].into_iter().collect();
        let _ = sel;
        assert!(matches!(
            primary_drag_action(Mode::Navigate, Some(9)),
            DragAction::Move { .. }
        ));
    }
}

#[cfg(test)]
mod wrap_probe {
    use eframe::egui;

    #[test]
    fn horizontal_wrapped_wraps_in_panel() {
        let ctx = egui::Context::default();
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
                            for i in 0..10 {
                                ui.button(format!("button-{i}-long-label"));
                            }
                        });
                        h = ui.min_rect().height();
                    });
                },
            );
        }
        assert!(h > 40.0, "wrap did not engage: panel height {h}");
    }
}

#[cfg(test)]
mod wrap_probe2 {
    use eframe::egui;

    // Repro of the toolbar wrap complaint: a narrow TopBottomPanel must
    // wrap the controls row onto a second line instead of clipping.
    #[test]
    fn wrapped_row_grows_past_one_line() {
        let ctx = egui::Context::default();
        let mut row_h = 0.0f32;
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
                            for i in 0..10 {
                                ui.button(format!("btn-{i}-long-text"));
                            }
                        });
                        row_h = ui.min_rect().height();
                    });
                },
            );
        }
        assert!(row_h > 40.0, "wrapped row only {row_h}px tall — no wrap");
    }

    // Same shape as the real toolbar: a plain `horizontal` sibling first,
    // then the `horizontal_wrapped` mode row — does the wrap still engage?
    #[test]
    fn wrapped_row_after_plain_horizontal_wraps() {
        let ctx = egui::Context::default();
        let mut h = 0.0f32;
        let mut wrapped_h = 0.0f32;
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
                        ui.horizontal(|ui| {
                            ui.button("File");
                            ui.button("Run");
                        });
                        let inner = ui.horizontal_wrapped(|ui| {
                            for i in 0..10 {
                                ui.button(format!("btn-{i}-long-text"));
                            }
                        });
                        wrapped_h = inner.response.rect.height();
                        h = ui.min_rect().height();
                    });
                },
            );
        }
        assert!(
            wrapped_h > 40.0,
            "wrapped row after sibling did not wrap: {wrapped_h}px (panel {h}px)"
        );
    }

    // Real toolbar mix: selectable buttons + separator + label + ComboBox +
    // tinted image + status label — find which widget poisons the wrap.
    #[test]
    fn wrapped_row_with_toolbar_mix_wraps() {
        let ctx = egui::Context::default();
        let mut wrapped_h = 0.0f32;
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
                        let inner = ui.horizontal_wrapped(|ui| {
                            let mut mode = 0usize;
                            for name in ["Navigate", "Insert", "Connect", "Move"] {
                                ui.selectable_value(&mut mode, usize::MAX, name);
                            }
                            ui.separator();
                            ui.label("plane");
                            ui.selectable_value(&mut 0, 0, "x-z");
                            ui.selectable_value(&mut 0, 1, "x-y");
                            ui.selectable_value(&mut 0, 2, "y-z");
                            ui.separator();
                            ui.selectable_value(&mut true, true, "synapses");
                            ui.selectable_value(&mut true, true, "grid");
                            egui::ComboBox::from_id_salt("gv")
                                .selected_text("auto")
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut 0, 0, "auto");
                                });
                            ui.separator();
                            ui.label("status text goes here");
                        });
                        wrapped_h = inner.response.rect.height();
                    });
                },
            );
        }
        assert!(
            wrapped_h > 40.0,
            "toolbar-mix row did not wrap: {wrapped_h}px"
        );
    }
}
