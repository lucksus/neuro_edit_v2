//! Central 3D viewport: the scene renders offscreen in `update`, and an
//! egui-wgpu paint callback blits it into the viewport rect. Input: camera
//! controls, GPU picking, box select, work-plane move/insert/connect.

use eframe::{egui, egui_wgpu};
use egui::epaint::PaintCallbackInfo;
use glam::{Mat4, Vec2, Vec3};

use crate::app::{EditorApp, Mode, MoveDrag};
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
    let d = rd.dot(nrm);
    if d.abs() < 1e-9 {
        return None;
    }
    let t = (off - ro.dot(nrm)) / d;
    (t > 0.0).then(|| ro + rd * t)
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
                    on_pick(app, res, ctx, mouse_vp, aspect);
                }
            }

            // ---- pick request on click (and a cheap hover pick each frame
            // so the status bar + context menu know what's under the cursor)
            if resp.clicked()
                || (resp.drag_started_by(egui::PointerButton::Primary) && app.mode == Mode::Move)
            {
                if let Some(mv) = mouse_vp {
                    let px = (mv.x * w_px as f32).min(w_px as f32 - 1.0) as u32;
                    let py = (mv.y * h_px as f32).min(h_px as f32 - 1.0) as u32;
                    with_gpu(rs, |gpu| gpu.request_pick(px, py, w_px, h_px, false));
                }
            } else if resp.hovered() && !resp.dragged() {
                if let Some(mv) = mouse_vp {
                    let px = (mv.x * w_px as f32).min(w_px as f32 - 1.0) as u32;
                    let py = (mv.y * h_px as f32).min(h_px as f32 - 1.0) as u32;
                    with_gpu(rs, |gpu| gpu.request_pick(px, py, w_px, h_px, true));
                }
            }

            context_menu(app, &resp);

            // ---- mode interactions ----
            mode_input(app, &resp, rect, mouse_vp, aspect);

            // ---- uploads + offscreen scene render ----
            let now_ms = app.ctl.shared.read().unwrap().snap.time_ms;
            let sel: Vec<u32> = app.selection.iter().copied().collect();
            let mut pvs = Vec::new();
            if app.show_grid {
                scene::floor_grid(app.camera.target, &mut pvs);
            }
            if app.mode == Mode::Move || app.mode == Mode::Insert {
                scene::work_plane(app.camera.target, app.plane, &mut pvs);
            }
            if let Some(g) = &app.geo {
                scene::group_boxes(g, &mut pvs);
                if app.mode == Mode::Connect {
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
            let parts = app.particles.instances(now_ms);
            let pv_n = pvs.len() as u32;
            let show_syn = app.show_synapses;
            with_gpu(rs, |gpu| {
                gpu.write_cam(
                    &app.camera,
                    aspect,
                    h_px as f32,
                    ctx.input(|i| i.time) as f32,
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
                    planes_on,
                });
                gpu.render_offscreen(w_px, h_px);
            });

            ui.painter()
                .add(egui_wgpu::Callback::new_paint_callback(rect, BlitScene));

            overlay_2d(app, ui, rect, aspect);

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

fn camera_input(app: &mut EditorApp, ctx: &egui::Context, resp: &egui::Response) {
    let dt = ctx.input(|i| i.stable_dt.min(0.05));
    app.camera.tick(dt);
    if resp.dragged_by(egui::PointerButton::Secondary) {
        let d = resp.drag_delta();
        app.camera.orbit(d.x, -d.y);
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
            app.mode = Mode::Navigate;
            app.connect_from = None;
            app.move_drag = None;
        }
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
        if i.key_pressed(egui::Key::V) && !i.modifiers.ctrl {
            app.mode = Mode::Navigate;
        }
        if i.key_pressed(egui::Key::F) {
            app.frame_all();
        }
        if i.modifiers.ctrl && i.key_pressed(egui::Key::C) {
            app.copy_selection(false);
        }
        if i.modifiers.ctrl && i.key_pressed(egui::Key::X) {
            app.copy_selection(true);
        }
        if i.modifiers.ctrl && i.key_pressed(egui::Key::V) {
            app.paste(None);
        }
    });
}

/// Right-click context menu on the entity under the cursor (or empty space).
fn context_menu(app: &mut EditorApp, resp: &egui::Response) {
    resp.context_menu(|ui| {
        let ent = app
            .hover
            .and_then(|i| app.geo.as_ref().and_then(|g| entity_of(g, i)));
        match ent {
            Some(Entity::Neuron(n)) => {
                ui.label(format!("neuron {n}"));
                ui.separator();
                if ui.button("Watch v(t)").clicked() {
                    app.watch(vec![n]);
                    ui.close();
                }
                if ui.button("Connect from here").clicked() {
                    app.mode = Mode::Connect;
                    app.connect_from = app.hover;
                    ui.close();
                }
                if ui.button("Delete").clicked() {
                    app.delete_entity(n);
                    ui.close();
                }
            }
            Some(Entity::Input(k)) => {
                if let Some(g) = &app.geo {
                    ui.label(format!("input {}", g.inputs[k as usize].name));
                }
                ui.separator();
                if ui.button("Set value…").clicked() {
                    app.open_input_dialog(k);
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
                if ui.button("Insert neuron here").clicked() {
                    app.mode = Mode::Insert;
                    ui.close();
                }
                if ui.button("Paste").clicked() {
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

fn on_pick(
    app: &mut EditorApp,
    res: Option<u32>,
    ctx: &egui::Context,
    mouse_vp: Option<Vec2>,
    aspect: f32,
) {
    match app.mode {
        Mode::Connect => {
            // Only a hit on a neuron changes state: a missed click keeps the
            // pending source (dense scenes make small cubes easy to miss),
            // so Esc is the only way to abort a pending connection.
            if let Some(inst) = res {
                match app.geo.as_ref().and_then(|g| entity_of(g, inst)) {
                    Some(Entity::Neuron(n)) => {
                        let a = app
                            .connect_from
                            .take()
                            .and_then(|fi| app.geo.as_ref().and_then(|g| entity_of(g, fi)))
                            .and_then(|e| match e {
                                Entity::Neuron(a) => Some(a),
                                _ => None,
                            });
                        if let Some(a) = a {
                            app.connect_neurons(a, n);
                        } else {
                            app.connect_from = Some(inst);
                            app.status = "connect: pick target".into();
                        }
                    }
                    _ => {
                        app.status = "connect: that's not a neuron".into();
                    }
                }
            } else if app.connect_from.is_some() {
                app.status = "connect: missed — click a target neuron".into();
            }
        }
        Mode::Move => {
            // click starts a move drag if it landed on a selected entity
            if let (Some(inst), Some(mv)) = (res, mouse_vp) {
                if app.selection.contains(&inst) {
                    if let Some(g) = &app.geo {
                        let (ro, rd) = ray(&app.camera, aspect, mv.x, mv.y);
                        if let Some(hit) = plane_hit(app.plane, app.camera.target, ro, rd) {
                            let ids: Vec<u32> = app.selection.iter().copied().collect();
                            let start_pos: Vec<Vec3> =
                                ids.iter().filter_map(|&i| inst_pos(g, i)).collect();
                            app.move_drag = Some(MoveDrag {
                                plane_pt: hit,
                                start_pos,
                                ids,
                            });
                        }
                    }
                }
            }
        }
        _ => {
            // Navigate: select/clear; dbl-click focus handled separately
            let shift = ctx.input(|i| i.modifiers.shift);
            match res {
                Some(inst) => {
                    if !shift {
                        app.selection.clear();
                    }
                    app.selection.insert(inst);
                    app.last_pick = Some(inst);
                }
                None => {
                    if !shift {
                        app.selection.clear();
                    }
                    app.last_pick = None;
                }
            }
        }
    }
}

fn mode_input(
    app: &mut EditorApp,
    resp: &egui::Response,
    rect: egui::Rect,
    mouse_vp: Option<Vec2>,
    aspect: f32,
) {
    match app.mode {
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
        Mode::Move => {
            if resp.dragged_by(egui::PointerButton::Primary) {
                if let (Some(mv), Some(md)) = (mouse_vp, app.move_drag.clone()) {
                    let (ro, rd) = ray(&app.camera, aspect, mv.x, mv.y);
                    if let Some(hit) = plane_hit(app.plane, app.camera.target, ro, rd) {
                        let delta = hit - md.plane_pt;
                        app.preview_delta = Some(delta);
                        app.status =
                            format!("move Δ({:.0},{:.0},{:.0})", delta.x, delta.y, delta.z);
                    }
                }
            }
            if resp.drag_stopped_by(egui::PointerButton::Primary) {
                if let (Some(md), Some(d)) = (app.move_drag.take(), app.preview_delta.take()) {
                    app.apply_move(&md, d);
                    app.status.clear();
                }
            }
        }
        Mode::Navigate => {
            if resp.drag_started_by(egui::PointerButton::Primary) {
                if let Some(p) = resp.interact_pointer_pos() {
                    app.box_sel = Some((p, p));
                }
            }
            if resp.dragged_by(egui::PointerButton::Primary) {
                if let (Some((a, _)), Some(cur)) = (app.box_sel, resp.interact_pointer_pos()) {
                    app.box_sel = Some((a, cur));
                }
            }
            if resp.drag_stopped_by(egui::PointerButton::Primary) {
                if let Some((a, b)) = app.box_sel.take() {
                    let shift = resp.ctx.input(|i| i.modifiers.shift);
                    box_select(app, rect, a, b, aspect, shift);
                }
            }
            // double-click focus: CPU nearest on ray
            if resp.double_clicked() {
                if let (Some(g), Some(mv)) = (&app.geo, mouse_vp) {
                    let (ro, rd) = ray(&app.camera, aspect, mv.x, mv.y);
                    match nearest_on_ray(g, ro, rd, 25.0) {
                        Some(inst) => {
                            if let Some(p) = inst_pos(g, inst) {
                                app.camera.focus(p);
                            }
                        }
                        None => {
                            if let Some(hit) = plane_hit(app.plane, app.camera.target, ro, rd) {
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

/// Nearest entity to a ray within `max_t` world units (double-click focus).
fn nearest_on_ray(g: &GeoSnap, ro: Vec3, rd: Vec3, max_t: f32) -> Option<u32> {
    let mut best = None;
    let mut best_d = max_t;
    for (i, p) in g.pos.iter().enumerate() {
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
    if let Some(g) = &app.geo {
        let mut drawn = 0;
        for &id in &app.selection {
            if drawn > 64 {
                break;
            }
            if let Some(p) = inst_pos(g, id) {
                if let Some(nd) = project(&app.camera, aspect, p) {
                    let s = egui::pos2(
                        rect.min.x + nd.x * rect.width(),
                        rect.min.y + nd.y * rect.height(),
                    );
                    let r = 7.0f32;
                    let col = egui::Color32::from_rgb(255, 120, 40);
                    for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                        painter.line_segment(
                            [
                                egui::pos2(s.x + dx * r, s.y + dy * r),
                                egui::pos2(s.x + dx * r * 0.4, s.y + dy * r),
                            ],
                            egui::Stroke::new(1.5_f32, col),
                        );
                        painter.line_segment(
                            [
                                egui::pos2(s.x + dx * r, s.y + dy * r),
                                egui::pos2(s.x + dx * r, s.y + dy * r * 0.4),
                            ],
                            egui::Stroke::new(1.5_f32, col),
                        );
                    }
                    drawn += 1;
                }
            }
        }
    }
    let hint = match app.mode {
        Mode::Navigate => "navigate: click select · drag = box select · right-drag orbit · wheel zoom · WASD/QE pan · dbl-click focus · I insert · C connect · M move · Del delete",
        Mode::Insert => "insert: click to place a neuron on the work plane · Esc to exit",
        Mode::Connect => "connect: click source then target neuron · Esc to abort",
        Mode::Move => "move: drag a selected object on the work plane · Esc to exit",
    };
    painter.text(
        egui::pos2(rect.min.x + 8.0, rect.max.y - 8.0),
        egui::Align2::LEFT_BOTTOM,
        hint,
        egui::FontId::monospace(11.0),
        egui::Color32::from_gray(180),
    );
}
