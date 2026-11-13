//! Per-frame shared-state sync: `sync_shared` folds the worker's
//! `Shared` mailbox into app fields; `plane_frame` feeds grid-plane
//! textures to the GPU.
//!
//! Moved verbatim out of `app.rs` (Stage 19).

use crate::gpu::Gpu;
use crate::snap::GeoSnap;

use super::EditorApp;
use super::PlotTab;

impl EditorApp {
    /// Pull fresh shared state (geometry + snapshot) into the UI.
    pub(super) fn sync_shared(&mut self, gpu: &mut Gpu) {
        // Any ServerMsg that landed since the last frame bumps dirty_seq —
        // that's our "repaint on transport" signal for Stage 14 E.
        //
        // One lock: copy out everything the frame needs, drop it, then
        // act — upload_ready/files_refresh send client messages and
        // save_download opens a file dialog, none of which may hold the
        // worker's lock.
        struct Taken {
            dirty_seq: u64,
            file_data: Option<(String, String)>,
            file_written: Option<String>,
            last_err: Option<String>,
            files: Option<(String, Vec<proto::FileEntry>)>,
            usage: Option<(u64, u64)>,
            upload_ready: Option<String>,
            file_op: Option<String>,
            download: Option<(String, Vec<u8>)>,
            saved: Vec<(String, u64)>,
            loaded: Vec<(String, u64)>,
            jobs: Vec<proto::JobInfo>,
            job_progress: std::collections::BTreeMap<u64, Vec<Vec<(String, f64)>>>,
            job_detail: Option<crate::snap::JobDetailSnap>,
            geo_new: Option<std::sync::Arc<GeoSnap>>,
            snap: crate::snap::Snap,
            running: bool,
            sim_built: bool,
            busy: bool,
            net_gen: u64,
            cfg: proto::Config,
            backend: proto::Backend,
        }
        let Taken {
            dirty_seq,
            file_data,
            file_written,
            last_err,
            files,
            usage,
            upload_ready,
            file_op,
            download,
            saved,
            loaded,
            jobs,
            job_progress,
            job_detail,
            geo_new,
            snap,
            running,
            sim_built,
            busy,
            net_gen,
            cfg,
            backend,
        } = {
            let mut sh = self.ctl.shared.write().unwrap();
            Taken {
                dirty_seq: sh.dirty_seq,
                file_data: sh.file_data.take(),
                file_written: sh.file_written.take(),
                last_err: sh.last_error.take(),
                files: sh.files.take(),
                usage: sh.usage.take(),
                upload_ready: sh.upload_ready.take(),
                file_op: sh.file_op.take(),
                download: match sh.download.take() {
                    // still streaming — put it back
                    done @ Some((_, _, false)) => {
                        sh.download = done;
                        None
                    }
                    Some((path, bytes, true)) => Some((path, bytes)),
                    None => None,
                },
                saved: sh.saved_ack.take().into_iter().collect(),
                loaded: sh.loaded_ack.take().into_iter().collect(),
                jobs: sh.jobs.clone(),
                job_progress: sh.job_progress.clone(),
                job_detail: sh.job_detail.take(),
                geo_new: sh.geo.clone(),
                snap: sh.snap.clone(),
                running: sh.running,
                sim_built: sh.sim_built,
                busy: sh.busy,
                net_gen: sh.net_gen,
                cfg: sh.cfg.clone(),
                backend: sh.backend,
            }
        };
        if dirty_seq != self.last_dirty_seq {
            self.net_touched = true;
            self.last_dirty_seq = dirty_seq;
        }
        if let Some((path, text)) = file_data {
            self.script.file = path;
            self.script.text = text;
            self.status = "loaded".into();
        }
        if let Some(p) = file_written {
            // Q8: free the upload slot — the queued upload finished
            // (chunks end with FileWritten). Without this the second
            // upload waited forever behind the completed first.
            if self.files.upload.as_ref().map(|u| u.0.as_str()) == Some(p.as_str()) {
                self.files.upload = None;
            }
            self.status = format!("wrote {p}");
        }
        if let Some((dir, entries)) = files {
            if self.files.local_root.is_none() && dir == self.files.cwd {
                self.files.entries = entries.clone();
            }
            self.files.file_list = Some((dir, entries));
        }
        if let Some((used, quota)) = usage {
            self.files.usage = Some((used, quota));
        }
        if let Some(p) = upload_ready {
            self.upload_ready(&p);
        }
        if let Some(op) = file_op {
            // a mkdir/delete/rename landed — relist
            self.files_refresh();
            self.status = op;
        }
        if let Some((path, bytes)) = download {
            self.save_download(&path, &bytes);
        }
        // Q6: the document goes clean only on the server's Saved —
        // the ack carries the version it actually wrote.
        for (path, version) in saved {
            self.saved_version = version;
            self.net_path = Some(path.clone());
            self.push_recent(path.clone());
            self.status = format!("saved {path}");
        }
        // Q7: net_path/recent move on confirmed load only.
        for (path, _) in loaded {
            self.net_path = Some(path.clone());
            self.push_recent(path.clone());
            self.selection.clear();
            self.status = format!("loaded {path}");
        }
        if let Some(e) = last_err {
            // U1: the latest error rides the status line until the user
            // acts again; Q8: an "upload …" error frees the slot too.
            if e.starts_with("upload") {
                self.files.upload = None;
                self.files.status = e.clone();
            }
            self.status = e.clone();
            self.console.last_error = Some(e);
        }
        self.jobs.list = jobs;
        self.jobs.progress = job_progress;
        if let Some(d) = job_detail {
            // a re-opened job's history becomes plottable too
            self.jobs.progress.insert(d.id, d.progress.clone());
            self.jobs.detail = Some(d);
        }
        // Toolbar widgets read the worker's live cfg so script-side
        // `sim.set_dt`/`sim.set_noise`/`sim.set_backend` calls show up.
        self.sim.dt_ms = cfg.dt_ms;
        self.sim.noise_hz = cfg.noise_rate_hz;
        self.sim.noise_kick = cfg.noise_kick_mv;
        self.sim.backend = backend;
        let gen_bumped = net_gen != self.last_net_gen;
        if gen_bumped {
            self.last_net_gen = net_gen;
            self.reset_net_state();
        }
        if let Some(g) = geo_new {
            if g.version != self.geo_version {
                gpu.set_scene(&g);
                self.geo_version = g.version;
                // Auto-frame on empty → non-empty (first net, script-built
                // grid on a blank canvas, rebuild after New) and whenever a
                // dialog asked for it (frame_pending).
                // Floor grid + camera clamp follow the net's own baseline.
                self.camera.floor_y = g.floor_y();
                self.inst_half = gpu
                    .scene
                    .as_ref()
                    .map(|s| s.inst_half.clone())
                    .unwrap_or_default();
                let was_empty = self.geo.as_ref().is_none_or(|old| old.pos.is_empty());
                if !g.pos.is_empty() && (was_empty || self.frame_pending) {
                    let (mn, mx) = gpu.scene.as_ref().unwrap().bounds;
                    self.camera.frame(mn, mx);
                }
                self.frame_pending = false;
                self.geo = Some(g);
            }
        }
        // A loaded example asked to auto-watch its neurons — apply it
        // once the new Geo covers the ids. The gen bump from a script's
        // net.clear may land a frame AFTER the covering Geo and clear the
        // selection again — keep re-applying until update_watch has sent
        // the watch (watch_sent matches), then stop.
        if let Some(ids) = self.script.example_watch {
            if ids.is_empty() || self.wall_ms > self.script.example_deadline_ms {
                self.script.example_watch = None;
            } else {
                // Only a Geo newer than the Run click counts — the pre-run
                // net can already cover the ids, and watching it just gets
                // wiped by the eval's net.clear → reset_net_state.
                let covered =
                    example_geo_covers(self.script.example_base_ver, self.geo.as_deref(), ids);
                if covered {
                    // Rebuild the HashSet only when it differs — a fresh set
                    // gets a new random iteration order and update_watch would
                    // resend the same Watch list every frame.
                    if self.selection.len() != ids.len()
                        || ids.iter().any(|n| !self.selection.contains(n))
                    {
                        self.selection = ids.iter().copied().collect();
                    }
                    self.win.plots = true;
                    self.plots.tab = PlotTab::Voltage;
                }
                // Retire only after the watch has stayed applied across a
                // quiet stretch — a net-gen bump inside the eval (net.clear)
                // may land one frame before its Geo and clear the selection
                // again; this re-applies on the next frame.
                if covered
                    && self.watch_sent.len() == ids.len()
                    && ids.iter().all(|n| self.watch_sent.contains(n))
                    && !gen_bumped
                {
                    self.script.example_quiet += 1;
                    if self.script.example_quiet > 30 {
                        self.script.example_watch = None;
                        self.script.example_quiet = 0;
                    }
                } else {
                    self.script.example_quiet = 0;
                }
            }
        }
        if snap.seq != self.last_snap_seq {
            self.last_snap_seq = snap.seq;
            self.plots.last_sim_t = snap.time_ms;
            self.sim.speed_peak = self.sim.speed_peak.max(snap.steps_per_sec);
            self.activity = snap.activity.to_vec();
            self.grid_traces = snap.grid_traces.clone();
            self.input_values = snap.input_values.clone();
            if let Some(g) = &self.geo {
                self.particles
                    .feed(g, &snap.counts, snap.time_ms, self.wall_ms);
            }
            // raster bin: ids with count>0 — folded into the histories
            // by PlotHist::feed, which also spots the sim clock going
            // backwards (Reset, sim.reset(), another client) and clears.
            let fired: Vec<u32> = snap
                .counts
                .iter()
                .enumerate()
                .filter(|(_, c)| **c > 0)
                .map(|(i, _)| i as u32)
                .collect();
            // watched_v/_u are parallel vectors on the wire — pair by id,
            // u missing ⇒ empty series (older servers).
            let watched: Vec<(u32, &[f32], &[f32])> = snap
                .watched_v
                .iter()
                .map(|(id, vs)| {
                    let us = snap
                        .watched_u
                        .iter()
                        .find(|(i, _)| i == id)
                        .map(|(_, us)| us.as_slice())
                        .unwrap_or(&[]);
                    (*id, vs.as_slice(), us)
                })
                .collect();
            self.plots
                .hist
                .feed(snap.time_ms, fired, &watched, self.sim.dt_ms as f64);
            // A reset while the user was panned away snaps the plots
            // back to live view — a rewind always wants "now".
            if self.plots.hist.resets != self.plots.hist_resets_seen {
                self.plots.hist_resets_seen = self.plots.hist.resets;
                self.plots.win.follow = true;
            }
        }
        self.sim.running = running;
        self.sim.sim_built = sim_built;
        self.sim.busy = busy;
    }

    /// Per-frame grid-plane bookkeeping for the renderer: quad params and
    /// hide bits (cheap, every call), and the reduced activity textures
    /// only when the trace data or display config actually changed (a
    /// 1M-texel plane is a ~4MB padded upload — not per-frame).
    ///
    /// Returns the per-plane draw flags for `DrawParams::planes_on`.
    pub(crate) fn plane_frame(&mut self, gpu: &mut Gpu, vp_h_px: f32) -> Option<Vec<bool>> {
        let g = self.geo.clone()?;
        if g.grids.is_empty() {
            self.plane_key = 0;
            return None;
        }
        let eye = self.camera.eye();
        let mut on = Vec::with_capacity(g.grids.len());
        let mut params = Vec::with_capacity(g.grids.len());
        let mut cfgs = Vec::with_capacity(g.grids.len());
        let mut bits = vec![0u32; g.pos.len() / 32 + 1];
        for gs in &g.grids {
            let cfg = self
                .plane_cfgs
                .get(&(gs.group as u32))
                .copied()
                .unwrap_or_default();
            cfgs.push(cfg);
            let dist = (glam::Vec3::from(crate::plane::center(gs)) - eye).length();
            let show = match self.grid_view {
                crate::plane::GridView::Planes => true,
                crate::plane::GridView::Cubes => false,
                crate::plane::GridView::Auto => {
                    // A small selection touching this grid flips it back
                    // to cubes so its neurons stay editable; big selects
                    // keep the plane. Members are neuron ids == instance
                    // ids for neurons.
                    let sel_touches = self.selection.len() <= crate::plane::SEL_CUBE_MAX
                        && !self.selection.is_empty()
                        && gs.members.iter().any(|m| self.selection.contains(m));
                    !sel_touches && crate::plane::texel_px(gs, dist, vp_h_px) < crate::plane::LOD_PX
                }
            };
            on.push(show);
            let (ua, va) = crate::plane::plane_dims(gs, &cfg);
            let (origin, du, dv) = crate::plane::quad(gs, &cfg);
            params.push(crate::gpu::PlaneParamsGpu {
                origin,
                du,
                dv,
                wh: [gs.dims[ua], gs.dims[va], 0, 0],
            });
            if show {
                mark_bits(&mut bits, &gs.members);
            }
        }
        gpu.write_plane_params(&params);
        gpu.write_hide_bits(&bits);

        let need_upload = self.plane_snap != self.last_snap_seq
            || self.plane_geo != self.geo_version
            || self.plane_cfgs_last != cfgs;
        if need_upload {
            for (i, gs) in g.grids.iter().enumerate() {
                let tr = self.grid_traces.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
                let (w, h, pix, pick) = crate::plane::reduce(gs, tr, &cfgs[i]);
                gpu.ensure_plane_tex(i, w, h);
                gpu.write_plane(i, &pix, &pick);
            }
            self.plane_snap = self.last_snap_seq;
            self.plane_geo = self.geo_version;
            self.plane_cfgs_last = cfgs.clone();
        }
        // SceneKey contribution: everything this call fed the GPU.
        let mut h = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        h.write(bytemuck::cast_slice(&params));
        for c in &cfgs {
            format!("{c:?}").hash(&mut h);
        }
        h.write(bytemuck::cast_slice(&bits));
        on.hash(&mut h);
        need_upload.hash(&mut h);
        self.plane_key = h.finish();
        Some(on)
    }
}

/// Hide-bits for plane members. Member ids come over the wire — an
/// out-of-range id must not panic the frame.
pub(super) fn mark_bits(bits: &mut [u32], members: &[u32]) {
    for &m in members {
        if let Some(w) = bits.get_mut(m as usize >> 5) {
            *w |= 1 << (m & 31);
        }
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
    fn mark_bits_ignores_out_of_range_member_ids() {
        let mut bits = vec![0u32; 2]; // covers ids 0..63
        mark_bits(&mut bits, &[0, 63]);
        mark_bits(&mut bits, &[64, u32::MAX]); // must not panic
        assert_eq!(bits[0], 1);
        assert_eq!(bits[1], 1 << 31);
    }
}
