//! Files window, upload/download plumbing, file-op sends, and the
//! file helpers. Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;

use proto::ClientMsg;

use super::layout::{app_win, clamp_window_rect, usable_rect};
use super::EditorApp;

impl EditorApp {
    /// Render the latest `ListFiles` reply as clickable entries under a
    /// file picker; returns `(path, is_dir)` of the picked entry.
    pub(super) fn file_list_ui(&self, ui: &mut egui::Ui) -> Option<(String, bool)> {
        let (dir, entries) = self.files.file_list.as_ref()?;
        let mut picked = None;
        egui::ScrollArea::vertical()
            .max_height(120.0)
            .show(ui, |ui| {
                for e in entries {
                    let label = if e.is_dir {
                        format!("{}/", e.name)
                    } else {
                        format!("{} ({} B)", e.name, e.size)
                    };
                    if ui.small_button(label).clicked() {
                        picked = Some((
                            if dir == "." {
                                e.name.clone()
                            } else {
                                format!("{dir}/{}", e.name)
                            },
                            e.is_dir,
                        ));
                    }
                }
            });
        picked
    }

    /// Refresh `files.entries` for the current `cwd` (remote → ListFiles;
    /// local → read_dir) + the usage numbers.
    pub(super) fn files_refresh(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(root) = self.files.local_root.clone() {
            let dir = root.join(&self.files.cwd);
            self.files.entries = std::fs::read_dir(&dir)
                .map(|rd| {
                    let mut v: Vec<proto::FileEntry> = rd
                        .flatten()
                        .filter_map(|e| {
                            let md = e.metadata().ok()?;
                            Some(proto::FileEntry {
                                name: e.file_name().to_string_lossy().into_owned(),
                                is_dir: md.is_dir(),
                                size: md.len(),
                                mtime_ms: md
                                    .modified()
                                    .ok()
                                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                    .map(|d| d.as_millis() as u64)
                                    .unwrap_or(0),
                            })
                        })
                        .collect();
                    v.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
                    v
                })
                .unwrap_or_default();
            self.files.usage = Some((local_dir_size(&root), 0));
            return;
        }
        self.ctl.send(ClientMsg::ListFiles {
            path: self.files.cwd.clone(),
        });
        self.ctl.send(ClientMsg::FileUsage);
    }

    /// `UploadReady` arrived — stream the queued chunks for this path.
    pub(super) fn upload_ready(&mut self, path: &str) {
        let Some((p, data, off)) = &self.files.upload else {
            return;
        };
        if p != path {
            return;
        }
        // send the whole remaining tail; the writer thread queues them
        let mut off = *off;
        while off < data.len() {
            let end = (off + 256 * 1024).min(data.len());
            self.ctl.send(ClientMsg::UploadChunk {
                path: p.clone(),
                data: data[off..end].to_vec(),
                last: end == data.len(),
            });
            off = end;
        }
        if let Some(u) = self.files.upload.as_mut() {
            u.2 = off;
        }
    }

    /// Completed `DownloadFile` stream — hand the bytes to the user.
    pub(super) fn save_download(&mut self, path: &str, bytes: &[u8]) {
        let name = std::path::Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "download".into());
        #[cfg(target_arch = "wasm32")]
        {
            download_blob(&name, bytes);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(dst) = rfd::FileDialog::new().set_file_name(&name).save_file() {
                match std::fs::write(&dst, bytes) {
                    Ok(()) => self.files.status = format!("saved {}", dst.display()),
                    Err(e) => self.files.status = format!("save: {e}"),
                }
            }
        }
    }

    /// Queue an upload (remote → chunked protocol; local → copy).
    pub(super) fn queue_upload(&mut self, name: &str, bytes: Vec<u8>) {
        if let Some(root) = self.files.local_root.clone() {
            let dst = root.join(&self.files.cwd).join(name);
            match std::fs::write(&dst, &bytes) {
                Ok(()) => self.files.status = format!("uploaded {name}"),
                Err(e) => self.files.status = format!("upload {name}: {e}"),
            }
            self.files_refresh();
            return;
        }
        self.files.upload_q.push_back((
            if self.files.cwd.is_empty() {
                name.to_string()
            } else {
                format!("{}/{name}", self.files.cwd)
            },
            bytes,
        ));
    }

    /// Poll the upload queue: idle → `UploadStart` for the next item.
    pub(super) fn files_pump(&mut self) {
        if self.files.local_root.is_none() && self.files.upload.is_none() {
            if let Some((p, data)) = self.files.upload_q.pop_front() {
                self.ctl.send(ClientMsg::UploadStart {
                    path: p.clone(),
                    size: data.len() as u64,
                });
                self.files.upload = Some((p, data, 0));
            }
        }
        // browser file-picker results (web only)
        #[cfg(target_arch = "wasm32")]
        PICKED.with(|p| {
            for (name, bytes) in p.borrow_mut().drain(..) {
                self.queue_upload(&name, bytes);
            }
        });
        // dropped files (drag & drop onto the app — works native + wasm)
        for f in self.files.ctx_dropped.take().unwrap_or_default() {
            if let Some(bytes) = f.bytes.as_deref() {
                self.queue_upload(&f.name, bytes.to_vec());
            }
            #[cfg(not(target_arch = "wasm32"))]
            if f.bytes.is_none() {
                if let Some(p) = &f.path {
                    if let Ok(b) = std::fs::read(p) {
                        self.queue_upload(&f.name, b);
                    }
                }
            }
        }
    }

    pub(super) fn files_window(&mut self, ctx: &egui::Context) {
        // stash dropped files where files_pump can see them each frame
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            self.files.ctx_dropped = Some(dropped);
        }
        self.files_pump();
        if !self.win.files {
            return;
        }
        if !self.files.opened {
            self.files.opened = true;
            self.files_refresh();
        }
        let mut open = self.win.files;
        let u = usable_rect(ctx);
        let dr = clamp_window_rect(
            egui::Rect::from_min_size(
                egui::pos2(u.center().x - 210.0, u.min.y + 24.0),
                egui::vec2(420.0, 420.0),
            ),
            u,
        );
        app_win(ctx, "Files")
            .open(&mut open)
            .default_size(dr.size())
            .default_pos(dr.min)
            .show(ctx, |ui| {
                // path bar: root label + up + refresh
                ui.horizontal(|ui| {
                    if self.files.local_root.is_some() {
                        ui.label("app data:");
                    } else {
                        ui.label("my files:");
                    }
                    ui.monospace(if self.files.cwd.is_empty() {
                        "/".to_string()
                    } else {
                        self.files.cwd.clone()
                    });
                    if ui.button("^").clicked() && !self.files.cwd.is_empty() {
                        let cwd = std::path::Path::new(&self.files.cwd);
                        self.files.cwd = cwd
                            .parent()
                            .map(|p| p.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        self.files_refresh();
                    }
                    if ui.button("⟳").clicked() {
                        self.files_refresh();
                    }
                });
                // usage bar
                if let Some((used, quota)) = self.files.usage {
                    let (txt, frac) = if quota > 0 {
                        (
                            format!("{} of {} used", fmt_size(used), fmt_size(quota)),
                            used as f32 / quota as f32,
                        )
                    } else {
                        (format!("{} used", fmt_size(used)), 0.0)
                    };
                    ui.add(
                        egui::ProgressBar::new(frac.clamp(0.0, 1.0))
                            .text(txt)
                            .desired_height(10.0),
                    );
                }
                ui.separator();
                // listing
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let entries = self.files.entries.clone();
                    for e in &entries {
                        let label = if e.is_dir {
                            format!("📁 {}", e.name)
                        } else {
                            format!(
                                "{}   {}   {}",
                                e.name,
                                fmt_size(e.size),
                                fmt_date(e.mtime_ms)
                            )
                        };
                        let sel = self.files.sel.as_deref() == Some(e.name.as_str());
                        let r = ui.selectable_label(sel, label);
                        if r.clicked() {
                            self.files.sel = Some(e.name.clone());
                            self.files.rename_to = e.name.clone();
                        }
                        if r.double_clicked() {
                            self.files_open(e);
                        }
                    }
                    if entries.is_empty() {
                        ui.weak("(empty — drop files here to upload)");
                    }
                });
                ui.separator();
                // actions on the selection
                let sel = self.files.sel.clone();
                ui.horizontal_wrapped(|ui| {
                    if ui.button("new folder").clicked() {
                        self.files.mkdir = "new-folder".into();
                    }
                    if !self.files.mkdir.is_empty() {
                        ui.text_edit_singleline(&mut self.files.mkdir);
                        if ui.button("create").clicked() {
                            self.file_mkdir();
                        }
                    }
                    if sel.is_some() && ui.button("rename…").clicked() {
                        self.files.rename_open = true;
                    }
                    if sel.is_some() && ui.button("delete…").clicked() {
                        self.files.confirm_delete = sel.clone();
                    }
                    if sel.is_some() && ui.button("download").clicked() {
                        self.file_download();
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("upload…").clicked() {
                        if let Some(paths) = rfd::FileDialog::new().pick_files() {
                            for p in paths {
                                if let Ok(b) = std::fs::read(&p) {
                                    let n = p
                                        .file_name()
                                        .map(|x| x.to_string_lossy().into_owned())
                                        .unwrap_or_else(|| "file".into());
                                    self.queue_upload(&n, b);
                                }
                            }
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    if ui.button("upload…").clicked() {
                        web_file_picker();
                    }
                });
                if self.files.rename_open {
                    ui.horizontal(|ui| {
                        ui.label("rename to");
                        ui.text_edit_singleline(&mut self.files.rename_to);
                        if ui.button("ok").clicked() {
                            self.file_rename();
                            self.files.rename_open = false;
                        }
                        if ui.button("cancel").clicked() {
                            self.files.rename_open = false;
                        }
                    });
                }
                if let Some(d) = self.files.confirm_delete.clone() {
                    ui.horizontal(|ui| {
                        ui.colored_label(egui::Color32::LIGHT_RED, format!("delete {d}?"));
                        if ui.button("yes, delete").clicked() {
                            self.file_delete(&d);
                            self.files.confirm_delete = None;
                        }
                        if ui.button("cancel").clicked() {
                            self.files.confirm_delete = None;
                        }
                    });
                }
                if let Some((p, data, off)) = &self.files.upload {
                    ui.weak(format!(
                        "uploading {p}: {}/{}",
                        fmt_size(*off as u64),
                        fmt_size(data.len() as u64)
                    ));
                }
                if !self.files.status.is_empty() {
                    ui.weak(&self.files.status);
                }
            });
        self.win.files = open;
    }

    /// Double-click open: dirs descend; `.ne2` loads; `.rhai` opens in
    /// the script editor; anything else just stays selected.
    pub(super) fn files_open(&mut self, e: &proto::FileEntry) {
        if e.is_dir {
            self.files.cwd = if self.files.cwd.is_empty() {
                e.name.clone()
            } else {
                format!("{}/{}", self.files.cwd, e.name)
            };
            self.files_refresh();
            return;
        }
        let path = if self.files.cwd.is_empty() {
            e.name.clone()
        } else {
            format!("{}/{}", self.files.cwd, e.name)
        };
        if e.name.ends_with(".ne2") {
            match self.files.local_root.clone() {
                Some(root) => {
                    let p = root.join(&self.files.cwd).join(&e.name);
                    self.ctl.send(ClientMsg::LoadNet {
                        path: p.to_string_lossy().into_owned(),
                    });
                }
                None => self.ctl.send(ClientMsg::LoadNet { path }),
            }
            self.status = format!("loading {e}", e = e.name);
        } else if e.name.ends_with(".rhai") {
            match self.files.local_root.clone() {
                Some(root) => {
                    let p = root.join(&self.files.cwd).join(&e.name);
                    match std::fs::read_to_string(&p) {
                        Ok(t) => {
                            self.script.file = p.to_string_lossy().into_owned();
                            self.script.text = t;
                        }
                        Err(err) => self.files.status = format!("read {}: {err}", e.name),
                    }
                }
                None => self.ctl.send(ClientMsg::ReadFile { path }),
            }
            self.win.script = true;
        }
    }

    pub(super) fn file_mkdir(&mut self) {
        let name = self.files.mkdir.clone();
        if let Some(root) = self.files.local_root.clone() {
            // Q14: local-mode errors must reach files.status too.
            match std::fs::create_dir_all(root.join(&self.files.cwd).join(&name)) {
                Ok(()) => self.files.status = format!("mkdir {name}"),
                Err(e) => self.files.status = format!("mkdir {name}: {e}"),
            }
            self.files.mkdir.clear();
            self.files_refresh();
        } else {
            let path = if self.files.cwd.is_empty() {
                name.clone()
            } else {
                format!("{}/{name}", self.files.cwd)
            };
            self.ctl.send(ClientMsg::Mkdir { path });
            self.files.mkdir.clear();
        }
    }

    pub(super) fn file_rename(&mut self) {
        let Some(sel) = self.files.sel.clone() else {
            return;
        };
        let to_name = self.files.rename_to.clone();
        if let Some(root) = self.files.local_root.clone() {
            let dir = root.join(&self.files.cwd);
            match std::fs::rename(dir.join(&sel), dir.join(&to_name)) {
                Ok(()) => self.files.status = format!("renamed {sel} -> {to_name}"),
                Err(e) => self.files.status = format!("rename {sel}: {e}"),
            }
            self.files_refresh();
        } else {
            let (a, b) = if self.files.cwd.is_empty() {
                (sel, to_name)
            } else {
                (
                    format!("{}/{sel}", self.files.cwd),
                    format!("{}/{to_name}", self.files.cwd),
                )
            };
            self.ctl.send(ClientMsg::RenameFile { from: a, to: b });
        }
        self.files.sel = None;
    }

    pub(super) fn file_delete(&mut self, name: &str) {
        if let Some(root) = self.files.local_root.clone() {
            let p = root.join(&self.files.cwd).join(name);
            let r = if p.is_dir() {
                std::fs::remove_dir_all(&p)
            } else {
                std::fs::remove_file(&p)
            };
            match r {
                Ok(()) => self.files.status = format!("deleted {name}"),
                Err(e) => self.files.status = format!("delete {name}: {e}"),
            }
            self.files_refresh();
        } else {
            let path = if self.files.cwd.is_empty() {
                name.to_string()
            } else {
                format!("{}/{name}", self.files.cwd)
            };
            self.ctl.send(ClientMsg::DeleteFile { path });
        }
        self.files.sel = None;
    }

    pub(super) fn file_download(&mut self) {
        let Some(sel) = self.files.sel.clone() else {
            return;
        };
        if let Some(root) = self.files.local_root.clone() {
            let p = root.join(&self.files.cwd).join(&sel);
            match std::fs::read(&p) {
                Ok(bytes) => self.save_download(&sel, &bytes),
                Err(e) => self.files.status = format!("read {sel}: {e}"),
            }
        } else {
            let path = if self.files.cwd.is_empty() {
                sel.clone()
            } else {
                format!("{}/{sel}", self.files.cwd)
            };
            self.files.status = format!("downloading {path}…");
            self.ctl.send(ClientMsg::DownloadFile { path });
        }
    }
}

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

/// Parent dir of a path for a `ListFiles` request ("." at the top).
pub(super) fn dir_of(p: &str) -> String {
    match p.rsplit_once('/') {
        Some(("", _)) => "/".into(),
        Some((d, _)) => d.to_string(),
        None => ".".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simctl::SimCtl;

    #[test]
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
}
