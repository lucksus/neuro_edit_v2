//! Native login flow: saved-token reuse, the credentials dialog, and
//! the background connect attempt (all `#[cfg(not(wasm))]` — the module
//! is gated the same way in `mod.rs`).
//!
//! Moved verbatim out of `app.rs` (Stage 19).

use eframe::egui;

use crate::simctl::SimCtl;

use super::layout::app_win;
use super::{EditorApp, LoginDlg};

impl EditorApp {
    /// `--connect http(s)://…` entry point: reuse the persisted token
    /// when it matches this server (falling back to the dialog when it
    /// expired or never existed).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_saved_login(&mut self, url: &str) {
        if let Some((u, tok)) = self.login.saved_token.clone() {
            if u == url {
                match SimCtl::connect_secure(url, &tok) {
                    Ok(ctl) => {
                        self.ctl = ctl;
                        self.login.connect_url = url.to_string();
                        self.login.connect_tok = tok;
                        self.login.reconnect_delay = 2.0;
                        self.status = "connected".into();
                        return;
                    }
                    Err(_) => self.login.saved_token = None, // expired — re-login
                }
            }
        }
        self.begin_login(url, "");
    }

    /// Ask for credentials before the real transport exists; called by
    /// main's `--connect http(s)://` path after `EditorApp::new`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn begin_login(&mut self, url: &str, email: &str) {
        self.login.dlg = Some(LoginDlg {
            url: url.to_string(),
            email: email.to_string(),
            password: String::new(),
            remember: true,
            busy: false,
            error: String::new(),
            rx: None,
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn login_window(&mut self, ctx: &egui::Context) {
        let Some(lg) = &mut self.login.dlg else {
            return;
        };
        // result of the background attempt
        let mut done: Option<Result<(SimCtl, String), String>> = None;
        if let Some(rx) = &lg.rx {
            if let Ok(r) = rx.try_recv() {
                done = Some(r);
            }
        }
        match done {
            Some(Ok((ctl, token))) => {
                self.ctl = ctl;
                self.login.connect_url = lg.url.clone();
                self.login.connect_tok = token.clone();
                self.login.reconnect_delay = 2.0;
                if lg.remember {
                    self.login.saved_token = Some((lg.url.clone(), token));
                }
                self.login.dlg = None;
                self.status = "connected".into();
                return;
            }
            Some(Err(e)) => {
                lg.busy = false;
                lg.error = e;
                lg.rx = None;
            }
            None => {}
        }
        let mut keep = true;
        app_win(ctx, format!("Log in — {}", lg.url))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label("email");
                ui.text_edit_singleline(&mut lg.email);
                ui.label("password");
                let r = ui.add(egui::TextEdit::singleline(&mut lg.password).password(true));
                ui.checkbox(&mut lg.remember, "remember me on this machine");
                if !lg.error.is_empty() {
                    ui.colored_label(egui::Color32::LIGHT_RED, &lg.error);
                }
                ui.add_enabled_ui(!lg.busy, |ui| {
                    if ui.button("Log in").clicked()
                        || (r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                    {
                        lg.busy = true;
                        lg.error.clear();
                        let (tx, rx) = std::sync::mpsc::channel();
                        let (url, email, pw) =
                            (lg.url.clone(), lg.email.clone(), lg.password.clone());
                        std::thread::spawn(move || {
                            let r = SimCtl::remote_login(&url, &email, &pw).and_then(|tok| {
                                SimCtl::connect_secure(&url, &tok).map(|c| (c, tok))
                            });
                            let _ = tx.send(r.map_err(|e| e.to_string()));
                        });
                        lg.rx = Some(rx);
                    }
                });
                if ui.button("Cancel").clicked() {
                    keep = false;
                }
            });
        if !keep {
            self.login.dlg = None;
        }
    }
}
