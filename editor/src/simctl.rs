//! The editor's wire endpoint: `send` accepts `ClientMsg`s, a fold keeps
//! `Shared` current from `ServerMsg`s. Two transports, one API:
//!
//! - **native** (`SimCtl::start[_cfg]` for the in-process worker, or
//!   `SimCtl::connect` for a `server` over TCP+tungstenite WebSocket);
//! - **wasm** (`SimCtl::connect` over the browser `WebSocket` — the only
//!   transport there; sending before `onopen` queues).
//!
//! `send_edit` attaches the `base` version the server checks (stale
//! edits are refused + resynced); `Shared::net_version` tracks the
//! last-applied version and `edits_in_flight` lets several edits go out
//! in one frame with optimistic bases.

use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use std::sync::Arc;
#[cfg(any(feature = "native", target_arch = "wasm32"))]
use std::sync::RwLock;

use proto::{ClientMsg, NetEdit};

#[cfg(any(feature = "native", target_arch = "wasm32"))]
use crate::snap::Shared;
use crate::snap::SharedRef;

pub use proto;

pub use proto::PLANE_TAU_MS;
#[cfg(feature = "native")]
pub use sim::serve::geo_from_net;

pub struct SimCtl {
    /// Request channel — the transport's writer drains it (native only;
    /// wasm sends hit the browser socket directly from `send`).
    #[cfg(not(target_arch = "wasm32"))]
    tx: mpsc::Sender<(u64, ClientMsg)>,
    pub shared: SharedRef,
    /// Ask a running chunked run to stop early.
    pub abort: Arc<AtomicBool>,
    /// Remote transports speak the file/dataset/job protocol against
    /// the server's sandbox; local ones are the user's own filesystem.
    pub remote: bool,
    imp: Imp,
}

#[cfg(not(target_arch = "wasm32"))]
struct Imp {
    worker: Option<std::thread::JoinHandle<()>>,
    pump: Option<std::thread::JoinHandle<()>>,
    /// Remote only: kills the socket so the pump's `recv` unblocks.
    remote_shutdown: Option<std::net::TcpStream>,
}

#[cfg(target_arch = "wasm32")]
struct Imp {
    ws: web_sys::WebSocket,
    /// Encoded frames sent before `onopen`.
    queue: std::rc::Rc<std::cell::RefCell<Vec<Vec<u8>>>>,
    // Callbacks kept alive for the connection's life.
    _onmsg: wasm_bindgen::closure::Closure<dyn FnMut(web_sys::MessageEvent)>,
    _onopen: wasm_bindgen::closure::Closure<dyn FnMut(web_sys::Event)>,
    _onclose: wasm_bindgen::closure::Closure<dyn FnMut(web_sys::CloseEvent)>,
}

impl SimCtl {
    #[cfg(feature = "native")]
    pub fn start(net: net_format::Network, backend: sim::BackendKind, seed: u64) -> SimCtl {
        Self::start_cfg(net, backend, seed, sim::SimConfig::default())
    }

    /// Local transport: the sim worker runs `sim::serve::run` on a thread,
    /// speaking `ClientMsg`/`ServerMsg` over channels — the same protocol
    /// a remote transport uses. A pump thread folds `ServerMsg`s into
    /// `Shared`, so the UI keeps one read model regardless of where the
    /// sim lives.
    #[cfg(feature = "native")]
    pub fn start_cfg(
        net: net_format::Network,
        backend: sim::BackendKind,
        seed: u64,
        cfg: sim::SimConfig,
    ) -> SimCtl {
        let shared: SharedRef = Arc::new(RwLock::new(Shared::default()));
        let abort = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<(u64, ClientMsg)>();
        let (etx, erx) = mpsc::channel::<proto::ServerEvent>();

        let ab = abort.clone();
        let worker = std::thread::Builder::new()
            .name("sim-worker".into())
            .spawn(move || {
                let emit: sim::serve::Emit = std::rc::Rc::new(std::cell::RefCell::new(move |ev| {
                    let _ = etx.send(ev);
                }));
                sim::serve::run(
                    net,
                    sim::serve::ServeOpts {
                        backend,
                        seed,
                        cfg,
                        data_dir: None,
                        max_upload_bytes: usize::MAX,
                        quota_bytes: 0,
                        abort: ab,
                    },
                    rx,
                    emit,
                );
            })
            .expect("spawn sim worker");

        let sh = shared.clone();
        let pump = std::thread::Builder::new()
            .name("sim-fold".into())
            .spawn(move || {
                let mut fold = crate::snap::Fold::default();
                for ev in erx {
                    if let Ok(mut w) = sh.write() {
                        fold.apply(&mut w, ev.msg);
                    }
                }
            })
            .expect("spawn sim fold pump");

        let ctl = SimCtl {
            tx,
            shared,
            abort,
            remote: false,
            imp: Imp {
                worker: Some(worker),
                pump: Some(pump),
                remote_shutdown: None,
            },
        };
        // hydrate (the worker also broadcasts an initial Geo, but a
        // targeted Hello/Status keeps the handshake uniform with remote)
        ctl.send(ClientMsg::Hello {
            proto: proto::PROTO_VERSION,
        });
        ctl
    }

    /// Remote transport, native: the same protocol over TCP+tungstenite.
    /// `addr` is a `server` process's bind address (`host:port`).
    #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
    pub fn connect(addr: &str) -> anyhow::Result<SimCtl> {
        Self::connect_url(&format!("ws://{addr}"), None)
    }

    /// Authenticated remote transport: `ws(s)://host/ws` with a session
    /// token as `Authorization: Bearer` (multi-user server).
    #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
    pub fn connect_secure(base_url: &str, token: &str) -> anyhow::Result<SimCtl> {
        let ws_url = base_url
            .trim_end_matches('/')
            .replacen("https://", "wss://", 1)
            .replacen("http://", "ws://", 1)
            + "/ws";
        Self::connect_url(&ws_url, Some(token))
    }

    #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
    fn connect_url(url: &str, token: Option<&str>) -> anyhow::Result<SimCtl> {
        let shared: SharedRef = Arc::new(RwLock::new(Shared::default()));
        let abort = Arc::new(AtomicBool::new(false));
        let bearer = token.map(|t| format!("Bearer {t}"));
        let hdrs: Vec<(&str, &str)> = match &bearer {
            Some(b) => vec![("authorization", b.as_str())],
            None => vec![],
        };
        let conn = server::ClientConn::connect_url(url, &hdrs)
            .map_err(|e| anyhow::anyhow!("connect {url}: {e}"))?;
        let (mut wsend, mut wrecv) = conn.into_split()?;
        let remote_shutdown = wrecv.shutdown_stream().ok();

        let (tx, rx) = mpsc::channel::<(u64, ClientMsg)>();
        let writer = std::thread::Builder::new()
            .name("sim-ws-tx".into())
            .spawn(move || {
                for (_, c) in rx {
                    let bye = matches!(c, ClientMsg::Bye);
                    if wsend.send(&c).is_err() || bye {
                        break;
                    }
                }
            })
            .expect("spawn ws writer");

        let sh = shared.clone();
        let reader = std::thread::Builder::new()
            .name("sim-ws-rx".into())
            .spawn(move || {
                let mut fold = crate::snap::Fold::default();
                while let Ok(m) = wrecv.recv() {
                    if let Ok(mut w) = sh.write() {
                        fold.apply(&mut w, m);
                    }
                }
            })
            .expect("spawn ws reader");

        let ctl = SimCtl {
            tx,
            shared,
            abort,
            remote: true,
            imp: Imp {
                worker: Some(writer),
                pump: Some(reader),
                remote_shutdown,
            },
        };
        ctl.send(ClientMsg::Hello {
            proto: proto::PROTO_VERSION,
        });
        Ok(ctl)
    }

    /// A not-connected transport (multi-user login pending): sends go
    /// nowhere, `shared` stays default — lets the app boot so the login
    /// dialog can run before the real socket exists.
    #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
    pub fn offline() -> SimCtl {
        let (tx, _rx) = mpsc::channel::<(u64, ClientMsg)>();
        std::mem::forget(_rx); // sends return Err — send() ignores them
        SimCtl {
            tx,
            shared: Arc::new(RwLock::new(Shared::default())),
            abort: Arc::new(AtomicBool::new(false)),
            remote: true,
            imp: Imp {
                worker: None,
                pump: None,
                remote_shutdown: None,
            },
        }
    }

    /// `POST /api/login` on a multi-user server → the session token for
    /// `connect_secure`. Errors carry the server's sentence verbatim.
    #[cfg(all(feature = "native", not(target_arch = "wasm32")))]
    pub fn remote_login(base_url: &str, email: &str, password: &str) -> anyhow::Result<String> {
        let url = format!("{}/api/login", base_url.trim_end_matches('/'));
        let cfg = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build();
        let agent = ureq::Agent::new_with_config(cfg);
        let mut resp = agent
            .post(&url)
            .send_form([("email", email), ("password", password)])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let v: serde_json::Value = resp
            .body_mut()
            .read_to_string()
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        if resp.status() == 200 {
            v.get("token")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| anyhow::anyhow!("no token in reply"))
        } else {
            Err(anyhow::anyhow!(
                "{}",
                v.get("error")
                    .and_then(|e| e.as_str())
                    .unwrap_or("login failed")
            ))
        }
    }

    /// Remote transport, wasm: the browser `WebSocket` speaks the same
    /// bincode frames. No threads — `onmessage` folds straight into
    /// `Shared`; sends before `onopen` queue up.
    #[cfg(target_arch = "wasm32")]
    pub fn connect(addr: &str) -> anyhow::Result<SimCtl> {
        use wasm_bindgen::{closure::Closure, JsCast};
        let url = if addr.contains("://") {
            addr.to_string()
        } else {
            format!("ws://{addr}")
        };
        let ws = web_sys::WebSocket::new(&url).map_err(|e| anyhow::anyhow!("ws {url}: {e:?}"))?;
        ws.set_binary_type(web_sys::BinaryType::Arraybuffer);

        let shared: SharedRef = Arc::new(RwLock::new(Shared::default()));
        let abort = Arc::new(AtomicBool::new(false));
        let queue = std::rc::Rc::new(std::cell::RefCell::new(Vec::<Vec<u8>>::new()));

        let sh = shared.clone();
        let onmsg = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
            if let Ok(buf) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
                let bytes = js_sys::Uint8Array::new(&buf).to_vec();
                if let Ok(m) = proto::decode::<proto::ServerMsg>(&bytes) {
                    if let Ok(mut w) = sh.write() {
                        FOLD.with(|f| f.borrow_mut().apply(&mut w, m));
                    }
                }
            }
        }) as Box<dyn FnMut(web_sys::MessageEvent)>);
        ws.set_onmessage(Some(onmsg.as_ref().unchecked_ref()));

        let (q, ws_open) = (queue.clone(), ws.clone());
        let onopen = Closure::wrap(Box::new(move |_: web_sys::Event| {
            for b in q.borrow_mut().drain(..) {
                let _ = ws_open.send_with_u8_array(&b);
            }
        }) as Box<dyn FnMut(web_sys::Event)>);
        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));

        let sh = shared.clone();
        let onclose = Closure::wrap(Box::new(move |_: web_sys::CloseEvent| {
            if let Ok(mut w) = sh.write() {
                w.log("! connection to server closed");
            }
        }) as Box<dyn FnMut(web_sys::CloseEvent)>);
        ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));

        let ctl = SimCtl {
            shared,
            abort,
            imp: Imp {
                ws,
                queue,
                _onmsg: onmsg,
                _onopen: onopen,
                _onclose: onclose,
            },
            remote: true,
        };
        ctl.send(ClientMsg::Hello {
            proto: proto::PROTO_VERSION,
        });
        Ok(ctl)
    }

    /// Send a request — native queues it for the transport writer; wasm
    /// hits the browser socket directly (queued until `onopen`).
    pub fn send(&self, c: ClientMsg) {
        #[cfg(target_arch = "wasm32")]
        {
            let b = proto::encode(&c);
            if self.imp.ws.ready_state() == web_sys::WebSocket::OPEN {
                let _ = self.imp.ws.send_with_u8_array(&b);
            } else {
                self.imp.queue.borrow_mut().push(b);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.tx.send((0, c));
    }

    /// Versioned edit: the base is the last version the server confirmed
    /// plus the edits we've sent since — optimistic ordering that keeps
    /// rapid edits working over a channel. A refusal resyncs us.
    pub fn send_edit(&self, edit: NetEdit) {
        let base = {
            let mut sh = self.shared.write().unwrap();
            let b = sh.net_version + sh.edits_in_flight;
            sh.edits_in_flight += 1;
            b
        };
        self.send(ClientMsg::Edit { base, edit });
    }

    /// Stop a run NOW: flip the shared flag an in-flight eval/run watches,
    /// then queue `ClientMsg::Abort` so `running` clears once the worker
    /// drains. (The queued command alone can't interrupt a blocked eval.)
    pub fn abort_now(&self) {
        self.abort.store(true, Ordering::Relaxed);
        self.send(ClientMsg::Abort);
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for SimCtl {
    fn drop(&mut self) {
        let _ = self.tx.send((0, ClientMsg::Bye));
        if let Some(h) = self.imp.worker.take() {
            let _ = h.join();
        }
        // remote: the server never closes our socket — kill it so the
        // reader's blocking recv returns an error and exits
        if let Some(s) = &self.imp.remote_shutdown {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
        // worker gone → event channel closed → pump exits
        if let Some(h) = self.imp.pump.take() {
            let _ = h.join();
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for SimCtl {
    fn drop(&mut self) {
        let _ = self.imp.ws.close();
    }
}

// The fold lives on the one browser thread — TLS avoids a shared cell.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static FOLD: std::cell::RefCell<crate::snap::Fold> =
        std::cell::RefCell::new(crate::snap::Fold::default());
}
