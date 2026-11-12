//! Local transport: the sim worker runs `sim::serve::run` on a thread,
//! speaking `ClientMsg`/`ServerMsg` over channels — the same protocol a
//! remote transport would use. A pump thread folds `ServerMsg`s into
//! `Shared`, so the UI keeps one read model regardless of where the sim
//! lives.
//!
//! `SimCtl::send_edit` attaches the `base` version the server checks
//! (stale edits are refused + resynced); `Shared::net_version` tracks
//! the last-applied version and `edits_in_flight` lets several edits go
//! out in one frame with optimistic bases.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, RwLock};
use std::thread::JoinHandle;

use net_format::Network;
use proto::{ClientMsg, NetEdit, ServerEvent};
use sim::{BackendKind, SimConfig};

use crate::snap::{Fold, Shared, SharedRef};

pub use proto;
pub use sim::serve::{geo_from_net, PLANE_TAU_MS};

pub struct SimCtl {
    tx: mpsc::Sender<(u64, ClientMsg)>,
    pub shared: SharedRef,
    /// Ask a running chunked run to stop early.
    pub abort: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    pump: Option<JoinHandle<()>>,
    /// Remote only: kills the socket so the pump's `recv` unblocks.
    remote_shutdown: Option<std::net::TcpStream>,
}

impl SimCtl {
    pub fn start(net: Network, backend: BackendKind, seed: u64) -> SimCtl {
        Self::start_cfg(net, backend, seed, SimConfig::default())
    }

    pub fn start_cfg(net: Network, backend: BackendKind, seed: u64, cfg: SimConfig) -> SimCtl {
        let shared: SharedRef = Arc::new(RwLock::new(Shared::default()));
        let abort = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<(u64, ClientMsg)>();
        let (etx, erx) = mpsc::channel::<ServerEvent>();

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
                let mut fold = Fold::default();
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
            worker: Some(worker),
            pump: Some(pump),
            remote_shutdown: None,
        };
        // hydrate (the worker also broadcasts an initial Geo, but a
        // targeted Hello/Status keeps the handshake uniform with remote)
        ctl.send(ClientMsg::Hello {
            proto: proto::PROTO_VERSION,
        });
        ctl
    }

    /// Remote transport: the same protocol over WebSocket. `addr` is a
    /// `server` process's bind address (`host:port`). The UI sees the
    /// same `Shared` — nothing downstream knows the sim is remote.
    pub fn connect(addr: &str) -> anyhow::Result<SimCtl> {
        let shared: SharedRef = Arc::new(RwLock::new(Shared::default()));
        let abort = Arc::new(AtomicBool::new(false));
        let conn = server::ClientConn::connect(addr)
            .map_err(|e| anyhow::anyhow!("connect ws://{addr}: {e}"))?;
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
                let mut fold = Fold::default();
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
            worker: Some(writer),
            pump: Some(reader),
            remote_shutdown,
        };
        ctl.send(ClientMsg::Hello {
            proto: proto::PROTO_VERSION,
        });
        Ok(ctl)
    }

    pub fn send(&self, c: ClientMsg) {
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
        let _ = self.tx.send((0, ClientMsg::Abort));
    }
}

impl Drop for SimCtl {
    fn drop(&mut self) {
        let _ = self.tx.send((0, ClientMsg::Bye));
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
        // remote: the server never closes our socket — kill it so the
        // reader's blocking recv returns an error and exits
        if let Some(s) = &self.remote_shutdown {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
        // worker gone → event channel closed → pump exits
        if let Some(h) = self.pump.take() {
            let _ = h.join();
        }
    }
}
