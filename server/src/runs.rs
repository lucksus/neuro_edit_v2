//! Per-user sim runs for multi-user mode. Each active user gets their
//! own `sim::serve` worker (own cmd channel, own job registry, own
//! `data_dir` = `<root>/users/<user-id>/`) — the same piece `Server`
//! builds for the shared run, held per user id.
//!
//! Lifecycle: a run starts on the user's first WS connect; it stops
//! after `--idle-minutes` with no client attached and no running job;
//! `--max-runs` caps live runs; disabling/deleting a user drops their
//! run at once (its clients' sockets close with the writer half).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use net_format::Network;
use proto::{ClientMsg, ServerEvent};
use sim::serve::{Emit, ServeOpts};
use sim::{BackendKind, SimConfig};

use crate::jobs::{JobCtx, JobRegistry};
use crate::Clients;

/// One user's live run — mirrors `Server`'s pieces.
pub struct UserRun {
    pub cmd_tx: mpsc::Sender<(u64, ClientMsg)>,
    pub clients: Clients,
    pub jobs: Arc<JobRegistry>,
    /// Live client sockets — `stop` shuts them down so a disabled user's
    /// connections close at once (dropping writers alone only
    /// half-closes; the reader side would sit on read forever).
    pub socks: Mutex<Vec<std::net::TcpStream>>,
    abort: Arc<AtomicBool>,
    /// Last time a client was attached (or the run started).
    last_active: Mutex<Instant>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl UserRun {
    /// Attached clients right now.
    pub fn client_count(&self) -> usize {
        self.clients.lock().unwrap().len()
    }
    pub fn touch(&self) {
        *self.last_active.lock().unwrap() = Instant::now();
    }
    fn idle_for(&self) -> Duration {
        self.last_active.lock().unwrap().elapsed()
    }
    fn has_live_job(&self) -> bool {
        self.jobs
            .list()
            .iter()
            .any(|j| j.status == "running" || j.status == "cancelling")
    }
    /// Stop the worker and every client socket. `&self` because live
    /// connections hold `Arc`s — `Arc::try_unwrap` would fail exactly
    /// when it matters (a connected client being disabled).
    fn stop(&self) {
        self.abort.store(true, Ordering::Relaxed);
        let _ = self.cmd_tx.send((0, ClientMsg::Bye));
        self.clients.lock().unwrap().clear();
        for s in self.socks.lock().unwrap().drain(..) {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
        if let Some(w) = self.worker.lock().unwrap().take() {
            let _ = w.join();
        }
    }
}

pub struct RunsOpts {
    pub backend: BackendKind,
    pub seed: u64,
    pub cfg: SimConfig,
    /// `<root>` — user dirs live at `<root>/users/<id>/`.
    pub users_root: PathBuf,
    pub max_upload_bytes: usize,
    pub max_runs: usize,
    pub idle: Duration,
}

/// Registry of live runs. Shared between the HTTP/WS handler threads
/// and the reaper.
pub struct Runs {
    opts: RunsOpts,
    runs: Mutex<HashMap<i64, Arc<UserRun>>>,
}

impl Runs {
    pub fn new(opts: RunsOpts) -> Arc<Self> {
        let r = Arc::new(Self {
            opts,
            runs: Mutex::new(HashMap::new()),
        });
        let me = r.clone();
        std::thread::Builder::new()
            .name("run-reaper".into())
            .spawn(move || loop {
                std::thread::sleep(Duration::from_secs(10));
                me.reap();
            })
            .expect("spawn run reaper");
        r
    }

    /// The user's run — started on demand. `Err("full")` when
    /// `--max-runs` is reached; the WS/landing text explains politely.
    pub fn get_or_start(
        &self,
        user_id: i64,
        quota_bytes: u64,
    ) -> Result<Arc<UserRun>, &'static str> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(r) = runs.get(&user_id) {
            r.touch();
            return Ok(r.clone());
        }
        if runs.len() >= self.opts.max_runs {
            return Err("full");
        }
        let dir = self.opts.users_root.join(user_id.to_string());
        if std::fs::create_dir_all(&dir).is_err() {
            return Err("fs");
        }
        let run = self.spawn(dir, quota_bytes);
        run.touch();
        let run = Arc::new(run);
        runs.insert(user_id, run.clone());
        Ok(run)
    }

    fn spawn(&self, data_dir: PathBuf, quota_bytes: u64) -> UserRun {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let clients: Clients = Arc::new(Mutex::new(HashMap::new()));
        let abort = Arc::new(AtomicBool::new(false));
        let jobs = Arc::new(JobRegistry::new(JobCtx {
            backend: self.opts.backend,
            seed: self.opts.seed,
            cfg: self.opts.cfg.clone(),
            data_dir: Some(data_dir.clone()),
        }));
        let opts = ServeOpts {
            backend: self.opts.backend,
            seed: self.opts.seed,
            cfg: self.opts.cfg.clone(),
            data_dir: Some(data_dir),
            max_upload_bytes: self.opts.max_upload_bytes,
            quota_bytes,
            abort: abort.clone(),
        };
        let cls = clients.clone();
        let worker = std::thread::Builder::new()
            .name("sim-user".into())
            .spawn(move || {
                let emit: Emit =
                    std::rc::Rc::new(std::cell::RefCell::new(move |ev: ServerEvent| {
                        let bytes = proto::encode(&ev.msg);
                        let cls = cls.lock().unwrap();
                        match ev.to {
                            Some(id) => {
                                if let Some(tx) = cls.get(&id) {
                                    let _ = tx.send(bytes);
                                }
                            }
                            None => {
                                for tx in cls.values() {
                                    let _ = tx.send(bytes.clone());
                                }
                            }
                        }
                    }));
                sim::serve::run(Network::new(), opts, cmd_rx, emit);
            })
            .expect("spawn user sim worker");
        UserRun {
            cmd_tx,
            clients,
            jobs,
            socks: Mutex::new(Vec::new()),
            abort,
            last_active: Mutex::new(Instant::now()),
            worker: Mutex::new(Some(worker)),
        }
    }

    /// Reap runs idle beyond `--idle-minutes` with no clients and no
    /// live job. `pub` so tests/tools can force the tick.
    pub fn reap(&self) {
        let idle = self.opts.idle;
        let dead: Vec<i64> = self
            .runs
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, r)| r.client_count() == 0 && !r.has_live_job() && r.idle_for() > idle)
            .map(|(id, _)| *id)
            .collect();
        for id in dead {
            if let Some(r) = self.runs.lock().unwrap().remove(&id) {
                eprintln!("user {id}: run idle — stopping");
                r.stop();
            }
        }
    }

    /// Disable/delete path: the user's run stops and its client
    /// writers (and so their sockets) close at once.
    pub fn drop_user(&self, user_id: i64) {
        if let Some(r) = self.runs.lock().unwrap().remove(&user_id) {
            r.stop();
        }
    }

    /// Which users have a live run — the admin table's "run" column.
    pub fn live(&self) -> Vec<i64> {
        self.runs.lock().unwrap().keys().copied().collect()
    }

    pub fn count(&self) -> usize {
        self.runs.lock().unwrap().len()
    }
}
