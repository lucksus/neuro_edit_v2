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
    pub cmd_tx: mpsc::SyncSender<(u64, ClientMsg)>,
    pub clients: Clients,
    pub jobs: Arc<JobRegistry>,
    /// Live client sockets — `stop` shuts them down so a disabled user's
    /// connections close at once (dropping writers alone only
    /// half-closes; the reader side would sit on read forever).
    pub socks: Mutex<Vec<std::net::TcpStream>>,
    abort: Arc<AtomicBool>,
    /// Hard kill — set once by `stop`, never cleared, watched by every
    /// eval/run/build hook so the worker exits promptly even when the
    /// cmd queue holds an `Eval` that would re-clear `abort`.
    shutdown: Arc<AtomicBool>,
    /// Last time a client was attached (or the run started).
    last_active: Mutex<Instant>,
    worker: Mutex<Option<JoinHandle<()>>>,
    /// Live while the worker thread runs: set before spawn, cleared by
    /// a drop guard inside the thread — so a panic clears it too.
    /// (17C: `stop()` `.take()`s the join handle, so "no handle" is
    /// NOT a liveness signal.)
    alive: Arc<AtomicBool>,
}

/// Clears the worker's `alive` flag on scope exit — normal return or
/// panic unwinding alike (17C).
struct WorkerAlive(Arc<AtomicBool>);
impl Drop for WorkerAlive {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

/// How long `stop()` waits for the worker before detaching — a run
/// whose hooks are all armed exits within a few ms; this bound only
/// covers a pathological case and keeps `stop()` itself from stalling
/// the admin action (or reaper) that called it.
const STOP_JOIN_TIMEOUT: Duration = Duration::from_secs(3);

impl UserRun {
    /// Attached clients right now.
    pub fn client_count(&self) -> usize {
        self.clients.lock().unwrap().len()
    }
    /// The run's soft-abort flag — `ws_client_loop` sets it directly on
    /// `ClientMsg::Abort` so Stop reaches a worker busy in an eval
    /// (a queued Abort would wait behind the eval it means to kill).
    pub fn abort_flag(&self) -> Arc<AtomicBool> {
        self.abort.clone()
    }
    pub fn touch(&self) {
        *self.last_active.lock().unwrap() = Instant::now();
    }
    fn idle_for(&self) -> Duration {
        self.last_active.lock().unwrap().elapsed()
    }
    /// Worker exited on its own (panic or finished) — a dead run must
    /// be replaced, not left in the map to soak up client connections.
    /// Reads the same liveness flag as `worker_exited` (17C): after
    /// `stop()` took the handle, "no handle" would lie.
    pub fn dead(&self) -> bool {
        self.worker_exited()
    }
    /// The worker thread has really exited — the drop guard ran, even
    /// through a panic. Unlike `dead`-via-handle this stays meaningful
    /// after `stop()` detached.
    pub fn worker_exited(&self) -> bool {
        !self.alive.load(Ordering::Relaxed)
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
    ///
    /// Order matters: `shutdown` first (unconditional), then `abort`,
    /// then Bye (redundant — the loop also breaks on `shutdown`), then
    /// close the transports, then a BOUNDED join. Everything is
    /// fire-and-forget so `stop()` itself can never block.
    fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.abort.store(true, Ordering::Relaxed);
        self.jobs.cancel_all();
        let _ = self.cmd_tx.try_send((0, ClientMsg::Bye));
        self.clients.lock().unwrap().clear();
        for s in self.socks.lock().unwrap().drain(..) {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
        if let Some(w) = self.worker.lock().unwrap().take() {
            let deadline = Instant::now() + STOP_JOIN_TIMEOUT;
            while !w.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            if w.is_finished() {
                let _ = w.join();
            } else {
                // Detach: the shutdown flag is still armed, so the worker
                // exits at its next hook poll. We simply stop waiting.
                eprintln!("run worker slow to die — detaching (shutdown armed)");
            }
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
    /// `user_id → status` lookup for the reaper: a user whose account is
    /// no longer `active` (CLI `admin disable` runs in another process,
    /// so only the DB tells us) loses their run on the next tick.
    pub user_status: Option<Arc<dyn Fn(i64) -> Option<String> + Send + Sync>>,
    /// Sim-size ceiling per run (server flags `--max-neurons` &c.).
    pub sim_limits: sim::SimLimits,
    /// Live-job caps per user + the wall-clock limit each job gets.
    pub max_jobs: usize,
    pub job_wall: Duration,
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
    /// A run whose worker died (panic) is replaced instead of handed
    /// out — without this, clients attach to a dead cmd channel and the
    /// run only clears at the next idle reap (up to `--idle-minutes`).
    pub fn get_or_start(
        &self,
        user_id: i64,
        quota_bytes: u64,
    ) -> Result<Arc<UserRun>, &'static str> {
        // A worker that dies instantly after spawn would loop forever —
        // bound the replacements.
        for _ in 0..4 {
            let mut runs = self.runs.lock().unwrap();
            if let Some(r) = runs.get(&user_id) {
                if r.dead() {
                    eprintln!("user {user_id}: worker died — replacing the run");
                    let dead = runs.remove(&user_id).unwrap();
                    // Release the map before stop(): it can take seconds.
                    drop(runs);
                    dead.stop();
                    continue;
                }
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
            return Ok(run);
        }
        Err("fs")
    }

    fn spawn(&self, data_dir: PathBuf, quota_bytes: u64) -> UserRun {
        let (cmd_tx, cmd_rx) = mpsc::sync_channel(crate::CMD_QUEUE);
        let clients: Clients = Arc::new(Mutex::new(HashMap::new()));
        let abort = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));
        let jobs = Arc::new(JobRegistry::new(JobCtx {
            backend: self.opts.backend,
            seed: self.opts.seed,
            cfg: self.opts.cfg.clone(),
            data_dir: Some(data_dir.clone()),
            quota_bytes,
            sim_limits: self.opts.sim_limits,
            max_jobs: self.opts.max_jobs,
            wall: self.opts.job_wall,
        }));
        let opts = ServeOpts {
            backend: self.opts.backend,
            seed: self.opts.seed,
            cfg: self.opts.cfg.clone(),
            data_dir: Some(data_dir),
            max_upload_bytes: self.opts.max_upload_bytes,
            quota_bytes,
            abort: abort.clone(),
            shutdown: shutdown.clone(),
            sim_limits: self.opts.sim_limits,
        };
        let cls = clients.clone();
        let alive = Arc::new(AtomicBool::new(true));
        let alive_guard = WorkerAlive(alive.clone());
        let worker = std::thread::Builder::new()
            .name("sim-user".into())
            .spawn(move || {
                // 17C: cleared when this thread exits — return or panic.
                let _alive_guard = alive_guard;
                let emit: Emit =
                    std::rc::Rc::new(std::cell::RefCell::new(move |ev: ServerEvent| {
                        let bytes = proto::encode(&ev.msg);
                        let mut cls = cls.lock().unwrap();
                        crate::fanout(&mut cls, ev.to, &bytes);
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
            shutdown,
            last_active: Mutex::new(Instant::now()),
            worker: Mutex::new(Some(worker)),
            alive,
        }
    }

    /// Reap runs idle beyond `--idle-minutes` with no clients and no
    /// live job; also runs whose account left `active` status (CLI
    /// `admin disable`/delete in another process) and runs whose worker
    /// already died. `pub` so tests/tools can force the tick.
    pub fn reap(&self) {
        let idle = self.opts.idle;
        let status_of = &self.opts.user_status;
        let dead: Vec<i64> = self
            .runs
            .lock()
            .unwrap()
            .iter()
            .filter(|(&id, r)| {
                r.dead()
                    || status_of
                        .as_ref()
                        .map(|f| f(id).as_deref() != Some("active"))
                        .unwrap_or(false)
                    || (r.client_count() == 0 && !r.has_live_job() && r.idle_for() > idle)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in dead {
            // Removal in its own statement: `stop()` may take seconds —
            // holding the map lock across it stalls every other user's
            // connect (that's the whole-server freeze from the review).
            let run = self.runs.lock().unwrap().remove(&id);
            if let Some(r) = run {
                eprintln!("user {id}: run reaped — stopping");
                r.stop();
            }
        }
    }

    /// Disable/delete path: the user's run stops and its client
    /// writers (and so their sockets) close at once.
    pub fn drop_user(&self, user_id: i64) {
        let run = self.runs.lock().unwrap().remove(&user_id);
        if let Some(r) = run {
            r.stop();
        }
    }

    /// Which users have a live run — the admin table's "run" column.
    pub fn live(&self) -> Vec<i64> {
        self.runs.lock().unwrap().keys().copied().collect()
    }

    /// The user's live run, if any — used by the admin quota action to
    /// push `SetQuota` onto a running worker.
    pub fn get(&self, user_id: i64) -> Option<Arc<UserRun>> {
        self.runs.lock().unwrap().get(&user_id).cloned()
    }

    pub fn count(&self) -> usize {
        self.runs.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    fn opts(
        users_root: PathBuf,
        status: Option<Arc<dyn Fn(i64) -> Option<String> + Send + Sync>>,
    ) -> RunsOpts {
        RunsOpts {
            backend: BackendKind::Cpu,
            seed: 1,
            cfg: SimConfig::default(),
            users_root,
            max_upload_bytes: usize::MAX,
            max_runs: 4,
            idle: Duration::from_secs(3600),
            user_status: status,
            sim_limits: sim::SimLimits::default(),
            max_jobs: 0,
            job_wall: Duration::MAX,
        }
    }

    fn wait_dead(r: &Arc<UserRun>) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !r.dead() {
            assert!(Instant::now() < deadline, "worker never exited");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// S12: a run whose worker died is replaced, not handed to the
    /// next client (which would see a dead cmd channel and a silent
    /// disconnect).
    #[test]
    fn dead_run_is_replaced() {
        let dir = std::env::temp_dir().join(format!("runs_{}", std::process::id()));
        let runs = Runs::new(opts(dir.clone(), None));
        let r1 = runs.get_or_start(5, 0).unwrap();
        // kill the worker politely — Bye returns from run()
        r1.cmd_tx.try_send((0, ClientMsg::Bye)).unwrap();
        wait_dead(&r1);
        let r2 = runs.get_or_start(5, 0).unwrap();
        assert!(!Arc::ptr_eq(&r1, &r2), "dead run must be replaced");
        assert!(!r2.dead());
        r2.stop();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// S11: a user disabled from another process (CLI `admin
    /// disable`) is dropped by the next reaper tick — the status
    /// closure is the only channel into the live server.
    #[test]
    fn reaper_drops_disabled_user() {
        let dir = std::env::temp_dir().join(format!("runs11_{}", std::process::id()));
        let status = Arc::new(StdMutex::new("active".to_string()));
        let st = status.clone();
        let runs = Runs::new(opts(
            dir.clone(),
            Some(Arc::new(move |_: i64| Some(st.lock().unwrap().clone()))),
        ));
        let r = runs.get_or_start(7, 0).unwrap();
        assert_eq!(runs.live(), vec![7]);
        // CLI disable lands in the DB — the closure now reports it
        *status.lock().unwrap() = "disabled".into();
        runs.reap();
        assert!(runs.live().is_empty(), "disabled user's run must drop");
        wait_dead(&r);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
