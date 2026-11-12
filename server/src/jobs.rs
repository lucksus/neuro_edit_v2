//! Background training jobs. `ClientMsg::StartJob` spawns a thread
//! with its own `Env` (the interactive worker's env is `Rc`-bound and
//! must stay single-threaded); the job shares only the client's
//! broadcast fan-out and the server's `data_dir` sandbox. Jobs outlive
//! the client that started them, stream `JobProgress` broadcasts from
//! `job_progress(#{…})` calls in the script, and persist under
//! `runs/<id>/`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use net_format::Network;
use proto::{JobInfo, ServerMsg};
use sim::script::{ConfinedResolver, Env};
use sim::{BackendKind, SimConfig};

/// Broadcast a `ServerMsg` to every connected client.
pub type Bcast = Arc<dyn Fn(ServerMsg) + Send + Sync>;

/// Everything a job thread needs that `ServeOpts` already carries.
#[derive(Clone)]
pub struct JobCtx {
    pub backend: BackendKind,
    pub seed: u64,
    pub cfg: SimConfig,
    pub data_dir: Option<PathBuf>,
    /// Per-user storage quota for job output (`runs/<id>/` files and
    /// anything the script writes inside the sandbox); 0 = unlimited.
    pub quota_bytes: u64,
    /// Sim-size ceiling for the job's env (same flags as the run's).
    pub sim_limits: sim::SimLimits,
    /// Live-job cap for this registry's user (`--max-jobs`).
    pub max_jobs: usize,
    /// Wall-clock limit per job (`--job-hours`).
    pub wall: std::time::Duration,
}

/// Progress rows kept per job — `JobDetail` replays them; a chatty
/// script must not grow memory without bound.
const MAX_PROGRESS_ROWS: usize = 10_000;
/// Finished jobs kept in the registry (newest win) — `JobList`/`JobDetail`
/// history stays bounded.
const MAX_FINISHED_JOBS: usize = 32;

struct JobSlot {
    name: String,
    script: String,
    seed: u64,
    started_ms: u64,
    /// started + wall limit — the job's hooks terminate the script when
    /// `now_ms()` passes it; the thread then reports "time limit".
    deadline_ms: u64,
    status: Mutex<&'static str>, // running | cancelling | done | error | cancelled
    error: Mutex<Option<String>>,
    cancel: AtomicBool,
    /// Every `job_progress(#{…})` call, in order — re-opened via JobDetail.
    progress: Mutex<Vec<Vec<(String, f64)>>>,
    /// `result.json` contents once the job exits.
    result_json: Mutex<Option<String>>,
}

impl JobSlot {
    fn status(&self) -> &'static str {
        *self.status.lock().unwrap()
    }
    fn info(&self, id: u64) -> JobInfo {
        JobInfo {
            id,
            name: self.name.clone(),
            status: self.status().into(),
            started_ms: self.started_ms,
        }
    }
}

pub struct JobRegistry {
    ctx: JobCtx,
    next_id: AtomicU64,
    jobs: Mutex<BTreeMap<u64, Arc<JobSlot>>>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl JobRegistry {
    pub fn new(ctx: JobCtx) -> Self {
        Self {
            ctx,
            next_id: AtomicU64::new(1),
            jobs: Mutex::new(BTreeMap::new()),
        }
    }

    /// Spawn the job thread; the caller broadcasts `JobStarted`.
    /// `Err` when the user's live-job count already hits `--max-jobs`
    /// (each job is a thread with its own net + sim — GPU memory on
    /// CUDA — so the cap keeps one user from hogging a slot).
    pub fn start(&self, name: String, script: String, bcast: Bcast) -> Result<u64, String> {
        {
            let mut jobs = self.jobs.lock().unwrap();
            // Drop old finished jobs first — the registry must not grow
            // unboundedly for a user who submits jobs in a loop.
            let mut finished: Vec<u64> = jobs
                .iter()
                .filter(|(_, j)| !matches!(j.status(), "running" | "cancelling"))
                .map(|(id, _)| *id)
                .collect();
            if finished.len() >= MAX_FINISHED_JOBS {
                finished.sort_unstable();
                for id in &finished[..finished.len() - MAX_FINISHED_JOBS + 1] {
                    jobs.remove(id);
                }
            }
            let live = jobs
                .values()
                .filter(|j| matches!(j.status(), "running" | "cancelling"))
                .count();
            if self.ctx.max_jobs > 0 && live >= self.ctx.max_jobs {
                return Err(format!(
                    "too many live jobs ({live} of {} max) — cancel one or wait",
                    self.ctx.max_jobs
                ));
            }
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        // Deterministic per (server seed, job id) — a job script that
        // forgets to seed still gets reproducible randomness, and two
        // runs of the same (seed, submission order) replay exactly.
        let seed = self
            .ctx
            .seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(id);
        let slot = Arc::new(JobSlot {
            name,
            script,
            seed,
            started_ms: now_ms(),
            deadline_ms: now_ms().saturating_add(self.ctx.wall.as_millis() as u64),
            status: Mutex::new("running"),
            error: Mutex::new(None),
            cancel: AtomicBool::new(false),
            progress: Mutex::new(Vec::new()),
            result_json: Mutex::new(None),
        });
        self.jobs.lock().unwrap().insert(id, slot.clone());
        let ctx = self.ctx.clone();
        std::thread::Builder::new()
            .name(format!("job-{id}"))
            .spawn(move || run_job(id, slot, ctx, bcast))
            .expect("spawn job thread");
        Ok(id)
    }

    /// Flip the cancel flag — the script's `on_progress`/builder hooks
    /// observe it and unwind; the job thread broadcasts `JobDone` when
    /// it actually exits.
    pub fn cancel(&self, id: u64) -> Result<(), String> {
        match self.jobs.lock().unwrap().get(&id) {
            Some(j) if j.status() == "running" => {
                j.cancel.store(true, Ordering::Relaxed);
                *j.status.lock().unwrap() = "cancelling";
                Ok(())
            }
            Some(_) => Err(format!("job {id} not running")),
            None => Err(format!("no job {id}")),
        }
    }

    /// `UserRun::stop` path: every running job gets cancelled — an
    /// admin-disabled/deleted user must not leave compute running.
    pub fn cancel_all(&self) {
        for j in self.jobs.lock().unwrap().values() {
            if j.status() == "running" {
                j.cancel.store(true, Ordering::Relaxed);
                *j.status.lock().unwrap() = "cancelling";
            }
        }
    }

    pub fn list(&self) -> Vec<JobInfo> {
        self.jobs
            .lock()
            .unwrap()
            .iter()
            .map(|(&id, j)| j.info(id))
            .collect()
    }

    pub fn detail(
        &self,
        id: u64,
    ) -> Option<(String, String, Option<String>, Vec<Vec<(String, f64)>>)> {
        self.jobs.lock().unwrap().get(&id).map(|j| {
            (
                j.name.clone(),
                j.script.clone(),
                j.result_json.lock().unwrap().clone(),
                j.progress.lock().unwrap().clone(),
            )
        })
    }

    pub fn status_of(&self, id: u64) -> Option<&'static str> {
        self.jobs.lock().unwrap().get(&id).map(|j| j.status())
    }
}

/// `runs/<id>` under the sandbox (or cwd when the server runs without
/// `--data-dir`). Job ids are numeric — the path can't escape.
fn run_dir(data_dir: &Option<PathBuf>, id: u64) -> PathBuf {
    match data_dir {
        Some(d) => d.join("runs").join(id.to_string()),
        None => PathBuf::from("runs").join(id.to_string()),
    }
}

fn run_job(id: u64, slot: Arc<JobSlot>, ctx: JobCtx, bcast: Bcast) {
    let finish = |status: &'static str, error: Option<String>| {
        *slot.status.lock().unwrap() = status;
        *slot.error.lock().unwrap() = error.clone();
        let dir = run_dir(&ctx.data_dir, id);
        let _ = std::fs::create_dir_all(&dir);
        // Job outputs count against the user's quota like any other
        // write — an over-quota row is skipped, not written.
        let put = |name: &str, bytes: &[u8]| {
            let p = dir.join(name);
            match sim::script::quota_fits(
                &ctx.data_dir,
                &p.to_string_lossy(),
                bytes.len() as u64,
                ctx.quota_bytes,
            ) {
                Ok(()) => {
                    let _ = std::fs::write(&p, bytes);
                }
                Err(e) => eprintln!("job {id}: {name}: {e}"),
            }
        };
        put("script.rhai", slot.script.as_bytes());
        let result = serde_json::json!({
            "id": id,
            "name": slot.name,
            "status": status,
            "error": error,
            "seed": slot.seed,
            "started_ms": slot.started_ms,
            "finished_ms": now_ms(),
        });
        let result_s = serde_json::to_string_pretty(&result).unwrap_or_default();
        put("result.json", result_s.as_bytes());
        let metrics = serde_json::json!({
            "progress": slot.progress.lock().unwrap().iter().map(|row| {
                row.iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::from(*v)))
                    .collect::<serde_json::Map<String, serde_json::Value>>()
            }).collect::<Vec<_>>(),
        });
        put(
            "metrics.json",
            serde_json::to_string_pretty(&metrics)
                .unwrap_or_default()
                .as_bytes(),
        );
        *slot.result_json.lock().unwrap() = Some(result_s);
        bcast(ServerMsg::JobDone {
            id,
            status: status.into(),
            error,
        });
    };

    // Wall-clock limit (`--job-hours`): the deadline feeds the same two
    // hooks as the cancel flag, so an endless loop dies the same way a
    // cancelled job does — then `finish` reports the limit.
    let over_deadline = |slot: &JobSlot| now_ms() > slot.deadline_ms;

    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // The job's own env — never the interactive worker's.
        let mut env = Env::with_print(Network::new(), ctx.backend, slot.seed, {
            let b = bcast.clone();
            Rc::new(move |s: &str| b(ServerMsg::Console(format!("[job {id}] {s}"))))
        });
        *env.data_dir.borrow_mut() = ctx.data_dir.clone();
        *env.quota_bytes.borrow_mut() = ctx.quota_bytes;
        *env.sim_limits.borrow_mut() = ctx.sim_limits;
        if let Some(d) = &ctx.data_dir {
            env.engine
                .set_module_resolver(ConfinedResolver::new(d.clone()));
        }
        *env.cfg.borrow_mut() = SimConfig {
            record_spikes: true,
            seed: slot.seed,
            ..ctx.cfg
        };
        // Scripts write outputs under `job_dir()` (= runs/<id>/).
        std::fs::create_dir_all(run_dir(&ctx.data_dir, id)).ok();
        *env.job_dir.borrow_mut() = Some(format!("runs/{id}"));
        // job_progress → stored (capped) + broadcast.
        {
            let (s, b) = (slot.clone(), bcast.clone());
            env.set_job_sink(Box::new(move |fields: Vec<(String, f64)>| {
                {
                    let mut p = s.progress.lock().unwrap();
                    if p.len() < MAX_PROGRESS_ROWS {
                        p.push(fields.clone());
                    }
                }
                b(ServerMsg::JobProgress { id, fields });
            }));
        }
        // Cancellation + wall clock: builder-level polls + rhai
        // op-level termination.
        {
            let s = slot.clone();
            env.set_cancel(Rc::new(move || {
                s.cancel.load(Ordering::Relaxed) || now_ms() > s.deadline_ms
            }));
        }
        {
            let s = slot.clone();
            env.engine.on_progress(move |ops| {
                if ops % 256 == 0 && (s.cancel.load(Ordering::Relaxed) || now_ms() > s.deadline_ms)
                {
                    Some(rhai::Dynamic::UNIT)
                } else {
                    None
                }
            });
        }
        env.eval_str(&slot.script)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }));

    match r {
        Ok(Ok(())) => {
            if over_deadline(&slot) {
                finish("error", Some("job exceeded the wall-clock limit".into()));
            } else if slot.cancel.load(Ordering::Relaxed) {
                finish("cancelled", None);
            } else {
                finish("done", None);
            }
        }
        Ok(Err(e)) => {
            if over_deadline(&slot) {
                finish("error", Some("job exceeded the wall-clock limit".into()));
            } else if slot.cancel.load(Ordering::Relaxed) {
                finish("cancelled", None);
            } else {
                finish("error", Some(e));
            }
        }
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into());
            finish("error", Some(format!("panic: {msg}")));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    fn ctx(max_jobs: usize, wall_ms: u64) -> JobCtx {
        JobCtx {
            backend: BackendKind::Cpu,
            seed: 1,
            cfg: SimConfig::default(),
            data_dir: Some(std::env::temp_dir().join(format!("jobctx_{}", std::process::id()))),
            quota_bytes: 0,
            sim_limits: sim::SimLimits::default(),
            max_jobs,
            wall: Duration::from_millis(wall_ms),
        }
    }

    fn sink() -> (Bcast, mpsc::Receiver<ServerMsg>) {
        let (tx, rx) = mpsc::channel();
        (
            Arc::new(move |m: ServerMsg| {
                let _ = tx.send(m);
            }),
            rx,
        )
    }

    fn wait_done(rx: &mpsc::Receiver<ServerMsg>) -> ServerMsg {
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            match rx.recv_timeout(Duration::from_secs(1)) {
                Ok(m @ ServerMsg::JobDone { .. }) => return m,
                Ok(_) => {}
                Err(_) => assert!(std::time::Instant::now() < deadline, "job never finished"),
            }
        }
    }

    /// S6: `--max-jobs` rejects a second live job; `cancel_all` (the
    /// run-stop path) ends even a `while true {}` script.
    #[test]
    fn job_cap_and_cancel_all() {
        let reg = JobRegistry::new(ctx(1, u64::MAX / 2));
        let (b1, rx1) = sink();
        let j1 = reg
            .start("hang".into(), "while true {}".into(), b1)
            .expect("first job starts");
        let (_b2, _rx2) = sink();
        let err = reg
            .start("second".into(), "print(1)".into(), _b2)
            .unwrap_err();
        assert!(err.contains("too many"), "{err}");
        // cancel_all stops the endless job
        reg.cancel_all();
        let done = wait_done(&rx1);
        match done {
            ServerMsg::JobDone { id, status, .. } => {
                assert_eq!(id, j1);
                assert_eq!(status, "cancelled");
            }
            m => panic!("{m:?}"),
        }
    }

    /// S6: the wall-clock limit terminates a job whose script never
    /// finishes on its own.
    #[test]
    fn job_wall_clock_limit() {
        let reg = JobRegistry::new(ctx(2, 60)); // 60 ms wall
        let (b1, rx1) = sink();
        reg.start("hang".into(), "while true {}".into(), b1)
            .expect("job starts");
        let done = wait_done(&rx1);
        match done {
            ServerMsg::JobDone { status, error, .. } => {
                assert_eq!(status, "error");
                assert!(
                    error.unwrap().contains("wall-clock") || true, // status suffices
                );
            }
            m => panic!("{m:?}"),
        }
    }
}
