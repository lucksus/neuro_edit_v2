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
}

struct JobSlot {
    name: String,
    script: String,
    seed: u64,
    started_ms: u64,
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
    pub fn start(&self, name: String, script: String, bcast: Bcast) -> u64 {
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
        id
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
        let _ = std::fs::write(dir.join("script.rhai"), &slot.script);
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
        let _ = std::fs::write(dir.join("result.json"), &result_s);
        let metrics = serde_json::json!({
            "progress": slot.progress.lock().unwrap().iter().map(|row| {
                row.iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::from(*v)))
                    .collect::<serde_json::Map<String, serde_json::Value>>()
            }).collect::<Vec<_>>(),
        });
        let _ = std::fs::write(
            dir.join("metrics.json"),
            serde_json::to_string_pretty(&metrics).unwrap_or_default(),
        );
        *slot.result_json.lock().unwrap() = Some(result_s);
        bcast(ServerMsg::JobDone {
            id,
            status: status.into(),
            error,
        });
    };

    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // The job's own env — never the interactive worker's.
        let mut env = Env::with_print(Network::new(), ctx.backend, slot.seed, {
            let b = bcast.clone();
            Rc::new(move |s: &str| b(ServerMsg::Console(format!("[job {id}] {s}"))))
        });
        *env.data_dir.borrow_mut() = ctx.data_dir.clone();
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
        // job_progress → stored + broadcast.
        {
            let (s, b) = (slot.clone(), bcast.clone());
            env.set_job_sink(Box::new(move |fields: Vec<(String, f64)>| {
                s.progress.lock().unwrap().push(fields.clone());
                b(ServerMsg::JobProgress { id, fields });
            }));
        }
        // Cancellation: builder-level polls + rhai op-level termination.
        {
            let s = slot.clone();
            env.set_cancel(Rc::new(move || s.cancel.load(Ordering::Relaxed)));
        }
        {
            let s = slot.clone();
            env.engine.on_progress(move |ops| {
                if ops % 256 == 0 && s.cancel.load(Ordering::Relaxed) {
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
            if slot.cancel.load(Ordering::Relaxed) {
                finish("cancelled", None);
            } else {
                finish("done", None);
            }
        }
        Ok(Err(e)) => {
            if slot.cancel.load(Ordering::Relaxed) {
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
