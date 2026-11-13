//! Stage 12 integration tests: real multi-user servers on random
//! loopback ports (temp state/data dirs, `--cpu`). HTTP via a tiny
//! blocking client; WS via `ClientConn` against `/ws`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use proto::{ClientMsg, ServerMsg};
use server::accounts::Accounts;
use server::multi::{Multi, MultiOpts};
use server::ClientConn;
use sim::BackendKind;

// ---------------------------------------------------------------------------
// tiny blocking HTTP client (enough for forms, cookies, redirects)
// ---------------------------------------------------------------------------

struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Resp {
    fn header(&self, k: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(k))
            .map(|(_, v)| v.as_str())
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    fn cookie(&self, name: &str) -> Option<String> {
        self.header("set-cookie")?
            .split(';')
            .next()?
            .strip_prefix(&format!("{name}="))
            .map(|s| s.to_string())
    }
}

fn http(addr: &str, method: &str, path: &str, headers: &[(&str, &str)], body: &str) -> Resp {
    let mut s = TcpStream::connect(addr).unwrap();
    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    req.push_str(body);
    s.write_all(req.as_bytes()).unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let sep = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| p + 4)
        .unwrap_or(raw.len());
    let head = String::from_utf8_lossy(&raw[..sep]).to_string();
    let mut lines = head.split("\r\n");
    let status: u16 = lines
        .next()
        .unwrap_or("")
        .split_whitespace()
        .nth(1)
        .unwrap_or("0")
        .parse()
        .unwrap_or(0);
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    Resp {
        status,
        headers,
        body: raw[sep..].to_vec(),
    }
}

fn form(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={}", urlenc(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn urlenc(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            _ => format!("%{:02X}", c as u32),
        })
        .collect()
}

fn origin_for(addr: &str) -> String {
    format!("http://{addr}")
}

fn login(addr: &str, email: &str, pw: &str) -> Resp {
    http(
        addr,
        "POST",
        "/login",
        &[
            ("content-type", "application/x-www-form-urlencoded"),
            ("origin", &origin_for(addr)),
        ],
        &form(&[("email", email), ("password", pw)]),
    )
}

fn signup(addr: &str, email: &str, name: &str, pw: &str, note: &str) -> Resp {
    http(
        addr,
        "POST",
        "/signup",
        &[
            ("content-type", "application/x-www-form-urlencoded"),
            ("origin", &origin_for(addr)),
        ],
        &form(&[
            ("email", email),
            ("name", name),
            ("password", pw),
            ("note", note),
        ]),
    )
}

fn ws_addr(addr: &str, token: Option<&str>) -> anyhow::Result<ClientConn> {
    match token {
        Some(t) => ClientConn::connect_url(
            &format!("ws://{addr}/ws"),
            &[("authorization", &format!("Bearer {t}"))],
        ),
        None => ClientConn::connect_url(&format!("ws://{addr}/ws"), &[]),
    }
}

// ---------------------------------------------------------------------------
// fixture: a Multi server on a random port
// ---------------------------------------------------------------------------

struct Fixture {
    addr: String,
    multi: Arc<Multi>,
    root: PathBuf,
}

fn tmp_root(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ne_multi_{}_{}_{}",
        std::process::id(),
        tag,
        server::accounts::now_ms()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn start(tag: &str, f: impl FnOnce(&mut MultiOpts)) -> Fixture {
    let root = tmp_root(tag);
    let state = root.join("state");
    let data = root.join("data");
    let web = root.join("web");
    std::fs::create_dir_all(&web).unwrap();
    std::fs::write(web.join("index.html"), "<html>app</html>").unwrap();
    let accounts = Arc::new(Accounts::open(&state).unwrap());
    let mut opts = MultiOpts {
        data_root: data,
        web_dir: web,
        public_url: None,
        trust_proxy: vec![],
        backend: BackendKind::Cpu,
        seed: 1,
        cfg: sim::SimConfig::default(),
        max_upload_bytes: 64 << 20,
        max_runs: 4,
        idle: Duration::from_secs(900),
        default_quota_bytes: 4 << 20,
        sim_limits: sim::SimLimits::default(),
        max_jobs: 0,
        job_wall: Duration::MAX,
        max_connections: 512,
    };
    f(&mut opts);
    let multi = Arc::new(Multi::new(accounts, opts));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let m = multi.clone();
    std::thread::spawn(move || m.serve(listener).unwrap());
    Fixture { addr, multi, root }
}

fn mkuser(fx: &Fixture, email: &str, name: &str) -> server::accounts::User {
    let a = &fx.multi.accounts;
    let u = a
        .create_user(email, name, "correct horse 12", "t", 4 << 20)
        .unwrap();
    a.set_status(u.id, "active").unwrap();
    a.by_id(u.id).unwrap()
}

fn token_of(fx: &Fixture, email: &str) -> String {
    let r = login(&fx.addr, email, "correct horse 12");
    r.cookie("ne_session")
        .unwrap_or_else(|| panic!("login {email} failed: {} {}", r.status, r.text()))
}

/// First ServerMsg within `t` matching `f`.
fn recv_until(
    c: &mut ClientConn,
    t: Duration,
    mut f: impl FnMut(&ServerMsg) -> bool,
) -> Option<ServerMsg> {
    let deadline = Instant::now() + t;
    while Instant::now() < deadline {
        match c.recv_opt(deadline - Instant::now()) {
            Ok(Some(m)) if f(&m) => return Some(m),
            Ok(Some(_)) | Ok(None) => {}
            Err(_) => return None,
        }
    }
    None
}

// ---------------------------------------------------------------------------
// 1. signup → pending → login refused → activate → login → WS
// ---------------------------------------------------------------------------

#[test]
fn signup_pending_activate_login_ws() {
    let fx = start("flow", |_| {});
    let r = signup(&fx.addr, "a@x.dev", "Ann", "correct horse 12", "trying it");
    assert_eq!(303, r.status);
    // pending login → spec'd sentence
    let r = login(&fx.addr, "a@x.dev", "correct horse 12");
    assert!(r.text().contains("waiting for activation"), "{}", r.text());
    // admin activates (same store path the CLI uses)
    let u = fx.multi.accounts.by_email("a@x.dev").unwrap();
    assert_eq!("pending", u.status);
    fx.multi.accounts.set_status(u.id, "active").unwrap();
    // login now issues the cookie…
    let tok = token_of(&fx, "a@x.dev");
    // …and the WS opens and speaks.
    let mut c = ws_addr(&fx.addr, Some(&tok)).expect("ws connect");
    c.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })
    .unwrap();
    let m = recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Hello { .. })
    });
    assert!(m.is_some(), "no Hello");
}

// ---------------------------------------------------------------------------
// 2. WS + file calls refused without a session (and with a bad token)
// ---------------------------------------------------------------------------

#[test]
fn ws_refused_without_session() {
    let fx = start("noauth", |_| {});
    assert!(ws_addr(&fx.addr, None).is_err(), "no-session ws must fail");
    assert!(
        ws_addr(
            &fx.addr,
            Some("deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
        )
        .is_err(),
        "bad-token ws must fail"
    );
    // and the refusal is a real HTTP 403, not a hang
    let r = http(&fx.addr, "GET", "/ws", &[("upgrade", "websocket")], "");
    assert_eq!(403, r.status);
}

// ---------------------------------------------------------------------------
// 3. user A cannot reach user B — path tricks or protocol ids
// ---------------------------------------------------------------------------

#[test]
fn users_cannot_cross() {
    let fx = start("isolation", |_| {});
    let a = mkuser(&fx, "a@x.dev", "Ann");
    let b = mkuser(&fx, "b@x.dev", "Bob");
    let ta = token_of(&fx, "a@x.dev");
    let tb = token_of(&fx, "b@x.dev");
    let mut ca = ws_addr(&fx.addr, Some(&ta)).unwrap();
    let mut cb = ws_addr(&fx.addr, Some(&tb)).unwrap();
    for c in [&mut ca, &mut cb] {
        c.send(&ClientMsg::Hello {
            proto: proto::PROTO_VERSION,
        })
        .unwrap();
        recv_until(c, Duration::from_secs(5), |m| {
            matches!(m, ServerMsg::Hello { .. })
        });
    }
    // B plants a file; A tries every trick to read it.
    cb.send(&ClientMsg::WriteFile {
        path: "secret.txt".into(),
        data: b"hunter2".to_vec(),
    })
    .unwrap();
    recv_until(&mut cb, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::FileWritten { .. })
    });

    for bad in [
        format!("../{}/secret.txt", b.id),
        "../secret.txt".into(),
        fx.multi
            .opts_root()
            .join(format!("users/{}/secret.txt", b.id))
            .to_string_lossy()
            .into_owned(),
    ] {
        ca.send(&ClientMsg::ReadFile { path: bad.clone() }).unwrap();
        let m = recv_until(&mut ca, Duration::from_secs(5), |m| {
            matches!(m, ServerMsg::FileData { .. } | ServerMsg::Error(_))
        });
        match m {
            Some(ServerMsg::Error(_)) => {}
            other => panic!("{bad}: expected Error, got {other:?}"),
        }
    }
    // A planted symlink inside A's root pointing into B's.
    #[cfg(unix)]
    {
        let ad = fx.multi.opts_root().join("users").join(a.id.to_string());
        let bd = fx.multi.opts_root().join("users").join(b.id.to_string());
        std::fs::create_dir_all(&ad).unwrap();
        std::os::unix::fs::symlink(&bd, ad.join("peek")).unwrap();
        ca.send(&ClientMsg::ReadFile {
            path: "peek/secret.txt".into(),
        })
        .unwrap();
        let m = recv_until(&mut ca, Duration::from_secs(5), |m| {
            matches!(m, ServerMsg::FileData { .. } | ServerMsg::Error(_))
        });
        assert!(
            matches!(m, Some(ServerMsg::Error(_))),
            "symlink peek: {m:?}"
        );
    }
    // A's dataset + job are invisible to B.
    ca.send(&ClientMsg::UploadDataset {
        name: "ads".into(),
        signal_ext: "f32".into(),
        signal: vec![0; 16],
        meta_json: "{\"channels\":1,\"dt_ms\":1.0,\"steps\":4}".into(),
        labels_json: "{\"events_ms\":[1.0]}".into(),
    })
    .unwrap();
    recv_until(&mut ca, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::FileWritten { .. })
    });
    cb.send(&ClientMsg::ListFiles {
        path: "datasets".into(),
    })
    .unwrap();
    let m = recv_until(&mut cb, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::FileList { .. } | ServerMsg::Error(_))
    });
    match m {
        Some(ServerMsg::FileList { entries, .. }) => {
            assert!(entries.is_empty(), "B sees A's datasets: {entries:?}")
        }
        Some(ServerMsg::Error(_)) => {} // dir absent for B — also fine
        other => panic!("{other:?}"),
    }
    ca.send(&ClientMsg::StartJob {
        name: "ajob".into(),
        script: "print(1)".into(),
    })
    .unwrap();
    cb.send(&ClientMsg::ListJobs).unwrap();
    let m = recv_until(&mut cb, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::JobList { .. })
    });
    match m {
        Some(ServerMsg::JobList { jobs }) => {
            assert!(
                jobs.iter().all(|j| j.name != "ajob"),
                "B sees A's job: {jobs:?}"
            )
        }
        other => panic!("{other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 4. disable → live WS closes, next request refused
// ---------------------------------------------------------------------------

#[test]
fn disable_kills_session_and_ws() {
    let fx = start("disable", |_| {});
    let u = mkuser(&fx, "v@x.dev", "Vic");
    let tok = token_of(&fx, "v@x.dev");
    let mut c = ws_addr(&fx.addr, Some(&tok)).unwrap();
    c.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })
    .unwrap();
    recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Hello { .. })
    });
    // admin disables — same calls the back office makes
    let a = &fx.multi.accounts;
    a.set_status(u.id, "disabled").unwrap();
    a.delete_user_sessions(u.id);
    fx.multi.runs.drop_user(u.id);
    // the socket dies…
    let mut dead = false;
    for _ in 0..30 {
        if c.recv_opt(Duration::from_millis(200)).is_err() {
            dead = true;
            break;
        }
    }
    assert!(dead, "ws stayed alive after disable");
    // …and HTTP with the old cookie is logged out.
    let r = http(
        &fx.addr,
        "GET",
        "/app",
        &[("cookie", &format!("ne_session={tok}"))],
        "",
    );
    assert_eq!(303, r.status, "disabled user's /app should redirect");
}

// ---------------------------------------------------------------------------
// 5. quota: over-quota upload fails, no partial file
// ---------------------------------------------------------------------------

#[test]
fn quota_refuses_and_leaves_no_partial() {
    let fx = start("quota", |o| o.default_quota_bytes = 1 << 20);
    let u = mkuser(&fx, "q@x.dev", "Quo");
    fx.multi.accounts.set_quota(u.id, 100).unwrap(); // 100 B quota
    let tok = token_of(&fx, "q@x.dev");
    let mut c = ws_addr(&fx.addr, Some(&tok)).unwrap();
    c.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })
    .unwrap();
    recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Hello { .. })
    });
    c.send(&ClientMsg::WriteFile {
        path: "seed".into(),
        data: vec![0u8; 90],
    })
    .unwrap();
    recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::FileWritten { .. })
    });
    // 50 more bytes won't fit
    c.send(&ClientMsg::UploadStart {
        path: "big.bin".into(),
        size: 50,
    })
    .unwrap();
    let m = recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::UploadReady { .. } | ServerMsg::Error(_))
    });
    assert!(
        matches!(m, Some(ServerMsg::Error(_))),
        "over-quota upload: {m:?}"
    );
    // no partial file anywhere in the user dir
    let udir = fx.multi.opts_root().join("users").join(u.id.to_string());
    let names: Vec<String> = std::fs::read_dir(&udir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        !names.iter().any(|n| n.contains("upload") || n == "big.bin"),
        "{names:?}"
    );
}

// ---------------------------------------------------------------------------
// 6. chunked upload + download round-trip
// ---------------------------------------------------------------------------

#[test]
fn chunked_round_trip() {
    let fx = start("chunks", |o| o.max_upload_bytes = 8 << 20);
    let _u = mkuser(&fx, "c@x.dev", "Chu");
    let tok = token_of(&fx, "c@x.dev");
    let mut c = ws_addr(&fx.addr, Some(&tok)).unwrap();
    c.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })
    .unwrap();
    recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Hello { .. })
    });
    // 3 MB of deterministic bytes in 512 KB chunks
    let payload: Vec<u8> = (0..3 * 1024 * 1024u32)
        .map(|i| i.wrapping_mul(2654435761) as u8)
        .collect();
    c.send(&ClientMsg::UploadStart {
        path: "big.bin".into(),
        size: payload.len() as u64,
    })
    .unwrap();
    recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::UploadReady { .. })
    })
    .expect("upload not ready");
    for (i, chunk) in payload.chunks(512 * 1024).enumerate() {
        let n = 3 * 1024 * 1024 / (512 * 1024);
        c.send(&ClientMsg::UploadChunk {
            path: "big.bin".into(),
            data: chunk.to_vec(),
            last: i == n - 1,
        })
        .unwrap();
    }
    recv_until(&mut c, Duration::from_secs(10), |m| {
        matches!(m, ServerMsg::FileWritten { .. })
    })
    .expect("no FileWritten");
    c.send(&ClientMsg::DownloadFile {
        path: "big.bin".into(),
    })
    .unwrap();
    let mut got = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match c.recv_opt(deadline - Instant::now()) {
            Ok(Some(ServerMsg::FileChunk { data, done, .. })) => {
                got.extend_from_slice(&data);
                if done {
                    break;
                }
            }
            Ok(Some(_)) | Ok(None) => {
                if Instant::now() >= deadline {
                    panic!("download stalled at {} B", got.len());
                }
            }
            Err(e) => panic!("download: {e}"),
        }
    }
    assert_eq!(payload, got);
}

// ---------------------------------------------------------------------------
// 7. /admin: 404 for non-admin; POSTs refuse bad CSRF / foreign Origin
// ---------------------------------------------------------------------------

#[test]
fn admin_gated_and_post_checks() {
    let fx = start("admin", |_| {});
    let admin = fx
        .multi
        .accounts
        .create_admin("root@x.dev", "Root", "correct horse 12")
        .unwrap()
        .0;
    let _u = mkuser(&fx, "n@x.dev", "Non");
    let admin_tok = token_of(&fx, "root@x.dev");
    let user_tok = token_of(&fx, "n@x.dev");
    // non-admin sees 404
    let r = http(
        &fx.addr,
        "GET",
        "/admin",
        &[("cookie", &format!("ne_session={user_tok}"))],
        "",
    );
    assert_eq!(404, r.status);
    let r = http(&fx.addr, "GET", "/admin", &[], "");
    assert_eq!(404, r.status);
    // admin page renders — pull the csrf token out of a form
    let r = http(
        &fx.addr,
        "GET",
        "/admin",
        &[("cookie", &format!("ne_session={admin_tok}"))],
        "",
    );
    assert_eq!(200, r.status);
    let body = r.text();
    let csrf = body
        .split("name=csrf value=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .expect("no csrf in admin page")
        .to_string();
    let post = |hdrs: &[(&str, &str)], body: &str| {
        let mut h: Vec<(String, String)> =
            vec![("cookie".into(), format!("ne_session={admin_tok}"))];
        for (k, v) in hdrs {
            h.push((k.to_string(), v.to_string()));
        }
        let refs: Vec<(&str, &str)> = h.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        http(&fx.addr, "POST", "/admin/disable", &refs, body)
    };
    // no CSRF → refused
    let r = post(
        &[
            ("content-type", "application/x-www-form-urlencoded"),
            ("origin", &origin_for(&fx.addr)),
        ],
        &form(&[("id", &"1".to_string())]),
    );
    assert_eq!(403, r.status);
    // foreign Origin → refused
    let r = post(
        &[
            ("content-type", "application/x-www-form-urlencoded"),
            ("origin", "https://evil.example"),
        ],
        &form(&[("id", &"1".to_string()), ("csrf", &csrf)]),
    );
    assert_eq!(403, r.status);
    // proper CSRF + Origin → works
    let r = post(
        &[
            ("content-type", "application/x-www-form-urlencoded"),
            ("origin", &origin_for(&fx.addr)),
        ],
        &form(&[("id", &format!("{}", admin.id)), ("csrf", &csrf)]),
    );
    assert_eq!(303, r.status, "{}", r.text());
}

// ---------------------------------------------------------------------------
// 8. the DB holds no plaintext password and no raw session token
// ---------------------------------------------------------------------------

#[test]
fn db_holds_no_secrets() {
    let fx = start("nosecret", |_| {});
    let _u = mkuser(&fx, "s@x.dev", "Sec");
    let tok = token_of(&fx, "s@x.dev");
    let db = fx.root.join("state").join("accounts.db");
    let conn = rusqlite::Connection::open(&db).unwrap();
    let ph: String = conn
        .query_row("SELECT pass_hash FROM users LIMIT 1", [], |r| r.get(0))
        .unwrap();
    assert!(ph.starts_with("$argon2"), "{ph}");
    assert!(!ph.contains("correct horse"), "{ph}");
    let hashes: Vec<String> = conn
        .prepare("SELECT hash FROM sessions")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .flatten()
        .collect();
    assert!(!hashes.is_empty());
    assert!(hashes.iter().all(|h| *h != tok), "raw token stored");
}

// ---------------------------------------------------------------------------
// 9. login rate limit: 10 fails / 10 min → 429
// ---------------------------------------------------------------------------

#[test]
fn login_rate_limit() {
    let fx = start("ratelimit", |_| {});
    let _u = mkuser(&fx, "r@x.dev", "Rat");
    for _ in 0..10 {
        login(&fx.addr, "r@x.dev", "wrong password!!");
    }
    let r = login(&fx.addr, "r@x.dev", "correct horse 12");
    assert_eq!(429, r.status);
    assert!(r.text().contains("too many tries"));
}

// ---------------------------------------------------------------------------
// 10. --max-runs cap and idle stop
// ---------------------------------------------------------------------------

#[test]
fn max_runs_cap_and_idle_stop() {
    let fx = start("caps", |o| {
        o.max_runs = 1;
        o.idle = Duration::from_millis(1);
    });
    let _a = mkuser(&fx, "a@x.dev", "Ann");
    let _b = mkuser(&fx, "b@x.dev", "Bob");
    let ta = token_of(&fx, "a@x.dev");
    let tb = token_of(&fx, "b@x.dev");
    let mut ca = ws_addr(&fx.addr, Some(&ta)).unwrap();
    ca.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })
    .unwrap();
    recv_until(&mut ca, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Hello { .. })
    });
    assert_eq!(1, fx.multi.runs.count());
    // second user gets the "server is full" sentence as an Error frame
    let mut cb = ws_addr(&fx.addr, Some(&tb)).unwrap();
    let m = recv_until(&mut cb, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Error(_))
    });
    match m {
        Some(ServerMsg::Error(e)) => assert!(e.contains("server is full"), "{e}"),
        other => panic!("{other:?}"),
    }
    // idle: A's run dies once A disconnects and the reaper ticks
    drop(ca);
    std::thread::sleep(Duration::from_millis(300));
    fx.multi.runs.reap();
    assert_eq!(0, fx.multi.runs.count(), "idle run not reaped");
    // now B can get in
    let mut cb = ws_addr(&fx.addr, Some(&tb)).unwrap();
    cb.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })
    .unwrap();
    recv_until(&mut cb, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Hello { .. })
    })
    .expect("B's run didn't start");
}

// ---------------------------------------------------------------------------
// --trust-proxy: XFF honoured only from declared proxy peers
// ---------------------------------------------------------------------------

#[test]
fn trust_proxy_gates_xff() {
    let fx = start("xff", |o| {
        o.trust_proxy = vec!["127.0.0.1".parse().unwrap()]
    });
    let _u = mkuser(&fx, "x@x.dev", "Xen");
    // From a trusted peer (loopback here), a spoofed XFF would let an
    // attacker borrow someone else's rate-limit bucket — verify the
    // forwarded IP is what lands in the limiter, and that with NO
    // trusted proxies the header is ignored.
    for i in 0..10 {
        let r = login(&fx.addr, "x@x.dev", "nope nope nope");
        let _ = (r, i);
    }
    // the (ip,email) pair bucket is full → 429 even with the right password
    let r = login(&fx.addr, "x@x.dev", "correct horse 12");
    assert_eq!(429, r.status);
    // S9 fix: a DIFFERENT forwarded IP is a different pair — the email's
    // high ceiling (30) isn't hit, so the victim isn't locked out from
    // other networks (the old flat email bucket gave a 429 here).
    let r = http(
        &fx.addr,
        "POST",
        "/login",
        &[
            ("content-type", "application/x-www-form-urlencoded"),
            ("origin", &origin_for(&fx.addr)),
            ("x-forwarded-for", "9.9.9.9"),
        ],
        &form(&[("email", "x@x.dev"), ("password", "correct horse 12")]),
    );
    assert_eq!(303, r.status, "S9: pair-limit only, not email lockout");
    // Untrusted XFF can't poison a victim IP's bucket either: without
    // trust, XFF "1.2.3.4" must not have accumulated fails — the fixture
    // below proves peer-vs-header handling by re-running with no trust.
    let fx2 = start("xff2", |_| {}); // trust_proxy empty → header ignored
    let _u = mkuser(&fx2, "y@x.dev", "Yen");
    for _ in 0..10 {
        http(
            &fx2.addr,
            "POST",
            "/login",
            &[
                ("content-type", "application/x-www-form-urlencoded"),
                ("origin", &origin_for(&fx2.addr)),
                ("x-forwarded-for", "8.8.8.8"),
            ],
            &form(&[("email", "zz@x.dev"), ("password", "nope nope nope")]),
        );
    }
    // the fails landed on the real peer (loopback) bucket, not 8.8.8.8 —
    // a different email on the same peer is now ip-limited.
    let r = login(&fx2.addr, "y@x.dev", "correct horse 12");
    assert_eq!(429, r.status);
}

// ---------------------------------------------------------------------------
// Stage 16 — security + crash fixes
// ---------------------------------------------------------------------------

/// S1: cookie-authenticated WS upgrades must pass the Origin check —
/// a cross-site upgrade is WS-CSRF. Bearer clients (native editor)
/// carry no ambient credentials, so no-Origin is fine for them.
#[test]
fn ws_cookie_requires_our_origin() {
    let fx = start("s1", |_| {});
    let _u = mkuser(&fx, "o@x.dev", "Ori");
    let tok = token_of(&fx, "o@x.dev");
    let cookie = format!("ne_session={tok}");

    // cookie + foreign Origin → 403
    let bad = ClientConn::connect_url(
        &format!("ws://{}/ws", fx.addr),
        &[("cookie", &cookie), ("origin", "https://evil.example")],
    );
    assert!(bad.is_err(), "cookie+foreign Origin must be refused");

    // cookie + no Origin → 403 (browsers always send Origin on WS)
    let none = ClientConn::connect_url(&format!("ws://{}/ws", fx.addr), &[("cookie", &cookie)]);
    assert!(none.is_err(), "cookie+missing Origin must be refused");

    // cookie + our Origin → connects
    let good = ClientConn::connect_url(
        &format!("ws://{}/ws", fx.addr),
        &[("cookie", &cookie), ("origin", &origin_for(&fx.addr))],
    );
    assert!(good.is_ok(), "cookie+own Origin must connect");

    // Bearer + no Origin → connects (native client)
    let bearer = ws_addr(&fx.addr, Some(&tok));
    assert!(bearer.is_ok(), "bearer without Origin must connect");
}

/// S11: `admin disable` runs in a second process — simulate it with a
/// second Accounts handle on the same state dir. The live socket dies
/// on its next inbound frame (the validity recheck hits the DB).
#[test]
fn cli_disable_drops_live_ws() {
    let fx = start("s11", |_| {});
    let u = mkuser(&fx, "d@x.dev", "Dot");
    let tok = token_of(&fx, "d@x.dev");
    let mut c = ws_addr(&fx.addr, Some(&tok)).unwrap();
    c.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })
    .unwrap();
    recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Hello { .. })
    });

    // the CLI path: a *second* Accounts handle — the running server
    // can only notice through the DB.
    let cli = Accounts::open(&fx.root.join("state")).unwrap();
    cli.set_status(u.id, "disabled").unwrap();
    cli.delete_user_sessions(u.id);
    assert!(
        fx.multi.accounts.session(&tok).is_none(),
        "server should see the session gone"
    );

    // next inbound frame trips the recheck → socket closes. The check
    // is throttled to 500ms per socket (a frame storm shouldn't hammer
    // the DB), so wait out the window before poking.
    std::thread::sleep(Duration::from_millis(600));
    c.send(&ClientMsg::Eval {
        code: "print(\"poke\")".into(),
    })
    .unwrap();
    let mut dead = false;
    for _ in 0..40 {
        if c.recv_opt(Duration::from_millis(200)).is_err() {
            dead = true;
            break;
        }
    }
    assert!(dead, "ws stayed alive after CLI disable");
}

/// S2/S3: `drop_user` while an eval is running must return promptly —
/// the whole point is that a stuck script can't stall admin actions
/// or the HTTP front door.
#[test]
fn drop_user_during_eval_is_fast() {
    let fx = start("s2", |_| {});
    let u = mkuser(&fx, "s@x.dev", "Sal");
    let tok = token_of(&fx, "s@x.dev");
    let mut c = ws_addr(&fx.addr, Some(&tok)).unwrap();
    c.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })
    .unwrap();
    recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Hello { .. })
    });
    // two endless evals queued: the first runs, the second waits.
    // The second must ALSO be endless — a finishing eval would clear
    // `abort`, and then only the `shutdown` flag ends the next eval.
    c.send(&ClientMsg::Eval {
        code: "while true {}".into(),
    })
    .unwrap();
    c.send(&ClientMsg::Eval {
        code: "while true {}".into(),
    })
    .unwrap();
    recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Status { busy: true, .. })
    });

    // D (Stage 17): prove the worker really exits and the front door
    // stays live *while* stop() is still draining the endless eval.
    let run = fx.multi.runs.get(u.id).expect("live run");
    let runs = fx.multi.runs.clone();
    let uid = u.id;
    let stopper = std::thread::spawn(move || {
        runs.drop_user(uid);
    });
    // While stop() is still running, HTTP must answer — previously a
    // map-lock held across the join stalled every request.
    let t = Instant::now();
    let r = http(&fx.addr, "GET", "/", &[], "");
    assert_eq!(200, r.status);
    assert!(
        t.elapsed() < Duration::from_secs(5),
        "GET / stalled behind drop_user's join ({:?})",
        t.elapsed()
    );
    stopper.join().expect("drop_user panicked");
    assert!(
        t.elapsed() < Duration::from_secs(10),
        "drop_user stalled {:?}",
        t.elapsed()
    );
    // and the worker thread actually exited — the eval saw the
    // shutdown flag, it wasn't merely detached. worker_exited reads
    // the thread's own liveness flag: stop() took the join handle, so
    // dead()-via-handle is trivially true and proves nothing.
    let deadline = Instant::now() + Duration::from_secs(5);
    while !run.worker_exited() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        run.worker_exited(),
        "worker thread still alive after drop_user"
    );
}

/// S2/S3: Abort over the socket ends an endless eval — the flag is
/// set on the run directly, not queued behind the very eval it kills.
#[test]
fn abort_over_socket_ends_loop_eval() {
    let fx = start("s3", |_| {});
    let _u = mkuser(&fx, "ab@x.dev", "Abe");
    let tok = token_of(&fx, "ab@x.dev");
    let mut c = ws_addr(&fx.addr, Some(&tok)).unwrap();
    c.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })
    .unwrap();
    recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Hello { .. })
    });
    c.send(&ClientMsg::Eval {
        code: "while true {}".into(),
    })
    .unwrap();
    recv_until(&mut c, Duration::from_secs(5), |m| {
        matches!(m, ServerMsg::Status { busy: true, .. })
    });
    c.send(&ClientMsg::Abort).unwrap();
    // the eval dies; the worker still answers
    c.send(&ClientMsg::Eval {
        code: "print(\"alive\")".into(),
    })
    .unwrap();
    let m = recv_until(
        &mut c,
        Duration::from_secs(10),
        |m| matches!(m, ServerMsg::Console(s) if s == "alive"),
    );
    assert!(m.is_some(), "worker never recovered from Abort");
}

/// S18: admin POSTs refuse self-disable/self-delete, and unknown ids
/// report "no such user" instead of "activated".
#[test]
fn admin_post_guards() {
    let fx = start("s18", |_| {});
    let admin = fx
        .multi
        .accounts
        .create_admin("root@x.dev", "Root", "correct horse 12")
        .unwrap()
        .0;
    let _v = mkuser(&fx, "victim@x.dev", "Vic");
    let admin_tok = token_of(&fx, "root@x.dev");
    let r = http(
        &fx.addr,
        "GET",
        "/admin",
        &[("cookie", &format!("ne_session={admin_tok}"))],
        "",
    );
    let csrf = r
        .text()
        .split("name=csrf value=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .expect("no csrf")
        .to_string();

    let post = |act: &str, pairs: &[(&str, String)]| -> Resp {
        let mut body = format!("csrf={}", urlenc(&csrf));
        for (k, v) in pairs {
            body.push_str(&format!("&{k}={}", urlenc(v)));
        }
        http(
            &fx.addr,
            "POST",
            &format!("/admin/{act}"),
            &[
                ("cookie", &format!("ne_session={admin_tok}")),
                ("content-type", "application/x-www-form-urlencoded"),
                ("origin", &origin_for(&fx.addr)),
            ],
            &body,
        )
    };

    // unknown id → "no such user", not "activated"
    let r = post("activate", &[("id", "424242".to_string())]);
    assert_eq!(303, r.status);
    assert!(
        r.header("location").unwrap().contains("no+such+user"),
        "{}",
        r.header("location").unwrap()
    );

    // self-disable refused (also the last-admin case with one admin)
    let r = post("disable", &[("id", admin.id.to_string())]);
    assert_eq!(303, r.status);
    assert!(
        r.header("location").unwrap().contains("cannot"),
        "{}",
        r.header("location").unwrap()
    );

    // self-delete refused too
    let r = post(
        "delete",
        &[
            ("id", admin.id.to_string()),
            ("confirm", "root@x.dev".to_string()),
        ],
    );
    assert!(r.header("location").unwrap().contains("cannot"));

    // a real disable still works
    let v = fx.multi.accounts.by_email("victim@x.dev").unwrap();
    let r = post("disable", &[("id", v.id.to_string())]);
    assert!(
        r.header("location").unwrap().contains("disabled"),
        "{}",
        r.header("location").unwrap()
    );
}

/// S10: per-IP signup limit — five hits and the sixth is a 429, so a
/// loop can't bury real signups or burn argon2 hashes forever.
#[test]
fn signup_rate_limited() {
    let fx = start("s10", |_| {});
    for i in 0..5 {
        let r = signup(
            &fx.addr,
            &format!("s{i}@x.dev"),
            "Su",
            "correct horse 12",
            "hi",
        );
        assert!(
            r.status != 429,
            "signup {i} rate-limited too early: {}",
            r.status
        );
    }
    let r = signup(&fx.addr, "s5@x.dev", "Su", "correct horse 12", "hi");
    assert_eq!(429, r.status);
}

/// S16: past --max-connections the front door answers 503 instead of
/// spawning unbounded threads.
#[test]
fn conn_cap_answers_503() {
    let fx = start("s16", |o| o.max_connections = 3);
    // three idle sockets pin the cap — the handlers wait on their read
    let _held: Vec<TcpStream> = (0..3)
        .map(|_| TcpStream::connect(&fx.addr).unwrap())
        .collect();
    std::thread::sleep(Duration::from_millis(400));
    // read_to_end may race the RST when the server drops the socket
    // after its 503 — the status line is what matters.
    let mut s = TcpStream::connect(&fx.addr).unwrap();
    s.write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut buf = Vec::new();
    let _ = s.read_to_end(&mut buf);
    let head = String::from_utf8_lossy(&buf);
    assert!(head.starts_with("HTTP/1.1 503"), "{head}");
}

/// Stage 16B: /app's editor shell needs wasm + its own (file-based)
/// scripts — a narrow CSP that still has no 'unsafe-inline' scripts;
/// the account/admin pages keep the strict no-script policy.
#[test]
fn app_csp_allows_wasm_not_inline_scripts() {
    let fx = start("csp", |_| {});
    let _u = mkuser(&fx, "c@x.dev", "Cee");
    let tok = token_of(&fx, "c@x.dev");
    let cookie = format!("ne_session={tok}");

    // a real index.html under --web
    std::fs::write(
        fx.root.join("web").join("index.html"),
        "<html><head></head><body><script type=\"module\" src=\"boot.mjs\"></script></body></html>",
    )
    .unwrap();

    fn script_src(csp: &str) -> String {
        csp.split(';')
            .map(str::trim)
            .find(|d| d.starts_with("script-src"))
            .unwrap_or("")
            .to_string()
    }

    let r = http(&fx.addr, "GET", "/app", &[("cookie", &cookie)], "");
    assert_eq!(200, r.status);
    let csp = r.header("content-security-policy").unwrap().to_string();
    assert!(
        csp.contains("wasm-unsafe-eval"),
        "app CSP must allow wasm: {csp}"
    );
    assert!(
        !script_src(&csp).contains("unsafe-inline"),
        "app script-src must not allow inline scripts: {csp}"
    );
    assert!(
        !csp.contains("unsafe-eval\"") && !csp.contains("'unsafe-eval'"),
        "no unsafe-eval anywhere: {csp}"
    );

    // /app/index.html via serve_static gets the same app policy
    let r = http(
        &fx.addr,
        "GET",
        "/app/index.html",
        &[("cookie", &cookie)],
        "",
    );
    assert_eq!(200, r.status);
    assert!(r
        .header("content-security-policy")
        .unwrap()
        .contains("wasm-unsafe-eval"));

    // the account pages keep the strict policy — no script-src, no wasm
    for path in ["/", "/login", "/admin", "/signup"] {
        let r = http(&fx.addr, "GET", path, &[("cookie", &cookie)], "");
        let csp = r
            .header("content-security-policy")
            .unwrap_or("")
            .to_string();
        if r.status == 200 {
            assert!(
                csp.contains("default-src 'self'") && !csp.contains("wasm-unsafe-eval"),
                "{path} must keep the strict policy: {csp}"
            );
            assert_eq!("", script_src(&csp), "{path} must not allow scripts");
        }
    }
}

/// The bundle keeps its file names across builds, so /app/* must make
/// the browser revalidate: a fresh page paired with a cached old
/// editor.js/wasm once dialled `ws://` from https and read as a WebGPU
/// failure. no-cache + ETag; a matching If-None-Match gets a bodyless
/// 304, a rebuilt file a new tag.
#[test]
fn app_bundle_revalidates_with_etag() {
    let fx = start("etag", |_| {});
    let _u = mkuser(&fx, "e@x.dev", "Ee");
    let tok = token_of(&fx, "e@x.dev");
    let cookie = format!("ne_session={tok}");
    let js = fx.root.join("web").join("editor.js");
    std::fs::write(&js, "export default 1;").unwrap();

    let r = http(
        &fx.addr,
        "GET",
        "/app/editor.js",
        &[("cookie", &cookie)],
        "",
    );
    assert_eq!(200, r.status);
    assert_eq!(Some("no-cache"), r.header("cache-control"));
    let tag = r.header("etag").expect("etag on a static file").to_string();
    assert_eq!("export default 1;", r.text());

    let r = http(
        &fx.addr,
        "GET",
        "/app/editor.js",
        &[("cookie", &cookie), ("if-none-match", &tag)],
        "",
    );
    assert_eq!(304, r.status);
    assert_eq!("", r.text());
    assert_eq!(Some(tag.as_str()), r.header("etag"));

    // a rebuild (new size + mtime) must not match the old tag
    std::thread::sleep(Duration::from_millis(20));
    std::fs::write(&js, "export default 22;").unwrap();
    let r = http(
        &fx.addr,
        "GET",
        "/app/editor.js",
        &[("cookie", &cookie), ("if-none-match", &tag)],
        "",
    );
    assert_eq!(200, r.status);
    assert_eq!("export default 22;", r.text());
    assert_ne!(Some(tag.as_str()), r.header("etag"));

    // the shell itself stays no-store, with no validator
    std::fs::write(
        fx.root.join("web").join("index.html"),
        "<html><head></head></html>",
    )
    .unwrap();
    let r = http(
        &fx.addr,
        "GET",
        "/app/index.html",
        &[("cookie", &cookie)],
        "",
    );
    assert_eq!(Some("no-store"), r.header("cache-control"));
    assert_eq!(None, r.header("etag"));
}
