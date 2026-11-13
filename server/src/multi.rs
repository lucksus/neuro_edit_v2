//! Multi-user mode (`server --accounts <state-dir>`): real HTTP on the
//! same blocking-thread model as the WS path — one connection, one
//! thread, `Connection: close`. No async runtime: pages are
//! server-rendered forms and the WS upgrade hands the socket to the
//! user's own sim run (`runs.rs`). nginx terminates TLS; `--public-url`
//! tells the server its outward origin for cookies and Origin checks.

use std::io::{Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use sim::{BackendKind, SimConfig};

use crate::accounts::{Accounts, Login, User, SESSION_COOKIE};
use crate::pages;
use crate::runs::{Runs, RunsOpts};

pub struct MultiOpts {
    /// `<root>` — users at `<root>/users/<id>/`, deleted users archived
    /// to `<root>/deleted/`.
    pub data_root: PathBuf,
    /// `editor/dist-web` — served under `/app` to active users.
    pub web_dir: PathBuf,
    /// Outward origin, e.g. `https://neuroedit.ad4m.dev` — decides the
    /// cookie's `Secure` flag and the accepted `Origin` on POSTs.
    pub public_url: Option<String>,
    /// Proxy IPs whose `X-Forwarded-*` headers are honoured; anything
    /// else speaking those headers is ignored (it can only spoof its
    /// own rate-limit bucket otherwise).
    pub trust_proxy: Vec<IpAddr>,
    pub backend: BackendKind,
    pub seed: u64,
    pub cfg: SimConfig,
    pub max_upload_bytes: usize,
    pub max_runs: usize,
    pub idle: Duration,
    pub default_quota_bytes: u64,
    /// Sim-size ceiling per user run (`--max-neurons/--max-syns/--max-ring-mb`).
    pub sim_limits: sim::SimLimits,
    /// Live-job cap per user (`--max-jobs`) and wall-clock limit
    /// (`--job-hours`).
    pub max_jobs: usize,
    pub job_wall: Duration,
    /// Concurrent-connection ceiling (`--max-connections`) — threads
    /// are per-conn blocking, so the count is the resource.
    pub max_connections: usize,
}

pub struct Multi {
    pub accounts: Arc<Accounts>,
    pub runs: Arc<Runs>,
    opts: MultiOpts,
    /// Live connections in `serve()` — the `max_connections` gate.
    conns: std::sync::atomic::AtomicUsize,
}

/// Parsed HTTP request — head + bounded body (`Content-Length`; forms
/// stay small, files travel the chunked WS protocol instead).
struct Req {
    method: String,
    target: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    peer: IpAddr,
}

impl Req {
    fn header(&self, k: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(k))
            .map(|(_, v)| v.as_str())
    }
    fn path(&self) -> &str {
        self.target.split(['?', '#']).next().unwrap_or("/")
    }
    fn query(&self, k: &str) -> Option<String> {
        let q = self.target.split('?').nth(1)?;
        url_decode_form(q.as_bytes())
            .into_iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v)
    }
    fn form(&self) -> Vec<(String, String)> {
        url_decode_form(&self.body)
    }
    fn cookie(&self, name: &str) -> Option<String> {
        self.header("cookie")?.split(';').find_map(|c| {
            let c = c.trim();
            c.strip_prefix(&format!("{name}=")).map(|v| v.to_string())
        })
    }
    fn bearer(&self) -> Option<&str> {
        self.header("authorization")?.trim().strip_prefix("Bearer ")
    }
    /// Rate-limit IP: the XFF *last* hop — the one our trusted proxy
    /// appended — so a client can't seed the head of the list with a
    /// fake address. Spoofable headers from anyone else are ignored.
    fn client_ip(&self, trust: &[IpAddr]) -> IpAddr {
        if trust.contains(&self.peer) {
            if let Some(xff) = self.header("x-forwarded-for") {
                if let Some(last) = xff.split(',').last() {
                    if let Ok(ip) = last.trim().parse() {
                        return ip;
                    }
                }
            }
        }
        self.peer
    }
}

/// Byte-wise form decoding — never indexes into `&str` (`%E2` mid-
/// char would panic the old `s[i+1..i+3]` slice), and splits on raw
/// bytes so a lone `%` can't corrupt the pair boundaries either.
fn url_decode_form(body: &[u8]) -> Vec<(String, String)> {
    fn hex(b: u8) -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    }
    fn dec(bs: &[u8]) -> String {
        let mut out = Vec::with_capacity(bs.len());
        let mut i = 0;
        while i < bs.len() {
            match bs[i] {
                b'%' if i + 2 < bs.len()
                    && hex(bs[i + 1]).is_some()
                    && hex(bs[i + 2]).is_some() =>
                {
                    out.push(hex(bs[i + 1]).unwrap() * 16 + hex(bs[i + 2]).unwrap());
                    i += 3;
                }
                b'+' => {
                    out.push(b' ');
                    i += 1;
                }
                c => {
                    out.push(c);
                    i += 1;
                }
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }
    body.split(|&b| b == b'&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p
                .iter()
                .position(|&b| b == b'=')
                .map(|e| (&p[..e], &p[e + 1..]))
                .unwrap_or((p, &[][..]));
            (dec(k), dec(v))
        })
        .collect()
}

/// Read one request: head to `\r\n\r\n`, then `Content-Length` body
/// (64 KB cap — forms only). Returns None on WS upgrade handled by the
/// caller checking `Upgrade` itself.
fn read_req(stream: &mut TcpStream) -> anyhow::Result<Option<Req>> {
    let mut buf = Vec::with_capacity(2048);
    let mut chunk = [0u8; 4096];
    let head_end;
    loop {
        if let Some(p) = find_subslice(&buf, b"\r\n\r\n") {
            head_end = p + 4;
            break;
        }
        if buf.len() > 32 * 1024 {
            anyhow::bail!("request head too long");
        }
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.split("\r\n");
    let rl: Vec<&str> = lines.next().unwrap_or("").split_whitespace().collect();
    let (method, target) = (
        rl.first().unwrap_or(&"GET").to_string(),
        rl.get(1).unwrap_or(&"/").to_string(),
    );
    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    let len: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0)
        .min(64 * 1024);
    let peer = stream
        .peer_addr()
        .map(|a| a.ip())
        .unwrap_or(IpAddr::from([127, 0, 0, 1]));
    while buf.len() < head_end + len {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = buf[head_end..buf.len().min(head_end + len)].to_vec();
    Ok(Some(Req {
        method,
        target,
        headers,
        body,
        peer,
    }))
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Strict policy for the server-rendered pages: no scripts at all,
/// inline style allowed (the pages use style attributes).
const CSP_STRICT: &str = "default-src 'self'; style-src 'self' 'unsafe-inline'";
/// `/app`'s editor shell: the wasm-bindgen glue needs WebAssembly, the
/// bootstrap lives in boot.mjs (no inline scripts, no eval).
const CSP_APP: &str =
    "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'";

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
    extra: &str,
) -> anyhow::Result<()> {
    respond_csp(stream, status, content_type, body, extra, CSP_STRICT)
}

fn respond_csp(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
    extra: &str,
    csp: &str,
) -> anyhow::Result<()> {
    // Baseline headers on everything; HTML responses (which include
    // the authenticated pages) additionally refuse framing and forbid
    // caching so a back-button can't replay a dead session's view.
    // Referrer-Policy is `same-origin`, not `no-referrer`: under
    // `no-referrer` browsers send `Origin: null` on form POSTs, and
    // `origin_ok` then refuses our own /signup and /login forms.
    let security = if content_type.contains("text/html") {
        format!(
            "X-Frame-Options: DENY\r\nX-Content-Type-Options: nosniff\r\n\
             Referrer-Policy: same-origin\r\nCache-Control: no-store\r\n\
             Content-Security-Policy: {csp}\r\n"
        )
    } else {
        "X-Content-Type-Options: nosniff\r\n".to_string()
    };
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Connection: close\r\n{security}{extra}\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    Ok(())
}

fn redirect(stream: &mut TcpStream, to: &str, extra: &str) -> anyhow::Result<()> {
    respond(
        stream,
        "303 See Other",
        "text/plain",
        b"",
        &format!("Location: {to}\r\n{extra}"),
    )
}

fn not_found(stream: &mut TcpStream) -> anyhow::Result<()> {
    respond(stream, "404 Not Found", "text/plain", b"not found", "")
}

impl Multi {
    pub fn new(accounts: Arc<Accounts>, opts: MultiOpts) -> Self {
        std::fs::create_dir_all(opts.data_root.join("users")).ok();
        std::fs::create_dir_all(opts.data_root.join("deleted")).ok();
        let acct = accounts.clone();
        let runs = Runs::new(RunsOpts {
            backend: opts.backend,
            seed: opts.seed,
            cfg: opts.cfg.clone(),
            users_root: opts.data_root.join("users"),
            max_upload_bytes: opts.max_upload_bytes,
            max_runs: opts.max_runs,
            idle: opts.idle,
            // The reaper consults the DB each tick — CLI `admin
            // disable`/delete runs in another process, so this closure
            // is the only channel that reaches it.
            user_status: Some(Arc::new(move |id: i64| acct.by_id(id).map(|u| u.status))),
            sim_limits: opts.sim_limits,
            max_jobs: opts.max_jobs,
            job_wall: opts.job_wall,
        });
        Self {
            accounts,
            runs,
            opts,
            conns: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// `<root>` — users at `<root>/users/<id>/`.
    pub fn opts_root(&self) -> &Path {
        &self.opts.data_root
    }

    /// Blocking accept loop — same shape as `Server::accept_web`.
    /// Conns are capped (`--max-connections`) because each one is a
    /// thread; idle reads/writes time out so a dead peer can't pin its
    /// thread forever.
    pub fn serve(self: &Arc<Self>, listener: TcpListener) -> anyhow::Result<()> {
        const IO_TIMEOUT: Duration = Duration::from_secs(120);
        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("accept: {e}");
                    continue;
                }
            };
            if self.conns.fetch_add(1, Ordering::Relaxed) >= self.opts.max_connections {
                self.conns.fetch_sub(1, Ordering::Relaxed);
                let _ = respond(
                    &mut stream,
                    "503 Service Unavailable",
                    "text/plain",
                    b"too busy",
                    "Retry-After: 5\r\n",
                );
                continue;
            }
            let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
            let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
            let me = self.clone();
            std::thread::spawn(move || {
                let mut s = stream;
                if let Err(e) = me.handle(&mut s) {
                    let _ = e; // closed sockets are routine
                }
                me.conns.fetch_sub(1, Ordering::Relaxed);
            });
        }
        Ok(())
    }

    /// Cookie session → the user, only while `active`. Disabled/
    /// deleted users' sessions are gone from the DB so this is None.
    /// Returns `(user, csrf)` — `session_token` below also reports how
    /// the request authenticated (cookie vs Bearer) and its raw token.
    fn session_user(&self, req: &Req) -> Option<(User, String)> {
        let token = req
            .cookie(SESSION_COOKIE)
            .or_else(|| req.bearer().map(str::to_string))?;
        let (uid, csrf) = self.accounts.session(&token)?;
        let u = self.accounts.by_id(uid)?;
        (u.status == "active").then_some((u, csrf))
    }

    /// `(user, token, via_cookie)` for the request — the WS handshake
    /// needs all three: `via_cookie` gates the Origin check (a browser
    /// cross-site upgrade is only dangerous with ambient cookies), and
    /// `token` feeds the live-connection validity recheck.
    fn session_full(&self, req: &Req) -> Option<(User, String, bool)> {
        let (token, via_cookie) = match req.cookie(SESSION_COOKIE) {
            Some(t) => (t, true),
            None => (req.bearer()?.to_string(), false),
        };
        let (uid, _csrf) = self.accounts.session(&token)?;
        let u = self.accounts.by_id(uid)?;
        (u.status == "active").then_some((u, token, via_cookie))
    }

    /// The origin POSTs must match: `--public-url` when set, else the
    /// request's own scheme+host (X-Forwarded-Proto honoured from a
    /// trusted proxy only).
    fn expected_origin(&self, req: &Req) -> String {
        if let Some(u) = &self.opts.public_url {
            return u.trim_end_matches('/').to_string();
        }
        let proto = if self.opts.trust_proxy.contains(&req.peer) {
            req.header("x-forwarded-proto")
                .unwrap_or("http")
                .to_string()
        } else {
            "http".to_string()
        };
        format!("{proto}://{}", req.header("host").unwrap_or("localhost"))
    }

    fn origin_ok(&self, req: &Req) -> bool {
        match req.header("origin") {
            Some(o) => o.trim_end_matches('/') == self.expected_origin(req),
            None => false,
        }
    }

    fn secure_cookie(&self) -> bool {
        self.opts
            .public_url
            .as_deref()
            .map(|u| u.starts_with("https://"))
            .unwrap_or(false)
    }

    fn cookie_header(&self, token: &str) -> String {
        format!(
            "Set-Cookie: {SESSION_COOKIE}={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=2592000{}",
            if self.secure_cookie() { "; Secure" } else { "" }
        )
    }

    fn quota_of(&self, u: &User) -> u64 {
        if u.quota_bytes > 0 {
            u.quota_bytes
        } else {
            self.opts.default_quota_bytes
        }
    }

    fn handle(&self, stream: &mut TcpStream) -> anyhow::Result<()> {
        stream.set_nodelay(true).ok();
        let req = match read_req(stream)? {
            Some(r) => r,
            None => return Ok(()),
        };
        // WebSocket upgrade — the editor's channel.
        if req
            .header("upgrade")
            .map(|v| v.eq_ignore_ascii_case("websocket"))
            == Some(true)
        {
            return self.handle_ws(stream, req);
        }
        let path = req.path().to_string();
        let sess = self.session_user(&req);
        match (req.method.as_str(), path.as_str()) {
            ("GET", "/") => {
                let full = self.runs.count() >= self.opts.max_runs;
                let (u, csrf, used) = match &sess {
                    Some((u, c)) => {
                        let used = sim::serve::dir_size(
                            &self.opts.data_root.join("users").join(u.id.to_string()),
                        );
                        // render the effective quota (0 on the row = default)
                        let mut u = u.clone();
                        u.quota_bytes = self.quota_of(&u);
                        (Some(u), Some(c.clone()), used)
                    }
                    None => (None, None, 0),
                };
                let body = pages::landing(
                    u.as_ref(),
                    csrf.as_deref(),
                    used,
                    req.query("msg").as_deref(),
                    req.query("err").as_deref(),
                    full,
                    "/app",
                );
                respond(
                    stream,
                    "200 OK",
                    "text/html; charset=utf-8",
                    body.as_bytes(),
                    "",
                )
            }
            ("POST", "/signup") => {
                if !self.origin_ok(&req) {
                    return respond(stream, "403 Forbidden", "text/plain", b"bad origin", "");
                }
                // One argon2 hash + one row per request — cap both the
                // per-IP rate and the total pending pile so a loop
                // can't bury real signups in /admin.
                let ip = req.client_ip(&self.opts.trust_proxy);
                let key = format!("signup:{ip}");
                if self.accounts.limiter.limited_n(&key, 5) {
                    return respond(
                        stream,
                        "429 Too Many Requests",
                        "text/plain",
                        b"too many signups - try later",
                        "",
                    );
                }
                self.accounts.limiter.record_fail(&key);
                if self.accounts.pending_count() >= 1000 {
                    return respond(
                        stream,
                        "503 Service Unavailable",
                        "text/plain",
                        b"signups are full - ask an admin",
                        "",
                    );
                }
                let f: std::collections::HashMap<_, _> = req.form().into_iter().collect();
                match self.accounts.create_user(
                    f.get("email").map(String::as_str).unwrap_or(""),
                    f.get("name").map(String::as_str).unwrap_or(""),
                    f.get("password").map(String::as_str).unwrap_or(""),
                    f.get("note").map(String::as_str).unwrap_or(""),
                    self.opts.default_quota_bytes,
                ) {
                    Ok(_) => redirect(stream, "/?msg=account+created+—+waiting+for+activation", ""),
                    Err(e) => {
                        let body = pages::landing(None, None, 0, None, Some(&e), false, "/app");
                        respond(
                            stream,
                            "200 OK",
                            "text/html; charset=utf-8",
                            body.as_bytes(),
                            "",
                        )
                    }
                }
            }
            // Login CSRF: a foreign Origin means another site is
            // logging our visitor into the attacker's account. The API
            // route is Bearer/native — no Origin header at all — so
            // the check only applies when a browser sends one.
            ("POST", "/login") => {
                if !self.origin_ok(&req) {
                    return respond(stream, "403 Forbidden", "text/plain", b"bad origin", "");
                }
                self.login(stream, &req, false)
            }
            ("POST", "/api/login") => {
                if req.header("origin").is_some() && !self.origin_ok(&req) {
                    return respond(stream, "403 Forbidden", "text/plain", b"bad origin", "");
                }
                self.login(stream, &req, true)
            }
            ("POST", "/logout") => {
                if let Some((_, csrf)) = &sess {
                    let ok_csrf = req
                        .form()
                        .into_iter()
                        .any(|(k, v)| k == "csrf" && &v == csrf);
                    if !ok_csrf || !self.origin_ok(&req) {
                        return respond(stream, "403 Forbidden", "text/plain", b"forbidden", "");
                    }
                }
                if let Some(t) = req.cookie(SESSION_COOKIE) {
                    self.accounts.delete_session(&t);
                } else if let Some(t) = req.bearer() {
                    self.accounts.delete_session(t);
                }
                redirect(
                    stream,
                    "/",
                    &format!("Set-Cookie: {SESSION_COOKIE}=; HttpOnly; Path=/; Max-Age=0\r\n"),
                )
            }
            ("GET", "/admin") => match &sess {
                Some((u, csrf)) if u.is_admin => {
                    let body = pages::admin(
                        &self.accounts.list(),
                        &self.runs.live(),
                        csrf,
                        req.query("did").as_deref(),
                        self.opts.default_quota_bytes,
                    );
                    respond(
                        stream,
                        "200 OK",
                        "text/html; charset=utf-8",
                        body.as_bytes(),
                        "",
                    )
                }
                _ => not_found(stream),
            },
            ("POST", p) if p.starts_with("/admin/") => self.admin_post(stream, &req, sess, p),
            ("GET", "/app") | ("GET", "/app/") => match &sess {
                Some(_) => {
                    // Rebase the editor shell under /app/ so its relative
                    // imports (editor.js, editor_bg.wasm, webgpu-help.mjs,
                    // icons, manifest) hit the session-gated /app/* route.
                    let html = std::fs::read_to_string(self.opts.web_dir.join("index.html"))
                        .unwrap_or_default()
                        .replacen("<head>", "<head><base href=\"/app/\">", 1);
                    respond_csp(
                        stream,
                        "200 OK",
                        "text/html; charset=utf-8",
                        html.as_bytes(),
                        "",
                        CSP_APP,
                    )
                }
                None => redirect(stream, "/", ""),
            },
            ("GET", p) if p.starts_with("/app/") => match &sess {
                Some(_) => serve_static(
                    stream,
                    &self.opts.web_dir,
                    &p[5..],
                    req.header("if-none-match"),
                ),
                None => redirect(stream, "/", ""),
            },
            ("GET", "/assets/about.png") => {
                respond(stream, "200 OK", "image/png", pages::ABOUT_PNG, "")
            }
            ("GET", "/assets/favicon.png") | ("GET", "/favicon.ico") => {
                respond(stream, "200 OK", "image/png", pages::FAVICON_PNG, "")
            }
            ("GET", "/assets/shot_dark.png") => {
                respond(stream, "200 OK", "image/png", pages::SHOT_DARK, "")
            }
            ("GET", "/assets/shot_move.png") => {
                respond(stream, "200 OK", "image/png", pages::SHOT_MOVE, "")
            }
            ("GET", "/assets/shot_connect.png") => {
                respond(stream, "200 OK", "image/png", pages::SHOT_CONNECT, "")
            }
            _ => not_found(stream),
        }
    }

    /// `/login` (form → redirect/page) and `/api/login` (JSON for the
    /// native client). Rate limit: 10 fails / 10 min per IP and per
    /// email; pending/disabled carry the spec'd texts.
    fn login(&self, stream: &mut TcpStream, req: &Req, api: bool) -> anyhow::Result<()> {
        let f: std::collections::HashMap<_, _> = req.form().into_iter().collect();
        let email = f
            .get("email")
            .map(String::as_str)
            .unwrap_or("")
            .to_lowercase();
        let password = f.get("password").map(String::as_str).unwrap_or("");
        let ip = req.client_ip(&self.opts.trust_proxy);
        // Three buckets: per-IP (spray), per (email,ip) pair (the
        // actual brute-force knob at 10/10min), and a high per-email
        // ceiling (30) so an attacker can only slow — never lock — the
        // victim's own login from other IPs.
        let (ip_key, em_key, pair_key) = (
            format!("ip:{ip}"),
            format!("email:{email}"),
            format!("pair:{ip}|{email}"),
        );
        let lim = &self.accounts.limiter;
        let too_many = lim.limited(&ip_key) || lim.limited_n(&em_key, 30) || lim.limited(&pair_key);
        let out: (&str, Option<String>, Option<(i64, String)>) = if too_many {
            ("too many tries — wait a few minutes", None, None)
        } else {
            match self.accounts.verify_login(&email, password) {
                Login::Ok(u) => {
                    lim.clear(&em_key);
                    lim.clear(&pair_key);
                    match self.accounts.create_session(u.id) {
                        Ok((tok, csrf)) => {
                            self.accounts.touch_login(u.id);
                            ("ok", Some(tok), Some((u.id, csrf)))
                        }
                        Err(_) => ("internal error", None, None),
                    }
                }
                Login::Pending => {
                    lim.record_fail(&ip_key);
                    lim.record_fail(&em_key);
                    lim.record_fail(&pair_key);
                    ("your account is waiting for activation", None, None)
                }
                Login::Disabled => {
                    lim.record_fail(&ip_key);
                    lim.record_fail(&em_key);
                    lim.record_fail(&pair_key);
                    ("your account is disabled", None, None)
                }
                Login::Bad => {
                    lim.record_fail(&ip_key);
                    lim.record_fail(&em_key);
                    lim.record_fail(&pair_key);
                    ("wrong email or password", None, None)
                }
            }
        };
        let (text, token, _who) = out;
        if api {
            let (status, body) = match &token {
                Some(t) => ("200 OK", serde_json::json!({"token": t}).to_string()),
                None => (
                    if text.starts_with("too many") {
                        "429 Too Many Requests"
                    } else {
                        "403 Forbidden"
                    },
                    serde_json::json!({"error": text}).to_string(),
                ),
            };
            return respond(stream, status, "application/json", body.as_bytes(), "");
        }
        match token {
            Some(t) => redirect(stream, "/", &format!("{}\r\n", self.cookie_header(&t))),
            None => {
                let kind = if text.starts_with("too many") {
                    "err"
                } else {
                    "info"
                };
                let body = pages::landing_with(text, kind);
                let status = if text.starts_with("too many") {
                    "429 Too Many Requests"
                } else {
                    "200 OK"
                };
                respond(
                    stream,
                    status,
                    "text/html; charset=utf-8",
                    body.as_bytes(),
                    "",
                )
            }
        }
    }

    /// `/admin/<action>` POSTs: admin-only, CSRF + Origin enforced.
    fn admin_post(
        &self,
        stream: &mut TcpStream,
        req: &Req,
        sess: Option<(User, String)>,
        path: &str,
    ) -> anyhow::Result<()> {
        let Some((admin, csrf)) = sess else {
            return not_found(stream);
        };
        if !admin.is_admin {
            return not_found(stream);
        }
        if !self.origin_ok(req) {
            return respond(stream, "403 Forbidden", "text/plain", b"bad origin", "");
        }
        let f: std::collections::HashMap<_, _> = req.form().into_iter().collect();
        if f.get("csrf").map(String::as_str) != Some(csrf.as_str()) {
            return respond(stream, "403 Forbidden", "text/plain", b"bad csrf", "");
        }
        let id: i64 = f.get("id").and_then(|s| s.parse().ok()).unwrap_or(-1);
        let act = path.trim_start_matches("/admin/");
        let Some(target) = self.accounts.by_id(id) else {
            return redirect(stream, "/admin?did=no+such+user", "");
        };
        // You can't disable, delete, or de-admin yourself — and the
        // same for the last active admin either way around.
        let demoting = matches!(act, "disable" | "delete" | "revoke-admin")
            && (id == admin.id || (target.is_admin && !self.accounts.has_other_admin(id)));
        if demoting {
            let why = if id == admin.id {
                "cannot do that to yourself"
            } else {
                "last active admin"
            };
            return redirect(stream, &format!("/admin?did={}", why.replace(' ', "+")), "");
        }
        let did = match act {
            "activate" => {
                self.accounts.set_status(id, "active").ok();
                "activated"
            }
            "disable" => {
                self.accounts.set_status(id, "disabled").ok();
                self.accounts.delete_user_sessions(id);
                self.runs.drop_user(id);
                "disabled"
            }
            "quota" => {
                let mb: u64 = f.get("mb").and_then(|s| s.parse().ok()).unwrap_or(0);
                let bytes = mb.saturating_mul(1 << 20);
                self.accounts.set_quota(id, bytes).ok();
                // A live run keeps the old value — push the new one.
                if let Some(run) = self.runs.get(id) {
                    let _ = run.cmd_tx.try_send((0, proto::ClientMsg::SetQuota(bytes)));
                }
                "quota set"
            }
            "grant-admin" => {
                self.accounts.set_admin(id, true).ok();
                "admin granted"
            }
            "revoke-admin" => {
                self.accounts.set_admin(id, false).ok();
                "admin revoked"
            }
            "delete" => {
                let ok = f.get("confirm").map(String::as_str) == Some(target.email.as_str());
                if ok {
                    self.runs.drop_user(id);
                    self.accounts.delete_user(id).ok();
                    let users = self.opts.data_root.join("users");
                    let from = users.join(id.to_string());
                    if from.exists() {
                        let to = self
                            .opts
                            .data_root
                            .join("deleted")
                            .join(format!("{id}-{}", crate::accounts::now_ms()));
                        let _ = std::fs::rename(&from, &to);
                    }
                    "deleted"
                } else {
                    "confirm email mismatch"
                }
            }
            _ => return not_found(stream),
        };
        redirect(stream, &format!("/admin?did={}", did.replace(' ', "+")), "")
    }

    /// `GET /ws` (+ Upgrade): session via cookie or Bearer → the user's
    /// run (started on demand) → the shared client loop.
    fn handle_ws(&self, stream: &mut TcpStream, req: Req) -> anyhow::Result<()> {
        let Some((u, token, via_cookie)) = self.session_full(&req) else {
            return respond(stream, "403 Forbidden", "text/plain", b"login required", "");
        };
        // Cookie sessions ride on ambient browser auth — a cross-site
        // upgrade is a WS-CSRF. Bearer clients (native editor) carry no
        // ambient credentials, so a missing Origin is fine for them but
        // a present one must still be ours when it authenticates by
        // cookie.
        if via_cookie && !self.origin_ok(&req) {
            return respond(stream, "403 Forbidden", "text/plain", b"bad origin", "");
        }
        // Live-connection invalidation: CLI `admin disable`/delete and
        // session revocation all land in the DB; recheck (throttled)
        // per inbound frame so the socket dies on its next message.
        let accounts = self.accounts.clone();
        let uid = u.id;
        let last =
            std::sync::Mutex::new((std::time::Instant::now() - Duration::from_millis(600), true));
        let valid: crate::ValidCheck = Some(Arc::new(move || {
            let mut l = last.lock().unwrap();
            if l.0.elapsed() < Duration::from_millis(500) {
                return l.1;
            }
            l.0 = std::time::Instant::now();
            let ok = accounts.session(&token).map(|(id, _)| id) == Some(uid)
                && accounts
                    .by_id(uid)
                    .map(|u| u.status == "active")
                    .unwrap_or(false);
            l.1 = ok;
            ok
        }));
        let run = match self.runs.get_or_start(u.id, self.quota_of(&u)) {
            Ok(r) => r,
            Err(_) => {
                // Upgrade succeeds then immediately reports + closes —
                // the editor surfaces the sentence verbatim.
                let key = req.header("sec-websocket-key").unwrap_or("").to_string();
                ws_accept(stream, &key)?;
                let mut ws = tungstenite::WebSocket::from_raw_socket(
                    stream.try_clone()?,
                    tungstenite::protocol::Role::Server,
                    None,
                );
                ws.send(tungstenite::Message::Binary(
                    proto::encode(&proto::ServerMsg::Error(
                        "the server is full, try again in a few minutes".into(),
                    ))
                    .into(),
                ))
                .ok();
                return Ok(());
            }
        };
        let key = req.header("sec-websocket-key").unwrap_or("").to_string();
        ws_accept(stream, &key)?;
        run.socks.lock().unwrap().push(stream.try_clone()?);
        let ws = tungstenite::WebSocket::from_raw_socket(
            stream.try_clone()?,
            tungstenite::protocol::Role::Server,
            Some(crate::server_ws_config()),
        );
        let id = RUN_CLIENT_ID.fetch_add(1, Ordering::Relaxed);
        crate::ws_client_loop(
            ws,
            id,
            run.cmd_tx.clone(),
            run.clients.clone(),
            run.jobs.clone(),
            run.abort_flag(),
            valid,
        )?;
        run.touch();
        Ok(())
    }
}

static RUN_CLIENT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn ws_accept(stream: &mut TcpStream, key: &str) -> anyhow::Result<()> {
    let accept = tungstenite::handshake::derive_accept_key(key.as_bytes());
    stream.write_all(
        format!(
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\
             Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        .as_bytes(),
    )?;
    Ok(())
}

/// Static file from `web_dir` (the `--web` tree), component-safe.
///
/// `editor.js` / `editor_bg.wasm` keep their names across builds, so
/// without a cache header a browser may pair a fresh page with a stale
/// bundle (one that still dialled `ws://` from https). Every file goes
/// out `no-cache` with an ETag (size + mtime): the browser revalidates
/// on each load and gets a 304 while the build is unchanged.
fn serve_static(
    stream: &mut TcpStream,
    web_dir: &Path,
    rel: &str,
    if_none_match: Option<&str>,
) -> anyhow::Result<()> {
    let rel = rel.split(['?', '#']).next().unwrap_or("");
    let rel = if rel.is_empty() { "index.html" } else { rel };
    let mut path = web_dir.to_path_buf();
    let mut bad = false;
    for c in std::path::Path::new(rel).components() {
        match c {
            std::path::Component::Normal(n) => path.push(n),
            _ => bad = true,
        }
    }
    let mime = match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") | Some("mjs") => "text/javascript",
        Some("wasm") => "application/wasm",
        Some("css") => "text/css",
        Some("png") => "image/png",
        Some("svg") => "image/svg+xml",
        Some("json") | Some("map") => "application/json",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    };
    // The only html under web_dir is the editor shell — it needs the
    // /app policy (wasm), not the strict page policy.
    let csp = if mime.contains("text/html") {
        CSP_APP
    } else {
        CSP_STRICT
    };
    // html already goes out no-store (respond_csp); no validator there
    let html = mime.contains("text/html");
    let etag = if bad || html {
        None
    } else {
        static_etag(&path)
    };
    if let (Some(tag), Some(inm)) = (&etag, if_none_match) {
        if inm.split(',').any(|t| t.trim() == tag || t.trim() == "*") {
            let extra = format!("Cache-Control: no-cache\r\nETag: {tag}\r\n");
            return respond_csp(stream, "304 Not Modified", mime, b"", &extra, csp);
        }
    }
    match if bad { None } else { std::fs::read(&path).ok() } {
        Some(b) => {
            let extra = match &etag {
                _ if html => String::new(),
                Some(tag) => format!("Cache-Control: no-cache\r\nETag: {tag}\r\n"),
                None => "Cache-Control: no-cache\r\n".to_string(),
            };
            respond_csp(stream, "200 OK", mime, &b, &extra, csp)
        }
        None => not_found(stream),
    }
}

/// Weak validator for a static file: size + mtime, so a rebuilt bundle
/// gets a new tag without hashing 10 MB of wasm per request.
fn static_etag(path: &Path) -> Option<String> {
    let md = std::fs::metadata(path).ok()?;
    if !md.is_file() {
        return None;
    }
    let mtime = md
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    Some(format!(
        "W/\"{:x}-{:x}-{:x}\"",
        md.len(),
        mtime.as_secs(),
        mtime.subsec_nanos()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req_with(headers: &[(&str, &str)], peer: IpAddr) -> Req {
        Req {
            method: "GET".into(),
            target: "/".into(),
            headers: headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            body: Vec::new(),
            peer,
        }
    }

    /// S15: `%` followed by a multi-byte char used to slice a `&str`
    /// mid-char and panic the connection thread. Byte-wise decode must
    /// not panic on any of these.
    #[test]
    fn url_decode_never_panics() {
        for body in [
            b"%a\xc3\xa9" as &[u8], // %a then é
            b"%zz",
            b"%",
            b"%e2%82", // truncated utf-8
            b"a=%&b=%x",
            b"name=J%C3%B8rgen+Ok",
        ] {
            let pairs = url_decode_form(body);
            let _ = pairs; // reaching here without panic is the test
        }
        let pairs = url_decode_form(b"k=v%202&plus=a+b");
        assert_eq!(
            pairs,
            vec![
                ("k".to_string(), "v 2".to_string()),
                ("plus".to_string(), "a b".to_string())
            ]
        );
    }

    /// S9: the rate-limit IP is the LAST XFF hop (the one our proxy
    /// appended) — the client-controlled first hop must not be used.
    #[test]
    fn client_ip_takes_last_xff_hop() {
        let proxy: IpAddr = "10.0.0.1".parse().unwrap();
        let r = req_with(&[("x-forwarded-for", "1.2.3.4, 5.6.7.8")], proxy);
        assert_eq!(r.client_ip(&[proxy]), "5.6.7.8".parse::<IpAddr>().unwrap());
        // an untrusted peer's header is ignored entirely
        let r2 = req_with(
            &[("x-forwarded-for", "1.2.3.4")],
            "9.9.9.9".parse().unwrap(),
        );
        assert_eq!(r2.client_ip(&[proxy]), "9.9.9.9".parse::<IpAddr>().unwrap());
    }
}
