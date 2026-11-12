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
}

pub struct Multi {
    pub accounts: Arc<Accounts>,
    pub runs: Arc<Runs>,
    opts: MultiOpts,
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
    /// Rate-limit IP: the XFF first hop, but only when the TCP peer is
    /// a declared proxy (`--trust-proxy`).
    fn client_ip(&self, trust: &[IpAddr]) -> IpAddr {
        if trust.contains(&self.peer) {
            if let Some(xff) = self.header("x-forwarded-for") {
                if let Some(first) = xff.split(',').next() {
                    if let Ok(ip) = first.trim().parse() {
                        return ip;
                    }
                }
            }
        }
        self.peer
    }
}

fn url_decode_form(body: &[u8]) -> Vec<(String, String)> {
    let dec = |s: &str| -> String {
        let mut out = Vec::with_capacity(s.len());
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            match b[i] {
                b'%' if i + 2 < b.len() => {
                    let h = u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap_or(b'?');
                    out.push(h);
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
    };
    String::from_utf8_lossy(body)
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
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

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
    extra: &str,
) -> anyhow::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Connection: close\r\n{extra}\r\n",
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
        let runs = Runs::new(RunsOpts {
            backend: opts.backend,
            seed: opts.seed,
            cfg: opts.cfg.clone(),
            users_root: opts.data_root.join("users"),
            max_upload_bytes: opts.max_upload_bytes,
            max_runs: opts.max_runs,
            idle: opts.idle,
        });
        Self {
            accounts,
            runs,
            opts,
        }
    }

    /// `<root>` — users at `<root>/users/<id>/`.
    pub fn opts_root(&self) -> &Path {
        &self.opts.data_root
    }

    /// Blocking accept loop — same shape as `Server::accept_web`.
    pub fn serve(self: &Arc<Self>, listener: TcpListener) -> anyhow::Result<()> {
        for stream in listener.incoming() {
            let stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("accept: {e}");
                    continue;
                }
            };
            let me = self.clone();
            std::thread::spawn(move || {
                let mut s = stream;
                if let Err(e) = me.handle(&mut s) {
                    let _ = e; // closed sockets are routine
                }
            });
        }
        Ok(())
    }

    /// Cookie session → the user, only while `active`. Disabled/
    /// deleted users' sessions are gone from the DB so this is None.
    fn session_user(&self, req: &Req) -> Option<(User, String)> {
        let token = req
            .cookie(SESSION_COOKIE)
            .or_else(|| req.bearer().map(str::to_string))?;
        let (uid, csrf) = self.accounts.session(&token)?;
        let u = self.accounts.by_id(uid)?;
        (u.status == "active").then_some((u, csrf))
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
                        (Some(u.clone()), Some(c.clone()), used)
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
            ("POST", "/login") => self.login(stream, &req, false),
            ("POST", "/api/login") => self.login(stream, &req, true),
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
                Some(_) => serve_static(stream, &self.opts.web_dir, "index.html"),
                None => redirect(stream, "/", ""),
            },
            ("GET", p) if p.starts_with("/app/") => match &sess {
                Some(_) => serve_static(stream, &self.opts.web_dir, &p[5..]),
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
        let (ip_key, em_key) = (format!("ip:{ip}"), format!("email:{email}"));
        let lim = &self.accounts.limiter;
        let too_many = lim.limited(&ip_key) || lim.limited(&em_key);
        let out: (&str, Option<String>, Option<(i64, String)>) = if too_many {
            ("too many tries — wait a few minutes", None, None)
        } else {
            match self.accounts.verify_login(&email, password) {
                Login::Ok(u) => {
                    lim.clear(&em_key);
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
                    ("your account is waiting for activation", None, None)
                }
                Login::Disabled => {
                    lim.record_fail(&ip_key);
                    lim.record_fail(&em_key);
                    ("your account is disabled", None, None)
                }
                Login::Bad => {
                    lim.record_fail(&ip_key);
                    lim.record_fail(&em_key);
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
                self.accounts.set_quota(id, mb * (1 << 20)).ok();
                "quota set"
            }
            "grant-admin" => {
                self.accounts.set_admin(id, true).ok();
                "admin granted"
            }
            "revoke-admin" => {
                if id == admin.id {
                    "cannot revoke yourself"
                } else {
                    self.accounts.set_admin(id, false).ok();
                    "admin revoked"
                }
            }
            "delete" => {
                let ok = f.get("confirm").map(String::as_str)
                    == self.accounts.by_id(id).map(|u| u.email).as_deref();
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
        let sess = self.session_user(&req);
        let Some((u, _csrf)) = sess else {
            return respond(stream, "403 Forbidden", "text/plain", b"login required", "");
        };
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
            None,
        );
        let id = RUN_CLIENT_ID.fetch_add(1, Ordering::Relaxed);
        crate::ws_client_loop(
            ws,
            id,
            run.cmd_tx.clone(),
            run.clients.clone(),
            run.jobs.clone(),
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
fn serve_static(stream: &mut TcpStream, web_dir: &Path, rel: &str) -> anyhow::Result<()> {
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
    match if bad { None } else { std::fs::read(&path).ok() } {
        Some(b) => respond(stream, "200 OK", mime, &b, ""),
        None => not_found(stream),
    }
}
