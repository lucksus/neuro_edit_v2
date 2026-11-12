//! WebSocket transport for `sim::serve`: one worker thread runs the
//! protocol core; each connected client gets a reader + writer thread.
//! `ClientMsg`s arrive as bincode frames tagged with the connection's id;
//! `ServerEvent`s route to one client (`to: Some`) or broadcast (`None`).

use std::collections::HashMap;
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use net_format::Network;
use proto::{ClientMsg, ServerEvent, ServerMsg};
use sim::serve::{Emit, ServeOpts};

/// One connected client's outbound channel (encoded WS frames).
type ClientTx = mpsc::Sender<Vec<u8>>;
type Clients = Arc<Mutex<HashMap<u64, ClientTx>>>;

pub struct Server {
    cmd_tx: mpsc::Sender<(u64, ClientMsg)>,
    clients: Clients,
    next_id: Arc<AtomicU64>,
    worker: JoinHandle<()>,
    /// Shared stop flag (same one the worker watches).
    pub abort: Arc<AtomicBool>,
}

impl Server {
    /// Start the worker; call `accept` on a listener to serve clients.
    pub fn start(net: Network, opts: ServeOpts) -> Server {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let clients: Clients = Arc::new(Mutex::new(HashMap::new()));
        let next_id = Arc::new(AtomicU64::new(1));

        let abort = opts.abort.clone();
        let cls = clients.clone();
        let worker = std::thread::Builder::new()
            .name("sim-serve".into())
            .spawn(move || {
                // demux: broadcast → every client; targeted → one.
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
                sim::serve::run(net, opts, cmd_rx, emit);
            })
            .expect("spawn sim worker");
        Server {
            cmd_tx,
            clients,
            next_id,
            worker,
            abort,
        }
    }

    /// Accept loop — blocks forever; each connection gets an id and two
    /// threads. Returns only on listener errors.
    pub fn accept(&self, listener: TcpListener) -> anyhow::Result<()> {
        self.accept_web(listener, None)
    }

    /// `accept` that also serves static files from `web_dir` on plain
    /// HTTP requests — one port carries the page, its wasm bundle and
    /// the WebSocket the page connects back to (`server --web`).
    pub fn accept_web(
        &self,
        listener: TcpListener,
        web_dir: Option<std::path::PathBuf>,
    ) -> anyhow::Result<()> {
        for stream in listener.incoming() {
            let stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("accept: {e}");
                    continue;
                }
            };
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let cmd_tx = self.cmd_tx.clone();
            let clients = self.clients.clone();
            let dir = web_dir.clone();
            std::thread::spawn(move || {
                if let Err(e) = handle_client(stream, id, cmd_tx, clients, dir) {
                    eprintln!("client {id}: {e}");
                }
            });
        }
        Ok(())
    }

    /// Same accept loop on a background thread — the `Server` stays
    /// usable (tests drive `send`/`shutdown` while clients connect).
    pub fn spawn_acceptor(&self, listener: TcpListener) -> JoinHandle<anyhow::Result<()>> {
        self.spawn_acceptor_web(listener, None)
    }

    /// `spawn_acceptor` variant that also serves `web_dir` (`--web`).
    pub fn spawn_acceptor_web(
        &self,
        listener: TcpListener,
        web_dir: Option<std::path::PathBuf>,
    ) -> JoinHandle<anyhow::Result<()>> {
        let cmd_tx = self.cmd_tx.clone();
        let clients = self.clients.clone();
        let next_id = self.next_id.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let stream = match stream {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("accept: {e}");
                        continue;
                    }
                };
                let id = next_id.fetch_add(1, Ordering::Relaxed);
                let cmd_tx = cmd_tx.clone();
                let clients = clients.clone();
                let dir = web_dir.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle_client(stream, id, cmd_tx, clients, dir) {
                        eprintln!("client {id}: {e}");
                    }
                });
            }
            Ok(())
        })
    }

    pub fn send(&self, m: ClientMsg) {
        let _ = self.cmd_tx.send((0, m));
    }

    /// Stop the worker (clients stay connected to a dead backend — used
    /// by tests; normal shutdown is just process exit).
    pub fn shutdown(self) {
        let _ = self.cmd_tx.send((0, ClientMsg::Bye));
        let _ = self.worker.join();
    }
}

/// What a just-accepted connection asked for on its first bytes.
enum FrontReq {
    /// `Upgrade: websocket` — carries the Sec-WebSocket-Key to answer.
    Ws(String),
    /// Plain HTTP — the request-target path to serve from `web_dir`.
    Http(String),
}

/// Read the HTTP request head (client blocks for our response, so
/// nothing after `\r\n\r\n` can arrive yet) and classify it.
fn sniff_request(stream: &std::net::TcpStream) -> anyhow::Result<FrontReq> {
    use std::io::Read;
    let mut s = stream;
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 2048];
    loop {
        if find_subslice(&buf, b"\r\n\r\n").is_some() {
            break;
        }
        if buf.len() > 16 * 1024 {
            anyhow::bail!("request head too long");
        }
        let n = s.read(&mut chunk)?;
        if n == 0 {
            anyhow::bail!("eof in request head");
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let text = String::from_utf8_lossy(&buf);
    let mut lines = text.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_string();
    let mut upgrade = false;
    let mut key = None;
    for line in lines {
        if line.is_empty() {
            break;
        }
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        match k.trim().to_ascii_lowercase().as_str() {
            "upgrade" if v.trim().eq_ignore_ascii_case("websocket") => upgrade = true,
            "sec-websocket-key" => key = Some(v.trim().to_string()),
            _ => {}
        }
    }
    Ok(if upgrade {
        match key {
            Some(k) => FrontReq::Ws(k),
            None => anyhow::bail!("websocket upgrade without Sec-WebSocket-Key"),
        }
    } else {
        FrontReq::Http(path)
    })
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Serve one static GET from `web_dir`, then close — no keep-alive.
fn serve_http(
    mut stream: std::net::TcpStream,
    target: &str,
    web_dir: &std::path::Path,
) -> anyhow::Result<()> {
    use std::io::Write;
    let rel = target
        .split(['?', '#'])
        .next()
        .unwrap_or("/")
        .trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };
    let mut path = web_dir.to_path_buf();
    let mut bad = false;
    for c in std::path::Path::new(rel).components() {
        match c {
            std::path::Component::Normal(n) => path.push(n),
            _ => bad = true,
        }
    }
    let body = if bad { None } else { std::fs::read(&path).ok() };
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
    let (status, ct, bytes): (&str, &str, Vec<u8>) = match body {
        Some(b) => ("200 OK", mime, b),
        None => ("404 Not Found", "text/plain", b"not found".to_vec()),
    };
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        bytes.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}

fn handle_client(
    stream: std::net::TcpStream,
    id: u64,
    cmd_tx: mpsc::Sender<(u64, ClientMsg)>,
    clients: Clients,
    web_dir: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    stream.set_nodelay(true).ok();
    let ws = match &web_dir {
        // --web: the same port serves the page over plain HTTP and the
        // wire over Upgrade requests — split them on the request head.
        Some(dir) => match sniff_request(&stream)? {
            FrontReq::Http(path) => return serve_http(stream, &path, dir),
            FrontReq::Ws(key) => {
                use std::io::Write;
                let accept = tungstenite::handshake::derive_accept_key(key.as_bytes());
                let mut s = stream;
                s.write_all(
                    format!(
                        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\
                         Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
                    )
                    .as_bytes(),
                )?;
                tungstenite::WebSocket::from_raw_socket(
                    s,
                    tungstenite::protocol::Role::Server,
                    None,
                )
            }
        },
        None => tungstenite::accept(stream)?,
    };
    // One socket → reader + writer halves via two WebSockets on dup'd fds.
    let mut w_read = ws;
    let write_stream = w_read.get_ref().try_clone()?;
    let mut w_write = tungstenite::WebSocket::from_raw_socket(
        write_stream,
        tungstenite::protocol::Role::Server,
        None,
    );

    // outbound: this client's queue → the write half
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    clients.lock().unwrap().insert(id, tx);
    let cls2 = clients.clone();
    let writer = std::thread::spawn(move || {
        for bytes in rx {
            if w_write
                .send(tungstenite::Message::Binary(bytes.into()))
                .is_err()
            {
                break;
            }
        }
    });

    // inbound: bincode frames → the worker. `Bye` is swallowed: a
    // client leaving must not stop the shared worker — `Server::shutdown`
    // injects Bye straight into cmd_tx, and the socket close cleans this
    // client up either way.
    let r = (|| -> anyhow::Result<()> {
        loop {
            match w_read.read() {
                Ok(tungstenite::Message::Binary(b)) => match proto::decode::<ClientMsg>(&b) {
                    Ok(ClientMsg::Bye) => {}
                    Ok(m) => {
                        if cmd_tx.send((id, m)).is_err() {
                            return Ok(()); // worker gone
                        }
                    }
                    Err(e) => eprintln!("client {id}: bad frame: {e}"),
                },
                Ok(tungstenite::Message::Close(_)) | Err(_) => return Ok(()),
                Ok(_) => {} // pings/text ignored
            }
        }
    })();
    cls2.lock().unwrap().remove(&id);
    drop(writer);
    r
}

/// Convenience for tests/tools: a typed client side of the wire.
pub struct ClientConn {
    ws: tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>,
}

impl ClientConn {
    pub fn connect(addr: &str) -> anyhow::Result<Self> {
        let (ws, _resp) = tungstenite::connect(format!("ws://{addr}"))?;
        Ok(Self { ws })
    }
    pub fn send(&mut self, m: &ClientMsg) -> anyhow::Result<()> {
        self.ws
            .send(tungstenite::Message::Binary(proto::encode(m).into()))?;
        Ok(())
    }
    /// Next `ServerMsg`, or `None` if nothing arrives within `timeout`.
    /// Uses the socket's read timeout; error kinds other than a plain
    /// timeout are returned.
    pub fn recv_opt(&mut self, timeout: Duration) -> anyhow::Result<Option<ServerMsg>> {
        use std::io::ErrorKind;
        use tungstenite::stream::MaybeTlsStream;
        if let MaybeTlsStream::Plain(s) = self.ws.get_mut() {
            s.set_read_timeout(Some(timeout))?;
        }
        match self.ws.read() {
            Ok(tungstenite::Message::Binary(b)) => Ok(Some(proto::decode::<ServerMsg>(&b)?)),
            Ok(_) => Ok(None),
            Err(tungstenite::Error::Io(e))
                if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) =>
            {
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }
    pub fn recv(&mut self) -> anyhow::Result<ServerMsg> {
        use tungstenite::stream::MaybeTlsStream;
        if let MaybeTlsStream::Plain(s) = self.ws.get_mut() {
            s.set_read_timeout(None)?;
        }
        loop {
            match self.ws.read()? {
                tungstenite::Message::Binary(b) => {
                    return Ok(proto::decode::<ServerMsg>(&b)?);
                }
                _ => {}
            }
        }
    }

    /// Split into a blocking writer + reader so two threads can each own
    /// one direction (e.g. the editor's send/pump threads). `ws://` only.
    pub fn into_split(mut self) -> anyhow::Result<(WsWriter, WsReader)> {
        use tungstenite::stream::MaybeTlsStream;
        let dup = match self.ws.get_mut() {
            MaybeTlsStream::Plain(s) => s.try_clone()?,
            _ => anyhow::bail!("can't split a non-TCP websocket"),
        };
        let w =
            tungstenite::WebSocket::from_raw_socket(dup, tungstenite::protocol::Role::Client, None);
        Ok((WsWriter(w), WsReader(self.ws)))
    }
}

/// Write half of a split `ClientConn`.
pub struct WsWriter(tungstenite::WebSocket<std::net::TcpStream>);
/// Read half of a split `ClientConn`.
pub struct WsReader(
    tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>,
);

impl WsWriter {
    pub fn send(&mut self, m: &ClientMsg) -> anyhow::Result<()> {
        self.0
            .send(tungstenite::Message::Binary(proto::encode(m).into()))?;
        Ok(())
    }
}

impl WsReader {
    /// A second handle to the socket for `shutdown()` — lets an owner
    /// unblock a `recv()` parked on another thread (drop/shutdown paths).
    pub fn shutdown_stream(&self) -> std::io::Result<std::net::TcpStream> {
        use tungstenite::stream::MaybeTlsStream;
        match self.0.get_ref() {
            MaybeTlsStream::Plain(s) => s.try_clone(),
            _ => Err(std::io::Error::other("non-TCP websocket")),
        }
    }
    /// Blocking read; `Err` once the server drops the connection.
    pub fn recv(&mut self) -> anyhow::Result<ServerMsg> {
        loop {
            match self.0.read()? {
                tungstenite::Message::Binary(b) => return Ok(proto::decode::<ServerMsg>(&b)?),
                _ => {}
            }
        }
    }
}
