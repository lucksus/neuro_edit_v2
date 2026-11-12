# Deploying neuro_edit as a hosted web app

Multi-user mode turns `server` into a small web application: a landing
page with signup/login, an admin back office at `/admin`, the wasm
editor at `/app`, and a per-user simulator run + per-user file space
behind it. Everything is served over plain HTTP — put nginx in front
for TLS.

The production target is `https://neuroedit.ad4m.dev`: neuro_edit runs
on **miguel** bound to its LAN address, and TLS terminates at an nginx
on **Marvin** (a separate box on the same LAN) which proxies to miguel.

## Layout

```
<state-dir>/accounts.db     users, sessions, rate-limit buckets (SQLite, WAL)
<data-dir>/users/<id>/      one sandboxed file root per user
<data-dir>/deleted/         dirs of deleted users are moved here, never rm'ed
```

Each active user gets a simulator run lazily on their first `/app` or
`/ws` connect, rooted at their own `users/<id>/` — runs never share
mutable state. Runs stop after `--idle-minutes` (default 15) without
clients or jobs, and `--max-runs` (default 4) caps how many are live.

## The server (on miguel)

```sh
# build once
cargo build --release -p server
editor/web/build.sh                       # → editor/dist-web (the /app bundle)

server \
  --bind 10.0.0.20:9337 \                 # miguel's LAN address — accounts mode allows it
  --accounts /var/lib/neuroedit/state \
  --data-dir /var/lib/neuroedit/data \
  --web /opt/neuroedit/dist-web \
  --public-url https://neuroedit.ad4m.dev \
  --trust-proxy 10.0.0.10 \               # Marvin's LAN address (comma-separate for several)
  --max-upload-mb 64 \
  --idle-minutes 15 \
  --max-runs 4 \
  --default-quota-mb 2048
```

Flags that matter in multi-user mode:

- `--accounts DIR` enables the mode; `accounts.db` lives inside. It
  requires `--data-dir` (per-user roots) and `--web` (the `/app` bundle).
- `--public-url` is the outward origin. It sets the session cookie's
  `Secure` flag (https only) and the `Origin` every POST must carry, and
  is the base for links the server emits. Set it to
  `https://neuroedit.ad4m.dev` even though the server itself only speaks
  HTTP — the browser only ever sees the https origin.
- `--trust-proxy <ip>[,<ip>]` is an allowlist, not a boolean. The server
  honours `X-Forwarded-For` (login rate limiting) and
  `X-Forwarded-Proto` only when the direct TCP peer is one of these
  addresses; the same headers from anyone else are ignored, so a user
  cannot spoof a proxy. List Marvin's LAN IP — and nothing else.
- Without `--accounts` the server is unchanged: one shared run, and any
  non-loopback `--bind` is refused.

## First admin

```sh
server admin add --accounts /var/lib/neuroedit/state \
                 --email nico@example.com --name "Nico"
# reads the password from stdin — it never lands in argv, ps, or history
```

`admin list`, `admin activate <email>`, `admin disable <email>` take the
same `--accounts` flag for ops from cron/ssh. Day-to-day activation is
meant to happen in `/admin` instead.

## systemd unit (on miguel)

`/etc/systemd/system/neuroedit.service`:

```ini
[Unit]
Description=neuro_edit multi-user server
After=network-online.target

[Service]
User=neuroedit
ExecStart=/opt/neuroedit/server \
    --bind 10.0.0.20:9337 \
    --accounts /var/lib/neuroedit/state \
    --data-dir /var/lib/neuroedit/data \
    --web /opt/neuroedit/dist-web \
    --public-url https://neuroedit.ad4m.dev \
    --trust-proxy 10.0.0.10
Restart=on-failure
RestartSec=3
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/neuroedit

[Install]
WantedBy=multi-user.target
```

## nginx (on Marvin — the proxy box)

The server block lives on the box that terminates TLS and points at
miguel's LAN address:

```nginx
server {
    listen 443 ssl;
    server_name neuroedit.ad4m.dev;

    ssl_certificate     /etc/letsencrypt/live/neuroedit.ad4m.dev/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/neuroedit.ad4m.dev/privkey.pem;

    # uploads are chunked protocol frames, but /signup etc. are normal
    # POSTs — keep this generous anyway (matches --max-upload-mb + slack)
    client_max_body_size 80m;

    location / {
        proxy_pass http://10.0.0.20:9337;     # miguel
        proxy_http_version 1.1;

        # WebSocket upgrade (/ws keeps an open conn per editor)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 1h;
        proxy_send_timeout 1h;

        # the app reads these ONLY because miguel's --trust-proxy
        # allowlists this box
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Host $host;
    }
}
```

Notes:

- `/ws` upgrades like any other path; without the `Upgrade`/`Connection`
  headers the editor will connect, get a 400, and stay on "connecting".
- The long timeouts keep an idle editor's socket alive; the per-user run
  still stops on `--idle-minutes`, the socket itself is cheap.
- miguel's firewall should allow 9337 **only** from Marvin. If a user
  could reach miguel directly their `X-Forwarded-For` would be ignored
  anyway (not allowlisted) — but defence in depth is cheap here.
- Plain HTTP between Marvin and miguel is acceptable only because the
  LAN hop is trusted; nothing secret crosses it except session tokens —
  keep the network segment clean.

## Backups

Two things are irreplaceable:

- `<state-dir>/accounts.db` — users, sessions, signup notes. SQLite in
  WAL mode; snapshot it safely with
  `sqlite3 accounts.db ".backup accounts-$(date +%F).db"` instead of
  copying the file while the server runs.
- `<data-dir>/users/` — every user's files. A nightly
  `rsync -a /var/lib/neuroedit/data/users/ backup:/srv/neuroedit/users/`
  is enough; `<data-dir>/deleted/` is retention, not live data.

Everything else — the wasm bundle, the binary, the unit file — is
reproducible from the repo.

## Checklist

1. `cargo build --release -p server`; `editor/web/build.sh` →
   `dist-web` next to the binary.
2. systemd unit installed; `systemctl start neuroedit`.
3. `server admin add --accounts …` — create the first admin.
4. nginx server block on Marvin with the five `proxy_set_header` lines.
5. miguel firewall: 9337 reachable only from Marvin's IP.
6. Browse `https://neuroedit.ad4m.dev/` → landing page; sign up;
   activate the user in `/admin`; log in; `/app` loads the editor.
