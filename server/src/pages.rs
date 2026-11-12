//! Server-rendered pages for multi-user mode: landing (login/signup),
//! `/admin` back office. One dark stylesheet, the Stage 10 artwork and
//! cyan/amber accents, no framework — small forms, no SPA.

use crate::accounts::User;
use crate::runs::Runs;
use std::sync::Arc;

pub const ABOUT_PNG: &[u8] = include_bytes!("../assets/about.png");
pub const FAVICON_PNG: &[u8] = include_bytes!("../assets/favicon.png");
pub const SHOT_DARK: &[u8] = include_bytes!("../assets/stage11_dark_theme.png");
pub const SHOT_MOVE: &[u8] = include_bytes!("../assets/stage11_move_preview.png");
pub const SHOT_CONNECT: &[u8] = include_bytes!("../assets/stage11_connect_input.png");

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

const CSS: &str = r#"
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; background:#111318; color:#d7dbe0;
       font:15px/1.5 system-ui, "Segoe UI", sans-serif; }
a { color:#3ec6d8; }
.wrap { max-width:880px; margin:0 auto; padding:24px 16px 64px; }
header { display:flex; align-items:center; gap:12px; }
header img.logo { width:40px; height:40px; }
h1 { font-size:22px; margin:0; color:#f0f2f5; }
h1 .amber { color:#ffb43a; }
h2 { font-size:16px; color:#f0f2f5; margin:28px 0 8px; }
.hero { width:100%; border-radius:8px; margin:20px 0;
        border:1px solid #262a33; background:#0b0d10; }
.lede { color:#aab0ba; max-width:60ch; }
.card { background:#1a1d24; border:1px solid #262a33; border-radius:8px;
        padding:16px; margin:12px 0; }
.cols { display:flex; gap:16px; flex-wrap:wrap; }
.cols > .card { flex:1 1 300px; }
label { display:block; font-size:13px; color:#9aa1ac; margin:10px 0 3px; }
input, button, textarea { font:inherit; }
input[type=email], input[type=password], input[type=text],
input[type=number], textarea {
  width:100%; padding:8px 10px; border-radius:6px; border:1px solid #333945;
  background:#12151b; color:#e6e9ee; }
button, .btn { display:inline-block; margin-top:12px; padding:8px 16px;
  border-radius:6px; border:1px solid #2b8f9d; background:#173741;
  color:#7fe0ef; cursor:pointer; text-decoration:none; }
button:hover, .btn:hover { background:#1d4550; }
.btn.amber { border-color:#a97a26; background:#3d2f13; color:#ffce7a; }
.btn.amber:hover { background:#4a3a18; }
.msg { padding:10px 12px; border-radius:6px; margin:12px 0; }
.msg.err { background:#3d1f24; border:1px solid #7c3a45; color:#f0a8b0; }
.msg.ok { background:#1c3527; border:1px solid #2f7d52; color:#9fe0b8; }
.msg.info { background:#22303d; border:1px solid #3a607c; color:#a8ccec; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid #262a33;
         vertical-align:top; }
th { color:#9aa1ac; font-weight:600; }
.badge { display:inline-block; padding:1px 8px; border-radius:10px;
         font-size:12px; }
.badge.pending { background:#3d2f13; color:#ffce7a; }
.badge.active { background:#1c3527; color:#9fe0b8; }
.badge.disabled { background:#3d1f24; color:#f0a8b0; }
.shots { display:flex; gap:12px; flex-wrap:wrap; margin-top:8px; }
.shots img { flex:1 1 240px; max-width:100%; border-radius:6px;
             border:1px solid #262a33; }
.small { font-size:13px; color:#9aa1ac; }
form.inline { display:inline; }
form.inline button { margin:2px 4px 2px 0; padding:3px 10px; font-size:13px; }
.usage { height:8px; background:#262a33; border-radius:4px; overflow:hidden;
         margin:6px 0 2px; }
.usage > div { height:100%; background:#2b8f9d; }
ul.feat { padding-left:18px; } ul.feat li { margin:4px 0; }
"#;

fn page(title: &str, body: &str) -> String {
    format!(
        "<!doctype html><html lang=en><head><meta charset=utf-8>\
         <meta name=viewport content=\"width=device-width, initial-scale=1\">\
         <title>{title} · neuro_edit</title>\
         <link rel=icon href=\"/assets/favicon.png\" type=\"image/png\">\
         <style>{CSS}</style></head><body><div class=wrap>{body}</div></body></html>"
    )
}

fn header() -> String {
    "<header><img class=logo src=\"/assets/favicon.png\" alt=\"\">\
     <h1>neuro<span class=amber>_edit</span></h1></header>"
        .to_string()
}

fn fmt_mb(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.1} GB", b as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.1} MB", b as f64 / (1u64 << 20) as f64)
    }
}

/// `GET /` — landing. Logged out: hero, blurb, features, shots,
/// login+signup cards. Logged in (active): open-editor + usage + logout.
/// `msg`/`err` carry login/signup outcomes back to the form.
pub fn landing(
    user: Option<&User>,
    session_csrf: Option<&str>,
    used_bytes: u64,
    msg: Option<&str>,
    err: Option<&str>,
    full: bool,
    app_url: &str,
) -> String {
    let mut b = header();
    b.push_str("<img class=hero src=\"/assets/about.png\" alt=\"neuro_edit network view\">");
    b.push_str(
        "<p class=lede>neuro_edit is an editor and simulator for large \
         spiking-neuron networks. You draw networks of neurons on a 3D \
         canvas, drive them with inputs, and watch them run — locally on \
         your machine, or here on this server.</p>",
    );
    if let Some(m) = msg {
        b.push_str(&format!("<div class=\"msg ok\">{}</div>", esc(m)));
    }
    if let Some(e) = err {
        b.push_str(&format!("<div class=\"msg err\">{}</div>", esc(e)));
    }
    if full {
        b.push_str("<div class=\"msg info\">the server is full, try again in a few minutes</div>");
    }

    match (user, session_csrf) {
        (Some(u), Some(csrf)) => {
            let pct = if u.quota_bytes > 0 {
                (used_bytes * 100 / u.quota_bytes.max(1)).min(100)
            } else {
                0
            };
            b.push_str(&format!(
                "<div class=card><b>{name}</b> <span class=small>({mail})</span>\
                 <div class=usage><div style=\"width:{pct}%\"></div></div>\
                 <div class=small>{used} of {quota} used</div>\
                 <a class=\"btn amber\" href=\"{app}\">Open editor</a>\
                 <form class=inline method=post action=\"/logout\" style=\"float:right\">\
                 <input type=hidden name=csrf value=\"{csrf}\">\
                 <button>Log out</button></form></div>",
                name = esc(&u.name),
                mail = esc(&u.email),
                used = fmt_mb(used_bytes),
                quota = fmt_mb(u.quota_bytes),
                app = app_url,
                csrf = esc(csrf),
                pct = pct,
            ));
        }
        _ => {
            b.push_str(
                "<div class=cols>\
                 <div class=card><h2>Log in</h2>\
                 <form method=post action=\"/login\">\
                 <label>Email</label><input type=email name=email required autocomplete=email>\
                 <label>Password</label><input type=password name=password required autocomplete=current-password>\
                 <button>Log in</button></form></div>\
                 <div class=card><h2>Sign up</h2>\
                 <form method=post action=\"/signup\">\
                 <label>Email</label><input type=email name=email required autocomplete=email>\
                 <label>Display name</label><input type=text name=name required>\
                 <label>Password (12+ characters)</label><input type=password name=password required minlength=12 autocomplete=new-password>\
                 <label>What do you want to use it for? <span class=small>(optional)</span></label>\
                 <textarea name=note rows=2></textarea>\
                 <button>Sign up</button>\
                 <p class=small>New accounts stay pending until an admin activates them.</p></form></div>\
                 </div>",
            );
        }
    }

    b.push_str(
        "<h2>What it does</h2><ul class=feat>\
         <li>Edit spiking networks: neurons, synapses, input sources, 2D/3D sheets.</li>\
         <li>Simulate on CPU or GPU (CUDA) and watch activity live in the editor.</li>\
         <li>Rhai scripts drive experiments from inside the app.</li>\
         <li>Datasets + liquid-state training/readout run as background jobs.</li>\
         <li>The plane view shows large sheets at once.</li>\
         </ul>\
         <div class=shots>\
         <img src=\"/assets/shot_dark.png\" alt=\"editor, dark theme\">\
         <img src=\"/assets/shot_move.png\" alt=\"moving neurons with live preview\">\
         <img src=\"/assets/shot_connect.png\" alt=\"connecting an input to a neuron\">\
         </div>\
         <p class=small>The in-browser editor needs WebGPU — \
         <a href=\"https://webgpureport.org\">webgpureport.org</a> checks your browser, \
         and the app shows setup steps if it is missing.</p>",
    );
    page("welcome", &b)
}

/// `/` after POST /login with a known-state account — same page, one
/// info line. `text` is the spec'd pending/disabled/429 sentence.
pub fn landing_with(text: &str, kind: &str) -> String {
    landing(None, None, 0, None, None, false, "/app").replace(
        "<div class=cols>",
        &format!(
            "<div class=\"msg {kind}\">{}</div><div class=cols>",
            esc(text)
        ),
    )
}

/// `GET /admin` — the back office user table (admins only; caller 404s
/// otherwise). Pending users float to the top with a count.
pub fn admin(users: &[User], live_runs: &[i64], csrf: &str, flash: Option<&str>) -> String {
    let pending = users.iter().filter(|u| u.status == "pending").count();
    let mut b = header();
    b.push_str(&format!(
        "<div class=card><b>Back office</b> — {pending} pending activation</div>"
    ));
    if let Some(f) = flash {
        b.push_str(&format!("<div class=\"msg ok\">{}</div>", esc(f)));
    }
    b.push_str(
        "<table><tr><th>User</th><th>Status</th><th>Note</th><th>Created</th>\
         <th>Last login</th><th>Storage</th><th>Run</th><th></th></tr>",
    );
    for u in users {
        let created = fmt_date(u.created_ms);
        let login = if u.last_login_ms == 0 {
            "—".into()
        } else {
            fmt_date(u.last_login_ms)
        };
        let quota_mb = u.quota_bytes / (1 << 20);
        let run = if live_runs.contains(&u.id) {
            "live"
        } else {
            "—"
        };
        b.push_str(&format!(
            "<tr><td><b>{}</b><br><span class=small>{}</span></td>\
             <td><span class=\"badge {}\">{}{}</span></td>\
             <td class=small>{}</td><td class=small>{created}</td><td class=small>{login}</td>\
             <td class=small>quota {} MB</td><td class=small>{run}</td><td>",
            esc(&u.name),
            esc(&u.email),
            u.status,
            u.status,
            if u.is_admin { " · admin" } else { "" },
            esc(&u.note),
            quota_mb,
        ));
        // Actions — each its own small form with the session CSRF.
        let form = |action: &str, extra: &str, label: &str| {
            format!(
                "<form class=inline method=post action=\"/admin/{action}\">\
                 <input type=hidden name=id value={id}><input type=hidden name=csrf value=\"{c}\">\
                 {extra}<button>{label}</button></form>",
                id = u.id,
                c = esc(csrf),
            )
        };
        match u.status.as_str() {
            "pending" | "disabled" => b.push_str(&form("activate", "", "activate")),
            _ => b.push_str(&form("disable", "", "disable")),
        }
        b.push_str(&form(
            "quota",
            "<input type=number name=mb min=0 style=\"width:90px;display:inline\" placeholder=\"MB\">",
            "set quota",
        ));
        if u.is_admin {
            b.push_str(&form("revoke-admin", "", "revoke admin"));
        } else {
            b.push_str(&form("grant-admin", "", "make admin"));
        }
        b.push_str(&form(
            "delete",
            &format!(
                "<input type=text name=confirm placeholder=\"type {}\" \
                 style=\"width:120px;display:inline\">",
                esc(&u.email)
            ),
            "delete",
        ));
        b.push_str("</td></tr>");
    }
    b.push_str("</table><p class=small><a href=\"/\">← landing</a></p>");
    page("admin", &b)
}

fn fmt_date(ms: u64) -> String {
    // Day-precision UTC without a chrono dep: civil-from-days.
    let days = (ms / 86_400_000) as i64;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Pending-user count line for the landing header (`full` line lives
/// in `landing` itself). Kept tiny on purpose.
pub fn _runs_full(runs: &Arc<Runs>, max: usize) -> bool {
    runs.count() >= max
}
