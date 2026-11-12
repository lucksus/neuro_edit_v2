//! Multi-user account store: SQLite at `<state-dir>/accounts.db`
//! (rusqlite bundled — no system sqlite dependency).
//!
//! Users carry a `pending | active | disabled` status — signups land
//! `pending` until an admin activates them from the back office or
//! `server admin activate`. Passwords are argon2id hashes; sessions are
//! 256-bit tokens stored only as SHA-256 so the DB alone leaks neither.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use rand::rngs::OsRng;
use rand::RngCore;
use rusqlite::{params, Connection};
use sha2::{Digest, Sha256};

/// Browser cookie name carrying the session token.
pub const SESSION_COOKIE: &str = "ne_session";
/// Session lifetime: 30 days.
pub const SESSION_TTL_MS: u64 = 30 * 24 * 3600 * 1000;
/// Minimum password length (signup + admin add enforce it).
pub const MIN_PASSWORD_LEN: usize = 12;

#[derive(Debug, Clone, PartialEq)]
pub struct User {
    pub id: i64,
    pub email: String,
    pub name: String,
    /// "pending" | "active" | "disabled"
    pub status: String,
    pub is_admin: bool,
    pub created_ms: u64,
    pub last_login_ms: u64,
    pub quota_bytes: u64,
    pub note: String,
}

/// What `verify_login` found — the login page maps Pending/Disabled to
/// their spec'd texts and keeps Bad identical for wrong email and wrong
/// password (no account enumeration).
#[derive(Debug, PartialEq)]
pub enum Login {
    Ok(User),
    Pending,
    Disabled,
    Bad,
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn token256() -> String {
    let mut b = [0u8; 32];
    OsRng.fill_bytes(&mut b);
    hex(&b)
}

fn sha256_hex(s: &str) -> String {
    hex(&Sha256::digest(s.as_bytes()))
}

fn hash_password(pw: &str) -> anyhow::Result<String> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(pw.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| anyhow::anyhow!("hash: {e}"))
}

/// Verify `pw` against an argon2 PHC string.
fn verify_password(pw: &str, phc: &str) -> bool {
    match PasswordHash::new(phc) {
        Ok(h) => Argon2::default().verify_password(pw.as_bytes(), &h).is_ok(),
        Err(_) => false,
    }
}

/// Rate limiting for login: 10 failures per key per 10 minutes.
/// Keys are `ip:<addr>` and `email:<addr>` buckets. In-memory — a
/// server restart resets the counters (documented in deploy.md).
pub struct RateLimiter {
    fails: Mutex<HashMap<String, Vec<Instant>>>,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            fails: Mutex::new(HashMap::new()),
        }
    }
    /// True once `key` already has `limit` failures inside the window.
    /// Lookup does not insert — a flood of unique keys can't grow the
    /// map; a periodic sweep clears the ones that do get written.
    pub fn limited_n(&self, key: &str, limit: usize) -> bool {
        let mut m = self.fails.lock().unwrap();
        let now = Instant::now();
        if m.len() > 50_000 {
            m.retain(|_, v| {
                v.iter()
                    .any(|t| now.duration_since(*t) < Duration::from_secs(600))
            });
        }
        m.get(key)
            .map(|v| {
                v.iter()
                    .filter(|t| now.duration_since(**t) < Duration::from_secs(600))
                    .count()
            })
            .map(|n| n >= limit)
            .unwrap_or(false)
    }
    /// The login default: 10 failures per key per 10 minutes.
    pub fn limited(&self, key: &str) -> bool {
        self.limited_n(key, 10)
    }
    pub fn record_fail(&self, key: &str) {
        let mut m = self.fails.lock().unwrap();
        m.entry(key.to_string()).or_default().push(Instant::now());
    }
    /// Successful login clears the email bucket.
    pub fn clear(&self, key: &str) {
        self.fails.lock().unwrap().remove(key);
    }
}

pub struct Accounts {
    conn: Mutex<Connection>,
    pub state_dir: PathBuf,
    pub limiter: RateLimiter,
}

impl Accounts {
    pub fn open(state_dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(state_dir)?;
        let conn = Connection::open(state_dir.join("accounts.db"))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS users(
                 id INTEGER PRIMARY KEY,
                 email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                 name TEXT NOT NULL,
                 pass_hash TEXT NOT NULL,
                 status TEXT NOT NULL DEFAULT 'pending',
                 is_admin INTEGER NOT NULL DEFAULT 0,
                 created_ms INTEGER NOT NULL,
                 last_login_ms INTEGER NOT NULL DEFAULT 0,
                 quota_bytes INTEGER NOT NULL DEFAULT 0,
                 note TEXT NOT NULL DEFAULT '');
             CREATE TABLE IF NOT EXISTS sessions(
                 hash TEXT PRIMARY KEY,
                 user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                 csrf TEXT NOT NULL,
                 created_ms INTEGER NOT NULL,
                 expires_ms INTEGER NOT NULL);
             CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
            state_dir: state_dir.to_path_buf(),
            limiter: RateLimiter::new(),
        })
    }

    fn row_to_user(r: &rusqlite::Row) -> rusqlite::Result<User> {
        Ok(User {
            id: r.get(0)?,
            email: r.get(1)?,
            name: r.get(2)?,
            status: r.get(4)?,
            is_admin: r.get::<_, i64>(5)? != 0,
            created_ms: r.get::<_, i64>(6)? as u64,
            last_login_ms: r.get::<_, i64>(7)? as u64,
            quota_bytes: r.get::<_, i64>(8)? as u64,
            note: r.get(9)?,
        })
    }

    const USER_COLS: &'static str =
        "id, email, name, pass_hash, status, is_admin, created_ms, last_login_ms, quota_bytes, note";

    /// Signup path: email lowercased+trimmed, unique; status `pending`.
    /// Returns Err("…") text safe to show on the form.
    pub fn create_user(
        &self,
        email: &str,
        name: &str,
        password: &str,
        note: &str,
        quota_bytes: u64,
    ) -> Result<User, String> {
        let email = email.trim().to_lowercase();
        if !email.contains('@') || email.len() < 3 {
            return Err("that does not look like an email address".into());
        }
        let name = name.trim();
        if name.is_empty() {
            return Err("display name is required".into());
        }
        if name.chars().count() > 80 {
            return Err("display name is too long (80 characters max)".into());
        }
        if note.chars().count() > 500 {
            return Err("note is too long (500 characters max)".into());
        }
        if email.len() > 254 {
            return Err("email is too long".into());
        }
        if password.len() < MIN_PASSWORD_LEN {
            return Err(format!(
                "password needs at least {MIN_PASSWORD_LEN} characters"
            ));
        }
        let hash = hash_password(password).map_err(|e| e.to_string())?;
        let conn = self.conn.lock().unwrap();
        let r = conn.execute(
            "INSERT INTO users(email,name,pass_hash,status,is_admin,created_ms,quota_bytes,note)
             VALUES(?1,?2,?3,'pending',0,?4,?5,?6)",
            params![email, name, hash, now_ms() as i64, quota_bytes as i64, note],
        );
        match r {
            Ok(_) => self
                .by_email_locked(&conn, &email)
                .map_err(|e| e.to_string()),
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err("an account with that email already exists".into())
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Operator path (`server admin add`): creates an already-active
    /// admin. On an existing email it *replaces* the account: new
    /// password and name, admin+active forced, all prior sessions
    /// deleted — re-running `admin add` is the password-reset path.
    /// Returns `(user, replaced)`.
    pub fn create_admin(
        &self,
        email: &str,
        name: &str,
        password: &str,
    ) -> anyhow::Result<(User, bool)> {
        let email = email.trim().to_lowercase();
        if password.len() < MIN_PASSWORD_LEN {
            anyhow::bail!("password needs at least {MIN_PASSWORD_LEN} characters");
        }
        let hash = hash_password(password)?;
        let conn = self.conn.lock().unwrap();
        let replaced = self.by_email_locked(&conn, &email).is_ok();
        conn.execute(
            "INSERT INTO users(email,name,pass_hash,status,is_admin,created_ms,quota_bytes,note)
             VALUES(?1,?2,?3,'active',1,?4,0,'')
             ON CONFLICT(email) DO UPDATE SET
                 pass_hash=excluded.pass_hash,
                 name=excluded.name,
                 status='active',
                 is_admin=1",
            params![email, name, hash, now_ms() as i64],
        )?;
        let u = self.by_email_locked(&conn, &email)?;
        // Whatever sessions the old account held die with its password.
        conn.execute("DELETE FROM sessions WHERE user_id=?1", params![u.id])?;
        Ok((u, replaced))
    }

    fn by_email_locked(&self, conn: &Connection, email: &str) -> rusqlite::Result<User> {
        conn.query_row(
            &format!("SELECT {} FROM users WHERE email=?1", Self::USER_COLS),
            params![email],
            Self::row_to_user,
        )
    }

    pub fn by_email(&self, email: &str) -> Option<User> {
        let conn = self.conn.lock().unwrap();
        self.by_email_locked(&conn, &email.trim().to_lowercase())
            .ok()
    }

    pub fn by_id(&self, id: i64) -> Option<User> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            &format!("SELECT {} FROM users WHERE id=?1", Self::USER_COLS),
            params![id],
            Self::row_to_user,
        )
        .ok()
    }

    /// Login check. `Pending`/`Disabled` let the form show the spec'd
    /// texts; `Bad` is identical for unknown email and wrong password —
    /// and the argon2 verify runs either way so timing doesn't leak
    /// which emails exist.
    pub fn verify_login(&self, email: &str, password: &str) -> Login {
        let email = email.trim().to_lowercase();
        // Read the row under the lock, verify outside it — argon2 takes
        // tens of ms and every other request needs this same lock.
        let row: Option<(String, User)> = {
            let conn = self.conn.lock().unwrap();
            conn.query_row(
                &format!(
                    "SELECT {}, pass_hash FROM users WHERE email=?1",
                    Self::USER_COLS
                ),
                params![email],
                |r| {
                    // pass_hash is column 3 in USER_COLS — row_to_user reads it
                    // positionally; select it twice so both shapes work.
                    Ok((r.get::<_, String>(3)?, Self::row_to_user(r)?))
                },
            )
            .ok()
        };
        match row {
            Some((phc, u)) => {
                if !verify_password(password, &phc) {
                    return Login::Bad;
                }
                match u.status.as_str() {
                    "active" => Login::Ok(u),
                    "pending" => Login::Pending,
                    _ => Login::Disabled,
                }
            }
            None => {
                // Dummy verify to match the taken path's cost.
                verify_password(
                    password,
                    "$argon2id$v=19$m=19456,t=2,p=1$AAAAAAAAAAA$0000000000000000000000000000000000000000000",
                );
                Login::Bad
            }
        }
    }

    pub fn touch_login(&self, user_id: i64) {
        let _ = self.conn.lock().unwrap().execute(
            "UPDATE users SET last_login_ms=?1 WHERE id=?2",
            params![now_ms() as i64, user_id],
        );
    }

    /// Pending first, then alphabetical — the back office's sort order.
    pub fn list(&self) -> Vec<User> {
        let conn = self.conn.lock().unwrap();
        let mut st = conn
            .prepare(&format!(
                "SELECT {} FROM users ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, email",
                Self::USER_COLS
            ))
            .unwrap();
        st.query_map([], Self::row_to_user)
            .unwrap()
            .flatten()
            .collect()
    }

    pub fn set_status(&self, user_id: i64, status: &str) -> rusqlite::Result<usize> {
        self.conn.lock().unwrap().execute(
            "UPDATE users SET status=?1 WHERE id=?2",
            params![status, user_id],
        )
    }

    pub fn set_quota(&self, user_id: i64, quota_bytes: u64) -> rusqlite::Result<usize> {
        self.conn.lock().unwrap().execute(
            "UPDATE users SET quota_bytes=?1 WHERE id=?2",
            params![quota_bytes as i64, user_id],
        )
    }

    pub fn set_admin(&self, user_id: i64, is_admin: bool) -> rusqlite::Result<usize> {
        self.conn.lock().unwrap().execute(
            "UPDATE users SET is_admin=?1 WHERE id=?2",
            params![is_admin as i64, user_id],
        )
    }

    pub fn delete_user(&self, user_id: i64) -> rusqlite::Result<usize> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM sessions WHERE user_id=?1", params![user_id])?;
        conn.execute("DELETE FROM users WHERE id=?1", params![user_id])
    }

    /// New session: returns the raw token (cookie value) and this
    /// session's CSRF token. Only the SHA-256 is stored.
    pub fn create_session(&self, user_id: i64) -> anyhow::Result<(String, String)> {
        let token = token256();
        let csrf = token256();
        self.conn.lock().unwrap().execute(
            "INSERT INTO sessions(hash,user_id,csrf,created_ms,expires_ms) VALUES(?1,?2,?3,?4,?5)",
            params![
                sha256_hex(&token),
                user_id,
                csrf,
                now_ms() as i64,
                (now_ms() + SESSION_TTL_MS) as i64
            ],
        )?;
        Ok((token, csrf))
    }

    /// `(user_id, csrf)` for a still-valid session token.
    pub fn session(&self, token: &str) -> Option<(i64, String)> {
        if token.len() > 128 {
            return None;
        }
        self.conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT user_id, csrf FROM sessions WHERE hash=?1 AND expires_ms>?2",
                params![sha256_hex(token), now_ms() as i64],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .ok()
    }

    pub fn delete_session(&self, token: &str) {
        let _ = self.conn.lock().unwrap().execute(
            "DELETE FROM sessions WHERE hash=?1",
            params![sha256_hex(token)],
        );
    }

    /// Disable/delete path: every live session of the user dies.
    pub fn delete_user_sessions(&self, user_id: i64) {
        let _ = self
            .conn
            .lock()
            .unwrap()
            .execute("DELETE FROM sessions WHERE user_id=?1", params![user_id]);
    }

    /// Housekeeping: expired sessions are deleted lazily here — called
    /// on `open` and once in a while by the server.
    pub fn sweep_expired_sessions(&self) {
        let _ = self.conn.lock().unwrap().execute(
            "DELETE FROM sessions WHERE expires_ms<=?1",
            params![now_ms() as i64],
        );
    }

    /// Pending-account count (signup cap — S10).
    pub fn pending_count(&self) -> i64 {
        self.conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM users WHERE status='pending'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0)
    }

    /// Admins other than `except_id` — the last-admin guard (S18).
    pub fn has_other_admin(&self, except_id: i64) -> bool {
        self.conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM users WHERE is_admin=1 AND status='active' AND id!=?1",
                params![except_id],
                |r| r.get::<_, i64>(0),
            )
            .map(|n| n > 0)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn login_roundtrip() {
        let dir = std::env::temp_dir().join(format!("acct_{}", std::process::id()));
        let a = Accounts::open(&dir).unwrap();
        let u = a
            .create_user("a@x.dev", "Ann", "correct horse 12", "n", 1 << 20)
            .unwrap();
        a.set_status(u.id, "active").unwrap();
        match a.verify_login("a@x.dev", "correct horse 12") {
            Login::Ok(_) => {}
            other => panic!("{other:?}"),
        }
        assert_eq!(Login::Bad, a.verify_login("a@x.dev", "wrong wrong wrong"));
        assert_eq!(
            Login::Bad,
            a.verify_login("nobody@x.dev", "correct horse 12")
        );
        let (tok, _csrf) = a.create_session(u.id).unwrap();
        assert!(a.session(&tok).is_some());
        a.delete_session(&tok);
        assert!(a.session(&tok).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// S4: `admin add` on an existing email replaces the account —
    /// new password + name, old sessions dead, `replaced` reported.
    #[test]
    fn admin_add_replaces_existing_account() {
        let dir = std::env::temp_dir().join(format!("acct_s4_{}", std::process::id()));
        let a = Accounts::open(&dir).unwrap();
        let u = a
            .create_user("v@x.dev", "Old Name", "old password 123", "", 0)
            .unwrap();
        a.set_status(u.id, "active").unwrap();
        let (tok, _) = a.create_session(u.id).unwrap();
        assert!(a.session(&tok).is_some());

        let (u2, replaced) = a
            .create_admin("v@x.dev", "New Name", "new password 456")
            .unwrap();
        assert!(replaced);
        assert_eq!(u2.id, u.id);
        assert_eq!(u2.name, "New Name");
        assert!(u2.is_admin);
        // old password dead, new password works, old session gone
        assert_eq!(Login::Bad, a.verify_login("v@x.dev", "old password 123"));
        assert!(matches!(
            a.verify_login("v@x.dev", "new password 456"),
            Login::Ok(_)
        ));
        assert!(a.session(&tok).is_none());
        // a fresh email reports not-replaced
        let (_u3, replaced) = a.create_admin("n@x.dev", "N", "fresh password 1").unwrap();
        assert!(!replaced);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// S9: `limited`/`limited_n` don't insert keys on lookup — a flood
    /// of unique keys can't grow the map.
    #[test]
    fn rate_limiter_lookup_does_not_insert() {
        let lim = RateLimiter::new();
        for i in 0..10_000 {
            assert!(!lim.limited(&format!("k{i}")));
        }
        assert!(lim.fails.lock().unwrap().is_empty());
        // recorded failures still count
        for _ in 0..10 {
            lim.record_fail("victim");
        }
        assert!(lim.limited("victim"));
        assert!(lim.limited_n("victim2", 1) == false);
        lim.record_fail("victim2");
        assert!(lim.limited_n("victim2", 1));
    }
}
