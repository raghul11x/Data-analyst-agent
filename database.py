import sqlite3
import json
import os
import time
import base64
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "agentic.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id            TEXT PRIMARY KEY,
            dataset_name  TEXT,
            dataset_path  TEXT,
            question      TEXT,
            final_report  TEXT,
            created_at    REAL,
            updated_at    REAL
        );

        CREATE TABLE IF NOT EXISTS plots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT REFERENCES sessions(id) ON DELETE CASCADE,
            plot_b64    TEXT,
            created_at  REAL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT REFERENCES sessions(id) ON DELETE CASCADE,
            role        TEXT,
            content     TEXT,
            created_at  REAL
        );

        CREATE TABLE IF NOT EXISTS prep_sessions (
            id          TEXT PRIMARY KEY,
            filename    TEXT,
            csv_path    TEXT,
            log_json    TEXT DEFAULT '[]',
            code        TEXT DEFAULT '',
            created_at  REAL
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_plots_session    ON plots(session_id);
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        """)

def create_session(sid: str, dataset_name: str, dataset_path: str, question: str = ""):
    now = time.time()
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, dataset_name, dataset_path, question, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, dataset_name, dataset_path, question, now, now)
        )

def update_final_report(sid: str, report: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET final_report=?, updated_at=? WHERE id=?",
            (report, time.time(), sid)
        )

def add_plot(sid: str, plot_b64: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO plots (session_id, plot_b64, created_at) VALUES (?, ?, ?)",
            (sid, plot_b64, time.time())
        )

def add_message(sid: str, role: str, content: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (sid, role, content, time.time())
        )

def get_messages(sid: str) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at",
            (sid,)
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]

def get_plots(sid: str) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT plot_b64 FROM plots WHERE session_id=? ORDER BY created_at",
            (sid,)
        ).fetchall()
    return [r["plot_b64"] for r in rows]

def get_session(sid: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    if not row:
        return None
    return dict(row)

def get_full_session(sid: str) -> dict | None:
    session = get_session(sid)
    if not session:
        return None
    session["plots"]    = get_plots(sid)
    session["messages"] = get_messages(sid)
    return session

def list_sessions(limit: int = 10) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, dataset_name, created_at, final_report IS NOT NULL as has_report, "
            "(SELECT COUNT(*) FROM plots WHERE session_id=sessions.id) as plot_count "
            "FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

def delete_session(sid: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE id=?", (sid,))

def get_dataset_path(sid: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT dataset_path FROM sessions WHERE id=?", (sid,)
        ).fetchone()
    return row["dataset_path"] if row else ""

def create_prep_session(sid: str, filename: str, csv_path: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO prep_sessions (id, filename, csv_path, created_at) VALUES (?,?,?,?)",
            (sid, filename, csv_path, time.time())
        )

def update_prep_log(sid: str, log: list, code: str = ""):
    with get_conn() as conn:
        conn.execute(
            "UPDATE prep_sessions SET log_json=?, code=? WHERE id=?",
            (json.dumps(log), code, sid)
        )

def get_prep_session(sid: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM prep_sessions WHERE id=?", (sid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["log"] = json.loads(d.pop("log_json", "[]"))
    return d

def get_latest_prep_session() -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM prep_sessions WHERE log_json != '[]' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["log"] = json.loads(d.pop("log_json", "[]"))
    return d

def evict_old_sessions(ttl_seconds: int = 86400 * 7):
    """Delete sessions older than ttl_seconds (default 7 days)."""
    cutoff = time.time() - ttl_seconds
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE created_at < ?", (cutoff,))
        conn.execute("DELETE FROM prep_sessions WHERE created_at < ?", (cutoff,))
