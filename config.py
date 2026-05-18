import os
import tempfile
import threading
import time

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL   = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

SESSIONS      = {}
_session_lock = threading.Lock()
SESSION_TTL   = 3600

def session_plot_path(session_id: str) -> str:
    """Unique per-session plot file in the OS temp directory."""
    return os.path.join(tempfile.gettempdir(), f"plot_{session_id}.png")

def cleanup_session(session_id: str):
    """
    Delete the session's temp CSV and plot file, then remove it from SESSIONS.
    Call with _session_lock held, or from a single-threaded context.
    """
    session = SESSIONS.pop(session_id, None)
    if not session:
        return
    for key in ("dataset_path", "plot_path"):
        path = session.get(key)
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass

def evict_old_sessions():
    """Remove sessions older than SESSION_TTL. Call at the start of each new analysis."""
    now = time.time()
    with _session_lock:
        stale = [
            sid for sid, s in SESSIONS.items()
            if now - s.get("created_at", now) > SESSION_TTL
        ]
        for sid in stale:
            cleanup_session(sid)
