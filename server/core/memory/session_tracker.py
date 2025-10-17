"""Simple persistent tracker for session statistics (JSON-backed)."""

import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SessionTracker:
    def __init__(self, *, storage_path: str | None = None):
        default_path = Path(os.getenv("SESSION_STATS_PATH", "data/session_stats.json"))
        self._path = Path(storage_path) if storage_path else default_path
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except Exception:
                self._data = {}

    def _save(self) -> None:
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._data, indent=2))
        tmp_path.replace(self._path)

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        now = time.time()
        user_entry = self._data.setdefault(
            user_id,
            {
                "total_sessions": 0,
                "total_turns": 0,
                "total_time_seconds": 0.0,
                "sessions": {},  # Track individual sessions
            },
        )
        user_entry.setdefault("total_turns", 0)
        user_entry.setdefault("total_time_seconds", 0.0)
        user_entry.setdefault("sessions", {})

        # Use provided session_id or generate one
        if session_id is None:
            session_id = f"{user_id}_{int(now)}_{os.urandom(4).hex()[:8]}"

        # Initialize session data
        session_data = {
            "session_id": session_id,
            "session_start_epoch": now,
            "session_start_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "session_turns": 0,
            "session_elapsed": 0.0,
        }
        user_entry["sessions"][session_id] = session_data

        # Update user totals
        user_entry["total_sessions"] = len(user_entry["sessions"])
        user_entry["current_session_id"] = session_id

        self._save()
        return dict(session_data)

    def record_turn(self, user_id: str, session_id: Optional[str], turn_duration_sec: float) -> Dict[str, Any]:
        user_entry = self._data.get(user_id)
        if not user_entry:
            # Create user entry if it doesn't exist
            user_entry = self._data[user_id] = {
                "total_sessions": 0,
                "total_turns": 0,
                "total_time_seconds": 0.0,
                "sessions": {},
            }

        # Use current session if none provided
        if session_id is None:
            session_id = user_entry.get("current_session_id")
            if not session_id:
                # Generate a session ID if none exists
                session_id = f"{user_id}_{int(time.time())}_{os.urandom(4).hex()[:8]}"

        # Get or create session entry
        session_data = user_entry["sessions"].setdefault(session_id, {
            "session_id": session_id,
            "session_start_epoch": time.time(),
            "session_start_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_turns": 0,
            "session_elapsed": 0.0,
        })

        # Update session stats
        session_data["session_turns"] = int(session_data.get("session_turns", 0)) + 1
        start_epoch = float(session_data.get("session_start_epoch", time.time()))
        session_data["session_elapsed"] = max(time.time() - start_epoch, 0.0)

        # Update user totals
        user_entry["total_turns"] = int(user_entry.get("total_turns", 0)) + 1
        user_entry["total_time_seconds"] = float(user_entry.get("total_time_seconds", 0.0)) + max(turn_duration_sec, 0.0)
        user_entry["current_session_id"] = session_id
        # Ensure total_sessions reflects current session count
        try:
            user_entry["total_sessions"] = len(user_entry.get("sessions", {}))
        except Exception:
            user_entry["total_sessions"] = user_entry.get("total_sessions", 1)

        self._save()
        # Return a merged view including user totals so headers can show counts
        merged = dict(session_data)
        merged.update({
            "total_sessions": int(user_entry.get("total_sessions", 1)),
            "total_turns": int(user_entry.get("total_turns", 0)),
            "total_time_seconds": float(user_entry.get("total_time_seconds", 0.0)),
            "current_session": int(user_entry.get("current_session_index", len(user_entry.get("sessions", {}))))
        })
        return merged

    def end_session(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        user_entry = self._data.get(user_id)
        if not user_entry:
            return {}

        # Use current session if none specified
        if session_id is None:
            session_id = user_entry.get("current_session_id")

        if session_id and session_id in user_entry.get("sessions", {}):
            session_data = user_entry["sessions"][session_id]
            start_epoch = float(session_data.get("session_start_epoch", time.time()))
            session_data["session_elapsed"] = max(time.time() - start_epoch, 0.0)
            self._save()
            return dict(session_data)

        return {}

    def get_stats(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        user_entry = self._data.get(user_id, {})
        if not user_entry:
            return {}

        if session_id and session_id in user_entry.get("sessions", {}):
            # Return specific session stats
            return dict(user_entry["sessions"][session_id])
        else:
            # Return user-level stats
            return dict(user_entry)

    def get_user_identity(self, user_id: str) -> Optional[str]:
        """Get the current identity for a user."""
        user_entry = self._data.get(user_id, {})
        return user_entry.get("current_identity")

    def set_user_identity(self, user_id: str, identity: str) -> None:
        """Set the current identity for a user."""
        if user_id not in self._data:
            self._data[user_id] = {"total_sessions": 0, "sessions": {}}
        
        # Update current identity
        old_identity = self._data[user_id].get("current_identity")
        if old_identity != identity:
            self._data[user_id]["current_identity"] = identity
            self._data[user_id]["identity_history"] = self._data[user_id].get("identity_history", [])
            self._data[user_id]["identity_history"].append({
                "identity": identity,
                "timestamp": time.time(),
                "previous": old_identity
            })
            self._save()
            try:
                logger.debug(f"[SessionTracker] User identity updated: {user_id}: {old_identity} → {identity}")
            except Exception:
                pass

    def get_user_identity(self, user_id: str) -> Optional[str]:
        """Get the current identity for a user."""
        user_entry = self._data.get(user_id, {})
        return user_entry.get("current_identity")
