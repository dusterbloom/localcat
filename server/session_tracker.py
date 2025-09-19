"""Simple persistent tracker for session statistics."""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any


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

    def start_session(self, user_id: str) -> Dict[str, Any]:
        now = time.time()
        entry = self._data.setdefault(
            user_id,
            {
                "total_sessions": 0,
                "total_turns": 0,
                "total_time_seconds": 0.0,
            },
        )
        entry.setdefault("total_turns", 0)
        entry.setdefault("total_time_seconds", 0.0)
        entry["total_sessions"] = int(entry.get("total_sessions", 0)) + 1
        entry["current_session"] = entry["total_sessions"]
        entry["session_start_epoch"] = now
        entry["session_start_iso"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        entry["session_turns"] = 0
        entry["session_elapsed"] = 0.0
        self._save()
        return dict(entry)

    def record_turn(self, user_id: str, turn_duration_sec: float) -> Dict[str, Any]:
        entry = self._data.setdefault(
            user_id,
            {
                "total_sessions": 1,
                "current_session": 1,
                "total_turns": 0,
                "total_time_seconds": 0.0,
                "session_start_epoch": time.time(),
                "session_start_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                "session_turns": 0,
            },
        )
        entry["session_turns"] = int(entry.get("session_turns", 0)) + 1
        entry["total_turns"] = int(entry.get("total_turns", 0)) + 1
        entry["total_time_seconds"] = float(entry.get("total_time_seconds", 0.0)) + max(turn_duration_sec, 0.0)
        start_epoch = float(entry.get("session_start_epoch", time.time()))
        entry["session_elapsed"] = max(time.time() - start_epoch, 0.0)
        self._save()
        return dict(entry)

    def end_session(self, user_id: str) -> Dict[str, Any]:
        entry = self._data.get(user_id)
        if not entry:
            return {}
        start_epoch = float(entry.get("session_start_epoch", time.time()))
        entry["session_elapsed"] = max(time.time() - start_epoch, 0.0)
        self._save()
        return dict(entry)

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        entry = self._data.get(user_id, {})
        return dict(entry)
