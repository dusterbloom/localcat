"""Database-backed session tracker for full session persistence."""

import os
import sqlite3
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseSessionTracker:
    """Session tracker that uses SQLite database for persistence.

    Compatible with the original localcat sessions.db schema.
    """

    def __init__(self, *, db_path: str | None = None):
        # Single source of truth: SESSION_DB_PATH (relative to server/)
        env_path = os.getenv("SESSION_DB_PATH")
        default_path = Path(env_path) if env_path else Path("data/sessions.db")
        self._db_path = Path(db_path) if db_path else default_path
        logger.info(f"Session DB path resolved to: {self._db_path}")

        # Ensure directory exists
        if not self._db_path.parent.exists():
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema if not exists."""
        with self._get_connection() as conn:
            # Create sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time INTEGER NOT NULL,
                    end_time INTEGER,
                    summary TEXT,
                    message_count INTEGER DEFAULT 0,
                    extraction_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now')),
                    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)

            # Create indices
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user_time
                ON sessions(user_id, start_time)
            """)

            # Create session_messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp INTEGER DEFAULT (strftime('%s', 'now')),
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)

            # Create session_summaries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    summary_type TEXT DEFAULT 'auto',
                    created_at INTEGER DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)

            # Create session_knowledge_links table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_knowledge_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    relation_type TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at INTEGER DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Start a new session for a user."""
        now = int(time.time())

        # Generate session_id if not provided
        if session_id is None:
            session_id = f"{user_id}_{now}_{os.urandom(4).hex()[:8]}"

        with self._get_connection() as conn:
            # Insert new session
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, user_id, start_time, message_count, extraction_count)
                VALUES (?, ?, ?, 0, 0)
            """, (session_id, user_id, now))

            # Get user stats
            cursor = conn.execute("""
                SELECT COUNT(*) as total_sessions,
                       SUM(message_count) as total_messages
                FROM sessions
                WHERE user_id = ?
            """, (user_id,))
            row = cursor.fetchone()

            total_sessions = row['total_sessions'] if row else 1
            total_messages = row['total_messages'] if row and row['total_messages'] else 0

        return {
            "session_id": session_id,
            "user_id": user_id,
            "session_start_epoch": now,
            "session_start_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "current_session": total_sessions,
        }

    def record_turn(self, user_id: str, session_id: str, turn_duration_sec: float) -> Dict[str, Any]:
        """Record a conversation turn."""
        with self._get_connection() as conn:
            # Update message count
            conn.execute("""
                UPDATE sessions
                SET message_count = message_count + 1,
                    updated_at = strftime('%s', 'now')
                WHERE session_id = ?
            """, (session_id,))

            # Get session info
            cursor = conn.execute("""
                SELECT start_time, message_count
                FROM sessions
                WHERE session_id = ?
            """, (session_id,))

            session = cursor.fetchone()
            if not session:
                # Session doesn't exist, create it
                return self.start_session(user_id, session_id)

            # Get user totals
            cursor = conn.execute("""
                SELECT COUNT(*) as total_sessions,
                       SUM(message_count) as total_turns
                FROM sessions
                WHERE user_id = ?
            """, (user_id,))

            totals = cursor.fetchone()

            now = time.time()
            session_elapsed = now - session['start_time']

            # Calculate current session number
            cursor = conn.execute("""
                SELECT COUNT(*) as session_num
                FROM sessions
                WHERE user_id = ? AND start_time <= ?
            """, (user_id, session['start_time']))

            session_num = cursor.fetchone()['session_num']

        return {
            "session_id": session_id,
            "session_turns": session['message_count'],
            "session_elapsed": session_elapsed,
            "total_sessions": totals['total_sessions'] if totals else 1,
            "total_turns": totals['total_turns'] if totals and totals['total_turns'] else 0,
            "current_session": session_num,
        }

    def get_stats(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get current session statistics."""
        with self._get_connection() as conn:
            # Get session info
            cursor = conn.execute("""
                SELECT start_time, message_count
                FROM sessions
                WHERE session_id = ?
            """, (session_id,))

            session = cursor.fetchone()
            if not session:
                return {}

            # Get user totals
            cursor = conn.execute("""
                SELECT COUNT(*) as total_sessions,
                       SUM(message_count) as total_turns,
                       MIN(start_time) as first_session_time
                FROM sessions
                WHERE user_id = ?
            """, (user_id,))

            totals = cursor.fetchone()

            # Calculate current session number
            cursor = conn.execute("""
                SELECT COUNT(*) as session_num
                FROM sessions
                WHERE user_id = ? AND start_time <= ?
            """, (user_id, session['start_time']))

            session_num = cursor.fetchone()['session_num']

            now = time.time()
            session_elapsed = now - session['start_time']
            total_time = now - totals['first_session_time'] if totals and totals['first_session_time'] else 0

        return {
            "session_id": session_id,
            "session_turns": session['message_count'],
            "session_elapsed": session_elapsed,
            "total_sessions": totals['total_sessions'] if totals else 1,
            "total_turns": totals['total_turns'] if totals and totals['total_turns'] else 0,
            "total_time_seconds": total_time,
            "current_session": session_num,
        }

    def end_session(self, user_id: str, session_id: str):
        """Mark a session as ended."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE sessions
                SET end_time = strftime('%s', 'now'),
                    updated_at = strftime('%s', 'now')
                WHERE session_id = ?
            """, (session_id,))

    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the session history."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO session_messages (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, json.dumps(metadata) if metadata else None))

    def add_summary(self, session_id: str, summary_text: str, summary_type: str = "auto"):
        """Add a summary to the session."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO session_summaries (session_id, summary_text, summary_type)
                VALUES (?, ?, ?)
            """, (session_id, summary_text, summary_type))

    def get_session_history(self, session_id: str) -> list:
        """Get all messages for a session."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT role, content, timestamp, metadata
                FROM session_messages
                WHERE session_id = ?
                ORDER BY timestamp
            """, (session_id,))

            return [dict(row) for row in cursor.fetchall()]

    def get_user_sessions(self, user_id: str, limit: int = 10) -> list:
        """Get recent sessions for a user."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT session_id, start_time, end_time, message_count, summary
                FROM sessions
                WHERE user_id = ?
                ORDER BY start_time DESC
                LIMIT ?
            """, (user_id, limit))

            return [dict(row) for row in cursor.fetchall()]


# Backward compatibility: Use the database tracker by default
SessionTracker = DatabaseSessionTracker