"""
Session tracking and header management for memory system.

Handles session metadata injection and turn tracking.
"""

import os
import time
import hashlib
from typing import Dict, Optional, Any, List
from loguru import logger
from .config_manager import MemoryConfiguration


class SessionManager:
    """
    Manage session tracking and headers.

    Responsibilities:
    - Track session metadata (ID, user, agent)
    - Inject session headers into context
    - Track conversation turns
    - Measure session duration
    """

    def __init__(
        self,
        session_id: str,
        user_eid: str,
        agent_eid: str,
        config: MemoryConfiguration,
        session_tracker=None
    ):
        self.session_id = session_id
        self.user_eid = user_eid
        self.agent_eid = agent_eid
        self.config = config
        self.session_tracker = session_tracker

        self._turn_count = 0
        self._start_time = time.time()
        self._last_user_speech_time = None
        self._display_user_id = user_eid  # For display purposes
        self._display_agent_id = agent_eid  # For display purposes

        # Session header tag
        self._session_header_tag = "[Session Context]"

    def get_session_header(self, stats: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate session header for context injection.

        Returns:
            Formatted session header string
        """
        if not self.config.session_header_enabled:
            return ""

        current_time = int(time.time())
        duration = current_time - int(self._start_time)

        # Use coarse-grained values to preserve LLM KV cache
        # Only show date without time to avoid cache invalidation
        system_date = time.strftime("%Y-%m-%d")

        # Round session durations to nearest 5 minutes to reduce cache invalidation
        session_minutes = int(duration / 60)
        session_minutes_rounded = (session_minutes // 5) * 5  # Round to nearest 5min

        # Show anonymous label in ephemeral mode
        if self.config.ephemeral_mode:
            display_user = "anonymous"
        else:
            display_user = self._display_user_id
            display_agent = self._display_agent_id

        
        
        # Get session stats from tracker if available
        total_sessions = 1
        total_time = duration
        current_session = 1

        tracker_stats = stats
        if tracker_stats is None and self.session_tracker:
            try:
                tracker_stats = self.session_tracker.get_stats(self.user_eid, self.session_id)
            except Exception as e:
                logger.debug(f"Failed to get session stats: {e}")

        if tracker_stats:
            try:
                total_sessions = int(tracker_stats.get("total_sessions", 1))
                total_time = float(tracker_stats.get("total_time_seconds", duration))
                current_session = int(tracker_stats.get("current_session", total_sessions))
            except Exception as e:
                logger.debug(f"Failed to normalize tracker stats: {e}")

        # Round total time to nearest 5 minutes
        total_minutes = int(total_time / 60)
        total_minutes_rounded = (total_minutes // 5) * 5

        lines = [
            self._session_header_tag,
            f"Date: {system_date}",
            f"User: {display_user}",
            f"Agent: {display_agent}",
            f"Session #{current_session}",
            f"Total sessions: {total_sessions}",
        ]

        # Only add timing info if significant (>= 5 min)
        if session_minutes_rounded >= 5:
            lines.append(f"Session: ~{session_minutes_rounded}min")
        if total_minutes_rounded >= 5 and total_minutes_rounded != session_minutes_rounded:
            lines.append(f"Total time: ~{total_minutes_rounded}min")

        return "\n".join(lines)

    async def mark_user_speaking(self):
        """Mark user started speaking"""
        self._last_user_speech_time = time.time()

    def increment_turn(self):
        """Increment turn counter"""
        self._turn_count += 1

    def record_turn_metrics(self, elapsed_ms: float) -> Optional[Dict[str, Any]]:
        """Record turn metrics via session tracker"""
        if not self.session_tracker:
            return None
            
        try:
            stats = self.session_tracker.record_turn(
                self.user_eid, 
                self.session_id, 
                elapsed_ms / 1000.0
            )
            return stats
        except Exception as e:
            logger.warning(f"Failed to record turn metrics: {e}")
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get session metrics"""
        current_time = time.time()
        
        base_metrics = {
            "session_id": self.session_id,
            "turn_count": self._turn_count,
            "duration_seconds": int(current_time - self._start_time),
            "user_eid": self.user_eid,
            "agent_eid": self.agent_eid
        }

        # Add tracker stats if available
        if self.session_tracker:
            try:
                stats = self.session_tracker.get_stats(self.user_eid, self.session_id)
                if stats:
                    base_metrics.update({
                        "session_turns": int(stats.get("session_turns", self._turn_count)),
                        "total_turns": int(stats.get("total_turns", self._turn_count)),
                        "total_sessions": int(stats.get("total_sessions", 1)),
                        "total_time_seconds": float(stats.get("total_time_seconds", base_metrics["duration_seconds"])),
                        "session_start_iso": stats.get("session_start_iso"),
                        "current_session": int(stats.get("current_session", 1))
                    })
            except Exception as e:
                logger.debug(f"Failed to get tracker stats: {e}")

        return base_metrics

    def set_user_identity(self, user_id: str) -> None:
        """
        Switch the active user identity for headers and future indexing.
        
        Args:
            user_id: New user identifier
        """
        try:
            user_id = (user_id or "").strip() or self.user_eid
            if user_id != self.user_eid:
                # Store original user_id for session tracking to avoid case sensitivity issues
                self._display_user_id = user_id  # For display purposes
                # Keep the original case for session tracking compatibility
                session_user_id = self._normalize_user_id_for_session(user_id)
                
                if session_user_id != self.user_eid:
                    logger.info(f"[SessionManager] User identity changed from '{self.user_eid}' to '{session_user_id}' (display: '{user_id}')")
                    self.user_eid = session_user_id
        except Exception as e:
            logger.warning(f"[SessionManager] Failed to set user identity: {e}")
    
    def _normalize_user_id_for_session(self, user_id: str) -> str:
        """
        Normalize user_id for session tracking to avoid case sensitivity issues.
        
        This ensures that session tracking works consistently regardless of
        how the user's name is capitalized by speaker recognition.
        """
        # Use the original environment variable value if available
        env_user_id = os.getenv("USER_ID", "")
        if env_user_id.lower() == user_id.lower():
            return env_user_id
        
        # Otherwise use lowercase for consistency
        return user_id.lower()

    def start_session(self) -> Optional[Dict[str, Any]]:
        """Start session tracking via session tracker"""
        if not self.session_tracker:
            return None
            
        try:
            stats = self.session_tracker.start_session(self.user_eid, self.session_id)
            logger.debug(f"[SessionManager] Session started: {self.user_eid}/{self.session_id}")
            return stats
        except Exception as e:
            logger.warning(f"[SessionManager] Failed to start session: {e}")
            return None

    def end_session(self) -> None:
        """End session tracking via session tracker"""
        if not self.session_tracker:
            return
            
        try:
            self.session_tracker.end_session(self.user_eid, self.session_id)
            logger.debug(f"[SessionManager] Session ended: {self.user_eid}/{self.session_id}")
        except Exception as e:
            logger.warning(f"[SessionManager] Failed to end session: {e}")

    def is_session_owned_by_user(self, user_id: str, session_id: str) -> bool:
        """
        Check if a session belongs to a user.
        
        This can be used for cross-session memory access control.
        """
        if not self.session_tracker or not session_id or not user_id:
            return False
            
        try:
            return self.session_tracker.is_session_owned_by_user(session_id, user_id)
        except Exception as e:
            logger.debug(f"Failed to check session ownership: {e}")
            return False

    def get_recent_conversation_chunks(self, limit: int = 10) -> List[tuple]:
        """
        Get recent conversation chunks for this session.
        
        Returns:
            List of (text, timestamp) tuples
        """
        # This would need access to the memory store
        # For now, return empty - integration will handle this
        return []

    def should_inject_header(self) -> bool:
        """Check if session header should be injected"""
        if self.config.ephemeral_mode:
            return False
        return self.config.session_header_enabled

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        seconds = max(float(seconds), 0.0)
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours}h {mins}m {secs}s"
        if mins:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def ensure_session_header(
        self,
        context_aggregator,
        *,
        stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ensure the session header is present (or removed) in the context aggregator.

        Args:
            context_aggregator: Pipeline context aggregator with user() accessor.
            stats: Optional stats returned by session tracker to avoid duplicate lookups.
        """
        if context_aggregator is None:
            return

        try:
            context = context_aggregator.user().context
        except Exception as exc:
            logger.debug(f"[SessionManager] Context aggregator unavailable: {exc}")
            return

        try:
            messages = list(context.get_messages())
        except Exception as exc:
            logger.debug(f"[SessionManager] Failed to read context messages: {exc}")
            return

        # Ephemeral mode disables headers altogether
        if self.config.ephemeral_mode or not self.config.session_header_enabled:
            filtered = [msg for msg in messages if not self._is_session_header(msg)]
            try:
                context.set_messages(filtered)
            except Exception as exc:
                logger.debug(f"[SessionManager] Failed to clear session header: {exc}")
            return

        header_text = self.get_session_header(stats)
        if not header_text:
            return

        header_message = {"role": "system", "content": header_text}
        existing_idx = self._find_session_header_index(messages)

        if existing_idx is None:
            messages.insert(0, header_message)
        else:
            messages[existing_idx] = header_message

        try:
            context.set_messages(messages)
        except Exception as exc:
            logger.debug(f"[SessionManager] Failed to update session header: {exc}")

    def _is_session_header(self, message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        if message.get("role") != "system":
            return False
        content = message.get("content")
        return isinstance(content, str) and content.startswith(self._session_header_tag)

    def _find_session_header_index(self, messages: list) -> Optional[int]:
        for idx, message in enumerate(messages):
            if self._is_session_header(message):
                return idx
        return None
