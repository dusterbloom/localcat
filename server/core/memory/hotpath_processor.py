"""
Orchestrator for the HotPath memory system.

This module wires together the specialized components created during the
God-object breakup:
  - MemoryConfiguration for env-driven settings
  - SessionManager for session bookkeeping and headers
  - ContextInjector for fast context insertion
  - MemoryFrameProcessor for Pipecat frame routing
  - ContextCompactor for infinite context via gradient-bang pattern

HotPathMemoryProcessor now focuses on dependency wiring, lifecycle
management, and Pipecat integration.
"""

import os
import sys
import time
from typing import Optional, Dict, Any, List

from loguru import logger

# Add local pipecat to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

PIPECAT_IMPORT_DISABLED = os.getenv("PIPECAT_DISABLE_IMPORT", "").lower() in ("1", "true", "yes", "on")
if not PIPECAT_IMPORT_DISABLED:
    try:
        from pipecat.frames.frames import Frame, TranscriptionFrame  # type: ignore
        from pipecat.processors.frame_processor import FrameProcessor as BaseProcessor, FrameDirection  # type: ignore
        PIPECAT_AVAILABLE = True
    except Exception as exc:
        logger.warning(f"Pipecat import failed ({exc!r}); using stub BaseProcessor")
        PIPECAT_AVAILABLE = False
else:
    logger.debug("Pipecat import disabled via PIPECAT_DISABLE_IMPORT")
    PIPECAT_AVAILABLE = False

if not PIPECAT_AVAILABLE:
    from .frame_processor import Frame, TranscriptionFrame, FrameDirection  # type: ignore

    class BaseProcessor:  # type: ignore
        async def process_frame(self, frame, direction):
            return

        async def push_frame(self, frame, direction):
            pass

from .memory_store import MemoryStore, Paths
from .memory_hotpath import HotMemory
from .context import MemoryContextFrame
from .context_formatter import ContextFormatter
from .config_manager import MemoryConfiguration
from .context_injector import ContextInjector
from .frame_processor import MemoryFrameProcessor
from .session_manager import SessionManager
from .context_compactor import ContextCompactor

# Import intent service when available (optional dependency)
try:
    from ..intent import get_intent_service
    INTENT_SERVICE_AVAILABLE = True
except ImportError:
    INTENT_SERVICE_AVAILABLE = False
    logger.warning("Intent service not available - using standard memory processing")


_HOTMEM_LOG_SINK_ADDED = False


class HotPathMemoryProcessor(BaseProcessor):
    """
    Thin orchestration layer for the HotPath memory components.

    Responsibilities:
    - Instantiate and wire core components
    - Delegate frame handling to MemoryFrameProcessor
    - Emit MemoryContextFrame + optional handshake on injections
    - Keep session headers and metrics up to date
    """

    def __init__(
        self,
        sqlite_path: Optional[str] = None,
        lmdb_dir: Optional[str] = None,
        user_id: str = "default-user",
        enable_metrics: bool = True,
        context_aggregator=None,
        *,
        session_tracker: Optional[Any] = None,
        agent_id: Optional[str] = None,
        config: Optional[MemoryConfiguration] = None,
        hot_memory: Optional[HotMemory] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        super().__init__()

        self.context_aggregator = context_aggregator
        self.session_tracker = session_tracker
        self._last_tracker_stats: Optional[Dict[str, Any]] = None

        self.config = config or MemoryConfiguration.from_env()
        self.config.user_id = user_id or getattr(self.config, "user_id", None) or "default-user"
        self.config.agent_id = agent_id or getattr(self.config, "agent_id", None) or "locat"

        if sqlite_path:
            self.config.sqlite_path = sqlite_path
        if lmdb_dir:
            self.config.lmdb_dir = lmdb_dir

        self.config.enable_metrics = bool(enable_metrics)
        self.config.metrics_enabled = bool(enable_metrics)

        self._handshake_enabled = bool(self.config.handshake_enabled)
        self._metrics_enabled = bool(self.config.metrics_enabled)

        self._configure_logging()

        self.paths = Paths(
            sqlite_path=self.config.sqlite_path,
            lmdb_dir=self.config.lmdb_dir,
        )

        self.store = memory_store or MemoryStore(self.paths)
        self.hot = hot_memory or HotMemory(self.store)

        try:
            self.hot.prewarm("en")
        except Exception as exc:
            logger.debug(f"HotMemory prewarm failed: {exc}")

        try:
            self.hot.rebuild_from_store()
        except Exception as exc:
            logger.warning(f"Could not rebuild HotMemory from store (starting fresh): {exc}")

        # Backfill Enhanced FTS slot tags for existing rows once per startup
        try:
            from .enhanced_fts import EnhancedFTS
            EnhancedFTS(self.store).reindex_existing_data()
        except Exception as exc:
            logger.debug(f"[HotMem] Enhanced FTS reindex skipped: {exc}")

        self.session_id = self._generate_session_id(self.config.user_id)
        self.hot.agent_eid = f"agent:{self.config.agent_id}"
        self.hot.current_user_id = self.config.user_id
        self.hot.current_session_id = self.session_id
        # Make configuration available to retrieval logic
        try:
            self.hot.config = self.config  # type: ignore[attr-defined]
        except Exception:
            pass

        self.session_manager = SessionManager(
            session_id=self.session_id,
            user_eid=self.config.user_id,
            agent_eid=self.config.agent_id,
            config=self.config,
            session_tracker=self.session_tracker,
        )

        self._context_formatter = ContextFormatter(
            max_bullets=self.config.bullets_max,
            inject_role=self.config.inject_role,
            inject_header=self.config.inject_header,
        )

        self.context_injector = ContextInjector(
            hot_memory=self.hot,
            config=self.config,
            formatter=self._context_formatter,
            context_aggregator=context_aggregator,
        )

        self.context_compactor = ContextCompactor(
            max_context_tokens=int(os.getenv("VOICE_AGENT_LLM_MAX_TOKENS", "4096"))
        )

        self.intent_service = self._init_intent_service()

        self.frame_processor = MemoryFrameProcessor(
            config=self.config,
            context_injector=self.context_injector,
            session_manager=self.session_manager,
            hot_memory=self.hot,
            intent_service=self.intent_service,
            on_turn_processed=self._on_turn_processed,
            context_compactor=self.context_compactor,
        )

        self._last_metrics_log = time.time()

        # Start session tracking and inject initial header
        stats = self.session_manager.start_session()
        self._last_tracker_stats = stats
        self.session_manager.ensure_session_header(self.context_aggregator, stats=stats)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Delegate frame handling to MemoryFrameProcessor and emit context frames."""
        await super().process_frame(frame, direction)

        prior_injections = self.context_injector.get_injection_count()

        async for outgoing in self.frame_processor.process_frame(frame, direction):
            await self.push_frame(outgoing, direction)

        if self.context_injector.get_injection_count() > prior_injections:
            bullets = self.context_injector.get_last_injected_bullets()
            await self._emit_memory_frames(bullets, direction)

        if (
            self._handshake_enabled
            and isinstance(frame, TranscriptionFrame)
            and getattr(frame, "is_final", None) in (True, None)
            and not self.context_injector.has_signaled_turn_ready()
        ):
            await self._emit_handshake(direction)

        if self.context_aggregator:
            self.session_manager.ensure_session_header(
                self.context_aggregator,
                stats=self._last_tracker_stats,
            )

    def set_ephemeral_mode(self, enabled: bool) -> None:
        """Enable/disable ephemeral mode across components."""
        enabled = bool(enabled)
        self.config.ephemeral_mode = enabled
        self.session_manager.config.ephemeral_mode = enabled
        self.frame_processor.set_ephemeral_mode(enabled)

        if enabled:
            logger.info("[HotMem] Ephemeral mode ENABLED: memory storage and retrieval bypassed")
        else:
            logger.info("[HotMem] Ephemeral mode DISABLED: normal memory processing restored")

        self.session_manager.ensure_session_header(
            self.context_aggregator,
            stats=self._last_tracker_stats,
        )

    def set_user_identity(self, user_id: str) -> None:
        """Update user identity across session and memory components."""
        self.session_manager.set_user_identity(user_id)
        self.config.user_id = self.session_manager.user_eid
        self.hot.current_user_id = self.session_manager.user_eid

        self.session_manager.ensure_session_header(
            self.context_aggregator,
            stats=self._last_tracker_stats,
        )

    def refresh_session_header(self) -> None:
        """Force session header refresh."""
        self.session_manager.ensure_session_header(
            self.context_aggregator,
            stats=self._last_tracker_stats,
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """Expose memory statistics for diagnostics."""
        return {
            "session_id": self.session_id,
            "hot_metrics": self.hot.get_metrics(),
            "store_metrics": self.store.get_metrics(),
        }

    async def cleanup(self):
        """Cleanup lifecycle resources."""
        try:
            await self.frame_processor.cleanup()
        finally:
            try:
                self.session_manager.end_session()
            except Exception as exc:
                logger.debug(f"[HotMem] Session cleanup failed: {exc}")

    async def _emit_memory_frames(self, bullets: List[str], direction: FrameDirection) -> None:
        """Emit MemoryContextFrame + optional handshake after successful injection."""
        if not bullets:
            return
        try:
            await self.push_frame(
                MemoryContextFrame(self.config.inject_role, self.config.inject_header, bullets),
                direction,
            )
        except Exception as exc:
            logger.debug(f"[HotMem] Failed to push MemoryContextFrame: {exc}")

        await self._emit_handshake(direction)

    async def _emit_handshake(self, direction: FrameDirection) -> None:
        """Emit handshake frame when memory context is ready (even if empty)."""
        if not self._handshake_enabled or self.context_injector.has_signaled_turn_ready():
            return
        try:
            await self.push_frame(MemoryContextReadyFrame(), direction)
            self.context_injector.mark_turn_ready()
        except Exception as exc:
            logger.debug(f"[HotMem] Failed to push handshake frame: {exc}")

    def _on_turn_processed(self, elapsed_ms: float, stats: Optional[Dict[str, Any]]) -> None:
        """Callback invoked by MemoryFrameProcessor after final transcription."""
        self._last_tracker_stats = stats
        if self._metrics_enabled:
            self._log_metrics(elapsed_ms)

        if self.context_aggregator:
            self.session_manager.ensure_session_header(self.context_aggregator, stats=stats)

    def _log_metrics(self, elapsed_ms: float) -> None:
        """Log periodic performance metrics."""
        now = time.time()
        interval = max(10, getattr(self.config, "metrics_log_interval", 30))

        if now - self._last_metrics_log < interval:
            return

        metrics = self.hot.get_metrics()
        store_metrics = self.store.get_metrics()

        logger.debug(f"[HotMem] Turn processed in {elapsed_ms:.1f}ms")
        for name, stats in metrics.items():
            logger.debug(f"[HotMem] {name}: {stats}")
        for name, stats in store_metrics.items():
            logger.debug(f"[HotMem Store] {name}: {stats}")

        self._last_metrics_log = now

    def _init_intent_service(self):
        """Initialize intent service when available and enabled."""
        if not INTENT_SERVICE_AVAILABLE:
            return None
        if not getattr(self.config, "intent_aware_processing", True):
            return None

        try:
            service = get_intent_service()
            logger.info("[HotMem] Intent-aware processing enabled")
            return service
        except Exception as exc:
            logger.warning(f"[HotMem] Failed to initialize intent service: {exc}")
            return None

    def _configure_logging(self) -> None:
        """Configure file logging once per process."""
        global _HOTMEM_LOG_SINK_ADDED
        if _HOTMEM_LOG_SINK_ADDED:
            return

        try:
            log_path = os.getenv(
                "HOTMEM_LOG_FILE",
                os.path.join(os.path.dirname(__file__), ".logs", "hotmem.log"),
            )
            log_level = os.getenv("HOTMEM_LOG_LEVEL", "DEBUG").upper()
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            logger.add(
                log_path,
                rotation="10 MB",
                retention="10 days",
                enqueue=True,
                level=log_level,
                backtrace=False,
                diagnose=False,
            )
            _HOTMEM_LOG_SINK_ADDED = True
        except Exception as exc:
            logger.warning(f"HotMem file logging not enabled: {exc}")

    @staticmethod
    def _generate_session_id(user_id: str) -> str:
        """Generate a unique session identifier."""
        return f"{user_id}_{int(time.time())}_{os.urandom(4).hex()}"


class MemoryContextReadyFrame(Frame):
    """Handshake frame announcing memory context availability."""

    pass
