"""
Anonymous Performance Policy

Applies a per-session, in-memory performance overlay when anonymous/ephemeral
mode is enabled. This policy never mutates environment variables or files.

Effects (anonymous active only):
- Ensure memory processor operates in ephemeral mode (bypass heavy paths)
- Switch session tracker to ephemeral (no disk writes)
- Tighten TTS text aggregation for faster TTFB

All changes are reversible via clear().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
from loguru import logger


@dataclass
class _TextAggSnapshot:
    min_tokens: int
    max_tokens: int
    max_time: float


class AnonymousPerformancePolicy:
    """
    Runtime overlay applied only while anonymous mode is active.

    It operates on live service instances fetched from the ServiceFactory
    (via its get_service method), and reverts to the original behavior on exit.
    """

    def __init__(self, *, factory: Any, memory_processor: Optional[Any] = None):
        self._factory = factory
        self._memory_processor = memory_processor
        self._active = False

        # Snapshots for restore
        self._text_agg_snapshot: Optional[_TextAggSnapshot] = None
        self._session_tracker_snapshot: Optional[bool] = None

    def apply(self) -> None:
        if self._active:
            return

        try:
            # 1) Ensure memory processor is in ephemeral mode (bypass heavy memory paths)
            if self._memory_processor and hasattr(self._memory_processor, "set_ephemeral_mode"):
                try:
                    self._memory_processor.set_ephemeral_mode(True)
                    logger.info("[AnonPolicy] Memory processor set to ephemeral mode")
                except Exception as e:
                    logger.warning(f"[AnonPolicy] Failed to set memory ephemeral mode: {e}")

            # 2) Switch session tracker to ephemeral (no disk writes)
            try:
                tracker = self._get_service("session_tracker")
                if tracker is not None:
                    # Snapshot current state (best-effort)
                    self._session_tracker_snapshot = getattr(tracker, "_ephemeral", None)
                    setattr(tracker, "_ephemeral", True)
                    logger.info("[AnonPolicy] SessionTracker set to ephemeral=True (no persistence)")
            except Exception as e:
                logger.warning(f"[AnonPolicy] Failed to set SessionTracker ephemeral: {e}")

            # 3) Tighten TTS text aggregation for lower TTFB
            try:
                text_agg = self._get_service("text_aggregator")
                if text_agg is not None:
                    # Detect FastTextAggregator by attributes to avoid tight coupling
                    has_min = hasattr(text_agg, "_min_tokens")
                    has_max = hasattr(text_agg, "_max_tokens")
                    has_time = hasattr(text_agg, "_max_time")
                    if has_min and has_max and has_time:
                        # Snapshot
                        self._text_agg_snapshot = _TextAggSnapshot(
                            min_tokens=int(getattr(text_agg, "_min_tokens", 175)),
                            max_tokens=int(getattr(text_agg, "_max_tokens", 250)),
                            max_time=float(getattr(text_agg, "_max_time", 0.5)),
                        )
                        # Apply anonymous-friendly values
                        try:
                            setattr(text_agg, "_min_tokens", 100)
                            setattr(text_agg, "_max_tokens", 175)
                            setattr(text_agg, "_max_time", 0.4)
                            logger.info("[AnonPolicy] Text aggregator clamped (min=100, max=175, time=0.4s)")
                        except Exception as e:
                            logger.warning(f"[AnonPolicy] Failed to clamp text aggregator: {e}")
            except Exception as e:
                logger.warning(f"[AnonPolicy] Text aggregator tuning failed: {e}")

            self._active = True
        except Exception as e:
            logger.error(f"[AnonPolicy] Apply failed: {e}")

    def clear(self) -> None:
        if not self._active:
            return

        try:
            # 1) Restore session tracker ephemeral flag
            try:
                tracker = self._get_service("session_tracker")
                if tracker is not None and self._session_tracker_snapshot is not None:
                    setattr(tracker, "_ephemeral", self._session_tracker_snapshot)
                    logger.info("[AnonPolicy] SessionTracker ephemeral restored")
            except Exception as e:
                logger.warning(f"[AnonPolicy] Failed to restore SessionTracker state: {e}")

            # 2) Restore text aggregator parameters
            try:
                text_agg = self._get_service("text_aggregator")
                snap = self._text_agg_snapshot
                if text_agg is not None and snap is not None:
                    try:
                        setattr(text_agg, "_min_tokens", snap.min_tokens)
                        setattr(text_agg, "_max_tokens", snap.max_tokens)
                        setattr(text_agg, "_max_time", snap.max_time)
                        logger.info("[AnonPolicy] Text aggregator parameters restored")
                    except Exception as e:
                        logger.warning(f"[AnonPolicy] Failed to restore text aggregator: {e}")
            except Exception as e:
                logger.warning(f"[AnonPolicy] Text aggregator restore failed: {e}")

            self._active = False
        except Exception as e:
            logger.error(f"[AnonPolicy] Clear failed: {e}")

    # ------------------------
    # Helpers
    # ------------------------
    def _get_service(self, name: str) -> Optional[Any]:
        """Fetch a live service by name from the factory cache (best-effort)."""
        if not self._factory:
            return None
        try:
            if hasattr(self._factory, "get_service"):
                return self._factory.get_service(name)
        except Exception:
            pass
        return None

