"""
Configuration Management for Memory Systems

Centralized configuration following the Single Responsibility Principle.
Handles parsing and validation of 40+ environment variables with type safety.

NOTE: This module now uses the unified configuration base classes and parsing
utilities from config.base_config and config.parsers to eliminate code duplication.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from loguru import logger

# Import unified configuration base and parsers
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config.base_config import BaseConfiguration
from config.parsers import _parse_bool, _parse_int, _parse_float, _parse_list


@dataclass
class MemoryConfiguration(BaseConfiguration):
    """
    Complete configuration for HotPath memory system.

    Single source of truth for all memory-related settings.
    Consolidates configuration from the hotpath_processor.py god object.

    Inherits from BaseConfiguration for unified config management and validation.
    """

    # Core settings
    enabled: bool = True
    bullets_max: int = 3
    interim_min_words: int = 6
    inject_role: str = "system"
    inject_header: str = "Use the following factual context if helpful."

    # Retrieval settings
    sources: List[str] = field(default_factory=lambda: ["graph"])
    convo_index_enabled: bool = False
    max_turn_pairs: int = 4
    ctx_window_enabled: bool = True
    ctx_max_pairs: int = 4

    # Token-aware context management (prevent degradation in long chats)
    llm_context_max_tokens: int = 3000
    llm_context_prune_threshold: float = 0.70
    llm_context_min_turns: int = 3

    # Token budget and filtering
    token_budget: int = 300
    max_bullets: int = 2
    filter_quality: bool = True

    # Session tracking
    session_tracking_enabled: bool = True
    session_header_enabled: bool = True
    user_id: str = "default-user"
    agent_id: str = "locat"

    # Background processing
    summarization_enabled: bool = False
    summary_base_url: str = "http://127.0.0.1:1234/v1"
    summary_api_key: str = ""
    summary_model: str = "llama-3.2-3b-instruct"
    summary_interval_secs: float = 60.0
    summary_max_tokens: int = 160
    summary_max_messages: int = 10
    summary_window_mode: str = "turn_pairs"
    summary_turn_pairs: int = 5

    # Ephemeral mode
    ephemeral_mode: bool = False
    ephemeral_ttl_seconds: int = 3600
    excluded_phrases: List[str] = field(default_factory=list)

    # Performance and caching
    retrieval_timeout_ms: int = 50
    cache_enabled: bool = True
    metrics_enabled: bool = True
    metrics_log_interval: int = 60
    enable_metrics: bool = True

    # Audio intelligence
    audio_intel_enabled: bool = True
    audio_intel_intro_pipeline: bool = True

    # Intent service
    intent_aware_processing: bool = True
    intent_classification_enabled: bool = True

    # Frame tracing
    trace_frames: bool = False
    handshake_enabled: bool = True

    # Composite scoring weights
    rerank_weights: Optional[Dict[str, float]] = None
    weight_graph: float = 0.3
    weight_convo: float = 0.4
    weight_summary: float = 0.2
    weight_semantic: float = 0.1

    # Storage paths
    sqlite_path: Optional[str] = None
    lmdb_dir: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing"""
        # Ensure lists are properly initialized
        if not self.sources:
            self.sources = ["graph"]
        if not self.excluded_phrases:
            self.excluded_phrases = []

        # Validate configuration values
        if self.bullets_max < 0:
            self.bullets_max = 3
        if self.interim_min_words < 0:
            self.interim_min_words = 6
        if self.inject_role not in ("user", "system"):
            self.inject_role = "system"

    @classmethod
    def from_env(cls) -> 'MemoryConfiguration':
        """
        Parse all memory configuration from environment variables.

        Supports both MEMORY_* and legacy HOTMEM_* prefixes for backward compatibility.

        Returns:
            MemoryConfiguration with all settings
        """

        def get_env(key: str, legacy_key: str = None) -> Optional[str]:
            """Get env var with fallback to legacy key"""
            value = os.getenv(f"MEMORY_{key}")
            if value is None and legacy_key:
                value = os.getenv(f"HOTMEM_{legacy_key}")
            return value

        # Core settings
        enabled = _parse_bool(get_env("ENABLED", "ENABLED") or "true")
        bullets_max = _parse_int(get_env("BULLETS_MAX", "BULLETS_MAX"), default=3)
        interim_min_words = _parse_int(get_env("INTERIM_MIN_WORDS", "INTERIM_MIN_WORDS"), default=6)
        inject_role = get_env("INJECT_ROLE", "INJECT_ROLE") or "system"
        inject_header = get_env("INJECT_HEADER", "INJECT_HEADER") or "Use the following factual context if helpful."

        # Retrieval settings
        sources = _parse_list(get_env("SOURCES", "SOURCES"), default=["graph"])
        convo_index_enabled = _parse_bool(get_env("CONVO_INDEX", "CONVO_INDEX") or "false")
        max_turn_pairs = _parse_int(get_env("MAX_TURN_PAIRS"), default=4)
        ctx_window_enabled = _parse_bool(get_env("CONTEXT_SLIDING_WINDOW", "CTX_PRUNE_ENABLED") or "true")
        ctx_max_pairs = _parse_int(get_env("CONTEXT_MAX_TURN_PAIRS", "CTX_MAX_PAIRS"), default=4)

        # Token-aware context management
        llm_context_max_tokens = _parse_int(os.getenv("LLM_CONTEXT_MAX_TOKENS"), default=3000)
        llm_context_prune_threshold = _parse_float(os.getenv("LLM_CONTEXT_PRUNE_THRESHOLD"), default=0.70)
        llm_context_min_turns = _parse_int(os.getenv("LLM_CONTEXT_MIN_TURNS"), default=3)

        # Token budget and filtering
        token_budget = _parse_int(get_env("TOKEN_BUDGET"), default=300)
        max_bullets = _parse_int(get_env("MAX_BULLETS"), default=2)
        filter_quality = _parse_bool(get_env("FILTER_QUALITY") or "true")

        # Session tracking
        session_tracking_enabled = _parse_bool(get_env("SESSION_TRACKING") or "true")
        session_header_enabled = _parse_bool(get_env("SESSION_HEADER") or "true")
        user_id = get_env("USER_ID") or "default-user"
        agent_id = get_env("AGENT_ID") or "locat"

        # Background processing
        summarization_enabled = _parse_bool(get_env("SUMMARIZER_ENABLED", "SUMMARY_ENABLED") or "false")
        summary_base_url = get_env("SUMMARIZER_BASE_URL") or "http://127.0.0.1:1234/v1"
        summary_api_key = get_env("SUMMARIZER_API_KEY") or ""
        summary_model = get_env("SUMMARIZER_MODEL") or "llama-3.2-3b-instruct"
        summary_interval_secs = _parse_float(get_env("SUMMARIZER_INTERVAL_SECS"), default=60.0)
        summary_max_tokens = _parse_int(get_env("SUMMARIZER_MAX_TOKENS"), default=160)
        summary_max_messages = _parse_int(get_env("SUMMARIZER_MAX_MESSAGES"), default=10)
        summary_window_mode = get_env("SUMMARIZER_WINDOW_MODE") or "turn_pairs"
        summary_turn_pairs = _parse_int(get_env("SUMMARIZER_TURN_PAIRS"), default=5)

        # Ephemeral mode
        ephemeral_mode = _parse_bool(get_env("EPHEMERAL_MODE") or "false")
        ephemeral_ttl_seconds = _parse_int(get_env("EPHEMERAL_TTL"), default=3600)
        
        # Excluded phrases
        ex_phr = get_env("EXCLUDED_MEMORY_PHRASES", "EXCLUDED_PHRASES") or ""
        fixed = get_env("ENROLLMENT_FIXED_PHRASE") or ""
        excluded_phrases = []
        if ex_phr:
            excluded_phrases.extend([p.strip() for p in ex_phr.split("||") if p.strip()])
        if fixed:
            excluded_phrases.append(fixed.strip())

        # Performance and caching
        retrieval_timeout_ms = _parse_int(get_env("RETRIEVAL_TIMEOUT_MS"), default=50)
        cache_enabled = _parse_bool(get_env("CACHE_ENABLED") or "true")
        metrics_enabled = _parse_bool(get_env("METRICS_ENABLED", "METRICS") or "true")
        metrics_log_interval = _parse_int(get_env("METRICS_LOG_INTERVAL"), default=60)
        enable_metrics = _parse_bool(get_env("ENABLE_METRICS") or "true")

        # Audio intelligence
        audio_intel_enabled = _parse_bool(get_env("AUDIO_INTELLIGENCE_ENABLED") or "true")
        audio_intel_intro_pipeline = _parse_bool(get_env("AUDIO_INTEL_INTRO_PIPELINE") or "true")

        # Intent service
        intent_aware_processing = _parse_bool(get_env("INTENT_CLASSIFICATION_ENABLED") or "true")
        intent_classification_enabled = _parse_bool(get_env("INTENT_CLASSIFICATION_ENABLED") or "true")

        # Frame tracing
        trace_frames = _parse_bool(get_env("MEMORY_TRACE_FRAMES", "TRACE_FRAMES") or "false")
        handshake_enabled = _parse_bool(get_env("MEMORY_ENABLE_HANDSHAKE", "HANDSHAKE_ENABLED") or "true")

        # Composite scoring weights
        rerank_weights = None
        rerank_weights_json = get_env("RERANK_WEIGHTS")
        if rerank_weights_json:
            try:
                import json
                rerank_weights = json.loads(rerank_weights_json)
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Invalid MEMORY_RERANK_WEIGHTS JSON: {rerank_weights_json}")

        weight_graph = _parse_float(get_env("WEIGHT_GRAPH", "WEIGHT_GRAPH"), default=0.3)
        weight_convo = _parse_float(get_env("WEIGHT_CONVO", "WEIGHT_CONVO"), default=0.4)
        weight_summary = _parse_float(get_env("WEIGHT_SUMMARY", "WEIGHT_SUMMARY"), default=0.2)
        weight_semantic = _parse_float(get_env("WEIGHT_SEMANTIC", "WEIGHT_SEMANTIC"), default=0.1)

        # Storage paths
        sqlite_path = get_env("SQLITE_PATH", "SQLITE")
        lmdb_dir = get_env("LMDB_DIR", "LMDB_DIR")

        return cls(
            # Core settings
            enabled=enabled,
            bullets_max=bullets_max,
            interim_min_words=interim_min_words,
            inject_role=inject_role,
            inject_header=inject_header,

            # Retrieval settings
            sources=sources,
            convo_index_enabled=convo_index_enabled,
            max_turn_pairs=max_turn_pairs,
            ctx_window_enabled=ctx_window_enabled,
            ctx_max_pairs=ctx_max_pairs,

            # Token-aware context management
            llm_context_max_tokens=llm_context_max_tokens,
            llm_context_prune_threshold=llm_context_prune_threshold,
            llm_context_min_turns=llm_context_min_turns,

            # Token budget and filtering
            token_budget=token_budget,
            max_bullets=max_bullets,
            filter_quality=filter_quality,

            # Session tracking
            session_tracking_enabled=session_tracking_enabled,
            session_header_enabled=session_header_enabled,
            user_id=user_id,
            agent_id=agent_id,

            # Background processing
            summarization_enabled=summarization_enabled,
            summary_base_url=summary_base_url,
            summary_api_key=summary_api_key,
            summary_model=summary_model,
            summary_interval_secs=summary_interval_secs,
            summary_max_tokens=summary_max_tokens,
            summary_max_messages=summary_max_messages,
            summary_window_mode=summary_window_mode,
            summary_turn_pairs=summary_turn_pairs,

            # Ephemeral mode
            ephemeral_mode=ephemeral_mode,
            ephemeral_ttl_seconds=ephemeral_ttl_seconds,
            excluded_phrases=excluded_phrases,

            # Performance and caching
            retrieval_timeout_ms=retrieval_timeout_ms,
            cache_enabled=cache_enabled,
            metrics_enabled=metrics_enabled,
            metrics_log_interval=metrics_log_interval,
            enable_metrics=enable_metrics,

            # Audio intelligence
            audio_intel_enabled=audio_intel_enabled,
            audio_intel_intro_pipeline=audio_intel_intro_pipeline,

            # Intent service
            intent_aware_processing=intent_aware_processing,
            intent_classification_enabled=intent_classification_enabled,

            # Frame tracing
            trace_frames=trace_frames,
            handshake_enabled=handshake_enabled,

            # Composite scoring weights
            rerank_weights=rerank_weights,
            weight_graph=weight_graph,
            weight_convo=weight_convo,
            weight_summary=weight_summary,
            weight_semantic=weight_semantic,

            # Storage paths
            sqlite_path=sqlite_path,
            lmdb_dir=lmdb_dir
        )

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings.

        Returns:
            List of validation warning messages (empty if all valid)
        """
        warnings = []

        if self.bullets_max < 1 or self.bullets_max > 10:
            warnings.append(f"bullets_max={self.bullets_max} outside recommended range [1-10]")

        if self.retrieval_timeout_ms > 100:
            warnings.append(f"retrieval_timeout_ms={self.retrieval_timeout_ms} exceeds 100ms (impacts latency)")

        if not self.sources:
            warnings.append("No retrieval sources configured (memory will not work)")

        if self.token_budget < 100:
            warnings.append(f"token_budget={self.token_budget} too low (minimum 100 recommended)")

        if self.summary_interval_secs < 10 and self.summarization_enabled:
            warnings.append(f"summary_interval_secs={self.summary_interval_secs} too frequent (minimum 10s recommended)")

        if self.weight_graph + self.weight_convo + self.weight_summary + self.weight_semantic > 1.0:
            warnings.append("Sum of source weights exceeds 1.0 (weights should be normalized)")

        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging/debugging."""
        return {
            # Core settings
            "enabled": self.enabled,
            "bullets_max": self.bullets_max,
            "interim_min_words": self.interim_min_words,
            "inject_role": self.inject_role,
            "inject_header": self.inject_header,

            # Retrieval settings
            "sources": self.sources,
            "convo_index_enabled": self.convo_index_enabled,
            "max_turn_pairs": self.max_turn_pairs,
            "ctx_window_enabled": self.ctx_window_enabled,
            "ctx_max_pairs": self.ctx_max_pairs,

            # Token budget and filtering
            "token_budget": self.token_budget,
            "max_bullets": self.max_bullets,
            "filter_quality": self.filter_quality,

            # Session tracking
            "session_tracking_enabled": self.session_tracking_enabled,
            "session_header_enabled": self.session_header_enabled,
            "user_id": self.user_id,
            "agent_id": self.agent_id,

            # Background processing
            "summarization_enabled": self.summarization_enabled,
            "summary_window_mode": self.summary_window_mode,
            "summary_turn_pairs": self.summary_turn_pairs,

            # Performance
            "retrieval_timeout_ms": self.retrieval_timeout_ms,
            "metrics_enabled": self.metrics_enabled,

            # Ephemeral mode
            "ephemeral_mode": self.ephemeral_mode,
            "excluded_phrases_count": len(self.excluded_phrases)
        }

    def log_configuration(self) -> None:
        """Log current configuration for debugging."""
        config_dict = self.to_dict()
        logger.info(f"Memory system configuration: {config_dict}")


# Global configuration instance (following existing pattern)
_memory_config: Optional[MemoryConfiguration] = None


def get_memory_config() -> MemoryConfiguration:
    """
    Get the global memory configuration instance.

    This follows the singleton pattern to ensure consistent configuration
    across the memory system.
    """
    global _memory_config
    if _memory_config is None:
        _memory_config = MemoryConfiguration.from_env()
        _memory_config.log_configuration()
    return _memory_config


def reload_memory_config() -> MemoryConfiguration:
    """
    Reload configuration from environment variables.

    Useful for testing and configuration updates.
    """
    global _memory_config
    _memory_config = MemoryConfiguration.from_env()
    _memory_config.log_configuration()
    return _memory_config
