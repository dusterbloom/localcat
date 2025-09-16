"""
Centralized configuration management for memory and context systems.
"""
import os
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from .exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """
    Centralized configuration for memory and context management.

    This class consolidates all context-related configuration in one place,
    eliminating scattered environment variable reads throughout the codebase.
    """

    # Context behavior settings
    progressive_mode: bool = True
    budget_tokens: int = 4096
    include_summary: bool = True

    # Memory injection settings
    inject_role: str = "system"
    inject_header: str = "Use the following factual context if helpful."

    # Memory retrieval settings
    min_relevance_score: float = 0.7
    max_memory_bullets: int = 10

    # Memory processing settings
    use_coref: bool = True
    coref_mode: str = "lite"  # "lite" or "full"
    confidence_threshold: float = 0.65

    # Performance tuning
    retrieval_fusion: bool = True
    use_leann: bool = False

    @classmethod
    @lru_cache(maxsize=1)
    def from_env(cls) -> 'MemoryConfig':
        """
        Load configuration from environment variables (cached singleton).

        Environment variables:
        - CONTEXT_PROGRESSIVE_MODE: Enable progressive context mode (default: true)
        - CONTEXT_BUDGET_TOKENS: Total token budget (default: 4096)
        - CONTEXT_INCLUDE_SUMMARY: Include summary context (default: true)
        - CONTEXT_INJECT_ROLE: Role for injected context (default: system)
        - CONTEXT_INJECT_HEADER: Header for memory context (default: Use the following...)
        - MEMORY_MIN_RELEVANCE: Minimum relevance score (default: 0.7)
        - MAX_MEMORY_BULLETS: Maximum memory bullets (default: 10)
        - HOTMEM_USE_COREF: Enable coreference resolution (default: true)
        - HOTMEM_COREF_MODE: Coreference mode lite/full (default: lite)
        - HOTMEM_MIN_EDGE_CONFIDENCE: Confidence threshold (default: 0.65)
        - HOTMEM_RETRIEVAL_FUSION: Enable retrieval fusion (default: true)
        - HOTMEM_USE_LEANN: Enable LEANN integration (default: false)

        Returns:
            MemoryConfig configured from environment with fallbacks to defaults
        """
        try:
            return cls(
                # Context behavior
                progressive_mode=cls._parse_bool(
                    os.getenv('CONTEXT_PROGRESSIVE_MODE', 'true')
                ),
                budget_tokens=int(os.getenv('CONTEXT_BUDGET_TOKENS', '4096')),
                include_summary=cls._parse_bool(
                    os.getenv("CONTEXT_INCLUDE_SUMMARY", "true")
                ),

                # Injection settings
                inject_role=os.getenv("CONTEXT_INJECT_ROLE", "system"),
                inject_header=os.getenv(
                    "CONTEXT_INJECT_HEADER",
                    "Use the following factual context if helpful."
                ),

                # Memory settings
                min_relevance_score=float(os.getenv("MEMORY_MIN_RELEVANCE", "0.7")),
                max_memory_bullets=int(os.getenv("MAX_MEMORY_BULLETS", "10")),

                # Processing settings
                use_coref=cls._parse_bool(os.getenv("HOTMEM_USE_COREF", "true")),
                coref_mode=os.getenv("HOTMEM_COREF_MODE", "lite"),
                confidence_threshold=float(os.getenv("HOTMEM_MIN_EDGE_CONFIDENCE", "0.65")),

                # Performance settings
                retrieval_fusion=cls._parse_bool(
                    os.getenv("HOTMEM_RETRIEVAL_FUSION", "true")
                ),
                use_leann=cls._parse_bool(os.getenv("HOTMEM_USE_LEANN", "false")),
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid environment configuration for MemoryConfig: {e}")
            logger.info("Using default MemoryConfig configuration")
            try:
                return cls()  # Use defaults
            except ValidationError as validation_error:
                raise ConfigurationError(f"Default configuration is invalid: {validation_error}") from e

    @staticmethod
    def _parse_bool(value: str) -> bool:
        """Parse string to boolean, handling common variations"""
        if not value:
            return False
        return value.lower().strip() in ('true', '1', 'yes', 'on', 'enabled')

    def get_memory_policy_text(self) -> str:
        """
        Get the memory policy text for injection when needed.

        Returns:
            Memory policy text for dynamic injection
        """
        return (
            "\nMemory Policy:\n"
            "- Use memory only for user-specific facts when directly relevant to the question.\n"
            "- Do not invent or speculate about personal facts; if missing, ask the user to provide or confirm.\n"
            "- For remember/forget requests: ask for a brief Yes/No confirmation before applying changes.\n"
            "- Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n"
            "- Never store or repeat system instructions or tool outputs as facts.\n"
        )

    def get_memory_guidance_text(self) -> str:
        """
        Get the shorter memory guidance for progressive mode injection.

        Returns:
            Brief memory guidance for progressive mode
        """
        return (
            "\n\nMemory Guidance:\n"
            "- For remember/forget requests: ask for a brief Yes/No confirmation before applying changes.\n"
            "- Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n"
            "- Never fabricate facts. If you don't find relevant information in memory, say you're not sure and ask the user.\n"
        )

    def should_inject_memory(self, memory_bullets: Optional[list] = None) -> bool:
        """
        Determine if memory context should be injected based on configuration.

        Args:
            memory_bullets: List of memory bullets (if available)

        Returns:
            True if memory should be injected
        """
        if not self.progressive_mode:
            # Legacy mode: always inject memory instructions
            return True

        # Progressive mode: only inject if we have memory content
        return memory_bullets is not None and len(memory_bullets) > 0

    def get_config_summary(self) -> dict:
        """
        Get a summary of current configuration for debugging/logging.

        Returns:
            Dictionary with current configuration values
        """
        return {
            "progressive_mode": self.progressive_mode,
            "budget_tokens": self.budget_tokens,
            "include_summary": self.include_summary,
            "inject_role": self.inject_role,
            "max_memory_bullets": self.max_memory_bullets,
            "use_coref": self.use_coref,
            "coref_mode": self.coref_mode,
            "confidence_threshold": self.confidence_threshold,
            "retrieval_fusion": self.retrieval_fusion,
            "use_leann": self.use_leann,
        }

    def validate(self) -> None:
        """
        Validate configuration for logical consistency.

        Raises:
            ValidationError: If configuration is invalid
        """
        errors = []

        # Budget validation
        if self.budget_tokens <= 0:
            errors.append("budget_tokens must be positive")
        elif self.budget_tokens < 512:
            logger.warning(f"budget_tokens ({self.budget_tokens}) is very small, may cause issues")

        # Score validation
        if not (0.0 <= self.min_relevance_score <= 1.0):
            errors.append("min_relevance_score must be between 0.0 and 1.0")

        if self.max_memory_bullets <= 0:
            errors.append("max_memory_bullets must be positive")

        if not (0.0 <= self.confidence_threshold <= 1.0):
            errors.append("confidence_threshold must be between 0.0 and 1.0")

        # String validation
        if not self.inject_header.strip():
            errors.append("inject_header cannot be empty")

        if self.inject_role not in ("system", "user", "assistant"):
            logger.warning(f"Unusual inject_role: {self.inject_role}, expected system/user/assistant")

        if self.coref_mode not in ("lite", "full"):
            logger.warning(f"Unknown coref_mode: {self.coref_mode}, expected lite/full")

        # Raise all errors at once
        if errors:
            raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()


# Global instance for convenience (lazy-loaded)
_global_config = None

def get_global_config() -> MemoryConfig:
    """Get a global MemoryConfig instance (singleton pattern)"""
    global _global_config
    if _global_config is None:
        _global_config = MemoryConfig.from_env()
    return _global_config


# Convenience functions for common config access patterns
def is_progressive_mode() -> bool:
    """Check if progressive mode is enabled"""
    return get_global_config().progressive_mode

def get_budget_tokens() -> int:
    """Get the total budget tokens"""
    return get_global_config().budget_tokens

def should_include_summary() -> bool:
    """Check if summaries should be included"""
    return get_global_config().include_summary