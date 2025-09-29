"""
Coreference Integration for HotPath Memory Processor

Provides factory functions to create coreference-enabled memory processing
without breaking existing code. Follows the factory pattern for clean integration.
"""

from typing import Optional, List
from loguru import logger

from .processors.coreference import CoreferenceProcessor
from .processors.base import TextProcessor
from .extractors.ud import UDExtractor
from .config import get_memory_config, MemoryConfig


def create_coreference_processor(config: Optional[MemoryConfig] = None) -> Optional[CoreferenceProcessor]:
    """
    Create a CoreferenceProcessor based on configuration.

    Args:
        config: Memory configuration (uses global config if None)

    Returns:
        CoreferenceProcessor instance if enabled, None otherwise
    """
    if config is None:
        config = get_memory_config()

    if not config.coreference.enabled:
        logger.debug("Coreference resolution disabled in configuration")
        return None

    if not config.processors.enabled:
        logger.warning("Coreference enabled but processors disabled - skipping coreference")
        return None

    try:
        processor = CoreferenceProcessor(
            timeout_ms=config.coreference.timeout_ms,
            min_text_length=config.coreference.min_text_length,
            lang=config.coreference.lang
        )
        logger.info(f"Created CoreferenceProcessor with {config.coreference.timeout_ms}ms timeout")
        return processor

    except Exception as e:
        logger.warning(f"Failed to create CoreferenceProcessor: {e}")
        if config.coreference.fallback_enabled:
            logger.info("Continuing without coreference resolution (fallback enabled)")
            return None
        else:
            raise


def create_enhanced_ud_extractor(host, config: Optional[MemoryConfig] = None) -> UDExtractor:
    """
    Create a UDExtractor with optional coreference processing.

    Args:
        host: Host object providing extraction methods
        config: Memory configuration (uses global config if None)

    Returns:
        UDExtractor instance with optional text processing
    """
    if config is None:
        config = get_memory_config()

    text_processors: List[TextProcessor] = []

    # Add coreference processor if enabled
    coreference_processor = create_coreference_processor(config)
    if coreference_processor is not None:
        text_processors.append(coreference_processor)
        logger.info("Added coreference resolution to extraction pipeline")

    # Future processors can be added here
    # e.g., named entity normalization, spell correction, etc.

    if text_processors:
        logger.info(f"Created UDExtractor with {len(text_processors)} text processors")
        return UDExtractor(host, text_processors=text_processors)
    else:
        logger.debug("Created standard UDExtractor (no text processing)")
        return UDExtractor(host)


def get_coreference_metrics(extractor: UDExtractor) -> dict:
    """
    Get coreference processing metrics from an extractor.

    Args:
        extractor: UDExtractor instance

    Returns:
        Dictionary with coreference metrics or empty dict if not available
    """
    try:
        metrics = extractor.get_processor_metrics()
        for metric in metrics:
            if metric.get("processor") == "coreference":
                return metric
        return {"error": "No coreference processor found"}

    except Exception as e:
        logger.warning(f"Failed to get coreference metrics: {e}")
        return {"error": str(e)}


def log_coreference_status(config: Optional[MemoryConfig] = None) -> None:
    """
    Log the current coreference resolution status for debugging.

    Args:
        config: Memory configuration (uses global config if None)
    """
    if config is None:
        config = get_memory_config()

    logger.info("=== Coreference Resolution Status ===")
    logger.info(f"Coreference enabled: {config.coreference.enabled}")
    logger.info(f"Processors enabled: {config.processors.enabled}")

    if config.coreference.enabled:
        logger.info(f"Timeout: {config.coreference.timeout_ms}ms")
        logger.info(f"Min text length: {config.coreference.min_text_length}")
        logger.info(f"Language: {config.coreference.lang}")
        logger.info(f"Fallback enabled: {config.coreference.fallback_enabled}")

        # Test if spacy-coref is available
        try:
            from .nlp_manager import get_nlp_with_coref
            nlp = get_nlp_with_coref(config.coreference.lang)
            if nlp is not None:
                logger.info("✅ spacy-coref model loaded successfully")
            else:
                logger.warning("❌ Failed to load spacy-coref model")
        except Exception as e:
            logger.warning(f"❌ spacy-coref not available: {e}")

    logger.info("=====================================")


# Convenience function for hotpath processor integration
def should_use_coreference(config: Optional[MemoryConfig] = None) -> bool:
    """
    Check if coreference resolution should be used.

    Args:
        config: Memory configuration (uses global config if None)

    Returns:
        True if coreference should be enabled
    """
    if config is None:
        config = get_memory_config()

    return (
        config.coreference.enabled and
        config.processors.enabled and
        config.enabled  # Overall memory system enabled
    )