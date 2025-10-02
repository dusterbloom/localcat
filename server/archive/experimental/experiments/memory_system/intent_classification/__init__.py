"""
Intent Classification Module for LocalCat Voice Agent

This module provides intent classification services to enable smart routing
of user utterances for optimized memory processing and conversation flow.

Refactored architecture with separated concerns:
- service: Thin orchestrator (main interface)
- strategies: Centralized configuration
- cache: Performance optimization
- metrics: Monitoring and analytics
- router: Routing decisions
- exceptions: Error handling
"""

# Main service interface
from .service import IntentService, get_intent_service

# Individual components (for advanced usage)
from .strategies import get_intent_strategies, MemoryProcessingStrategy, IntentCategory
from .cache import IntentCache, cached_intent_classifier
from .metrics import IntentMetrics, get_intent_metrics
from .router import IntentRouter, get_intent_router
from .exceptions import (
    IntentClassificationError, ModelLoadError, ClassificationTimeoutError,
    LowConfidenceError, CacheLookupError, InvalidIntentError,
    IntentExceptionHandler
)

# Main exports for public API
__all__ = [
    # Primary interface
    'IntentService', 'get_intent_service',

    # Components (for advanced usage)
    'get_intent_strategies', 'MemoryProcessingStrategy', 'IntentCategory',
    'IntentCache', 'cached_intent_classifier',
    'IntentMetrics', 'get_intent_metrics',
    'IntentRouter', 'get_intent_router',

    # Exception handling
    'IntentClassificationError', 'ModelLoadError', 'ClassificationTimeoutError',
    'LowConfidenceError', 'CacheLookupError', 'InvalidIntentError',
    'IntentExceptionHandler'
]