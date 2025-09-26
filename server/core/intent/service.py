"""
Refactored Intent Service - Thin Orchestrator
Follows Single Responsibility Principle by delegating to focused components
"""

import os
import time
from typing import Dict, Any, Optional
from loguru import logger

# Handle both package and direct execution imports
try:
    from ..fast_intent_classifier import FastIntentClassifier
    from .strategies import get_intent_strategies
    from .cache import IntentCache
    from .metrics import IntentMetrics, get_intent_metrics
    from .router import IntentRouter, get_intent_router
    from .exceptions import (
        IntentClassificationError, ModelLoadError, ClassificationTimeoutError,
        IntentExceptionHandler, safe_classify_with_fallback
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from fast_intent_classifier import FastIntentClassifier
    from strategies import get_intent_strategies
    from cache import IntentCache
    from metrics import IntentMetrics, get_intent_metrics
    from router import IntentRouter, get_intent_router
    from exceptions import (
        IntentClassificationError, ModelLoadError, ClassificationTimeoutError,
        IntentExceptionHandler, safe_classify_with_fallback
    )


class IntentService:
    """
    Thin orchestrator for intent classification
    Delegates to focused components following Single Responsibility Principle
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 confidence_threshold: float = 0.7,
                 cache_size: int = 128,
                 enable_metrics: bool = True,
                 enable_caching: bool = True):
        """
        Initialize intent service

        Args:
            model_name: Hugging Face model name
            confidence_threshold: Minimum confidence for classification
            cache_size: Size of LRU cache
            enable_metrics: Whether to track performance metrics
            enable_caching: Whether to enable result caching
        """
        self.model_name = model_name or os.getenv("INTENT_MODEL", "Falconsai/intent_classification")
        self.confidence_threshold = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", str(confidence_threshold)))
        self.enabled = os.getenv("INTENT_CLASSIFICATION_ENABLED", "true").lower() == "true"

        # Initialize components
        self.classifier: Optional[FastIntentClassifier] = None
        self.cache: Optional[IntentCache] = None
        self.metrics: IntentMetrics = get_intent_metrics() if enable_metrics else None
        self.router: IntentRouter = get_intent_router()

        # Configuration
        self.enable_metrics = enable_metrics
        self.enable_caching = enable_caching and self.enabled
        self.cache_size = cache_size

        # Initialize if enabled
        if self.enabled:
            self._initialize_components()

        logger.info(f"Intent service initialized: enabled={self.enabled}, model={self.model_name}, "
                   f"caching={self.enable_caching}, metrics={self.enable_metrics}")

    def _initialize_components(self) -> None:
        """Initialize all service components"""
        try:
            # Initialize classifier
            self.classifier = FastIntentClassifier(self.model_name)
            self.classifier.initialize()

            # Initialize cache if enabled
            if self.enable_caching:
                self.cache = IntentCache(max_size=self.cache_size, track_timing=True)

            logger.debug("Intent service components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize intent service components: {e}")
            self.enabled = False
            raise ModelLoadError(self.model_name, e)

    async def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify intent for given text

        Args:
            text: Text to classify

        Returns:
            Classification result with routing information

        Raises:
            IntentClassificationError: If classification fails
        """
        if not self.enabled or not self.classifier:
            return self._get_fallback_result("Service disabled or not initialized")

        start_time = time.perf_counter()

        try:
            # Use cache if available
            if self.cache:
                result, was_cached = await self.cache.get_or_compute(text, self._classify_uncached)
                result['cached'] = was_cached
            else:
                result = await self._classify_uncached(text)
                result['cached'] = False

            # Add timing information
            total_time_ms = (time.perf_counter() - start_time) * 1000
            result['total_processing_time_ms'] = total_time_ms

            # Get routing decision
            routing_decision = self.router.make_routing_decision(
                intent=result['intent'],
                confidence=result['confidence'],
                text_context=text,
                fallback=result.get('fallback', False)
            )

            # Add routing information to result
            result.update({
                'strategy': routing_decision.strategy.value,
                'skip_memory': routing_decision.skip_memory,
                'category': routing_decision.category.value,
                'routing_reasoning': routing_decision.reasoning
            })

            # Record metrics
            if self.metrics:
                self.metrics.record_classification(
                    text=text,
                    intent=result['intent'],
                    confidence=result['confidence'],
                    processing_time_ms=total_time_ms,
                    cached=result['cached'],
                    fallback=result.get('fallback', False),
                    model_name=self.model_name
                )

            return result

        except Exception as e:
            logger.error(f"Intent classification failed for text '{text[:50]}...': {e}")

            # Use exception handler for graceful fallback
            fallback_result = IntentExceptionHandler.handle_classification_error(e, text)

            # Still record metrics for failures
            if self.metrics:
                self.metrics.record_classification(
                    text=text,
                    intent=fallback_result['intent'],
                    confidence=fallback_result['confidence'],
                    processing_time_ms=fallback_result['processing_time_ms'],
                    cached=False,
                    fallback=True,
                    model_name=self.model_name
                )

            return fallback_result

    async def _classify_uncached(self, text: str) -> Dict[str, Any]:
        """Perform actual classification without caching"""
        if not self.classifier:
            raise ModelLoadError(self.model_name, None)

        try:
            # Classify with the underlying classifier
            result = self.classifier.predict(text, self.confidence_threshold)

            # Standardize result format
            return {
                'intent': result['intent'],
                'confidence': result['confidence'],
                'fallback': result.get('low_confidence', False),
                'model_label': result.get('model_label', ''),
                'inference_time_ms': result.get('inference_time_ms', 0.0)
            }

        except Exception as e:
            raise IntentClassificationError(f"Classifier prediction failed: {e}", {
                'text_length': len(text),
                'model_name': self.model_name
            })

    def _get_fallback_result(self, reason: str) -> Dict[str, Any]:
        """Get safe fallback result"""
        return {
            'intent': 'general_chat',
            'confidence': 0.0,
            'fallback': True,
            'reason': reason,
            'strategy': 'minimal',
            'skip_memory': True,
            'category': 'conversational',
            'cached': False,
            'inference_time_ms': 0.0,
            'total_processing_time_ms': 0.0
        }

    # Convenience methods for backward compatibility
    def should_skip_memory_processing(self, intent: str) -> bool:
        """Check if intent should skip memory processing"""
        return self.router.should_skip_memory_processing(intent)

    def get_memory_processing_strategy(self, intent: str) -> str:
        """Get memory processing strategy for an intent"""
        return self.router.get_memory_processing_strategy(intent)

    def get_intent_categories(self) -> Dict[str, list]:
        """Get all intent categories"""
        return self.router.get_intent_categories()

    # Service management methods
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'service_enabled': self.enabled,
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'caching_enabled': self.enable_caching,
            'metrics_enabled': self.enable_metrics
        }

        # Add classifier stats if available
        if self.classifier:
            classifier_stats = self.classifier.get_performance_stats()
            stats['classifier'] = classifier_stats

        # Add cache stats if available
        if self.cache:
            cache_info = self.cache.get_info()
            stats['cache'] = cache_info

        # Add metrics if available
        if self.metrics:
            metrics_stats = self.metrics.get_current_stats()
            stats['metrics'] = metrics_stats

        # Add router configuration
        router_summary = self.router.get_routing_summary()
        stats['routing'] = router_summary

        return stats

    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        status = {
            'healthy': True,
            'issues': [],
            'components': {}
        }

        # Check service enabled
        status['components']['service'] = {'healthy': self.enabled}
        if not self.enabled:
            status['healthy'] = False
            status['issues'].append("Intent service is disabled")

        # Check classifier
        status['components']['classifier'] = {'healthy': self.classifier is not None}
        if self.enabled and not self.classifier:
            status['healthy'] = False
            status['issues'].append("Classifier not initialized")

        # Check cache
        status['components']['cache'] = {'healthy': not self.enable_caching or self.cache is not None}
        if self.enable_caching and not self.cache:
            status['issues'].append("Cache enabled but not initialized")

        # Check router configuration
        router_issues = self.router.validate_routing_config()
        status['components']['router'] = {'healthy': len(router_issues) == 0, 'issues': router_issues}
        if router_issues:
            status['issues'].extend(router_issues)

        # Check performance if metrics available
        if self.metrics:
            metrics_stats = self.metrics.get_current_stats()
            if isinstance(metrics_stats, dict) and 'fallback_rate' in metrics_stats:
                fallback_rate = metrics_stats['fallback_rate']
                if fallback_rate > 0.5:  # More than 50% fallbacks is concerning
                    status['issues'].append(f"High fallback rate: {fallback_rate:.1%}")

        status['healthy'] = len(status['issues']) == 0
        return status

    def clear_cache(self) -> None:
        """Clear classification cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Intent classification cache cleared")

    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        if self.metrics:
            self.metrics.reset_metrics()
            logger.info("Intent classification metrics reset")

    def get_optimization_suggestions(self) -> list[str]:
        """Get performance optimization suggestions"""
        suggestions = []

        if not self.enabled:
            return ["Intent classification is disabled - consider enabling for better performance"]

        # Get suggestions from metrics
        if self.metrics:
            metrics_suggestions = self.metrics.get_optimization_suggestions()
            suggestions.extend(metrics_suggestions)

        # Add cache suggestions
        if not self.enable_caching:
            suggestions.append("Consider enabling caching to improve response times")
        elif self.cache:
            cache_info = self.cache.get_info()
            cache_stats = cache_info.get('statistics', {})
            if cache_stats.get('hit_rate_percent', 0) < 30:
                suggestions.append("Low cache hit rate - consider increasing cache size or improving key normalization")

        return suggestions if suggestions else ["Performance looks good! No optimization suggestions at this time."]


# Singleton instance following established pattern
_intent_service = None

def get_intent_service() -> IntentService:
    """Get or create intent service singleton"""
    global _intent_service
    if _intent_service is None:
        _intent_service = IntentService()
    return _intent_service


if __name__ == "__main__":
    # Test the refactored service
    import asyncio

    async def test_intent_service():
        print("Testing Refactored Intent Service")
        print("=" * 40)

        service = get_intent_service()

        # Check health
        health = service.get_health_status()
        print(f"Service Health: {'✅ Healthy' if health['healthy'] else '❌ Issues'}")
        if health['issues']:
            for issue in health['issues']:
                print(f"  - {issue}")

        print("\nTesting Classifications:")
        test_cases = [
            "Remember that I like coffee",
            "Hello how are you today?",
            "What did I tell you about my job?",
            "Goodbye see you later"
        ]

        for text in test_cases:
            result = await service.classify_intent(text)
            print(f"\nText: '{text}'")
            print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
            print(f"  Strategy: {result['strategy']}")
            print(f"  Skip memory: {result['skip_memory']}")
            print(f"  Time: {result.get('total_processing_time_ms', 0):.2f}ms")
            print(f"  Cached: {result.get('cached', False)}")

        print("\nPerformance Stats:")
        stats = service.get_performance_stats()
        if 'metrics' in stats and stats['metrics'] != {'status': 'no_data'}:
            metrics = stats['metrics']
            print(f"  Total classifications: {metrics.get('total_classifications', 0)}")
            print(f"  Average latency: {metrics.get('avg_latency_ms', 0):.2f}ms")
            print(f"  Cache hit rate: {metrics.get('cache_hit_rate', 0)*100:.1f}%")

        print("\nOptimization Suggestions:")
        suggestions = service.get_optimization_suggestions()
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")

    asyncio.run(test_intent_service())