"""
Intent Service Wrapper for LocalCat Voice Agent
Provides lightweight intent classification for smart memory processing

Based on the integration guide, adapted for FastIntentClassifier instead of Rasa.
"""

import os
import time
from typing import Optional, Dict, Any, List
from functools import lru_cache
from loguru import logger
import asyncio

# Handle both package and direct execution imports
try:
    from ..fast_intent_classifier import FastIntentClassifier
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from fast_intent_classifier import FastIntentClassifier


class IntentService:
    """
    Intent classification service wrapper
    Follows singleton pattern from the integration guide
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 confidence_threshold: float = 0.7,
                 cache_size: int = 128):
        """
        Initialize intent service

        Args:
            model_name: Hugging Face model name
            confidence_threshold: Minimum confidence for classification
            cache_size: Size of LRU cache for repeated phrases
        """
        self.model_name = model_name or os.getenv("INTENT_MODEL", "Falconsai/intent_classification")
        self.confidence_threshold = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", str(confidence_threshold)))
        self.enabled = os.getenv("INTENT_CLASSIFICATION_ENABLED", "true").lower() == "true"

        # Performance settings
        self.log_classification_time = os.getenv("INTENT_LOG_CLASSIFICATION_TIME", "false").lower() == "true"
        self.log_routing_decisions = os.getenv("INTENT_LOG_ROUTING_DECISIONS", "false").lower() == "true"

        # Initialize classifier
        self.classifier = None
        if self.enabled:
            try:
                self.classifier = FastIntentClassifier(self.model_name)
                logger.info(f"Intent service initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize intent classifier: {e}")
                self.enabled = False

        # LRU cache for repeated classifications
        self._cached_classify = lru_cache(maxsize=cache_size)(self._classify_uncached)

        # Performance metrics
        self.metrics = {
            'total_classifications': 0,
            'cache_hits': 0,
            'avg_latency_ms': 0.0,
            'fallback_count': 0
        }

    async def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify intent for given text with caching

        Returns:
            {
                'intent': str,
                'confidence': float,
                'fallback': bool,
                'processing_time_ms': float,
                'cached': bool
            }
        """
        if not self.enabled or not self.classifier:
            return self._fallback_result()

        start_time = time.perf_counter()

        # Try cache first
        text_normalized = text.strip().lower()
        try:
            result = await self._cached_classify(text_normalized, text)
            result['cached'] = True
            self.metrics['cache_hits'] += 1
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            result = await self._classify_uncached(text_normalized, text)
            result['cached'] = False

        # Record metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result['processing_time_ms'] = elapsed_ms

        self._update_metrics(elapsed_ms, result['fallback'])

        if self.log_classification_time:
            cache_status = "cached" if result.get('cached') else "fresh"
            logger.debug(f"Intent classification ({cache_status}): {result['intent']} "
                        f"({result['confidence']:.2f}) in {elapsed_ms:.1f}ms")

        return result

    async def _classify_uncached(self, text_normalized: str, original_text: str) -> Dict[str, Any]:
        """Perform actual classification without caching"""
        try:
            result = self.classifier.predict(original_text, self.confidence_threshold)

            # Map to standardized format
            return {
                'intent': result['intent'],
                'confidence': result['confidence'],
                'fallback': result.get('low_confidence', False),
                'model_label': result.get('model_label', ''),
                'inference_time_ms': result['inference_time_ms']
            }
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._fallback_result()

    def _fallback_result(self) -> Dict[str, Any]:
        """Return safe fallback result"""
        return {
            'intent': 'general_chat',
            'confidence': 0.0,
            'fallback': True,
            'processing_time_ms': 0.0,
            'cached': False
        }

    def get_intent_categories(self) -> Dict[str, List[str]]:
        """
        Get intent categories for routing decisions
        Following the guide's categorization (lines 380-385)
        """
        return {
            'memory_operations': ['remember_fact', 'recall_query', 'forget_request', 'memory_check'],
            'conversational': ['general_chat', 'greeting', 'goodbye', 'affirmation', 'negation'],
            'clarification': ['clarification', 'correction', 'continuation'],
            'skip_memory': self._get_skip_memory_intents(),
            'capability_queries': ['capability_query']
        }

    def _get_skip_memory_intents(self) -> List[str]:
        """Get intents that should skip memory processing"""
        default_skip = ['general_chat', 'greeting', 'goodbye', 'affirmation', 'negation']
        env_skip = os.getenv("INTENT_SKIP_MEMORY_FOR", "")

        if env_skip:
            return [intent.strip() for intent in env_skip.split(',') if intent.strip()]
        return default_skip

    def should_skip_memory_processing(self, intent: str) -> bool:
        """Check if intent should skip memory processing"""
        skip_intents = self.get_intent_categories()['skip_memory']
        should_skip = intent in skip_intents

        if should_skip and self.log_routing_decisions:
            logger.info(f"[Intent Routing] Skipping memory processing for intent: {intent}")

        return should_skip

    def get_memory_processing_strategy(self, intent: str) -> str:
        """
        Get memory processing strategy based on intent
        Following the guide's smart routing approach
        """
        strategies = {
            'remember_fact': 'storage_focused',
            'recall_query': 'retrieval_focused',
            'forget_request': 'deletion_focused',
            'memory_check': 'lookup_focused',
            'general_chat': 'minimal',
            'clarification': 'contextual',
            'correction': 'recent_context',
            'continuation': 'contextual',
            'greeting': 'skip',
            'goodbye': 'skip',
            'affirmation': 'minimal',
            'negation': 'minimal',
            'capability_query': 'standard'
        }

        strategy = strategies.get(intent, 'standard')

        if self.log_routing_decisions:
            logger.debug(f"[Intent Routing] Strategy for '{intent}': {strategy}")

        return strategy

    def _update_metrics(self, latency_ms: float, is_fallback: bool):
        """Update performance metrics"""
        self.metrics['total_classifications'] += 1

        # Update rolling average
        total = self.metrics['total_classifications']
        current_avg = self.metrics['avg_latency_ms']
        self.metrics['avg_latency_ms'] = (current_avg * (total - 1) + latency_ms) / total

        if is_fallback:
            self.metrics['fallback_count'] += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total = self.metrics['total_classifications']
        if total == 0:
            return {'status': 'no_data'}

        cache_hit_rate = self.metrics['cache_hits'] / total if total > 0 else 0
        fallback_rate = self.metrics['fallback_count'] / total if total > 0 else 0

        return {
            'total_classifications': total,
            'avg_latency_ms': self.metrics['avg_latency_ms'],
            'cache_hit_rate': cache_hit_rate,
            'fallback_rate': fallback_rate,
            'model': self.model_name,
            'enabled': self.enabled
        }

    def clear_cache(self):
        """Clear the classification cache"""
        self._cached_classify.cache_clear()
        logger.debug("Intent classification cache cleared")


# Singleton instance following guide pattern (lines 388-395)
_intent_service = None

def get_intent_service() -> IntentService:
    """Get or create intent service singleton"""
    global _intent_service
    if _intent_service is None:
        _intent_service = IntentService()
    return _intent_service


# Convenience function for testing
async def test_intent_service():
    """Test intent service functionality"""
    service = get_intent_service()

    test_cases = [
        "Remember that I like coffee",
        "What did I tell you about my job?",
        "How are you doing today?",
        "Forget what I said about that",
        "Hello there",
        "Yes that's correct",
        "What can you help me with?"
    ]

    print("Testing Intent Service:")
    print("-" * 60)

    for text in test_cases:
        result = await service.classify_intent(text)
        strategy = service.get_memory_processing_strategy(result['intent'])
        skip_memory = service.should_skip_memory_processing(result['intent'])

        print(f"Text: {text}")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
        print(f"Strategy: {strategy} | Skip memory: {skip_memory}")
        print(f"Time: {result['processing_time_ms']:.2f}ms | Cached: {result.get('cached', False)}")
        print()

    print("Performance Stats:")
    stats = service.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_intent_service())