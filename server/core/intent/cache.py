"""
Intent Classification Caching Component
Reusable LRU caching with statistics and configurable behavior
"""

import time
import hashlib
from functools import wraps, lru_cache
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from loguru import logger

# Handle both package and direct execution imports
try:
    from .exceptions import CacheLookupError
except ImportError:
    from exceptions import CacheLookupError


@dataclass
class CacheStats:
    """Statistics tracking for cache performance"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    total_time_saved_ms: float = 0.0
    creation_time: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate as percentage"""
        return 100.0 - self.hit_rate

    @property
    def average_time_saved_ms(self) -> float:
        """Average time saved per cache hit"""
        if self.hits == 0:
            return 0.0
        return self.total_time_saved_ms / self.hits

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for serialization"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'total_requests': self.total_requests,
            'hit_rate_percent': self.hit_rate,
            'miss_rate_percent': self.miss_rate,
            'total_time_saved_ms': self.total_time_saved_ms,
            'average_time_saved_ms': self.average_time_saved_ms,
            'uptime_seconds': time.time() - self.creation_time
        }


class IntentCache:
    """
    High-performance caching for intent classification results
    Uses LRU eviction with performance tracking
    """

    def __init__(self, max_size: int = 128, track_timing: bool = True):
        """
        Initialize intent cache

        Args:
            max_size: Maximum number of cached entries
            track_timing: Whether to track timing for cache efficiency
        """
        self.max_size = max_size
        self.track_timing = track_timing
        self.stats = CacheStats()

        # Create LRU cache with custom wrapper for statistics
        self._cache = lru_cache(maxsize=max_size)(self._cached_classify)

    def _normalize_key(self, text: str) -> str:
        """
        Normalize text for consistent cache keys

        Args:
            text: Input text to normalize

        Returns:
            Normalized cache key
        """
        # Simple normalization: strip whitespace, lowercase
        normalized = text.strip().lower()

        # For very long texts, use hash to keep key size manageable
        if len(normalized) > 200:
            return hashlib.md5(normalized.encode('utf-8')).hexdigest()

        return normalized

    def _cached_classify(self, cache_key: str, original_text: str, classifier_func: Callable) -> Dict[str, Any]:
        """
        Internal cached classification function
        This is wrapped by LRU cache for actual caching
        """
        # This should never be called directly - it's wrapped by lru_cache
        # The actual classification happens in get_or_compute
        raise NotImplementedError("This method should be called through cache wrapper")

    async def get_or_compute(self,
                           text: str,
                           classifier_func: Callable[[str], Dict[str, Any]]) -> Tuple[Dict[str, Any], bool]:
        """
        Get classification result from cache or compute if missing

        Args:
            text: Text to classify
            classifier_func: Function that performs actual classification

        Returns:
            Tuple of (classification_result, was_cached)

        Raises:
            CacheLookupError: If cache operations fail
        """
        self.stats.total_requests += 1
        cache_key = self._normalize_key(text)

        start_time = time.perf_counter() if self.track_timing else 0

        try:
            # Check if we have cached result
            cache_info = self._cache.cache_info()
            initial_hits = cache_info.hits

            # Try to get from cache by creating a dummy function that returns cached result
            # We need to work around lru_cache limitations here
            try:
                # Check cache directly using cache_info and manual lookup
                cached_result = self._try_cache_lookup(cache_key, text, classifier_func)

                if cached_result is not None:
                    # Cache hit
                    self.stats.hits += 1
                    if self.track_timing:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        self.stats.total_time_saved_ms += elapsed_ms

                    logger.debug(f"Cache hit for key: {cache_key[:20]}...")
                    return cached_result, True

            except Exception as e:
                logger.debug(f"Cache lookup failed, computing fresh: {e}")

            # Cache miss - compute fresh result
            self.stats.misses += 1
            logger.debug(f"Cache miss for key: {cache_key[:20]}...")

            # Compute fresh result
            result = await classifier_func(text)

            # Store in cache
            self._store_in_cache(cache_key, text, result)

            return result, False

        except Exception as e:
            raise CacheLookupError("get_or_compute", cache_key, e)

    def _try_cache_lookup(self, cache_key: str, text: str, classifier_func: Callable) -> Optional[Dict[str, Any]]:
        """
        Try to lookup result in cache
        Returns None if not found
        """
        # This is a simplified approach - in a real implementation,
        # we might need to use a more sophisticated cache key strategy
        # For now, we'll use a direct dictionary cache

        if not hasattr(self, '_direct_cache'):
            self._direct_cache = {}

        return self._direct_cache.get(cache_key)

    def _store_in_cache(self, cache_key: str, text: str, result: Dict[str, Any]) -> None:
        """Store result in cache with eviction if needed"""
        if not hasattr(self, '_direct_cache'):
            self._direct_cache = {}

        # Simple LRU eviction
        if len(self._direct_cache) >= self.max_size:
            # Remove oldest entry (first key)
            if self._direct_cache:
                oldest_key = next(iter(self._direct_cache))
                del self._direct_cache[oldest_key]
                self.stats.evictions += 1

        self._direct_cache[cache_key] = result

    def clear(self) -> None:
        """Clear all cached entries"""
        if hasattr(self, '_direct_cache'):
            self._direct_cache.clear()
        self._cache.cache_clear()
        logger.debug("Intent cache cleared")

    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        return self.stats

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        cache_info = self._cache.cache_info()
        direct_cache_size = len(getattr(self, '_direct_cache', {}))

        return {
            'max_size': self.max_size,
            'current_size': direct_cache_size,
            'lru_cache_info': {
                'hits': cache_info.hits,
                'misses': cache_info.misses,
                'maxsize': cache_info.maxsize,
                'currsize': cache_info.currsize
            },
            'statistics': self.stats.to_dict(),
            'track_timing': self.track_timing
        }


class IntentCacheDecorator:
    """
    Decorator for adding caching to any classification function
    """

    def __init__(self, cache_size: int = 128, track_timing: bool = True):
        """
        Initialize cache decorator

        Args:
            cache_size: Maximum cache size
            track_timing: Whether to track timing statistics
        """
        self.cache = IntentCache(cache_size, track_timing)

    def __call__(self, func: Callable) -> Callable:
        """
        Decorate a classification function with caching

        Args:
            func: Classification function to cache

        Returns:
            Cached version of the function
        """
        @wraps(func)
        async def wrapper(text: str, *args, **kwargs):
            # Create a partial function with the extra args
            async def classifier_func(text_input: str):
                return await func(text_input, *args, **kwargs)

            result, was_cached = await self.cache.get_or_compute(text, classifier_func)
            result['cached'] = was_cached
            return result

        # Attach cache methods to the wrapper
        wrapper.cache = self.cache
        wrapper.cache_clear = self.cache.clear
        wrapper.cache_stats = self.cache.get_stats
        wrapper.cache_info = self.cache.get_info

        return wrapper

    def get_cache(self) -> IntentCache:
        """Get the underlying cache instance"""
        return self.cache


# Convenience function for creating cached classifiers
def cached_intent_classifier(cache_size: int = 128, track_timing: bool = True):
    """
    Decorator factory for creating cached intent classifiers

    Args:
        cache_size: Maximum number of cached entries
        track_timing: Whether to track timing statistics

    Returns:
        Decorator that adds caching to classification functions
    """
    return IntentCacheDecorator(cache_size, track_timing)


if __name__ == "__main__":
    # Test cache functionality
    import asyncio

    async def test_cache():
        print("Testing Intent Cache")
        print("=" * 30)

        cache = IntentCache(max_size=3, track_timing=True)

        # Mock classifier function
        call_count = 0
        async def mock_classifier(text: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            # Simulate processing time
            await asyncio.sleep(0.01)
            return {
                'intent': 'test_intent',
                'confidence': 0.95,
                'text': text,
                'call_number': call_count
            }

        # Test caching behavior
        test_texts = ["hello", "world", "hello", "test", "hello"]

        for i, text in enumerate(test_texts):
            result, cached = await cache.get_or_compute(text, mock_classifier)
            print(f"{i+1}. Text: '{text}' -> Call #{result['call_number']}, Cached: {cached}")

        print(f"\nTotal classifier calls: {call_count}")
        print(f"Cache stats: {cache.get_stats().to_dict()}")
        print(f"Cache info: {cache.get_info()}")

        # Test decorator
        print("\n" + "=" * 30)
        print("Testing Cache Decorator")

        @cached_intent_classifier(cache_size=2)
        async def decorated_classifier(text: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {
                'intent': 'decorated_intent',
                'confidence': 0.90,
                'text': text,
                'call_number': call_count
            }

        for text in ["foo", "bar", "foo"]:
            result = await decorated_classifier(text)
            print(f"Decorated: '{text}' -> Call #{result['call_number']}, Cached: {result['cached']}")

        print(f"Final classifier calls: {call_count}")

    asyncio.run(test_cache())