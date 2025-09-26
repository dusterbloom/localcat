"""
Intent Classification Metrics Component
Performance tracking and monitoring for intent classification operations
"""

import time
import statistics
from collections import deque, defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Deque
from loguru import logger


@dataclass
class ClassificationMetric:
    """Single classification metric entry"""
    timestamp: float
    text_length: int
    intent: str
    confidence: float
    processing_time_ms: float
    cached: bool
    fallback: bool
    model_name: str = ""


@dataclass
class PerformanceWindow:
    """Performance metrics for a specific time window"""
    start_time: float
    end_time: float
    total_classifications: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    cache_hit_rate: float
    fallback_rate: float
    confidence_avg: float
    most_common_intents: Dict[str, int]


class IntentMetrics:
    """
    Comprehensive metrics tracking for intent classification
    Provides performance monitoring, alerting, and optimization insights
    """

    def __init__(self, max_history: int = 10000, window_size_minutes: int = 5):
        """
        Initialize metrics tracking

        Args:
            max_history: Maximum number of classification records to keep
            window_size_minutes: Size of rolling performance windows
        """
        self.max_history = max_history
        self.window_size_seconds = window_size_minutes * 60

        # Classification history
        self.history: Deque[ClassificationMetric] = deque(maxlen=max_history)

        # Real-time counters
        self.total_classifications = 0
        self.total_cache_hits = 0
        self.total_fallbacks = 0
        self.total_processing_time_ms = 0.0

        # Performance tracking
        self.latency_samples: Deque[float] = deque(maxlen=1000)  # Last 1000 samples for percentiles
        self.confidence_samples: Deque[float] = deque(maxlen=1000)

        # Intent distribution tracking
        self.intent_counts = Counter()
        self.intent_avg_latency = defaultdict(list)
        self.intent_avg_confidence = defaultdict(list)

        # Performance windows
        self.performance_windows: List[PerformanceWindow] = []
        self.last_window_time = time.time()

        # Alerting thresholds
        self.latency_alert_threshold_ms = 1000.0  # Alert if p95 > 1000ms
        self.confidence_alert_threshold = 0.5    # Alert if avg confidence < 50%
        self.fallback_rate_alert_threshold = 0.2  # Alert if fallback rate > 20%

        logger.debug(f"Intent metrics initialized with {max_history} max history, {window_size_minutes}min windows")

    def record_classification(self,
                            text: str,
                            intent: str,
                            confidence: float,
                            processing_time_ms: float,
                            cached: bool = False,
                            fallback: bool = False,
                            model_name: str = "") -> None:
        """
        Record a classification event

        Args:
            text: Input text that was classified
            intent: Predicted intent
            confidence: Classification confidence score
            processing_time_ms: Time taken for classification
            cached: Whether result came from cache
            fallback: Whether this was a fallback result
            model_name: Name of the model used
        """
        timestamp = time.time()

        # Create metric record
        metric = ClassificationMetric(
            timestamp=timestamp,
            text_length=len(text),
            intent=intent,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            cached=cached,
            fallback=fallback,
            model_name=model_name
        )

        # Add to history
        self.history.append(metric)

        # Update counters
        self.total_classifications += 1
        if cached:
            self.total_cache_hits += 1
        if fallback:
            self.total_fallbacks += 1
        self.total_processing_time_ms += processing_time_ms

        # Update samples for percentile calculations
        self.latency_samples.append(processing_time_ms)
        if not fallback:  # Only track confidence for real predictions
            self.confidence_samples.append(confidence)

        # Update intent-specific metrics
        self.intent_counts[intent] += 1
        self.intent_avg_latency[intent].append(processing_time_ms)
        self.intent_avg_confidence[intent].append(confidence)

        # Check if we need to create a new performance window
        if timestamp - self.last_window_time >= self.window_size_seconds:
            self._create_performance_window()
            self.last_window_time = timestamp

        # Check for alerts
        self._check_alerts()

    def _create_performance_window(self) -> None:
        """Create a performance window for the current period"""
        if not self.history:
            return

        current_time = time.time()
        window_start = current_time - self.window_size_seconds

        # Get metrics from the current window
        window_metrics = [m for m in self.history if m.timestamp >= window_start]

        if not window_metrics:
            return

        # Calculate window statistics
        latencies = [m.processing_time_ms for m in window_metrics]
        confidences = [m.confidence for m in window_metrics if not m.fallback]
        cached_count = sum(1 for m in window_metrics if m.cached)
        fallback_count = sum(1 for m in window_metrics if m.fallback)

        window = PerformanceWindow(
            start_time=window_start,
            end_time=current_time,
            total_classifications=len(window_metrics),
            avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0.0),
            p99_latency_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else (max(latencies) if latencies else 0.0),
            cache_hit_rate=cached_count / len(window_metrics) if window_metrics else 0.0,
            fallback_rate=fallback_count / len(window_metrics) if window_metrics else 0.0,
            confidence_avg=statistics.mean(confidences) if confidences else 0.0,
            most_common_intents=dict(Counter(m.intent for m in window_metrics).most_common(5))
        )

        self.performance_windows.append(window)

        # Keep only recent windows (last 24 hours worth)
        max_windows = int(24 * 60 / (self.window_size_seconds / 60))
        if len(self.performance_windows) > max_windows:
            self.performance_windows = self.performance_windows[-max_windows:]

        logger.debug(f"Created performance window: {len(window_metrics)} classifications, "
                    f"avg latency: {window.avg_latency_ms:.1f}ms, cache hit rate: {window.cache_hit_rate:.1%}")

    def _check_alerts(self) -> None:
        """Check for performance alerts based on current metrics"""
        if self.total_classifications < 10:  # Need minimum data for reliable alerts
            return

        # Check latency alert
        if len(self.latency_samples) >= 20:
            p95_latency = statistics.quantiles(list(self.latency_samples), n=20)[18]
            if p95_latency > self.latency_alert_threshold_ms:
                logger.warning(f"High latency alert: P95 latency {p95_latency:.1f}ms > {self.latency_alert_threshold_ms}ms")

        # Check confidence alert
        if len(self.confidence_samples) >= 10:
            avg_confidence = statistics.mean(self.confidence_samples)
            if avg_confidence < self.confidence_alert_threshold:
                logger.warning(f"Low confidence alert: Average confidence {avg_confidence:.3f} < {self.confidence_alert_threshold:.3f}")

        # Check fallback rate alert
        fallback_rate = self.total_fallbacks / self.total_classifications
        if fallback_rate > self.fallback_rate_alert_threshold:
            logger.warning(f"High fallback rate alert: {fallback_rate:.1%} > {self.fallback_rate_alert_threshold:.1%}")

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if self.total_classifications == 0:
            return {"status": "no_data"}

        cache_hit_rate = self.total_cache_hits / self.total_classifications
        fallback_rate = self.total_fallbacks / self.total_classifications
        avg_latency = self.total_processing_time_ms / self.total_classifications

        stats = {
            'total_classifications': self.total_classifications,
            'cache_hit_rate': cache_hit_rate,
            'fallback_rate': fallback_rate,
            'avg_latency_ms': avg_latency,
        }

        # Add percentile information if we have enough samples
        if len(self.latency_samples) >= 2:
            latencies = list(self.latency_samples)
            stats.update({
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'median_latency_ms': statistics.median(latencies),
            })

            if len(latencies) >= 20:
                quantiles = statistics.quantiles(latencies, n=20)
                stats.update({
                    'p95_latency_ms': quantiles[18],
                    'p99_latency_ms': quantiles[19] if len(latencies) >= 100 else quantiles[-1]
                })

        # Add confidence statistics
        if self.confidence_samples:
            confidences = list(self.confidence_samples)
            stats.update({
                'avg_confidence': statistics.mean(confidences),
                'min_confidence': min(confidences),
                'max_confidence': max(confidences),
            })

        # Add intent distribution
        stats['intent_distribution'] = dict(self.intent_counts.most_common(10))

        return stats

    def get_intent_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed analysis per intent"""
        analysis = {}

        for intent in self.intent_counts:
            latencies = self.intent_avg_latency[intent]
            confidences = self.intent_avg_confidence[intent]

            analysis[intent] = {
                'count': self.intent_counts[intent],
                'avg_latency_ms': statistics.mean(latencies) if latencies else 0.0,
                'avg_confidence': statistics.mean(confidences) if confidences else 0.0,
                'min_latency_ms': min(latencies) if latencies else 0.0,
                'max_latency_ms': max(latencies) if latencies else 0.0,
            }

        return analysis

    def get_performance_trend(self, hours: int = 1) -> List[PerformanceWindow]:
        """Get performance trend for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        return [w for w in self.performance_windows if w.end_time >= cutoff_time]

    def get_optimization_suggestions(self) -> List[str]:
        """Get performance optimization suggestions based on metrics"""
        suggestions = []

        if self.total_classifications < 10:
            return ["Insufficient data for optimization suggestions"]

        # Cache optimization
        cache_hit_rate = self.total_cache_hits / self.total_classifications
        if cache_hit_rate < 0.3:
            suggestions.append(f"Low cache hit rate ({cache_hit_rate:.1%}). Consider increasing cache size or improving key normalization.")

        # Latency optimization
        if len(self.latency_samples) >= 20:
            p95_latency = statistics.quantiles(list(self.latency_samples), n=20)[18]
            if p95_latency > 100:
                suggestions.append(f"High P95 latency ({p95_latency:.1f}ms). Consider using a faster model or optimizing inference.")

        # Confidence optimization
        if len(self.confidence_samples) >= 10:
            avg_confidence = statistics.mean(self.confidence_samples)
            if avg_confidence < 0.7:
                suggestions.append(f"Low average confidence ({avg_confidence:.3f}). Consider retraining model or adjusting confidence threshold.")

        # Fallback rate optimization
        fallback_rate = self.total_fallbacks / self.total_classifications
        if fallback_rate > 0.1:
            suggestions.append(f"High fallback rate ({fallback_rate:.1%}). Consider improving model training data or adjusting thresholds.")

        # Intent distribution analysis
        total_intents = len(self.intent_counts)
        if total_intents > 0:
            most_common_intent_count = max(self.intent_counts.values())
            if most_common_intent_count / self.total_classifications > 0.8:
                dominant_intent = max(self.intent_counts, key=self.intent_counts.get)
                suggestions.append(f"Intent '{dominant_intent}' dominates ({most_common_intent_count/self.total_classifications:.1%}). Consider rebalancing training data.")

        return suggestions if suggestions else ["Performance looks good! No optimization suggestions at this time."]

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing or clean starts)"""
        self.history.clear()
        self.total_classifications = 0
        self.total_cache_hits = 0
        self.total_fallbacks = 0
        self.total_processing_time_ms = 0.0
        self.latency_samples.clear()
        self.confidence_samples.clear()
        self.intent_counts.clear()
        self.intent_avg_latency.clear()
        self.intent_avg_confidence.clear()
        self.performance_windows.clear()
        self.last_window_time = time.time()
        logger.info("Intent metrics reset")


# Global metrics instance for easy access
_metrics_instance = None

def get_intent_metrics() -> IntentMetrics:
    """Get or create the global intent metrics instance"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = IntentMetrics()
    return _metrics_instance


if __name__ == "__main__":
    # Test metrics functionality
    print("Testing Intent Metrics")
    print("=" * 30)

    metrics = IntentMetrics(max_history=100, window_size_minutes=1)

    # Simulate some classifications
    import random

    intents = ["remember_fact", "general_chat", "recall_query", "greeting"]

    for i in range(50):
        intent = random.choice(intents)
        confidence = random.uniform(0.6, 0.95)
        latency = random.uniform(10, 200)
        cached = random.random() < 0.3  # 30% cache hit rate
        fallback = random.random() < 0.05  # 5% fallback rate

        metrics.record_classification(
            text=f"test message {i}",
            intent=intent,
            confidence=confidence,
            processing_time_ms=latency,
            cached=cached,
            fallback=fallback,
            model_name="test-model"
        )

    print("Current Stats:")
    stats = metrics.get_current_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nIntent Analysis:")
    analysis = metrics.get_intent_analysis()
    for intent, data in analysis.items():
        print(f"  {intent}: {data['count']} calls, {data['avg_latency_ms']:.1f}ms avg, {data['avg_confidence']:.3f} confidence")

    print("\nOptimization Suggestions:")
    suggestions = metrics.get_optimization_suggestions()
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")