"""
Lightweight timing tracker for retrieval pipeline.

Usage:
    tracker = TimingTracker()
    tracker.start("graph_collection")
    # ... operation ...
    tracker.end("graph_collection")

    breakdown = tracker.get_breakdown()
    # {"graph_collection": 12.5, "convo_collection": 8.3, ...}
"""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TimingTracker:
    """Thread-safe timing tracker for retrieval stages."""

    _stages: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _active: Dict[str, float] = field(default_factory=dict)
    _start_time: float = field(default_factory=time.perf_counter)

    def start(self, stage: str) -> None:
        """Start timing a stage."""
        self._active[stage] = time.perf_counter()

    def end(self, stage: str) -> float:
        """End timing a stage and return duration in ms."""
        if stage not in self._active:
            return 0.0

        duration_ms = (time.perf_counter() - self._active[stage]) * 1000
        self._stages[stage].append(duration_ms)
        del self._active[stage]
        return duration_ms

    def mark(self, stage: str) -> None:
        """Mark a point-in-time stage (duration from tracker start)."""
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self._stages[stage].append(elapsed_ms)

    def get_breakdown(self) -> Dict[str, float]:
        """Get timing breakdown (sum of all durations per stage)."""
        return {
            stage: sum(durations)
            for stage, durations in self._stages.items()
        }

    def get_total(self) -> float:
        """Get total elapsed time from tracker creation."""
        return (time.perf_counter() - self._start_time) * 1000

    def to_dict(self) -> Dict[str, any]:
        """Export to dict for logging."""
        breakdown = self.get_breakdown()
        total = self.get_total()
        return {
            "total_ms": total,
            "breakdown_ms": breakdown,
            "budget_remaining_ms": 100.0 - total,  # Against 100ms SLO
            "over_budget": total > 100.0
        }


@dataclass
class LatencyStats:
    """Aggregated latency statistics."""

    samples: List[float] = field(default_factory=list)

    def add(self, value_ms: float):
        """Add a latency sample."""
        self.samples.append(value_ms)

    def get_percentiles(self) -> Dict[str, float]:
        """Calculate P50, P95, P99."""
        if not self.samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

        try:
            import numpy as np
            return {
                "p50": float(np.percentile(self.samples, 50)),
                "p95": float(np.percentile(self.samples, 95)),
                "p99": float(np.percentile(self.samples, 99)),
                "mean": float(np.mean(self.samples))
            }
        except ImportError:
            # Fallback if numpy not available
            sorted_samples = sorted(self.samples)
            n = len(sorted_samples)
            return {
                "p50": sorted_samples[int(n * 0.50)] if n > 0 else 0.0,
                "p95": sorted_samples[int(n * 0.95)] if n > 0 else 0.0,
                "p99": sorted_samples[int(n * 0.99)] if n > 0 else 0.0,
                "mean": sum(self.samples) / n if n > 0 else 0.0
            }

    def clear(self):
        """Clear all samples."""
        self.samples.clear()
