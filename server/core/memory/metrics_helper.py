"""
Memory Metrics Helper

Simple utilities to time extraction/retrieval/injection phases and aggregate percentiles.
Used by performance tests only, no runtime dependency.
"""

import time
import statistics
from typing import List, Dict, Any, Optional, Callable, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Result of a timing measurement"""
    operation: str
    duration_ms: float
    metadata: Optional[Dict[str, Any]] = None


class MetricsCollector:
    """Collects and aggregates timing metrics"""
    
    def __init__(self):
        self.measurements: List[TimingResult] = []
        self._start_time: Optional[float] = None
    
    def start_timing(self, operation: str) -> None:
        """Start timing an operation"""
        self._start_time = time.perf_counter()
        self._current_operation = operation
    
    def end_timing(self, metadata: Optional[Dict[str, Any]] = None) -> float:
        """End timing and record the result"""
        if self._start_time is None:
            raise ValueError("Must call start_timing() before end_timing()")
        
        duration_ms = (time.perf_counter() - self._start_time) * 1000
        
        result = TimingResult(
            operation=self._current_operation,
            duration_ms=duration_ms,
            metadata=metadata
        )
        
        self.measurements.append(result)
        self._start_time = None
        
        return duration_ms
    
    def add_measurement(self, operation: str, duration_ms: float, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a measurement directly"""
        result = TimingResult(
            operation=operation,
            duration_ms=duration_ms,
            metadata=metadata
        )
        self.measurements.append(result)
    
    @contextmanager
    def time_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for timing operations"""
        self.start_timing(operation)
        try:
            yield
        finally:
            self.end_timing(metadata)
    
    def get_measurements(self, operation: Optional[str] = None) -> List[float]:
        """Get all measurements for an operation (or all if None)"""
        if operation is None:
            return [m.duration_ms for m in self.measurements]
        
        return [m.duration_ms for m in self.measurements if m.operation == operation]
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, float]:
        """Get statistics for measurements"""
        measurements = self.get_measurements(operation)
        
        if not measurements:
            return {}
        
        measurements_sorted = sorted(measurements)
        n = len(measurements)
        
        stats = {
            'count': n,
            'mean_ms': statistics.mean(measurements),
            'median_ms': statistics.median(measurements),
            'min_ms': min(measurements),
            'max_ms': max(measurements),
            'std_ms': statistics.stdev(measurements) if n > 1 else 0.0
        }
        
        # Calculate percentiles
        for p in [50, 75, 90, 95, 99]:
            idx = int(p / 100 * (n - 1))
            stats[f'p{p}_ms'] = measurements_sorted[idx]
        
        return stats
    
    def print_summary(self, operation: Optional[str] = None) -> None:
        """Print a summary of measurements"""
        if operation:
            stats = self.get_stats(operation)
            if stats:
                print(f"\n=== {operation} Performance Summary ===")
                print(f"Count: {stats['count']}")
                print(f"Mean: {stats['mean_ms']:.2f}ms")
                print(f"Median: {stats['median_ms']:.2f}ms")
                print(f"P50: {stats['p50_ms']:.2f}ms")
                print(f"P90: {stats['p90_ms']:.2f}ms")
                print(f"P95: {stats['p95_ms']:.2f}ms")
                print(f"P99: {stats['p99_ms']:.2f}ms")
                print(f"Min: {stats['min_ms']:.2f}ms")
                print(f"Max: {stats['max_ms']:.2f}ms")
                print(f"Std: {stats['std_ms']:.2f}ms")
            else:
                print(f"No measurements found for operation: {operation}")
        else:
            # Print summary for all operations
            operations = set(m.operation for m in self.measurements)
            for op in sorted(operations):
                self.print_summary(op)
    
    def clear(self) -> None:
        """Clear all measurements"""
        self.measurements.clear()
        self._start_time = None


def time_function(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Time a function call and return (result, duration_ms)
    
    Args:
        func: Function to call
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Tuple of (function_result, duration_in_ms)
    """
    start_time = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        return result, (time.perf_counter() - start_time) * 1000
    except Exception as e:
        duration = (time.perf_counter() - start_time) * 1000
        raise e


def benchmark_multiple_runs(func: Callable, runs: int = 10, 
                           warmup_runs: int = 3) -> Dict[str, float]:
    """
    Benchmark a function over multiple runs with warmup
    
    Args:
        func: Function to benchmark (should take no args)
        runs: Number of timed runs
        warmup_runs: Number of warmup runs (not timed)
    
    Returns:
        Dictionary with statistics
    """
    collector = MetricsCollector()
    
    # Warmup runs
    for _ in range(warmup_runs):
        try:
            func()
        except Exception:
            pass  # Ignore warmup errors
    
    # Timed runs
    for i in range(runs):
        try:
            with collector.time_operation("benchmark_run"):
                func()
        except Exception as e:
            logger.warning(f"Benchmark run {i} failed: {e}")
    
    return collector.get_stats("benchmark_run")


# Test corpus for consistent measurements
TEST_SENTENCES = [
    "I live in New York City.",
    "My name is John Smith.",
    "I work as a software engineer.",
    "I have two brothers and one sister.",
    "My favorite color is blue.",
    "I graduated from Stanford University.",
    "I drive a Tesla Model 3.",
    "My phone number is 555-1234.",
    "I was born in 1990.",
    "I enjoy hiking and photography."
]

TEST_RETRIEVAL_QUERIES = [
    "Where do I live?",
    "What is my name?",
    "What is my job?",
    "Do I have siblings?",
    "What is my favorite color?"
]
