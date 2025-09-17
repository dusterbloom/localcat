"""
Memory Operation Timing Tracer
Provides detailed timing metrics for memory write/update/retrieve operations
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict, deque
import statistics
from loguru import logger


@dataclass
class MemoryOperationTiming:
    """Single memory operation timing measurement"""
    operation: str  # 'write', 'update', 'retrieve', 'flush'
    component: str  # 'store', 'session_store', 'retriever', 'facade'
    duration_ms: float
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    turn_id: int = 0


@dataclass
class MemoryTimingStats:
    """Aggregated timing statistics for memory operations"""
    operation: str
    component: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    recent_timings: deque = field(default_factory=lambda: deque(maxlen=100))


class MemoryTimingTracer:
    """High-performance memory operation timing tracer"""

    def __init__(self, max_measurements: int = 1000):
        self.max_measurements = max_measurements
        self.measurements: List[MemoryOperationTiming] = []
        self.stats: Dict[str, MemoryTimingStats] = {}
        self.active_operations: Dict[str, float] = {}
        self.lock = threading.RLock()

        # Performance thresholds (ms)
        self.thresholds = {
            'write': 50,     # Single entity/edge write
            'update': 30,    # Entity/edge update
            'retrieve': 100, # Memory retrieval
            'flush': 200,    # Batch flush operations
            'session_write': 25,   # Session message storage
            'session_update': 20,  # Session updates
            'session_retrieve': 50, # Session context retrieval
        }

    def start_operation(self, operation: str, component: str,
                       session_id: str = "", turn_id: int = 0,
                       details: Optional[Dict[str, Any]] = None) -> str:
        """Start timing a memory operation"""
        op_id = f"{component}::{operation}::{time.time()}"

        with self.lock:
            self.active_operations[op_id] = time.perf_counter()

        logger.debug(f"⏱️ START {component}.{operation} [id={op_id[-8:]}]")
        return op_id

    def end_operation(self, op_id: str, details: Optional[Dict[str, Any]] = None) -> Optional[MemoryOperationTiming]:
        """End timing and record measurement"""
        end_time = time.perf_counter()

        with self.lock:
            start_time = self.active_operations.pop(op_id, None)
            if start_time is None:
                logger.warning(f"⚠️ Unknown operation ID: {op_id}")
                return None

            # Parse operation details
            parts = op_id.split("::")
            if len(parts) < 2:
                return None
            component, operation = parts[0], parts[1]

            duration_ms = (end_time - start_time) * 1000

            measurement = MemoryOperationTiming(
                operation=operation,
                component=component,
                duration_ms=duration_ms,
                timestamp=end_time,
                details=details or {},
                session_id=details.get('session_id', '') if details else '',
                turn_id=details.get('turn_id', 0) if details else 0
            )

            # Store measurement
            self.measurements.append(measurement)
            if len(self.measurements) > self.max_measurements:
                self.measurements.pop(0)

            # Update stats
            self._update_stats(measurement)

            # Check thresholds
            threshold_key = f"{component}_{operation}" if f"{component}_{operation}" in self.thresholds else operation
            threshold = self.thresholds.get(threshold_key, self.thresholds.get(operation, 200))

            status = "🔥" if duration_ms > threshold * 2 else "⚠️" if duration_ms > threshold else "✅"

            logger.info(f"{status} {component}.{operation}: {duration_ms:.1f}ms (threshold: {threshold}ms)")

            return measurement

    @contextmanager
    def time_operation(self, operation: str, component: str,
                      session_id: str = "", turn_id: int = 0,
                      details: Optional[Dict[str, Any]] = None):
        """Context manager for timing operations"""
        op_id = self.start_operation(operation, component, session_id, turn_id, details)
        try:
            yield op_id
        finally:
            self.end_operation(op_id, details)

    def _update_stats(self, measurement: MemoryOperationTiming):
        """Update aggregated statistics"""
        key = f"{measurement.component}.{measurement.operation}"

        if key not in self.stats:
            self.stats[key] = MemoryTimingStats(
                operation=measurement.operation,
                component=measurement.component
            )

        stats = self.stats[key]
        stats.count += 1
        stats.total_ms += measurement.duration_ms
        stats.min_ms = min(stats.min_ms, measurement.duration_ms)
        stats.max_ms = max(stats.max_ms, measurement.duration_ms)
        stats.mean_ms = stats.total_ms / stats.count

        # Update recent timings for percentiles
        stats.recent_timings.append(measurement.duration_ms)

        # Calculate percentiles
        if stats.recent_timings:
            sorted_timings = sorted(stats.recent_timings)
            n = len(sorted_timings)
            stats.p50_ms = sorted_timings[int(n * 0.5)]
            stats.p95_ms = sorted_timings[int(n * 0.95)]
            stats.p99_ms = sorted_timings[int(n * 0.99)]

    def get_stats(self, component: Optional[str] = None) -> Dict[str, MemoryTimingStats]:
        """Get current statistics"""
        with self.lock:
            if component:
                return {k: v for k, v in self.stats.items() if v.component == component}
            return self.stats.copy()

    def get_operation_summary(self) -> Dict[str, Any]:
        """Get operation performance summary"""
        with self.lock:
            summary = {
                "total_measurements": len(self.measurements),
                "active_operations": len(self.active_operations),
                "operation_breakdown": {},
                "component_breakdown": {},
                "slow_operations": [],
                "recommendations": []
            }

            # Group by operation type
            operation_groups = defaultdict(list)
            component_groups = defaultdict(list)

            for stat in self.stats.values():
                operation_groups[stat.operation].append(stat)
                component_groups[stat.component].append(stat)

            # Summarize by operation
            for op, stats_list in operation_groups.items():
                total_calls = sum(s.count for s in stats_list)
                avg_p95 = sum(s.p95_ms * s.count for s in stats_list) / total_calls if total_calls > 0 else 0

                summary["operation_breakdown"][op] = {
                    "total_calls": total_calls,
                    "avg_p95_ms": avg_p95,
                    "components": len(stats_list)
                }

            # Summarize by component
            for comp, stats_list in component_groups.items():
                total_calls = sum(s.count for s in stats_list)
                avg_p95 = sum(s.p95_ms * s.count for s in stats_list) / total_calls if total_calls > 0 else 0

                summary["component_breakdown"][comp] = {
                    "total_calls": total_calls,
                    "avg_p95_ms": avg_p95,
                    "operations": len(stats_list)
                }

            # Identify slow operations
            for key, stat in self.stats.items():
                threshold_key = f"{stat.component}_{stat.operation}"
                threshold = self.thresholds.get(threshold_key, self.thresholds.get(stat.operation, 200))

                if stat.p95_ms > threshold * 1.5:
                    summary["slow_operations"].append({
                        "operation": key,
                        "p95_ms": stat.p95_ms,
                        "threshold_ms": threshold,
                        "slowdown_factor": stat.p95_ms / threshold,
                        "call_count": stat.count
                    })

            # Generate recommendations
            summary["recommendations"] = self._generate_recommendations(summary)

            return summary

    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        for slow_op in summary["slow_operations"]:
            factor = slow_op["slowdown_factor"]
            op_name = slow_op["operation"]

            if "write" in op_name and factor > 3:
                recommendations.append(f"🔥 Optimize {op_name}: {factor:.1f}x slower than target")
            elif "retrieve" in op_name and factor > 2:
                recommendations.append(f"⚠️ Consider caching for {op_name}: {factor:.1f}x slower")
            elif "flush" in op_name and factor > 2:
                recommendations.append(f"💡 Batch operations for {op_name}: {factor:.1f}x slower")

        return recommendations

    def print_report(self, component: Optional[str] = None, top_n: int = 15):
        """Print comprehensive memory timing report"""
        print("\n" + "="*80)
        print("🗄️ MEMORY OPERATION TIMING REPORT")
        print("="*80)

        summary = self.get_operation_summary()

        print(f"\n📊 OVERVIEW:")
        print(f"   Total measurements: {summary['total_measurements']}")
        print(f"   Active operations: {summary['active_operations']}")

        # Operation breakdown
        print(f"\n🎯 OPERATION BREAKDOWN:")
        for op, data in sorted(summary["operation_breakdown"].items(),
                              key=lambda x: x[1]["avg_p95_ms"], reverse=True):
            print(f"   {op:15} {data['total_calls']:6} calls  avg_p95={data['avg_p95_ms']:6.1f}ms  ({data['components']} components)")

        # Component breakdown
        print(f"\n🏗️ COMPONENT BREAKDOWN:")
        for comp, data in sorted(summary["component_breakdown"].items(),
                                key=lambda x: x[1]["avg_p95_ms"], reverse=True):
            print(f"   {comp:15} {data['total_calls']:6} calls  avg_p95={data['avg_p95_ms']:6.1f}ms  ({data['operations']} operations)")

        # Slow operations
        if summary["slow_operations"]:
            print(f"\n🚨 SLOW OPERATIONS:")
            for slow in summary["slow_operations"]:
                print(f"   🔥 {slow['operation']:35} {slow['p95_ms']:6.1f}ms ({slow['slowdown_factor']:.1f}x slower)")

        # Detailed stats
        stats = self.get_stats(component)
        if stats:
            print(f"\n📈 DETAILED STATISTICS {'('+component+')' if component else ''}:")
            sorted_stats = sorted(stats.items(), key=lambda x: x[1].p95_ms, reverse=True)

            for i, (key, stat) in enumerate(sorted_stats[:top_n]):
                status = "🔥" if stat.p95_ms > 200 else "⚠️" if stat.p95_ms > 50 else "✅"
                print(f"   {status} {key:35} {stat.count:4}x  "
                      f"p95={stat.p95_ms:6.1f}ms  mean={stat.mean_ms:6.1f}ms  "
                      f"min={stat.min_ms:6.1f}ms  max={stat.max_ms:6.1f}ms")

        # Recommendations
        if summary["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in summary["recommendations"]:
                print(f"   {rec}")

        print("="*80)

    def export_measurements(self, filepath: str):
        """Export measurements to JSON"""
        import json

        with self.lock:
            data = {
                "measurements": [
                    {
                        "operation": m.operation,
                        "component": m.component,
                        "duration_ms": m.duration_ms,
                        "timestamp": m.timestamp,
                        "details": m.details,
                        "session_id": m.session_id,
                        "turn_id": m.turn_id
                    }
                    for m in self.measurements
                ],
                "summary": self.get_operation_summary()
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"📁 Exported {len(self.measurements)} measurements to {filepath}")


# Global tracer instance
_global_memory_tracer: Optional[MemoryTimingTracer] = None

def get_memory_tracer() -> MemoryTimingTracer:
    """Get global memory timing tracer"""
    global _global_memory_tracer
    if _global_memory_tracer is None:
        _global_memory_tracer = MemoryTimingTracer()
    return _global_memory_tracer


# Decorator for timing methods
def time_memory_operation(operation: str, component: str = None):
    """Decorator to time memory operations"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            tracer = get_memory_tracer()
            comp = component or self.__class__.__name__

            with tracer.time_operation(operation, comp):
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    tracer = MemoryTimingTracer()

    # Simulate memory operations
    print("🧪 Testing memory timing tracer...")

    # Simulate various operations
    operations = [
        ("write", "store", 15, 45),
        ("update", "store", 10, 35),
        ("retrieve", "retriever", 50, 150),
        ("flush", "store", 100, 300),
        ("session_write", "session_store", 5, 25),
        ("session_retrieve", "session_store", 20, 80),
    ]

    import random

    for _ in range(50):
        op, comp, min_ms, max_ms = random.choice(operations)
        duration = random.uniform(min_ms, max_ms) / 1000

        with tracer.time_operation(op, comp, "test_session", 1):
            time.sleep(duration)

    # Print report
    tracer.print_report()