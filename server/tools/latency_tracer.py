#!/usr/bin/env python3
"""
LATENCY TRACER - Comprehensive pipeline latency analysis tool
Traces latency throughout the entire voice pipeline from STT to TTS
"""

import time
import asyncio
import logging
import json
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import threading
import os
import sys

# Add server path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class LatencyMeasurement:
    """Single latency measurement"""
    component: str
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    thread_id: str = ""
    session_id: str = ""
    turn_id: int = 0

@dataclass
class LatencyStats:
    """Aggregated latency statistics"""
    component: str
    operation: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    measurements: List[float] = field(default_factory=list)

class LatencyTracer:
    """High-performance latency tracer for voice pipeline"""

    def __init__(self, max_measurements: int = 1000, enable_detailed_logging: bool = True):
        self.max_measurements = max_measurements
        self.enable_detailed_logging = enable_detailed_logging
        self.measurements: List[LatencyMeasurement] = []
        self.active_traces: Dict[str, float] = {}
        self.stats: Dict[str, LatencyStats] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger("LatencyTracer")

        # Pipeline stage tracking
        self.pipeline_stages = [
            "vad_detection",
            "speech_to_text",
            "intent_classification",
            "memory_extraction",
            "memory_retrieval",
            "llm_inference",
            "text_to_speech",
            "audio_output"
        ]

        # Critical thresholds (ms)
        self.thresholds = {
            "vad_detection": 50,
            "speech_to_text": 500,
            "intent_classification": 100,  # Rule-based target
            "memory_extraction": 200,
            "memory_retrieval": 50,
            "llm_inference": 2000,
            "text_to_speech": 1000,
            "audio_output": 100,
            "total_pipeline": 3000  # End-to-end target
        }

    def start_trace(self, component: str, operation: str,
                   session_id: str = "", turn_id: int = 0,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start timing a component operation"""
        trace_id = f"{component}::{operation}::{time.time()}"

        with self.lock:
            self.active_traces[trace_id] = time.perf_counter()

        if self.enable_detailed_logging:
            self.logger.debug(f"⏱️  START {component}.{operation} [trace_id={trace_id}]")

        return trace_id

    def end_trace(self, trace_id: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[LatencyMeasurement]:
        """End timing and record measurement"""
        end_time = time.perf_counter()

        with self.lock:
            start_time = self.active_traces.pop(trace_id, None)
            if start_time is None:
                self.logger.warning(f"⚠️  Unknown trace_id: {trace_id}")
                return None

            # Parse trace_id
            parts = trace_id.split("::")
            if len(parts) < 2:
                return None
            component, operation = parts[0], parts[1]

            duration_ms = (end_time - start_time) * 1000

            measurement = LatencyMeasurement(
                component=component,
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                metadata=metadata or {},
                thread_id=str(threading.get_ident()),
                session_id=metadata.get('session_id', '') if metadata else '',
                turn_id=metadata.get('turn_id', 0) if metadata else 0
            )

            # Store measurement
            self.measurements.append(measurement)
            if len(self.measurements) > self.max_measurements:
                self.measurements.pop(0)  # Remove oldest

            # Update stats
            self._update_stats(measurement)

            # Check thresholds
            threshold = self.thresholds.get(component, self.thresholds.get(operation, float('inf')))
            status = "🔥" if duration_ms > threshold * 2 else "⚠️" if duration_ms > threshold else "✅"

            if self.enable_detailed_logging:
                self.logger.info(f"{status} END {component}.{operation}: {duration_ms:.1f}ms (threshold: {threshold}ms)")

            return measurement

    @asynccontextmanager
    async def trace_async(self, component: str, operation: str,
                         session_id: str = "", turn_id: int = 0,
                         metadata: Optional[Dict[str, Any]] = None):
        """Async context manager for tracing"""
        trace_id = self.start_trace(component, operation, session_id, turn_id, metadata)
        try:
            yield trace_id
        finally:
            self.end_trace(trace_id, metadata)

    def trace_sync(self, component: str, operation: str,
                  session_id: str = "", turn_id: int = 0,
                  metadata: Optional[Dict[str, Any]] = None):
        """Sync context manager for tracing"""
        class SyncTraceContext:
            def __init__(self, tracer, comp, op, sid, tid, meta):
                self.tracer = tracer
                self.component = comp
                self.operation = op
                self.session_id = sid
                self.turn_id = tid
                self.metadata = meta
                self.trace_id = None

            def __enter__(self):
                self.trace_id = self.tracer.start_trace(
                    self.component, self.operation,
                    self.session_id, self.turn_id, self.metadata
                )
                return self.trace_id

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.trace_id:
                    self.tracer.end_trace(self.trace_id, self.metadata)

        return SyncTraceContext(self, component, operation, session_id, turn_id, metadata)

    def _update_stats(self, measurement: LatencyMeasurement):
        """Update aggregated statistics"""
        key = f"{measurement.component}.{measurement.operation}"

        if key not in self.stats:
            self.stats[key] = LatencyStats(
                component=measurement.component,
                operation=measurement.operation
            )

        stats = self.stats[key]
        stats.count += 1
        stats.total_ms += measurement.duration_ms
        stats.min_ms = min(stats.min_ms, measurement.duration_ms)
        stats.max_ms = max(stats.max_ms, measurement.duration_ms)
        stats.mean_ms = stats.total_ms / stats.count

        # Keep rolling window for percentiles
        stats.measurements.append(measurement.duration_ms)
        if len(stats.measurements) > 100:  # Keep last 100 measurements
            stats.measurements.pop(0)

        # Calculate percentiles
        if stats.measurements:
            sorted_measurements = sorted(stats.measurements)
            n = len(sorted_measurements)
            stats.p50_ms = sorted_measurements[int(n * 0.5)]
            stats.p95_ms = sorted_measurements[int(n * 0.95)]
            stats.p99_ms = sorted_measurements[int(n * 0.99)]

    def get_stats(self, component: Optional[str] = None) -> Dict[str, LatencyStats]:
        """Get current statistics"""
        with self.lock:
            if component:
                return {k: v for k, v in self.stats.items() if v.component == component}
            return self.stats.copy()

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get end-to-end pipeline performance summary"""
        with self.lock:
            summary = {
                "total_measurements": len(self.measurements),
                "active_traces": len(self.active_traces),
                "stage_performance": {},
                "bottlenecks": [],
                "recommendations": []
            }

            # Analyze each pipeline stage
            for stage in self.pipeline_stages:
                stage_stats = [v for k, v in self.stats.items() if stage in k.lower()]
                if stage_stats:
                    # Aggregate stats for this stage
                    total_p95 = max(s.p95_ms for s in stage_stats)
                    total_mean = sum(s.mean_ms for s in stage_stats) / len(stage_stats)
                    threshold = self.thresholds.get(stage, float('inf'))

                    summary["stage_performance"][stage] = {
                        "p95_ms": total_p95,
                        "mean_ms": total_mean,
                        "threshold_ms": threshold,
                        "status": "critical" if total_p95 > threshold * 2 else "warning" if total_p95 > threshold else "good",
                        "operations": len(stage_stats)
                    }

                    # Identify bottlenecks
                    if total_p95 > threshold * 1.5:
                        summary["bottlenecks"].append({
                            "stage": stage,
                            "p95_ms": total_p95,
                            "threshold_ms": threshold,
                            "slowdown_factor": total_p95 / threshold
                        })

            # Generate recommendations
            summary["recommendations"] = self._generate_recommendations(summary)

            return summary

    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        for bottleneck in summary["bottlenecks"]:
            stage = bottleneck["stage"]
            factor = bottleneck["slowdown_factor"]

            if stage == "intent_classification" and factor > 10:
                recommendations.append(f"🔥 CRITICAL: Switch from SOTA to rule-based classifier ({factor:.1f}x slower)")
            elif stage == "memory_extraction" and factor > 2:
                recommendations.append(f"⚠️ Consider optimizing extraction pipeline ({factor:.1f}x slower)")
            elif stage == "llm_inference" and factor > 1.5:
                recommendations.append(f"💡 Consider smaller/faster LLM model ({factor:.1f}x slower)")
            elif stage == "text_to_speech" and factor > 1.5:
                recommendations.append(f"🎵 Consider faster TTS model or streaming ({factor:.1f}x slower)")

        return recommendations

    def print_report(self, component: Optional[str] = None, top_n: int = 10):
        """Print comprehensive latency report"""
        print("\n" + "="*80)
        print("🚀 LOCALCAT VOICE PIPELINE LATENCY REPORT")
        print("="*80)

        # Pipeline summary
        summary = self.get_pipeline_summary()
        print(f"\n📊 PIPELINE OVERVIEW:")
        print(f"   Total measurements: {summary['total_measurements']}")
        print(f"   Active traces: {summary['active_traces']}")

        # Stage performance
        print(f"\n🎯 STAGE PERFORMANCE:")
        for stage, perf in summary["stage_performance"].items():
            status_emoji = {"critical": "🔥", "warning": "⚠️", "good": "✅"}[perf["status"]]
            print(f"   {status_emoji} {stage:20} p95={perf['p95_ms']:6.1f}ms  mean={perf['mean_ms']:6.1f}ms  (threshold: {perf['threshold_ms']}ms)")

        # Bottlenecks
        if summary["bottlenecks"]:
            print(f"\n🚨 BOTTLENECKS:")
            for bottleneck in summary["bottlenecks"]:
                print(f"   🔥 {bottleneck['stage']:20} {bottleneck['p95_ms']:6.1f}ms ({bottleneck['slowdown_factor']:.1f}x slower)")

        # Recommendations
        if summary["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in summary["recommendations"]:
                print(f"   {rec}")

        # Detailed stats
        stats = self.get_stats(component)
        if stats:
            print(f"\n📈 DETAILED STATISTICS {'('+component+')' if component else ''}:")
            sorted_stats = sorted(stats.items(), key=lambda x: x[1].p95_ms, reverse=True)

            for i, (key, stat) in enumerate(sorted_stats[:top_n]):
                status = "🔥" if stat.p95_ms > 500 else "⚠️" if stat.p95_ms > 100 else "✅"
                print(f"   {status} {key:35} {stat.count:4}x  "
                      f"p95={stat.p95_ms:6.1f}ms  mean={stat.mean_ms:6.1f}ms  "
                      f"min={stat.min_ms:6.1f}ms  max={stat.max_ms:6.1f}ms")

        print("="*80)

    def export_measurements(self, filepath: str):
        """Export measurements to JSON for analysis"""
        with self.lock:
            data = {
                "measurements": [
                    {
                        "component": m.component,
                        "operation": m.operation,
                        "duration_ms": m.duration_ms,
                        "timestamp": m.start_time,
                        "metadata": m.metadata,
                        "session_id": m.session_id,
                        "turn_id": m.turn_id
                    }
                    for m in self.measurements
                ],
                "stats": {
                    k: {
                        "component": v.component,
                        "operation": v.operation,
                        "count": v.count,
                        "mean_ms": v.mean_ms,
                        "p95_ms": v.p95_ms,
                        "p99_ms": v.p99_ms
                    }
                    for k, v in self.stats.items()
                },
                "pipeline_summary": self.get_pipeline_summary()
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"📁 Exported {len(self.measurements)} measurements to {filepath}")

# Global tracer instance
_global_tracer: Optional[LatencyTracer] = None

def get_tracer() -> LatencyTracer:
    """Get global tracer instance"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = LatencyTracer()
    return _global_tracer

def trace_function(component: str, operation: Optional[str] = None):
    """Decorator to trace function execution"""
    def decorator(func):
        op_name = operation or func.__name__

        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                async with tracer.trace_async(component, op_name):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.trace_sync(component, op_name):
                    return func(*args, **kwargs)
            return sync_wrapper

    return decorator

# Example usage and testing
if __name__ == "__main__":
    import random

    # Create tracer
    tracer = LatencyTracer(enable_detailed_logging=True)

    # Test sync tracing
    print("🧪 Testing latency tracer...")

    # Simulate pipeline stages
    stages = [
        ("vad_detection", 20, 30),
        ("speech_to_text", 300, 600),
        ("intent_classification", 10, 2000),  # Simulate SOTA vs rule-based
        ("memory_extraction", 100, 1000),
        ("memory_retrieval", 20, 50),
        ("llm_inference", 1000, 3000),
        ("text_to_speech", 500, 2000),
        ("audio_output", 50, 150)
    ]

    # Simulate 50 pipeline runs
    for turn in range(50):
        session_id = f"session_{turn // 10}"

        for stage, min_ms, max_ms in stages:
            # Simulate processing time
            sim_time = random.uniform(min_ms, max_ms) / 1000

            with tracer.trace_sync(stage, "process", session_id, turn):
                time.sleep(sim_time)

    # Print report
    tracer.print_report()

    # Export data
    tracer.export_measurements("/tmp/latency_trace.json")