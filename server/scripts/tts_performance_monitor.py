#!/usr/bin/env python3
"""
TTS Performance Monitoring Script
Real-time monitoring and analysis of TTS latency optimizations.
"""

import json
import time
import asyncio
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import sys
import os
from pathlib import Path

# Add server root to path
server_root = Path(__file__).parent.parent
sys.path.insert(0, str(server_root))

@dataclass
class TTSPerformanceMetrics:
    """TTS performance metrics data structure."""
    timestamp: datetime
    ttfb_ms: float
    chunk_size: int
    target_chunk: int
    generation_time_ms: float
    audio_bytes: int
    text_length: int
    processing_time_ms: float = 0.0
    
class TTSPerformanceMonitor:
    """Real-time TTS performance monitoring and analysis."""
    
    def __init__(self, alert_threshold_ms: float = 800.0):
        self.metrics_history: List[TTSPerformanceMetrics] = []
        self.alert_threshold_ms = alert_threshold_ms
        self.start_time = datetime.now()
        
        # Performance targets
        self.target_avg_ttfb = 600.0
        self.target_max_ttfb = 800.0
        self.target_variance = 100.0
        
    def record_metric(self, metric_data: Dict[str, Any]):
        """Record a TTS performance metric."""
        metric = TTSPerformanceMetrics(
            timestamp=datetime.now(),
            ttfb_ms=metric_data.get('ttfb_ms', 0.0),
            chunk_size=metric_data.get('bytes', 0),
            target_chunk=metric_data.get('target_chunk', 0),
            generation_time_ms=metric_data.get('total_ms', 0.0),
            audio_bytes=metric_data.get('audio_bytes', 0),
            text_length=metric_data.get('text_length', 0),
            processing_time_ms=metric_data.get('processing_time_ms', 0.0)
        )
        
        self.metrics_history.append(metric)
        
        # Check for performance alerts
        if metric.ttfb_ms > self.alert_threshold_ms:
            self._trigger_alert(metric)
    
    def _trigger_alert(self, metric: TTSPerformanceMetrics):
        """Trigger performance alert for high latency."""
        print(f"🚨 PERFORMANCE ALERT: TTFB {metric.ttfb_ms:.1f}ms > {self.alert_threshold_ms}ms")
        print(f"   Timestamp: {metric.timestamp}")
        print(f"   Chunk size: {metric.chunk_size} bytes (target: {metric.target_chunk})")
        print(f"   Text length: {metric.text_length} chars")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Calculate current performance statistics."""
        if not self.metrics_history:
            return {"status": "No data available"}
        
        recent_metrics = [m for m in self.metrics_history 
                         if m.timestamp > datetime.now() - timedelta(minutes=5)]
        
        if not recent_metrics:
            recent_metrics = self.metrics_history[-10:]  # Last 10 metrics
        
        ttfb_values = [m.ttfb_ms for m in recent_metrics if m.ttfb_ms > 0]
        chunk_sizes = [m.chunk_size for m in recent_metrics if m.chunk_size > 0]
        
        if not ttfb_values:
            return {"status": "No valid TTFB data"}
        
        stats = {
            "time_period": "5 minutes" if len(recent_metrics) < len(self.metrics_history) else "all time",
            "total_requests": len(self.metrics_history),
            "recent_requests": len(recent_metrics),
            "ttfb": {
                "avg": statistics.mean(ttfb_values),
                "median": statistics.median(ttfb_values),
                "min": min(ttfb_values),
                "max": max(ttfb_values),
                "p95": self._percentile(ttfb_values, 95),
                "p99": self._percentile(ttfb_values, 99),
                "variance": statistics.pvariance(ttfb_values) ** 0.5 if len(ttfb_values) > 1 else 0,
                "std_dev": statistics.stdev(ttfb_values) if len(ttfb_values) > 1 else 0
            },
            "chunk_size": {
                "avg": statistics.mean(chunk_sizes) if chunk_sizes else 0,
                "min": min(chunk_sizes) if chunk_sizes else 0,
                "max": max(chunk_sizes) if chunk_sizes else 0,
                "range": max(chunk_sizes) - min(chunk_sizes) if chunk_sizes else 0
            }
        }
        
        # Add target compliance
        stats["targets_met"] = self._check_targets(stats)
        
        return stats
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _check_targets(self, stats: Dict[str, Any]) -> Dict[str, bool]:
        """Check if performance targets are being met."""
        ttfb_stats = stats.get("ttfb", {})
        
        return {
            "avg_ttfb": ttfb_stats.get("avg", float('inf')) <= self.target_avg_ttfb,
            "max_ttfb": ttfb_stats.get("max", float('inf')) <= self.target_max_ttfb,
            "variance": ttfb_stats.get("variance", float('inf')) <= self.target_variance,
            "p95_ttfb": ttfb_stats.get("p95", float('inf')) <= self.target_max_ttfb
        }
    
    def print_dashboard(self):
        """Print real-time performance dashboard."""
        stats = self.get_current_stats()
        
        if "status" in stats:
            print(f"📊 TTS Performance Monitor: {stats['status']}")
            return
        
        print("\n" + "="*60)
        print("🚀 TTS PERFORMANCE DASHBOARD")
        print("="*60)
        
        ttfb = stats["ttfb"]
        targets = stats["targets_met"]
        
        print(f"📈 Time-to-First-Byte (TTFB) Metrics:")
        print(f"   Average: {ttfb['avg']:.1f}ms {'✅' if targets['avg_ttfb'] else '❌'}")
        print(f"   Median:  {ttfb['median']:.1f}ms")
        print(f"   P95:     {ttfb['p95']:.1f}ms {'✅' if targets['p95_ttfb'] else '❌'}")
        print(f"   P99:     {ttfb['p99']:.1f}ms")
        print(f"   Max:     {ttfb['max']:.1f}ms {'✅' if targets['max_ttfb'] else '❌'}")
        print(f"   Variance: {ttfb['variance']:.1f}ms {'✅' if targets['variance'] else '❌'}")
        
        chunk = stats["chunk_size"]
        print(f"\n📦 Chunk Size Metrics:")
        print(f"   Average: {chunk['avg']:.0f} bytes")
        print(f"   Range:   {chunk['min']:.0f} - {chunk['max']:.0f} bytes")
        print(f"   Target:  2KB - 4KB range")
        
        print(f"\n📊 Session Summary:")
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Recent Period:  {stats['time_period']}")
        print(f"   Monitor Runtime: {(datetime.now() - self.start_time).total_seconds():.1f}s")
        
        print(f"\n🎯 Target Compliance: {sum(targets.values())}/{len(targets)} met")
        for target, met in targets.items():
            status = "✅" if met else "❌"
            print(f"   {target}: {status}")
    
    def export_report(self, filename: str = None):
        """Export performance report to JSON file."""
        if not filename:
            filename = f"tts_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "monitoring_period": {
                "start": self.start_time.isoformat(),
                "end": datetime.now().isoformat()
            },
            "configuration": {
                "alert_threshold_ms": self.alert_threshold_ms,
                "target_avg_ttfb": self.target_avg_ttfb,
                "target_max_ttfb": self.target_max_ttfb,
                "target_variance": self.target_variance
            },
            "current_stats": self.get_current_stats(),
            "all_metrics": [asdict(m) for m in self.metrics_history]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Performance report exported to: {filename}")

async def simulate_tts_load_test(monitor: TTSPerformanceMonitor, duration_seconds: int = 60):
    """Simulate TTS load testing with realistic data."""
    print(f"🧪 Starting {duration_seconds}s load test...")
    
    test_phrases = [
        "Hello",
        "How are you today?",
        "This is a test of the voice system",
        "The quick brown fox jumps over the lazy dog",
        "I am ready to assist you with your request",
        "Please tell me what you need help with",
        "This system provides ultra-low latency text-to-speech",
        "We have optimized the performance significantly"
    ]
    
    start_time = time.time()
    request_count = 0
    
    while time.time() - start_time < duration_seconds:
        # Simulate request with realistic latency
        phrase = test_phrases[request_count % len(test_phrases)]
        
        # Simulate optimized TTFB (should be better than original)
        base_latency = 500 + len(phrase) * 2
        noise = (hash(phrase + str(request_count)) % 100) - 50
        ttfb = base_latency + noise
        
        # Simulate chunk size (2KB-4KB range)
        target_chunk = 2048 + (len(phrase) * 20)
        target_chunk = min(max(target_chunk, 2048), 4096)
        
        chunk_size = target_chunk + ((request_count * 17) % 500) - 250
        chunk_size = min(max(chunk_size, 1024), 4500)
        
        metric_data = {
            "ttfb_ms": max(ttfb, 100),  # Ensure positive
            "bytes": chunk_size,
            "target_chunk": target_chunk,
            "total_ms": ttfb + 50,
            "audio_bytes": int(chunk_size * 0.9),
            "text_length": len(phrase),
            "processing_time_ms": ttfb * 0.1
        }
        
        monitor.record_metric(metric_data)
        request_count += 1
        
        # Simulate request rate (2-5 requests per second)
        await asyncio.sleep(0.2 + (request_count % 3) * 0.1)
    
    print(f"✅ Load test completed: {request_count} requests simulated")

async def main():
    """Main performance monitoring function."""
    print("🚀 Starting TTS Performance Monitor")
    print("="*50)
    
    monitor = TTSPerformanceMonitor(alert_threshold_ms=800.0)
    
    # Simulate some initial data
    print("📊 Simulating initial performance data...")
    await simulate_tts_load_test(monitor, 30)
    
    # Show dashboard
    monitor.print_dashboard()
    
    # Export report
    monitor.export_report()

if __name__ == "__main__":
    asyncio.run(main())
