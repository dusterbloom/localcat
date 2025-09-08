"""
Memory Metrics Module

Handles performance monitoring and metrics collection for the HotMem system.
Provides comprehensive performance tracking and reporting.

Author: SOLID Refactoring
"""

import os
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict, deque
import threading
import json
import statistics

from loguru import logger

class MemoryMetrics:
    """
    Handles performance monitoring and metrics collection for the HotMem system.
    
    Responsibilities:
    - Performance metrics collection and tracking
    - Latency monitoring and reporting
    - Resource usage monitoring
    - Health checks and alerting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Metrics configuration
        self.metrics_window = self.config.get('metrics_window', 3600)  # 1 hour
        self.health_check_interval = self.config.get('health_check_interval', 60)  # 1 minute
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'extraction_latency_ms': 500,
            'storage_latency_ms': 100,
            'retrieval_latency_ms': 200,
            'error_rate': 0.1,  # 10%
            'memory_usage_mb': 1024,  # 1GB
        })
        
        # Performance metrics
        self.extraction_times = deque(maxlen=1000)
        self.storage_times = deque(maxlen=1000)
        self.retrieval_times = deque(maxlen=1000)
        self.quality_scores = deque(maxlen=1000)
        
        # Operation counts
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        
        # Resource monitoring
        self.memory_usage = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=1000)
        
        # Health status
        self.health_status = 'healthy'
        self.last_health_check = time.time()
        self.health_issues = []
        
        # Background monitoring
        self.monitoring_active = True
        self.monitor_thread = None
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def record_extraction_time(self, duration_ms: int, success: bool = True) -> None:
        """Record extraction operation time."""
        self.extraction_times.append(duration_ms)
        self.operation_counts['extractions'] += 1
        
        if not success:
            self.error_counts['extractions'] += 1
        
        self._check_performance_thresholds('extraction', duration_ms)
    
    def record_storage_time(self, duration_ms: int, success: bool = True) -> None:
        """Record storage operation time."""
        self.storage_times.append(duration_ms)
        self.operation_counts['storage_ops'] += 1
        
        if not success:
            self.error_counts['storage_ops'] += 1
        
        self._check_performance_thresholds('storage', duration_ms)
    
    def record_retrieval_time(self, duration_ms: int, success: bool = True) -> None:
        """Record retrieval operation time."""
        self.retrieval_times.append(duration_ms)
        self.operation_counts['retrievals'] += 1
        
        if not success:
            self.error_counts['retrievals'] += 1
        
        self._check_performance_thresholds('retrieval', duration_ms)
    
    def record_quality_score(self, score: float) -> None:
        """Record quality assessment score."""
        self.quality_scores.append(score)
    
    def record_memory_usage(self, usage_mb: float) -> None:
        """Record memory usage."""
        self.memory_usage.append(usage_mb)
        
        # Check memory threshold
        if usage_mb > self.alert_thresholds['memory_usage_mb']:
            self._add_health_issue('high_memory_usage', 
                                 f"Memory usage {usage_mb:.1f}MB exceeds threshold")
    
    def record_cpu_usage(self, usage_percent: float) -> None:
        """Record CPU usage."""
        self.cpu_usage.append(usage_percent)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'timestamp': time.time(),
            'health_status': self.health_status,
            'health_issues': self.health_issues.copy(),
            
            # Performance metrics
            'extraction': self._calculate_stats(self.extraction_times),
            'storage': self._calculate_stats(self.storage_times),
            'retrieval': self._calculate_stats(self.retrieval_times),
            'quality': self._calculate_stats(self.quality_scores),
            
            # Resource usage
            'memory_usage': self._calculate_stats(self.memory_usage),
            'cpu_usage': self._calculate_stats(self.cpu_usage),
            
            # Operation counts
            'operations': dict(self.operation_counts),
            'errors': dict(self.error_counts),
            
            # Error rates
            'error_rates': self._calculate_error_rates(),
            
            # System info
            'uptime': time.time() - self.last_health_check,
        }
        
        return summary
    
    def _calculate_stats(self, values: deque) -> Dict[str, float]:
        """Calculate basic statistics for a list of values."""
        if not values:
            return {'count': 0}
        
        values_list = list(values)
        return {
            'count': len(values_list),
            'min': min(values_list),
            'max': max(values_list),
            'mean': statistics.mean(values_list),
            'median': statistics.median(values_list),
            'p95': self._percentile(values_list, 95),
            'p99': self._percentile(values_list, 99),
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _calculate_error_rates(self) -> Dict[str, float]:
        """Calculate error rates for different operations."""
        error_rates = {}
        
        for op_type in ['extractions', 'storage_ops', 'retrievals']:
            total = self.operation_counts.get(op_type, 0)
            errors = self.error_counts.get(op_type, 0)
            
            if total > 0:
                error_rates[op_type] = errors / total
            else:
                error_rates[op_type] = 0.0
        
        return error_rates
    
    def _check_performance_thresholds(self, operation: str, duration_ms: int) -> None:
        """Check if performance exceeds thresholds."""
        threshold_key = f'{operation}_latency_ms'
        threshold = self.alert_thresholds.get(threshold_key)
        
        if threshold and duration_ms > threshold:
            issue = f'{operation}_latency_high'
            message = f"{operation} latency {duration_ms}ms exceeds threshold {threshold}ms"
            self._add_health_issue(issue, message)
    
    def _add_health_issue(self, issue_type: str, message: str) -> None:
        """Add a health issue."""
        issue_key = f"{issue_type}_{int(time.time())}"
        
        self.health_issues.append({
            'type': issue_type,
            'message': message,
            'timestamp': time.time(),
            'key': issue_key,
        })
        
        # Update health status
        if self.health_status == 'healthy':
            self.health_status = 'degraded'
        
        logger.warning(f"Health issue: {message}")
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        self.last_health_check = time.time()
        
        # Clear old health issues (older than 1 hour)
        cutoff_time = time.time() - 3600
        self.health_issues = [issue for issue in self.health_issues 
                            if issue['timestamp'] > cutoff_time]
        
        # Check error rates
        error_rates = self._calculate_error_rates()
        for op_type, rate in error_rates.items():
            if rate > self.alert_thresholds['error_rate']:
                self._add_health_issue('high_error_rate', 
                                     f"{op_type} error rate {rate:.2%} exceeds threshold")
        
        # Check recent performance
        if self.extraction_times:
            recent_avg = statistics.mean(list(self.extraction_times)[-10:])
            if recent_avg > self.alert_thresholds['extraction_latency_ms']:
                self._add_health_issue('high_avg_extraction_latency',
                                     f"Recent average extraction latency {recent_avg:.1f}ms")
        
        # Update overall health status
        if not self.health_issues:
            self.health_status = 'healthy'
        elif len(self.health_issues) > 5:
            self.health_status = 'critical'
        else:
            self.health_status = 'degraded'
        
        return {
            'status': self.health_status,
            'issues': self.health_issues.copy(),
            'timestamp': self.last_health_check,
            'metrics': self.get_performance_summary(),
        }
    
    def get_recent_metrics(self, minutes: int = 60) -> Dict[str, Any]:
        """Get metrics from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        
        return {
            'time_window_minutes': minutes,
            'cutoff_time': cutoff_time,
            'extraction_count': len([t for t in self.extraction_times if t > cutoff_time]),
            'storage_count': len([t for t in self.storage_times if t > cutoff_time]),
            'retrieval_count': len([t for t in self.retrieval_times if t > cutoff_time]),
            'total_operations': sum(self.operation_counts.values()),
            'total_errors': sum(self.error_counts.values()),
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in various formats."""
        metrics_data = self.get_performance_summary()
        
        if format == 'json':
            return json.dumps(metrics_data, indent=2)
        elif format == 'csv':
            return self._export_to_csv(metrics_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_to_csv(self, metrics_data: Dict[str, Any]) -> str:
        """Export metrics to CSV format."""
        lines = []
        lines.append("timestamp,metric_type,metric_name,value")
        
        timestamp = metrics_data['timestamp']
        
        # Add performance metrics
        for metric_type in ['extraction', 'storage', 'retrieval', 'quality']:
            if metric_type in metrics_data:
                for metric_name, value in metrics_data[metric_type].items():
                    if metric_name != 'count':
                        lines.append(f"{timestamp},{metric_type},{metric_name},{value}")
        
        return "\n".join(lines)
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.extraction_times.clear()
        self.storage_times.clear()
        self.retrieval_times.clear()
        self.quality_scores.clear()
        self.memory_usage.clear()
        self.cpu_usage.clear()
        
        self.operation_counts.clear()
        self.error_counts.clear()
        
        self.health_issues.clear()
        self.health_status = 'healthy'
        
        logger.info("All metrics reset")
    
    def _start_background_monitoring(self) -> None:
        """Start background monitoring thread."""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Perform health check
                    self.perform_health_check()
                    
                    # Sleep for health check interval
                    time.sleep(self.health_check_interval)
                    
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                    time.sleep(10)  # Brief pause before retry
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Background monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Background monitoring stopped")
    
    def get_alert_config(self) -> Dict[str, Any]:
        """Get current alert configuration."""
        return {
            'alert_thresholds': self.alert_thresholds.copy(),
            'health_check_interval': self.health_check_interval,
            'metrics_window': self.metrics_window,
        }
    
    def update_alert_config(self, new_config: Dict[str, Any]) -> None:
        """Update alert configuration."""
        self.alert_thresholds.update(new_config.get('alert_thresholds', {}))
        self.health_check_interval = new_config.get('health_check_interval', self.health_check_interval)
        self.metrics_window = new_config.get('metrics_window', self.metrics_window)
        
        logger.info("Alert configuration updated")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_monitoring()