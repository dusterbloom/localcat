"""
Performance benchmarking suite for the text-based pipeline.

This module provides comprehensive performance testing capabilities including
latency measurement, throughput analysis, and resource utilization tracking.
"""

import asyncio
import time
import psutil
import tracemalloc
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import statistics
import json
from datetime import datetime

from .text_pipeline_tester import TextPipelineTester, TestCase, TestResult

@dataclass
class BenchmarkMetrics:
    """Performance benchmark metrics."""
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_rps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate: float
    total_operations: int
    duration_seconds: float

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    name: str
    test_cases: List[TestCase]
    concurrency: int = 1
    iterations: int = 10
    duration_seconds: Optional[int] = None
    warmup_iterations: int = 3
    enable_memory_tracking: bool = True
    enable_cpu_tracking: bool = True

class PipelineBenchmark:
    """Performance benchmarking for the text pipeline."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark."""
        self.config = config
        self.tester = TextPipelineTester()
        self.results: List[TestResult] = []
        self.logger = None

    def setup_logging(self):
        """Setup logging for benchmark."""
        import logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    async def warmup(self):
        """Warm up the pipeline before benchmarking."""
        self.logger.info(f"Running {self.config.warmup_iterations} warmup iterations...")

        for i in range(self.config.warmup_iterations):
            test_case = self.config.test_cases[i % len(self.config.test_cases)]
            await self.tester.run_single_test(test_case)

        self.logger.info("Warmup complete")

    async def run_single_iteration(self, test_case: TestCase) -> TestResult:
        """Run a single benchmark iteration."""
        return await self.tester.run_single_test(test_case)

    async def run_concurrent_benchmark(self) -> BenchmarkMetrics:
        """Run benchmark with concurrent execution."""
        self.setup_logging()
        await self.warmup()

        self.logger.info(f"Starting concurrent benchmark: {self.config.name}")
        self.logger.info(f"Concurrency: {self.config.concurrency}, Iterations: {self.config.iterations}")

        # Start resource monitoring
        if self.config.enable_memory_tracking:
            tracemalloc.start()

        start_time = time.time()
        latencies = []
        errors = 0
        successful_operations = 0

        # Create executor for concurrent execution
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            tasks = []

            # Submit all tasks
            for i in range(self.config.iterations):
                test_case = self.config.test_cases[i % len(self.config.test_cases)]
                task = asyncio.create_task(self.run_single_iteration(test_case))
                tasks.append(task)

            # Wait for all tasks to complete and collect results
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                    self.logger.error(f"Task failed: {result}")
                else:
                    if result.success:
                        latencies.append(result.metrics.latency_ms)
                        successful_operations += 1
                    else:
                        errors += 1
                    self.results.append(result)

        # Calculate duration
        duration = time.time() - start_time

        # Collect resource usage
        memory_usage = 0
        cpu_usage = 0

        if self.config.enable_memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            memory_usage = peak / 1024 / 1024  # Convert to MB
            tracemalloc.stop()

        if self.config.enable_cpu_tracking:
            cpu_usage = psutil.cpu_percent()

        # Calculate metrics
        if latencies:
            metrics = BenchmarkMetrics(
                avg_latency_ms=statistics.mean(latencies),
                p50_latency_ms=statistics.median(latencies),
                p95_latency_ms=statistics.quantiles(latencies, n=20)[18],  # 95th percentile
                p99_latency_ms=statistics.quantiles(latencies, n=100)[98],  # 99th percentile
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                throughput_rps=successful_operations / duration,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                error_rate=errors / (errors + successful_operations),
                total_operations=successful_operations + errors,
                duration_seconds=duration
            )
        else:
            metrics = BenchmarkMetrics(
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                min_latency_ms=0,
                max_latency_ms=0,
                throughput_rps=0,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                error_rate=1.0,
                total_operations=errors,
                duration_seconds=duration
            )

        return metrics

    async def run_duration_benchmark(self) -> BenchmarkMetrics:
        """Run benchmark for a fixed duration."""
        self.setup_logging()
        await self.warmup()

        self.logger.info(f"Starting duration benchmark: {self.config.name}")
        self.logger.info(f"Duration: {self.config.duration_seconds}s, Concurrency: {self.config.concurrency}")

        # Start resource monitoring
        if self.config.enable_memory_tracking:
            tracemalloc.start()

        start_time = time.time()
        latencies = []
        errors = 0
        successful_operations = 0

        async def worker():
            """Worker function for duration-based benchmark."""
            nonlocal errors, successful_operations
            test_index = 0

            while time.time() - start_time < self.config.duration_seconds:
                test_case = self.config.test_cases[test_index % len(self.config.test_cases)]
                test_index += 1

                try:
                    result = await self.run_single_iteration(test_case)
                    if result.success:
                        latencies.append(result.metrics.latency_ms)
                        successful_operations += 1
                    else:
                        errors += 1
                    self.results.append(result)
                except Exception as e:
                    errors += 1
                    self.logger.error(f"Worker error: {e}")

        # Create concurrent workers
        workers = [asyncio.create_task(worker()) for _ in range(self.config.concurrency)]

        # Wait for duration to complete
        await asyncio.sleep(self.config.duration_seconds)

        # Cancel workers
        for worker in workers:
            worker.cancel()

        # Wait for workers to finish cleanup
        await asyncio.gather(*workers, return_exceptions=True)

        # Calculate duration
        duration = time.time() - start_time

        # Collect resource usage
        memory_usage = 0
        cpu_usage = 0

        if self.config.enable_memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            memory_usage = peak / 1024 / 1024  # Convert to MB
            tracemalloc.stop()

        if self.config.enable_cpu_tracking:
            cpu_usage = psutil.cpu_percent()

        # Calculate metrics
        if latencies:
            metrics = BenchmarkMetrics(
                avg_latency_ms=statistics.mean(latencies),
                p50_latency_ms=statistics.median(latencies),
                p95_latency_ms=statistics.quantiles(latencies, n=20)[18],
                p99_latency_ms=statistics.quantiles(latencies, n=100)[98],
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                throughput_rps=successful_operations / duration,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                error_rate=errors / (errors + successful_operations),
                total_operations=successful_operations + errors,
                duration_seconds=duration
            )
        else:
            metrics = BenchmarkMetrics(
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                min_latency_ms=0,
                max_latency_ms=0,
                throughput_rps=0,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                error_rate=1.0,
                total_operations=errors,
                duration_seconds=duration
            )

        return metrics

    async def run_benchmark(self) -> BenchmarkMetrics:
        """Run the benchmark based on configuration."""
        if self.config.duration_seconds:
            return await self.run_duration_benchmark()
        else:
            return await self.run_concurrent_benchmark()

    def generate_report(self, metrics: BenchmarkMetrics) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report."""
        report = {
            'benchmark_name': self.config.name,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'concurrency': self.config.concurrency,
                'iterations': self.config.iterations,
                'duration_seconds': self.config.duration_seconds,
                'warmup_iterations': self.config.warmup_iterations,
                'test_cases_count': len(self.config.test_cases)
            },
            'metrics': asdict(metrics),
            'performance_analysis': self.analyze_performance(metrics),
            'recommendations': self.generate_recommendations(metrics)
        }

        return report

    def analyze_performance(self, metrics: BenchmarkMetrics) -> Dict[str, Any]:
        """Analyze performance metrics and provide insights."""
        analysis = {
            'latency_grade': self.grade_latency(metrics.avg_latency_ms),
            'throughput_grade': self.grade_throughput(metrics.throughput_rps),
            'resource_efficiency': self.grade_resource_efficiency(metrics),
            'scalability_assessment': self.assess_scalability(metrics)
        }

        return analysis

    def grade_latency(self, avg_latency_ms: float) -> str:
        """Grade latency performance."""
        if avg_latency_ms < 100:
            return "Excellent"
        elif avg_latency_ms < 500:
            return "Good"
        elif avg_latency_ms < 1000:
            return "Fair"
        else:
            return "Poor"

    def grade_throughput(self, throughput_rps: float) -> str:
        """Grade throughput performance."""
        if throughput_rps > 10:
            return "Excellent"
        elif throughput_rps > 5:
            return "Good"
        elif throughput_rps > 1:
            return "Fair"
        else:
            return "Poor"

    def grade_resource_efficiency(self, metrics: BenchmarkMetrics) -> str:
        """Grade resource efficiency."""
        efficiency_score = metrics.throughput_rps / max(metrics.memory_usage_mb, 1)

        if efficiency_score > 5:
            return "Excellent"
        elif efficiency_score > 2:
            return "Good"
        elif efficiency_score > 0.5:
            return "Fair"
        else:
            return "Poor"

    def assess_scalability(self, metrics: BenchmarkMetrics) -> Dict[str, Any]:
        """Assess scalability potential."""
        return {
            'cpu_bound': metrics.cpu_usage_percent > 70,
            'memory_bound': metrics.memory_usage_mb > 1000,
            'latency_sensitive': metrics.p99_latency_ms > 1000,
            'throughput_limited': metrics.throughput_rps < 1
        }

    def generate_recommendations(self, metrics: BenchmarkMetrics) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if metrics.avg_latency_ms > 500:
            recommendations.append("Consider optimizing the slowest pipeline components")
            recommendations.append("Implement caching for frequent operations")

        if metrics.memory_usage_mb > 500:
            recommendations.append("Optimize memory usage patterns")
            recommendations.append("Consider memory pooling or recycling")

        if metrics.error_rate > 0.05:
            recommendations.append("Investigate and fix error sources")
            recommendations.append("Implement better error handling")

        if metrics.throughput_rps < 2:
            recommendations.append("Consider parallelization opportunities")
            recommendations.append("Optimize I/O operations")

        return recommendations

    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save benchmark report to file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        if self.logger:
            self.logger.info(f"Benchmark report saved to {filepath}")

# Standard benchmark configurations
STANDARD_BENCHMARKS = {
    'basic_throughput': BenchmarkConfig(
        name="Basic Throughput Test",
        test_cases=STANDARD_TEST_CASES,
        concurrency=1,
        iterations=50,
        warmup_iterations=5
    ),
    'concurrent_load': BenchmarkConfig(
        name="Concurrent Load Test",
        test_cases=STANDARD_TEST_CASES,
        concurrency=5,
        iterations=100,
        warmup_iterations=10
    ),
    'stress_test': BenchmarkConfig(
        name="Stress Test",
        test_cases=STANDARD_TEST_CASES,
        concurrency=10,
        iterations=200,
        warmup_iterations=20
    ),
    'sustained_load': BenchmarkConfig(
        name="Sustained Load Test",
        test_cases=STANDARD_TEST_CASES,
        concurrency=3,
        duration_seconds=60,
        warmup_iterations=10
    )
}