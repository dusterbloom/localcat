#!/usr/bin/env python3
"""
Comprehensive test runner for LocalCat testing suite.

This script runs all available tests and generates a consolidated report.
"""

import asyncio
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any

class TestRunner:
    """Comprehensive test runner for LocalCat."""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()

    def run_command(self, cmd: List[str], cwd: Path, description: str) -> Dict[str, Any]:
        """Run a command and capture results."""
        print(f"\n🧪 Running {description}...")
        print(f"📍 Command: {' '.join(cmd)}")
        print(f"📁 Directory: {cwd}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'duration': time.time() - self.start_time
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Timeout exceeded',
                'duration': time.time() - self.start_time
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - self.start_time
            }

    def run_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("=" * 60)
        print("🏆 PERFORMANCE BENCHMARKS")
        print("=" * 60)

        benchmark_dir = Path(__file__).parent.parent / "benchmarks"
        result = self.run_command(
            ["python", "run_quick_benchmarks.py"],
            benchmark_dir,
            "Performance Benchmarks"
        )

        if result['success']:
            # Try to read benchmark results
            results_file = benchmark_dir / "benchmark_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    benchmark_data = json.load(f)
                    result['benchmark_data'] = benchmark_data

        self.results['benchmarks'] = result
        return result

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        print("\n" + "=" * 60)
        print("🔗 INTEGRATION TESTS")
        print("=" * 60)

        integration_dir = Path(__file__).parent.parent / "integration"

        # Run pipeline tests
        pipeline_result = self.run_command(
            ["python", "run_pipeline_tests.py"],
            integration_dir,
            "Pipeline Integration Tests"
        )

        # Run minimal tests
        minimal_result = self.run_command(
            ["python", "run_minimal_tests.py"],
            integration_dir,
            "Minimal Integration Tests"
        )

        integration_results = {
            'pipeline_tests': pipeline_result,
            'minimal_tests': minimal_result,
            'success': pipeline_result['success'] and minimal_result['success']
        }

        self.results['integration'] = integration_results
        return integration_results

    def run_v7_comparison(self) -> Dict[str, Any]:
        """Run V7 vs baseline comparison."""
        print("\n" + "=" * 60)
        print("⚡ V7 PERFORMANCE COMPARISON")
        print("=" * 60)

        server_dir = Path(__file__).parent.parent
        result = self.run_command(
            ["python", "-c", """
from components.extraction.enhanced_level3_extractor import QualityExtractor
from components.extraction.extraction_strategies import EnhancedLevel3ExtractionStrategy
import spacy
import time

# V7 (Enhanced Level3)
v7_extractor = QualityExtractor()
v7_times = []
v7_relations = []
texts = [
    'My wife is at Google since 2020.',
    'John lives in Seattle and works at Microsoft since 2018.'
]
for text in texts:
    doc = spacy.load('en_core_web_sm')(text)
    t0 = time.perf_counter()
    kg = v7_extractor.extract_quality_kg(doc)
    t1 = time.perf_counter()
    v7_times.append((t1 - t0) * 1000)
    v7_relations.append(len(kg['relations']))

# Baseline (Level3)
baseline_extractor = EnhancedLevel3ExtractionStrategy()
baseline_times = []
baseline_relations = []
for text in texts:
    t0 = time.perf_counter()
    rels = baseline_extractor.extract(text)
    t1 = time.perf_counter()
    baseline_times.append((t1 - t0) * 1000)
    baseline_relations.append(len(rels))

v7_avg = sum(v7_times)/len(v7_times)
baseline_avg = sum(baseline_times)/len(baseline_times)
improvement = baseline_avg / v7_avg if v7_avg > 0 else 0

print(f"V7 Average: {v7_avg:.1f}ms")
print(f"Baseline Average: {baseline_avg:.1f}ms")
print(f"Improvement: {improvement:.1f}x faster")
"""],
            server_dir,
            "V7 vs Baseline Comparison"
        )

        self.results['v7_comparison'] = result
        return result

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = time.time() - self.start_time

        report = {
            'timestamp': time.time(),
            'total_duration': total_duration,
            'results': self.results,
            'summary': {
                'total_tests': 3,
                'passed_tests': sum(1 for r in self.results.values() if r.get('success', False)),
                'success_rate': sum(1 for r in self.results.values() if r.get('success', False)) / len(self.results)
            }
        }

        return report

    def print_summary(self, report: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("🏁 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 60)

        summary = report['summary']
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1%}")
        print(f"⏱️  Total Duration: {report['total_duration']:.1f}s")

        print("\n📋 Individual Results:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            print(f"  {test_name.replace('_', ' ').title():20} {status}")

        # Print V7 comparison results if available
        if 'v7_comparison' in self.results and self.results['v7_comparison']['success']:
            v7_output = self.results['v7_comparison']['stdout']
            if 'Improvement:' in v7_output:
                print(f"\n🚀 V7 Performance Improvement: {v7_output.split('Improvement: ')[1].strip()}")

        # Print benchmark summary if available
        if 'benchmarks' in self.results and 'benchmark_data' in self.results['benchmarks']:
            benchmark_data = self.results['benchmarks']['benchmark_data']
            if 'summary' in benchmark_data:
                bench_summary = benchmark_data['summary']
                print(f"\n🏆 Benchmark Summary: {bench_summary['passed_tests']}/{bench_summary['total_tests']} tests passed")
                for rec in bench_summary.get('recommendations', []):
                    print(f"  💡 {rec}")

    async def run_all_tests(self):
        """Run all tests and generate report."""
        print("🚀 Starting comprehensive LocalCat test suite...")
        print(f"⏰  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Run all test suites
        await asyncio.gather(
            asyncio.to_thread(self.run_benchmarks),
            asyncio.to_thread(self.run_integration_tests),
            asyncio.to_thread(self.run_v7_comparison)
        )

        # Generate report
        report = self.generate_report()

        # Save report
        report_file = Path(__file__).parent / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        self.print_summary(report)

        print(f"\n📄 Detailed report saved to: {report_file}")
        print(f"🎉 Test suite completed in {report['total_duration']:.1f}s")

        return report

async def main():
    """Main entry point."""
    runner = TestRunner()
    report = await runner.run_all_tests()
    return report

if __name__ == "__main__":
    asyncio.run(main())