#!/usr/bin/env python3
"""
Main test runner for the text-based pipeline testing framework.

This script provides a command-line interface for running comprehensive
tests on the localcat pipeline excluding STT/TTS components.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from tests.pipeline.text_pipeline_tester import TextPipelineTester, TestCase, STANDARD_TEST_CASES
from tests.pipeline.performance_benchmark import PipelineBenchmark, STANDARD_BENCHMARKS

def setup_test_environment():
    """Setup the test environment."""
    print("🔧 Setting up test environment...")

    # Check if we're in the correct directory
    if not Path("components/memory/hotmemory_facade.py").exists():
        print("❌ Error: Please run this script from the server directory")
        sys.exit(1)

    print("✅ Test environment ready")

async def run_basic_tests(test_cases: Optional[List[TestCase]] = None, output_file: Optional[str] = None):
    """Run basic pipeline tests."""
    print("🧪 Running basic pipeline tests...")

    tester = TextPipelineTester()
    cases = test_cases or STANDARD_TEST_CASES

    try:
        report = await tester.run_test_suite(cases)

        # Print summary
        print(f"\n📊 Test Results Summary:")
        print(f"   Total Tests: {report['total_tests']}")
        print(f"   Passed: {report['passed_tests']}")
        print(f"   Failed: {report['failed_tests']}")
        print(f"   Success Rate: {report['aggregate_metrics']['success_rate']:.2%}")
        print(f"   Avg Latency: {report['aggregate_metrics']['avg_latency_ms']:.2f}ms")
        print(f"   Avg Memory Ops: {report['aggregate_metrics']['avg_memory_operations']:.2f}")

        if output_file:
            tester.save_report(report, output_file)
            print(f"📄 Detailed report saved to: {output_file}")

        return report

    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        raise

async def run_performance_benchmarks(benchmark_names: Optional[List[str]] = None, output_dir: Optional[str] = None):
    """Run performance benchmarks."""
    print("🚀 Running performance benchmarks...")

    output_dir = Path(output_dir or "benchmark_results")
    output_dir.mkdir(exist_ok=True)

    benchmarks_to_run = benchmark_names or list(STANDARD_BENCHMARKS.keys())
    all_reports = {}

    for benchmark_name in benchmarks_to_run:
        if benchmark_name not in STANDARD_BENCHMARKS:
            print(f"⚠️  Unknown benchmark: {benchmark_name}")
            continue

        print(f"\n🏃 Running benchmark: {benchmark_name}")

        try:
            benchmark = PipelineBenchmark(STANDARD_BENCHMARKS[benchmark_name])
            metrics = await benchmark.run_benchmark()
            report = benchmark.generate_report(metrics)

            # Print summary
            print(f"   Avg Latency: {metrics.avg_latency_ms:.2f}ms")
            print(f"   P95 Latency: {metrics.p95_latency_ms:.2f}ms")
            print(f"   Throughput: {metrics.throughput_rps:.2f} req/s")
            print(f"   Memory Usage: {metrics.memory_usage_mb:.2f}MB")
            print(f"   Error Rate: {metrics.error_rate:.2%}")

            # Save report
            report_file = output_dir / f"{benchmark_name}_report.json"
            benchmark.save_report(report, str(report_file))
            print(f"   📄 Report saved to: {report_file}")

            all_reports[benchmark_name] = report

        except Exception as e:
            print(f"❌ Benchmark '{benchmark_name}' failed: {e}")
            all_reports[benchmark_name] = {"error": str(e)}

    # Save combined report
    if all_reports:
        combined_file = output_dir / "combined_benchmark_report.json"
        with open(combined_file, 'w') as f:
            json.dump(all_reports, f, indent=2)
        print(f"📄 Combined report saved to: {combined_file}")

    return all_reports

async def run_quality_evaluation():
    """Run quality evaluation tests."""
    print("🔍 Running quality evaluation...")

    # Create quality-focused test cases
    quality_cases = [
        TestCase(
            name="Memory Recall Test",
            input_text="Remember that I told you I have a dog named Max?",
            expected_intent="FACT",
            expected_entities=["dog", "Max"],
            expected_relations=["has_pet"]
        ),
        TestCase(
            name="Context Understanding Test",
            input_text="Given what I just said about my job, what career advice would you give me?",
            expected_intent="QUESTION",
            expected_entities=["job", "career"],
            expected_relations=[]
        ),
        TestCase(
            name="Multi-turn Consistency Test",
            input_text="Actually, let me correct what I said earlier about my education",
            expected_intent="CORRECTION",
            expected_entities=["education"],
            expected_relations=[]
        )
    ]

    tester = TextPipelineTester()
    report = await tester.run_test_suite(quality_cases)

    print(f"\n🎯 Quality Evaluation Results:")
    print(f"   Intent Accuracy: {report['aggregate_metrics']['avg_intent_accuracy']:.2%}")
    print(f"   Extraction Quality: {report['aggregate_metrics']['avg_extraction_quality']:.2%}")
    print(f"   Retrieval Accuracy: {report['aggregate_metrics']['avg_retrieval_accuracy']:.2%}")
    print(f"   Memory Efficiency: {report['aggregate_metrics']['avg_memory_efficiency']:.2%}")

    return report

async def run_latency_optimization_test():
    """Run latency optimization tests."""
    print("⚡ Running latency optimization tests...")

    # Create test cases that exercise different latency-sensitive paths
    latency_cases = [
        TestCase(
            name="Fast Path Test",
            input_text="Hello",
            expected_intent="GREETING",
            expected_entities=[],
            expected_relations=[]
        ),
        TestCase(
            name="Memory Access Test",
            input_text="What did I tell you about my family?",
            expected_intent="PURE_QUESTION",
            expected_entities=["family"],
            expected_relations=[]
        ),
        TestCase(
            name="Context Building Test",
            input_text="Can you summarize our conversation so far?",
            expected_intent="PURE_QUESTION",
            expected_entities=["conversation"],
            expected_relations=[]
        )
    ]

    tester = TextPipelineTester()
    report = await tester.run_test_suite(latency_cases)

    print(f"\n⚡ Latency Results:")
    print(f"   Average Latency: {report['aggregate_metrics']['avg_latency_ms']:.2f}ms")
    print(f"   Fastest: {min(r.metrics.latency_ms for r in tester.results):.2f}ms")
    print(f"   Slowest: {max(r.metrics.latency_ms for r in tester.results):.2f}ms")

    # Analyze latency components
    print("\n🔧 Latency Analysis:")
    for result in tester.results:
        print(f"   {result.test_case.name}: {result.metrics.latency_ms:.2f}ms")

    return report

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LocalCat Pipeline Testing Framework")
    parser.add_argument("--mode", choices=["basic", "performance", "quality", "latency", "all"],
                       default="all", help="Test mode to run")
    parser.add_argument("--output", help="Output file/directory for reports")
    parser.add_argument("--benchmarks", nargs="+", help="Specific benchmarks to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    setup_test_environment()

    if args.mode == "all":
        print("🚀 Running comprehensive pipeline test suite...\n")

        # Run all test types
        basic_report = await run_basic_tests()
        print()

        perf_reports = await run_performance_benchmarks(args.benchmarks, args.output)
        print()

        quality_report = await run_quality_evaluation()
        print()

        latency_report = await run_latency_optimization_test()

        print("\n🎉 All tests completed!")

    elif args.mode == "basic":
        await run_basic_tests(output_file=args.output)

    elif args.mode == "performance":
        await run_performance_benchmarks(args.benchmarks, args.output)

    elif args.mode == "quality":
        await run_quality_evaluation()

    elif args.mode == "latency":
        await run_latency_optimization_test()

if __name__ == "__main__":
    asyncio.run(main())