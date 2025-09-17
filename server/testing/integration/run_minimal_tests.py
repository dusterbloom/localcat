#!/usr/bin/env python3
"""
Simple test runner for minimal pipeline testing.
"""

import asyncio
import argparse
import sys
from pathlib import Path

from tests.pipeline.minimal_pipeline_tester import MinimalPipelineTester, SIMPLE_TEST_CASES

async def run_basic_tests():
    """Run basic pipeline tests using the minimal tester."""
    print("🧪 Running minimal pipeline tests...")

    tester = MinimalPipelineTester()

    try:
        report = await tester.run_test_suite(SIMPLE_TEST_CASES)

        # Print summary
        print(f"\n📊 Test Results Summary:")
        print(f"   Total Tests: {report['total_tests']}")
        print(f"   Passed: {report['passed_tests']}")
        print(f"   Failed: {report['failed_tests']}")
        print(f"   Success Rate: {report['success_rate']:.2%}")
        print(f"   Avg Latency: {report['average_latency']:.2f}ms")
        print(f"   Avg Memory Ops: {report['average_memory_ops']:.2f}")
        print(f"   Total Memories Created: {report['total_memories_created']}")

        # Print individual results
        print(f"\n🔍 Individual Test Results:")
        for result in report['individual_results']:
            status = "✅" if result['metrics']['success'] else "❌"
            print(f"   {status} {result['test_case']['name']}: {result['metrics']['latency_ms']:.2f}ms")
            if result['error']:
                print(f"      Error: {result['error']}")

        return report

    except Exception as e:
        print(f"❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LocalCat Minimal Pipeline Testing")
    parser.add_argument("--output", help="Output file for test report")

    args = parser.parse_args()

    # Check if we're in the right directory
    if not Path("components/memory/hotmemory_facade.py").exists():
        print("❌ Error: Please run this script from the server directory")
        sys.exit(1)

    print("🚀 Starting minimal pipeline test runner...")

    report = await run_basic_tests()

    if report and args.output:
        tester = MinimalPipelineTester()
        tester.save_report(report, args.output)
        print(f"📄 Report saved to: {args.output}")

    print("\n🎉 Test run completed!")

if __name__ == "__main__":
    asyncio.run(main())