#!/usr/bin/env python
"""
Run all tests for the LocalCat streaming server
"""

import sys
import os
import asyncio
import subprocess
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test_file(test_path: Path) -> bool:
    """Run a single test file and return success status"""
    logger.info(f"Running {test_path.name}...")

    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            logger.success(f"✓ {test_path.name} passed")
            return True
        else:
            logger.error(f"✗ {test_path.name} failed")
            if result.stdout:
                logger.error("  ----- stdout -----")
                logger.error(result.stdout[-2000:])
            if result.stderr:
                logger.error("  ----- stderr -----")
                logger.error(result.stderr[-2000:])
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"✗ {test_path.name} timed out")
        return False
    except Exception as e:
        logger.error(f"✗ {test_path.name} error: {e}")
        return False


def main():
    """Run all test suites"""
    logger.info("=" * 60)
    logger.info("LOCALCAT STREAMING TEST SUITE")
    logger.info("=" * 60)

    tests_dir = Path(__file__).parent

    # Define test suites
    test_suites = {
        "Unit Tests": list(tests_dir.glob("unit/*.py")),
        "Integration Tests": list(tests_dir.glob("integration/*.py"))
    }

    all_results = []

    for suite_name, test_files in test_suites.items():
        if not test_files:
            continue

        logger.info(f"\n{suite_name}:")
        logger.info("-" * 40)

        for test_file in test_files:
            if test_file.name == "__init__.py":
                continue
            passed = run_test_file(test_file)
            all_results.append((suite_name, test_file.name, passed))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed_count = sum(1 for _, _, passed in all_results if passed)
    total_count = len(all_results)

    for suite, test_name, passed in all_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} | {suite:20} | {test_name}")

    logger.info("-" * 60)
    logger.info(f"Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        logger.success("\n🎉 All tests passed!")
        return True
    else:
        logger.error(f"\n❌ {total_count - passed_count} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
