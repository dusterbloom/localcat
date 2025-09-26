#!/usr/bin/env python
"""
Run all tests for the LocalCat streaming server
Supports different test categories for CI/CD and development
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mitigate macOS OpenMP shared-memory issues during tests
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")


def run_pytest(args, category="all"):
    """Run tests using pytest with proper configuration"""
    cmd = [sys.executable, "-m", "pytest", "-v"]

    if category == "ci":
        # Fast CI tests only
        cmd.extend(["-m", "ci"])
        logger.info("Running CI test suite (fast tests only)")
    elif category == "fast":
        # Fast tests, no models required
        cmd.extend(["-m", "fast and not slow and not requires_models"])
        logger.info("Running fast test suite")
    elif category == "unit":
        # Unit tests only
        cmd.extend(["tests/unit", "-m", "not integration"])
        logger.info("Running unit tests")
    elif category == "integration":
        # Integration tests
        cmd.extend(["tests/integration"])
        logger.info("Running integration tests")
    elif category == "slow":
        # Run slow tests
        cmd.extend(["-m", "slow", "--run-slow"])
        logger.info("Running slow tests (this may take a while)")
    else:
        # Run all tests
        if args.skip_slow:
            cmd.extend(["-m", "not slow"])
            logger.info("Running all tests except slow tests")
        else:
            cmd.append("--run-slow")
            logger.info("Running all tests including slow tests")

    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing"])

    # Set CI environment if requested
    if args.ci_mode:
        os.environ["CI"] = "1"
        logger.info("CI mode enabled")

    # Run tests
    try:
        logger.info(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to run pytest: {e}")
        return False


def run_test_file(test_path: Path, timeout=60):
    """Run a single test file directly with Python (legacy mode)"""
    logger.info(f"Running {test_path.name} directly...")

    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            timeout=timeout
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
        logger.error(f"✗ {test_path.name} timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"✗ {test_path.name} error: {e}")
        return False


def main():
    """Run test suite with various options"""
    parser = argparse.ArgumentParser(description="Run LocalCat test suite")
    parser.add_argument(
        "--category",
        choices=["all", "ci", "fast", "unit", "integration", "slow"],
        default="all",
        help="Test category to run"
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow tests"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy test runner (runs files directly)"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Run in CI mode (sets CI env var, skips certain tests)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per test in seconds (legacy mode only)"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("LOCALCAT STREAMING TEST SUITE")
    logger.info("=" * 60)

    if args.legacy:
        # Legacy mode: run test files directly
        logger.info("Using legacy test runner")

        tests_dir = Path(__file__).parent
        test_suites = {
            "Unit Tests": list(tests_dir.glob("unit/*.py")),
            "Integration Tests": list(tests_dir.glob("integration/*.py"))
        }

        all_results = []

        for suite_name, test_files in test_suites.items():
            if not test_files:
                continue

            # Filter based on category
            if args.category == "unit" and "Integration" in suite_name:
                continue
            if args.category == "integration" and "Unit" in suite_name:
                continue

            logger.info(f"\n{suite_name}:")
            logger.info("-" * 40)

            for test_file in test_files:
                if test_file.name == "__init__.py":
                    continue
                if test_file.name.startswith("__"):
                    continue

                # Skip slow tests if requested
                if args.skip_slow and any(slow in test_file.name for slow in ["performance", "e2e", "stability"]):
                    logger.info(f"Skipping slow test: {test_file.name}")
                    continue

                result = run_test_file(test_file, args.timeout)
                all_results.append(result)

        # Summary
        passed = sum(all_results)
        failed = len(all_results) - passed

        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("-" * 60)
        logger.info(f"Total: {len(all_results)} | Passed: {passed} | Failed: {failed}")

        if failed == 0:
            logger.success("✅ ALL TESTS PASSED!")
            return 0
        else:
            logger.error(f"❌ {failed} TESTS FAILED")
            return 1

    else:
        # Modern mode: use pytest
        success = run_pytest(args, args.category)

        if success:
            logger.success("\n✅ TEST SUITE PASSED!")
            return 0
        else:
            logger.error("\n❌ TEST SUITE FAILED")
            return 1


if __name__ == "__main__":
    sys.exit(main())