#!/usr/bin/env python3
"""
Simple test for greeting detection substring bug fix.
Tests the _should_suppress_memory_injection logic directly.
"""

import sys
from loguru import logger

# Set up minimal logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<level>{level: <8}</level> | {message}")

def test_greeting_detection_fix():
    """Test that word boundary matching fixes the substring bug"""
    print("\n" + "="*80)
    print("TEST: Greeting Detection Substring Bug Fix")
    print("="*80)

    from unittest.mock import Mock
    from core.memory.retrieval import Retrieval

    # Create mock host
    host = Mock()
    host.entity_index = {}
    host.recency_buffer = Mock()
    host.store = Mock()
    host.current_user_id = "test_user"
    host.current_session_id = "test_session"

    retriever = Retrieval(host)

    # Test cases
    test_cases = [
        ("Do you know my location?", False, "Should NOT suppress - valid memory query"),
        ("Do you know my favorite number?", False, "Should NOT suppress - valid memory query"),
        ("hello", True, "Should suppress - actual greeting"),
        ("hi there", True, "Should suppress - actual greeting"),
        ("hey yo", True, "Should suppress - actual greeting with 'yo'"),
        ("yo", True, "Should suppress - standalone 'yo' greeting"),
        ("you are great", False, "Should NOT suppress - 'you' contains 'yo' substring but not a greeting"),
    ]

    all_passed = True

    for query, expected_suppress, description in test_cases:
        print(f"\nQuery: '{query}'")
        print(f"Expected: {expected_suppress} ({description})")

        actual_suppress = retriever._should_suppress_memory_injection(query)

        print(f"Actual:   {actual_suppress}")

        if actual_suppress == expected_suppress:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("🎉 All greeting detection tests passed!")
        return 0
    else:
        print("⚠️  Some greeting detection tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = test_greeting_detection_fix()
    sys.exit(exit_code)
