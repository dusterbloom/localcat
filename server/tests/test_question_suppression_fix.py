#!/usr/bin/env python3
"""
Test that memory recall questions are NOT suppressed.

This tests the critical fix for the issue where "Do you know my dog's name?"
was incorrectly suppressing memory injection.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory.retrieval import Retrieval
from unittest.mock import Mock


def test_memory_recall_questions_not_suppressed():
    """Test that memory recall questions don't suppress memory injection."""

    # Create mock host
    host = Mock()
    host.entity_index = {}
    host.recency_buffer = []
    host.store = Mock()

    retrieval = Retrieval(host)

    # Memory recall questions that should NOT be suppressed
    memory_recall_queries = [
        "Do you know my dog's name?",
        "Do you know where I live?",
        "Can you tell me what my favorite food is?",
        "Do you remember what I told you about my job?",
        "Can you recall our conversation from yesterday?",
        "Do you know when my birthday is?",
        "What did I say about my family?",
        "Do you know who my friends are?",
    ]

    print("\n" + "="*70)
    print("TESTING: Memory Recall Questions Should NOT Be Suppressed")
    print("="*70 + "\n")

    all_pass = True

    for query in memory_recall_queries:
        suppressed = retrieval._should_suppress_memory_injection(query)
        status = "❌ FAIL" if suppressed else "✅ PASS"

        if suppressed:
            all_pass = False
            print(f"{status}: '{query}'")
            print(f"  → WRONGLY suppressed (should allow memory)")
        else:
            print(f"{status}: '{query}'")
            print(f"  → Correctly allows memory retrieval")

    # Generic capability questions that SHOULD be suppressed
    generic_queries = [
        "What can you do?",
        "Can you help me?",
        "How are you?",
        "What are you?",
    ]

    print("\n" + "="*70)
    print("TESTING: Generic Questions Should Be Suppressed")
    print("="*70 + "\n")

    for query in generic_queries:
        suppressed = retrieval._should_suppress_memory_injection(query)
        status = "✅ PASS" if suppressed else "❌ FAIL"

        if not suppressed:
            all_pass = False
            print(f"{status}: '{query}'")
            print(f"  → WRONGLY allowed (should suppress)")
        else:
            print(f"{status}: '{query}'")
            print(f"  → Correctly suppressed")

    # Summary
    print("\n" + "="*70)
    if all_pass:
        print("✅ ALL TESTS PASSED")
        print("="*70)
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        return False


if __name__ == "__main__":
    success = test_memory_recall_questions_not_suppressed()
    sys.exit(0 if success else 1)
