#!/usr/bin/env python3
"""Test script to verify V2 Enhanced Rule Classifier fixes"""

import sys
import os

# Activate virtual environment
sys.path.insert(0, '/Users/peppi/Dev/localcat/server')
os.chdir('/Users/peppi/Dev/localcat/server')

# Test the fixed V2 classifier
def test_v2_classifier():
    from components.memory.memory_intent import get_intent_classifier

    # Force V2 classifier
    os.environ["USE_DISTILBERT_CLASSIFIER"] = "false"
    os.environ["USE_BASIC_CLASSIFIER"] = "false"

    classifier = get_intent_classifier()

    print("🧪 Testing Fixed V2 Enhanced Rule Classifier")
    print("=" * 50)

    test_cases = [
        ("Hello there", "GREETING", False, False),
        ("I work at Google", "FACT", True, False),
        ("What is the weather?", "QUESTION", False, True),
        ("My name is John", "FACT", True, False),
        ("Can you help me?", "REQUEST", False, True),
        ("No, actually it's Microsoft", "CORRECTION", True, True),
        ("Thanks!", "ACKNOWLEDGMENT", False, False),
        ("Wow that's amazing!", "REACTION", False, False),
    ]

    all_passed = True

    for text, expected_type, expected_memory, expected_retrieval in test_cases:
        result = classifier.analyze(text)

        success = result is not None
        intent_str = result.intent.name if result else "None"

        print(f"Text: '{text}'")
        print(f"  Result: {intent_str} (expected {expected_type})")

        if result:
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Requires Memory: {getattr(result, 'requires_memory', 'N/A')} (expected {expected_memory})")
            print(f"  Requires Retrieval: {getattr(result, 'requires_retrieval', 'N/A')} (expected {expected_retrieval})")

            if intent_str != expected_type:
                print(f"  ❌ WRONG INTENT: got {intent_str}, expected {expected_type}")
                all_passed = False
            elif not hasattr(result, 'requires_memory') or not hasattr(result, 'requires_retrieval'):
                print(f"  ❌ MISSING ATTRIBUTES: requires_memory or requires_retrieval")
                all_passed = False
            else:
                print(f"  ✅ CORRECT")
        else:
            print(f"  ❌ RETURNED None!")
            all_passed = False

        print()

    print("=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! V2 classifier is working correctly.")
    else:
        print("❌ Some tests failed. V2 classifier needs more fixes.")

    return all_passed

if __name__ == "__main__":
    test_v2_classifier()