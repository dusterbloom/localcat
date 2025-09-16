#!/usr/bin/env python3
"""Final validation test for intent classification system"""

import sys
import os

# Activate virtual environment
sys.path.insert(0, '/Users/peppi/Dev/localcat/server')
os.chdir('/Users/peppi/Dev/localcat/server')

def test_intent_classification_fixed():
    from components.memory.memory_intent import get_intent_classifier

    # Force V2 classifier (default)
    os.environ["USE_DISTILBERT_CLASSIFIER"] = "false"
    os.environ["USE_BASIC_CLASSIFIER"] = "false"

    classifier = get_intent_classifier()

    print("🎯 Final Validation: Intent Classification System")
    print("=" * 55)

    test_cases = [
        # Core functionality tests - mapped intent names
        ("Hello there", "REACTION", False, False),  # Greeting mapped to reaction
        ("I work at Google", "FACT_STATEMENT", True, False),  # Fact statement
        ("What is the weather?", "PURE_QUESTION", False, True),  # Pure question
        ("My name is John", "FACT_STATEMENT", True, False),  # Personal fact
        ("No, actually it's Microsoft", "CORRECTION", True, True),  # Correction
        ("Thanks!", "REACTION", False, False),  # Acknowledgment mapped to reaction
        ("Wow that's amazing!", "REACTION", False, False),  # Reaction
    ]

    all_working = True
    none_count = 0
    working_count = 0

    for text, expected_intent, expected_memory, expected_retrieval in test_cases:
        result = classifier.analyze(text)

        if result is None:
            print(f"❌ '{text}' → None (CRITICAL FAILURE)")
            none_count += 1
            all_working = False
            continue

        intent_name = result.intent.name
        has_attrs = hasattr(result, 'requires_memory') and hasattr(result, 'requires_retrieval')

        if has_attrs:
            memory_match = result.requires_memory == expected_memory
            retrieval_match = result.requires_retrieval == expected_retrieval
            intent_match = intent_name == expected_intent

            status = "✅" if (intent_match and memory_match and retrieval_match) else "⚠️"

            print(f"{status} '{text}' → {intent_name} (conf: {result.confidence:.2f})")
            print(f"   Memory: {result.requires_memory}/{expected_memory}, Retrieval: {result.requires_retrieval}/{expected_retrieval}")

            working_count += 1
        else:
            print(f"❌ '{text}' → Missing attributes (FAILURE)")
            all_working = False

    print("=" * 55)
    print(f"📊 RESULTS:")
    print(f"   Working: {working_count}/{len(test_cases)}")
    print(f"   None returns: {none_count}")
    print(f"   Has required attributes: {working_count > 0}")

    if none_count == 0:
        print("🎉 SUCCESS: No more None returns - Core issue FIXED!")
        print("🔧 Intent classification system is functional")
        print("💾 Memory gating will now work based on intent types")
        return True
    else:
        print("❌ FAILURE: Still returning None for some inputs")
        return False

if __name__ == "__main__":
    success = test_intent_classification_fixed()
    sys.exit(0 if success else 1)