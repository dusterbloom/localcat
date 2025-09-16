#!/usr/bin/env python3
"""
Debug intent classification to see why facts are being misclassified
"""

import os
import sys

# Add server path and activate environment
sys.path.insert(0, '/Users/peppi/Dev/localcat/server')
os.chdir('/Users/peppi/Dev/localcat/server')


def test_intent_classification():
    """Test intent classification on example facts"""
    print("🎯 INTENT CLASSIFICATION DEBUG")
    print("=" * 50)

    from components.memory.memory_intent import get_intent_classifier

    # Initialize classifier
    classifier = get_intent_classifier()

    test_cases = [
        "My favorite color is blue",
        "I work at TechCorp as a software engineer",
        "I live in San Francisco on Market Street",
        "My dog's name is Rex and he loves tennis balls",
        "I went to Stanford University for computer science",
        "What is the weather like today?",
        "Hello there!",
        "Actually, his name is Buddy"
    ]

    print(f"Testing {len(test_cases)} statements:")
    print()

    for text in test_cases:
        intent = classifier.analyze(text, "en")

        needs_memory = getattr(intent, 'requires_memory', True)
        needs_retrieval = getattr(intent, 'requires_retrieval', True)

        print(f"Text: '{text}'")
        print(f"  Intent: {intent.intent.value}")
        print(f"  Confidence: {intent.confidence:.3f}")
        print(f"  Needs memory: {needs_memory}")
        print(f"  Needs retrieval: {needs_retrieval}")
        print(f"  Classifier: {getattr(intent, 'classifier', 'unknown')}")
        print()

if __name__ == "__main__":
    test_intent_classification()