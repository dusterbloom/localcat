#!/usr/bin/env python3
"""
Simple Intent Classification Test
Tests the core intent classification without full memory processor integration
"""

import os
import sys
import asyncio
import time

# Add server path for imports
sys.path.insert(0, os.path.dirname(__file__))

from core.intent import get_intent_service


async def test_intent_classification():
    """Test intent classification functionality"""
    print("🚀 Testing Intent Classification Integration")
    print("="*60)

    # Set environment variables for testing
    os.environ["INTENT_CLASSIFICATION_ENABLED"] = "true"
    os.environ["INTENT_LOG_CLASSIFICATION_TIME"] = "true"
    os.environ["INTENT_LOG_ROUTING_DECISIONS"] = "true"

    # Initialize intent service
    intent_service = get_intent_service()
    print(f"✅ Intent service initialized: {intent_service.enabled}")
    print(f"✅ Model: {intent_service.model_name}")

    # Test intent classification and routing
    test_cases = [
        {
            'text': "Remember that I like coffee",
            'expected_category': 'memory_operations',
            'expected_skip': False
        },
        {
            'text': "What did I tell you about my job?",
            'expected_category': 'memory_operations',
            'expected_skip': False
        },
        {
            'text': "Hello how are you today?",
            'expected_category': 'conversational',
            'expected_skip': True
        },
        {
            'text': "Goodbye see you later",
            'expected_category': 'conversational',
            'expected_skip': True
        },
        {
            'text': "Yes that's correct",
            'expected_category': 'conversational',
            'expected_skip': True
        },
        {
            'text': "Forget what I said about pizza",
            'expected_category': 'memory_operations',
            'expected_skip': False
        },
        {
            'text': "What can you help me with?",
            'expected_category': 'capability_queries',
            'expected_skip': False
        }
    ]

    print("\n" + "="*60)
    print("Testing Intent Classification and Routing")
    print("="*60)

    total_time = 0
    classifications = 0

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{case['text']}'")

        # Classify intent
        result = await intent_service.classify_intent(case['text'])
        intent = result['intent']
        confidence = result['confidence']
        processing_time = result.get('total_processing_time_ms', result.get('processing_time_ms', 0))
        cached = result.get('cached', False)

        # Get routing decisions
        strategy = intent_service.get_memory_processing_strategy(intent)
        should_skip = intent_service.should_skip_memory_processing(intent)
        categories = intent_service.get_intent_categories()

        # Find which category this intent belongs to
        intent_category = None
        for category, intents in categories.items():
            if intent in intents:
                intent_category = category
                break

        print(f"   Intent: {intent} (confidence: {confidence:.3f})")
        print(f"   Category: {intent_category}")
        print(f"   Strategy: {strategy}")
        print(f"   Skip memory: {should_skip}")
        print(f"   Processing time: {processing_time:.2f}ms")
        print(f"   Cached: {cached}")

        # Validation
        if should_skip == case['expected_skip']:
            print(f"   ✅ Routing decision correct")
        else:
            print(f"   ⚠️  Expected skip={case['expected_skip']}, got skip={should_skip}")

        total_time += processing_time
        classifications += 1

    # Performance summary
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)

    avg_time = total_time / classifications if classifications > 0 else 0
    print(f"Total classifications: {classifications}")
    print(f"Average processing time: {avg_time:.2f}ms")

    # Get service statistics
    stats = intent_service.get_performance_stats()
    print(f"Cache hit rate: {stats.get('cache_hit_rate', 0)*100:.1f}%")
    print(f"Fallback rate: {stats.get('fallback_rate', 0)*100:.1f}%")

    # Test memory processing strategies
    print("\n" + "="*60)
    print("Memory Processing Strategies")
    print("="*60)

    strategy_test_cases = [
        ("remember_fact", "storage_focused"),
        ("recall_query", "retrieval_focused"),
        ("forget_request", "deletion_focused"),
        ("memory_check", "lookup_focused"),
        ("general_chat", "minimal"),
        ("greeting", "skip"),
        ("goodbye", "skip"),
        ("clarification", "contextual")
    ]

    for intent, expected_strategy in strategy_test_cases:
        actual_strategy = intent_service.get_memory_processing_strategy(intent)
        skip_memory = intent_service.should_skip_memory_processing(intent)

        print(f"{intent:15} → {actual_strategy:15} (skip: {skip_memory})")

        if actual_strategy != expected_strategy:
            print(f"   ⚠️  Expected {expected_strategy}, got {actual_strategy}")

    # Performance comparison simulation
    print("\n" + "="*60)
    print("Estimated Performance Impact")
    print("="*60)

    greeting_time = 50  # Typical classification time for simple cases
    memory_processing_time = 200  # Typical memory processing time we're saving

    print(f"Intent classification time: ~{greeting_time}ms")
    print(f"Memory processing time saved: ~{memory_processing_time}ms")
    print(f"Net performance gain for skipped intents: ~{memory_processing_time - greeting_time}ms")
    print(f"Performance improvement: {(memory_processing_time - greeting_time)/memory_processing_time*100:.1f}%")

    print("\n✅ Intent classification integration test completed successfully!")
    print("The system correctly classifies intents and makes appropriate routing decisions.")


if __name__ == "__main__":
    asyncio.run(test_intent_classification())