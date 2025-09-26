#!/usr/bin/env python3
"""
Simple Intent Classification Pipeline Integration Test

This is a focused test to validate that the complete pipeline from
bot.py → VoiceAgentFactory → Intent Classification works correctly.
"""

import os
import sys
import asyncio

# Add server path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config import VoiceAgentConfig
from core.factory import VoiceAgentFactory


async def test_full_pipeline_integration():
    """Test the complete integration pipeline without complex mocks"""
    print("🚀 Testing Full Intent Classification Pipeline Integration")
    print("=" * 70)

    # Set test environment
    os.environ["INTENT_CLASSIFICATION_ENABLED"] = "true"

    try:
        # 1. Test VoiceAgentFactory creates intent service (like bot.py does)
        print("\n1. Testing VoiceAgentFactory creates intent service...")
        config = VoiceAgentConfig.from_env()
        factory = VoiceAgentFactory(config)

        intent_service = factory.create_intent_service()
        assert intent_service is not None, "Intent service should be created"
        assert intent_service.enabled == True, "Intent service should be enabled"
        print("✅ VoiceAgentFactory successfully creates intent service")

        # 2. Test intent service classification works
        print("\n2. Testing intent classification works...")
        result = await intent_service.classify_intent("Hello there!")
        assert 'intent' in result, "Result should contain intent"
        assert 'confidence' in result, "Result should contain confidence"
        assert 'strategy' in result, "Result should contain strategy"
        assert 'skip_memory' in result, "Result should contain skip_memory"

        greeting_skip = result['skip_memory']
        greeting_intent = result['intent']
        print(f"✅ Greeting classified as: {greeting_intent} (skip_memory: {greeting_skip})")

        # 3. Test memory fact classification
        result = await intent_service.classify_intent("Remember that I like coffee")
        memory_skip = result['skip_memory']
        memory_intent = result['intent']
        print(f"✅ Memory fact classified as: {memory_intent} (skip_memory: {memory_skip})")

        # 4. Test that greetings skip memory but facts don't
        assert greeting_skip == True, "Greetings should skip memory processing"
        assert memory_skip == False, "Memory facts should NOT skip memory processing"
        print("✅ Memory routing decisions work correctly")

        # 5. Test singleton consistency
        print("\n3. Testing intent service singleton consistency...")
        intent_service_2 = factory.create_intent_service()
        assert intent_service is intent_service_2, "Should return same singleton instance"
        print("✅ Intent service singleton works correctly")

        # 6. Test different intent strategies
        print("\n4. Testing intent strategies...")
        test_cases = [
            ("Hello how are you?", True, "conversational"),
            ("Remember that I like coffee", False, "memory_operations"),  # More explicit memory phrase
            ("What did I tell you about my job?", False, "memory_operations"),
            ("Goodbye", True, "conversational"),
            ("Yes that's correct", True, "conversational")
        ]

        for text, expected_skip, expected_category in test_cases:
            result = await intent_service.classify_intent(text)
            actual_skip = result['skip_memory']

            # Get category
            categories = intent_service.get_intent_categories()
            actual_category = None
            for category, intents in categories.items():
                if result['intent'] in intents:
                    actual_category = category
                    break

            assert actual_skip == expected_skip, f"'{text}' should have skip={expected_skip}, got {actual_skip}"

            if expected_category == "conversational":
                assert actual_category == "conversational", f"'{text}' should be conversational"
            elif expected_category == "memory_operations":
                assert actual_category == "memory_operations", f"'{text}' should be memory_operations"

            print(f"  ✅ '{text[:30]}...' → {result['intent']} (skip: {actual_skip})")

        # 7. Test performance
        print("\n5. Testing performance requirements...")
        import time

        total_time = 0
        num_tests = 5

        for i in range(num_tests):
            start = time.perf_counter()
            await intent_service.classify_intent(f"Test message {i}")
            end = time.perf_counter()

            latency = (end - start) * 1000  # Convert to ms
            total_time += latency

            # Each classification should be under 200ms
            assert latency < 200, f"Classification {i} took {latency:.2f}ms (over 200ms limit)"

        avg_latency = total_time / num_tests
        print(f"✅ Average latency: {avg_latency:.2f}ms (under 200ms target)")

        # 8. Test error handling
        print("\n6. Testing error handling...")

        # Empty text
        result = await intent_service.classify_intent("")
        assert 'intent' in result, "Should handle empty text gracefully"

        # Very long text
        long_text = "word " * 500
        result = await intent_service.classify_intent(long_text)
        assert 'intent' in result, "Should handle long text gracefully"

        print("✅ Error handling works correctly")

        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("The complete pipeline from bot.py → VoiceAgentFactory → Intent Classification works correctly")
        print("\nKey Validations:")
        print("  ✅ VoiceAgentFactory creates intent service properly")
        print("  ✅ Intent classification works end-to-end")
        print("  ✅ Memory routing decisions are correct")
        print("  ✅ Singleton pattern maintains consistency")
        print("  ✅ All intent strategies work as expected")
        print("  ✅ Performance meets requirements (<200ms)")
        print("  ✅ Error handling is robust")

        return True

    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        try:
            del os.environ["INTENT_CLASSIFICATION_ENABLED"]
        except KeyError:
            pass


if __name__ == "__main__":
    success = asyncio.run(test_full_pipeline_integration())
    if success:
        print("\n✅ Integration test completed successfully")
        exit(0)
    else:
        print("\n❌ Integration test failed")
        exit(1)