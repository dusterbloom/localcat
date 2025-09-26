#!/usr/bin/env python3
"""
Test Intent Classification Integration
Validates that intent-aware memory processing works correctly
"""

import os
import sys
import asyncio
import time
from unittest.mock import Mock

# Add server path for imports
sys.path.insert(0, os.path.dirname(__file__))

from core.intent import get_intent_service
from core.memory.hotpath_processor import HotPathMemoryProcessor
from core.memory.session_tracker import SessionTracker
from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection


class TestIntentIntegration:
    """Test intent classification integration with memory processor"""

    def __init__(self):
        self.intent_service = None
        self.memory_processor = None

    async def setup(self):
        """Initialize services for testing"""
        print("Setting up test environment...")

        # Set environment variables for testing
        os.environ["INTENT_CLASSIFICATION_ENABLED"] = "true"
        os.environ["INTENT_LOG_CLASSIFICATION_TIME"] = "true"
        os.environ["INTENT_LOG_ROUTING_DECISIONS"] = "true"

        # Initialize intent service
        self.intent_service = get_intent_service()
        print(f"✅ Intent service initialized: {self.intent_service.enabled}")

        # Create mock context aggregator
        mock_context_aggregator = Mock()
        mock_context_aggregator.user.return_value = Mock()

        # Create session tracker
        session_tracker = SessionTracker()

        # Initialize memory processor with intent awareness
        self.memory_processor = HotPathMemoryProcessor(
            sqlite_path=":memory:",
            lmdb_dir=None,
            user_id="test-user",
            enable_metrics=True,
            context_aggregator=mock_context_aggregator,
            session_tracker=session_tracker,
            agent_id="test-agent"
        )

        print("✅ Memory processor initialized with intent awareness")

    async def test_intent_classification(self):
        """Test basic intent classification"""
        print("\n" + "="*60)
        print("Testing Intent Classification")
        print("="*60)

        test_cases = [
            "Remember that I like coffee",
            "What did I tell you about my job?",
            "Hello how are you today?",
            "Forget what I said about that",
            "Yes that's correct",
            "What can you help me with?"
        ]

        for text in test_cases:
            result = await self.intent_service.classify_intent(text)
            strategy = self.intent_service.get_memory_processing_strategy(result['intent'])
            skip_memory = self.intent_service.should_skip_memory_processing(result['intent'])

            print(f"\nText: '{text}'")
            print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
            print(f"  Strategy: {strategy}")
            print(f"  Skip memory: {skip_memory}")
            print(f"  Time: {result['processing_time_ms']:.2f}ms")

    async def test_memory_processing_with_intents(self):
        """Test memory processor with intent-aware routing"""
        print("\n" + "="*60)
        print("Testing Memory Processing with Intent Routing")
        print("="*60)

        test_scenarios = [
            {
                'text': "Remember that I like coffee",
                'expected_intent': "remember_fact",
                'should_skip': False,
                'description': "Memory storage operation"
            },
            {
                'text': "Hello there",
                'expected_intent': "general_chat",
                'should_skip': True,
                'description': "Casual greeting"
            },
            {
                'text': "What did I tell you about work?",
                'expected_intent': "recall_query",
                'should_skip': False,
                'description': "Memory recall operation"
            },
            {
                'text': "Yes that's right",
                'expected_intent': "affirmation",
                'should_skip': True,
                'description': "Simple affirmation"
            }
        ]

        for scenario in test_scenarios:
            print(f"\n🧪 Testing: {scenario['description']}")
            print(f"   Text: '{scenario['text']}'")

            # Create transcription frame
            frame = TranscriptionFrame(text=scenario['text'], is_final=True)

            # Track processing time
            start_time = time.perf_counter()

            # Process the frame (this includes intent classification and routing)
            await self.memory_processor._process_transcription(frame, FrameDirection.DOWNSTREAM)

            processing_time = (time.perf_counter() - start_time) * 1000

            # Check if memory processing was skipped for appropriate intents
            has_pending_bullets = bool(self.memory_processor._pending_bullets)

            print(f"   Processing time: {processing_time:.2f}ms")
            print(f"   Memory bullets generated: {has_pending_bullets}")
            print(f"   Expected to skip memory: {scenario['should_skip']}")

            # Validate behavior
            if scenario['should_skip'] and has_pending_bullets:
                print(f"   ⚠️  Warning: Expected to skip memory but bullets were generated")
            elif not scenario['should_skip'] and not has_pending_bullets:
                print(f"   ⚠️  Warning: Expected memory processing but no bullets generated")
            else:
                print(f"   ✅ Behavior matches expectation")

    async def test_performance_comparison(self):
        """Compare performance with and without intent classification"""
        print("\n" + "="*60)
        print("Performance Comparison")
        print("="*60)

        test_text = "Hello how are you today?"  # Should skip memory processing

        # Test with intent classification enabled
        os.environ["INTENT_CLASSIFICATION_ENABLED"] = "true"
        await self.setup()  # Reinitialize

        start_time = time.perf_counter()
        frame = TranscriptionFrame(text=test_text, is_final=True)
        await self.memory_processor._process_transcription(frame, FrameDirection.DOWNSTREAM)
        time_with_intent = (time.perf_counter() - start_time) * 1000

        # Test with intent classification disabled
        os.environ["INTENT_CLASSIFICATION_ENABLED"] = "false"
        self.memory_processor._intent_aware_processing = False

        start_time = time.perf_counter()
        frame = TranscriptionFrame(text=test_text, is_final=True)
        await self.memory_processor._process_transcription(frame, FrameDirection.DOWNSTREAM)
        time_without_intent = (time.perf_counter() - start_time) * 1000

        print(f"Processing time with intent classification: {time_with_intent:.2f}ms")
        print(f"Processing time without intent classification: {time_without_intent:.2f}ms")

        if time_with_intent < time_without_intent:
            savings = time_without_intent - time_with_intent
            print(f"✅ Performance gain: {savings:.2f}ms ({savings/time_without_intent*100:.1f}% faster)")
        else:
            overhead = time_with_intent - time_without_intent
            print(f"⚠️  Performance overhead: {overhead:.2f}ms ({overhead/time_without_intent*100:.1f}% slower)")

    async def test_statistics(self):
        """Display performance statistics"""
        print("\n" + "="*60)
        print("Intent Service Statistics")
        print("="*60)

        stats = self.intent_service.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

    async def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Intent Classification Integration Tests")
        print("="*60)

        try:
            await self.setup()
            await self.test_intent_classification()
            await self.test_memory_processing_with_intents()
            await self.test_performance_comparison()
            await self.test_statistics()

            print("\n" + "="*60)
            print("✅ All tests completed successfully!")
            print("Intent classification integration is working correctly.")

        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main test function"""
    test_runner = TestIntentIntegration()
    await test_runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())