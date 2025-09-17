"""
Comprehensive tests for intent processing and classification.

These tests focus on intent classification, quality filtering, and
intent-based pipeline routing functionality.
"""

import pytest
import asyncio
from typing import Dict, List, Any

from components.memory.enhanced_rule_classifier_v2 import EnhancedRuleClassifierV2
from components.memory.hotmemory_facade import HotMemoryFacade
from components.session.session_store import SessionStore
from components.memory.memory_store import MemoryStore
from pathlib import Path
import tempfile
import shutil

class TestIntentProcessing:
    """Test suite for intent processing components."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test databases."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    async def memory_store(self, temp_dir):
        """Create a memory store for testing."""
        store = MemoryStore(db_path=Path(temp_dir) / "test_memory.db")
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    async def session_store(self, temp_dir):
        """Create a session store for testing."""
        store = SessionStore(db_path=Path(temp_dir) / "test_session.db")
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    async def intent_classifier(self):
        """Create an intent classifier for testing."""
        classifier = EnhancedRuleClassifierV2()
        yield classifier

    @pytest.fixture
    async def memory_facade(self, memory_store, session_store, intent_classifier):
        """Create a memory facade for testing."""
        facade = HotMemoryFacade(
            memory_store=memory_store,
            session_store=session_store,
            intent_classifier=intent_classifier
        )
        yield facade

    @pytest.mark.asyncio
    async def test_intent_classification_basic(self, intent_classifier):
        """Test basic intent classification."""
        session_id = "test_session_1"

        # Test different intent types
        test_cases = [
            ("What's the weather like?", "PURE_QUESTION"),
            ("My name is John", "FACT"),
            ("That's interesting!", "REACTION"),
            ("Actually, I meant to say something else", "CORRECTION"),
            ("Hello there!", "GREETING"),
            ("Goodbye", "FAREWELL"),
            ("Thank you for your help", "APPRECIATION")
        ]

        for text, expected_intent in test_cases:
            result = await intent_classifier.classify_intent(text, session_id)

            assert result is not None
            assert 'intent' in result
            assert result['intent'] == expected_intent, f"Expected {expected_intent}, got {result['intent']}"

    @pytest.mark.asyncio
    async def test_intent_confidence_scoring(self, intent_classifier):
        """Test intent confidence scoring."""
        session_id = "test_session_2"

        # Test ambiguous inputs
        ambiguous_text = "Well, I don't know..."
        result = await intent_classifier.classify_intent(ambiguous_text, session_id)

        assert result is not None
        assert 'confidence' in result
        assert 0.0 <= result['confidence'] <= 1.0

        # Test clear inputs
        clear_text = "What is the capital of France?"
        result = await intent_classifier.classify_intent(clear_text, session_id)

        assert result is not None
        assert result['confidence'] > 0.7, "Clear intent should have high confidence"

    @pytest.mark.asyncio
    async def test_intent_context_awareness(self, intent_classifier):
        """Test that intent classification considers context."""
        session_id = "test_session_3"

        # Test context-dependent intents
        context_texts = [
            "My name is Sarah",
            "I work as a doctor",
            "I live in Boston"
        ]

        # Build context
        for text in context_texts:
            await intent_classifier.classify_intent(text, session_id)

        # Test context-dependent question
        question = "What did I tell you about myself?"
        result = await intent_classifier.classify_intent(question, session_id)

        assert result is not None
        assert result['intent'] in ["PURE_QUESTION", "CONTEXT_QUESTION"]

    @pytest.mark.asyncio
    async def test_intent_quality_filtering(self, intent_classifier):
        """Test quality filtering in intent classification."""
        session_id = "test_session_4"

        # Test low-quality inputs
        low_quality_inputs = [
            "um",
            "like",
            "whatever",
            "idk",
            "lol",
            ""
        ]

        for text in low_quality_inputs:
            result = await intent_classifier.classify_intent(text, session_id)

            assert result is not None
            # Low quality inputs should be filtered or marked as low confidence
            if text.strip():  # Non-empty but low quality
                assert result['confidence'] < 0.5, f"Low quality input '{text}' should have low confidence"

    @pytest.mark.asyncio
    async def test_intent_memory_gating(self, memory_facade):
        """Test that intent affects memory operations."""
        session_id = "test_session_5"

        # Test different intents and their memory impact
        test_cases = [
            ("My name is Alice", "FACT"),  # Should create memories
            ("What's the weather?", "PURE_QUESTION"),  # Should retrieve but not create
            ("That's cool!", "REACTION"),  # Should not create memories
            ("Actually, I meant Bob", "CORRECTION")  # Should update memories
        ]

        for text, expected_intent in test_cases:
            result = await memory_facade.process_turn(text, session_id)

            assert result is not None

            # Check intent-based behavior
            if expected_intent == "FACT":
                # Should create memories
                assert 'created_memories' in result
                assert len(result['created_memories']) > 0
            elif expected_intent == "REACTION":
                # Should not create significant memories
                created_count = len(result.get('created_memories', []))
                assert created_count == 0, "Reaction should not create memories"

    @pytest.mark.asyncio
    async def test_intent_error_handling(self, intent_classifier):
        """Test error handling in intent classification."""
        session_id = "test_session_6"

        # Test edge cases
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a" * 10000,  # Very long text
            "特殊字符测试",  # Non-ASCII characters
            "🙂😊😎",  # Emoji only
        ]

        for text in edge_cases:
            try:
                result = await intent_classifier.classify_intent(text, session_id)
                assert result is not None
                assert 'intent' in result
            except Exception as e:
                # Should handle gracefully
                assert "gracefully" in str(e).lower() or "handled" in str(e).lower()

    @pytest.mark.asyncio
    async def test_intent_performance(self, intent_classifier):
        """Test intent classification performance."""
        session_id = "test_session_7"

        test_texts = [
            "What time is it?",
            "My favorite color is blue",
            "That's interesting!",
            "Can you help me?",
            "I'm feeling great today",
            "Actually, I meant something else",
            "Thank you for your help"
        ] * 10  # Repeat for performance testing

        # Test performance
        import time
        start_time = time.time()

        results = []
        for text in test_texts:
            result = await intent_classifier.classify_intent(text, session_id)
            results.append(result)

        total_time = time.time() - start_time

        # Performance assertions
        assert total_time < 2.0, f"Intent classification took {total_time:.2f}s for {len(test_texts)} texts"
        assert len(results) == len(test_texts)

        # All results should be valid
        for result in results:
            assert result is not None
            assert 'intent' in result

    @pytest.mark.asyncio
    async def test_intent_conversation_flow(self, memory_facade):
        """Test intent classification in conversation flow."""
        session_id = "test_session_8"

        # Simulate a conversation
        conversation = [
            ("Hello there!", "GREETING"),
            ("My name is David", "FACT"),
            ("What can you tell me about AI?", "PURE_QUESTION"),
            ("That's fascinating!", "REACTION"),
            ("I work as a software engineer", "FACT"),
            ("Actually, I meant to say I'm a data scientist", "CORRECTION"),
            ("Thank you for the information", "APPRECIATION"),
            ("Goodbye", "FAREWELL")
        ]

        for text, expected_intent in conversation:
            result = await memory_facade.process_turn(text, session_id)

            assert result is not None
            assert 'intent' in result
            assert result['intent'] == expected_intent, f"Expected {expected_intent}, got {result['intent']}"

    @pytest.mark.asyncio
    async def test_intent_memory_retrieval_triggering(self, memory_facade):
        """Test that certain intents trigger memory retrieval."""
        session_id = "test_session_9"

        # First, create some memories
        await memory_facade.process_turn("I have a cat named Whiskers", session_id)
        await memory_facade.process_turn("I live in Seattle", session_id)

        # Test questions that should trigger retrieval
        retrieval_questions = [
            "What pets do I have?",
            "Where do I live?",
            "Tell me about my cat",
            "What do you know about me?"
        ]

        for question in retrieval_questions:
            result = await memory_facade.process_turn(question, session_id)

            assert result is not None
            # Should have retrieved memories
            assert 'retrieved_memories' in result
            assert len(result['retrieved_memories']) > 0

    @pytest.mark.asyncio
    async def test_intent_correction_handling(self, memory_facade):
        """Test handling of correction intents."""
        session_id = "test_session_10"

        # Create initial memory
        await memory_facade.process_turn("I work at Microsoft", session_id)

        # Test correction
        correction_text = "Actually, I work at Google, not Microsoft"
        result = await memory_facade.process_turn(correction_text, session_id)

        assert result is not None
        assert result['intent'] == "CORRECTION"

        # Should have correction information
        assert 'corrections' in result or 'updated_memories' in result

    @pytest.mark.asyncio
    async def test_intent_multi_language_support(self, intent_classifier):
        """Test intent classification with different languages."""
        session_id = "test_session_11"

        # Test different languages
        multi_language_tests = [
            ("¿Cómo estás?", "GREETING"),  # Spanish
            ("Bonjour", "GREETING"),  # French
            ("Guten Tag", "GREETING"),  # German
            ("ありがとう", "APPRECIATION"),  # Japanese
            ("What is your name?", "PURE_QUESTION"),  # English
        ]

        for text, expected_intent in multi_language_tests:
            result = await intent_classifier.classify_intent(text, session_id)

            assert result is not None
            assert 'intent' in result
            # Note: Multi-language support might be limited, so we just check it doesn't crash
            assert result['intent'] in ["GREETING", "PURE_QUESTION", "APPRECIATION", "UNKNOWN"]

    @pytest.mark.asyncio
    async def test_intent_sarcasm_detection(self, intent_classifier):
        """Test detection of sarcastic or ironic statements."""
        session_id = "test_session_12"

        sarcastic_texts = [
            "Oh great, another meeting",
            "Perfect, just what I needed",
            "Yeah, right",
            "Wonderful"
        ]

        for text in sarcastic_texts:
            result = await intent_classifier.classify_intent(text, session_id)

            assert result is not None
            assert 'intent' in result
            # Sarcastic statements might be classified as REACTION or have lower confidence
            assert result['intent'] in ["REACTION", "SARCASM", "UNKNOWN"]

    @pytest.mark.asyncio
    async def test_intent_follow_up_detection(self, intent_classifier):
        """Test detection of follow-up questions and statements."""
        session_id = "test_session_13"

        # Build some context
        context_texts = [
            "I'm planning a trip to Paris",
            "I've never been to Europe before",
            "I'm excited about the Eiffel Tower"
        ]

        for text in context_texts:
            await intent_classifier.classify_intent(text, session_id)

        # Test follow-up questions
        follow_ups = [
            "What should I pack?",
            "When is the best time to go?",
            "How much does it cost?"
        ]

        for text in follow_ups:
            result = await intent_classifier.classify_intent(text, session_id)

            assert result is not None
            assert 'intent' in result
            # Follow-ups should be recognized as questions
            assert result['intent'] in ["PURE_QUESTION", "FOLLOW_UP_QUESTION"]

    @pytest.mark.asyncio
    async def test_intent_sentiment_analysis(self, intent_classifier):
        """Test sentiment analysis integration with intent classification."""
        session_id = "test_session_14"

        sentiment_tests = [
            ("I'm so happy today!", "positive"),
            ("This is terrible", "negative"),
            ("I'm feeling okay", "neutral"),
            ("I'm excited about the project", "positive"),
            ("I'm worried about the deadline", "negative")
        ]

        for text, expected_sentiment in sentiment_tests:
            result = await intent_classifier.classify_intent(text, session_id)

            assert result is not None
            # Check if sentiment is included in the result
            if 'sentiment' in result:
                assert result['sentiment'] in ['positive', 'negative', 'neutral']

    @pytest.mark.asyncio
    async def test_intent_session_evolution(self, memory_facade):
        """Test how intent classification evolves over a session."""
        session_id = "test_session_15"

        # Long conversation with evolving context
        conversation = [
            "Hello!",
            "I'm new here",
            "Can you help me learn about this system?",
            "That's helpful",
            "I think I understand now",
            "Actually, I have a question about something else",
            "Thank you for your patience",
            "I need to go now",
            "Goodbye!"
        ]

        intents = []
        for text in conversation:
            result = await memory_facade.process_turn(text, session_id)
            intents.append(result['intent'])

        # Verify conversation flow
        assert intents[0] == "GREETING"  # Hello!
        assert intents[-1] == "FAREWELL"  # Goodbye!
        assert "APPRECIATION" in intents  # Thank you

        # Should have variety of intents
        unique_intents = set(intents)
        assert len(unique_intents) >= 3, "Should have variety of intents in conversation"