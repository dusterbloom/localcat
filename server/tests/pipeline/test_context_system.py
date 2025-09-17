"""
Comprehensive tests for the context system components.

These tests focus on context building, token budgeting, progressive context,
and overall context management functionality.
"""

import pytest
import asyncio
from typing import Dict, List, Any

from components.context.context_orchestrator import pack_context
from components.context.memory_config import get_global_config
from components.context.memory_config import MemoryConfig
from components.memory.hotmemory_facade import HotMemoryFacade
from components.memory.enhanced_rule_classifier_v2 import EnhancedRuleClassifierV2
from components.session.session_store import SessionStore
from components.memory.memory_store import MemoryStore
from pathlib import Path
import tempfile
import shutil

class TestContextSystem:
    """Test suite for context system components."""

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
    def context_config(self):
        """Create a context configuration for testing."""
        return get_global_config()

    @pytest.fixture
    async def memory_facade(self, memory_store, session_store):
        """Create a memory facade for testing."""
        classifier = EnhancedRuleClassifierV2()
        facade = HotMemoryFacade(
            memory_store=memory_store,
            session_store=session_store,
            intent_classifier=classifier
        )
        yield facade

    @pytest.mark.asyncio
    async def test_context_building_basic(self, context_config):
        """Test basic context building functionality."""
        token_budget = 1000

        # Build context
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        memory_bullets = []
        summary_text = None

        packed_messages, stats = pack_context(
            messages=messages,
            memory_bullets=memory_bullets,
            summary_text=summary_text,
            budget_tokens=token_budget,
            progressive_mode=True
        )

        # Verify structure
        assert packed_messages is not None
        assert isinstance(packed_messages, list)
        assert stats is not None
        assert 'tokens_total' in stats

        # Verify token budget adherence
        assert stats['tokens_total'] <= token_budget

    @pytest.mark.asyncio
    async def test_context_with_memories(self, context_orchestrator, memory_facade):
        """Test context building with memories."""
        session_id = "test_session_2"
        token_budget = 2000

        # First, create some memories
        await memory_facade.process_turn("My name is Alice and I work as a software engineer", session_id)
        await memory_facade.process_turn("I have a dog named Buddy who loves to play fetch", session_id)
        await memory_facade.process_turn("I live in San Francisco and enjoy hiking on weekends", session_id)

        # Build context
        context_result = await context_orchestrator.pack_context(session_id, token_budget)

        # Verify memories are included
        assert context_result is not None
        assert len(context_result['memory_bullets']) > 0

        # Check that relevant memories are included
        memory_text = ' '.join(context_result['memory_bullets']).lower()
        assert 'alice' in memory_text
        assert 'software engineer' in memory_text
        assert 'buddy' in memory_text

    @pytest.mark.asyncio
    async def test_progressive_context_mode(self, context_orchestrator, memory_facade):
        """Test progressive context mode."""
        session_id = "test_session_3"
        token_budget = 1500

        # Enable progressive mode
        context_orchestrator.config.progressive_context_enabled = True

        # Create memories
        await memory_facade.process_turn("I am a doctor specializing in cardiology", session_id)
        await memory_facade.process_turn("I work at General Hospital", session_id)
        await memory_facade.process_turn("I have been practicing medicine for 15 years", session_id)

        # Build context in progressive mode
        context_result = await context_orchestrator.pack_context(session_id, token_budget)

        # Verify progressive context features
        assert context_result is not None
        assert 'progressive_context' in context_result
        assert context_result['progressive_context'] is True

        # Check memory prioritization
        assert len(context_result['memory_bullets']) > 0

    @pytest.mark.asyncio
    async def test_token_budget_adherence(self, context_orchestrator):
        """Test that context respects token budget."""
        session_id = "test_session_4"

        # Test with different budget sizes
        budgets = [500, 1000, 2000, 4000]

        for budget in budgets:
            context_result = await context_orchestrator.pack_context(session_id, budget)

            assert context_result is not None
            assert context_result['token_count'] <= budget

            # Context should still be meaningful even with small budgets
            if budget >= 500:
                assert len(context_result['context']) > 0

    @pytest.mark.asyncio
    async def test_context_prioritization(self, context_orchestrator, memory_facade):
        """Test that context prioritizes important memories."""
        session_id = "test_session_5"
        token_budget = 1000

        # Create memories with different importance levels
        await memory_facade.process_turn("I am allergic to peanuts", session_id)  # High importance
        await memory_facade.process_turn("I like the color blue", session_id)  # Low importance
        await memory_facade.process_turn("My mother's name is Sarah", session_id)  # Medium importance
        await memory_facade.process_turn("I prefer coffee over tea", session_id)  # Low importance

        # Build context with limited budget
        context_result = await context_orchestrator.pack_context(session_id, token_budget)

        # Verify prioritization
        assert context_result is not None
        memory_text = ' '.join(context_result['memory_bullets']).lower()

        # High importance memory should be included
        assert 'allergic' in memory_text or 'peanut' in memory_text

        # Not all low importance memories may fit
        # (this is more of an integration test)

    @pytest.mark.asyncio
    async def test_context_session_isolation(self, context_orchestrator, memory_facade):
        """Test that contexts are isolated between sessions."""
        session_1 = "test_session_6a"
        session_2 = "test_session_6b"

        # Create different memories in each session
        await memory_facade.process_turn("I am a vegetarian", session_1)
        await memory_facade.process_turn("I am a meat lover", session_2)

        # Build contexts
        context_1 = await context_orchestrator.pack_context(session_1, 1000)
        context_2 = await context_orchestrator.pack_context(session_2, 1000)

        # Verify isolation
        assert context_1 is not None
        assert context_2 is not None

        text_1 = ' '.join(context_1['memory_bullets']).lower()
        text_2 = ' '.join(context_2['memory_bullets']).lower()

        assert 'vegetarian' in text_1
        assert 'meat lover' in text_2
        assert 'vegetarian' not in text_2
        assert 'meat lover' not in text_1

    @pytest.mark.asyncio
    async def test_context_with_summary(self, context_orchestrator, memory_facade):
        """Test context building with session summary."""
        session_id = "test_session_7"
        token_budget = 2000

        # Create many memories to trigger summarization
        for i in range(20):
            await memory_facade.process_turn(f"This is memory number {i} about various topics in my life", session_id)

        # Build context
        context_result = await context_orchestrator.pack_context(session_id, token_budget)

        # Verify summary integration
        assert context_result is not None
        assert 'summary_context' in context_result

        # Should have either summary or memories (or both)
        assert len(context_result['memory_bullets']) > 0 or len(context_result['summary_context']) > 0

    @pytest.mark.asyncio
    async def test_context_performance(self, context_orchestrator, memory_facade):
        """Test context building performance."""
        session_id = "test_session_8"
        token_budget = 1500

        # Create many memories
        for i in range(100):
            await memory_facade.process_turn(f"Test memory {i} with various content", session_id)

        # Test performance
        import time
        start_time = time.time()

        context_result = await context_orchestrator.pack_context(session_id, token_budget)

        build_time = time.time() - start_time

        # Performance assertions
        assert build_time < 2.0, f"Context building took {build_time:.2f}s, should be under 2s"
        assert context_result is not None
        assert context_result['token_count'] <= token_budget

    @pytest.mark.asyncio
    async def test_context_memory_injection(self, context_orchestrator, memory_facade):
        """Test that memories are properly injected into context."""
        session_id = "test_session_9"
        token_budget = 2000

        # Create structured memories
        await memory_facade.process_turn("My name is Bob and I am 30 years old", session_id)
        await memory_facade.process_turn("I work at Tesla as an engineer", session_id)
        await memory_facade.process_turn("I live in Austin, Texas", session_id)

        # Build context
        context_result = await context_orchestrator.pack_context(session_id, token_budget)

        # Verify memory injection
        assert context_result is not None
        assert len(context_result['memory_bullets']) > 0

        # Check for specific memories
        memory_text = ' '.join(context_result['memory_bullets']).lower()
        assert 'bob' in memory_text
        assert 'tesla' in memory_text
        assert 'austin' in memory_text

        # Check for proper formatting
        bullets = context_result['memory_bullets']
        for bullet in bullets:
            assert isinstance(bullet, str)
            assert len(bullet.strip()) > 0

    @pytest.mark.asyncio
    async def test_context_system_message(self, context_orchestrator):
        """Test system message generation."""
        session_id = "test_session_10"
        token_budget = 1000

        # Build context
        context_result = await context_orchestrator.pack_context(session_id, token_budget)

        # Verify system message
        assert context_result is not None
        assert 'system_message' in context_result
        assert len(context_result['system_message']) > 0

        # System message should be properly formatted
        system_message = context_result['system_message']
        assert isinstance(system_message, str)
        assert len(system_message.strip()) > 0

    @pytest.mark.asyncio
    async def test_context_token_counting(self, context_orchestrator, memory_facade):
        """Test accurate token counting."""
        session_id = "test_session_11"

        # Create some memories
        await memory_facade.process_turn("This is a test memory for token counting", session_id)
        await memory_facade.process_turn("Another test memory with more content", session_id)

        # Test with different budgets
        for budget in [500, 1000, 2000]:
            context_result = await context_orchestrator.pack_context(session_id, budget)

            assert context_result is not None
            assert context_result['token_count'] <= budget

            # Token count should be reasonably accurate
            # (allowing for some approximation)
            estimated_tokens = len(context_result['context']) // 4  # Rough estimate
            assert abs(context_result['token_count'] - estimated_tokens) < budget * 0.2

    @pytest.mark.asyncio
    async def test_context_empty_session(self, context_orchestrator):
        """Test context building for empty session."""
        session_id = "test_session_12"
        token_budget = 1000

        # Build context for empty session
        context_result = await context_orchestrator.pack_context(session_id, token_budget)

        # Verify graceful handling
        assert context_result is not None
        assert 'context' in context_result
        assert 'memory_bullets' in context_result

        # Should have basic context structure even without memories
        assert len(context_result['context']) > 0
        assert len(context_result['system_message']) > 0

    @pytest.mark.asyncio
    async def test_context_large_memory_handling(self, context_orchestrator, memory_facade):
        """Test handling of large amounts of memories."""
        session_id = "test_session_13"
        token_budget = 4000

        # Create many memories
        for i in range(200):
            await memory_facade.process_turn(f"Extended memory content number {i} with detailed information about various aspects of life", session_id)

        # Build context
        context_result = await context_orchestrator.pack_context(session_id, token_budget)

        # Verify handling
        assert context_result is not None
        assert context_result['token_count'] <= token_budget

        # Should have some memories included
        assert len(context_result['memory_bullets']) > 0

        # Performance should still be reasonable
        assert len(context_result['memory_bullets']) <= 50  # Should be limited/curated

    @pytest.mark.asyncio
    async def test_context_configuration(self, context_orchestrator):
        """Test context configuration options."""
        session_id = "test_session_14"
        token_budget = 1000

        # Test different configurations
        original_progressive = context_orchestrator.config.progressive_context_enabled

        # Test with progressive mode disabled
        context_orchestrator.config.progressive_context_enabled = False
        context_result_1 = await context_orchestrator.pack_context(session_id, token_budget)
        assert context_result_1 is not None

        # Test with progressive mode enabled
        context_orchestrator.config.progressive_context_enabled = True
        context_result_2 = await context_orchestrator.pack_context(session_id, token_budget)
        assert context_result_2 is not None

        # Restore original configuration
        context_orchestrator.config.progressive_context_enabled = original_progressive

        # Both should produce valid contexts
        assert context_result_1['token_count'] <= token_budget
        assert context_result_2['token_count'] <= token_budget