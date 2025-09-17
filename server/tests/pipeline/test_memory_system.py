"""
Comprehensive tests for the memory system components.

These tests focus on memory creation, retrieval, quality filtering,
and overall memory management functionality.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

from components.memory.hotmemory_facade import HotMemoryFacade
from components.memory.memory_store import MemoryStore
from components.session.session_store import SessionStore
from components.memory.enhanced_rule_classifier_v2 import EnhancedRuleClassifierV2

class TestMemorySystem:
    """Test suite for memory system components."""

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
    async def test_memory_creation(self, memory_facade):
        """Test basic memory creation functionality."""
        session_id = "test_session_1"
        input_text = "My name is John and I live in New York"

        # Process the turn
        result = await memory_facade.process_turn(input_text, session_id)

        # Verify memory was created
        assert result is not None
        assert 'created_memories' in result
        assert len(result['created_memories']) > 0

        # Check entity extraction
        entities = [mem for mem in result['created_memories'] if mem.get('type') == 'entity']
        assert len(entities) >= 2  # Should extract "John" and "New York"

    @pytest.mark.asyncio
    async def test_memory_retrieval(self, memory_facade):
        """Test memory retrieval functionality."""
        session_id = "test_session_2"

        # First, create some memories
        await memory_facade.process_turn("I have a dog named Max", session_id)
        await memory_facade.process_turn("Max is a golden retriever", session_id)

        # Now test retrieval
        query_text = "Tell me about my dog"
        intent_result = await memory_facade.intent_classifier.classify_intent(query_text, session_id)

        retrieved_memories = await memory_facade.retrieve_relevant_memories(
            query_text, session_id, intent_result
        )

        assert retrieved_memories is not None
        assert len(retrieved_memories) > 0

        # Check that retrieved memories are relevant
        dog_memories = [mem for mem in retrieved_memories if 'dog' in str(mem).lower() or 'max' in str(mem).lower()]
        assert len(dog_memories) > 0

    @pytest.mark.asyncio
    async def test_memory_quality_filtering(self, memory_facade):
        """Test memory quality filtering."""
        session_id = "test_session_3"

        # Process various quality inputs
        high_quality = "My daughter Emma is studying computer science at Stanford University"
        low_quality = "um like yeah whatever"
        medium_quality = "I work at Google as a software engineer"

        await memory_facade.process_turn(high_quality, session_id)
        await memory_facade.process_turn(low_quality, session_id)
        await memory_facade.process_turn(medium_quality, session_id)

        # Test retrieval quality filtering
        query_text = "What do you know about me?"
        intent_result = await memory_facade.intent_classifier.classify_intent(query_text, session_id)

        retrieved_memories = await memory_facade.retrieve_relevant_memories(
            query_text, session_id, intent_result
        )

        # Should filter out low quality memories
        assert retrieved_memories is not None
        # High and medium quality memories should be present
        memory_texts = [str(mem).lower() for mem in retrieved_memories]
        assert any('emma' in text or 'stanford' in text for text in memory_texts)
        assert any('google' in text for text in memory_texts)

    @pytest.mark.asyncio
    async def test_memory_persistence(self, memory_store, session_store):
        """Test that memories persist across sessions."""
        session_id = "test_session_4"

        # Create a classifier and facade
        classifier = EnhancedRuleClassifierV2()
        facade = HotMemoryFacade(
            memory_store=memory_store,
            session_store=session_store,
            intent_classifier=classifier
        )

        # Create memories
        await facade.process_turn("I love playing tennis on weekends", session_id)

        # Create new facade instance (simulating restart)
        new_facade = HotMemoryFacade(
            memory_store=memory_store,
            session_store=session_store,
            intent_classifier=classifier
        )

        # Test retrieval with new instance
        query_text = "What are my hobbies?"
        intent_result = await new_facade.intent_classifier.classify_intent(query_text, session_id)

        retrieved_memories = await new_facade.retrieve_relevant_memories(
            query_text, session_id, intent_result
        )

        assert retrieved_memories is not None
        assert len(retrieved_memories) > 0
        memory_texts = [str(mem).lower() for mem in retrieved_memories]
        assert any('tennis' in text for text in memory_texts)

    @pytest.mark.asyncio
    async def test_memory_concurrent_access(self, memory_facade):
        """Test concurrent memory operations."""
        session_id = "test_session_5"

        # Create multiple concurrent memory operations
        tasks = []
        test_inputs = [
            "I have a cat named Whiskers",
            "Whiskers is black and white",
            "My cat loves to sleep in the sun",
            "I feed Whiskers twice a day"
        ]

        for input_text in test_inputs:
            task = memory_facade.process_turn(input_text, session_id)
            tasks.append(task)

        # Wait for all operations to complete
        results = await asyncio.gather(*tasks)

        # Verify all operations succeeded
        assert len(results) == len(test_inputs)
        for result in results:
            assert result is not None
            assert 'created_memories' in result

        # Test retrieval
        query_text = "Tell me about my cat"
        intent_result = await memory_facade.intent_classifier.classify_intent(query_text, session_id)

        retrieved_memories = await memory_facade.retrieve_relevant_memories(
            query_text, session_id, intent_result
        )

        assert retrieved_memories is not None
        assert len(retrieved_memories) >= 2  # Should have multiple cat-related memories

    @pytest.mark.asyncio
    async def test_memory_relationship_extraction(self, memory_facade):
        """Test that relationships between entities are properly extracted."""
        session_id = "test_session_6"
        input_text = "My brother Michael works at Apple as a designer"

        result = await memory_facade.process_turn(input_text, session_id)

        # Check that relationships were extracted
        assert result is not None
        assert 'created_memories' in result

        # Look for relationship memories
        relations = [mem for mem in result['created_memories'] if mem.get('type') == 'relation']
        assert len(relations) > 0

        # Verify specific relationship
        relation_found = False
        for relation in relations:
            relation_data = relation.get('data', {})
            if (relation_data.get('subject') == 'Michael' and
                relation_data.get('object') == 'Apple' and
                'works_at' in relation_data.get('relation_type', '').lower()):
                relation_found = True
                break

        assert relation_found, "Should have extracted 'Michael works_at Apple' relationship"

    @pytest.mark.asyncio
    async def test_memory_session_isolation(self, memory_facade):
        """Test that memories are properly isolated between sessions."""
        session_1 = "test_session_7a"
        session_2 = "test_session_7b"

        # Create memories in different sessions
        await memory_facade.process_turn("I am a doctor", session_1)
        await memory_facade.process_turn("I am a teacher", session_2)

        # Test retrieval from session 1
        query_text = "What is my profession?"
        intent_result = await memory_facade.intent_classifier.classify_intent(query_text, session_1)

        memories_1 = await memory_facade.retrieve_relevant_memories(
            query_text, session_1, intent_result
        )

        # Test retrieval from session 2
        memories_2 = await memory_facade.retrieve_relevant_memories(
            query_text, session_2, intent_result
        )

        # Verify session isolation
        assert memories_1 is not None
        assert memories_2 is not None

        texts_1 = [str(mem).lower() for mem in memories_1]
        texts_2 = [str(mem).lower() for mem in memories_2]

        assert any('doctor' in text for text in texts_1)
        assert not any('doctor' in text for text in texts_2)
        assert any('teacher' in text for text in texts_2)
        assert not any('teacher' in text for text in texts_1)

    @pytest.mark.asyncio
    async def test_memory_update_and_correction(self, memory_facade):
        """Test memory update and correction functionality."""
        session_id = "test_session_8"

        # Create initial memory
        await memory_facade.process_turn("I work at Microsoft as a developer", session_id)

        # Test correction
        correction_text = "Actually, I work at Google, not Microsoft"
        result = await memory_facade.process_turn(correction_text, session_id)

        assert result is not None
        assert 'corrections' in result or 'updated_memories' in result

        # Test retrieval to see if correction was applied
        query_text = "Where do I work?"
        intent_result = await memory_facade.intent_classifier.classify_intent(query_text, session_id)

        retrieved_memories = await memory_facade.retrieve_relevant_memories(
            query_text, session_id, intent_result
        )

        assert retrieved_memories is not None
        memory_texts = [str(mem).lower() for mem in retrieved_memories]

        # Should find Google, not Microsoft (or both with correction context)
        google_found = any('google' in text for text in memory_texts)
        assert google_found, "Should find corrected information about working at Google"

    @pytest.mark.asyncio
    async def test_memory_search_performance(self, memory_facade):
        """Test memory search performance."""
        session_id = "test_session_9"

        # Create many memories
        for i in range(50):
            await memory_facade.process_turn(f"This is test memory number {i} about various topics", session_id)

        # Test search performance
        import time
        start_time = time.time()

        query_text = "What do you know about test memories?"
        intent_result = await memory_facade.intent_classifier.classify_intent(query_text, session_id)

        retrieved_memories = await memory_facade.retrieve_relevant_memories(
            query_text, session_id, intent_result
        )

        search_time = time.time() - start_time

        # Performance assertions
        assert search_time < 1.0, f"Search took {search_time:.2f}s, should be under 1s"
        assert retrieved_memories is not None
        assert len(retrieved_memories) > 0

    @pytest.mark.asyncio
    async def test_memory_quality_scoring(self, memory_facade):
        """Test memory quality scoring mechanisms."""
        session_id = "test_session_10"

        # Test inputs with different quality levels
        high_quality = "My daughter Sarah graduated from MIT with a degree in aerospace engineering"
        medium_quality = "I like to watch movies sometimes"
        low_quality = "lol yeah whatever"

        # Process each input
        result_high = await memory_facade.process_turn(high_quality, session_id)
        result_medium = await memory_facade.process_turn(medium_quality, session_id)
        result_low = await memory_facade.process_turn(low_quality, session_id)

        # Check quality scoring
        assert result_high is not None
        assert result_medium is not None
        assert result_low is not None

        # High quality should produce more/better memories
        high_memories = result_high.get('created_memories', [])
        medium_memories = result_medium.get('created_memories', [])
        low_memories = result_low.get('created_memories', [])

        # High quality should have more entities/relations
        assert len(high_memories) >= len(medium_memories)
        assert len(medium_memories) >= len(low_memories)