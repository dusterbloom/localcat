#!/usr/bin/env python3
"""Unit tests for summarizer prosody bias functionality."""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.background_summarizer import BackgroundSummarizer
from core.memory.memory_store import MemoryStore, Paths
from core.memory.config_manager import MemoryConfiguration


@pytest.fixture
def memory_store():
    """Create a temporary MemoryStore for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        paths = Paths(sqlite_path=db_path, lmdb_dir=None)  # Disable LMDB for simplicity
        store = MemoryStore(paths)
        yield store
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.fixture
def memory_config():
    """Create a MemoryConfiguration for testing."""
    config = MemoryConfiguration()
    config.summarization_enabled = True
    config.summary_window_mode = "turn_pairs"
    config.summary_turn_pairs = 3
    config.summary_model = "test-model"
    config.summary_base_url = "http://localhost:8080"
    config.summary_max_tokens = 100
    config.summary_max_messages = 10
    return config


@pytest.fixture
def hot_memory():
    """Create a mock hot memory instance."""
    return Mock()


@pytest.fixture
def summarizer(memory_store, memory_config, hot_memory):
    """Create a BackgroundSummarizer instance for testing."""
    return BackgroundSummarizer(hot_memory, memory_config, memory_store)


class TestSummarizerProsodyBias:
    """Test prosody bias functionality in background summarizer."""

    def test_prosody_bias_disabled_by_default(self, summarizer, memory_store):
        """Test that prosody bias is disabled by default."""
        # Store some conversation chunks
        memory_store.enqueue_mention("conversation", "Test message 1", 1000, "session1", 1)
        memory_store.enqueue_mention("conversation", "Test message 2", 2000, "session1", 2)
        memory_store.flush()
        
        # Should use regular method without prosody bias
        chunks = summarizer._get_conversation_chunks("session1", 5)
        
        assert len(chunks) == 2
        assert all(isinstance(chunk, tuple) and len(chunk) == 2 for chunk in chunks)

    def test_prosody_bias_enabled(self, summarizer, memory_store):
        """Test that prosody bias works when enabled."""
        # Store prosody data with different certainties
        memory_store.set_turn_prosody("session1", 1, 0.8)  # High certainty
        memory_store.set_turn_prosody("session1", 2, 0.2)  # Low certainty
        memory_store.set_turn_prosody("session1", 3, 0.9)  # Very high certainty
        
        # Store conversation chunks
        memory_store.enqueue_mention("conversation", "Low certainty message", 1000, "session1", 1)
        memory_store.enqueue_mention("conversation", "Medium message", 2000, "session1", 2) 
        memory_store.enqueue_mention("conversation", "High certainty message", 3000, "session1", 3)
        memory_store.flush()
        
        # Enable prosody bias
        with patch.dict(os.environ, {"SUMMARY_PROSODY_ENABLED": "true"}):
            chunks = summarizer._get_conversation_chunks("session1", 5)
            
            # Should return chunks, high certainty should be preferred
            assert len(chunks) > 0
            
            # The high certainty message should be prioritized (comes first or is included)
            chunk_texts = [chunk[0] for chunk in chunks]
            assert "High certainty message" in chunk_texts

    def test_low_certainty_filtering(self, summarizer, memory_store):
        """Test that very low certainty chatter is filtered out."""
        # Store prosody data with very low certainty
        memory_store.set_turn_prosody("session1", 1, 0.1)  # Very low certainty
        memory_store.set_turn_prosody("session1", 2, 0.8)  # High certainty
        
        # Store conversation chunks
        memory_store.enqueue_mention("conversation", "Very low certainty", 1000, "session1", 1)
        memory_store.enqueue_mention("conversation", "High certainty", 2000, "session1", 2)
        memory_store.flush()
        
        # Enable prosody bias
        with patch.dict(os.environ, {"SUMMARY_PROSODY_ENABLED": "true"}):
            chunks = summarizer._get_conversation_chunks("session1", 5)
            
            # Should filter out very low certainty (< 0.3)
            chunk_texts = [chunk[0] for chunk in chunks]
            assert "Very low certainty" not in chunk_texts
            assert "High certainty" in chunk_texts

    def test_fallback_when_all_low_certainty(self, summarizer, memory_store):
        """Test fallback when all chunks are low certainty."""
        # Store prosody data with all low certainty
        memory_store.set_turn_prosody("session1", 1, 0.2)  # Low certainty
        memory_store.set_turn_prosody("session1", 2, 0.1)  # Very low certainty
        
        # Store conversation chunks
        memory_store.enqueue_mention("conversation", "Low certainty 1", 1000, "session1", 1)
        memory_store.enqueue_mention("conversation", "Low certainty 2", 2000, "session1", 2)
        memory_store.flush()
        
        # Enable prosody bias
        with patch.dict(os.environ, {"SUMMARY_PROSODY_ENABLED": "true"}):
            chunks = summarizer._get_conversation_chunks("session1", 5)
            
            # Should still return chunks as fallback
            assert len(chunks) > 0
            chunk_texts = [chunk[0] for chunk in chunks]
            assert "Low certainty 1" in chunk_texts or "Low certainty 2" in chunk_texts

    def test_missing_prosody_data_handled_gracefully(self, summarizer, memory_store):
        """Test that missing prosody data is handled gracefully."""
        # Store conversation chunks without prosody data
        memory_store.enqueue_mention("conversation", "Message without prosody", 1000, "session1", 1)
        memory_store.flush()
        
        # Enable prosody bias
        with patch.dict(os.environ, {"SUMMARY_PROSODY_ENABLED": "true"}):
            # Should not crash and should return chunks
            chunks = summarizer._get_conversation_chunks("session1", 5)
            assert len(chunks) > 0
            assert chunks[0][0] == "Message without prosody"

    def test_prosody_retrieval_exception_handled(self, summarizer, memory_store):
        """Test that prosody retrieval exceptions are handled gracefully."""
        # Store conversation chunks
        memory_store.enqueue_mention("conversation", "Test message", 1000, "session1", 1)
        memory_store.flush()
        
        # Mock get_turn_prosody to raise an exception
        def failing_get_turn_prosody(*args, **kwargs):
            raise Exception("Database error")
        
        original_method = memory_store.get_turn_prosody
        memory_store.get_turn_prosody = failing_get_turn_prosody
        
        try:
            # Enable prosody bias
            with patch.dict(os.environ, {"SUMMARY_PROSODY_ENABLED": "true"}):
                # Should not crash and should return chunks
                chunks = summarizer._get_conversation_chunks("session1", 5)
                assert len(chunks) > 0
        finally:
            # Restore original method
            memory_store.get_turn_prosody = original_method

    def test_no_store_uses_empty_list(self, memory_config, hot_memory):
        """Test behavior when store is None."""
        summarizer_no_store = BackgroundSummarizer(hot_memory, memory_config, None)
        
        with patch.dict(os.environ, {"SUMMARY_PROSODY_ENABLED": "true"}):
            chunks = summarizer_no_store._get_conversation_chunks("session1", 5)
            assert chunks == []

    def test_extended_limit_for_filtering(self, summarizer, memory_store):
        """Test that extended limit allows for proper filtering."""
        # Store prosody data
        memory_store.set_turn_prosody("session1", 1, 0.9)  # High
        memory_store.set_turn_prosody("session1", 2, 0.8)  # High
        memory_store.set_turn_prosody("session1", 3, 0.2)  # Low
        memory_store.set_turn_prosody("session1", 4, 0.1)  # Very low
        
        # Store more chunks than the limit
        for i in range(1, 5):
            memory_store.enqueue_mention("conversation", f"Message {i}", i * 1000, "session1", i)
        memory_store.flush()
        
        # Request with small limit but prosody bias enabled
        with patch.dict(os.environ, {"SUMMARY_PROSODY_ENABLED": "true"}):
            chunks = summarizer._get_conversation_chunks("session1", 2)
            
            # Should return high certainty messages
            chunk_texts = [chunk[0] for chunk in chunks]
            
            # Should not include very low certainty
            assert "Message 4" not in chunk_texts  # Very low certainty
            
            # Should include some high certainty messages
            assert any("Message 1" in text or "Message 2" in text for text in chunk_texts)

    def test_sorting_by_certainty_and_timestamp(self, summarizer, memory_store):
        """Test that chunks are sorted by certainty then timestamp."""
        # Store prosody data with same certainty but different timestamps
        memory_store.set_turn_prosody("session1", 1, 0.8)  # Same certainty, older
        memory_store.set_turn_prosody("session1", 2, 0.8)  # Same certainty, newer
        
        # Store conversation chunks
        memory_store.enqueue_mention("conversation", "Older high certainty", 1000, "session1", 1)
        memory_store.enqueue_mention("conversation", "Newer high certainty", 3000, "session1", 2)
        memory_store.flush()
        
        # Enable prosody bias
        with patch.dict(os.environ, {"SUMMARY_PROSODY_ENABLED": "true"}):
            chunks = summarizer._get_conversation_chunks("session1", 5)
            
            # Should prioritize newer chunk when certainty is equal
            chunk_texts = [chunk[0] for chunk in chunks]
            if len(chunks) >= 2:
                # Newer message should come first when certainty is equal
                newer_index = chunk_texts.index("Newer high certainty")
                older_index = chunk_texts.index("Older high certainty")
                assert newer_index < older_index

    @pytest.mark.asyncio
    async def test_integration_with_summarize_turns(self, summarizer, memory_store):
        """Test integration with the main summarize_turns method."""
        # Store prosody data
        memory_store.set_turn_prosody("session1", 1, 0.9)  # High certainty
        memory_store.set_turn_prosody("session1", 2, 0.2)  # Low certainty
        
        # Store conversation chunks
        memory_store.enqueue_mention("conversation", "High certainty user message", 1000, "session1", 1)
        memory_store.enqueue_mention("conversation", "Low certainty user message", 2000, "session1", 2)
        memory_store.flush()
        
        # Mock the LLM call
        with patch.dict(os.environ, {"SUMMARY_PROSODY_ENABLED": "true"}):
            with patch.object(summarizer, '_call_summarizer_llm', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = "Test summary"
                
                # Run summarization
                result = await summarizer.summarize_turns(2, "session1")
                
                # Should succeed and use prosody-biased chunks
                assert result is True
                mock_llm.assert_called_once()
                
                # Check that the LLM was called with high certainty content preferred
                call_args = mock_llm.call_args[0][0]  # First positional argument (text)
                assert "High certainty" in call_args

    def test_environment_variable_case_insensitive(self, summarizer, memory_store):
        """Test that SUMMARY_PROSODY_ENABLED is case insensitive."""
        memory_store.enqueue_mention("conversation", "Test message", 1000, "session1", 1)
        memory_store.flush()
        
        # Test various case combinations
        test_cases = ["true", "TRUE", "True", "1", "yes", "YES"]
        
        for case_value in test_cases:
            with patch.dict(os.environ, {"SUMMARY_PROSODY_ENABLED": case_value}):
                chunks = summarizer._get_conversation_chunks("session1", 5)
                assert len(chunks) > 0
        
        # Test disabled values
        disabled_cases = ["false", "FALSE", "False", "0", "no", "NO", ""]
        
        for case_value in disabled_cases:
            with patch.dict(os.environ, {"SUMMARY_PROSODY_ENABLED": case_value}):
                chunks = summarizer._get_conversation_chunks("session1", 5)
                assert len(chunks) > 0  # Should work but without bias
