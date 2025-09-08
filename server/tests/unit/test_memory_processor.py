"""
Unit tests for MemoryProcessor
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from processors.memory_processor import MemoryProcessor, MemoryProcessorConfig
from components.memory.memory_store import MemoryStore
from components.memory.memory_hotpath import HotMemory


class TestMemoryProcessor:
    """Test cases for MemoryProcessor"""
    
    def test_init(self, test_config):
        """Test MemoryProcessor initialization"""
        config = MemoryProcessorConfig(
            sqlite_path=test_config.database.sqlite_path,
            lmdb_dir=test_config.database.lmdb_dir,
            user_id="test-user",
            enable_metrics=True
        )
        
        processor = MemoryProcessor(config)
        
        assert processor.config.user_id == "test-user"
        assert processor.config.enable_metrics is True
        assert processor.config.max_facts_per_injection == 3
        assert processor._metrics['total_processed'] == 0
    
    def test_init_with_defaults(self):
        """Test MemoryProcessor initialization with defaults"""
        config = MemoryProcessorConfig()
        processor = MemoryProcessor(config)
        
        assert processor.config.user_id == "default-user"
        assert processor.config.enable_metrics is True
        assert processor.config.max_facts_per_injection == 3
    
    @pytest.mark.asyncio
    async def test_process_transcription_frame(self, memory_processor, mock_transcription_frame, mock_push_frame):
        """Test processing transcription frame"""
        push_frame, pushed_frames = mock_push_frame
        memory_processor.push_frame = push_frame
        
        # Mock the extraction and storage methods
        memory_processor._extract_memory_facts = AsyncMock(return_value=[
            {"text": "Test fact", "confidence": 0.8}
        ])
        memory_processor._store_memory_facts = AsyncMock()
        
        await memory_processor.process_frame(mock_transcription_frame, None)
        
        # Verify extraction was called
        memory_processor._extract_memory_facts.assert_called_once_with(mock_transcription_frame.text)
        
        # Verify storage was called
        memory_processor._store_memory_facts.assert_called_once()
        
        # Verify frame was passed through
        assert len(pushed_frames) == 1
        assert pushed_frames[0] == mock_transcription_frame
    
    @pytest.mark.asyncio
    async def test_process_llm_messages_frame_with_context(self, memory_processor, mock_llm_messages_frame, mock_push_frame):
        """Test processing LLM messages frame with memory context"""
        push_frame, pushed_frames = mock_push_frame
        memory_processor.push_frame = push_frame
        
        # Mock memory context retrieval
        memory_context = ["User mentioned they work at Acme Corp", "User lives in New York"]
        memory_processor._get_memory_context = AsyncMock(return_value=memory_context)
        
        await memory_processor.process_frame(mock_llm_messages_frame, None)
        
        # Verify context retrieval was called
        memory_processor._get_memory_context.assert_called_once_with(mock_llm_messages_frame.messages)
        
        # Verify enhanced frame was pushed
        assert len(pushed_frames) == 1
        enhanced_frame = pushed_frames[0]
        assert len(enhanced_frame.messages) == len(mock_llm_messages_frame.messages) + 1
        
        # Verify memory context was injected
        memory_message = enhanced_frame.messages[1]  # After system message
        assert memory_message['role'] == 'user'
        assert "Memory Context:" in memory_message['content']
        assert "User mentioned they work at Acme Corp" in memory_message['content']
    
    @pytest.mark.asyncio
    async def test_process_llm_messages_frame_without_context(self, memory_processor, mock_llm_messages_frame, mock_push_frame):
        """Test processing LLM messages frame without memory context"""
        push_frame, pushed_frames = mock_push_frame
        memory_processor.push_frame = push_frame
        
        # Mock empty memory context
        memory_processor._get_memory_context = AsyncMock(return_value=[])
        
        await memory_processor.process_frame(mock_llm_messages_frame, None)
        
        # Verify original frame was passed through
        assert len(pushed_frames) == 1
        assert pushed_frames[0] == mock_llm_messages_frame
    
    @pytest.mark.asyncio
    async def test_extract_memory_facts(self, memory_processor):
        """Test memory fact extraction"""
        test_text = "My name is John and I work at Acme Corporation."
        
        # Mock hot memory extraction
        expected_facts = [
            {"text": "John works at Acme Corporation", "confidence": 0.8}
        ]
        memory_processor.hot_memory.extract_facts = AsyncMock(return_value=expected_facts)
        
        result = await memory_processor._extract_memory_facts(test_text)
        
        assert result == expected_facts
        memory_processor.hot_memory.extract_facts.assert_called_once_with(test_text)
    
    @pytest.mark.asyncio
    async def test_store_memory_facts(self, memory_processor):
        """Test storing memory facts"""
        facts = [
            {"text": "Test fact 1", "type": "general", "confidence": 0.8, "metadata": {}},
            {"text": "Test fact 2", "type": "personal", "confidence": 0.9, "metadata": {}}
        ]
        
        # Mock memory store
        memory_processor.memory_store.add_fact = AsyncMock()
        
        await memory_processor._store_memory_facts(facts)
        
        # Verify each fact was stored
        assert memory_processor.memory_store.add_fact.call_count == 2
        
        # Verify calls with correct parameters
        for i, fact in enumerate(facts):
            call_args = memory_processor.memory_store.add_fact.call_args_list[i]
            assert call_args[1]['user_id'] == memory_processor.config.user_id
            assert call_args[1]['fact_text'] == fact['text']
            assert call_args[1]['fact_type'] == fact['type']
            assert call_args[1]['confidence'] == fact['confidence']
            assert call_args[1]['metadata'] == fact['metadata']
    
    @pytest.mark.asyncio
    async def test_get_memory_context(self, memory_processor):
        """Test getting memory context"""
        messages = [
            {"role": "user", "content": "Hello, I'm John from Acme Corp"},
            {"role": "assistant", "content": "Hello John! How can I help you today?"}
        ]
        
        expected_facts = [
            {"text": "John works at Acme Corporation", "confidence": 0.8}
        ]
        
        # Mock hot memory retrieval
        memory_processor.hot_memory.retrieve_relevant_facts = AsyncMock(return_value=expected_facts)
        
        result = await memory_processor._get_memory_context(messages)
        
        assert result == ["John works at Acme Corporation"]
        memory_processor.hot_memory.retrieve_relevant_facts.assert_called_once()
        
        # Verify the context text was passed correctly
        call_args = memory_processor.hot_memory.retrieve_relevant_facts.call_args[0]
        assert "Hello, I'm John from Acme Corp" in call_args[0]
        assert "Hello John! How can I help you today?" in call_args[0]
    
    def test_inject_memory_context(self, memory_processor):
        """Test injecting memory context into messages"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        memory_context = ["User works at Acme Corp", "User lives in New York"]
        
        result = memory_processor._inject_memory_context(messages, memory_context)
        
        # Verify memory message was injected
        assert len(result) == len(messages) + 1
        memory_message = result[1]  # After system message
        assert memory_message['role'] == 'user'
        assert "Memory Context:" in memory_message['content']
        assert "• User works at Acme Corp" in memory_message['content']
        assert "• User lives in New York" in memory_message['content']
        
        # Verify original messages are preserved
        assert result[0] == messages[0]
        assert result[2] == messages[1]
    
    def test_inject_memory_context_no_system_message(self, memory_processor):
        """Test injecting memory context when no system message exists"""
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        memory_context = ["User works at Acme Corp"]
        
        result = memory_processor._inject_memory_context(messages, memory_context)
        
        # Verify memory message was injected at beginning
        assert len(result) == len(messages) + 1
        memory_message = result[0]
        assert memory_message['role'] == 'user'
        assert "Memory Context:" in memory_message['content']
        
        # Verify original message is preserved
        assert result[1] == messages[0]
    
    def test_inject_memory_context_empty(self, memory_processor):
        """Test injecting empty memory context"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        result = memory_processor._inject_memory_context(messages, [])
        
        # Verify no injection occurred
        assert result == messages
    
    def test_update_metrics(self, memory_processor):
        """Test metrics update"""
        initial_total = memory_processor._metrics['total_processed']
        initial_avg = memory_processor._metrics['avg_latency_ms']
        
        memory_processor._update_metrics(0.1)  # 100ms
        
        # Verify metrics were updated
        assert memory_processor._metrics['total_processed'] == initial_total + 1
        assert memory_processor._metrics['avg_latency_ms'] == 100.0
    
    def test_get_metrics(self, memory_processor):
        """Test getting metrics"""
        # Set some test metrics
        memory_processor._metrics = {
            'total_processed': 10,
            'memory_hits': 5,
            'corrections_applied': 2,
            'avg_latency_ms': 150.0
        }
        
        metrics = memory_processor.get_metrics()
        
        assert metrics['total_processed'] == 10
        assert metrics['memory_hits'] == 5
        assert metrics['corrections_applied'] == 2
        assert metrics['avg_latency_ms'] == 150.0
    
    @pytest.mark.asyncio
    async def test_cleanup(self, memory_processor):
        """Test cleanup"""
        # Mock memory store cleanup
        memory_processor.memory_store.close = AsyncMock()
        
        await memory_processor.cleanup()
        
        memory_processor.memory_store.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_frame_error_handling(self, memory_processor, mock_transcription_frame, mock_push_frame):
        """Test error handling in process_frame"""
        push_frame, pushed_frames = mock_push_frame
        memory_processor.push_frame = push_frame
        
        # Mock extraction to raise an exception
        memory_processor._extract_memory_facts = AsyncMock(side_effect=Exception("Test error"))
        
        # Should not raise exception
        await memory_processor.process_frame(mock_transcription_frame, None)
        
        # Verify frame was still passed through
        assert len(pushed_frames) == 1
        assert pushed_frames[0] == mock_transcription_frame
    
    @pytest.mark.asyncio
    async def test_process_non_text_frame(self, memory_processor, mock_push_frame):
        """Test processing non-text frame"""
        from tests.conftest import MockFrame
        
        push_frame, pushed_frames = mock_push_frame
        memory_processor.push_frame = push_frame
        
        frame = MockFrame("test")
        
        await memory_processor.process_frame(frame, None)
        
        # Verify frame was passed through without processing
        assert len(pushed_frames) == 1
        assert pushed_frames[0] == frame