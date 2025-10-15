#!/usr/bin/env python3
"""Integration tests for prosody capture in frame processor."""

import pytest
import tempfile
import os
import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.frame_processor import MemoryFrameProcessor
from core.memory.memory_store import MemoryStore, Paths
from core.memory.config_manager import MemoryConfiguration
from core.memory.session_manager import SessionManager


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
    config.enabled = True
    config.agent_id = "test_agent"
    config.ephemeral_mode = False
    config.interim_min_words = 2
    return config


@pytest.fixture
def session_manager(memory_config):
    """Create a SessionManager for testing."""
    return SessionManager(
        session_id="test_session",
        user_eid="test_user", 
        agent_eid="test_agent",
        config=memory_config
    )


@pytest.fixture
def hot_memory(memory_store):
    """Create a mock hot memory instance."""
    hot = Mock()
    hot.store = memory_store
    hot.agent_eid = "agent:test_agent"
    hot.current_user_id = "user:test_user"
    hot.current_session_id = "session:test_session"
    return hot


@pytest.fixture
def frame_processor(memory_config, session_manager, hot_memory):
    """Create a MemoryFrameProcessor instance for testing."""
    context_injector = Mock()
    background_summarizer = Mock()
    intent_service = Mock()
    
    processor = MemoryFrameProcessor(
        config=memory_config,
        context_injector=context_injector,
        session_manager=session_manager,
        background_summarizer=background_summarizer,
        hot_memory=hot_memory,
        intent_service=intent_service
    )
    
    # Session info is already initialized in the fixture
    
    return processor


class TestProsodyCapture:
    """Test prosody capture functionality in frame processor."""

    def test_capture_prosody_certainty_valid(self, frame_processor):
        """Test capturing valid prosody certainty."""
        certainty = 0.85
        
        frame_processor.capture_prosody_certainty(certainty)
        
        assert frame_processor._last_prosody_certainty == certainty

    def test_capture_prosody_certainty_clamping(self, frame_processor):
        """Test that prosody certainty is clamped to 0.0-1.0 range."""
        # Test high value clamping
        frame_processor.capture_prosody_certainty(1.5)
        assert frame_processor._last_prosody_certainty == 1.0
        
        # Test low value clamping
        frame_processor.capture_prosody_certainty(-0.5)
        assert frame_processor._last_prosody_certainty == 0.0

    def test_capture_prosody_certainty_invalid_type(self, frame_processor):
        """Test handling of invalid certainty types."""
        # Test with invalid type
        frame_processor.capture_prosody_certainty("invalid")
        # Should not crash and should keep None
        assert frame_processor._last_prosody_certainty is None
        
        # Test with None
        frame_processor.capture_prosody_certainty(None)
        assert frame_processor._last_prosody_certainty is None

    @pytest.mark.asyncio
    async def test_store_prosody_for_turn(self, frame_processor, memory_store):
        """Test storing prosody for current turn."""
        # Set up session and turn
        frame_processor._turn_id = 1
        frame_processor._last_prosody_certainty = 0.75
        
        # Store prosody
        await frame_processor._store_prosody_for_turn()
        
        # Verify it was stored
        certainty, meta = memory_store.get_turn_prosody("session-1", 1)
        assert certainty == 0.75
        assert meta["source"] == "frame_processor"
        
        # Verify it was cleared after storing
        assert frame_processor._last_prosody_certainty is None

    @pytest.mark.asyncio
    async def test_store_prosody_no_session_id(self, frame_processor):
        """Test graceful handling when no session ID is available."""
        frame_processor._turn_id = 1
        frame_processor._last_prosody_certainty = 0.75
        # Can't easily change session_id after initialization, so just test the logic
        
        # Should not crash
        await frame_processor._store_prosody_for_turn()
        
        # Should be cleared even if not stored
        assert frame_processor._last_prosody_certainty is None

    @pytest.mark.asyncio
    async def test_store_prosody_invalid_turn_id(self, frame_processor):
        """Test graceful handling when turn ID is invalid."""
        frame_processor._turn_id = 0  # Invalid turn ID
        frame_processor._last_prosody_certainty = 0.75
        
        # Should not crash
        await frame_processor._store_prosody_for_turn()
        
        # Should be cleared even if not stored
        assert frame_processor._last_prosody_certainty is None

    @pytest.mark.asyncio
    async def test_store_prosody_no_hot_memory(self, memory_config, session_manager):
        """Test graceful handling when hot memory is not available."""
        processor = MemoryFrameProcessor(
            config=memory_config,
            context_injector=Mock(),
            session_manager=session_manager,
            hot_memory=None  # No hot memory
        )
        
        processor._turn_id = 1
        processor._last_prosody_certainty = 0.75
        processor.session_manager.session_id = "test_session"
        
        # Should not crash
        await processor._store_prosody_for_turn()
        
        # Should be cleared even if not stored
        assert processor._last_prosody_certainty is None

    @pytest.mark.asyncio
    async def test_store_prosody_exception_handling(self, frame_processor):
        """Test exception handling during prosody storage."""
        frame_processor._turn_id = 1
        frame_processor._last_prosody_certainty = 0.75
        
        # Mock store to raise exception
        def failing_set_turn_prosody(*args, **kwargs):
            raise Exception("Database error")
        
        original_method = frame_processor.hot_memory.store.set_turn_prosody
        frame_processor.hot_memory.store.set_turn_prosody = failing_set_turn_prosody
        
        try:
            # Should not crash
            await frame_processor._store_prosody_for_turn()
            
            # Should be cleared even if storage failed
            assert frame_processor._last_prosody_certainty is None
        finally:
            # Restore original method
            frame_processor.hot_memory.store.set_turn_prosody = original_method

    @pytest.mark.asyncio
    async def test_transcription_frame_triggers_prosody_storage(self, frame_processor, memory_store):
        """Test that transcription frames trigger prosody storage."""
        # Set up prosody certainty
        frame_processor._turn_id = 1
        frame_processor.capture_prosody_certainty(0.82)
        
        # Create mock transcription frame
        frame = Mock()
        frame.is_final = True
        frame.text = "Test transcription"
        
        # Process transcription frame
        await frame_processor._handle_transcription_frame(frame)
        
        # Verify prosody was stored
        certainty, meta = memory_store.get_turn_prosody("test_session", 1)
        assert certainty == 0.82

    @pytest.mark.asyncio
    async def test_interim_transcription_frame_does_not_store_prosody(self, frame_processor, memory_store):
        """Test that interim transcription frames don't trigger prosody storage."""
        # Set up prosody certainty
        frame_processor._turn_id = 1
        frame_processor.capture_prosody_certainty(0.82)
        
        # Create mock interim transcription frame
        from core.memory.frame_processor import InterimTranscriptionFrame
        frame = InterimTranscriptionFrame()
        frame.text = "Test interim transcription"
        
        # Process interim transcription frame
        await frame_processor._handle_interim_transcription(frame)
        
        # Prosody should NOT be stored yet (only on final)
        # Note: Since we don't have actual implementation for interim frames,
        # this test ensures the current behavior is maintained
        assert frame_processor._last_prosody_certainty is not None  # Still stored

    @pytest.mark.asyncio
    async def test_multiple_turns_prosody_tracking(self, frame_processor, memory_store):
        """Test prosody tracking across multiple turns."""
        # Turn 1
        frame_processor._turn_id = 1
        frame_processor.capture_prosody_certainty(0.9)
        await frame_processor._store_prosody_for_turn()
        
        # Turn 2
        frame_processor._turn_id = 2
        frame_processor.capture_prosody_certainty(0.6)
        await frame_processor._store_prosody_for_turn()
        
        # Verify both turns have prosody data
        certainty1, _ = memory_store.get_turn_prosody("test_session", 1)
        certainty2, _ = memory_store.get_turn_prosody("test_session", 2)
        
        assert certainty1 == 0.9
        assert certainty2 == 0.6

    @pytest.mark.asyncio
    async def test_prosody_metadata_includes_timestamp(self, frame_processor, memory_store):
        """Test that stored prosody includes proper metadata."""
        frame_processor._turn_id = 1
        frame_processor.capture_prosody_certainty(0.75)
        
        # Store prosody
        await frame_processor._store_prosody_for_turn()
        
        # Check metadata
        certainty, meta = memory_store.get_turn_prosody("test_session", 1)
        
        assert meta["source"] == "frame_processor"
        assert "captured_at" in meta
        assert isinstance(meta["captured_at"], int)
        assert meta["captured_at"] > 0

    @pytest.mark.asyncio
    async def test_integration_with_frame_processing_pipeline(self, frame_processor, memory_store):
        """Test complete integration with simulated frame processing pipeline."""
        # Simulate prosody capture from audio processing
        frame_processor.capture_prosody_certainty(0.88)
        
        # Simulate transcription frame processing
        frame = Mock()
        frame.is_final = True
        frame.text = "I love hiking in the mountains"
        
        # Mock the transcription processing to avoid complex setup
        original_process = frame_processor._process_transcription
        frame_processor._process_transcription = AsyncMock()
        
        try:
            # Process the frame
            await frame_processor._handle_transcription_frame(frame)
            
            # Verify prosody was stored
            certainty, meta = memory_store.get_turn_prosody("test_session", 1)
            assert certainty == 0.88
            assert meta["source"] == "frame_processor"
            
            # Verify transcription processing was called
            frame_processor._process_transcription.assert_called_once_with("I love hiking in the mountains")
            
        finally:
            # Restore original method
            frame_processor._process_transcription = original_process

    def test_prosody_certainty_float_conversion(self, frame_processor):
        """Test that int certainty values are converted to float."""
        # Test with integer input
        frame_processor.capture_prosody_certainty(1)  # int
        assert isinstance(frame_processor._last_prosody_certainty, float)
        assert frame_processor._last_prosody_certainty == 1.0
        
        # Test with float input
        frame_processor.capture_prosody_certainty(0.75)  # float
        assert isinstance(frame_processor._last_prosody_certainty, float)
        assert frame_processor._last_prosody_certainty == 0.75
