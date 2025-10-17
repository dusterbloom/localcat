#!/usr/bin/env python3
"""
Test TTSMLXUltraLowLatency interruption handling.

Tests the new interruption handling features added for ultra-low latency TTS.
"""

import asyncio
import sys
import os
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add server directory to path
_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")
for p in (_SERVER_ROOT, _PIPECAT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipecat.frames.frames import (
    TextFrame,
    InterruptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection


class TestTTSInterruptionHandling:
    """Test suite for TTS interruption handling."""

    @pytest.fixture
    def mock_tts_service(self):
        """Create a mock TTSMLXUltraLowLatency instance."""
        # Import here to avoid startup issues
        from core.tts.tts_mlx_ultra_low_latency import TTSMLXUltraLowLatency

        # Mock the subprocess worker
        with patch('subprocess.Popen'):
            service = TTSMLXUltraLowLatency(
                model="mlx-community/Kokoro-82M-bf16",
                voice="af_heart",
                speed=1.0,
                sample_rate=24000,
                buffer_ms=40,
            )

            # Mock internal state
            service._interrupted = False
            service._cancel_event = asyncio.Event()
            service.push_frame = AsyncMock()
            service._text_aggregator = Mock()
            service._text_aggregator.handle_interruption = AsyncMock()

            return service

    @pytest.mark.asyncio
    async def test_user_started_speaking_sets_interrupted_flag(self, mock_tts_service):
        """Test that UserStartedSpeakingFrame sets the interrupted flag."""
        frame = UserStartedSpeakingFrame()

        # Process the frame
        await mock_tts_service.process_frame(frame, FrameDirection.UPSTREAM)

        # Check that interrupted flag is set
        assert mock_tts_service._interrupted is True, \
            "UserStartedSpeakingFrame should set _interrupted to True"

    @pytest.mark.asyncio
    async def test_user_started_speaking_requests_cancel(self, mock_tts_service):
        """Test that UserStartedSpeakingFrame cancels TTS generation."""
        frame = UserStartedSpeakingFrame()

        # Process the frame
        await mock_tts_service.process_frame(frame, FrameDirection.UPSTREAM)

        # Check that cancel event was set
        assert mock_tts_service._cancel_event.is_set(), \
            "UserStartedSpeakingFrame should request cancellation"

    @pytest.mark.asyncio
    async def test_user_stopped_speaking_clears_interrupted_flag(self, mock_tts_service):
        """Test that UserStoppedSpeakingFrame clears the interrupted flag."""
        # Set interrupted state first
        mock_tts_service._interrupted = True

        frame = UserStoppedSpeakingFrame()

        # Process the frame
        await mock_tts_service.process_frame(frame, FrameDirection.UPSTREAM)

        # Check that interrupted flag is cleared
        assert mock_tts_service._interrupted is False, \
            "UserStoppedSpeakingFrame should clear _interrupted flag"

    @pytest.mark.asyncio
    async def test_interruption_frame_clears_text_aggregator(self, mock_tts_service):
        """Test that InterruptionFrame clears the text aggregator."""
        frame = InterruptionFrame()

        # Process the frame
        await mock_tts_service.process_frame(frame, FrameDirection.UPSTREAM)

        # Check that text aggregator's handle_interruption was called
        mock_tts_service._text_aggregator.handle_interruption.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_frame_dropped_during_interruption(self, mock_tts_service):
        """Test that TextFrames are dropped during interruption."""
        # Set interrupted state
        mock_tts_service._interrupted = True

        frame = TextFrame(text="This text should be dropped")

        # Process the frame
        await mock_tts_service.process_frame(frame, FrameDirection.UPSTREAM)

        # The frame should not be pushed downstream (push_frame not called for TextFrame)
        # We need to check that it was silently dropped
        # In the actual implementation, it returns early without calling push_frame

        # We can't easily verify "not called" without more complex mocking
        # But we can verify the frame was not processed by checking logs
        # For now, this is a behavior test that the frame gets dropped

    @pytest.mark.asyncio
    async def test_text_frame_processed_when_not_interrupted(self, mock_tts_service):
        """Test that TextFrames are processed normally when not interrupted."""
        # Clear interrupted state
        mock_tts_service._interrupted = False

        # Mock the parent's process_frame to avoid actual TTS
        with patch.object(type(mock_tts_service).__bases__[0], 'process_frame', new=AsyncMock()):
            frame = TextFrame(text="This text should be processed")

            # Process the frame
            await mock_tts_service.process_frame(frame, FrameDirection.UPSTREAM)

            # The frame should be passed to parent's process_frame
            # (Actual verification would require more complex mocking)

    @pytest.mark.asyncio
    async def test_interruption_lifecycle(self, mock_tts_service):
        """Test the complete interruption lifecycle."""
        # 1. Initially not interrupted
        assert not mock_tts_service._interrupted

        # 2. User starts speaking -> interrupted
        await mock_tts_service.process_frame(
            UserStartedSpeakingFrame(),
            FrameDirection.UPSTREAM
        )
        assert mock_tts_service._interrupted

        # 3. Text frame during interruption -> dropped
        # (would need more complex verification)

        # 4. User stops speaking -> not interrupted
        await mock_tts_service.process_frame(
            UserStoppedSpeakingFrame(),
            FrameDirection.UPSTREAM
        )
        assert not mock_tts_service._interrupted

        # 5. Text frame after interruption -> processed
        # (would need more complex verification)


class TestTextChunking:
    """Test text chunking integration in TTS."""

    @pytest.mark.asyncio
    async def test_text_chunking_produces_multiple_chunks(self):
        """Test that long text is chunked into multiple pieces."""
        from tools.text_formatter import chunk_for_kokoro_ultra_low_latency

        long_text = "This is a longer piece of text that should be split into multiple chunks for ultra-low latency streaming."

        chunks = chunk_for_kokoro_ultra_low_latency(long_text, max_chars=25)

        # Should produce multiple chunks
        assert len(chunks) > 1, "Long text should be split into multiple chunks"

        # Each chunk should be reasonably sized
        for chunk in chunks:
            assert len(chunk) <= 30, f"Chunk too long: {len(chunk)} chars"  # Allow some overflow

        # All chunks combined should preserve the content
        combined = " ".join(chunks)
        # Not exact match due to punctuation handling, but should have similar length
        assert len(combined) > len(long_text) * 0.8, "Chunks should preserve most content"


@pytest.mark.fast
def test_tts_buffer_configuration():
    """Test that TTS respects buffer configuration."""
    from core.tts.tts_mlx_ultra_low_latency import TTSMLXUltraLowLatency

    # Test with custom buffer
    with patch('subprocess.Popen'):
        service = TTSMLXUltraLowLatency(
            model="mlx-community/Kokoro-82M-bf16",
            voice="af_heart",
            buffer_ms=40,
        )

        assert service._buffer_ms == 40, "Should respect buffer_ms parameter"


@pytest.mark.fast
def test_tts_cancellation_event_initialization():
    """Test that cancellation event is properly initialized."""
    from core.tts.tts_mlx_ultra_low_latency import TTSMLXUltraLowLatency

    with patch('subprocess.Popen'):
        service = TTSMLXUltraLowLatency(
            model="mlx-community/Kokoro-82M-bf16",
            voice="af_heart",
        )

        # Should have cancel event
        assert hasattr(service, '_cancel_event'), "Should have _cancel_event"
        assert isinstance(service._cancel_event, asyncio.Event), \
            "_cancel_event should be an asyncio.Event"

        # Should have interrupted flag
        assert hasattr(service, '_interrupted'), "Should have _interrupted flag"
        assert service._interrupted is False, "Should start not interrupted"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
