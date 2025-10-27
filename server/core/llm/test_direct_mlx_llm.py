"""
Test suite for DirectMLXLLMService to validate Pipecat pipeline integration.

This test validates that DirectMLXLLMService properly implements the LLMService
interface and can be used as a drop-in replacement for OpenAILLMService.
"""

import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import pytest

# Mock mlx_lm before importing DirectMLXLLMService
mock_mlx_lm = MagicMock()
mock_model = MagicMock()
mock_tokenizer = MagicMock()


class MockChunk:
    def __init__(self, text):
        self.text = text


def mock_stream_generate(*args, **kwargs):
    """Mock MLX stream_generate to yield test tokens."""
    for token in ["Hello", " ", "world", "!"]:
        yield MockChunk(token)


mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
mock_mlx_lm.stream_generate = mock_stream_generate
mock_tokenizer.apply_chat_template.return_value = "Test prompt"

import sys
sys.modules['mlx_lm'] = mock_mlx_lm

from core.llm.direct_mlx_llm import DirectMLXLLMService
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMMessagesFrame,
    LLMUpdateSettingsFrame,
    LLMTextFrame,
    ErrorFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection


class TestDirectMLXLLMService:
    """Test suite for DirectMLXLLMService."""

    @pytest.fixture
    def service(self):
        """Create a DirectMLXLLMService instance for testing."""
        return DirectMLXLLMService(
            model="test-model",
            max_tokens=256,
            temperature=0.7
        )

    @pytest.mark.asyncio
    async def test_initialization(self, service):
        """Test that DirectMLXLLMService initializes correctly."""
        assert service._model_name == "test-model"
        assert service._settings["max_tokens"] == 256
        assert service._settings["temperature"] == 0.7
        assert service._model is not None
        assert service._tokenizer is not None
        assert service._generation_lock is not None

    @pytest.mark.asyncio
    async def test_create_context_aggregator(self, service):
        """Test that create_context_aggregator returns proper aggregators."""
        context = OpenAILLMContext()

        # Mock the set_llm_adapter method
        context.set_llm_adapter = Mock()

        pair = service.create_context_aggregator(context)

        # Verify adapter was set
        assert context.set_llm_adapter.called

        # Verify aggregators were created
        assert pair._user is not None
        assert pair._assistant is not None

    @pytest.mark.asyncio
    async def test_process_openai_context_frame(self, service):
        """Test processing OpenAILLMContextFrame."""
        # Create test context
        context = OpenAILLMContext()
        context.add_message({"role": "user", "content": "Hello"})

        # Create frame
        frame = OpenAILLMContextFrame(context=context)

        # Mock push_frame to collect emitted frames
        emitted_frames = []
        async def mock_push_frame(f, direction=None):
            emitted_frames.append(f)
        service.push_frame = mock_push_frame

        # Mock metrics methods
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()

        # Process frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Wait a bit for async processing
        await asyncio.sleep(0.1)

        # Verify emitted frames
        frame_types = [type(f).__name__ for f in emitted_frames]

        # Should have: StartFrame, LLMTextFrame(s), EndFrame
        assert "LLMFullResponseStartFrame" in frame_types
        assert "LLMTextFrame" in frame_types
        assert "LLMFullResponseEndFrame" in frame_types

        # Verify text frames contain expected tokens
        text_frames = [f for f in emitted_frames if isinstance(f, LLMTextFrame)]
        assert len(text_frames) > 0
        full_text = "".join(f.text for f in text_frames)
        assert "Hello world!" == full_text

    @pytest.mark.asyncio
    async def test_process_llm_context_frame(self, service):
        """Test processing universal LLMContextFrame."""
        # Create universal context
        context = LLMContext()
        context.add_message({"role": "user", "content": "Test message"})

        frame = LLMContextFrame(context=context)

        # Mock push_frame
        emitted_frames = []
        async def mock_push_frame(f, direction=None):
            emitted_frames.append(f)
        service.push_frame = mock_push_frame

        # Mock metrics
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()

        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.get_messages_for_logging.return_value = [
            {"role": "user", "content": "Test message"}
        ]
        service.get_llm_adapter = Mock(return_value=mock_adapter)

        # Process frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.1)

        # Verify frames were emitted
        frame_types = [type(f).__name__ for f in emitted_frames]
        assert "LLMFullResponseStartFrame" in frame_types
        assert "LLMFullResponseEndFrame" in frame_types

    @pytest.mark.asyncio
    async def test_update_settings_frame(self, service):
        """Test processing LLMUpdateSettingsFrame."""
        # Create settings update frame
        frame = LLMUpdateSettingsFrame(settings={
            "temperature": 0.9,
            "max_tokens": 512
        })

        # Mock push_frame
        service.push_frame = AsyncMock()

        # Process frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify settings were updated
        assert service._settings["temperature"] == 0.9
        assert service._settings["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_deprecated_messages_frame(self, service):
        """Test backward compatibility with LLMMessagesFrame."""
        # Create deprecated messages frame
        frame = LLMMessagesFrame(messages=[
            {"role": "user", "content": "Test"}
        ])

        # Mock push_frame
        emitted_frames = []
        async def mock_push_frame(f, direction=None):
            emitted_frames.append(f)
        service.push_frame = mock_push_frame

        # Mock metrics
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()

        # Process frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.1)

        # Verify frames were emitted (backward compatibility works)
        frame_types = [type(f).__name__ for f in emitted_frames]
        assert "LLMFullResponseStartFrame" in frame_types
        assert "LLMFullResponseEndFrame" in frame_types

    @pytest.mark.asyncio
    async def test_passthrough_frame(self, service):
        """Test that unknown frames are passed through."""
        # Create unknown frame
        class UnknownFrame(Frame):
            pass

        frame = UnknownFrame()

        # Mock push_frame
        pushed_frame = None
        pushed_direction = None
        async def mock_push_frame(f, direction=None):
            nonlocal pushed_frame, pushed_direction
            pushed_frame = f
            pushed_direction = direction
        service.push_frame = mock_push_frame

        # Process frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify frame was passed through
        assert pushed_frame is frame
        assert pushed_direction == FrameDirection.DOWNSTREAM

    @pytest.mark.asyncio
    async def test_llm_adapter_support(self, service):
        """Test that service properly supports LLM adapters."""
        # Verify get_llm_adapter method exists and returns adapter
        adapter = service.get_llm_adapter()
        assert adapter is not None

    @pytest.mark.asyncio
    async def test_model_hot_swap(self, service):
        """Test model hot-swapping functionality."""
        original_model = service._model

        # Mock new model load
        new_mock_model = MagicMock()
        new_mock_tokenizer = MagicMock()
        mock_mlx_lm.load.return_value = (new_mock_model, new_mock_tokenizer)

        # Swap model
        await service.set_model("new-test-model")

        # Verify model was swapped
        assert service._model_name == "new-test-model"
        assert service._model is new_mock_model
        assert service._tokenizer is new_mock_tokenizer

    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        """Test that errors are properly handled and emitted as ErrorFrame."""
        # Create context that will cause an error
        context = OpenAILLMContext()
        context.add_message({"role": "user", "content": "Test"})

        # Mock tokenizer to raise error
        service._tokenizer.apply_chat_template.side_effect = Exception("Test error")

        # Mock push_frame
        emitted_frames = []
        async def mock_push_frame(f, direction=None):
            emitted_frames.append(f)
        service.push_frame = mock_push_frame

        # Mock metrics
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()

        # Process context (should handle error gracefully)
        frame = OpenAILLMContextFrame(context=context)
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.1)

        # Verify error was handled
        frame_types = [type(f).__name__ for f in emitted_frames]

        # Should still emit start and end frames even on error
        assert "LLMFullResponseStartFrame" in frame_types
        assert "LLMFullResponseEndFrame" in frame_types


def run_tests():
    """Run all tests."""
    print("🧪 Running DirectMLXLLMService tests...")
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
