#!/usr/bin/env python3
"""
Test VisionContextInjector optimizations.

Tests the vision processing optimizations including:
- Image resize with aspect ratio preservation
- Context pruning to limit images
- Frame deduplication
- Keyword filtering
"""

import asyncio
import sys
import os
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from PIL import Image
import io

# Add server directory to path
_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")
for p in (_SERVER_ROOT, _PIPECAT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipecat.frames.frames import InputImageRawFrame, TextFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import FrameDirection


class TestVisionContextInjector:
    """Test suite for VisionContextInjector."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock OpenAILLMContext."""
        context = Mock()
        context.get_messages = Mock(return_value=[])
        context.set_messages = Mock()
        context.add_image_frame_message = Mock()
        return context

    @pytest.fixture
    def injector(self, mock_context):
        """Create a VisionContextInjector instance."""
        from core.video.vision_context_injector import VisionContextInjector

        # Mock environment variables for testing
        with patch.dict(os.environ, {
            'VISION_IMAGE_SIZE': '384',
            'VISION_IMAGE_QUALITY': '85',
            'VISION_MAX_IMAGES_IN_CONTEXT': '2',
            'VISION_ENABLE_DEDUPLICATION': 'true',
        }):
            injector = VisionContextInjector(
                context=mock_context,
                target_fps=0.5,
                inject_on_text=True,
                keyword_filter=False,
            )
            injector.push_frame = AsyncMock()
            return injector

    def create_test_image(self, width=1920, height=1080, color=(255, 0, 0)):
        """Create a test image as raw bytes."""
        img = Image.new('RGB', (width, height), color=color)
        return img.tobytes(), (width, height), 'RGB'

    @pytest.mark.asyncio
    async def test_image_resize_preserves_aspect_ratio(self, injector):
        """Test that image resize maintains aspect ratio."""
        # Create a 1920x1080 image (16:9 aspect ratio)
        original_data, original_size, format = self.create_test_image(1920, 1080)

        # Resize
        resized_data, new_size, output_format = injector._resize_image(
            original_data, original_size, format
        )

        # Check aspect ratio is preserved (within rounding)
        original_ratio = original_size[0] / original_size[1]
        new_ratio = new_size[0] / new_size[1]
        assert abs(original_ratio - new_ratio) < 0.01, \
            f"Aspect ratio not preserved: {original_ratio} -> {new_ratio}"

        # Check that it's actually smaller
        assert new_size[0] <= injector._target_size, \
            f"Width should be <= target size: {new_size[0]} > {injector._target_size}"
        assert new_size[1] <= injector._target_size, \
            f"Height should be <= target size: {new_size[1]} > {injector._target_size}"

    @pytest.mark.asyncio
    async def test_image_resize_reduces_data_size(self, injector):
        """Test that resizing reduces data size."""
        # Create a large image
        original_data, original_size, format = self.create_test_image(1920, 1080)

        # Resize
        resized_data, new_size, output_format = injector._resize_image(
            original_data, original_size, format
        )

        # Check that data size is reduced
        assert len(resized_data) < len(original_data), \
            f"Resized data should be smaller: {len(resized_data)} >= {len(original_data)}"

    @pytest.mark.asyncio
    async def test_image_resize_handles_rgba(self, injector):
        """Test that RGBA images are converted to RGB."""
        # Create an RGBA image
        img = Image.new('RGBA', (1920, 1080), color=(255, 0, 0, 128))
        original_data = img.tobytes()
        original_size = img.size
        format = 'RGBA'

        # Resize
        resized_data, new_size, output_format = injector._resize_image(
            original_data, original_size, format
        )

        # Should convert to RGB
        assert output_format == 'RGB', \
            f"RGBA should be converted to RGB, got: {output_format}"

    @pytest.mark.asyncio
    async def test_image_resize_disabled_when_target_size_negative(self, injector):
        """Test that resize is disabled when target_size is -1."""
        injector._target_size = -1

        original_data, original_size, format = self.create_test_image(1920, 1080)

        # Resize should be no-op
        resized_data, new_size, output_format = injector._resize_image(
            original_data, original_size, format
        )

        # Should return original unchanged
        assert resized_data == original_data
        assert new_size == original_size
        assert output_format == format

    @pytest.mark.asyncio
    async def test_context_pruning_removes_old_images(self, injector, mock_context):
        """Test that old images are pruned when limit exceeded."""
        # Set max images to 2
        injector._max_images = 2
        injector._injected_image_count = 4

        # Mock context with 4 image messages in OpenAI vision format
        mock_context.get_messages.return_value = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/..."}}]},  # oldest
            {"role": "user", "content": "some text"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/..."}}]},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/..."}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/..."}}]},  # newest
        ]

        # Prune old images
        injector._prune_old_images()

        # Should call set_messages with pruned list
        assert mock_context.set_messages.called, \
            "Should update context with pruned messages"

        # Verify the pruned messages - should keep only last 2 images
        pruned_messages = mock_context.set_messages.call_args[0][0]

        # Count remaining image messages
        image_count = 0
        for msg in pruned_messages:
            content = msg.get('content')
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        image_count += 1
                        break

        assert image_count == 2, f"Should keep only 2 images, got {image_count}"

    @pytest.mark.asyncio
    async def test_context_pruning_no_op_when_within_limit(self, injector, mock_context):
        """Test that pruning is skipped when within limit."""
        injector._max_images = 5
        injector._injected_image_count = 2

        # Mock context with 2 image messages (under limit)
        mock_context.get_messages.return_value = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/..."}}]},
            {"role": "user", "content": "some text"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/..."}}]},
        ]

        # Prune should be no-op
        injector._prune_old_images()

        # Should not call set_messages
        assert not mock_context.set_messages.called, \
            "Should not prune when within limit"

    @pytest.mark.asyncio
    async def test_deduplication_skips_identical_frames(self, injector):
        """Test that duplicate frames are skipped."""
        # Enable deduplication
        injector._enable_deduplication = True

        # First image
        image_data = b"test_image_data"

        # Should inject first time
        assert injector._should_inject_image(image_data), \
            "First image should be injected"

        # Should skip second time (duplicate)
        assert not injector._should_inject_image(image_data), \
            "Duplicate image should be skipped"

    @pytest.mark.asyncio
    async def test_deduplication_allows_different_frames(self, injector):
        """Test that different frames are not filtered."""
        injector._enable_deduplication = True

        image_data_1 = b"test_image_data_1"
        image_data_2 = b"test_image_data_2"

        # Should inject first image
        assert injector._should_inject_image(image_data_1), \
            "First image should be injected"

        # Should inject different image
        assert injector._should_inject_image(image_data_2), \
            "Different image should be injected"

    @pytest.mark.asyncio
    async def test_deduplication_disabled(self, injector):
        """Test that deduplication can be disabled."""
        injector._enable_deduplication = False

        image_data = b"test_image_data"

        # Should always inject when disabled
        assert injector._should_inject_image(image_data), \
            "Should inject when dedup disabled (1st)"
        assert injector._should_inject_image(image_data), \
            "Should inject when dedup disabled (2nd)"

    @pytest.mark.asyncio
    async def test_keyword_filtering_detects_vision_keywords(self, injector):
        """Test that vision keywords trigger image injection."""
        injector._keyword_filter = True
        injector._inject_on_text = True

        # Create a test image
        image_data, size, format = self.create_test_image(640, 480)
        injector._last_image = {
            'image': image_data,
            'size': size,
            'format': format
        }

        # Test text with vision keyword
        text_with_keyword = "Can you see this object?"
        frame = TextFrame(text=text_with_keyword)

        # Process frame
        await injector.process_frame(frame, FrameDirection.UPSTREAM)

        # Should inject image (add_image_frame_message called)
        assert injector._context.add_image_frame_message.called, \
            "Should inject image when vision keyword detected"

    @pytest.mark.asyncio
    async def test_keyword_filtering_skips_without_keywords(self, injector, mock_context):
        """Test that images are NOT injected without vision keywords."""
        injector._keyword_filter = True
        injector._inject_on_text = True
        injector._context = mock_context

        # Create a test image
        image_data, size, format = self.create_test_image(640, 480)
        injector._last_image = {
            'image': image_data,
            'size': size,
            'format': format
        }

        # Test text without vision keyword
        text_without_keyword = "Hello, how are you doing today?"
        frame = TextFrame(text=text_without_keyword)

        # Process frame
        await injector.process_frame(frame, FrameDirection.UPSTREAM)

        # Should NOT inject image (add_image_frame_message not called)
        assert not mock_context.add_image_frame_message.called, \
            "Should NOT inject image without vision keyword"

    @pytest.mark.asyncio
    async def test_keyword_filtering_disabled_injects_on_all_text(self, injector):
        """Test that with keyword filtering disabled, all text triggers injection."""
        injector._keyword_filter = False
        injector._inject_on_text = True

        # Create a test image
        image_data, size, format = self.create_test_image(640, 480)
        injector._last_image = {
            'image': image_data,
            'size': size,
            'format': format
        }

        # Test text without vision keyword
        text = "Random text without vision keywords"
        frame = TextFrame(text=text)

        # Process frame
        await injector.process_frame(frame, FrameDirection.UPSTREAM)

        # Should inject image (keyword filter disabled)
        assert injector._context.add_image_frame_message.called, \
            "Should inject image when keyword filter disabled"


@pytest.mark.fast
def test_vision_config_loading():
    """Test that vision configuration loads from environment."""
    from core.video.vision_context_injector import VisionContextInjector

    with patch.dict(os.environ, {
        'VISION_IMAGE_SIZE': '512',
        'VISION_IMAGE_QUALITY': '90',
        'VISION_MAX_IMAGES_IN_CONTEXT': '5',
        'VISION_ENABLE_DEDUPLICATION': 'false',
    }):
        mock_context = Mock()
        injector = VisionContextInjector(
            context=mock_context,
            target_fps=0.5,
        )

        assert injector._target_size == 512, "Should load VISION_IMAGE_SIZE"
        assert injector._image_quality == 90, "Should load VISION_IMAGE_QUALITY"
        assert injector._max_images == 5, "Should load VISION_MAX_IMAGES_IN_CONTEXT"
        assert injector._enable_deduplication is False, "Should load VISION_ENABLE_DEDUPLICATION"


@pytest.mark.fast
def test_vision_config_defaults():
    """Test that vision configuration uses sensible defaults."""
    from core.video.vision_context_injector import VisionContextInjector

    with patch.dict(os.environ, {}, clear=True):
        mock_context = Mock()
        injector = VisionContextInjector(
            context=mock_context,
            target_fps=0.5,
        )

        # Check defaults
        assert injector._target_size == 384, "Default image size should be 384"
        assert injector._image_quality == 85, "Default quality should be 85"
        assert injector._max_images == 2, "Default max images should be 2"
        assert injector._enable_deduplication is True, "Default dedup should be enabled"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
