"""Vision context injector - adds video frames to LLM context for streaming vision."""
import asyncio
from loguru import logger
from pipecat.frames.frames import Frame, InputImageRawFrame, TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.pipeline import FrameDirection
from typing import Optional


class VisionContextInjector(FrameProcessor):
    """
    Injects video frames into the LLM context for streaming vision.

    For continuous camera input, this processor intercepts InputImageRawFrame
    and directly adds it to the OpenAILLMContext using add_image_frame_message().

    This is different from UserImageRawFrame which is designed for function-call-based vision.
    """

    def __init__(self, context, target_fps: float = 2.0, inject_on_text: bool = True):
        """
        Args:
            context: OpenAILLMContext instance to inject images into
            target_fps: Frame rate for throttling (default 2fps)
            inject_on_text: Only inject images when user sends text (default True)
        """
        super().__init__()
        self._context = context
        self._target_fps = target_fps
        self._frame_interval = 1.0 / target_fps
        self._last_frame_time = 0.0
        self._last_image = None
        self._inject_on_text = inject_on_text
        logger.info(f"[VisionContextInjector] Initialized (fps={target_fps}, inject_on_text={inject_on_text})")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Always call parent first
        await super().process_frame(frame, direction)

        # Handle video frames - store the latest frame
        if isinstance(frame, InputImageRawFrame):
            current_time = asyncio.get_event_loop().time()
            if current_time - self._last_frame_time >= self._frame_interval:
                self._last_frame_time = current_time
                # Store the latest image
                self._last_image = {
                    'image': frame.image,
                    'size': frame.size,
                    'format': frame.format
                }
                logger.debug(f"[VisionContextInjector] Stored frame: {frame.size} ({frame.format})")
            # Don't push the frame downstream - we're handling it here
            return

        # When user sends text (from STT or text input), inject the latest image into context
        if isinstance(frame, (TextFrame, TranscriptionFrame)):
            text = getattr(frame, 'text', '')
            logger.debug(f"[VisionContextInjector] Received text frame: '{text[:50]}...' (has_image={self._last_image is not None})")

            if self._last_image and self._inject_on_text:
                logger.info(f"[VisionContextInjector] Injecting image into context with text: '{text[:50]}...'")
                try:
                    self._context.add_image_frame_message(
                        format=self._last_image['format'],
                        size=self._last_image['size'],
                        image=self._last_image['image'],
                        text=None  # Text is already in a separate frame
                    )
                    logger.info(f"[VisionContextInjector] ✓ Image added to LLM context")
                except Exception as e:
                    logger.error(f"[VisionContextInjector] Failed to add image to context: {e}")

        # Pass through all other frames
        await self.push_frame(frame, direction)
