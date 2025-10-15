"""Video frame throttler for vision models."""
import asyncio
import os
from loguru import logger
from pipecat.frames.frames import Frame, InputImageRawFrame, UserImageRawFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.pipeline import FrameDirection


class VideoFrameThrottler(FrameProcessor):
    """Throttle video frames to reduce LLM load. Vision models don't need 30fps."""

    def __init__(self, target_fps: float = 2.0):
        super().__init__()
        self._target_fps = target_fps
        self._frame_interval = 1.0 / target_fps
        self._last_frame_time = 0.0
        logger.info(f"[VideoThrottler] Initialized (target_fps={target_fps})")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # CRITICAL: Always call parent first (handles StartFrame and system frames)
        await super().process_frame(frame, direction)

        # Handle video frames specifically
        if isinstance(frame, InputImageRawFrame):
            current_time = asyncio.get_event_loop().time()
            if current_time - self._last_frame_time >= self._frame_interval:
                self._last_frame_time = current_time
                logger.debug(f"[VideoThrottler] Processing frame: {frame.size} ({frame.format})")

                # Convert to UserImageRawFrame for LLM vision processing
                user_frame = UserImageRawFrame(
                    image=frame.image,
                    size=frame.size,
                    format=frame.format,
                    user_id=os.getenv("USER_ID", "user"),
                )
                await self.push_frame(user_frame, direction)
            # Drop frame if too soon (don't push)
        else:
            # Pass through all non-image frames
            await self.push_frame(frame, direction)
