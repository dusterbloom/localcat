#!/usr/bin/env python3
"""
Streaming Text Aggregator for smooth TTS without hiccups.

This aggregator releases text more frequently than SentenceAggregator,
preventing the long pauses that occur when waiting for sentence boundaries.
"""

import asyncio
import time
from collections import deque
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TextFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class StreamingTextAggregator(FrameProcessor):
    """
    Aggregates text frames and releases them based on time or character count,
    rather than waiting for sentence boundaries. This prevents hiccups in TTS.
    """

    def __init__(self, base_max_time: float = 0.3, base_min_chars: int = 50, base_min_words: int = 8):
        """
        Initialize the intelligent streaming text aggregator with dynamic adjustment.

        Args:
            base_max_time: Base maximum time in seconds to wait before releasing
            base_min_chars: Base minimum characters before considering release
            base_min_words: Base minimum words before considering release
        """
        super().__init__()

        # Base parameters (will be adjusted dynamically)
        self._base_max_time = base_max_time
        self._base_min_chars = base_min_chars
        self._base_min_words = base_min_words

        # Dynamic parameters (adjusted based on performance)
        self._max_time = base_max_time
        self._min_chars = base_min_chars
        self._min_words = base_min_words

        # Performance tracking for intelligent adjustment
        self._tts_response_times = deque(maxlen=10)  # Last 10 TTS response times
        self._chunk_sizes = deque(maxlen=20)  # Last 20 chunk sizes
        self._last_adjustment_time = time.time()
        self._adjustment_interval = 5.0  # Adjust parameters every 5 seconds

        # Aggregation state
        self._aggregation = ""
        self._timer: Optional[asyncio.Task] = None
        self._last_release_time = asyncio.get_event_loop().time()
        self._response_start_time = None

    async def _release_text(self):
        """Release accumulated text to downstream processors."""
        if self._aggregation.strip():
            # Clean asterisks from text before sending to TTS
            clean_text = self._aggregation.replace('*', '').strip()
            if clean_text:  # Only send if there's text left after cleaning
                await self.push_frame(TextFrame(clean_text))

                # Track performance for dynamic adjustment
                chunk_size = len(clean_text)
                self._chunk_sizes.append(chunk_size)

                logger.debug(f"🚀 StreamingTextAggregator released: {chunk_size} chars (dynamic: min_chars={self._min_chars}, max_time={self._max_time:.2f})")

            self._aggregation = ""
            self._last_release_time = asyncio.get_event_loop().time()

        # Cancel any pending timer
        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    async def _schedule_release(self):
        """Schedule a release after max_time seconds."""
        if self._timer and not self._timer.done():
            self._timer.cancel()

        self._timer = asyncio.create_task(self._delayed_release())

    async def _delayed_release(self):
        """Release text after the timeout."""
        try:
            await asyncio.sleep(self._max_time)
            if self._aggregation.strip():
                await self._release_text()
        except asyncio.CancelledError:
            pass

    def _record_tts_performance(self, response_time: float):
        """Record TTS response time for performance analysis."""
        self._tts_response_times.append(response_time)

    def _adjust_parameters_intelligently(self):
        """Dynamically adjust chunking parameters based on performance metrics."""
        current_time = time.time()
        if current_time - self._last_adjustment_time < self._adjustment_interval:
            return  # Don't adjust too frequently

        if not self._tts_response_times or not self._chunk_sizes:
            return  # Not enough data yet

        # Calculate performance metrics
        avg_response_time = sum(self._tts_response_times) / len(self._tts_response_times)
        avg_chunk_size = sum(self._chunk_sizes) / len(self._chunk_sizes)

        # Intelligent adjustment logic
        if avg_response_time > 0.5:  # TTS is slow (>500ms)
            # Use larger chunks to reduce TTS requests
            self._min_chars = min(80, self._base_min_chars + 15)
            self._min_words = min(12, self._base_min_words + 2)
            self._max_time = min(0.5, self._base_max_time + 0.1)
            logger.debug(f"🐌 TTS slow ({avg_response_time:.2f}s), using larger chunks: {self._min_chars} chars, {self._min_words} words")
        elif avg_response_time < 0.2:  # TTS is fast (<200ms)
            # Use smaller chunks for more responsive streaming
            self._min_chars = max(30, self._base_min_chars - 10)
            self._min_words = max(5, self._base_min_words - 1)
            self._max_time = max(0.15, self._base_max_time - 0.05)
            logger.debug(f"🚀 TTS fast ({avg_response_time:.2f}s), using smaller chunks: {self._min_chars} chars, {self._min_words} words")
        else:
            # TTS performance is good, use base parameters
            self._min_chars = self._base_min_chars
            self._min_words = self._base_min_words
            self._max_time = self._base_max_time
            logger.debug(f"✅ TTS optimal ({avg_response_time:.2f}s), using base parameters")

        self._last_adjustment_time = current_time

    def _should_release_text(self, text: str) -> bool:
        """Determine if accumulated text should be released based on natural boundaries."""
        if len(text) < self._min_chars:  # Configurable minimum chunk size
            return False

        # Release at natural punctuation boundaries (more comprehensive)
        if text.rstrip().endswith((',', ';', ':', '—', '–', '-', '(', ')', '"', "'")):
            return True

        # Release at sentence endings (but not requiring full stops for streaming)
        if text.rstrip().endswith(('!', '?')):
            return True

        # Release at word boundaries (configurable minimum words)
        words = text.split()
        if len(words) >= self._min_words:
            return True

        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and aggregate text with smart releasing."""
        await super().process_frame(frame, direction)

        # We ignore interim transcription at this point
        if isinstance(frame, InterimTranscriptionFrame):
            return

        # Handle interruptions - clear accumulated text and cancel timers
        if isinstance(frame, (CancelFrame, InterruptionFrame, BotInterruptionFrame)):
            logger.debug("🛑 StreamingTextAggregator: Clearing TTS queue due to interruption")
            self._aggregation = ""
            if self._timer and not self._timer.done():
                self._timer.cancel()
                self._timer = None
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextFrame):
            # Clean asterisks from incoming text
            clean_text = frame.text.replace('*', '')
            self._aggregation += clean_text

            # Periodically adjust parameters based on performance
            self._adjust_parameters_intelligently()

            # Check if we should release based on smart boundaries
            if self._should_release_text(self._aggregation):
                await self._release_text()
            else:
                # Schedule a release if we haven't recently released
                current_time = asyncio.get_event_loop().time()
                if current_time - self._last_release_time >= self._max_time:
                    await self._schedule_release()

        elif isinstance(frame, EndFrame):
            # Release any remaining text on end
            if self._aggregation.strip():
                await self._release_text()
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        """Clean up any pending timers."""
        if self._timer and not self._timer.done():
            self._timer.cancel()
        await super().cleanup()