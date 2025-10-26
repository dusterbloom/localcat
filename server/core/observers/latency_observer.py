"""
Latency Observer - Tracks frame-level timing for sub-second latency optimization.
Monitors STT, LLM, and TTS processing times to identify bottlenecks.
"""

import time
from typing import Dict, Any
from loguru import logger

from pipecat.observers.base_observer import BaseObserver, FrameProcessed
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    StartInterruptionFrame,
    UserStartedSpeakingFrame,
)


class LatencyObserver(BaseObserver):
    """
    Custom observer for tracking voice AI latency metrics.

    Monitors frame processing times to ensure sub-second end-to-end latency:
    - STT: <150ms (Whisper transcription)
    - LLM: <250ms first token
    - TTS: <200ms first audio
    - Total: <800ms end-to-end
    """

    def __init__(self):
        super().__init__()
        self._timings: Dict[str, float] = {}
        self._session_start = time.time()
        self._frame_counts: Dict[str, int] = {}

        logger.info("Latency Observer initialized for sub-second monitoring")

    async def on_process_frame(self, data: FrameProcessed):
        """Track frame processing timing."""
        frame = data.frame
        direction = data.direction if hasattr(data, 'direction') else 'unknown'
        current_time = time.time()
        frame_type = frame.__class__.__name__

        # Track frame counts
        if frame_type not in self._frame_counts:
            self._frame_counts[frame_type] = 0
        self._frame_counts[frame_type] += 1

        # Monitor key latency points
        if isinstance(frame, UserStartedSpeakingFrame):
            self._timings['user_started'] = current_time
            logger.debug("User started speaking")

        elif isinstance(frame, TranscriptionFrame):
            if 'user_started' in self._timings:
                stt_latency = (current_time - self._timings['user_started']) * 1000
                self._timings['stt_complete'] = current_time
                logger.info(f"STT latency: {stt_latency:.1f}ms (target: <150ms)")

                if stt_latency > 150:
                    logger.warning(f"STT latency {stt_latency:.1f}ms exceeds target")

        elif isinstance(frame, LLMTextFrame):
            if 'stt_complete' in self._timings:
                llm_latency = (current_time - self._timings['stt_complete']) * 1000
                self._timings['llm_complete'] = current_time
                logger.info(f"LLM latency: {llm_latency:.1f}ms (target: <250ms)")

                if llm_latency > 250:
                    logger.warning(f"LLM latency {llm_latency:.1f}ms exceeds target")

        elif isinstance(frame, TTSStartedFrame):
            if 'llm_complete' in self._timings:
                tts_start_latency = (current_time - self._timings['llm_complete']) * 1000
                self._timings['tts_started'] = current_time
                logger.info(f"TTS start latency: {tts_start_latency:.1f}ms")

        elif isinstance(frame, TTSAudioRawFrame):
            if 'tts_started' in self._timings and 'user_started' in self._timings:
                # First TTS audio chunk
                if 'first_tts_audio' not in self._timings:
                    self._timings['first_tts_audio'] = current_time
                    end_to_end_latency = (current_time - self._timings['user_started']) * 1000
                    logger.info(f"End-to-end latency: {end_to_end_latency:.1f}ms (target: <800ms)")

                    if end_to_end_latency > 800:
                        logger.warning(f"End-to-end latency {end_to_end_latency:.1f}ms exceeds target")
                    elif end_to_end_latency < 500:
                        logger.info(f"Excellent latency: {end_to_end_latency:.1f}ms")

        elif isinstance(frame, TTSStoppedFrame):
            if 'tts_started' in self._timings:
                tts_total_latency = (current_time - self._timings['tts_started']) * 1000
                logger.info(f"TTS total latency: {tts_total_latency:.1f}ms (target: <200ms)")

        elif isinstance(frame, StartInterruptionFrame):
            logger.info("Interruption detected - monitoring barge-in performance")

    def get_metrics_report(self) -> Dict[str, Any]:
        """Generate latency metrics report."""
        session_duration = time.time() - self._session_start

        return {
            'session_duration_seconds': session_duration,
            'frame_counts': self._frame_counts,
            'timings': self._timings,
            'latency_targets': {
                'stt_max_ms': 150,
                'llm_max_ms': 250,
                'tts_max_ms': 200,
                'end_to_end_max_ms': 800
            }
        }