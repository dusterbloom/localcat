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
    UserStoppedSpeakingFrame,
)


class LatencyObserver(BaseObserver):
    """
    Custom observer for tracking voice AI latency metrics.

    Monitors frame processing times for optimization:
    - STT: Logged by DirectMLXWhisperSTT (~500-800ms actual processing)
    - LLM: <250ms first token (Time To First Token - TTFT)
    - TTS: First audio latency
    - Response: <500ms (LLM + TTS, excludes STT which runs in parallel with user speech)

    Note: Frame timing can be unreliable due to streaming and frame ordering.
    STT processing time is accurately logged by the STT service itself.
    """

    def __init__(self):
        super().__init__()
        self._timings: Dict[str, float] = {}
        self._session_start = time.time()
        self._frame_counts: Dict[str, int] = {}

        # Turn-based state tracking to prevent pollution across conversation turns
        self._current_turn_id = 0
        self._first_llm_token_measured = False  # Only measure first LLM token (TTFT)
        self._first_tts_audio_measured = False  # Only measure first TTS audio
        self._first_tts_started_measured = False  # Only log first TTS started
        self._first_tts_stopped_measured = False  # Only log first TTS stopped
        self._first_transcription_measured = False  # Only log first transcription

        logger.info("Latency Observer initialized for sub-second monitoring")

    def _reset_turn_state(self):
        """Reset timing state for a new conversation turn."""
        self._current_turn_id += 1
        self._timings.clear()
        self._first_llm_token_measured = False
        self._first_tts_audio_measured = False
        self._first_tts_started_measured = False
        self._first_tts_stopped_measured = False
        self._first_transcription_measured = False
        logger.debug(f"🔄 Turn {self._current_turn_id}: State reset for new conversation turn")

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
            # Reset state for new turn when user starts speaking
            self._reset_turn_state()
            self._timings['user_started'] = current_time
            logger.debug(f"👤 User started speaking (Turn {self._current_turn_id})")

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Mark when user finished speaking - this is the STT processing start point
            self._timings['user_stopped'] = current_time
            if 'user_started' in self._timings:
                speech_duration = (current_time - self._timings['user_started']) * 1000
                logger.debug(f"🗣️  User speech duration: {speech_duration:.0f}ms")

        elif isinstance(frame, TranscriptionFrame):
            # Mark STT complete - actual processing time logged by Whisper itself
            if not self._first_transcription_measured:
                self._timings['stt_complete'] = current_time
                self._first_transcription_measured = True
                # Note: Actual Whisper processing time is logged by DirectMLXWhisperSTT
                # Frame timing is unreliable due to streaming and frame ordering

        elif isinstance(frame, LLMTextFrame):
            # CRITICAL FIX: Only measure FIRST LLM token (Time To First Token - TTFT)
            # Streaming sends many LLMTextFrames; measuring all creates inflated latencies
            if 'stt_complete' in self._timings and not self._first_llm_token_measured:
                llm_latency = (current_time - self._timings['stt_complete']) * 1000
                self._timings['llm_first_token'] = current_time
                self._first_llm_token_measured = True
                logger.info(f"🧠 LLM TTFT (Time To First Token): {llm_latency:.1f}ms (target: <250ms)")

                if llm_latency > 250:
                    logger.warning(f"⚠️  LLM TTFT {llm_latency:.1f}ms exceeds target")
                elif llm_latency < 150:
                    logger.info(f"⚡ Excellent LLM TTFT: {llm_latency:.1f}ms")

        elif isinstance(frame, TTSStartedFrame):
            if 'llm_first_token' in self._timings and not self._first_tts_started_measured:
                tts_start_latency = (current_time - self._timings['llm_first_token']) * 1000
                self._timings['tts_started'] = current_time
                self._first_tts_started_measured = True
                logger.info(f"🔊 TTS started: {tts_start_latency:.1f}ms after first LLM token")

        elif isinstance(frame, TTSAudioRawFrame):
            # Measure first TTS audio chunk for response latency
            if 'stt_complete' in self._timings and not self._first_tts_audio_measured:
                self._timings['first_tts_audio'] = current_time
                self._first_tts_audio_measured = True

                # Response latency: transcription → first audio (LLM + TTS only, excludes STT)
                # STT time is logged separately by DirectMLXWhisperSTT
                response_latency = (current_time - self._timings['stt_complete']) * 1000

                # Log detailed breakdown if LLM timing is available
                if 'llm_first_token' in self._timings:
                    llm_ms = (self._timings['llm_first_token'] - self._timings['stt_complete']) * 1000
                    tts_ms = (current_time - self._timings['llm_first_token']) * 1000
                    logger.info(f"🎯 Response latency: {response_latency:.1f}ms = LLM({llm_ms:.0f}ms) + TTS({tts_ms:.0f}ms) [target: <500ms]")

                    if response_latency > 500:
                        logger.warning(f"⚠️  Response latency {response_latency:.1f}ms exceeds target")
                else:
                    logger.info(f"🎯 Response latency: {response_latency:.1f}ms (target: <500ms)")

        elif isinstance(frame, TTSStoppedFrame):
            if 'tts_started' in self._timings and not self._first_tts_stopped_measured:
                tts_total_latency = (current_time - self._timings['tts_started']) * 1000
                self._first_tts_stopped_measured = True
                logger.info(f"🔊 TTS completed: {tts_total_latency:.1f}ms total duration")
            # Turn complete - could reset here if needed for strict turn boundaries

        elif isinstance(frame, StartInterruptionFrame):
            logger.info("🚨 Interruption detected - user barge-in")

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