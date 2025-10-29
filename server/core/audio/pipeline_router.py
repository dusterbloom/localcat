"""
Pipeline Router - Strategy Pattern (Open/Closed Principle)

Routes frames between intro/enrollment and conversation pipelines
using Pipecat's ParallelPipeline pattern.
"""

import asyncio
from typing import Protocol, List, Optional
from loguru import logger

from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.frames.frames import Frame

from .enrollment_state import EnrollmentState, EnrollmentProgress


class PipelineStrategy(Protocol):
    """
    Interface for pipeline routing strategies (Dependency Inversion Principle).
    
    Allows different routing strategies to be implemented without
    modifying the router itself (Open/Closed Principle).
    """
    async def should_route_to_intro(self, frame: Frame) -> bool: ...
    async def should_route_to_conversation(self, frame: Frame) -> bool: ...


class SpeakerEnrollmentRouter(ParallelPipeline):
    """
    Routes frames between intro and conversation pipelines.
    
    Single Responsibility: Frame routing based on enrollment state.
    Open/Closed: Extensible via PipelineStrategy without modification.
    Liskov Substitution: Respects ParallelPipeline contract.
    """
    
    def __init__(
        self,
        intro_processors: List,
        conversation_processors: List,
        initial_state: EnrollmentState = EnrollmentState.INTRO,
        total_enrollment_samples: int = 3,
    ):
        """
        Initialize router with two pipeline paths.
        
        Args:
            intro_processors: List of processors for intro/enrollment path
            conversation_processors: List of processors for conversation path
            initial_state: Starting state (default: INTRO)
            total_enrollment_samples: Number of samples needed for enrollment
        """
        self._state = initial_state
        self._progress = EnrollmentProgress(
            current=0,
            total=total_enrollment_samples,
            state=initial_state
        )
        self._state_lock = asyncio.Lock()
        
        logger.debug(f"[EnrollmentRouter] Initialized with state: {self._state.value}")
        
        # Build parallel pipelines with filters
        super().__init__(
            # Intro/enrollment path
            [FunctionFilter(self._intro_filter)] + intro_processors,
            # Conversation path
            [FunctionFilter(self._conversation_filter)] + conversation_processors,
        )
    
    async def _intro_filter(self, frame: Frame) -> bool:
        """
        Filter for intro pipeline (Liskov Substitution - respects contract).

        Returns True if frame should go to intro pipeline.

        CRITICAL: System frames (StartFrame, EndFrame, CancelFrame) must ALWAYS
        pass through to ensure proper pipeline initialization.

        IMPORTANT: During CHOICE state, user transcriptions should go to CONVERSATION
        pipeline (which has LLM) to handle ambiguous responses. Only TextFrames
        (from coordinator) should go to INTRO pipeline for TTS.
        """
        from pipecat.frames.frames import (
            StartFrame, EndFrame, CancelFrame, SystemFrame,
            TranscriptionFrame, TranscriptionUpdateFrame, TextFrame
        )

        # ALWAYS allow system initialization frames through BOTH pipelines
        if isinstance(frame, (StartFrame, EndFrame, CancelFrame)):
            return True

        async with self._state_lock:
            # Special handling for CHOICE state: route user transcriptions to conversation
            # pipeline so the LLM can handle ambiguous responses
            if self._state == EnrollmentState.CHOICE:
                if isinstance(frame, (TranscriptionFrame, TranscriptionUpdateFrame)):
                    logger.info(f"[EnrollmentRouter] CHOICE state: Routing user transcription '{frame.text[:50]}...' to CONVERSATION pipeline for LLM processing")
                    return False  # Send to conversation pipeline
                elif isinstance(frame, TextFrame):
                    logger.debug(f"[EnrollmentRouter] CHOICE state: Routing TextFrame '{frame.text[:50]}...' to INTRO pipeline for TTS")
                    return True  # Send to intro pipeline for TTS
                else:
                    logger.debug(f"[EnrollmentRouter] CHOICE state: Routing {frame.__class__.__name__} to INTRO pipeline")
                    return True

            should_route = self._state in (
                EnrollmentState.INTRO,
                EnrollmentState.ENROLLING,
                EnrollmentState.TRANSITION,
                EnrollmentState.NAME_CAPTURE,
            )
            # Only log non-audio frames to reduce spam
            if should_route and not isinstance(frame, SystemFrame):
                from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
                if not isinstance(frame, (InputAudioRawFrame, OutputAudioRawFrame)):
                    logger.debug(f"[EnrollmentRouter] Routing {frame.__class__.__name__} to INTRO pipeline (state: {self._state.value})")
            return should_route
    
    async def _conversation_filter(self, frame: Frame) -> bool:
        """
        Filter for conversation pipeline (Liskov Substitution).

        Returns True if frame should go to conversation pipeline.

        CRITICAL: System frames (StartFrame, EndFrame, CancelFrame) must ALWAYS
        pass through to ensure proper pipeline initialization.

        IMPORTANT: During CHOICE state:
        - Accept user transcriptions as INPUT (to process with LLM)
        - Allow TTS/output frames as OUTPUT (so responses reach UI/transport)
        """
        from pipecat.frames.frames import (
            StartFrame, EndFrame, CancelFrame, SystemFrame,
            TranscriptionFrame, TranscriptionUpdateFrame,
            TTSTextFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame,
            InputAudioRawFrame, OutputAudioRawFrame
        )

        # ALWAYS allow system initialization frames through BOTH pipelines
        if isinstance(frame, (StartFrame, EndFrame, CancelFrame)):
            return True

        async with self._state_lock:
            # Special handling for CHOICE state
            if self._state == EnrollmentState.CHOICE:
                # Accept user transcriptions as input
                if isinstance(frame, (TranscriptionFrame, TranscriptionUpdateFrame)):
                    logger.info(f"[EnrollmentRouter] CHOICE state: Accepting user transcription '{frame.text[:50]}...' in CONVERSATION pipeline for context processing")
                    return True
                # Allow TTS and audio output frames (so assistant responses reach transport/UI)
                if isinstance(frame, (TTSTextFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame)):
                    logger.debug(f"[EnrollmentRouter] CHOICE state: Allowing {frame.__class__.__name__} in CONVERSATION pipeline (output)")
                    return True
                # Reject all other frames during CHOICE
                logger.debug(f"[EnrollmentRouter] CHOICE state: Rejecting {frame.__class__.__name__} in CONVERSATION pipeline")
                return False

            should_route = self._state == EnrollmentState.CONVERSATION
            # Only log non-audio frames to reduce spam
            if should_route and not isinstance(frame, SystemFrame):
                if not isinstance(frame, (InputAudioRawFrame, OutputAudioRawFrame)):
                    logger.info(f"[EnrollmentRouter] Routing {frame.__class__.__name__} to CONVERSATION pipeline")
            return should_route
    
    async def update_state(
        self,
        new_state: EnrollmentState,
        progress: int = 0,
        speaker_id: Optional[str] = None,
        consistency: float = 0.0,
    ):
        """
        Update router state (Interface Segregation - focused API).
        
        This is the only external interface for state management,
        keeping the API minimal and focused.
        
        Args:
            new_state: New enrollment state
            progress: Current progress count
            speaker_id: Optional speaker identifier
            consistency: Enrollment consistency score
        """
        async with self._state_lock:
            old_state = self._state
            self._state = new_state
            self._progress = EnrollmentProgress(
                current=progress,
                total=self._progress.total,
                state=new_state,
                speaker_id=speaker_id,
                consistency=consistency,
            )
            
            if old_state != new_state:
                logger.info(
                    f"[EnrollmentRouter] State transition: {old_state.value} → {new_state.value} "
                    f"(progress: {self._progress.current}/{self._progress.total})"
                )
    
    @property
    def current_state(self) -> EnrollmentState:
        """Get current state (thread-safe read)"""
        return self._state
    
    @property
    def current_progress(self) -> EnrollmentProgress:
        """Get current progress (thread-safe read)"""
        return self._progress
    
    def is_in_intro_mode(self) -> bool:
        """Check if currently in intro/enrollment mode"""
        return self._state in (
            EnrollmentState.INTRO,
            EnrollmentState.ENROLLING,
            EnrollmentState.TRANSITION
        )
    
    def is_in_conversation_mode(self) -> bool:
        """Check if currently in conversation mode"""
        return self._state == EnrollmentState.CONVERSATION
