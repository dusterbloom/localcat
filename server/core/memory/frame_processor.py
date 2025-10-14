"""
Frame processing for Pipecat pipeline integration.

Handles routing of different frame types through the memory system.
"""

import asyncio
import enum
import os
from typing import AsyncIterator, Optional, Dict, Any
from loguru import logger

# Pipecat imports (optional)
PIPECAT_IMPORT_DISABLED = os.getenv("PIPECAT_DISABLE_IMPORT", "").lower() in ("1", "true", "yes", "on")
if not PIPECAT_IMPORT_DISABLED:
    try:
        from pipecat.frames.frames import (  # type: ignore
            Frame,
            TranscriptionFrame,
            InterimTranscriptionFrame,
            LLMMessagesFrame,
            StartFrame,
            TextFrame,
        )
        from pipecat.processors.frame_processor import FrameProcessor, FrameDirection  # type: ignore
        PIPECAT_AVAILABLE = True
    except Exception as exc:
        logger.warning(f"Pipecat import failed ({exc!r}); using lightweight stubs")
        PIPECAT_AVAILABLE = False
else:
    logger.debug("Pipecat import disabled via PIPECAT_DISABLE_IMPORT")
    PIPECAT_AVAILABLE = False

if not PIPECAT_AVAILABLE:
    class Frame:  # type: ignore[no-redef]
        """Lightweight stub for tests."""
        pass

    class TranscriptionFrame(Frame):  # type: ignore[no-redef]
        """Stub transcription frame."""
        pass

    class InterimTranscriptionFrame(Frame):  # type: ignore[no-redef]
        """Stub interim transcription frame."""
        pass

    class LLMMessagesFrame(Frame):  # type: ignore[no-redef]
        """Stub LLM messages frame."""
        def __init__(self, messages=None):
            self.messages = messages or []

    class StartFrame(Frame):  # type: ignore[no-redef]
        """Stub start frame."""
        pass

    class TextFrame(Frame):  # type: ignore[no-redef]
        """Stub text frame."""
        def __init__(self, text: str = ""):
            self.text = text

    class FrameDirection(enum.Enum):  # type: ignore[no-redef]
        DOWNSTREAM = "downstream"
        UPSTREAM = "upstream"

    class FrameProcessor:  # type: ignore[no-redef]
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            return

        async def push_frame(self, frame: Frame, direction: FrameDirection):
            return

from .config_manager import MemoryConfiguration
from .quality_filter import QualityFilter


class MemoryFrameProcessor(FrameProcessor if PIPECAT_AVAILABLE else object):
    """
    Focused frame processor for memory system.

    Responsibilities:
    - Route frames to appropriate handlers
    - Integrate memory injection into pipeline
    - Handle user/assistant turns
    - Process transcriptions for memory extraction
    """

    def __init__(
        self,
        config: MemoryConfiguration,
        context_injector,
        session_manager,
        background_summarizer=None,
        hot_memory=None,
        intent_service=None,
        on_turn_processed=None,
        **kwargs
    ):
        """
        Initialize frame processor.
        
        Args:
            config: Memory configuration
            context_injector: Context injector instance
            session_manager: Session manager instance
            background_summarizer: Optional background summarizer
            hot_memory: Hot memory instance
            intent_service: Optional intent service for smart processing
            on_turn_processed: Optional callback invoked after processing a final transcription
        """
        if PIPECAT_AVAILABLE:
            super().__init__(**kwargs)
        
        self.config = config
        self.context_injector = context_injector
        self.session_manager = session_manager
        self.background_summarizer = background_summarizer
        self.hot_memory = hot_memory
        self.intent_service = intent_service
        self._on_turn_processed = on_turn_processed
        self.quality_filter = QualityFilter()
        
        # Frame processing state
        self._turn_id = 0
        self._ephemeral = config.ephemeral_mode

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection
    ) -> AsyncIterator[Frame]:
        """
        Process frame through memory pipeline.

        Routes different frame types to appropriate handlers.
        """
        # If memory is disabled or ephemeral, simply forward
        if not self.config.enabled or self._ephemeral:
            yield frame
            return

        # Handle StartFrame
        if isinstance(frame, StartFrame):
            await self._handle_start_frame(frame)
            yield frame
            return

        # Handle InterimTranscriptionFrame (Phase 0: pre-injection)
        if isinstance(frame, InterimTranscriptionFrame):
            await self._handle_interim_transcription(frame)
            yield frame
            return

        # Handle TranscriptionFrame (final processing)
        if isinstance(frame, TranscriptionFrame):
            await self._handle_transcription_frame(frame)
            yield frame
            return

        # Handle other frames (pass through)
        yield frame

    async def _handle_start_frame(self, frame: StartFrame) -> None:
        """Handle StartFrame initialization."""
        try:
            # Start background summarizer if configured for delta mode
            if (not self._ephemeral and 
                self.background_summarizer and 
                self.config.summarization_enabled and 
                self.config.summary_window_mode == "delta"):
                
                success = await self.background_summarizer.start_background_task(self.session_manager.session_id)
                if success:
                    logger.debug("[FrameProcessor] Background summarizer started")
                else:
                    logger.warning("[FrameProcessor] Failed to start background summarizer")
            
            # Provide role-aware IDs to HotMemory
            if self.hot_memory:
                try:
                    self.hot_memory.agent_eid = f"agent:{self.config.agent_id}"
                    self.hot_memory.current_user_id = self.session_manager.user_eid
                    self.hot_memory.current_session_id = self.session_manager.session_id
                except Exception as e:
                    logger.debug(f"[FrameProcessor] Failed to set hot memory identities: {e}")

        except Exception as e:
            logger.warning(f"[FrameProcessor] StartFrame handling failed: {e}")

    async def _handle_interim_transcription(self, frame: InterimTranscriptionFrame) -> None:
        """
        Handle interim transcription for early pre-injection.
        
        Phase 0: Retrieve and inject memory context before final transcription.
        """
        if self.context_injector._turn_has_preinjected_bullets:
            return  # Already pre-injected for this turn

        text = getattr(frame, 'text', '') or ''
        
        # Basic length threshold for pre-injection
        if len(text.strip().split()) < self.config.interim_min_words:
            return

        try:
            # Retrieve memory bullets for pre-injection
            bullets = await self.context_injector.retrieve_and_prepare_bullets(text, read_only=True)
            
            if bullets:
                # Inject immediately for low-latency response
                success = await self.context_injector.inject_memory_context()
                if success:
                    self.context_injector._turn_has_preinjected_bullets = True
                    logger.debug(f"[FrameProcessor] Interim pre-injection completed with {len(bullets)} bullets")

        except Exception as e:
            logger.error(f"[FrameProcessor] Interim transcription handling failed: {e}")

    async def _handle_transcription_frame(self, frame: TranscriptionFrame) -> None:
        """
        Handle final transcription for memory extraction and context refresh.
        
        This is the main entry point for memory processing.
        """
        is_final = getattr(frame, 'is_final', None)
        text = getattr(frame, 'text', '') or ''
        
        # WhisperSTTServiceMLX doesn't set is_final, so treat None as final (non-streaming)
        if is_final is True or is_final is None:
            logger.debug(f"[FrameProcessor] Processing transcription (is_final={is_final}): '{text}'")
            await self._process_transcription(text)
        else:
            logger.debug(f"[FrameProcessor] Skipping non-final transcription")

    async def _process_transcription(self, text: str) -> None:
        """
        Process user transcription with full memory pipeline.
        
        Args:
            text: User transcription text
        """
        if not text.strip():
            return

        self._turn_id += 1
        start_time = asyncio.get_event_loop().time()

        try:
            # Skip excluded phrases
            if self._is_excluded(text):
                logger.debug("[FrameProcessor] Skipping excluded phrase from memory processing")
                return

            # Intent classification for smart processing
            intent_result = None
            if (self.config.intent_aware_processing and 
                self.intent_service and 
                self.config.intent_classification_enabled):
                
                try:
                    intent_result = await self.intent_service.classify_intent(text)
                    logger.info(f"[FrameProcessor] Intent classified: {intent_result['intent']} "
                               f"(confidence: {intent_result['confidence']:.2f}, "
                               f"strategy: {intent_result.get('strategy', 'unknown')}, "
                               f"skip: {intent_result.get('skip_memory', False)})")
                except Exception as e:
                    logger.warning(f"[FrameProcessor] Intent classification failed: {e}")

            # Smart processing based on intent
            if intent_result and not intent_result.get('fallback', False):
                intent_name = intent_result['intent']
                strategy = intent_result.get('strategy', 'standard')
                skip_memory = intent_result.get('skip_memory', False)

                # Skip memory processing if routing decision says so
                if skip_memory:
                    logger.info(f"[FrameProcessor] Skipping memory processing for intent: {intent_name}")
                    return

                # Apply strategy-based processing
                logger.debug(f"[FrameProcessor] Using {strategy} strategy for intent: {intent_name}")
                focus = strategy if strategy != 'standard' else 'standard'
            else:
                focus = 'standard'

            # Process through hot memory
            if self.hot_memory:
                bullets, triples = self.hot_memory.process_turn(
                    text, 
                    self.session_manager.session_id, 
                    self._turn_id, 
                    focus=focus, 
                    intent=intent_result
                )
                
                logger.debug(f"[FrameProcessor] Extracted {len(triples)} facts, prepared {len(bullets)} bullets")
            else:
                bullets = []
                triples = []

            # Store conversation text for retrieval
            if self.hot_memory and self.hot_memory.store:
                if text.strip() and self.quality_filter.is_quality_for_storage(text):
                    now_ts = int(asyncio.get_event_loop().time() * 1000)
                    self.hot_memory.store.enqueue_mention(
                        self.session_manager.user_eid, 
                        text.strip(), 
                        now_ts, 
                        self.session_manager.session_id, 
                        self._turn_id
                    )
                    self.hot_memory.store.flush_if_needed()

            # Update context injector with new bullets
            if bullets:
                self.context_injector.set_pending_bullets(bullets)

            # Refresh injection if different from interim
            if self.context_injector.should_refresh_injection():
                await self.context_injector.inject_memory_context()
                logger.debug(f"[FrameProcessor] Final injection refreshed with {len(bullets)} bullets")

            # Update session tracking
            self.session_manager.increment_turn()
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            stats = self.session_manager.record_turn_metrics(elapsed_ms)

            if self._on_turn_processed:
                try:
                    self._on_turn_processed(elapsed_ms, stats)
                except Exception as exc:
                    logger.debug(f"[FrameProcessor] on_turn_processed callback failed: {exc}")

            # Trigger turn-based summary if configured
            if (self.background_summarizer and 
                self.config.summarization_enabled and 
                self.config.summary_window_mode == "turn_pairs"):
                
                if self.background_summarizer.should_summarize_turns(self._turn_id):
                    logger.info(f"[FrameProcessor] Triggering turn-based summary at turn {self._turn_id}")
                    asyncio.create_task(self.background_summarizer.summarize_turns(
                        self._turn_id, 
                        self.session_manager.session_id
                    ))

            # Reset pre-injection state for next turn
            self.context_injector.reset_turn_state()

        except Exception as e:
            logger.error(f"[FrameProcessor] Transcription processing failed: {e}")

    def _is_excluded(self, text: str) -> bool:
        """Check if text should be excluded from memory processing."""
        if not text or not self.config.excluded_phrases:
            return False
        
        tl = text.lower()
        for p in self.config.excluded_phrases:
            if p and p in tl:
                return True
        return False

    def set_ephemeral_mode(self, enabled: bool) -> None:
        """Enable/disable ephemeral mode."""
        self._ephemeral = bool(enabled)
        if self._ephemeral:
            logger.info("[FrameProcessor] Ephemeral mode ENABLED: memory processing bypassed")
        else:
            logger.info("[FrameProcessor] Ephemeral mode DISABLED: normal memory processing restored")

    def get_frame_processor_metrics(self) -> Dict[str, Any]:
        """Get frame processor metrics."""
        return {
            "turn_id": self._turn_id,
            "ephemeral_mode": self._ephemeral,
            "config_enabled": self.config.enabled
        }

    async def cleanup(self) -> None:
        """Cleanup when processor is destroyed."""
        try:
            # Generate final summary if needed
            if (self.background_summarizer and 
                self.config.summarization_enabled and 
                self._turn_id > 1 and 
                self._turn_id > self.background_summarizer._last_summarized_turn):
                
                logger.info(f"[FrameProcessor] Generating final summary for session (turns {self.background_summarizer._last_summarized_turn+1} to {self._turn_id})")
                try:
                    await self.background_summarizer.generate_final_summary(
                        self.session_manager.session_id, 
                        self._turn_id
                    )
                except asyncio.TimeoutError:
                    logger.warning("[FrameProcessor] Final summary generation timed out")

            # Stop background summarizer
            if self.background_summarizer:
                await self.background_summarizer.stop_background_task()

            # End session tracking
            if self.session_manager:
                self.session_manager.end_session()

            logger.debug("[FrameProcessor] Cleanup complete")

        except Exception as e:
            logger.error(f"[FrameProcessor] Cleanup error: {e}")
