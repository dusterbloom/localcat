"""
Voice Agent Factory - Centralized service creation with dependency injection.

This factory implements the Factory pattern to create all voice agent services
with proper dependency injection, eliminating tight coupling in bot.py and
enabling better testability and maintainability.
"""

import asyncio
import os
from typing import Dict, Any, Optional, Union
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor, RTVIObserver
from core.observers.latency_observer import LatencyObserver
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.filters.stt_mute_filter import (
    STTMuteFilter,
    STTMuteConfig,
    STTMuteStrategy
)

# Local imports
from config import VoiceAgentConfig
from core.factories.service_factory import ServiceFactory
from core.memory.session_tracker import SessionTracker

# Global ServiceFactory instance for shared services across all VoiceAgentFactory instances
# This prevents multiple LLM service creation and eliminates 40+ second loading delays
_global_service_factory = None


def get_global_service_factory(config: VoiceAgentConfig) -> ServiceFactory:
    """Get or create the global ServiceFactory instance."""
    global _global_service_factory
    if _global_service_factory is None:
        _global_service_factory = ServiceFactory(config)
        logger.info("🌐 Global ServiceFactory created for shared service caching")
    return _global_service_factory


class VoiceAgentFactory:
    """Factory for creating voice agent services with dependency injection."""

    def __init__(self, config: VoiceAgentConfig):
        self.config = config
        self._services_cache: Dict[str, Any] = {}
        # Use global ServiceFactory to share services across all VoiceAgentFactory instances
        # This prevents multiple LLM service creation and eliminates 40+ second loading delays
        self._service_factory = get_global_service_factory(config)

        # CRITICAL: Global lock for MLX operations (STT + TTS share MLX runtime)
        # This prevents heap corruption from concurrent MLX access on macOS Sequoia
        self.mlx_lock = asyncio.Lock()
        logger.info("🔒 MLX global lock initialized (prevents concurrent Metal access)")
        logger.info("🔄 VoiceAgentFactory using global ServiceFactory for service reuse")

    def create_transport(self, webrtc_connection: SmallWebRTCConnection) -> SmallWebRTCTransport:
        """Create WebRTC transport with VAD and turn detection."""
        transport = self._service_factory.create_transport(webrtc_connection)
        self._services_cache['transport'] = transport
        return transport

    def create_stt_service(self) -> Any:
        """Create STT service based on configuration."""
        stt = self._service_factory.create_stt_service()
        self._services_cache['stt'] = stt
        return stt

    def create_tts_service(self, use_boundaries: bool = True) -> Any:
        """
        Create TTS service based on configuration.

        Args:
            use_boundaries: Enable sentence boundary detection (default True).
                           Set False for intro messages to play as one unit.
        """
        tts = self._service_factory.create_tts_service(use_boundaries)
        self._services_cache['tts'] = tts
        return tts

    def create_llm_service(self) -> OpenAILLMService:
        """Create LLM service with streaming configuration."""
        # Check if LLM service is already cached in this VoiceAgentFactory instance
        if 'llm' in self._services_cache:
            logger.debug("✅ LLM service reused from VoiceAgentFactory cache")
            return self._services_cache['llm']

        # Create new LLM service and cache it
        llm = self._service_factory.create_llm_service()
        self._services_cache['llm'] = llm
        return llm

    def create_memory_processor(self, context_aggregator: Any, session_tracker: SessionTracker):
        """Create HotMem memory processor."""
        memory = self._service_factory.create_memory_processor(context_aggregator, session_tracker)
        self._services_cache['memory'] = memory
        return memory

    def create_audio_intelligence_processor(self) -> Any:
        """Create Audio Intelligence processor for speaker recognition, emotion, prosody."""
        audio_intel = self._service_factory.create_audio_intelligence_processor()
        if audio_intel:
            self._services_cache['audio_intelligence'] = audio_intel
        return audio_intel

    def create_hotmem_service(self, session_tracker: Optional[SessionTracker] = None):
        """Create HotMemService (Pipecat-compatible memory service)."""
        hotmem_service = self._service_factory.create_hotmem_service(session_tracker)
        self._services_cache['hotmem_service'] = hotmem_service
        return hotmem_service

    def create_intent_service(self) -> Optional[Any]:
        """Create intent classification service for smart memory processing."""
        intent_service = self._service_factory.create_intent_service()
        if intent_service:
            self._services_cache['intent'] = intent_service
        return intent_service

    def create_context_aggregator(self, llm_service: OpenAILLMService, system_instruction: str) -> Any:
        """Create LLM context aggregator."""
        # Delegate to ServiceFactory with reference to self for dynamic prompt rebuilding
        anonymous_aggregator = self._service_factory.create_context_aggregator(
            llm_service,
            system_instruction,
            factory_ref=self
        )
        self._services_cache['context'] = self._service_factory.get_service('context')
        self._services_cache['anonymous_aggregator'] = anonymous_aggregator
        return anonymous_aggregator

    def create_session_tracker(self) -> SessionTracker:
        """Create session tracker."""
        tracker = self._service_factory.create_session_tracker()
        self._services_cache['session_tracker'] = tracker
        return tracker

    def create_rtvi_processor(self) -> RTVIProcessor:
        """Create RTVI processor for client UI events."""
        rtvi = self._service_factory.create_rtvi_processor()
        self._services_cache['rtvi'] = rtvi
        return rtvi

    def create_mic_probe(self) -> Optional[Any]:
        """Create optional mic probe for debugging."""
        probe = self._service_factory.create_mic_probe()
        if probe:
            self._services_cache['mic_probe'] = probe
        return probe

    def create_text_aggregator(self) -> Any:
        """Create fast text aggregator for fluid speech."""
        aggregator = self._service_factory.create_text_aggregator()
        self._services_cache['text_aggregator'] = aggregator
        return aggregator

    def _has_existing_speaker_profiles(self) -> bool:
        """Check if any speaker profiles exist."""
        return self._service_factory.has_existing_speaker_profiles()

    def create_intro_aware_pipeline(
        self,
        transport: SmallWebRTCTransport,
        services: Dict[str, Any]
    ) -> Pipeline:
        """
        Create pipeline with intro/enrollment routing for improved UX.
        
        Uses ParallelPipeline pattern to route between:
        - Intro pipeline: Direct TTS feedback during enrollment
        - Conversation pipeline: Full LLM processing
        """
        from core.audio.pipeline_router import SpeakerEnrollmentRouter
        from core.audio.enrollment_coordinator import EnrollmentCoordinator
        from core.audio.enrollment_state import EnrollmentState
        
        # Determine initial state: if ephemeral choice is enabled, start with CHOICE
        has_profiles = self._has_existing_speaker_profiles()
        if self.config.enable_intro_pipeline and self.config.enable_ephemeral_choice and not self.config.force_intro:
            initial_state = EnrollmentState.CHOICE
        else:
            initial_state = (
                EnrollmentState.CONVERSATION if (has_profiles and self.config.skip_intro_for_returning and not self.config.force_intro)
                else EnrollmentState.INTRO
            )
        
        logger.info(
            f"[Factory] Creating intro-aware pipeline "
            f"(initial_state={initial_state.value}, has_profiles={has_profiles}, "
            f"flags: intro={self.config.enable_intro_pipeline}, choice={self.config.enable_ephemeral_choice}, force={self.config.force_intro})"
        )
        
        # CRITICAL: ParallelPipeline requires separate processor instances for each branch
        # Reusing the same TTS instance causes initialization errors
        
        # Define intro pipeline (bypasses LLM for direct feedback)
        # Create a separate TTS instance for intro messages
        # CRITICAL: Disable sentence boundaries so the full intro/instructions play as a single unit.
        # For Siri, this uses a no-boundaries mode; for Kokoro, aggregate_sentences/use_boundaries are disabled.
        intro_tts = self.create_tts_service(use_boundaries=False)

        # Optional: prevent TTS from being interrupted during intro/enrollment
        async def _no_intro_interruptions(frame) -> bool:
            try:
                from pipecat.frames.frames import InterruptionTaskFrame
                return not isinstance(frame, InterruptionTaskFrame)
            except Exception:
                return True

        from pipecat.processors.filters.function_filter import FunctionFilter as _FF

        # Debug: log TextFrames entering the INTRO TTS to ensure instructions are present
        async def _log_intro_text(frame) -> bool:
            from pipecat.frames.frames import TextFrame
            if isinstance(frame, TextFrame):
                txt = getattr(frame, 'text', '') or ''
                logger.debug(f"[Intro] TextFrame -> TTS: '{txt[:120]}{'...' if len(txt)>120 else ''}'")
            return True

        # Feed intro TextFrames directly to TTS; keep a small logger before TTS
        intro_processors = [
            _FF(_no_intro_interruptions),
            _FF(_log_intro_text),
            intro_tts
        ]
        
        # Define conversation pipeline (full processing)
        # IMPORTANT: Memory processor should only run in conversation branch
        
        # Enable barge-in: cancel TTS on user interruption
        async def _cancel_tts_on_interruption(frame) -> bool:
            try:
                from pipecat.frames.frames import InterruptionTaskFrame
                if isinstance(frame, InterruptionTaskFrame):
                    tts = services.get('tts')
                    if tts and hasattr(tts, 'request_cancel'):
                        await tts.request_cancel()
                    # Pass the interruption frame through to allow TTS to handle cleanup
                    # (TTS needs to see InterruptionFrame to clear its text aggregator)
                    return True
            except Exception:
                # Proceed if anything goes wrong; don't block pipeline
                return True
            return True

        from pipecat.processors.filters.function_filter import FunctionFilter as _FF

        # DEBUG: Frame tracking to diagnose duplication issue
        frame_tracker = {}

        async def _debug_before_memory(frame) -> bool:
            """Track frames entering conversation pipeline"""
            from pipecat.frames.frames import TranscriptionFrame
            if isinstance(frame, TranscriptionFrame):
                fid = frame.id
                if fid not in frame_tracker:
                    frame_tracker[fid] = {'text': frame.text, 'count': 0}
                frame_tracker[fid]['count'] += 1
                logger.debug(f"🔍 [PIPELINE IN] TranscriptionFrame id={fid} text='{frame.text[:50]}...' (seen {frame_tracker[fid]['count']}x)")
            return True

        async def _debug_before_llm(frame) -> bool:
            """Track frames entering LLM"""
            from pipecat.frames.frames import TranscriptionFrame, Frame
            if isinstance(frame, TranscriptionFrame):
                logger.debug(f"🔍 [LLM INPUT] id={frame.id} text='{frame.text[:50]}...'")
            return True

        async def _debug_after_llm(frame) -> bool:
            """Track frames generated by LLM"""
            from pipecat.frames.frames import TextFrame
            if isinstance(frame, TextFrame):
                logger.debug(f"🔍 [LLM OUTPUT] id={frame.id} text='{frame.text[:50]}...'")
            return True

        async def _debug_before_tts(frame) -> bool:
            """Track frames entering TTS"""
            from pipecat.frames.frames import TextFrame
            if isinstance(frame, TextFrame):
                logger.debug(f"🔍 [TTS INPUT] id={frame.id} text='{frame.text[:50]}...'")
            return True

        conversation_processors = [
            _FF(_cancel_tts_on_interruption),  # Barge-in handler
        ]

        # Vision context injector (if video enabled) - must be in conversation pipeline to see TranscriptionFrames
        if self.config.video_input_enabled:
            from core.video import VisionContextInjector
            context = services.get('context')
            if context:
                # Get vision optimization settings from environment
                keyword_filter = os.getenv("VISION_KEYWORD_FILTER", "true").lower() in ("true", "1", "yes")
                keywords = None
                if os.getenv("VISION_KEYWORDS"):
                    keywords = [k.strip() for k in os.getenv("VISION_KEYWORDS").split(",")]

                vision_injector = VisionContextInjector(
                    context=context,
                    target_fps=self.config.video_target_fps,
                    inject_on_text=True,
                    keyword_filter=keyword_filter,
                    keywords=keywords
                )
                conversation_processors.append(vision_injector)
                logger.info(f"📹 Vision context injector added to conversation pipeline "
                          f"({self.config.video_target_fps} fps, keyword_filter={keyword_filter})")
            else:
                logger.warning("📹 Video enabled but context not available - skipping vision injection")

        # Continue with rest of conversation pipeline with DEBUG filters
        conversation_processors.extend([
            _FF(_debug_before_memory),  # DEBUG: Track pipeline input
            services['memory'],
            services['context_aggregator'].user(),
            _FF(_debug_before_llm),  # DEBUG: Track LLM input
            services['llm'],
            _FF(_debug_after_llm),  # DEBUG: Track LLM output
            services['text_aggregator'],  # Intelligent sentence boundary detection (splits into sentences for TTS)
            _FF(_debug_before_tts),  # DEBUG: Track TTS input
            services['tts'],  # Main TTS instance
            services['context_aggregator'].assistant(),  # Capture assistant response for conversation memory
        ])
        
        # Determine enrollment sample count from AudioIntelligence (single source of truth)
        total_samples = 3
        try:
            ai = services.get('audio_intelligence')
            if ai is not None and hasattr(ai, '_auto_enroll_utterances'):
                total_samples = int(getattr(ai, '_auto_enroll_utterances'))
            else:
                # Fallback to environment variable
                total_samples = int(os.getenv('SPEAKER_AUTO_ENROLL_UTTERANCES', str(total_samples)))
        except Exception:
            total_samples = 3

        # Create router
        router = SpeakerEnrollmentRouter(
            intro_processors=intro_processors,
            conversation_processors=conversation_processors,
            initial_state=initial_state,
            total_enrollment_samples=total_samples,
        )
        
        # Create coordinator
        coordinator = EnrollmentCoordinator(
            router=router,
            profile_dir=self.config.speaker_profile_dir,
            skip_for_returning=self.config.skip_intro_for_returning,
            include_privacy_explanation=self.config.include_privacy_explanation,
            audio_intel=services.get('audio_intelligence'),
            memory=services.get('memory'),
            enable_ephemeral_choice=self.config.enable_ephemeral_choice,
            context_aggregator=services.get('context_aggregator'),
        )
        
        # Build main pipeline stages
        stages = [transport.input()]

        # Note: We add Audio Intelligence and EnrollmentCoordinator after STT,
        # so coordinator TextFrames reach the router directly without passing through STT.

        # Mic gate (KISS): Only block mic while the CHOICE prompt TTS is playing.
        # This avoids accidental interruption but doesn't block the user from answering.
        choice_tts_active = asyncio.Event()

        async def _track_choice_tts(frame) -> bool:
            from pipecat.frames.frames import TTSStartedFrame, TTSStoppedFrame
            state = router.current_state
            if state == EnrollmentState.CHOICE and isinstance(frame, TTSStartedFrame):
                choice_tts_active.set()
                logger.debug("[MicGate] CHOICE TTS started → mic muted")
            elif state == EnrollmentState.CHOICE and isinstance(frame, TTSStoppedFrame):
                if choice_tts_active.is_set():
                    choice_tts_active.clear()
                    logger.debug("[MicGate] CHOICE TTS finished → mic enabled")
            return True

        async def should_mute_enrollment(stt_filter: STTMuteFilter) -> bool:
            """Mute ONLY during CHOICE state when enrollment TTS is playing.

            This prevents echo/loopback from triggering false interruptions while
            the bot speaks enrollment prompts. Uses Pipecat's STTMuteFilter which
            properly blocks VAD frames (UserStartedSpeaking, etc.) unlike the old
            custom MicGate filter.
            """
            return (router.current_state == EnrollmentState.CHOICE and
                    choice_tts_active.is_set())

        # Create STTMuteFilter for enrollment (replaces old MicGate)
        # This blocks VAD frames + interruptions during CHOICE TTS only
        stt_mute_filter = STTMuteFilter(
            config=STTMuteConfig(
                strategies={STTMuteStrategy.CUSTOM},
                should_mute_callback=should_mute_enrollment
            )
        )

        # Optional mic probe
        mic_probe = services.get('mic_probe')
        if mic_probe:
            stages.append(mic_probe)

        # Create transcript processor for UI display
        transcript = TranscriptProcessor()

        # Add event handler to log all conversation content
        @transcript.event_handler("on_transcript_update")
        async def log_intro_conversation(processor, frame):
            """Log all conversation content to the log files."""
            for message in frame.messages:
                role = message.role.upper()
                content = message.content
                logger.info(f"📝 CONVERSATION [{role}]: {content}")

        # Core processing before router
        stages.extend([
            services['stt'],
            stt_mute_filter,  # Block VAD frames during enrollment CHOICE TTS
            transcript.user(),  # Capture user transcriptions for UI
            services['rtvi'],
        ])

        # Feed Audio Intelligence (if enabled) and then the EnrollmentCoordinator
        ai = services.get('audio_intelligence')
        if ai:
            stages.append(ai)
        stages.append(coordinator)

        # Router splits to intro vs conversation
        stages.append(router)

        # Add CHOICE TTS watcher AFTER router so it sees TTS frames from both pipelines
        stages.append(_FF(_track_choice_tts))

        # Light TTS output probe: helps diagnose muted enrollment speech
        # Logs TTSStarted/TTSText and first audio chunk per synthesis
        tts_audio_logged = {"seen": False}

        async def _log_tts_events(frame) -> bool:
            from pipecat.frames.frames import TTSStartedFrame, TTSTextFrame, TTSAudioRawFrame, TTSStoppedFrame
            try:
                if isinstance(frame, TTSStartedFrame):
                    tts_audio_logged["seen"] = False
                    logger.debug("[Pipeline] TTS started")
                elif isinstance(frame, TTSTextFrame):
                    txt = getattr(frame, 'text', '') or ''
                    logger.debug(f"[Pipeline] TTS text: '{txt[:80]}{'...' if len(txt) > 80 else ''}'")
                elif isinstance(frame, TTSAudioRawFrame):
                    if not tts_audio_logged["seen"]:
                        tts_audio_logged["seen"] = True
                        logger.debug(f"[Pipeline] First TTS audio chunk: {len(getattr(frame, 'audio', b''))} bytes")
                elif isinstance(frame, TTSStoppedFrame):
                    logger.debug("[Pipeline] TTS stopped")
            except Exception:
                pass
            return True

        stages.append(_FF(_log_tts_events))

        # Output audio first, then capture assistant transcripts (per Pipecat docs)
        stages.append(transport.output())

        # Add debug filter to see what's going to transcript.assistant()
        async def _debug_transcript_assistant(frame) -> bool:
            from pipecat.frames.frames import TTSTextFrame
            if isinstance(frame, TTSTextFrame):
                logger.info(f"🔍 [TRANSCRIPT.ASSISTANT INPUT] TTSTextFrame: '{frame.text[:100]}...'")
            return True

        stages.append(_FF(_debug_transcript_assistant))
        stages.append(transcript.assistant())
        
        pipeline = Pipeline(stages)
        self._services_cache['pipeline'] = pipeline
        self._services_cache['enrollment_router'] = router
        self._services_cache['enrollment_coordinator'] = coordinator
        
        return pipeline

    def create_pipeline(self, transport: SmallWebRTCTransport, services: Dict[str, Any]) -> Pipeline:
        """Create the main voice agent pipeline with optional audio intelligence."""
        # Use intro-aware pipeline if enabled
        if self.config.enable_intro_pipeline and services.get('audio_intelligence'):
            logger.info("🎭 Using intro-aware pipeline for enrollment UX")
            return self.create_intro_aware_pipeline(transport, services)

        # Standard pipeline (no enrollment UX)
        # Start with a top-level source; ParallelPipeline only fans out frames
        stages = [transport.input()]

        # Add RTVI processor right after transport input (per Pipecat docs)
        # This allows OutputTransportMessageUrgentFrame to flow downstream to transport.output()
        if services.get('rtvi'):
            stages.append(services['rtvi'])
            logger.debug("📡 RTVI processor added after transport.input()")

        # Optional parallel processors (non-blocking, event-driven)
        # These processors handle async events and must not block the main STT→LLM→TTS flow
        parallel_branch = []

        # Mic probe for diagnostics (if enabled)
        mic_probe = services.get('mic_probe')
        if mic_probe:
            parallel_branch.append(mic_probe)

        # Audio intelligence for speaker recognition (if enabled)
        audio_intel = services.get('audio_intelligence')
        if audio_intel:
            logger.info("🎤 Audio Intelligence enabled - speaker recognition active")
            parallel_branch.append(audio_intel)

        # Add ParallelPipeline if we have any side-processors
        # Single-branch ParallelPipeline ensures these processors don't block main flow
        if parallel_branch:
            stages.append(ParallelPipeline(parallel_branch))

        # Create transcript processor for UI display
        transcript = TranscriptProcessor()

        # Add event handler to log all conversation content
        @transcript.event_handler("on_transcript_update")
        async def log_conversation(processor, frame):
            """Log all conversation content to the log files."""
            for message in frame.messages:
                role = message.role.upper()
                content = message.content
                logger.info(f"📝 CONVERSATION [{role}]: {content}")

        stages.extend([
            services['stt'],
            transcript.user(),  # Capture user transcriptions for UI
        ])

        # Vision context injector (if video enabled) - MUST be after STT to see TranscriptionFrames
        if self.config.video_input_enabled:
            from core.video import VisionContextInjector
            context = services.get('context')
            if context:
                # Get vision optimization settings from environment
                keyword_filter = os.getenv("VISION_KEYWORD_FILTER", "true").lower() in ("true", "1", "yes")
                keywords = None
                if os.getenv("VISION_KEYWORDS"):
                    keywords = [k.strip() for k in os.getenv("VISION_KEYWORDS").split(",")]

                vision_injector = VisionContextInjector(
                    context=context,
                    target_fps=self.config.video_target_fps,
                    inject_on_text=True,
                    keyword_filter=keyword_filter,
                    keywords=keywords
                )
                stages.append(vision_injector)
                logger.info(f"📹 Vision context injector added after STT "
                          f"({self.config.video_target_fps} fps, keyword_filter={keyword_filter})")
            else:
                logger.warning("📹 Video enabled but context not available - skipping vision injection")

        # Continue with rest of pipeline
        stages.extend([
            services['memory'],
            services['context_aggregator'].user(),
            services['llm'],
            services['text_aggregator'],  # Intelligent sentence boundary detection
            services['tts'],
            # Output audio then capture assistant transcripts (per Pipecat docs)
            transport.output(),
            transcript.assistant(),
        ])

        pipeline = Pipeline(stages)
        self._services_cache['pipeline'] = pipeline
        return pipeline

    def create_pipeline_task(self, pipeline: Pipeline, rtvi_processor: Optional[RTVIProcessor] = None) -> PipelineTask:
        """Create pipeline task with latency profiling parameters."""
        # Configure for sub-second latency: 512 samples (32ms @ 16kHz)
        params = PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=16000,  # 16kHz for 32ms chunks (512 samples)
            audio_out_sample_rate=24000,  # 24kHz output for TTS
            send_initial_empty_metrics=False,
            report_only_initial_ttfb=True,
        )

        observers = []
        if rtvi_processor:
            observers.append(RTVIObserver(rtvi_processor))

        # Add latency observer for sub-second monitoring
        latency_observer = LatencyObserver()
        observers.append(latency_observer)

        task = PipelineTask(
            pipeline,
            params=params,
            observers=observers
        )
        self._services_cache['pipeline_task'] = task
        return task

    def get_service(self, name: str) -> Any:
        """Get a cached service by name."""
        return self._services_cache.get(name)

    def clear_cache(self):
        """Clear the services cache."""
        self._services_cache.clear()

    def create_voice_agent(self, webrtc_connection: SmallWebRTCConnection, system_instruction: str) -> Dict[str, Any]:
        """Create complete voice agent with all services configured.

        This is the main entry point that assembles all components.
        Returns a dictionary with all created services and the pipeline task.
        """
        # Create transport
        transport = self.create_transport(webrtc_connection)

        # Create STT service
        stt = self.create_stt_service()

        # Create TTS service
        tts = self.create_tts_service()

        # Create LLM service
        llm = self.create_llm_service()

        # Create context aggregator with system instruction
        context_aggregator = self.create_context_aggregator(llm, system_instruction)

        # Create session tracker
        session_tracker = self.create_session_tracker()

        # Create intent service for smart processing (optional)
        intent_service = self.create_intent_service()

        # Create memory service based on configuration
        memory_backend = os.getenv("MEMORY_BACKEND", "hotpath").lower()
        logger.debug(f"[Factory] MEMORY_BACKEND from env: '{memory_backend}'")

        if memory_backend == "hotmem":
            # Use HotMemService (Pipecat-compatible service)
            memory = self.create_hotmem_service(session_tracker)
            logger.info("Using HotMemService (Pipecat-compatible memory)")
        else:
            # Use HotPathMemoryProcessor (current processor)
            memory = self.create_memory_processor(context_aggregator, session_tracker)
            logger.info("Using HotPathMemoryProcessor (current memory processor)")

        # Link memory processor to anonymous aggregator for ephemeral mode control
        if hasattr(context_aggregator, '_memory_processor'):
            context_aggregator._memory_processor = memory
            logger.debug("[Factory] Linked memory processor to anonymous aggregator")

        # Create RTVI processor
        rtvi = self.create_rtvi_processor()

        # Create optional mic probe
        mic_probe = self.create_mic_probe()

        # Create audio intelligence processor (Session 1: Speaker recognition)
        audio_intelligence = self.create_audio_intelligence_processor()

        # Create text aggregator for intelligent sentence boundaries
        text_aggregator = self.create_text_aggregator()

        # Assemble all services
        services = {
            'transport': transport,
            'stt': stt,
            'tts': tts,
            'llm': llm,
            'context': self._services_cache.get('context'),  # Include context for event handlers
            'context_aggregator': context_aggregator,
            'session_tracker': session_tracker,
            'memory': memory,
            'rtvi': rtvi,
            'mic_probe': mic_probe,
            'intent': intent_service,  # Intent classification service (optional)
            'audio_intelligence': audio_intelligence,  # Audio intelligence (speaker, emotion, prosody)
            'text_aggregator': text_aggregator,  # Token-aware text aggregator for sentence boundaries
        }

        # Create pipeline
        pipeline = self.create_pipeline(transport, services)

        # Create pipeline task
        task = self.create_pipeline_task(pipeline, rtvi)

        return {
            **services,
            'pipeline': pipeline,
            'task': task
        }

    def build_system_prompt(self, skip_memory: bool = False, camera_active: bool = False) -> str:
        """
        Build dynamic system prompt based on configuration.

        Generates a clear, structured prompt that declares capabilities
        and provides guidance suitable for any SLM. Adapts based on
        enabled features.

        Args:
            skip_memory: If True, exclude memory section (for anonymous/ephemeral mode)
            camera_active: If True, include vision-related prompts

        Returns:
            Formatted system prompt string
        """
        sections = []

        # Identity
        sections.append("You are Locat, a locally-run AI voice assistant.")

        # Memory capabilities (if enabled AND not skipped for anonymous mode)
        if self.config.memory_enabled and not skip_memory:
            sections.append(self._build_memory_section())

        # Vision capabilities (if enabled)
        if self.config.video_input_enabled and self.config.vision_model_enabled:
            sections.append(self._build_vision_section())

        # Context management guidance
        sections.append(self._build_context_section())

        # Response guidelines
        sections.append(self._build_guidelines_section())

        return "\n\n".join(sections)

    def _build_memory_section(self) -> str:
        """Build memory capabilities section."""
        lines = ["MEMORY SYSTEM:"]
        lines.append("- You have access to conversation history and learned facts")
        lines.append(f"- Memory context appears with header: \"{self.config.memory_inject_header}\"")
        lines.append("- Use memory context when relevant to provide personalized responses")

        # List active sources
        sources = self.config.memory_sources.split(',')
        source_map = {
            'convo': 'recent conversation',
            'summary': 'conversation summaries',
            'graph': 'learned facts',
            'semantic': 'semantic search'
        }
        active = [source_map.get(s.strip(), s.strip()) for s in sources if s.strip() in source_map]
        if active:
            lines.append(f"- Sources: {', '.join(active)}")

        return "\n".join(lines)

    def _build_vision_section(self) -> str:
        """Build vision capabilities section."""
        lines = ["VISION:"]
        lines.append("- You can see what the user's camera shows")

        if self.config.vision_keyword_filter:
            keywords = [k.strip() for k in self.config.vision_keywords.split(',')][:8]
            if keywords:
                lines.append(f"- Vision activates for queries about: {', '.join(keywords)}, etc.")

        lines.append("- Describe what you see concisely when asked")
        return "\n".join(lines)

    def _build_context_section(self) -> str:
        """Build context management section."""
        lines = ["CONTEXT MANAGEMENT:"]
        lines.append(f"- Context window: {self.config.llm_context_max_tokens} tokens maximum")
        lines.append(f"- System maintains recent {self.config.llm_context_min_turns}+ conversation turns")
        lines.append("- Keep responses concise to preserve context capacity")
        return "\n".join(lines)

    def _build_guidelines_section(self) -> str:
        """Build response guidelines section."""
        lines = ["RESPONSE GUIDELINES:"]
        lines.append("- Be friendly, helpful, and conversational")
        lines.append("- Use short, natural responses suitable for voice interaction")
        lines.append("- Address users by name if mentioned in memory context")
        lines.append("- Focus on relevant details, avoid verbose explanations")
        return "\n".join(lines)
