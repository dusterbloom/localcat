"""
Voice Agent Factory - Centralized service creation with dependency injection.

This factory implements the Factory pattern to create all voice agent services
with proper dependency injection, eliminating tight coupling in bot.py and
enabling better testability and maintainability.
"""

import os
from typing import Dict, Any, Optional, Union
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.pipeline.pipeline import Pipeline
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

# Local imports
from config import VoiceAgentConfig
from core.memory.hotpath_processor import HotPathMemoryProcessor
from core.memory.session_tracker import SessionTracker
from core.memory import HotMemService

# Import intent service for smart processing
try:
    from core.intent import get_intent_service
    INTENT_SERVICE_AVAILABLE = True
except ImportError:
    INTENT_SERVICE_AVAILABLE = False
    logger.warning("Intent service not available in factory")
# Import database session tracker if available
try:
    from core.memory.db_session_tracker import DatabaseSessionTracker
    DB_TRACKER_AVAILABLE = True
except ImportError:
    DB_TRACKER_AVAILABLE = False
from core.tts.kokoro_professional import ProfessionalKokoroTTSService
from core.tts.kokoro_mlx import MLXKokoroTTSService

# # Import legacy TTS services for backward compatibility
# try:
#     from fastapi_streaming_tts import FastAPIStreamingTTS
#     FASTAPI_TTS_AVAILABLE = True
# except ImportError as e:
#     logger.warning(f"FastAPI TTS not available: {e}")
#     FASTAPI_TTS_AVAILABLE = False

# Import optional components
try:
    from mic_probe import MicProbe
    MIC_PROBE_AVAILABLE = True
except ImportError:
    MIC_PROBE_AVAILABLE = False


class VoiceAgentFactory:
    """Factory for creating voice agent services with dependency injection."""

    def __init__(self, config: VoiceAgentConfig):
        self.config = config
        self._services_cache: Dict[str, Any] = {}

    def create_transport(self, webrtc_connection: SmallWebRTCConnection) -> SmallWebRTCTransport:
        """Create WebRTC transport with VAD and turn detection."""
        # VAD configuration with backward compatibility
        vad_confidence = float(os.getenv("VAD_CONFIDENCE", "0.5"))
        vad_start_secs = float(os.getenv("VAD_START_SECS", "0.1"))
        # Use a more forgiving default stop window so brief pauses do not end the turn
        vad_stop_secs = float(os.getenv("VAD_STOP_SECS", "4.0"))
        vad_min_volume = float(os.getenv("VAD_MIN_VOLUME", "0.4"))

        vad_params = VADParams(
            confidence=vad_confidence,
            start_secs=vad_start_secs,
            stop_secs=max(vad_stop_secs, 0.8),
            min_volume=vad_min_volume,
        )

        ice_servers = [
            IceServer(urls="stun:stun.l.google.com:19302")
        ]

        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=vad_params),
                turn_analyzer=LocalSmartTurnAnalyzerV3(
                    params=SmartTurnParams(
                        stop_secs=float(os.getenv("SMART_TURN_STOP_SECS", "4.0")),
                        pre_speech_ms=float(os.getenv("SMART_TURN_PRE_SPEECH_MS", "300")),
                        max_duration_secs=float(os.getenv("SMART_TURN_MAX_DURATION_SECS", "16.0")),
                    )
                ),
            ),
        )

        self._services_cache['transport'] = transport
        return transport

    def create_stt_service(self) -> Any:
        """Create STT service based on configuration."""
        stt_config = self.config.get_component_config("stt")

        if self.config.stt_engine == "parakeet_streaming":
            try:
                # Try Parakeet streaming first (ultra-low latency)
                from core.stt.parakeet_streaming import ParakeetStreamingSTT
                logger.debug(f"Initializing Parakeet streaming STT with model: {stt_config.get('model', 'mlx-community/parakeet-tdt-0.6b-v3')}")
                stt = ParakeetStreamingSTT(
                    model_path=stt_config.get("model", "mlx-community/parakeet-tdt-0.6b-v3"),
                    language=stt_config.get("language", "en"),
                    chunk_duration=float(os.getenv("PARAKEET_CHUNK_DURATION", "1.0")),
                    enable_vad=os.getenv("PARAKEET_ENABLE_VAD", "false").lower() in ("1", "true", "yes"),
                    temperature=float(os.getenv("PARAKEET_TEMPERATURE", "0.0")),
                    confidence_threshold=float(os.getenv("PARAKEET_CONFIDENCE_THRESHOLD", "0.2")),
                    sentence_pause_threshold=float(os.getenv("PARAKEET_SENTENCE_PAUSE_THRESHOLD", "1.2")),
                    max_chunk_duration=float(os.getenv("PARAKEET_MAX_CHUNK_DURATION", "4.0")),
                    context_size=tuple(map(int, os.getenv("PARAKEET_CONTEXT_SIZE", "256,256").split(","))),
                    depth=int(os.getenv("PARAKEET_DEPTH", "3")),
                    volume_threshold=float(os.getenv("PARAKEET_VOLUME_THRESHOLD", "0.001"))
                )
                logger.info("✅ Parakeet streaming STT ready")
            except Exception as e:
                logger.warning(f"Parakeet streaming failed: {e}")
                try:
                    # Fallback to Parakeet batch mode (higher accuracy)
                    from core.stt.parakeet_batch import ParakeetBatchSTT
                    logger.debug("Falling back to Parakeet batch STT")
                    stt = ParakeetBatchSTT(
                        model_path=stt_config.get("model", "mlx-community/parakeet-tdt-0.6b-v3"),
                        language=stt_config.get("language", "en"),
                        confidence_threshold=float(os.getenv("PARAKEET_BATCH_CONFIDENCE_THRESHOLD", "0.3")),
                        temperature=float(os.getenv("PARAKEET_TEMPERATURE", "0.0"))
                    )
                    logger.info("✅ Parakeet batch STT ready (fallback)")
                except Exception as e2:
                    logger.error(f"Parakeet batch also failed: {e2}")
                    # Final fallback to Whisper MLX
                    logger.warning("Using Whisper MLX as final fallback")
                    stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)
        elif self.config.stt_engine == "parakeet":
            # Support legacy "parakeet" name for backward compatibility
            try:
                from core.stt.parakeet_streaming import ParakeetStreamingSTT
                logger.debug("Using Parakeet streaming STT (legacy name)")
                stt = ParakeetStreamingSTT(
                    model_path=stt_config.get("model", "mlx-community/parakeet-tdt-0.6b-v3"),
                    language=stt_config.get("language", "en"),
                    chunk_duration=float(os.getenv("PARAKEET_CHUNK_DURATION", "1.0")),
                    enable_vad=os.getenv("PARAKEET_ENABLE_VAD", "false").lower() in ("1", "true", "yes")
                )
                logger.info("✅ Parakeet streaming STT ready")
            except Exception as e:
                logger.error(f"❌ Parakeet STT failed: {e}", exc_info=True)
                logger.warning("Falling back to Whisper MLX batch mode")
                stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)
        elif self.config.stt_engine == "whisper_mlx":
            logger.debug("Using Whisper MLX batch mode")
            stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)
        else:
            logger.warning(f"Unknown STT engine: {self.config.stt_engine}, using Whisper MLX fallback")
            stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)

        self._services_cache['stt'] = stt
        return stt

    def create_tts_service(self) -> Any:
        """Create TTS service based on configuration."""
        tts_config = self.config.get_component_config("tts")

        if self.config.tts_engine == "kokoro_professional":
            logger.debug("Using Professional Kokoro TTS (default optimized)")
            tts = ProfessionalKokoroTTSService(
                voice=tts_config["voice"],
                speed=tts_config["speed"],
                sample_rate=tts_config["sample_rate"],
                fade_duration_ms=tts_config["fade_duration_ms"],
                target_peak_db=tts_config["target_peak_db"],
                enable_quality_logging=tts_config["enable_quality_logging"]
            )
            logger.info("✅ Professional Kokoro TTS ready")
        elif self.config.tts_engine == "kokoro_mlx":
            logger.debug("Using Ultra-Low Latency MLX Kokoro TTS")
            from core.tts.tts_mlx_ultra_low_latency import TTSMLXUltraLowLatency
            tts = TTSMLXUltraLowLatency(
                model="mlx-community/Kokoro-82M-bf16",
                voice=tts_config["voice"],
                speed=tts_config["speed"],
                sample_rate=tts_config["sample_rate"],
                buffer_ms=50  # 50ms buffer for optimal latency
            )
            logger.info("✅ Ultra-Low Latency MLX Kokoro TTS ready")
        elif self.config.tts_engine == "fastapi_streaming" and FASTAPI_TTS_AVAILABLE:
            logger.debug("Using FastAPI Streaming TTS (legacy)")
            tts = FastAPIStreamingTTS(
                voice=tts_config["voice"],
                speed=tts_config["speed"],
                sample_rate=tts_config["sample_rate"],
                socket_path="/tmp/fastapi-tts.sock"
            )
            logger.info("✅ FastAPI Streaming TTS ready")
        else:
            logger.warning(f"Unknown TTS engine: {self.config.tts_engine}, falling back to MLX Kokoro")
            tts = MLXKokoroTTSService(
                voice=tts_config["voice"],
                speed=tts_config["speed"],
                sample_rate=tts_config["sample_rate"]
            )
            logger.info("✅ MLX Kokoro TTS ready (fallback)")

        self._services_cache['tts'] = tts
        return tts

    def create_llm_service(self) -> OpenAILLMService:
        """Create LLM service with streaming configuration."""
        llm_config = self.config.get_component_config("llm")
        use_llm_streaming = os.getenv("LLM_USE_STREAMING", "true").lower() == "true"

        llm = OpenAILLMService(
            api_key=llm_config["api_key"],
            model=llm_config["model"],
            base_url=llm_config["base_url"],
            max_tokens=llm_config["max_tokens"],
            stream=use_llm_streaming,
            extra_body={
                "think": False,
                "stream": use_llm_streaming,
                "options": {
                    "num_predict": 768,
                    "temperature": llm_config["temperature"],
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_ctx": 4096,
                    "num_batch": 64,
                    "use_mlock": True,
                    "f16_kv": True,
                    "keep_alive": "15m"
                }
            },
        )

        if use_llm_streaming:
            logger.debug("LLM streaming enabled for lower latency")
        else:
            logger.debug("LLM streaming disabled, using batch mode")

        self._services_cache['llm'] = llm
        return llm

    def create_memory_processor(self, context_aggregator: Any, session_tracker: SessionTracker) -> HotPathMemoryProcessor:
        """Create HotMem memory processor."""
        memory = HotPathMemoryProcessor(
            sqlite_path=os.getenv("MEMORY_SQLITE_PATH", ":memory:"),
            lmdb_dir=os.getenv("MEMORY_LMDB_PATH", None),
            user_id=os.getenv("USER_ID", "default-user"),
            enable_metrics=True,
            context_aggregator=context_aggregator,
            session_tracker=session_tracker,
            agent_id=os.getenv("AGENT_ID", "locat"),
        )

        self._services_cache['memory'] = memory
        return memory

    def create_hotmem_service(self, session_tracker: Optional[SessionTracker] = None) -> HotMemService:
        """Create HotMemService (Pipecat-compatible memory service)."""
        # Create confidence strategy based on configuration
        confidence_strategy = self._create_confidence_strategy()

        hotmem_service = HotMemService(
            user_id=os.getenv("USER_ID", "default-user"),
            agent_id=os.getenv("AGENT_ID", "locat"),
            run_id=f"session_{os.getenv('USER_ID', 'default')}",
            sqlite_path=os.getenv("MEMORY_SQLITE_PATH"),
            lmdb_dir=os.getenv("MEMORY_LMDB_PATH"),
            session_tracker=session_tracker,
            confidence_strategy=confidence_strategy
        )

        self._services_cache['hotmem_service'] = hotmem_service
        logger.info(f"✅ HotMemService created with {type(confidence_strategy).__name__ if confidence_strategy else 'default'} confidence strategy")
        return hotmem_service

    def _create_confidence_strategy(self):
        """Create confidence strategy from environment configuration."""
        from core.memory.confidence_strategy import create_confidence_strategy

        strategy_name = os.getenv("CONFIDENCE_STRATEGY", "relation_type")

        try:
            strategy = create_confidence_strategy(strategy_name)
            logger.debug(f"Using confidence strategy: {strategy_name}")
            return strategy
        except ValueError as e:
            logger.warning(f"Invalid confidence strategy '{strategy_name}', using default: {e}")
            return None  # Will use default in HotMemory

    def create_intent_service(self) -> Optional[Any]:
        """Create intent classification service for smart memory processing."""
        if not INTENT_SERVICE_AVAILABLE:
            logger.debug("Intent service not available - skipping creation")
            return None

        if not os.getenv("INTENT_CLASSIFICATION_ENABLED", "true").lower() == "true":
            logger.debug("Intent classification disabled via environment variable")
            return None

        try:
            intent_service = get_intent_service()
            logger.info("✅ Intent classification service ready")
            self._services_cache['intent'] = intent_service
            return intent_service
        except Exception as e:
            logger.error(f"Failed to create intent service: {e}")
            return None

    def create_context_aggregator(self, llm_service: OpenAILLMService, system_instruction: str) -> Any:
        """Create LLM context aggregator."""
        context = OpenAILLMContext([{"role": "system", "content": system_instruction}])

        # Determine timeouts based on STT streaming capability
        use_streaming_stt = self.config.stt_engine in ["parakeet_streaming", "parakeet"]
        default_timeout = "0.12" if use_streaming_stt else "0.2"
        agg_timeout = float(os.getenv("LLM_AGGREGATION_TIMEOUT", default_timeout))
        turn_timeout = float(os.getenv("LLM_TURN_EMULATED_VAD_TIMEOUT", "0.4"))
        agg_interruptions = os.getenv("LLM_ENABLE_EMULATED_VAD_INTERRUPTION", "true").lower() in ("1", "true", "yes")

        context_aggregator = llm_service.create_context_aggregator(
            context,
            user_params=LLMUserAggregatorParams(
                aggregation_timeout=agg_timeout,
                turn_emulated_vad_timeout=turn_timeout,
                enable_emulated_vad_interruptions=agg_interruptions,
            ),
        )

        # Store both context and aggregator for access in event handlers
        self._services_cache['context'] = context
        self._services_cache['context_aggregator'] = context_aggregator
        return context_aggregator

    def create_session_tracker(self) -> SessionTracker:
        """Create session tracker."""
        # Use database tracker if configured and available
        use_db = os.getenv("SESSION_USE_DATABASE", "false").lower() in ("true", "1", "yes")

        if use_db and DB_TRACKER_AVAILABLE:
            logger.info("Using database-backed SessionTracker")
            tracker = DatabaseSessionTracker()
        else:
            logger.info("Using JSON-based SessionTracker")
            tracker = SessionTracker()

        self._services_cache['session_tracker'] = tracker
        return tracker

    def create_rtvi_processor(self) -> RTVIProcessor:
        """Create RTVI processor for client UI events."""
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
        self._services_cache['rtvi'] = rtvi
        return rtvi

    def create_mic_probe(self) -> Optional[Any]:
        """Create optional mic probe for debugging."""
        if MIC_PROBE_AVAILABLE and os.getenv("ENABLE_MIC_PROBE", "false").lower() in ("1", "true", "yes"):
            logger.debug("MicProbe enabled: logging mic input levels")
            probe = MicProbe()
            self._services_cache['mic_probe'] = probe
            return probe
        return None

    def create_text_aggregator(self) -> Any:
        """Create fast text aggregator for fluid speech."""
        from core.aggregators import FastTextAggregator
        aggregator = FastTextAggregator(min_tokens=175, max_tokens=250, max_time=0.5)
        self._services_cache['text_aggregator'] = aggregator
        return aggregator

    def create_pipeline(self, transport: SmallWebRTCTransport, services: Dict[str, Any]) -> Pipeline:
        """Create the main voice agent pipeline."""
        stages = [transport.input()]

        # Optional mic probe
        mic_probe = services.get('mic_probe')
        if mic_probe:
            stages.append(mic_probe)

        # Main processing pipeline
        stages.extend([
            services['stt'],
            services['rtvi'],
            services['memory'],
            services['context_aggregator'].user(),
            services['llm'],
            services['tts'],
            transport.output(),
            services['context_aggregator'].assistant(),
        ])

        pipeline = Pipeline(stages)
        self._services_cache['pipeline'] = pipeline
        return pipeline

    def create_pipeline_task(self, pipeline: Pipeline, rtvi_processor: Optional[RTVIProcessor] = None) -> PipelineTask:
        """Create pipeline task with appropriate parameters."""
        params = PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        )

        observers = []
        if rtvi_processor:
            observers.append(RTVIObserver(rtvi_processor))

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

        # Create RTVI processor
        rtvi = self.create_rtvi_processor()

        # Create optional mic probe
        mic_probe = self.create_mic_probe()

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
            'intent': intent_service  # Intent classification service (optional)
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