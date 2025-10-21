"""
Service Factory - Centralized creation of voice agent services.

Extracts service creation logic from VoiceAgentFactory to improve testability
and separation of concerns. Each service creation method is self-contained
and can be tested independently.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor

# Local imports
from config import VoiceAgentConfig
from core.memory.hotpath_processor import HotPathMemoryProcessor
from core.memory.session_tracker import SessionTracker
from core.memory import HotMemService
from core.memory.anonymous_context import AnonymousAwareContextAggregator

# Import intent service for smart processing
try:
    from core.intent import get_intent_service
    INTENT_SERVICE_AVAILABLE = True
except ImportError:
    INTENT_SERVICE_AVAILABLE = False
    logger.warning("Intent service not available in ServiceFactory")

# Import database session tracker if available
try:
    from core.memory.db_session_tracker import DatabaseSessionTracker
    DB_TRACKER_AVAILABLE = True
except ImportError:
    DB_TRACKER_AVAILABLE = False

from core.tts.kokoro_professional import ProfessionalKokoroTTSService
from core.tts.kokoro_mlx import MLXKokoroTTSService
from core.tts.siri_streaming import SiriStreamingTTSService

# Import optional components
try:
    from mic_probe import MicProbe
    MIC_PROBE_AVAILABLE = True
except ImportError:
    MIC_PROBE_AVAILABLE = False


def resolve_parakeet_model_path(model_id_or_path: str) -> str:
    """
    Resolve Parakeet model path for production (Tauri bundle) vs development.

    In production with HF_HUB_OFFLINE=1, HuggingFace can't resolve repo IDs
    to cached models. This function detects production mode and returns the
    absolute path to the bundled model.

    Args:
        model_id_or_path: HuggingFace model ID or local path

    Returns:
        Absolute path if in production, otherwise returns input unchanged
    """
    default_model_id = "mlx-community/parakeet-tdt-0.6b-v3"

    # Only resolve if we're in Tauri bundle and using default model
    if "TAURI_RESOURCE_DIR" in os.environ and model_id_or_path == default_model_id:
        hf_home = Path(os.environ.get("HF_HOME", ""))
        if hf_home.exists():
            bundled_model = hf_home / "hub" / "models--mlx-community--parakeet-tdt-0.6b-v3"
            if bundled_model.exists():
                logger.debug(f"Resolved Parakeet model to bundled path: {bundled_model}")
                return str(bundled_model)

    return model_id_or_path


class ServiceFactory:
    """Factory for creating individual voice agent services with dependency injection."""

    def __init__(self, config: VoiceAgentConfig):
        """
        Initialize service factory with configuration.

        Args:
            config: Voice agent configuration object
        """
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
                video_in_enabled=self.config.video_input_enabled,
                video_out_enabled=self.config.video_out_enabled,
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

        if self.config.video_input_enabled:
            logger.info(f"📹 Video input ENABLED (target_fps={self.config.video_target_fps})")

        self._services_cache['transport'] = transport
        return transport

    def create_stt_service(self) -> Any:
        """Create STT service based on configuration."""
        stt_config = self.config.get_component_config("stt")

        if self.config.stt_engine == "parakeet_streaming":
            try:
                # Try Parakeet streaming first (ultra-low latency)
                from core.stt.parakeet_streaming import ParakeetStreamingSTT
                model_path = resolve_parakeet_model_path(
                    stt_config.get("model", "mlx-community/parakeet-tdt-0.6b-v3")
                )
                logger.debug(f"Initializing Parakeet streaming STT with model: {model_path}")
                stt = ParakeetStreamingSTT(
                    model_path=model_path,
                    language=stt_config.get("language", "en"),
                    chunk_duration=float(os.getenv("PARAKEET_CHUNK_DURATION", "1.0")),
                    enable_vad=os.getenv("PARAKEET_ENABLE_VAD", "false").lower() in ("1", "true", "yes"),
                    temperature=float(os.getenv("PARAKEET_TEMPERATURE", "0.0")),
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
                    model_path = resolve_parakeet_model_path(
                        stt_config.get("model", "mlx-community/parakeet-tdt-0.6b-v3")
                    )
                    stt = ParakeetBatchSTT(
                        model_path=model_path,
                        language=stt_config.get("language", "en"),
                        temperature=float(os.getenv("PARAKEET_TEMPERATURE", "0.0"))
                    )
                    logger.info("✅ Parakeet batch STT ready (fallback)")
                except Exception as e2:
                    logger.error(f"Parakeet batch also failed: {e2}")
                    # Final fallback to Whisper MLX
                    logger.warning("Using Whisper MLX as final fallback")
                    stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)
        elif self.config.stt_engine == "parakeet_batch":
            # Explicit batch mode for quality comparison
            try:
                from core.stt.parakeet_batch import ParakeetBatchSTT
                logger.debug("Using Parakeet batch STT (explicit)")

                model_path = resolve_parakeet_model_path(
                    stt_config.get("model", "mlx-community/parakeet-tdt-0.6b-v3")
                )
                stt = ParakeetBatchSTT(
                    model_path=model_path,
                    language=stt_config.get("language", "en"),
                    temperature=float(os.getenv("PARAKEET_TEMPERATURE", "0.0"))
                )
                logger.info("✅ Parakeet batch STT ready")
            except Exception as e:
                logger.error(f"❌ Parakeet batch STT failed: {e}", exc_info=True)
                logger.warning("Falling back to Whisper MLX batch mode")
                stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)
        elif self.config.stt_engine == "parakeet":
            # Support legacy "parakeet" name for backward compatibility
            try:
                from core.stt.parakeet_streaming import ParakeetStreamingSTT
                logger.debug("Using Parakeet streaming STT (legacy name)")
                model_path = resolve_parakeet_model_path(
                    stt_config.get("model", "mlx-community/parakeet-tdt-0.6b-v3")
                )
                stt = ParakeetStreamingSTT(
                    model_path=model_path,
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

    def create_tts_service(self, use_boundaries: bool = True) -> Any:
        """
        Create TTS service based on configuration.

        Args:
            use_boundaries: Enable sentence boundary detection (default True).
                           Set False for intro messages to play as one unit.
        """
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
            # Use buffer_ms from environment or default to 40ms
            buffer_ms = int(os.getenv("TTS_BUFFER_MS", "40"))
            tts = TTSMLXUltraLowLatency(
                model="mlx-community/Kokoro-82M-bf16",
                voice=tts_config["voice"],
                speed=tts_config["speed"],
                sample_rate=tts_config["sample_rate"],
                buffer_ms=buffer_ms,  # Respects TTS_BUFFER_MS from .env
                use_boundaries=use_boundaries,  # Control sentence boundary detection
                aggregate_sentences=use_boundaries  # CRITICAL: Disable sentence aggregation for intro
            )
            logger.info("✅ Ultra-Low Latency MLX Kokoro TTS ready")
        elif self.config.tts_engine == "siri_streaming":
            logger.debug("Using Siri Streaming TTS (native macOS)")
            # Determine binary path (dev vs production)
            from pathlib import Path
            import os

            # Try multiple possible locations for the siri-tts binary
            binary_candidates = []

            # FIRST: Check TAURI_RESOURCE_DIR (Tauri bundle production mode)
            if "TAURI_RESOURCE_DIR" in os.environ:
                tauri_resource_dir = Path(os.environ["TAURI_RESOURCE_DIR"])
                binary_candidates.append(tauri_resource_dir / "sidecar" / "siri-tts" / "siri-tts")

            # Fallback paths for development and alternative production layouts
            binary_candidates.extend([
                # Production (Tauri bundle): Resources/siri-tts/siri-tts
                Path(__file__).parent.parent.parent / "siri-tts" / "siri-tts",
                # Development: app/src-tauri/sidecar/siri-tts/siri-tts
                Path(__file__).parent.parent.parent.parent / "app" / "src-tauri" / "sidecar" / "siri-tts" / "siri-tts",
                # Alternative production path: _up_/siri-tts/siri-tts
                Path(__file__).parent.parent.parent / "_up_" / "siri-tts" / "siri-tts",
            ])

            binary_path = None
            for candidate in binary_candidates:
                if candidate.exists():
                    binary_path = candidate
                    logger.debug(f"Found Siri TTS binary at: {binary_path}")
                    break

            if not binary_path:
                logger.error(f"Siri TTS binary not found in any of: {[str(p) for p in binary_candidates]}")
                raise FileNotFoundError(f"siri-tts binary not found. Searched: {binary_candidates}")

            tts = SiriStreamingTTSService(
                binary_path=str(binary_path),
                language=os.getenv("SIRI_TTS_LANGUAGE", "en-US"),
                voice_id=os.getenv("SIRI_TTS_VOICE_ID"),  # Optional override
                rate=float(os.getenv("SIRI_TTS_RATE", "0.52")),
                pitch=float(os.getenv("SIRI_TTS_PITCH", "1.0")),
                sample_rate=tts_config["sample_rate"]
            )
            logger.info("✅ Siri Streaming TTS ready")
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
        # Align with existing HOTMEM_* envs while supporting MEMORY_* overrides
        sqlite_path = os.getenv("MEMORY_SQLITE_PATH") or os.getenv("HOTMEM_SQLITE") or None
        lmdb_dir = os.getenv("MEMORY_LMDB_PATH") or os.getenv("HOTMEM_LMDB_DIR") or None

        memory = HotPathMemoryProcessor(
            sqlite_path=sqlite_path,
            lmdb_dir=lmdb_dir,
            user_id=os.getenv("USER_ID", "default-user"),
            enable_metrics=True,
            context_aggregator=context_aggregator,
            session_tracker=session_tracker,
            agent_id=os.getenv("AGENT_ID", "locat"),
        )

        self._services_cache['memory'] = memory
        return memory

    def create_audio_intelligence_processor(self) -> Any:
        """Create Audio Intelligence processor for speaker recognition, emotion, prosody."""
        audio_intel_enabled = os.getenv("AUDIO_INTELLIGENCE_ENABLED", "true").lower() in ("true", "1", "yes")

        if not audio_intel_enabled:
            logger.info("Audio intelligence disabled")
            return None

        try:
            from core.audio import AudioIntelligenceProcessor

            # Determine device (MPS for Apple Silicon, CPU fallback)
            device = "mps" if os.getenv("AUDIO_INTEL_USE_MPS", "true").lower() in ("true", "1", "yes") else "cpu"

            audio_intel = AudioIntelligenceProcessor(
                profile_dir=os.getenv("SPEAKER_PROFILE_DIR", "data/speaker_profiles"),
                similarity_threshold=float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.75")),
                min_utterance_duration_sec=float(os.getenv("SPEAKER_MIN_UTTERANCE_SEC", "1.0")),
                auto_enroll_utterances=int(os.getenv("SPEAKER_AUTO_ENROLL_UTTERANCES", "3")),
                consistency_threshold=float(os.getenv("SPEAKER_CONSISTENCY_THRESHOLD", "0.80")),
                sample_rate=16000,
                device=device,
                enable_emotion=os.getenv("AUDIO_INTEL_ENABLE_EMOTION", "true").lower() in ("true", "1", "yes"),
                enable_prosody=os.getenv("AUDIO_INTEL_ENABLE_PROSODY", "true").lower() in ("true", "1", "yes"),
                # Privacy-First
                privacy_mode=os.getenv("AUDIO_INTEL_PRIVACY_MODE", "auto_enroll"),
                require_consent=os.getenv("AUDIO_INTEL_REQUIRE_CONSENT", "false").lower() in ("true", "1", "yes"),
                consent_timeout_sec=int(os.getenv("AUDIO_INTEL_CONSENT_TIMEOUT_SEC", "300")),
            )

            logger.info(f"✅ Audio Intelligence processor ready on {device}")
            self._services_cache['audio_intelligence'] = audio_intel
            return audio_intel

        except ImportError as e:
            logger.error(f"AudioIntelligenceProcessor not available: {e}")
            logger.info("Install with: pip install speechbrain")
            return None
        except Exception as e:
            logger.error(f"Failed to create AudioIntelligenceProcessor: {e}")
            return None

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

    def create_context_aggregator(
        self,
        llm_service: OpenAILLMService,
        system_instruction: str,
        factory_ref: Optional[Any] = None
    ) -> Any:
        """
        Create LLM context aggregator.

        Args:
            llm_service: LLM service instance
            system_instruction: System prompt to use
            factory_ref: Reference to parent factory for dynamic prompt rebuilding
        """
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

        # Note: Context Guide removed - instructions for using memory bullets
        # should be integrated into the system_instruction (persona prompt) instead
        # This ensures proper message ordering: Session -> Persona -> Memory -> History

        # Store both context and aggregator for access in event handlers
        # Wrap with anonymous-aware functionality
        # Pass factory reference so anonymous mode can rebuild system prompt
        anonymous_aggregator = AnonymousAwareContextAggregator(
            context_aggregator,
            context,
            memory_processor=None,  # Will be linked after memory creation
            factory=factory_ref  # Pass factory for dynamic prompt rebuilding
        )

        self._services_cache['context'] = context
        self._services_cache['anonymous_aggregator'] = anonymous_aggregator
        return anonymous_aggregator

    def create_session_tracker(self) -> SessionTracker:
        """Create session tracker with ephemeral mode support."""
        from core.memory.config_manager import MemoryConfiguration

        # Get ephemeral mode from memory configuration
        memory_config = MemoryConfiguration.from_env()
        storage_path = os.getenv("SESSION_STATS_PATH")

        # Use database tracker if configured and available
        use_db = os.getenv("SESSION_USE_DATABASE", "false").lower() in ("true", "1", "yes")

        if use_db and DB_TRACKER_AVAILABLE:
            logger.info("Using database-backed SessionTracker")
            tracker = DatabaseSessionTracker()
        else:
            logger.info(
                f"Using JSON-based SessionTracker "
                f"(ephemeral={memory_config.ephemeral_mode})"
            )
            tracker = SessionTracker(
                storage_path=storage_path,
                ephemeral=memory_config.ephemeral_mode
            )

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

    def has_existing_speaker_profiles(self) -> bool:
        """Check if any speaker profiles exist."""
        try:
            from pathlib import Path
            profile_dir = Path(self.config.speaker_profile_dir)
            auto_dir = profile_dir / "auto_enrolled"

            if not auto_dir.exists():
                return False

            # Check for .pt files
            profiles = list(auto_dir.glob("*.pt"))
            has_profiles = len(profiles) > 0

            logger.debug(f"[ServiceFactory] Found {len(profiles)} existing speaker profiles")
            return has_profiles
        except Exception as e:
            logger.warning(f"[ServiceFactory] Error checking speaker profiles: {e}")
            return False

    def get_service(self, name: str) -> Any:
        """Get a cached service by name."""
        return self._services_cache.get(name)

    def clear_cache(self):
        """Clear the services cache."""
        self._services_cache.clear()
