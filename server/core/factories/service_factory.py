"""
Service Factory - Centralized creation of voice agent services.

Extracts service creation logic from VoiceAgentFactory to improve testability
and separation of concerns. Each service creation method is self-contained
and can be tested independently.
"""

import os
import time
from pathlib import Path
import threading
from typing import Dict, Any, Optional, List, Callable
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
from .utils.fallback_chain import FallbackChainManager, ChainExhaustedError
from .utils.model_resolver import resolve_parakeet_model_path
from .utils.service_validator import TTSServiceValidator
from .builders.stt_builder import STTServiceBuilder
from .builders.llm_builder import LLMServiceBuilder

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
from core.tts.kokoro_pytorch import KokoroPyTorchTTSService
from core.tts.siri_streaming import SiriStreamingTTSService
from .builders.tts_builder import TTSServiceBuilder

# Import optional components
try:
    from mic_probe import MicProbe
    MIC_PROBE_AVAILABLE = True
except ImportError:
    MIC_PROBE_AVAILABLE = False


def _prewarm_llm_service(llm_service: OpenAILLMService, llm_config: Dict[str, Any]) -> None:
    """
    Prewarm the LLM service to prevent cold start latency on first inference.

    Sends a minimal warmup request to LM Studio to ensure the model is loaded
    and ready for fast responses.
    """
    import asyncio
    import httpx

    async def _send_warmup_request():
        """Send minimal warmup request to LLM service."""
        try:
            # Extract base URL from service for direct HTTP call
            base_url = llm_config.get("base_url", "http://127.0.0.1:1234/v1")
            model = llm_config.get("model", "unknown")

            # Create minimal warmup request
            warmup_url = f"{base_url}/chat/completions"
            warmup_data = {
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,  # Minimal response
                "stream": False
            }

            logger.info(f"🔥 Prewarming LLM model: {model}")
            start_time = time.time()

            async with httpx.AsyncClient(timeout=60.0) as client:  # 60 second timeout for cold start
                response = await client.post(
                    warmup_url,
                    json=warmup_data,
                    headers={"Authorization": f"Bearer {llm_config.get('api_key', 'not-needed')}"}
                )

                if response.status_code == 200:
                    warmup_time = (time.time() - start_time) * 1000
                    logger.info(f"✅ LLM model prewarmed in {warmup_time:.1f}ms")
                else:
                    logger.warning(f"⚠️ LLM prewarm failed: HTTP {response.status_code}")

        except asyncio.TimeoutError:
            logger.warning("⚠️ LLM prewarm timed out (model may still be loading)")
        except Exception as e:
            logger.warning(f"⚠️ LLM prewarm failed: {e}")

    # Run warmup in background to not block service creation
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.create_task(_send_warmup_request())
    except RuntimeError:
        # No event loop, create new one for warmup
        try:
            asyncio.run(_send_warmup_request())
        except Exception:
            logger.debug("LLM prewarm skipped (no event loop available)")


## moved to utils/model_resolver.py


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
        # Concurrency-safe creation locks to avoid double-initialization
        self._stt_lock = threading.Lock()
        self._tts_lock = threading.Lock()
        self._llm_lock = threading.Lock()

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
        cache_enabled = os.getenv("SERVICE_FACTORY_CACHE_STT", "true").lower() in ("true", "1", "yes")
        if cache_enabled and 'stt' in self._services_cache:
            logger.debug("Using cached STT service")
            return self._services_cache['stt']

        with self._stt_lock:
            if cache_enabled and 'stt' in self._services_cache:
                logger.debug("Using cached STT service (post-lock)")
                return self._services_cache['stt']

            # Delegate to builder (preserves behavior)
            stt = STTServiceBuilder(self.config).build()

        if cache_enabled:
            self._services_cache['stt'] = stt
        return stt

    def create_tts_service(self, use_boundaries: bool = True) -> Any:
        """
        Create TTS service based on configuration.

        Args:
            use_boundaries: Enable sentence boundary detection (default True).
                           Set False for intro messages to play as one unit.
        """
        # Per-boundary caching (enabled by default)
        cache_enabled = os.getenv("SERVICE_FACTORY_CACHE_TTS", "true").lower() in ("true", "1", "yes")
        cache_key = f"tts_{bool(use_boundaries)}"
        if cache_enabled and cache_key in self._services_cache:
            logger.debug(f"Using cached TTS service for use_boundaries={use_boundaries}")
            return self._services_cache[cache_key]

        with self._tts_lock:
            if cache_enabled and cache_key in self._services_cache:
                logger.debug(f"Using cached TTS service for use_boundaries={use_boundaries} (post-lock)")
                return self._services_cache[cache_key]

            tts = TTSServiceBuilder(self.config, siri_creator=self._try_create_siri_tts).build(
                use_boundaries=use_boundaries
            )

            if cache_enabled:
                self._services_cache[cache_key] = tts
            return tts

    def _try_create_siri_tts(self, tts_config: Dict[str, Any], use_boundaries: bool) -> SiriStreamingTTSService:
        """Attempt to create Siri Streaming TTS if the sidecar is available."""
        # Try multiple possible locations for the siri-tts binary
        binary_candidates = []
        if "TAURI_RESOURCE_DIR" in os.environ:
            tauri_resource_dir = Path(os.environ["TAURI_RESOURCE_DIR"])
            binary_candidates.append(tauri_resource_dir / "sidecar" / "siri-tts" / "siri-tts")

        binary_candidates.extend([
            Path(__file__).parent.parent.parent / "siri-tts" / "siri-tts",
            Path(__file__).parent.parent.parent.parent / "app" / "src-tauri" / "sidecar" / "siri-tts" / "siri-tts",
            Path(__file__).parent.parent.parent / "_up_" / "siri-tts" / "siri-tts",
        ])

        binary_path = None
        for candidate in binary_candidates:
            try:
                if candidate.exists():
                    binary_path = candidate
                    logger.debug(f"Found Siri TTS binary at: {binary_path}")
                    break
            except Exception:
                continue

        if not binary_path:
            raise FileNotFoundError(f"siri-tts binary not found. Searched: {binary_candidates}")

        return SiriStreamingTTSService(
            binary_path=str(binary_path),
            language=os.getenv("SIRI_TTS_LANGUAGE", "en-US"),
            voice_id=os.getenv("SIRI_TTS_VOICE_ID"),
            use_system_voice=os.getenv("SIRI_TTS_USE_SYSTEM_VOICE", "false").lower() in ("1", "true", "yes"),
            rate=float(os.getenv("SIRI_TTS_RATE", "0.52")),
            pitch=float(os.getenv("SIRI_TTS_PITCH", "1.0")),
            sample_rate=tts_config["sample_rate"],
            no_boundaries=(not use_boundaries),
        )

    def create_llm_service(self) -> OpenAILLMService:
        """Create LLM service with streaming configuration."""
        # Check if LLM service is already cached
        if 'llm' in self._services_cache:
            logger.debug("Using cached LLM service")
            return self._services_cache['llm']

        with self._llm_lock:
            if 'llm' in self._services_cache:
                logger.debug("Using cached LLM service (post-lock)")
                return self._services_cache['llm']

            logger.debug("Creating new LLM service via builder")
            llm = LLMServiceBuilder(self.config).build()

            # Prewarm HTTP-based models to avoid cold start; Direct MLX doesn't need it
            llm_config = self.config.get_component_config("llm")
            if os.getenv("LLM_PREWARM", "true").lower() in ("true", "1", "yes") and not os.getenv("LLM_USE_DIRECT_MLX", "false").lower() in ("true", "1", "yes"):
                _prewarm_llm_service(llm, llm_config)

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

        # Resolve storage paths for production (expand ~ and $VARS)
        def _expand(p: str | None) -> str | None:
            if not p:
                return p
            try:
                return os.path.expanduser(os.path.expandvars(p))
            except Exception:
                return p

        hotmem_service = HotMemService(
            user_id=os.getenv("USER_ID", "default-user"),
            agent_id=os.getenv("AGENT_ID", "locat"),
            run_id=f"session_{os.getenv('USER_ID', 'default')}",
            sqlite_path=_expand(os.getenv("MEMORY_SQLITE_PATH")),
            lmdb_dir=_expand(os.getenv("MEMORY_LMDB_PATH")),
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
        # Vision injector will be linked later via set_vision_injector()
        anonymous_aggregator = AnonymousAwareContextAggregator(
            context_aggregator,
            context,
            memory_processor=None,  # Will be linked after memory creation
            factory=factory_ref,  # Pass factory for dynamic prompt rebuilding
            vision_injector=None  # Will be linked after vision injector creation

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
        # Align with benchmark findings: slightly smaller, steadier chunks
        # Allow environment overrides for rapid tuning
        def _get_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return default

        def _get_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except Exception:
                return default

        # Favor slightly larger spans to reduce too-frequent TTS calls
        # Allow LLM_* overrides for sentence-aware aggregation
        min_tokens = _get_int("LLM_MIN_TOKENS_FOR_TTS", _get_int("FAST_TEXT_MIN_TOKENS", 175))
        max_tokens = _get_int("LLM_MAX_TOKENS", _get_int("FAST_TEXT_MAX_TOKENS", 250))
        max_time = _get_float("FAST_TEXT_MAX_TIME", 0.5)
        sentence_delims = os.getenv("LLM_SENTENCE_DELIMITERS")

        aggregator = FastTextAggregator(
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            max_time=max_time,
            sentence_delimiters=sentence_delims
        )
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

    def _create_kokoro_pytorch_with_retry(self, tts_config: Dict[str, Any]) -> KokoroPyTorchTTSService:
        """
        Create Kokoro PyTorch TTS service with robust error handling and retry logic.

        This method addresses the root cause of Kokoro PyTorch initialization failures:
        - Model pre-validation before Metal lock acquisition
        - Multiple retry attempts with different configurations
        - Graceful fallback to alternative TTS engines if needed
        """
        import time
        import os
        from core.utils.model_validator import ModelValidationError

        max_retries = int(os.getenv("KOKORO_PYTORCH_MAX_RETRIES", "3"))
        retry_delay = float(os.getenv("KOKORO_PYTORCH_RETRY_DELAY", "2.0"))

        last_exception = None
        validator = TTSServiceValidator()

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"🔄 Kokoro PyTorch TTS initialization attempt {attempt + 1}/{max_retries + 1}")
                    time.sleep(retry_delay * attempt)  # Exponential backoff
                else:
                    logger.debug("🚀 Attempting Kokoro PyTorch TTS initialization")

                # Create the service with current configuration
                tts = KokoroPyTorchTTSService(
                    voice=tts_config["voice"],
                    speed=tts_config["speed"],
                    sample_rate=tts_config["sample_rate"]
                )

                # Verify the service is actually functional
                if validator.is_functional(tts):
                    return tts
                else:
                    logger.warning(f"⚠️ Kokoro PyTorch TTS created but verification failed (attempt {attempt + 1})")
                    last_exception = Exception("TTS service verification failed")

            except ModelValidationError as e:
                logger.error(f"❌ Kokoro PyTorch model validation failed (attempt {attempt + 1}): {e}")
                last_exception = e

                # Provide specific guidance for model validation failures
                if attempt == max_retries:
                    logger.error("💡 Model validation troubleshooting steps:")
                    logger.error("   1. Run server with internet to download models:")
                    logger.error("      python -c 'from kokoro import KPipeline; KPipeline(lang_code=\"a\", repo_id=\"hexgrad/Kokoro-82M\")'")
                    logger.error("   2. Check HUGGINGFACE_HUB_CACHE environment variable")
                    logger.error("   3. Verify model files exist in cache directory")
                    logger.error("   4. Try setting SKIP_TTS_VALIDATION=true for production bundles")

            except ImportError as e:
                logger.error(f"❌ Kokoro PyTorch import failed (attempt {attempt + 1}): {e}")
                last_exception = e
                logger.error("💡 Install with: pip install kokoro>=0.9.2")
                # Import errors won't be fixed by retries, break early
                break

            except Exception as e:
                logger.error(f"❌ Kokoro PyTorch TTS initialization failed (attempt {attempt + 1}): {e}")
                last_exception = e

                # Check for specific error types
                error_msg = str(e).lower()
                if "metal" in error_msg or "gpu" in error_msg:
                    logger.warning("💡 Metal/GPU error detected - this suggests concurrent Metal access")
                    logger.warning("   Try restarting the application and avoiding concurrent ML operations")
                elif "offline" in error_msg or "cache" in error_msg:
                    logger.warning("💡 Cache/offline error detected - model files may be missing")
                    logger.warning("   Try running with internet connectivity first to download models")

        # All retries failed, raise the last exception
        if last_exception:
            logger.error(f"💥 All Kokoro PyTorch TTS initialization attempts failed")

            # Suggest fallback options
            logger.info("💡 Fallback options:")
            logger.info("   1. Use kokoro_mlx engine (MLX-based, more stable on macOS)")
            logger.info("   2. Use siri_streaming engine (native macOS, fastest)")
            logger.info("   3. Set VOICE_AGENT_TTS_ENGINE=kokoro_mlx in .env file")

            raise last_exception
        else:
            raise Exception("Kokoro PyTorch TTS initialization failed with unknown error")

    ## moved to utils/service_validator.py

    def clear_cache(self):
        """Clear the services cache."""
        self._services_cache.clear()
