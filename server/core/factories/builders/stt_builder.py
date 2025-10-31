from typing import Any, Callable, Dict, List
import os

from loguru import logger

from config import VoiceAgentConfig
from core.factories.utils.fallback_chain import FallbackChainManager
from core.factories.utils.model_resolver import resolve_parakeet_model_path


class STTServiceBuilder:
    """Builds STT services with explicit fallback chains."""

    def __init__(self, config: VoiceAgentConfig, preloaded_models=None):
        self.config = config
        self.preloaded_models = preloaded_models  # NEW: Accept preloaded models

    def build(self) -> Any:
        stt_config = self.config.get_component_config("stt")
        engine = self.config.stt_engine

        # Creation functions
        def _create_parakeet_isolated() -> Any:
            from core.stt.parakeet_isolated import ParakeetIsolatedSTT
            model_path = resolve_parakeet_model_path(stt_config.get("model", "mlx-community/parakeet-tdt-0.6b-v3"))
            return ParakeetIsolatedSTT(
                model_path=model_path,
                streaming=os.getenv("PARAKEET_STREAMING", "true").lower() in ("1", "true", "yes"),
                context_size=tuple(map(int, os.getenv("PARAKEET_CONTEXT_SIZE", "256,256").split(","))),
                depth=int(os.getenv("PARAKEET_DEPTH", "3")),
            )

        def _create_parakeet_streaming() -> Any:
            from core.stt.parakeet_streaming import ParakeetStreamingSTT
            model_path = resolve_parakeet_model_path(stt_config.get("model", "mlx-community/parakeet-tdt-0.6b-v3"))
            return ParakeetStreamingSTT(
                model_path=model_path,
                language=stt_config.get("language", "en"),
                chunk_duration=float(os.getenv("PARAKEET_CHUNK_DURATION", "1.0")),
                enable_vad=os.getenv("PARAKEET_ENABLE_VAD", "false").lower() in ("1", "true", "yes"),
                temperature=float(os.getenv("PARAKEET_TEMPERATURE", "0.0")),
                sentence_pause_threshold=float(os.getenv("PARAKEET_SENTENCE_PAUSE_THRESHOLD", "1.2")),
                max_chunk_duration=float(os.getenv("PARAKEET_MAX_CHUNK_DURATION", "4.0")),
                context_size=tuple(map(int, os.getenv("PARAKEET_CONTEXT_SIZE", "256,256").split(","))),
                depth=int(os.getenv("PARAKEET_DEPTH", "3")),
                volume_threshold=float(os.getenv("PARAKEET_VOLUME_THRESHOLD", "0.001")),
            )

        def _create_parakeet_batch() -> Any:
            from core.stt.parakeet_batch import ParakeetBatchSTT
            model_path = resolve_parakeet_model_path(stt_config.get("model", "mlx-community/parakeet-tdt-0.6b-v3"))
            return ParakeetBatchSTT(
                model_path=model_path,
                language=stt_config.get("language", "en"),
                temperature=float(os.getenv("PARAKEET_TEMPERATURE", "0.0")),
            )

        def _create_macos_native() -> Any:
            from core.stt.macos_native import MacOSNativeSTT
            return MacOSNativeSTT(
                language=stt_config.get("language", "en-US"),
                sample_rate=int(os.getenv("STT_SAMPLE_RATE", "16000")),
                on_device=os.getenv("MACOS_STT_ON_DEVICE", "true").lower() in ("1", "true", "yes"),
            )

        def _create_whisper_batch() -> Any:
            # Import from service_factory so tests that monkeypatch sf.WhisperSTTServiceMLX/MLXModel see it
            from core.factories import service_factory as sf
            return sf.WhisperSTTServiceMLX(model=sf.MLXModel.MEDIUM)

        def _create_whisper_direct_mlx() -> Any:
            from core.stt.whisper_mlx import DirectMLXWhisperSTT
            # Use preloaded Whisper module if available
            preloaded_whisper = None
            if self.preloaded_models and hasattr(self.preloaded_models, 'whisper_module'):
                preloaded_whisper = self.preloaded_models.whisper_module
                logger.debug("🚀 STTBuilder using preloaded Whisper module")

            return DirectMLXWhisperSTT(
                model=stt_config.get("model", "mlx-community/whisper-small.en-mlx-q4"),
                language=stt_config.get("language", "en"),
                temperature=0.0,
                no_speech_threshold=0.6,
                hallucination_silence_threshold=0.3,
                _preloaded_whisper_module=preloaded_whisper,  # Pass preloaded module
            )

        chains: Dict[str, List[Callable[[], Any]]] = {
            "parakeet_isolated": [
                _create_parakeet_isolated,
                _create_parakeet_streaming,
                _create_macos_native,
                _create_whisper_batch,
            ],
            "parakeet_streaming": [
                _create_parakeet_streaming,
                _create_parakeet_batch,
                _create_macos_native,
                _create_whisper_batch,
            ],
            "parakeet_batch": [
                _create_parakeet_batch,
                _create_macos_native,
                _create_whisper_batch,
            ],
            "parakeet": [
                _create_parakeet_streaming,
                _create_macos_native,
                _create_whisper_batch,
            ],
            "macos_native": [
                _create_macos_native,
                _create_whisper_batch,
            ],
            "whisper_mlx_direct": [
                _create_whisper_direct_mlx,
                _create_macos_native,
                _create_whisper_batch,
            ],
            "whisper_mlx": [
                _create_whisper_batch,
            ],
        }

        chain = chains.get(engine, [_create_macos_native, _create_whisper_batch])
        stt = FallbackChainManager().execute(chain, context=f"STT({engine})")
        return stt
