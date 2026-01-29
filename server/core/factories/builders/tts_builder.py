from typing import Any, Dict, Callable
from loguru import logger

from config import VoiceAgentConfig
from core.factories.strategies.tts_strategies import (
    TTSCreationStrategy,
    KokoroMLXStrategy,
    KokoroProfessionalStrategy,
    KokoroPyTorchStrategy,
    SiriStreamingStrategy,
    SupertonicStrategy,
    Qwen3TTSStrategy,
)


class TTSServiceBuilder:
    """Builds TTS services using strategies and default macOS Siri fallback."""

    def __init__(self, config: VoiceAgentConfig, siri_creator: Callable[[Dict[str, Any], bool], Any]):
        self.config = config
        self._siri_creator = siri_creator

    def build(self, use_boundaries: bool = True) -> Any:
        tts_config = self.config.get_component_config("tts")
        engine = self.config.tts_engine

        def _strategy_for(engine_name: str) -> TTSCreationStrategy:
            if engine_name == "qwen3":
                return Qwen3TTSStrategy(self.config, tts_config)
            if engine_name == "supertonic":
                return SupertonicStrategy(self.config, tts_config)
            if engine_name == "kokoro_professional":
                return KokoroProfessionalStrategy(self.config, tts_config)
            if engine_name == "kokoro_mlx":
                return KokoroMLXStrategy(self.config, tts_config)
            if engine_name == "kokoro_pytorch":
                return KokoroPyTorchStrategy(self.config, tts_config)
            if engine_name == "siri_streaming":
                return SiriStreamingStrategy(self.config, tts_config, self._siri_creator)
            # Unknown → use Supertonic as default (fastest, most reliable)
            logger.warning(f"Unknown TTS engine '{engine_name}', defaulting to Supertonic")
            return SupertonicStrategy(self.config, tts_config)

        primary = _strategy_for(engine)

        # Define last-resort ordering consistent with current ServiceFactory behavior
        try:
            tts = primary.create(use_boundaries=use_boundaries)
            return tts
        except Exception as e:
            logger.error(f"❌ TTS primary strategy '{engine}' failed: {e}")

        # Fallback to Siri, then last resort depending on primary
        try:
            tts = SiriStreamingStrategy(self.config, tts_config, self._siri_creator).create(
                use_boundaries=use_boundaries
            )
            logger.info("✅ Siri Streaming TTS ready (fallback)")
            return tts
        except Exception as e2:
            logger.warning(f"Siri TTS fallback failed: {e2}")

        # Last resort
        if engine == "kokoro_mlx":
            return KokoroProfessionalStrategy(self.config, tts_config).create(use_boundaries=use_boundaries)
        # For professional and pytorch, fall back to MLX
        return KokoroMLXStrategy(self.config, tts_config).create(use_boundaries=use_boundaries)
