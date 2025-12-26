from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from loguru import logger

from config import VoiceAgentConfig
from core.factories.utils.service_validator import TTSServiceValidator


class TTSCreationStrategy(ABC):
    def __init__(self, config: VoiceAgentConfig, tts_config: Dict[str, Any]):
        self.config = config
        self.tts_config = tts_config

    @abstractmethod
    def create(self, use_boundaries: bool = True) -> Any:
        raise NotImplementedError


class KokoroMLXStrategy(TTSCreationStrategy):
    def create(self, use_boundaries: bool = True) -> Any:
        # Import via service_factory to honor tests' monkeypatching of MLXKokoroTTSService
        from core.factories import service_factory as sf
        return sf.MLXKokoroTTSService(
            voice=self.tts_config["voice"],
            speed=self.tts_config["speed"],
            sample_rate=self.tts_config["sample_rate"],
        )


class KokoroProfessionalStrategy(TTSCreationStrategy):
    def create(self, use_boundaries: bool = True) -> Any:
        # Import via service_factory to honor tests' monkeypatching of ProfessionalKokoroTTSService
        from core.factories import service_factory as sf
        # Use aggregate_sentences=True for both intro and conversation (same working pattern)
        return sf.ProfessionalKokoroTTSService(
            voice=self.tts_config["voice"],
            speed=self.tts_config["speed"],
            sample_rate=self.tts_config["sample_rate"],
            fade_duration_ms=self.tts_config["fade_duration_ms"],
            target_peak_db=self.tts_config["target_peak_db"],
            enable_quality_logging=self.tts_config["enable_quality_logging"],
            # aggregate_sentences defaults to True, which works for both intro and conversation
        )


class KokoroPyTorchStrategy(TTSCreationStrategy):
    def create(self, use_boundaries: bool = True) -> Any:
        from core.tts.kokoro_pytorch import KokoroPyTorchTTSService
        from core.utils.model_validator import ModelValidationError

        validator = TTSServiceValidator()
        max_retries = int(os.getenv("KOKORO_PYTORCH_MAX_RETRIES", "3"))
        retry_delay = float(os.getenv("KOKORO_PYTORCH_RETRY_DELAY", "2.0"))

        last_exception: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(
                        f"🔄 Kokoro PyTorch TTS initialization attempt {attempt + 1}/{max_retries + 1}"
                    )
                    time.sleep(retry_delay * attempt)
                else:
                    logger.debug("🚀 Attempting Kokoro PyTorch TTS initialization")

                tts = KokoroPyTorchTTSService(
                    voice=self.tts_config["voice"],
                    speed=self.tts_config["speed"],
                    sample_rate=self.tts_config["sample_rate"],
                )

                if validator.is_functional(tts):
                    return tts
                else:
                    last_exception = Exception("TTS service verification failed")
                    logger.warning(
                        f"⚠️ Kokoro PyTorch TTS created but verification failed (attempt {attempt + 1})"
                    )

            except ModelValidationError as e:
                last_exception = e
                if attempt == max_retries:
                    logger.error("💡 Model validation troubleshooting steps:")
                    logger.error("   1. Run server with internet to download models")
                    logger.error("   2. Check HUGGINGFACE_HUB_CACHE environment variable")
                    logger.error("   3. Verify model files exist in cache directory")
                    logger.error("   4. Try setting SKIP_TTS_VALIDATION=true for production bundles")
            except ImportError as e:
                logger.error(f"❌ Kokoro PyTorch import failed: {e}")
                logger.error("💡 Install with: pip install kokoro>=0.9.2")
                raise
            except Exception as e:
                last_exception = e
                # best-effort diagnostics
                emsg = str(e).lower()
                if "metal" in emsg or "gpu" in emsg:
                    logger.warning("💡 Metal/GPU error detected - suggests concurrent Metal access")
                elif "offline" in emsg or "cache" in emsg:
                    logger.warning("💡 Cache/offline error detected - models may be missing")

        if last_exception:
            logger.error("💥 All Kokoro PyTorch TTS initialization attempts failed")
            logger.info("💡 Fallback options: kokoro_mlx or siri_streaming")
            raise last_exception

        raise RuntimeError("Kokoro PyTorch TTS initialization failed with unknown error")


class SiriStreamingStrategy(TTSCreationStrategy):
    def __init__(self, config: VoiceAgentConfig, tts_config: Dict[str, Any], siri_creator: Callable[[Dict[str, Any], bool], Any]):
        super().__init__(config, tts_config)
        self._siri_creator = siri_creator

    def create(self, use_boundaries: bool = True) -> Any:
        return self._siri_creator(self.tts_config, use_boundaries)


class SupertonicStrategy(TTSCreationStrategy):
    """
    Strategy for Supertonic TTS - lightning-fast on-device synthesis.

    Supertonic is a 66M parameter model achieving 30-45x real-time on Apple Silicon.
    It handles complex text (numbers, dates, currency) automatically.
    """

    def create(self, use_boundaries: bool = True) -> Any:
        from core.tts.supertonic_service import SupertonicTTSService

        # Get Supertonic-specific config with fallbacks
        voice = self.tts_config.get("supertonic_voice", "F1")
        model_dir = self.tts_config.get("supertonic_model_dir") or os.getenv("SUPERTONIC_MODEL_PATH")
        total_steps = int(self.tts_config.get("supertonic_total_steps", 2))
        speed = float(self.tts_config.get("speed", 1.0))
        sample_rate = int(self.tts_config.get("sample_rate", 24000))

        logger.info(f"🎵 Creating Supertonic TTS: voice={voice}, steps={total_steps}, speed={speed}")

        return SupertonicTTSService(
            voice=voice,
            model_dir=model_dir,
            total_steps=total_steps,
            speed=speed,
            target_sample_rate=sample_rate,
            aggregate_sentences=use_boundaries,
        )
