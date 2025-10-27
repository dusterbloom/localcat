"""
Lightweight validators for service instances.
"""

from loguru import logger


class TTSServiceValidator:
    """Sanity checks for TTS services to ensure they are initialized."""

    def is_functional(self, tts_service) -> bool:
        try:
            if not hasattr(tts_service, "_pipeline"):
                logger.error("❌ TTS service missing _pipeline attribute")
                return False
            if not hasattr(tts_service, "_voice"):
                logger.error("❌ TTS service missing _voice attribute")
                return False
            if getattr(tts_service, "_pipeline", None) is None:
                logger.error("❌ TTS service pipeline is None")
                return False

            pipe = getattr(tts_service, "_pipeline", None)
            if hasattr(pipe, "lang_code"):
                logger.debug(f"✅ TTS service pipeline functional (lang_code: {pipe.lang_code})")
            else:
                logger.debug("✅ TTS service pipeline appears functional")
            return True
        except Exception as e:  # noqa: BLE001 - only used in tests
            logger.error(f"❌ TTS service validation failed: {e}")
            return False

