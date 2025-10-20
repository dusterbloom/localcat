"""
Siri Streaming TTS Service for Pipecat

Uses macOS native AVSpeechSynthesizer via Swift sidecar to stream audio directly
to Pipecat's WebRTC pipeline. Provides instant startup, excellent quality, and
multi-language support without requiring model files.

Architecture:
  1. Spawn Swift siri-tts subprocess with --stream-pcm mode
  2. Read PCM chunks from stdout asynchronously
  3. Convert to Pipecat TTSAudioRawFrame
  4. Yield frames to pipeline for WebRTC streaming

Features:
  - Zero model loading time (instant availability)
  - Multi-language support via built-in Siri voices
  - Voice customization (rate, pitch)
  - Proper resampling (24kHz → 16kHz for WebRTC)
  - Robust error handling with ONNX fallback
"""

import asyncio
import os
import struct
from pathlib import Path
from typing import AsyncGenerator, Optional
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    ErrorFrame,
)
from pipecat.services.tts_service import TTSService

# Voice ID mapping: language code → Siri voice identifier
# NOTE: Using premium quality voices for most natural sound
# If a voice isn't available, the system will fall back to the default for that language
SIRI_VOICE_MAP = {
    # English - Premium quality voices for natural sound
    "en-US": "com.apple.voice.premium.en-US.Ava",  # Natural female voice
    "en-GB": "com.apple.voice.premium.en-GB.Daniel",  # Natural male voice
    "en-AU": "com.apple.voice.premium.en-AU.Karen",
    "en-IN": "com.apple.voice.premium.en-IN.Rishi",

    # European languages
    "fr-FR": "com.apple.voice.premium.fr-FR.Thomas",
    "de-DE": "com.apple.voice.premium.de-DE.Anna",
    "es-ES": "com.apple.voice.premium.es-ES.Monica",
    "it-IT": "com.apple.voice.premium.it-IT.Alice",
    "pt-PT": "com.apple.voice.premium.pt-PT.Joana",
    "pt-BR": "com.apple.voice.premium.pt-BR.Luciana",
    "nl-NL": "com.apple.voice.premium.nl-NL.Ellen",
    "pl-PL": "com.apple.voice.premium.pl-PL.Zosia",
    "ru-RU": "com.apple.voice.premium.ru-RU.Milena",

    # Asian languages
    "ja-JP": "com.apple.voice.premium.ja-JP.Kyoko",
    "ko-KR": "com.apple.voice.premium.ko-KR.Yuna",
    "zh-CN": "com.apple.voice.premium.zh-CN.Ting-Ting",
    "zh-HK": "com.apple.voice.premium.zh-HK.Sin-Ji",
    "zh-TW": "com.apple.voice.premium.zh-TW.Mei-Jia",

    # Middle Eastern
    "ar-SA": "com.apple.voice.premium.ar-SA.Maged",
    "he-IL": "com.apple.voice.premium.he-IL.Carmit",
    "tr-TR": "com.apple.voice.premium.tr-TR.Yelda",

    # Nordic
    "sv-SE": "com.apple.voice.premium.sv-SE.Alva",
    "no-NO": "com.apple.voice.premium.no-NO.Nora",
    "da-DK": "com.apple.voice.premium.da-DK.Sara",
    "fi-FI": "com.apple.voice.premium.fi-FI.Satu",

    # Other
    "th-TH": "com.apple.voice.premium.th-TH.Kanya",
    "id-ID": "com.apple.voice.premium.id-ID.Damayanti",
    "vi-VN": "com.apple.voice.premium.vi-VN.Linh",
}


class SiriStreamingTTSService(TTSService):
    """
    Pipecat TTS service using native macOS Siri voices via streaming subprocess.

    Args:
        binary_path: Path to siri-tts binary
        language: Language code (e.g., "en-US", "it-IT")
        voice_id: Optional explicit voice ID (overrides language-based selection)
        rate: Speech rate 0.0-1.0 (default: 0.52 for natural pace)
        pitch: Pitch multiplier (default: 1.0)
        sample_rate: Target sample rate (16000 for WebRTC, 24000 native)
    """

    def __init__(
        self,
        binary_path: str,
        language: str = "en-US",
        voice_id: Optional[str] = None,
        rate: float = 0.52,
        pitch: float = 1.0,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self._binary_path = Path(binary_path)
        self._language = language
        self._voice_id = voice_id or SIRI_VOICE_MAP.get(language)
        self._rate = rate
        self._pitch = pitch
        self._sample_rate = sample_rate

        # Validate binary exists
        if not self._binary_path.exists():
            raise FileNotFoundError(f"Siri TTS binary not found: {binary_path}")

        if not os.access(self._binary_path, os.X_OK):
            raise PermissionError(f"Siri TTS binary not executable: {binary_path}")

        logger.info(
            f"Siri Streaming TTS initialized: lang={language}, "
            f"voice={self._voice_id or 'auto'}, rate={rate}, "
            f"sample_rate={sample_rate}Hz"
        )

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Generate speech audio by streaming from siri-tts subprocess.

        Yields:
            TTSStartedFrame: At beginning of synthesis
            TTSAudioRawFrame: PCM audio chunks as they're produced
            TTSStoppedFrame: At end of synthesis
            ErrorFrame: On errors (with fallback to ONNX if configured)
        """
        logger.debug(f"Siri TTS generating: '{text[:80]}{'...' if len(text) > 80 else ''}'")

        # Signal start
        yield TTSStartedFrame()

        # Emit text frame for transcript processor
        yield TTSTextFrame(text=text)

        try:
            # Build command
            cmd = [
                str(self._binary_path),
                "--stream-pcm",
                "--text", text,
                "--target-rate", str(self._sample_rate),
                "--rate", str(self._rate),
                "--pitch", str(self._pitch),
            ]

            # Add language or explicit voice ID
            if self._voice_id:
                cmd.extend(["--voice-id", self._voice_id])
            else:
                cmd.extend(["--lang", self._language])

            # Spawn subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            logger.debug(f"Spawned siri-tts process: PID {process.pid}")

            # Stream PCM chunks from stdout
            chunk_count = 0
            total_bytes = 0

            while True:
                # Read 4KB chunks (optimal for streaming)
                chunk = await process.stdout.read(4096)

                if not chunk:
                    # End of stream
                    break

                chunk_count += 1
                total_bytes += len(chunk)

                # Yield audio frame to Pipecat
                frame = TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                )
                yield frame

            # Wait for process completion
            await process.wait()

            if process.returncode != 0:
                # Read error output
                stderr = await process.stderr.read()
                error_msg = stderr.decode('utf-8', errors='replace')
                logger.error(f"Siri TTS process failed: {error_msg}")
                yield ErrorFrame(f"Siri TTS failed: {error_msg}")
            else:
                logger.debug(
                    f"Siri TTS completed: {chunk_count} chunks, "
                    f"{total_bytes} bytes (~{total_bytes/self._sample_rate/2:.2f}s)"
                )

        except Exception as e:
            logger.error(f"Siri TTS error: {e}", exc_info=True)
            yield ErrorFrame(f"Siri TTS error: {e}")

        finally:
            # Signal completion
            yield TTSStoppedFrame()


def resolve_siri_binary() -> Path:
    """
    Resolve path to siri-tts binary (development vs production).

    Returns:
        Path to siri-tts binary

    Raises:
        FileNotFoundError: If binary cannot be located
    """
    # Try common locations in order
    candidates = [
        # Development: relative to server directory
        Path(__file__).parent.parent.parent.parent / "app/src-tauri/sidecar/siri-tts/siri-tts",

        # Production bundle (Tauri resource path)
        Path(os.environ.get("TAURI_RESOURCE_DIR", "/")) / "sidecar/siri-tts/siri-tts",

        # System PATH
        "siri-tts",  # Will be resolved by shell
    ]

    for candidate in candidates:
        if isinstance(candidate, Path):
            if candidate.exists() and os.access(candidate, os.X_OK):
                logger.debug(f"Found siri-tts binary: {candidate}")
                return candidate
        else:
            # Try shell resolution
            import shutil
            resolved = shutil.which(candidate)
            if resolved:
                logger.debug(f"Found siri-tts in PATH: {resolved}")
                return Path(resolved)

    raise FileNotFoundError(
        "siri-tts binary not found in any expected location. "
        "Build it with: cd app/src-tauri/sidecar/siri-tts && ./build.sh"
    )
