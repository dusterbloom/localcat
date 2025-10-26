#!/usr/bin/env python3
"""
Whisper STT Worker - Process-isolated transcription service for background processing.
Runs in separate process to avoid Metal threading conflicts and enable parallel STT.

Communication: JSON over stdin/stdout
Commands:
  - {"cmd": "init", "model": "openai/whisper-small", "language": "en"}
  - {"cmd": "transcribe", "audio": "<base64>"}
  - {"cmd": "config"}  # Get configuration
"""

import os
import sys
import json
import base64
import traceback
import numpy as np
from typing import Optional

# Disable MLX lock in worker process - we own the entire Metal context
os.environ["MLX_DISABLE_LOCK"] = "1"

try:
    import mlx_whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class WhisperWorker:
    """Isolated Whisper STT worker for background processing."""

    def __init__(self):
        self._model = "openai/whisper-small"
        self._language = "en"
        self._sample_rate = 16000  # Whisper expects 16kHz

        print(json.dumps({
            "status": "Whisper worker initialized",
            "whisper_available": WHISPER_AVAILABLE
        }), flush=True)

    def initialize(self, model: str = "openai/whisper-small", language: str = "en"):
        """Initialize Whisper model in isolated process."""
        if not WHISPER_AVAILABLE:
            return {"error": "MLX Whisper not available"}

        try:
            print(json.dumps({"status": f"Loading model: {model}"}), flush=True)

            self._model = model
            self._language = language

            # Test the model
            import numpy as np
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            result = mlx_whisper.transcribe(test_audio, path_or_hf_repo=self._model)

            print(json.dumps({
                "success": True,
                "config": {
                    "model": self._model,
                    "language": self._language,
                    "sample_rate": self._sample_rate
                }
            }), flush=True)

            return {"success": True}

        except Exception as e:
            error_msg = f"Failed to initialize: {str(e)}"
            print(json.dumps({"error": error_msg}), flush=True)
            traceback.print_exc(file=sys.stderr)
            return {"error": error_msg}

    def transcribe(self, audio_b64: str):
        """Transcribe audio chunk (base64 encoded PCM16)."""
        if not WHISPER_AVAILABLE:
            print(json.dumps({"error": "Whisper not available"}), flush=True)
            return

        try:
            # Decode audio
            audio_bytes = base64.b64decode(audio_b64)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Transcribe using MLX Whisper
            result = mlx_whisper.transcribe(
                audio_float32,
                path_or_hf_repo=self._model,
                language=self._language
            )

            if result and "text" in result:
                text = str(result["text"]).strip()
                if text:
                    print(json.dumps({"text": text, "is_final": True}), flush=True)
                else:
                    print(json.dumps({"text": "", "is_final": True}), flush=True)
            else:
                print(json.dumps({"text": "", "is_final": True}), flush=True)

        except Exception as e:
            print(json.dumps({"error": f"Transcription failed: {str(e)}"}), flush=True)
            traceback.print_exc(file=sys.stderr)

    def get_config(self):
        """Return current configuration."""
        print(json.dumps({
            "model": self._model,
            "language": self._language,
            "sample_rate": self._sample_rate,
            "initialized": WHISPER_AVAILABLE
        }), flush=True)


def main():
    """Main worker loop - read commands from stdin, write results to stdout."""
    print(json.dumps({"status": "Whisper worker starting..."}), flush=True)
    worker = WhisperWorker()

    for line in sys.stdin:
        try:
            if not line.strip():
                continue

            req = json.loads(line.strip())
            cmd = req.get("cmd")

            if cmd == "init":
                worker.initialize(
                    model=req.get("model", "openai/whisper-small"),
                    language=req.get("language", "en")
                )

            elif cmd == "transcribe":
                worker.transcribe(req["audio"])

            elif cmd == "config":
                worker.get_config()

            else:
                print(json.dumps({"error": f"Unknown command: {cmd}"}), flush=True)

        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}), flush=True)
        except KeyError as e:
            print(json.dumps({"error": f"Missing parameter: {e}"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": f"Worker error: {str(e)}"}), flush=True)
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()