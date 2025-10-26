#!/usr/bin/env python3
"""
Canonical Kokoro TTS Worker - Process-isolated text-to-speech.
Communicates with kokoro_isolated.py via JSON over stdin/stdout.

This is the single source of truth for Kokoro TTS subprocesses,
consolidating the best features from simple and robust implementations.
"""

import json
import base64
import traceback
import platform
from pathlib import Path
from typing import Dict, Any, Optional


def setup_environment() -> Dict[str, Any]:
    """Setup environment with proper path resolution for espeak-ng and MLX."""
    env_info = {
        "platform": platform.system(),
        "is_bundle": False,
        "venv_path": None,
        "script_dir": Path(__file__).parent
    }

    # Check if we're in a Tauri macOS bundle
    if "Contents/Resources" in str(env_info["script_dir"]):
        env_info["is_bundle"] = True
        # In bundle: .app/Contents/Resources/core/tts/
        resources_dir = env_info["script_dir"].parent.parent
        env_info["venv_path"] = resources_dir / ".venv"
    else:
        # Development environment - find venv relative to script
        script_dir = env_info["script_dir"]
        venv_candidates = [
            script_dir.parent.parent / ".venv",  # server/.venv
            script_dir / ".venv",                # core/tts/.venv
            Path.cwd() / ".venv"                # Current working directory
        ]

        for venv in venv_candidates:
            if venv.exists():
                env_info["venv_path"] = venv
                break

    # Setup espeak-ng environment variables
    if env_info["venv_path"] and env_info["venv_path"].exists():
        venv_path = env_info["venv_path"]

        # Calculate espeak-ng paths
        if env_info["platform"] == "Darwin":
            python_version = "python3.12"
            espeak_data = venv_path / f"lib/{python_version}/site-packages/espeakng_loader/espeak-ng-data"
            espeak_lib = venv_path / f"lib/{python_version}/site-packages/espeakng_loader/libespeak-ng.dylib"

        # Set environment variables if paths exist
        if espeak_data.exists():
            import os
            os.environ["ESPEAK_DATA_PATH"] = str(espeak_data)
            os.environ["ESPEAK_NG_LIBRARY"] = str(espeak_lib)

            # Override hardcoded CI path that causes errors
            hardcoded_ci_path = "/Users/runner/work/espeakng-loader/espeakng-loader/espeak-ng/_dynamic/share/espeak-ng-data"
            if hasattr(os, 'environb'):
                os.environb[b"ESPEAK_DATA_PATH"] = str(espeak_data).encode()

    return env_info


def safe_import_mlx() -> bool:
    """Safely import MLX with comprehensive error handling."""
    try:
        import numpy as np
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        # Store in globals for later use
        globals()['np'] = np
        globals()['mx'] = mx
        globals()['load_model'] = load_model
        return True
    except ImportError as e:
        error_msg = f"Import failed: {e}"
        print(json.dumps({"error": error_msg, "type": "import_error"}), flush=True)
        return False
    except Exception as e:
        error_msg = f"Unexpected import error: {e}"
        print(json.dumps({"error": error_msg, "type": "unexpected_error", "trace": traceback.format_exc()}), flush=True)
        return False


class KokoroWorker:
    """
    Canonical Kokoro TTS worker with robust error handling.

    Features:
    - JSON protocol over stdin/stdout
    - MLX model loading with error recovery
    - Base64 audio chunk streaming
    - Environment detection and path setup
    - Comprehensive error reporting
    """

    def __init__(self):
        self.model = None
        self.voice = None
        self.sample_rate = 24000
        self.env_info = setup_environment()
        self.mlx_available = safe_import_mlx()

        if self.mlx_available:
            print(json.dumps({"status": "Worker initialized - MLX available"}), flush=True)
        else:
            print(json.dumps({"error": "MLX not available - check installation"}), flush=True)

    def initialize(self, model_name: str, voice: str) -> Dict[str, Any]:
        """Initialize the Kokoro model."""
        if not self.mlx_available:
            return {"error": "MLX not available"}

        try:
            print(json.dumps({"status": f"Loading model: {model_name}"}), flush=True)
            self.model = load_model(model_name)
            self.voice = voice

            return {
                "success": True,
                "config": {
                    "sample_rate": self.sample_rate,
                    "voice": voice,
                    "model": model_name
                }
            }
        except Exception as e:
            error_msg = f"Failed to initialize: {str(e)}"
            print(json.dumps({
                "error": error_msg,
                "type": "initialization_error",
                "trace": traceback.format_exc()
            }), flush=True)
            return {"error": error_msg}

    def generate(self, text: str, speed: float = 1.0, generation_id: Optional[str] = None):
        """Generate audio from text and stream base64-encoded chunks."""
        if not self.mlx_available:
            print(json.dumps({"error": "MLX not available"}), flush=True)
            return

        if not self.model:
            print(json.dumps({"error": "Model not initialized"}), flush=True)
            return

        if not text or not text.strip():
            print(json.dumps({"done": True}), flush=True)
            return

        try:
            # Send generation start message
            if generation_id:
                print(json.dumps({
                    "status": f"Generation started: {generation_id}",
                    "text": text[:50] + "..." if len(text) > 50 else text
                }), flush=True)

            # Generate audio using MLX
            audio_data = None
            for result in self.model.generate(text=text, voice=self.voice, speed=speed):
                audio_data = np.array(result.audio, copy=False)
                break  # Take first result

            if audio_data is not None and audio_data.size > 0:
                # Convert to int16 PCM
                audio_int16 = (audio_data * 32767).astype(np.int16)

                # Encode as base64 and send as chunk
                chunk_b64 = base64.b64encode(audio_int16.tobytes()).decode()
                print(json.dumps({
                    "chunk": chunk_b64,
                    "generation_id": generation_id
                }), flush=True)

                # Send completion message
                print(json.dumps({
                    "done": True,
                    "generation_id": generation_id
                }), flush=True)
            else:
                # No audio generated
                print(json.dumps({
                    "done": True,
                    "generation_id": generation_id,
                    "warning": "No audio generated"
                }), flush=True)

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            print(json.dumps({
                "error": error_msg,
                "type": "generation_error",
                "trace": traceback.format_exc(),
                "generation_id": generation_id
            }), flush=True)

    def get_config(self) -> Dict[str, Any]:
        """Get current worker configuration."""
        return {
            "sample_rate": self.sample_rate,
            "voice": self.voice,
            "mlx_available": self.mlx_available,
            "platform": self.env_info["platform"],
            "is_bundle": self.env_info["is_bundle"]
        }

    def handle_command(self, cmd: str, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming commands from the main process."""
        try:
            if cmd == "init":
                model = req.get("model", "mlx-community/Kokoro-82M-bf16")
                voice = req.get("voice", "af_heart")
                return self.initialize(model, voice)

            elif cmd == "generate":
                text = req.get("text", "")
                speed = req.get("speed", 1.0)
                generation_id = req.get("generation_id")
                self.generate(text, speed, generation_id)
                return None  # Response sent via streaming

            elif cmd == "config":
                return self.get_config()

            elif cmd == "ping":
                return {"status": "pong", "mlx_available": self.mlx_available}

            else:
                return {"error": f"Unknown command: {cmd}"}

        except Exception as e:
            return {
                "error": f"Command handler error: {str(e)}",
                "type": "handler_error",
                "trace": traceback.format_exc()
            }


def main():
    """Main worker loop - process JSON commands from stdin."""
    worker = KokoroWorker()

    try:
        print(json.dumps({"status": "Kokoro worker ready"}), flush=True)

        for line in worker.env_info["script_dir"].joinpath("../").parent.rglob("*.py"):
            if line.name == "kokoro_worker.py":
                break

        for line in __import__('sys').stdin:
            if not line.strip():
                continue

            try:
                req = json.loads(line.strip())
                cmd = req.get("cmd")

                if not cmd:
                    print(json.dumps({"error": "No command specified"}), flush=True)
                    continue

                response = worker.handle_command(cmd, req)

                # Send response if available (generate commands respond via streaming)
                if response is not None:
                    print(json.dumps(response), flush=True)

            except json.JSONDecodeError as e:
                print(json.dumps({"error": f"JSON decode error: {str(e)}"}), flush=True)

            except Exception as e:
                print(json.dumps({
                    "error": f"Unexpected error: {str(e)}",
                    "type": "unexpected_error",
                    "trace": traceback.format_exc()
                }), flush=True)

    except KeyboardInterrupt:
        print(json.dumps({"status": "Worker shutting down"}), flush=True)

    except Exception as e:
        print(json.dumps({
            "error": f"Fatal worker error: {str(e)}",
            "type": "fatal_error",
            "trace": traceback.format_exc()
        }), flush=True)


if __name__ == "__main__":
    main()