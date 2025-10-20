#!/usr/bin/env python3
"""
Kokoro ONNX TTS Worker Process
Runs in isolation to prevent library conflicts in Tauri bundles.

Enhanced with robust espeak-ng configuration and error diagnostics.
"""

import json
import sys
import base64
import os
import traceback
from pathlib import Path

# Enhanced startup diagnostics
print("=== Kokoro ONNX Worker Startup ===", file=sys.stderr, flush=True)
print(f"Python: {sys.version}", file=sys.stderr, flush=True)
print(f"Executable: {sys.executable}", file=sys.stderr, flush=True)
print(f"Working dir: {os.getcwd()}", file=sys.stderr, flush=True)
print(f"Script: {__file__}", file=sys.stderr, flush=True)

def setup_espeak_environment():
    """
    Configure espeak-ng environment with priority: parent env → venv → system.
    Returns tuple of (espeak_data_path, espeak_lib_path) or (None, None) if not found.
    """
    espeak_data = os.environ.get('ESPEAK_DATA_PATH')
    espeak_lib = os.environ.get('ESPEAK_NG_LIBRARY')

    # If parent process already set these (Tauri bundle), use them
    if espeak_data and espeak_lib:
        print(f"✅ Using parent ESPEAK_DATA_PATH: {espeak_data}", file=sys.stderr, flush=True)
        print(f"✅ Using parent ESPEAK_NG_LIBRARY: {espeak_lib}", file=sys.stderr, flush=True)
        return espeak_data, espeak_lib

    # Otherwise try to find in venv (development mode)
    print("⚠️  Parent env not set, searching for espeak-ng in venv...", file=sys.stderr, flush=True)

    try:
        # Try espeakng_loader first
        import espeakng_loader
        data_path = Path(espeakng_loader.__file__).parent / "espeak-ng-data"
        lib_path = Path(espeakng_loader.__file__).parent / "libespeak-ng.dylib"

        if data_path.exists() and lib_path.exists():
            espeak_data = str(data_path)
            espeak_lib = str(lib_path)
            os.environ['ESPEAK_DATA_PATH'] = espeak_data
            os.environ['ESPEAK_NG_LIBRARY'] = espeak_lib
            print(f"✅ Found espeak-ng in espeakng_loader: {espeak_data}", file=sys.stderr, flush=True)
            return espeak_data, espeak_lib
    except ImportError:
        print("⚠️  espeakng_loader not available", file=sys.stderr, flush=True)

    # Fallback: try relative to venv
    script_dir = Path(__file__).parent
    current = script_dir
    while current and current != Path('/'):
        venv_candidate = current / '.venv'
        if venv_candidate.exists():
            data_path = venv_candidate / 'lib/python3.12/site-packages/espeakng_loader/espeak-ng-data'
            lib_path = venv_candidate / 'lib/python3.12/site-packages/espeakng_loader/libespeak-ng.dylib'

            if data_path.exists() and lib_path.exists():
                espeak_data = str(data_path)
                espeak_lib = str(lib_path)
                os.environ['ESPEAK_DATA_PATH'] = espeak_data
                os.environ['ESPEAK_NG_LIBRARY'] = espeak_lib
                print(f"✅ Found espeak-ng in venv: {espeak_data}", file=sys.stderr, flush=True)
                return espeak_data, espeak_lib
            break
        current = current.parent

    print("❌ Could not find espeak-ng paths", file=sys.stderr, flush=True)
    return None, None

def main():
    """Worker main loop - reads JSON commands from stdin, writes responses to stdout."""
    # Set unbuffered output for lowest latency
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # Setup espeak-ng environment early
    espeak_data_fallback, espeak_lib_fallback = setup_espeak_environment()

    pipeline = None
    sample_rate = 24000

    print("=== Ready for commands ===", file=sys.stderr, flush=True)

    try:
        for line in sys.stdin:
            try:
                cmd = json.loads(line.strip())
                print(f"📥 Received command: {cmd.get('cmd')}", file=sys.stderr, flush=True)

                if cmd["cmd"] == "init":
                    print("🔧 Initializing Kokoro ONNX...", file=sys.stderr, flush=True)

                    # Get espeak paths with fallback BEFORE importing
                    espeak_data = os.environ.get('ESPEAK_DATA_PATH', espeak_data_fallback)
                    espeak_lib = os.environ.get('ESPEAK_NG_LIBRARY', espeak_lib_fallback)

                    # CRITICAL: Set environment variables BEFORE importing kokoro_onnx
                    # The library has hardcoded paths that need to be overridden
                    if espeak_data:
                        os.environ['ESPEAK_DATA_PATH'] = espeak_data
                        print(f"🔧 Pre-import ESPEAK_DATA_PATH: {espeak_data}", file=sys.stderr, flush=True)
                    if espeak_lib:
                        os.environ['ESPEAK_NG_LIBRARY'] = espeak_lib
                        print(f"🔧 Pre-import ESPEAK_NG_LIBRARY: {espeak_lib}", file=sys.stderr, flush=True)

                    # Import with error handling
                    try:
                        from kokoro_onnx import Kokoro
                        from kokoro_onnx.config import EspeakConfig
                        print("✅ Imported kokoro_onnx successfully", file=sys.stderr, flush=True)
                    except ImportError as e:
                        error_msg = f"Failed to import kokoro_onnx: {e}"
                        print(f"❌ {error_msg}", file=sys.stderr, flush=True)
                        print(json.dumps({"error": error_msg, "type": "ImportError"}), flush=True)
                        continue

                    model_name = cmd.get("model", "kokoro-v1.0.onnx")
                    voice = cmd.get("voice", "af_heart")

                    # Find bundled models (same logic as kokoro_professional.py)
                    script_dir = Path(__file__).parent.parent.parent
                    bundled_dir = script_dir / "models" / "kokoro"
                    model_path = bundled_dir / "kokoro-v1.0.onnx"
                    voices_path = bundled_dir / "voices-v1.0.bin"

                    if not model_path.exists():
                        print(f"⚠️  Bundled models not found at {bundled_dir}, trying cache...", file=sys.stderr, flush=True)
                        # Fallback to cache
                        cache_dir = Path.home() / ".cache" / "kokoro"
                        model_path = cache_dir / "kokoro-v1.0.onnx"
                        voices_path = cache_dir / "voices-v1.0.bin"

                    print(f"📦 Model path: {model_path}", file=sys.stderr, flush=True)
                    print(f"📦 Voices path: {voices_path}", file=sys.stderr, flush=True)
                    print(f"📦 Model exists: {model_path.exists()}", file=sys.stderr, flush=True)
                    print(f"📦 Voices exists: {voices_path.exists()}", file=sys.stderr, flush=True)

                    # Force-disable Espeak in worker: use Kokoro internal phonemizer to avoid dylib issues
                    espeak_config = None
                    print("🛑 Espeak disabled in worker (using internal phonemizer)", file=sys.stderr, flush=True)

                    # Initialize Kokoro with detailed error handling
                    try:
                        print("🚀 Initializing Kokoro pipeline...", file=sys.stderr, flush=True)
                        pipeline = Kokoro(
                            model_path=str(model_path),
                            voices_path=str(voices_path),
                            espeak_config=None
                        )
                        print("✅ Kokoro pipeline initialized successfully", file=sys.stderr, flush=True)
                    except Exception as e:
                        error_msg = f"Failed to initialize Kokoro: {e}"
                        print(f"❌ {error_msg}", file=sys.stderr, flush=True)
                        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                        print(json.dumps({"error": error_msg, "type": type(e).__name__}), flush=True)
                        continue

                    # Send success response
                    response = {
                        "success": True,
                        "config": {
                            "model": model_name,
                            "voice": voice,
                            "sample_rate": sample_rate,
                            "espeak_configured": espeak_config is not None
                        }
                    }
                    print(json.dumps(response), flush=True)
                    print("✅ Init complete, sent success response", file=sys.stderr, flush=True)

                elif cmd["cmd"] == "generate":
                    if not pipeline:
                        error_msg = "Pipeline not initialized"
                        print(f"❌ {error_msg}", file=sys.stderr, flush=True)
                        print(json.dumps({"error": error_msg}), flush=True)
                        continue

                    text = cmd["text"]
                    voice = cmd.get("voice", "af_heart")
                    speed = cmd.get("speed", 1.0)

                    print(f"🎤 Generating audio for: {text[:50]}...", file=sys.stderr, flush=True)

                    # Generate audio with error handling
                    try:
                        import numpy as np
                        print("🔊 Calling pipeline.create()...", file=sys.stderr, flush=True)
                        audio, sr = pipeline.create(text, voice=voice, speed=speed)
                        print(f"✅ Audio generated: {len(audio)} samples @ {sr}Hz", file=sys.stderr, flush=True)

                        # Convert float32 to int16
                        audio_int16 = (audio * 32767).astype(np.int16)

                        # Encode as base64 and send
                        audio_bytes = audio_int16.tobytes()
                        encoded = base64.b64encode(audio_bytes).decode('ascii')

                        print(json.dumps({"chunk": encoded}), flush=True)
                        print(json.dumps({"done": True}), flush=True)
                        print("✅ Audio chunk sent successfully", file=sys.stderr, flush=True)

                    except Exception as e:
                        error_msg = f"Audio generation failed: {e}"
                        print(f"❌ {error_msg}", file=sys.stderr, flush=True)
                        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                        print(json.dumps({"error": error_msg, "type": type(e).__name__}), flush=True)

                elif cmd["cmd"] == "diagnostics":
                    # Enhanced health check
                    espeak_data_env = os.environ.get('ESPEAK_DATA_PATH', '')
                    espeak_lib_env = os.environ.get('ESPEAK_NG_LIBRARY', '')

                    response = {
                        "mlx_available": False,  # ONNX doesn't use MLX
                        "onnx_available": pipeline is not None,
                        "paths_exist": {
                            "espeak_data": Path(espeak_data_env).exists() if espeak_data_env else False,
                            "espeak_lib": Path(espeak_lib_env).exists() if espeak_lib_env else False
                        },
                        "environment": {
                            "espeak_data": espeak_data_env,
                            "espeak_lib": espeak_lib_env,
                            "python": sys.executable,
                            "working_dir": os.getcwd()
                        }
                    }
                    print(json.dumps(response), flush=True)
                    print("✅ Diagnostics sent", file=sys.stderr, flush=True)

                else:
                    error_msg = f"Unknown command: {cmd.get('cmd')}"
                    print(f"❌ {error_msg}", file=sys.stderr, flush=True)
                    print(json.dumps({"error": error_msg}), flush=True)

            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON: {e}"
                print(f"❌ {error_msg}", file=sys.stderr, flush=True)
                print(json.dumps({"error": error_msg, "type": "JSONDecodeError"}), flush=True)
            except Exception as e:
                error_msg = f"Command error: {e}"
                print(f"❌ {error_msg}", file=sys.stderr, flush=True)
                print(f"Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                error_response = {"error": str(e), "type": type(e).__name__}
                print(json.dumps(error_response), flush=True)

    except KeyboardInterrupt:
        print("🛑 Worker interrupted", file=sys.stderr, flush=True)
    except Exception as e:
        error_msg = f"Worker fatal error: {e}"
        print(f"❌ {error_msg}", file=sys.stderr, flush=True)
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        print(json.dumps({"error": error_msg}), file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
