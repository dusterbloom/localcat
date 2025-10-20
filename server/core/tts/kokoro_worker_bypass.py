#!/usr/bin/env python3
"""
Kokoro worker that completely bypasses espeakng-loader to avoid hardcoded CI paths.
"""

import os
import sys
import json
import base64
import traceback

# Completely disable espeakng-loader by overriding its imports before anything else
class MockEspeakLoader:
    @staticmethod
    def get_data_path():
        # Return our bundled path instead of hardcoded CI path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        venv_dir = os.path.abspath(os.path.join(script_dir, '..', '.venv'))
        return os.path.join(venv_dir, 'lib/python3.12/site-packages/espeakng_loader/espeak-ng-data')
    
    @staticmethod
    def get_library_path():
        # Return our bundled library path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        venv_dir = os.path.abspath(os.path.join(script_dir, '..', '.venv'))
        lib_path = os.path.join(venv_dir, 'lib/python3.12/site-packages/espeakng_loader/libespeak-ng.dylib')
        return lib_path if os.path.exists(lib_path) else None

# Replace espeakng-loader in sys.modules before any imports
sys.modules['espeakng_loader'] = MockEspeakLoader()

# Force environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_dir = os.path.abspath(os.path.join(script_dir, '..', '.venv'))
os.environ['ESPEAK_DATA_PATH'] = os.path.join(venv_dir, 'lib/python3.12/site-packages/espeakng_loader/espeak-ng-data')
os.environ['ESPEAK_NG_LIBRARY'] = os.path.join(venv_dir, 'lib/python3.12/site-packages/espeakng_loader/libespeak-ng.dylib')

try:
    import numpy as np
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError as e:
    print(json.dumps({"error": f"Import failed: {e}"}), flush=True)
    MLX_AVAILABLE = False

class BypassKokoroWorker:
    """Kokoro worker that bypasses espeakng-loader hardcoded paths."""

    def __init__(self):
        self.model = None
        self.voice = None
        self.sample_rate = 24000

    def initialize(self, model_name, voice):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            print(json.dumps({"status": "Loading model..."}), flush=True)
            self.model = load_model(model_name)
            self.voice = voice
            print(json.dumps({"status": "Model loaded successfully"}), flush=True)
            return {"success": True, "config": {"sample_rate": self.sample_rate}}
        except Exception as e:
            print(json.dumps({"error": f"Failed to initialize: {str(e)}"}), flush=True)
            return {"error": f"Failed to initialize: {str(e)}"}

    def generate(self, text, speed=1.0):
        """Generate audio with simple streaming."""
        if not self.model:
            print(json.dumps({"error": "Not initialized"}), flush=True)
            return

        try:
            audio_data = np.array([])
            for result in self.model.generate(text=text, voice=self.voice, speed=speed):
                audio_data = np.array(result.audio, copy=False)
                break  # Just take first result
            
            if audio_data.size > 0:
                audio_int16 = (audio_data * 32767).astype(np.int16)
                chunk_b64 = base64.b64encode(audio_int16.tobytes()).decode()
                print(json.dumps({"chunk": chunk_b64}), flush=True)
            
            print(json.dumps({"done": True}), flush=True)
            
        except Exception as e:
            print(json.dumps({"error": str(e), "trace": traceback.format_exc()}), flush=True)

def main():
    """Main worker loop."""
    print(json.dumps({"status": "Worker started"}), flush=True)
    worker = BypassKokoroWorker()
    
    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            cmd = req.get("cmd")

            if cmd == "init":
                resp = worker.initialize(req["model"], req["voice"])
                print(json.dumps(resp), flush=True)

            elif cmd == "generate":
                speed = req.get("speed", 1.0)
                worker.generate(req["text"], speed)

            elif cmd == "config":
                print(json.dumps({
                    "sample_rate": worker.sample_rate
                }), flush=True)

            else:
                print(json.dumps({"error": f"Unknown command: {cmd}"}), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()
