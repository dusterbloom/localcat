#!/usr/bin/env python3
"""
Standalone Kokoro TTS worker process.

This worker runs in complete isolation to avoid Metal threading conflicts.
It communicates via JSON over stdin/stdout.

Usage:
    python kokoro_worker.py

Commands:
    {"cmd": "init", "model": "mlx-community/Kokoro-82M-bf16", "voice": "af_heart"}
    {"cmd": "generate", "text": "Hello world"}
"""

import sys
import json
import base64
import traceback
import numpy as np

# Add logging to worker
import logging
logging.basicConfig(level=logging.INFO, format='WORKER: %(message)s')

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class Worker:
    def __init__(self):
        self.model = None
        self.voice = None

    def initialize(self, model_name, voice):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            self.model = load_model(model_name)
            self.voice = voice
            # Test generation to ensure everything works
            list(self.model.generate(text="test", voice=voice, speed=1.0))
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    def generate(self, text):
        if not self.model:
            print(json.dumps({"error": "Not initialized"}), flush=True)
            return

        try:
            # Buffer multiple small chunks before sending to reduce overhead
            audio_buffer = []
            buffer_size = 4096  # 4KB chunks for balance between smoothness and responsiveness

            for result in self.model.generate(text=text, voice=self.voice, speed=1.0):
                audio_data = np.array(result.audio, copy=False)
                if audio_data.size == 0:
                    continue
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_buffer.append(audio_int16)

                # Check if we have enough data to send
                total_bytes = sum(a.nbytes for a in audio_buffer)
                if total_bytes >= buffer_size:
                    # Concatenate and send buffered audio
                    combined = np.concatenate(audio_buffer)
                    chunk_b64 = base64.b64encode(combined.tobytes()).decode()
                    print(json.dumps({"chunk": chunk_b64}), flush=True)
                    audio_buffer = []

            # Send any remaining buffered audio
            if audio_buffer:
                combined = np.concatenate(audio_buffer)
                chunk_b64 = base64.b64encode(combined.tobytes()).decode()
                print(json.dumps({"chunk": chunk_b64}), flush=True)

            print(json.dumps({"done": True}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


def main():
    """Main worker loop - reads commands from stdin, writes responses to stdout."""
    worker = Worker()
    
    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            if req["cmd"] == "init":
                resp = worker.initialize(req["model"], req["voice"])
                print(json.dumps(resp), flush=True)
            elif req["cmd"] == "generate":
                worker.generate(req["text"])
                continue
            else:
                resp = {"error": "Unknown command"}
                print(json.dumps(resp), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
