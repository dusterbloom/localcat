#!/usr/bin/env python3
"""
Kokoro worker that uses espeak-ng sidecar for proper bundle support.
This worker communicates with the Rust sidecar for phonemization.
"""

import os
import sys
import json
import base64
import traceback
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError as e:
    print(json.dumps({"error": f"Import failed: {e}"}), flush=True)
    MLX_AVAILABLE = False

class SidecarKokoroWorker:
    """Kokoro worker using espeak-ng sidecar for proper bundle support."""

    def __init__(self):
        self.model = None
        self.voice = None
        self.sample_rate = 24000
        self.sidecar_available = self._check_sidecar()
        
        print(json.dumps({
            "status": "Sidecar worker initialized",
            "sidecar_available": self.sidecar_available,
            "mlx_available": MLX_AVAILABLE
        }), flush=True)

    def _check_sidecar(self) -> bool:
        """Check if espeak-ng sidecar is available."""
        try:
            # Try calling the sidecar with a simple test
            result = subprocess.run(
                [sys.executable, "-c", "import requests; print(requests.post('http://127.0.0.1:7860/api/phonemize', json={'text': 'test'}).text)"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "test" in result.stdout.lower() or result.returncode == 0
        except Exception:
            # If sidecar not available, we'll fall back to direct generation
            return False

    def _phonemize_with_sidecar(self, text: str) -> Optional[str]:
        """Get phonemes using the Rust sidecar."""
        try:
            import requests
            response = requests.post(
                'http://127.0.0.1:7860/api/phonemize',
                json={'text': text},
                timeout=5
            )
            if response.status_code == 200:
                return response.text.strip()
        except Exception:
            pass
        return None

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
        """Generate audio with sidecar phonemization support."""
        if not self.model:
            print(json.dumps({"error": "Not initialized"}), flush=True)
            return

        try:
            # Try sidecar phonemization first
            phonemes = None
            if self.sidecar_available:
                phonemes = self._phonemize_with_sidecar(text)
                if phonemes:
                    print(json.dumps({"status": f"Using sidecar phonemes: {phonemes[:50]}..."}), flush=True)

            audio_data = np.array([])
            
            # Generate with or without phonemes
            if phonemes:
                # Use phonemes if sidecar worked
                for result in self.model.generate(
                    text=text, 
                    voice=self.voice, 
                    speed=speed,
                    phonemes=phonemes,  # Some Kokoro versions support this
                    use_boundaries=False
                ):
                    audio_data = np.array(result.audio, copy=False)
                    break
            else:
                # Direct generation without phonemizer
                for result in self.model.generate(
                    text=text, 
                    voice=self.voice, 
                    speed=speed,
                    phonemize=False,
                    use_boundaries=False
                ):
                    audio_data = np.array(result.audio, copy=False)
                    break
            
            if audio_data.size > 0:
                audio_int16 = (audio_data * 32767).astype(np.int16)
                chunk_b64 = base64.b64encode(audio_int16.tobytes()).decode()
                print(json.dumps({"chunk": chunk_b64}), flush=True)
            else:
                # Fallback tone
                duration = 0.3
                frequency = 440
                samples = int(self.sample_rate * duration)
                t = np.linspace(0, duration, samples, False)
                audio = (np.sin(2 * np.pi * frequency * t) * 0.1 * 32767).astype(np.int16)
                chunk_b64 = base64.b64encode(audio.tobytes()).decode()
                print(json.dumps({"chunk": chunk_b64}), flush=True)
            
            print(json.dumps({"done": True}), flush=True)
            
        except Exception as e:
            # Create fallback tone on any error
            print(json.dumps({"status": "Generation failed, creating fallback tone", "error": str(e)}), flush=True)
            duration = 0.3
            frequency = 800
            samples = int(self.sample_rate * duration)
            t = np.linspace(0, duration, samples, False)
            audio = (np.sin(2 * np.pi * frequency * t) * 0.1 * 32767).astype(np.int16)
            chunk_b64 = base64.b64encode(audio.tobytes()).decode()
            print(json.dumps({"chunk": chunk_b64}), flush=True)
            print(json.dumps({"done": True}), flush=True)

def main():
    """Main worker loop."""
    print(json.dumps({"status": "Sidecar worker starting..."}), flush=True)
    worker = SidecarKokoroWorker()
    
    for line in sys.stdin:
        try:
            if not line.strip():
                continue
                
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
                    "sample_rate": worker.sample_rate,
                    "sidecar_available": worker.sidecar_available
                }), flush=True)

            else:
                print(json.dumps({"error": f"Unknown command: {cmd}"}), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()
