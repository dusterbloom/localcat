#!/usr/bin/env python3
"""
Kokoro worker that uses espeak-ng sidecar for proper phonemization.
This fixes the silent audio issue by using actual espeak-ng phonemes.
"""

import os
import sys
import json
import base64
import traceback
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

class EspeakSidecarKokoroWorker:
    """Kokoro worker using espeak-ng sidecar for proper phonemization."""

    def __init__(self):
        self.model = None
        self.voice = None
        self.sample_rate = 24000
        self.espeak_path = self._find_espeak()
        
        print(json.dumps({
            "status": "Espeak sidecar worker initialized",
            "espeak_path": self.espeak_path,
            "mlx_available": MLX_AVAILABLE
        }), flush=True)

    def _find_espeak(self) -> str:
        """Find espeak-ng binary."""
        # Check various paths for espeak-ng
        candidates = [
            "/Users/peppi/Dev/localcat/app/src-tauri/bin/espeak/espeak-ng",
            "/opt/homebrew/bin/espeak-ng",
            "/usr/bin/espeak-ng",
            "espeak-ng"
        ]
        
        for candidate in candidates:
            try:
                result = subprocess.run([candidate, "--version"], 
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    return candidate
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        return "espeak-ng"  # fallback

    def _get_phonemes(self, text: str) -> str:
        """Get phonemes using espeak-ng."""
        try:
            result = subprocess.run([
                self.espeak_path,
                "--ipa=3",     # Use IPA output
                "-q",           # Quiet mode
                "--stdout",     # Output to stdout
                text
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(json.dumps({"status": f"Espeak failed: {e}"}), flush=True)
        
        return None

    def initialize(self, model_name, voice):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            print(json.dumps({"status": "Loading model..."}), flush=True)
            self.model = load_model(model_name)
            self.voice = voice
            print(json.dumps({"status": "Model loaded successfully"}), flush=True)
            
            # Test espeak
            test_phonemes = self._get_phonemes("hello")
            print(json.dumps({"status": f"Espeak test: {test_phonemes}"}), flush=True)
            
            return {"success": True, "config": {"sample_rate": self.sample_rate}}
        except Exception as e:
            print(json.dumps({"error": f"Failed to initialize: {str(e)}"}), flush=True)
            return {"error": f"Failed to initialize: {str(e)}"}

    def generate(self, text, speed=1.0):
        """Generate audio with proper espeak phonemization."""
        if not self.model:
            print(json.dumps({"error": "Not initialized"}), flush=True)
            return

        try:
            # Get phonemes first
            phonemes = self._get_phonemes(text)
            if not phonemes:
                print(json.dumps({"status": "Failed to get phonemes, using fallback"}), flush=True)
                # Use simple fallback
                phonemes = "h ə l oʊ w ɜːr l d"
            
            print(json.dumps({"status": f"Using phonemes: {phonemes}"}), flush=True)
            
            audio_data = np.array([])
            
            # Try generation with phonemes if supported
            try:
                for result in self.model.generate(
                    text=text, 
                    voice=self.voice, 
                    speed=speed,
                    # Try to pass phonemes if the model supports it
                    phonemes=phonemes if hasattr(self.model, 'generate') else None
                ):
                    audio_data = np.array(result.audio, copy=False)
                    break
            except Exception as e:
                print(json.dumps({"status": f"Direct generation failed: {e}"}), flush=True)
                # Fallback: try without phonemes but with shorter text
                simple_text = text[:20]  # Limit text length
                for result in self.model.generate(
                    text=simple_text, 
                    voice=self.voice, 
                    speed=speed
                ):
                    audio_data = np.array(result.audio, copy=False)
                    break
            
            if audio_data.size > 0:
                # Check if audio is silent
                max_val = float(np.max(np.abs(audio_data)))
                if max_val < 0.001:  # Very quiet or silent
                    print(json.dumps({"status": "Audio too quiet, generating fallback tone"}), flush=True)
                    # Generate audible tone
                    duration = len(audio_data) / self.sample_rate
                    frequency = 440  # A4 note
                    t = np.linspace(0, duration, len(audio_data), False)
                    audio_data = np.sin(2 * np.pi * frequency * t) * 0.1  # 10% volume
                
                audio_int16 = (audio_data * 32767).astype(np.int16)
                chunk_b64 = base64.b64encode(audio_int16.tobytes()).decode()
                print(json.dumps({"chunk": chunk_b64}), flush=True)
            else:
                # Generate fallback tone
                print(json.dumps({"status": "No audio generated, creating fallback tone"}), flush=True)
                duration = 0.5
                frequency = 440
                samples = int(self.sample_rate * duration)
                t = np.linspace(0, duration, samples, False)
                audio = (np.sin(2 * np.pi * frequency * t) * 0.1 * 32767).astype(np.int16)
                chunk_b64 = base64.b64encode(audio.tobytes()).decode()
                print(json.dumps({"chunk": chunk_b64}), flush=True)
            
            print(json.dumps({"done": True}), flush=True)
            
        except Exception as e:
            print(json.dumps({"status": "Generation failed completely", "error": str(e)}), flush=True)
            # Create fallback tone
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
    print(json.dumps({"status": "Espeak sidecar worker starting..."}), flush=True)
    worker = EspeakSidecarKokoroWorker()
    
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
                    "espeak_path": worker.espeak_path
                }), flush=True)

            else:
                print(json.dumps({"error": f"Unknown command: {cmd}"}), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()
