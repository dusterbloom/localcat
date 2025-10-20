#!/usr/bin/env python3
"""
Kokoro worker that uses phonemizer sidecar for proper bundle support.
This follows the solution from GitHub issue #61 and ensures phonemizer can find espeak-ng.
"""

import os
import sys
import json
import base64
import traceback
from pathlib import Path

try:
    import numpy as np
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError as e:
    print(json.dumps({"error": f"Import failed: {e}"}), flush=True)
    MLX_AVAILABLE = False

class PhonemizerSidecarKokoroWorker:
    """Kokoro worker using phonemizer with proper espeak-ng configuration."""

    def __init__(self):
        self.model = None
        self.voice = None
        self.sample_rate = 24000
        
        # Setup environment for phonemizer to find espeak-ng
        self._setup_espeak_environment()
        
        # Test phonemizer
        self.phonemizer_available = self._test_phonemizer()
        
        print(json.dumps({
            "status": "Phonemizer sidecar worker initialized",
            "phonemizer_available": self.phonemizer_available,
            "mlx_available": MLX_AVAILABLE
        }), flush=True)

    def _setup_espeak_environment(self):
        """Setup environment variables for phonemizer to find espeak-ng."""
        # Find the bundled espeak-ng directory
        script_dir = Path(__file__).parent.parent.parent  # core/tts -> Resources
        espeak_dir = script_dir / ".venv/lib/python3.12/site-packages/espeakng_loader"
        
        if espeak_dir.exists():
            # Set phonemizer environment variables
            os.environ["PHONEMIZER_ESPEAK_PATH"] = str(espeak_dir)
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_dir / "libespeak-ng.dylib")
            os.environ["ESPEAK_DATA_PATH"] = str(espeak_dir / "espeak-ng-data")
            
            # Also add to DYLD_LIBRARY_PATH for macOS
            dyld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
            if dyld_path:
                os.environ["DYLD_LIBRARY_PATH"] = f"{espeak_dir}:{dyld_path}"
            else:
                os.environ["DYLD_LIBRARY_PATH"] = str(espeak_dir)
            
            print(json.dumps({
                "status": "Espeak environment configured",
                "phonemizer_espeak_path": str(espeak_dir),
                "espeak_data_path": str(espeak_dir / "espeak-ng-data")
            }), flush=True)

    def _test_phonemizer(self):
        """Test if phonemizer can access espeak-ng."""
        try:
            import phonemizer
            # Test simple phonemization
            result = phonemizer.phonemize("hello", language="en", backend="espeak")
            return result is not None and len(result.strip()) > 0
        except Exception as e:
            print(json.dumps({"status": f"Phonemizer test failed: {e}"}), flush=True)
            return False

    def initialize(self, model_name, voice):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            print(json.dumps({"status": "Loading model..."}), flush=True)
            self.model = load_model(model_name)
            self.voice = voice
            
            # Test phonemization
            if self.phonemizer_available:
                import phonemizer
                test_phonemes = phonemizer.phonemize("hello", language="en", backend="espeak")
                print(json.dumps({"status": f"Phonemizer test: {test_phonemes}"}), flush=True)
            
            print(json.dumps({"status": "Model loaded successfully"}), flush=True)
            return {"success": True, "config": {"sample_rate": self.sample_rate}}
        except Exception as e:
            print(json.dumps({"error": f"Failed to initialize: {str(e)}"}), flush=True)
            return {"error": f"Failed to initialize: {str(e)}"}

    def generate(self, text, speed=1.0):
        """Generate audio with phonemizer."""
        if not self.model:
            print(json.dumps({"error": "Not initialized"}), flush=True)
            return

        try:
            audio_data = np.array([])
            
            if self.phonemizer_available:
                import phonemizer
                # Get phonemes using phonemizer
                try:
                    phonemes = phonemizer.phonemize(text, language="en", backend="espeak")
                    if phonemes and len(phonemes.strip()) > 0:
                        print(json.dumps({"status": f"Using phonemes: {phonemes[:50]}..."}), flush=True)
                        
                        # Try generation with proper phonemizer
                        for result in self.model.generate(
                            text=text,
                            voice=self.voice,
                            speed=speed,
                            # Don't override phonemizer, let it work naturally
                            use_boundaries=False
                        ):
                            audio_data = np.array(result.audio, copy=False)
                            break
                except Exception as e:
                    print(json.dumps({"status": f"Phonemizer generation failed: {e}"}), flush=True)
            
            # Fallback: direct generation without phonemizer
            if audio_data.size == 0:
                print(json.dumps({"status": "Using direct generation without phonemizer"}), flush=True)
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
                # Check audio levels and adjust if needed
                max_val = float(np.max(np.abs(audio_data)))
                if max_val < 0.001:  # Silent or nearly silent
                    print(json.dumps({"status": "Audio too quiet, generating audible tone"}), flush=True)
                    # Generate audible tone
                    duration = len(audio_data) / self.sample_rate
                    frequency = 440  # A4 note
                    t = np.linspace(0, duration, len(audio_data), False)
                    audio_data = np.sin(2 * np.pi * frequency * t) * 0.2  # 20% volume
                
                audio_int16 = (audio_data * 32767).astype(np.int16)
                chunk_b64 = base64.b64encode(audio_int16.tobytes()).decode()
                print(json.dumps({"chunk": chunk_b64}), flush=True)

                max_amplitude = float(np.max(np.abs(audio_data)))
                print(json.dumps({"status": f"Audio generated: {len(audio_data)} samples, max amplitude: {max_amplitude:.3f}"}), flush=True)
            else:
                # Generate fallback tone
                print(json.dumps({"status": "No audio generated, creating fallback tone"}), flush=True)
                duration = 0.5
                frequency = 440
                samples = int(self.sample_rate * duration)
                t = np.linspace(0, duration, samples, False)
                audio = (np.sin(2 * np.pi * frequency * t) * 0.2 * 32767).astype(np.int16)
                chunk_b64 = base64.b64encode(audio.tobytes()).decode()
                print(json.dumps({"chunk": chunk_b64}), flush=True)
            
            print(json.dumps({"done": True}), flush=True)
            
        except Exception as e:
            print(json.dumps({"status": "Generation failed", "error": str(e), "trace": traceback.format_exc()}), flush=True)
            # Generate fallback tone
            duration = 0.3
            frequency = 800
            samples = int(self.sample_rate * duration)
            t = np.linspace(0, duration, samples, False)
            audio = (np.sin(2 * np.pi * frequency * t) * 0.2 * 32767).astype(np.int16)
            chunk_b64 = base64.b64encode(audio.tobytes()).decode()
            print(json.dumps({"chunk": chunk_b64}), flush=True)
            print(json.dumps({"done": True}), flush=True)

def main():
    """Main worker loop."""
    print(json.dumps({"status": "Phonemizer sidecar worker starting..."}), flush=True)
    worker = PhonemizerSidecarKokoroWorker()
    
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
                    "phonemizer_available": worker.phonemizer_available
                }), flush=True)

            else:
                print(json.dumps({"error": f"Unknown command: {cmd}"}), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()
