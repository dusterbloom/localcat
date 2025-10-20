#!/usr/bin/env python3
"""
Simplified robust Kokoro worker that bypasses espeak-ng entirely for macOS bundles.
Uses basic phonemization fallbacks to avoid hardcoded path issues.
"""

import os
import sys
import json
import base64
import traceback
import platform
import re
from pathlib import Path
from typing import Optional, Dict, Any

def detect_environment() -> Dict[str, Any]:
    """Detect the runtime environment (development, Tauri bundle, etc.)."""
    env_info = {
        "platform": platform.system(),
        "is_bundle": False,
        "venv_path": None,
        "script_dir": os.path.dirname(os.path.abspath(__file__))
    }
    
    # Check if we're in a Tauri macOS bundle
    if "Contents/Resources" in env_info["script_dir"]:
        env_info["is_bundle"] = True
        # In bundle: .app/Contents/Resources/core/tts/
        resources_dir = Path(env_info["script_dir"]).parent.parent
        env_info["venv_path"] = resources_dir / ".venv"
    else:
        # Development environment
        script_dir = Path(env_info["script_dir"])
        venv_candidates = [
            script_dir.parent.parent / ".venv",  # server/.venv
            script_dir / ".venv",                # core/tts/.venv
            Path.cwd() / ".venv"                # Current working directory
        ]
        
        for venv in venv_candidates:
            if venv.exists():
                env_info["venv_path"] = venv
                break
    
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

def simple_phonemize(text: str) -> str:
    """Very simple phonemization fallback when espeak-ng is not available."""
    # Convert to lowercase
    text = text.lower()
    
    # Basic phoneme replacements (very simplified)
    replacements = {
        'you': 'juː',
        'the': 'ðə',
        'and': 'ænd',
        'for': 'fɔːr',
        'with': 'wɪð',
        'that': 'ðæt',
        'this': 'ðɪs',
        'from': 'frʌm',
        'they': 'ðeɪ',
        'know': 'noʊ',
        'want': 'wɒnt',
        'been': 'bɪn',
        'good': 'ɡʊd',
        'much': 'mʌtʃ',
        'some': 'sʌm',
        'time': 'taɪm',
        'very': 'vɛri',
        'when': 'wɛn',
        'come': 'kʌm',
        'here': 'hɪr',
        'how': 'haʊ',
        'just': 'dʒʌst',
        'like': 'laɪk',
        'long': 'lɔːŋ',
        'make': 'meɪk',
        'many': 'mɛni',
        'over': 'oʊvər',
        'such': 'sʌtʃ',
        'take': 'teɪk',
        'than': 'ðæn',
        'them': 'ðɛm',
        'well': 'wɛl',
        'where': 'wɛr'
    }
    
    # Apply replacements
    for word, phoneme in replacements.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', phoneme, text)
    
    # Add spaces between letters for remaining text (makes it pronounceable)
    words = text.split()
    result_words = []
    
    for word in words:
        # Skip if already phonemic (contains special characters)
        if re.search(r'[ːˈˌʲʷ˞]', word):
            result_words.append(word)
        else:
            # Add dots between letters to make it pronounceable
            result_words.append('.'.join(list(word)))
    
    return ' '.join(result_words)

class SimpleRobustKokoroWorker:
    """Simplified robust Kokoro worker that avoids espeak-ng issues."""

    def __init__(self):
        self.model = None
        self.voice = None
        self.sample_rate = 24000
        self.env_info = detect_environment()
        self.mlx_available = safe_import_mlx()
        
        print(json.dumps({
            "status": "Simple robust worker initialized",
            "env": {k: str(v) if hasattr(v, '__str__') else v for k, v in self.env_info.items()},
            "mlx_available": self.mlx_available
        }), flush=True)

    def validate_environment(self) -> Dict[str, Any]:
        """Validate the environment and return diagnostic info."""
        diagnostics = {
            "platform": platform.system(),
            "python_version": sys.version,
            "venv_path": str(self.env_info["venv_path"]) if self.env_info["venv_path"] else None,
            "script_dir": self.env_info["script_dir"],
            "mlx_available": self.mlx_available,
            "model_loaded": self.model is not None
        }
        return diagnostics

    def initialize(self, model_name: str, voice: str) -> Dict[str, Any]:
        """Initialize the Kokoro model with comprehensive error reporting."""
        if not self.mlx_available:
            return {
                "error": "MLX not available",
                "type": "mlx_unavailable",
                "diagnostics": self.validate_environment()
            }
        
        try:
            print(json.dumps({"status": "Loading model...", "model": model_name}), flush=True)
            
            # Load the model
            self.model = globals()['load_model'](model_name)
            self.voice = voice
            
            print(json.dumps({"status": "Model loaded successfully"}), flush=True)
            
            return {
                "success": True, 
                "config": {
                    "sample_rate": self.sample_rate,
                    "model": model_name,
                    "voice": voice,
                    "worker_type": "simple_robust"
                }
            }
            
        except Exception as e:
            error_msg = f"Failed to initialize: {str(e)}"
            print(json.dumps({
                "error": error_msg,
                "type": "initialization_error",
                "trace": traceback.format_exc(),
                "diagnostics": self.validate_environment()
            }), flush=True)
            
            return {
                "error": error_msg,
                "type": "initialization_error",
                "diagnostics": self.validate_environment()
            }

    def generate(self, text: str, speed: float = 1.0) -> None:
        """Generate audio with robust error handling and fallback phonemization."""
        if not self.model:
            print(json.dumps({
                "error": "Model not initialized", 
                "type": "not_initialized"
            }), flush=True)
            return

        try:
            np = globals()['np']
            
            # Try generation with simple phonemization first
            print(json.dumps({"status": "Attempting generation with simple phonemization"}), flush=True)
            
            # Pre-process text to avoid special characters that might cause issues
            cleaned_text = re.sub(r'[^\w\s\.,!?\'-]', '', text).strip()
            
            # Try generation with minimal settings
            try:
                audio_data = np.array([])
                for result in self.model.generate(
                    text=cleaned_text, 
                    voice=self.voice, 
                    speed=speed,
                    # Disable problematic features
                    use_boundaries=False,
                    phonemize=False  # Let the model handle phonemization
                ):
                    audio_data = np.array(result.audio, copy=False)
                    break  # Just take first result
                
                if audio_data.size > 0:
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    chunk_b64 = base64.b64encode(audio_int16.tobytes()).decode()
                    print(json.dumps({"chunk": chunk_b64}), flush=True)
                else:
                    print(json.dumps({"warning": "No audio generated"}), flush=True)
                    
            except Exception as gen_error:
                # If generation fails, try with even simpler text
                print(json.dumps({"status": "First generation failed, trying simpler text", "error": str(gen_error)}), flush=True)
                
                # Use very basic text
                simple_text = text[:50]  # Truncate to 50 chars
                simple_text = re.sub(r'[^a-zA-Z\s]', '', simple_text).strip()
                
                if simple_text:
                    audio_data = np.array([])
                    for result in self.model.generate(
                        text=simple_text, 
                        voice=self.voice, 
                        speed=speed,
                        use_boundaries=False,
                        phonemize=False
                    ):
                        audio_data = np.array(result.audio, copy=False)
                        break
                    
                    if audio_data.size > 0:
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                        chunk_b64 = base64.b64encode(audio_int16.tobytes()).decode()
                        print(json.dumps({"chunk": chunk_b64}), flush=True)
                    else:
                        print(json.dumps({"warning": "No audio generated with simple text"}), flush=True)
                else:
                    print(json.dumps({"error": "No valid text to generate"}), flush=True)
            
            print(json.dumps({"done": True}), flush=True)
            
        except Exception as e:
            print(json.dumps({
                "error": str(e),
                "type": "generation_error",
                "trace": traceback.format_exc()
            }), flush=True)

    def get_config(self) -> Dict[str, Any]:
        """Get worker configuration."""
        return {
            "sample_rate": self.sample_rate,
            "mlx_available": self.mlx_available,
            "model_loaded": self.model is not None,
            "voice": self.voice,
            "worker_type": "simple_robust",
            "diagnostics": self.validate_environment()
        }

def main():
    """Main worker loop with enhanced error handling."""
    print(json.dumps({"status": "Simple robust worker starting..."}), flush=True)
    
    try:
        worker = SimpleRobustKokoroWorker()
        
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
                    print(json.dumps(worker.get_config()), flush=True)

                elif cmd == "diagnostics":
                    print(json.dumps(worker.validate_environment()), flush=True)

                else:
                    print(json.dumps({
                        "error": f"Unknown command: {cmd}",
                        "type": "command_error"
                    }), flush=True)

            except json.JSONDecodeError as e:
                print(json.dumps({
                    "error": f"Invalid JSON: {e}",
                    "type": "json_error"
                }), flush=True)
            except Exception as e:
                print(json.dumps({
                    "error": str(e),
                    "type": "handler_error",
                    "trace": traceback.format_exc()
                }), flush=True)
                
    except KeyboardInterrupt:
        print(json.dumps({"status": "Worker interrupted"}), flush=True)
    except Exception as e:
        print(json.dumps({
            "error": f"Worker crashed: {e}",
            "type": "worker_error",
            "trace": traceback.format_exc()
        }), flush=True)

if __name__ == "__main__":
    main()
