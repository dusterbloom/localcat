#!/usr/bin/env python3
"""
Robust Kokoro worker with comprehensive error handling and proper espeakng-loader isolation.
Designed for macOS app bundles and development environments.
"""

import os
import sys
import json
import base64
import traceback
import platform
from pathlib import Path
from typing import Optional, Dict, Any

# Early patch for espeak-ng hardcoded paths
def patch_espeakng_paths():
    """Patch espeak-ng to avoid hardcoded CI paths."""
    # Create a patch for the hardcoded CI path
    hardcoded_ci_path = "/Users/runner/work/espeakng-loader/espeakng-loader/espeak-ng/_dynamic/share/espeak-ng-data"
    
    # Try to find the actual espeak-ng data and create a symlink or override
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.abspath(os.path.join(current_dir, '..', '.venv'))
    
    if platform.system() == "Darwin":
        espeak_data = os.path.join(venv_dir, "lib/python3.12/site-packages/espeakng_loader/espeak-ng-data")
    else:
        espeak_data = None
    
    if espeak_data and os.path.exists(espeak_data):
        # Create the directory structure that espeak-ng expects
        expected_dir = "/Users/runner/work/espeakng-loader/espeakng-loader/espeak-ng/_dynamic/share"
        os.makedirs(expected_dir, exist_ok=True)
        
        # Create a symlink from the expected location to our actual data
        expected_data_path = os.path.join(expected_dir, "espeak-ng-data")
        if os.path.exists(expected_data_path):
            os.remove(expected_data_path)
        os.symlink(espeak_data, expected_data_path)

# Apply the patch immediately
patch_espeakng_paths()

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
        # Find venv relative to script
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

def setup_espeakng_environment(env_info: Dict[str, Any]) -> bool:
    """Setup espeakng environment with proper path resolution."""
    try:
        venv_path = env_info["venv_path"]
        if not venv_path or not venv_path.exists():
            return False
        
        # Calculate espeak-ng paths
        if env_info["platform"] == "Darwin":
            python_version = "python3.12"
            espeak_data = venv_path / f"lib/{python_version}/site-packages/espeakng_loader/espeak-ng-data"
            espeak_lib = venv_path / f"lib/{python_version}/site-packages/espeakng_loader/libespeak-ng.dylib"
        elif env_info["platform"] == "Linux":
            python_version = "python3.12"
            espeak_data = venv_path / f"lib/{python_version}/site-packages/espeakng_loader/espeak-ng-data"
            espeak_lib = venv_path / f"lib/{python_version}/site-packages/espeakng_loader/libespeak-ng.so"
        else:  # Windows
            espeak_data = venv_path / "Lib/site-packages/espeakng_loader/espeak-ng-data"
            espeak_lib = venv_path / "Lib/site-packages/espeakng_loader/espeak-ng.dll"
        
        # Set environment variables
        if espeak_data.exists():
            os.environ["ESPEAK_DATA_PATH"] = str(espeak_data)
        else:
            # Fall back to the package's default
            try:
                import espeakng_loader
                os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()
            except ImportError:
                pass
        
        if espeak_lib.exists():
            os.environ["ESPEAK_NG_LIBRARY"] = str(espeak_lib)
        
        # Add library path for macOS/Linux
        if env_info["platform"] in ["Darwin", "Linux"]:
            lib_dir = str(espeak_lib.parent) if espeak_lib.exists() else ""
            if lib_dir:
                if env_info["platform"] == "Darwin":
                    os.environ["DYLD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
                else:  # Linux
                    os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        
        # CRITICAL: Override the hardcoded CI path that causes the error
        # This is the path that espeak-ng tries to use from CI builds
        hardcoded_ci_path = "/Users/runner/work/espeakng-loader/espeakng-loader/espeak-ng/_dynamic/share/espeak-ng-data"
        if os.environ.get("ESPEAK_DATA_PATH"):
            # Override the hardcoded path with our valid path
            os.environ["ESPEAK_DATA_PATH"] = str(espeak_data) if espeak_data.exists() else os.environ["ESPEAK_DATA_PATH"]
            # Add a mapping to prevent the hardcoded path from being used
            if hasattr(os, 'environb'):
                os.environb[b"ESPEAK_DATA_PATH"] = os.environ["ESPEAK_DATA_PATH"].encode()
        
        return True
    except Exception:
        return False

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

class RobustKokoroWorker:
    """Robust Kokoro worker with comprehensive error handling."""

    def __init__(self):
        self.model = None
        self.voice = None
        self.sample_rate = 24000
        self.env_info = detect_environment()
        self.mlx_available = False
        
        # Setup environment before any imports
        setup_success = setup_espeakng_environment(self.env_info)
        
        # Now try to import MLX
        self.mlx_available = safe_import_mlx()
        
        print(json.dumps({
            "status": "Worker initialized",
            "env": {k: str(v) if hasattr(v, '__str__') else v for k, v in self.env_info.items()},
            "espeakng_setup": setup_success,
            "mlx_available": self.mlx_available
        }), flush=True)

    def validate_environment(self) -> Dict[str, Any]:
        """Validate the environment and return diagnostic info."""
        diagnostics = {
            "platform": platform.system(),
            "python_version": sys.version,
            "venv_path": str(self.env_info["venv_path"]) if self.env_info["venv_path"] else None,
            "script_dir": self.env_info["script_dir"],
            "env_vars": {
                "ESPEAK_DATA_PATH": os.environ.get("ESPEAK_DATA_PATH"),
                "ESPEAK_NG_LIBRARY": os.environ.get("ESPEAK_NG_LIBRARY"),
                "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
                "DYLD_LIBRARY_PATH": os.environ.get("DYLD_LIBRARY_PATH"),
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH")
            },
            "paths_exist": {}
        }
        
        # Check critical paths
        espeak_data = Path(os.environ.get("ESPEAK_DATA_PATH", ""))
        espeak_lib = Path(os.environ.get("ESPEAK_NG_LIBRARY", ""))
        diagnostics["paths_exist"] = {
            "espeak_data": espeak_data.exists(),
            "espeak_lib": espeak_lib.exists(),
            "venv": self.env_info["venv_path"].exists() if self.env_info["venv_path"] else False
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
                    "voice": voice
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
        """Generate audio with robust error handling."""
        if not self.model:
            print(json.dumps({
                "error": "Model not initialized", 
                "type": "not_initialized"
            }), flush=True)
            return

        try:
            np = globals()['np']
            
            audio_data = np.array([])
            for result in self.model.generate(text=text, voice=self.voice, speed=speed):
                audio_data = np.array(result.audio, copy=False)
                break  # Just take first result
            
            if audio_data.size > 0:
                audio_int16 = (audio_data * 32767).astype(np.int16)
                chunk_b64 = base64.b64encode(audio_int16.tobytes()).decode()
                print(json.dumps({"chunk": chunk_b64}), flush=True)
            else:
                print(json.dumps({"warning": "No audio generated"}), flush=True)
            
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
            "diagnostics": self.validate_environment()
        }

def main():
    """Main worker loop with enhanced error handling."""
    print(json.dumps({"status": "Worker starting..."}), flush=True)
    
    try:
        worker = RobustKokoroWorker()
        
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
