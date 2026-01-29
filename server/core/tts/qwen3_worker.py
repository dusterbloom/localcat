#!/usr/bin/env python3
"""
Qwen3 TTS Worker - Process-isolated subprocess for Qwen3 TTS generation.

Uses qwen-tts package from Alibaba/Qwen for high-quality emotional TTS.
Supports voice cloning, emotional control via instruct parameter, and streaming.

Protocol (JSON via stdin/stdout):
- init: {"cmd": "init", "model": "...", "voice": "...", "model_type": "custom_voice|base|voice_design"}
- generate: {"cmd": "generate", "text": "...", "instruct": "...", "language": "..."}
- clone: {"cmd": "clone", "text": "...", "ref_audio": "...", "ref_text": "...", "language": "..."}
- cleanup: {"cmd": "cleanup"}

Response format:
- chunk: {"chunk": "<base64 audio>", "chunk_num": N, "bytes": M}
- done: {"done": true, "chunks": N, "total_ms": T, "ttfb_ms": T}
- error: {"error": "message"}
"""

import base64
import json
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np


def log(msg: str):
    """Log to stderr (doesn't interfere with JSON protocol on stdout)."""
    print(f"[qwen3-worker] {msg}", file=sys.stderr, flush=True)


class Qwen3TTSWorker:
    """Worker process for Qwen3 TTS generation."""

    def __init__(self):
        self.model = None
        self.model_type = "custom_voice"  # custom_voice, base, voice_design
        self.voice = "Ryan"  # Default English speaker
        self.sample_rate = 24000  # Will be updated from model output
        self.voice_clone_prompt = None  # Cached voice clone prompt for reuse
        self._initialized = False

    def initialize(self, model_name: str, voice: str, model_type: str = "custom_voice") -> dict:
        """Initialize Qwen3 TTS model."""
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            log(f"Initializing Qwen3 TTS: model={model_name}, voice={voice}, type={model_type}")

            # Determine device and dtype
            if torch.cuda.is_available():
                device_map = "cuda:0"
                dtype = torch.bfloat16
                attn_impl = "flash_attention_2"
                log("Using CUDA with Flash Attention 2")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_map = "mps"
                dtype = torch.float16  # MPS prefers float16
                attn_impl = "eager"  # Flash attention not available on MPS
                log("Using MPS (Apple Silicon)")
            else:
                device_map = "cpu"
                dtype = torch.float32
                attn_impl = "eager"
                log("Using CPU")

            # Load model
            start = time.time()
            self.model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=device_map,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
            load_time = (time.time() - start) * 1000

            self.model_type = model_type
            self.voice = voice
            self._initialized = True

            # Warmup with short text
            log("Warming up model...")
            warmup_start = time.time()
            self._generate_audio("Hello", language="English")
            warmup_time = (time.time() - warmup_start) * 1000

            log(f"Qwen3 TTS ready (load={load_time:.0f}ms, warmup={warmup_time:.0f}ms)")

            return {
                "success": True,
                "config": {
                    "model": model_name,
                    "voice": voice,
                    "model_type": model_type,
                    "sample_rate": self.sample_rate,
                    "device": device_map,
                    "load_time_ms": load_time,
                    "warmup_time_ms": warmup_time,
                }
            }

        except ImportError as e:
            log(f"qwen-tts package not installed: {e}")
            return {"success": False, "error": f"Install with: pip install qwen-tts. Error: {e}"}
        except Exception as e:
            log(f"Failed to initialize Qwen3 TTS: {e}")
            return {"success": False, "error": str(e)}

    def _generate_audio(
        self,
        text: str,
        language: str = "English",
        instruct: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Generate audio using CustomVoice model."""
        try:
            if self.model_type == "custom_voice":
                kwargs = {
                    "text": text,
                    "language": language,
                    "speaker": self.voice,
                }
                if instruct:
                    kwargs["instruct"] = instruct

                wavs, sr = self.model.generate_custom_voice(**kwargs)

            elif self.model_type == "voice_design":
                wavs, sr = self.model.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=instruct or "Natural speaking voice",
                )

            else:  # base model - requires voice clone
                if self.voice_clone_prompt:
                    wavs, sr = self.model.generate_voice_clone(
                        text=text,
                        language=language,
                        voice_clone_prompt=self.voice_clone_prompt,
                    )
                else:
                    # Fallback to default voice if no clone prompt
                    log("Warning: Base model without clone prompt, using default generation")
                    wavs, sr = self.model.generate_voice_clone(
                        text=text,
                        language=language,
                        ref_audio=None,  # Will fail, but handled in except
                    )

            self.sample_rate = sr
            return wavs[0] if wavs else None, sr

        except Exception as e:
            log(f"Generation error: {e}")
            return None, self.sample_rate

    def _generate_voice_clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        language: str = "English",
    ) -> Tuple[Optional[np.ndarray], int]:
        """Generate audio with voice cloning."""
        try:
            # Create reusable voice clone prompt if not cached
            if self.voice_clone_prompt is None:
                log(f"Creating voice clone prompt from: {ref_audio[:50]}...")
                self.voice_clone_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                log("Voice clone prompt created and cached")

            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=self.voice_clone_prompt,
            )

            self.sample_rate = sr
            return wavs[0] if wavs else None, sr

        except Exception as e:
            log(f"Voice clone error: {e}")
            return None, self.sample_rate

    def generate(
        self,
        text: str,
        language: str = "English",
        instruct: Optional[str] = None,
    ) -> None:
        """Generate audio and stream chunks via stdout."""
        if not self._initialized:
            print(json.dumps({"error": "Not initialized"}), flush=True)
            return

        start_time = time.time()
        ttfb_ms = None
        chunk_count = 0
        total_bytes = 0

        try:
            audio, sr = self._generate_audio(text, language, instruct)

            if audio is None:
                print(json.dumps({"error": "No audio generated"}), flush=True)
                return

            # Convert to int16 PCM
            if audio.dtype != np.int16:
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio

            audio_bytes = audio_int16.tobytes()

            # Stream in chunks for low latency
            chunk_size = int(sr * 0.1 * 2)  # 100ms chunks, 2 bytes per sample
            offset = 0

            while offset < len(audio_bytes):
                chunk = audio_bytes[offset:offset + chunk_size]
                chunk_count += 1
                total_bytes += len(chunk)

                if ttfb_ms is None:
                    ttfb_ms = (time.time() - start_time) * 1000

                response = {
                    "chunk": base64.b64encode(chunk).decode("ascii"),
                    "chunk_num": chunk_count,
                    "bytes": len(chunk),
                }
                print(json.dumps(response), flush=True)
                offset += chunk_size

            total_ms = (time.time() - start_time) * 1000
            print(json.dumps({
                "done": True,
                "chunks": chunk_count,
                "total_bytes": total_bytes,
                "total_ms": total_ms,
                "ttfb_ms": ttfb_ms or 0,
                "sample_rate": sr,
            }), flush=True)

        except Exception as e:
            log(f"Generate error: {e}")
            print(json.dumps({"error": str(e)}), flush=True)

    def clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        language: str = "English",
    ) -> None:
        """Generate audio with voice cloning and stream chunks via stdout."""
        if not self._initialized:
            print(json.dumps({"error": "Not initialized"}), flush=True)
            return

        start_time = time.time()
        ttfb_ms = None
        chunk_count = 0
        total_bytes = 0

        try:
            audio, sr = self._generate_voice_clone(text, ref_audio, ref_text, language)

            if audio is None:
                print(json.dumps({"error": "No audio generated"}), flush=True)
                return

            # Convert to int16 PCM
            if audio.dtype != np.int16:
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio

            audio_bytes = audio_int16.tobytes()

            # Stream in chunks for low latency
            chunk_size = int(sr * 0.1 * 2)  # 100ms chunks
            offset = 0

            while offset < len(audio_bytes):
                chunk = audio_bytes[offset:offset + chunk_size]
                chunk_count += 1
                total_bytes += len(chunk)

                if ttfb_ms is None:
                    ttfb_ms = (time.time() - start_time) * 1000

                response = {
                    "chunk": base64.b64encode(chunk).decode("ascii"),
                    "chunk_num": chunk_count,
                    "bytes": len(chunk),
                }
                print(json.dumps(response), flush=True)
                offset += chunk_size

            total_ms = (time.time() - start_time) * 1000
            print(json.dumps({
                "done": True,
                "chunks": chunk_count,
                "total_bytes": total_bytes,
                "total_ms": total_ms,
                "ttfb_ms": ttfb_ms or 0,
                "sample_rate": sr,
            }), flush=True)

        except Exception as e:
            log(f"Clone error: {e}")
            print(json.dumps({"error": str(e)}), flush=True)

    def cleanup(self) -> dict:
        """Cleanup resources."""
        try:
            self.model = None
            self.voice_clone_prompt = None
            self._initialized = False

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log("Cleanup complete")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}


def main():
    """Main worker loop - reads JSON commands from stdin, writes responses to stdout."""
    log("Qwen3 TTS worker starting...")
    worker = Qwen3TTSWorker()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
            command = cmd.get("cmd", "")

            if command == "init":
                model = cmd.get("model", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
                voice = cmd.get("voice", "Ryan")
                model_type = cmd.get("model_type", "custom_voice")
                result = worker.initialize(model, voice, model_type)
                print(json.dumps(result), flush=True)

            elif command == "generate":
                text = cmd.get("text", "")
                language = cmd.get("language", "English")
                instruct = cmd.get("instruct")
                worker.generate(text, language, instruct)

            elif command == "clone":
                text = cmd.get("text", "")
                ref_audio = cmd.get("ref_audio", "")
                ref_text = cmd.get("ref_text", "")
                language = cmd.get("language", "English")
                worker.clone(text, ref_audio, ref_text, language)

            elif command == "cleanup":
                result = worker.cleanup()
                print(json.dumps(result), flush=True)
                break

            else:
                print(json.dumps({"error": f"Unknown command: {command}"}), flush=True)

        except json.JSONDecodeError as e:
            log(f"JSON parse error: {e}")
            print(json.dumps({"error": f"Invalid JSON: {e}"}), flush=True)
        except Exception as e:
            log(f"Command error: {e}")
            print(json.dumps({"error": str(e)}), flush=True)

    log("Qwen3 TTS worker exiting")


if __name__ == "__main__":
    main()
