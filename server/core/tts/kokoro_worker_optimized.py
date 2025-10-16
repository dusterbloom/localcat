#!/usr/bin/env python3
"""
Optimized Kokoro TTS worker with ultra-low latency streaming.
Based on best practices from Kokoro FastAPI and LiveKit implementations.
"""

import sys
import json
import base64
import traceback
import numpy as np
import os
import time

# Add logging
import logging
logging.basicConfig(level=logging.INFO, format='WORKER: %(message)s')

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class OptimizedKokoroWorker:
    """Ultra-low latency Kokoro TTS worker with token-based chunking."""

    # Token chunk size configuration (optimized for consistent latency)
    TARGET_MIN_TOKENS = int(os.getenv("TTS_MIN_TOKENS", "150"))
    TARGET_MAX_TOKENS = int(os.getenv("TTS_MAX_TOKENS", "200"))
    ABSOLUTE_MAX_TOKENS = int(os.getenv("TTS_ABSOLUTE_MAX_TOKENS", "350"))

    # Audio buffer size for streaming (30-80ms adaptive target latency)
    AUDIO_BUFFER_MS = int(os.getenv("TTS_BUFFER_MS", "40"))  # 40ms buffer default

    def __init__(self):
        self.model = None
        self.voice = None
        self.sample_rate = 24000  # Kokoro default
        
        # Ultra-low latency buffer sizing (512-1024 bytes for immediate first-byte)
        self.buffer_bytes = (self.sample_rate * 2 * self.AUDIO_BUFFER_MS) // 1000
        # Very small chunks for ultra-low latency: 512-1024 byte range
        self.buffer_bytes = max(self.buffer_bytes, 1024)
        self.buffer_bytes = min(self.buffer_bytes, 2048)
        
        # Performance tracking
        self.prewarmed = False
        self.last_generation_time = 0
        self.warmup_count = 0

        logging.info(f"Initialized with token chunks: {self.TARGET_MIN_TOKENS}-{self.TARGET_MAX_TOKENS} (max: {self.ABSOLUTE_MAX_TOKENS})")
        logging.info(f"Audio buffer: {self.AUDIO_BUFFER_MS}ms ({self.buffer_bytes} bytes)")

    def initialize(self, model_name, voice):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            self.model = load_model(model_name)
            self.voice = voice

            # Enhanced prewarming for consistent first-chunk latency
            self._enhanced_prewarm()

            return {"success": True, "config": {
                "min_tokens": self.TARGET_MIN_TOKENS,
                "max_tokens": self.TARGET_MAX_TOKENS,
                "buffer_ms": self.AUDIO_BUFFER_MS,
                "sample_rate": self.sample_rate,
                "prewarmed": self.prewarmed,
                "warmup_count": self.warmup_count
            }}
        except Exception as e:
            return {"error": str(e)}

    def _enhanced_prewarm(self):
        """Minimal prewarming for ultra-low latency - just 2 generations."""
        try:
            start_time = time.time()

            # Warm-up 1: Short test
            list(self.model.generate(text="Hello", voice=self.voice, speed=1.0))
            self.warmup_count += 1

            # Warm-up 2: Medium-length sentence (covers most phoneme patterns)
            list(self.model.generate(
                text="The quick brown fox jumps over the lazy dog",
                voice=self.voice,
                speed=1.0
            ))
            self.warmup_count += 1

            self.prewarmed = True
            warmup_time = (time.time() - start_time) * 1000

            logging.info(f"✅ Prewarming completed: {warmup_time:.1f}ms, {self.warmup_count} generations")

        except Exception as e:
            logging.error(f"❌ Prewarming failed: {e}")
            # Fallback to single warm-up
            try:
                list(self.model.generate(text="test", voice=self.voice, speed=1.0))
                self.prewarmed = True
                self.warmup_count = 1
                logging.info("✅ Fallback prewarming completed")
            except Exception as fallback_e:
                logging.error(f"❌ Fallback prewarming failed: {fallback_e}")

    def generate(self, text, speed=1.0):
        """Generate audio with ultra-low latency streaming."""
        if not self.model:
            print(json.dumps({"error": "Not initialized"}), flush=True)
            return

        try:
            # Track timing for metrics
            start_time = time.time()
            first_chunk_sent = False

            # Audio accumulator for intelligent chunking
            audio_accumulator = []
            accumulated_bytes = 0
            chunk_count = 0
            chunks_sent = []

            # Generate audio with optimal speed (1.0 for lowest latency)
            speed = min(max(speed, 0.8), 1.2)  # Clamp speed to avoid artifacts

            for result in self.model.generate(text=text, voice=self.voice, speed=speed):
                audio_data = np.array(result.audio, copy=False)
                if audio_data.size == 0:
                    continue

                # Convert to 16-bit PCM
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_accumulator.append(audio_int16)
                accumulated_bytes += audio_int16.nbytes

                # Ultra-low latency streaming - send immediately
                # For 25-char chunks from tts_mlx_ultra_low_latency, this ensures
                # first audio bytes are sent within 40-80ms
                target_chunk_size = self.buffer_bytes

                # Stream when we hit buffer threshold OR on first chunk
                if accumulated_bytes >= target_chunk_size or not first_chunk_sent:
                    # Send first chunk ASAP for low time-to-first-byte
                    if not first_chunk_sent:
                        ttfb = (time.time() - start_time) * 1000
                        self.last_generation_time = ttfb
                        logging.info(f"⚡ TTFB: {ttfb:.1f}ms (target: {target_chunk_size}B)")
                        first_chunk_sent = True

                    # Concatenate accumulated audio efficiently
                    if len(audio_accumulator) == 1:
                        combined = audio_accumulator[0]
                    else:
                        combined = np.concatenate(audio_accumulator)

                    # Optimize array copy for streaming
                    chunk_b64 = base64.b64encode(combined.tobytes()).decode()
                    chunk_count += 1
                    
                    # Enhanced metadata for performance monitoring
                    print(json.dumps({
                        "chunk": chunk_b64,
                        "chunk_num": chunk_count,
                        "bytes": combined.nbytes,
                        "duration_ms": int((combined.size / self.sample_rate) * 1000),
                        "ttfb_ms": int(self.last_generation_time) if chunk_count == 1 else None,
                        "target_chunk": int(target_chunk_size)
                    }), flush=True)

                    # Reset accumulator
                    audio_accumulator = []
                    accumulated_bytes = 0

            # Send remaining audio
            if audio_accumulator:
                if len(audio_accumulator) == 1:
                    combined = audio_accumulator[0]
                else:
                    combined = np.concatenate(audio_accumulator)

                chunk_b64 = base64.b64encode(combined.tobytes()).decode()
                chunk_count += 1
                print(json.dumps({
                    "chunk": chunk_b64,
                    "chunk_num": chunk_count,
                    "bytes": combined.nbytes,
                    "duration_ms": int((combined.size / self.sample_rate) * 1000),
                    "final": True
                }), flush=True)

            # Send completion with metrics
            total_time = (time.time() - start_time) * 1000
            print(json.dumps({
                "done": True,
                "chunks": chunk_count,
                "total_ms": int(total_time),
                "ttfb_ms": int(ttfb) if first_chunk_sent else None
            }), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e), "trace": traceback.format_exc()}), flush=True)

    def generate_with_boundaries(self, text, speed=1.0):
        """Generate audio with sentence boundary detection for smoother chunks."""
        # Split text on natural boundaries (sentences, clauses)
        import re

        # Simple sentence splitter (can be enhanced)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                self.generate(sentence.strip(), speed)

                # Send boundary marker between sentences
                if i < len(sentences) - 1:
                    print(json.dumps({"boundary": "sentence"}), flush=True)


def main():
    """Main worker loop - reads commands from stdin, writes responses to stdout."""
    worker = OptimizedKokoroWorker()

    # Prewarm the model for ultra-low latency (40-80ms TTFB)
    prewarm = os.getenv("TTS_PREWARM", "true").lower() in ("true", "1", "yes")
    if prewarm:
        sys.stderr.write("Prewarming Kokoro model for ultra-low latency...\n")
        try:
            # Initialize with default model and voice
            model_name = os.getenv("TTS_MODEL", "mlx-community/Kokoro-82M-bf16")
            voice = os.getenv("TTS_VOICE", "af_heart")
            result = worker.initialize(model_name, voice)
            if result.get("status") == "ready":
                sys.stderr.write(f"✅ Model prewarmed successfully\n")
                # Generate a small test to fully warm up
                worker.generate("Test", 1.0)
                sys.stderr.write("✅ Model fully warmed with test generation\n")
        except Exception as e:
            sys.stderr.write(f"⚠️ Could not prewarm model: {e}\n")

    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            cmd = req.get("cmd")

            if cmd == "init":
                resp = worker.initialize(req["model"], req["voice"])
                print(json.dumps(resp), flush=True)

            elif cmd == "generate":
                speed = req.get("speed", 1.0)
                use_boundaries = req.get("use_boundaries", False)

                if use_boundaries:
                    worker.generate_with_boundaries(req["text"], speed)
                else:
                    worker.generate(req["text"], speed)

            elif cmd == "config":
                # Return current configuration
                print(json.dumps({
                    "min_tokens": worker.TARGET_MIN_TOKENS,
                    "max_tokens": worker.TARGET_MAX_TOKENS,
                    "buffer_ms": worker.AUDIO_BUFFER_MS,
                    "sample_rate": worker.sample_rate
                }), flush=True)

            else:
                print(json.dumps({"error": f"Unknown command: {cmd}"}), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e), "trace": traceback.format_exc()}), flush=True)


if __name__ == "__main__":
    main()
