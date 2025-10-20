#!/usr/bin/env python3
"""
Test kokoro-onnx with proper threading configuration for macOS
Based on: https://github.com/microsoft/onnxruntime/issues/20354
"""
import os
import soundfile as sf
from pathlib import Path

# CRITICAL: Set threading environment before any imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("🔧 Environment configured:")
print(f"   OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
print(f"   MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")

from kokoro_onnx import Kokoro

# Model paths
SERVER_ROOT = Path(__file__).resolve().parent
model_path = SERVER_ROOT / "models" / "kokoro" / "kokoro-v1.0.onnx"
voices_path = SERVER_ROOT / "models" / "kokoro" / "voices-v1.0.bin"

print(f"\n📁 Model paths:")
print(f"   Model: {model_path}")
print(f"   Exists: {model_path.exists()}")
print(f"   Voices: {voices_path}")
print(f"   Exists: {voices_path.exists()}")

if not model_path.exists() or not voices_path.exists():
    print("❌ Model files not found!")
    exit(1)

print("\n🔧 Initializing Kokoro with threading fix...")
try:
    kokoro = Kokoro(str(model_path), str(voices_path))
    print("✅ Kokoro initialized successfully!")
except Exception as e:
    print(f"❌ Init failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🎤 Generating audio...")
try:
    # Use lang="en-us" as in official examples
    samples, sample_rate = kokoro.create(
        "Hello. This is a test of the Kokoro TTS system!",
        voice="af_sarah",
        speed=1.0,
        lang="en-us"
    )
    print(f"✅ Audio generated: {len(samples)} samples at {sample_rate} Hz")
except Exception as e:
    print(f"❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Save to file
output_path = "/tmp/test_kokoro_threading.wav"
try:
    sf.write(output_path, samples, sample_rate)
    print(f"✅ Saved to {output_path}")
    print(f"\n🔊 Play with: afplay {output_path}")
except Exception as e:
    print(f"❌ Save failed: {e}")
    exit(1)

print("\n🎉 SUCCESS! Kokoro TTS is working!")
