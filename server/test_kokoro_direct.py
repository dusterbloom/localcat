#!/usr/bin/env python3
"""
Test kokoro-onnx by directly loading ONNX model, bypassing phonemizer
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import soundfile as sf
from pathlib import Path
import onnxruntime as ort

print("🔧 Testing ONNX Runtime directly...")

# Model paths
SERVER_ROOT = Path(__file__).resolve().parent
model_path = SERVER_ROOT / "models" / "kokoro" / "kokoro-v1.0.onnx"

print(f"Model: {model_path}")
print(f"Exists: {model_path.exists()}")

if not model_path.exists():
    print("❌ Model not found!")
    exit(1)

print("\n🔧 Creating ONNX session...")
try:
    # Configure session with single thread
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    print("✅ ONNX session created!")
    print(f"   Inputs: {[i.name for i in session.get_inputs()]}")
    print(f"   Outputs: {[o.name for i in session.get_outputs()]}")
    print("\n🎉 ONNX Runtime works! The hang is in kokoro-onnx wrapper, not ONNX itself")

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
