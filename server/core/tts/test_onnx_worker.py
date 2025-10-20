#!/usr/bin/env python3
"""
Test script for kokoro_onnx_worker.py
Simulates the parent process sending commands to the worker.
"""

import json
import subprocess
import sys
from pathlib import Path

def test_worker():
    """Test the ONNX worker subprocess."""

    worker_script = Path(__file__).parent / "kokoro_onnx_worker.py"

    if not worker_script.exists():
        print(f"❌ Worker script not found: {worker_script}")
        return False

    print("🚀 Starting ONNX worker...")

    # Start worker process
    process = subprocess.Popen(
        [sys.executable, str(worker_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )

    try:
        # Wait a moment for startup
        import time
        time.sleep(0.5)

        # Check if process started
        if process.poll() is not None:
            print(f"❌ Worker exited immediately with code: {process.poll()}")
            stderr = process.stderr.read()
            print(f"Stderr:\n{stderr}")
            return False

        print("✅ Worker started successfully")

        # Test 1: Init command
        print("\n📤 Test 1: Sending init command...")
        init_cmd = {
            "cmd": "init",
            "model": "kokoro-v1.0.onnx",
            "voice": "af_heart"
        }
        process.stdin.write(json.dumps(init_cmd) + "\n")
        process.stdin.flush()

        # Read response (with timeout)
        import select
        ready, _, _ = select.select([process.stdout], [], [], 30.0)

        if not ready:
            print("❌ No response from worker (timeout)")
            # Read stderr
            ready_err, _, _ = select.select([process.stderr], [], [], 0.1)
            if ready_err:
                stderr = process.stderr.read()
                print(f"Stderr:\n{stderr}")
            return False

        response_line = process.stdout.readline()
        if not response_line:
            print("❌ Empty response from worker")
            return False

        response = json.loads(response_line.strip())
        print(f"📥 Init response: {response}")

        if not response.get("success"):
            print("❌ Init failed")
            # Read stderr for diagnostics
            ready_err, _, _ = select.select([process.stderr], [], [], 0.1)
            if ready_err:
                stderr = process.stderr.read(1024)
                print(f"Stderr:\n{stderr}")
            return False

        print("✅ Init successful")

        # Test 2: Generate command
        print("\n📤 Test 2: Sending generate command...")
        generate_cmd = {
            "cmd": "generate",
            "text": "Hello, this is a test.",
            "voice": "af_heart",
            "speed": 1.0
        }
        process.stdin.write(json.dumps(generate_cmd) + "\n")
        process.stdin.flush()

        # Read audio chunk response
        ready, _, _ = select.select([process.stdout], [], [], 30.0)

        if not ready:
            print("❌ No audio response from worker (timeout)")
            # Read stderr
            ready_err, _, _ = select.select([process.stderr], [], [], 0.1)
            if ready_err:
                stderr = process.stderr.read()
                print(f"Stderr:\n{stderr}")
            return False

        chunk_line = process.stdout.readline()
        if not chunk_line:
            print("❌ Empty audio response from worker")
            return False

        chunk_response = json.loads(chunk_line.strip())

        if "error" in chunk_response:
            print(f"❌ Generation error: {chunk_response}")
            # Read stderr
            ready_err, _, _ = select.select([process.stderr], [], [], 0.1)
            if ready_err:
                stderr = process.stderr.read()
                print(f"Stderr:\n{stderr}")
            return False

        if "chunk" in chunk_response:
            print(f"✅ Received audio chunk ({len(chunk_response['chunk'])} bytes)")

            # Read done message
            done_line = process.stdout.readline()
            done_response = json.loads(done_line.strip())

            if done_response.get("done"):
                print("✅ Generation complete")
            else:
                print(f"⚠️  Unexpected response: {done_response}")
        else:
            print(f"❌ Unexpected response: {chunk_response}")
            return False

        # Test 3: Diagnostics command
        print("\n📤 Test 3: Sending diagnostics command...")
        diag_cmd = {"cmd": "diagnostics"}
        process.stdin.write(json.dumps(diag_cmd) + "\n")
        process.stdin.flush()

        ready, _, _ = select.select([process.stdout], [], [], 5.0)
        if ready:
            diag_line = process.stdout.readline()
            diag_response = json.loads(diag_line.strip())
            print(f"📥 Diagnostics: {json.dumps(diag_response, indent=2)}")

        print("\n🎉 All tests passed!")

        # Read any stderr output
        print("\n📋 Worker stderr output:")
        ready_err, _, _ = select.select([process.stderr], [], [], 0.1)
        if ready_err:
            stderr = process.stderr.read()
            print(stderr)

        return True

    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

if __name__ == "__main__":
    success = test_worker()
    sys.exit(0 if success else 1)
