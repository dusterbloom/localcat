#!/usr/bin/env python3
"""
Test script for kokoro_worker_optimized.py (MLX-based worker)
Simulates the parent process sending commands to the worker.
"""

import json
import subprocess
import sys
from pathlib import Path

def test_worker():
    """Test the MLX worker subprocess."""

    worker_script = Path(__file__).parent / "kokoro_worker_optimized.py"

    if not worker_script.exists():
        print(f"❌ Worker script not found: {worker_script}")
        return False

    print("🚀 Starting MLX Kokoro worker...")

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
        time.sleep(1.0)

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
            "model": "mlx-community/Kokoro-82M-bf16",
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
            "speed": 1.0
        }
        process.stdin.write(json.dumps(generate_cmd) + "\n")
        process.stdin.flush()

        # Read audio chunks until done
        print("📥 Reading audio chunks...")
        total_chunks = 0
        total_bytes = 0
        done = False

        while not done:
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
                print("❌ Empty response from worker")
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
                total_chunks += 1
                total_bytes += len(chunk_response['chunk'])
                ttfb = chunk_response.get('ttfb_ms')
                if ttfb:
                    print(f"✅ Received chunk {total_chunks} (TTFB: {ttfb}ms, {chunk_response.get('bytes', 0)} bytes)")
                else:
                    print(f"✅ Received chunk {total_chunks} ({chunk_response.get('bytes', 0)} bytes)")

            elif chunk_response.get("done"):
                done = True
                print(f"✅ Generation complete: {total_chunks} chunks, {total_bytes} base64 bytes")

        # Test 3: Config command
        print("\n📤 Test 3: Sending config command...")
        config_cmd = {"cmd": "config"}
        process.stdin.write(json.dumps(config_cmd) + "\n")
        process.stdin.flush()

        ready, _, _ = select.select([process.stdout], [], [], 5.0)
        if ready:
            config_line = process.stdout.readline()
            config_response = json.loads(config_line.strip())
            print(f"📥 Config: {json.dumps(config_response, indent=2)}")

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
