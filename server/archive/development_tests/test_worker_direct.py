#!/usr/bin/env python3
"""
Direct test of the Kokoro worker to debug sentence processing issues.
"""

import subprocess
import sys
import json
import time
import os

def test_worker_direct():
    """Test the worker process directly."""

    # Set environment variables for worker configuration
    env = os.environ.copy()
    env["KOKORO_MIN_TOKENS"] = "175"
    env["KOKORO_MAX_TOKENS"] = "250"
    env["KOKORO_BUFFER_MS"] = "80"

    # Start worker process
    worker = subprocess.Popen(
        [sys.executable, "kokoro_worker_optimized.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,  # Unbuffered
        env=env
    )

    def send_command(cmd):
        """Send command to worker and read response."""
        print(f"📤 Sending: {cmd}")
        worker.stdin.write(json.dumps(cmd) + "\n")
        worker.stdin.flush()

        # Read all responses until done
        responses = []
        while True:
            try:
                line = worker.stdout.readline()
                if not line:
                    break

                response = json.loads(line.strip())
                responses.append(response)
                print(f"📥 Received: {response}")

                # Stop if we get a done signal or error
                if response.get("done") or response.get("error"):
                    break

            except Exception as e:
                print(f"❌ Error reading response: {e}")
                break

        return responses

    try:
        print("🚀 Testing Kokoro Worker Direct Interface")
        print("=" * 50)

        # Initialize worker
        print("\n1. Initializing worker...")
        init_responses = send_command({
            "cmd": "init",
            "model": "mlx-community/Kokoro-82M-bf16",
            "voice": "af_heart"
        })

        if not init_responses or not init_responses[0].get("success"):
            print("❌ Worker initialization failed")
            return

        print("✅ Worker initialized successfully")

        # Test cases
        test_cases = [
            {
                "name": "Single sentence",
                "text": "Hello there!",
                "use_boundaries": False
            },
            {
                "name": "Two sentences (no boundaries)",
                "text": "Hello there! How are you doing?",
                "use_boundaries": False
            },
            {
                "name": "Two sentences (with boundaries)",
                "text": "Hello there! How are you doing?",
                "use_boundaries": True
            },
            {
                "name": "Three sentences (with boundaries)",
                "text": "That's a great question! I can help you with that. Let me know what you need.",
                "use_boundaries": True
            }
        ]

        for i, test_case in enumerate(test_cases):
            print(f"\n{i+2}. Testing: {test_case['name']}")
            print(f"   Text: {test_case['text']}")
            print(f"   Use boundaries: {test_case['use_boundaries']}")

            start_time = time.time()
            responses = send_command({
                "cmd": "generate",
                "text": test_case["text"],
                "speed": 1.0,
                "use_boundaries": test_case["use_boundaries"],
                "timeout_seconds": 15
            })

            duration = time.time() - start_time

            # Analyze responses
            chunks = [r for r in responses if "chunk" in r]
            boundaries = [r for r in responses if r.get("boundary")]
            errors = [r for r in responses if r.get("error")]
            done = [r for r in responses if r.get("done")]

            print(f"   📊 Results: {len(chunks)} chunks, {len(boundaries)} boundaries, {len(errors)} errors")
            print(f"   ⏱️  Duration: {duration:.2f}s")

            if errors:
                print(f"   ❌ Errors: {errors}")

            if done:
                print(f"   ✅ Completion: {done[0]}")

            print("   ---")

    finally:
        # Clean up
        try:
            worker.terminate()
            worker.wait(timeout=2)
        except:
            worker.kill()

if __name__ == "__main__":
    test_worker_direct()