#!/usr/bin/env python3
"""
VISION MODEL TEST
Specifically test the qwen3-vl-4b-instruct-mlx model that's causing 40+ second delays
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, Any
from loguru import logger

async def test_vision_model():
    """Test the problematic vision model"""

    base_url = "http://127.0.0.1:1234/v1"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer not-needed"
    }

    # Test 1: Simple text-only request
    print("🧪 Test 1: Vision model with text-only")
    payload = {
        "model": "qwen3-vl-4b-instruct-mlx",
        "messages": [
            {"role": "user", "content": "Hello, respond briefly."}
        ],
        "max_tokens": 20,
        "stream": False
    }

    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120) as response:
                data = await response.json()
                latency_ms = (time.time() - start_time) * 1000
                response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"   Text-only: {latency_ms:.1f}ms - '{response_text[:50]}...'")
    except Exception as e:
        print(f"   Text-only FAILED: {e}")

    # Test 2: Vision model with image
    print("\n🧪 Test 2: Vision model with image")
    payload_with_image = {
        "model": "qwen3-vl-4b-instruct-mlx",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "What do you see?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="}}
            ]}
        ],
        "max_tokens": 20,
        "stream": False
    }

    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/chat/completions", headers=headers, json=payload_with_image, timeout=120) as response:
                data = await response.json()
                latency_ms = (time.time() - start_time) * 1000
                response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"   With image: {latency_ms:.1f}ms - '{response_text[:50]}...'")
    except Exception as e:
        print(f"   With image FAILED: {e}")

    # Test 3: Streaming with vision model
    print("\n🧪 Test 3: Vision model streaming")
    payload_streaming = {
        "model": "qwen3-vl-4b-instruct-mlx",
        "messages": [
            {"role": "user", "content": "Hello, respond briefly."}
        ],
        "max_tokens": 20,
        "stream": True
    }

    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/chat/completions", headers=headers, json=payload_streaming, timeout=120) as response:
                response_chunks = []
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        try:
                            data = json.loads(line[6:])
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    response_chunks.append(delta["content"])
                        except:
                            continue

                latency_ms = (time.time() - start_time) * 1000
                response_text = "".join(response_chunks)
                print(f"   Streaming: {latency_ms:.1f}ms - '{response_text[:50]}...'")
    except Exception as e:
        print(f"   Streaming FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_vision_model())