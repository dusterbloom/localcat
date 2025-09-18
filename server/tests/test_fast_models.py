#!/usr/bin/env python3
"""
Test ultra-fast small models for <200ms inference
"""
import time
import requests
import json

def test_model_speed(model_id: str, prompt: str, max_tokens: int = 10):
    """Test inference speed for a specific model"""

    # Warmup
    for _ in range(2):
        requests.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "temperature": 0.0
            }
        )

    # Test queries
    test_cases = [
        ("Classify: Hello there!", "Intent classification"),
        ("Extract entities from: Sarah works at Google", "Entity extraction"),
        ("Answer: What color is the sky?", "Simple QA"),
        ("Sentiment: I love this product!", "Sentiment analysis"),
        ("Complete: The capital of France is", "Knowledge completion")
    ]

    results = []
    for query, task_type in test_cases:
        start = time.time()

        response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": query}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False
            }
        )

        elapsed = (time.time() - start) * 1000

        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            results.append({
                "task": task_type,
                "time_ms": elapsed,
                "output": content[:50]
            })

    return results

def main():
    print("🚀 Testing Ultra-Fast Small Language Models\n")
    print("Target: <200ms inference latency\n")
    print("="*60)

    models = [
        ("gemma-3-270m-it-mlx-8bit", 270, 10),  # 270M params
        ("llama-3.2-1b-instruct-4bit", 1000, 10),  # 1B params (baseline)
    ]

    for model_id, size_m, max_tokens in models:
        print(f"\n📊 Model: {model_id}")
        print(f"   Size: {size_m}M parameters")
        print(f"   Max tokens: {max_tokens}")
        print("-"*40)

        try:
            results = test_model_speed(model_id, "test", max_tokens)

            times = [r['time_ms'] for r in results]
            avg_time = sum(times) / len(times)

            print(f"✅ Average latency: {avg_time:.1f}ms")

            if avg_time < 200:
                print(f"🎯 MEETS TARGET! ({avg_time:.1f}ms < 200ms)")
            else:
                print(f"❌ Too slow ({avg_time:.1f}ms > 200ms)")

            print("\nTask breakdown:")
            for r in results:
                status = "✓" if r['time_ms'] < 200 else "✗"
                print(f"  {status} {r['task']:20} {r['time_ms']:.1f}ms")
                print(f"     → {r['output']}")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60)
    print("\n💡 Recommendations for <200ms inference:\n")

    print("1. **Gemma-3-270M** (if quality is acceptable)")
    print("   - 3-4x smaller than Llama 1B")
    print("   - Should achieve 50-150ms latency")
    print("   - Good for simple classification/extraction")
    print()
    print("2. **Consider even smaller models:**")
    print("   - SmolLM-135M (135M params)")
    print("   - TinyLlama-110M (110M params)")
    print("   - Phi-1.5 (130M params)")
    print("   - MobileLLM-125M (125M params)")
    print()
    print("3. **Optimization strategies:**")
    print("   - Use INT4 or INT8 quantization")
    print("   - Reduce max_tokens to minimum (5-20)")
    print("   - Use temperature=0 for deterministic output")
    print("   - Implement response caching")
    print("   - Use specialized models for specific tasks")
    print()
    print("4. **Alternative: Task-specific models**")
    print("   - BERT/DistilBERT for classification (~10-30ms)")
    print("   - Sentence transformers for embeddings (~5-20ms)")
    print("   - Flair for NER (~20-50ms)")

if __name__ == "__main__":
    main()