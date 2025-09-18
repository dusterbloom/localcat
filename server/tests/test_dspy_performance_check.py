#!/usr/bin/env python3
"""
Minimal test to check DSPy + Osaurs performance variance
"""
import dspy
import time
import requests

def test_performance():
    # Check Osaurs
    try:
        response = requests.get("http://127.0.0.1:8000/v1/models", timeout=2)
        models = response.json()
        print(f"✓ Osaurs running with models: {[m['id'] for m in models['data']]}")
    except Exception as e:
        print(f"❌ Osaurs not running: {e}")
        return

    # Test each model
    for model_id in ["llama-3.2-3b-instruct-4bit", "llama-3.2-1b-instruct-4bit"]:
        if not any(m['id'] == model_id for m in models['data']):
            print(f"\n⚠️  Model {model_id} not available")
            continue

        print(f"\n📊 Testing {model_id}:")

        # Configure DSPy
        lm = dspy.LM(
            model=f"openai/{model_id}",
            api_base="http://127.0.0.1:8000/v1",
            api_key="dummy",
            max_tokens=32,  # Reduced for speed
            temperature=0.0,  # Deterministic
        )
        dspy.settings.configure(lm=lm)

        # Simple signature
        class SimpleSignature(dspy.Signature):
            """Simple classification"""
            text: str = dspy.InputField()
            category: str = dspy.OutputField()

        # Test with Predict (fast)
        predictor = dspy.Predict(SimpleSignature)

        # Warmup
        print("  Warming up...")
        predictor(text="warmup")

        # Test queries
        queries = ["Hello!", "What's the weather?", "I work at Google"]
        times = []

        for query in queries:
            start = time.time()
            result = predictor(text=query)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            print(f"  '{query}' -> {elapsed:.1f}ms")

        avg = sum(times) / len(times)
        print(f"  Average: {avg:.1f}ms")

        # Now test with ChainOfThought (slower)
        print("\n  Testing ChainOfThought:")
        cot_predictor = dspy.ChainOfThought(SimpleSignature)

        # Warmup
        cot_predictor(text="warmup")

        cot_times = []
        for query in queries[:1]:  # Just one test
            start = time.time()
            result = cot_predictor(text=query)
            elapsed = (time.time() - start) * 1000
            cot_times.append(elapsed)
            print(f"  '{query}' -> {elapsed:.1f}ms (CoT)")

        print(f"\n  Predict vs ChainOfThought: {avg:.1f}ms vs {cot_times[0]:.1f}ms")
        print(f"  ChainOfThought is {cot_times[0]/avg:.1f}x slower")

if __name__ == "__main__":
    test_performance()