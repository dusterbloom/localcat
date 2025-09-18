#!/usr/bin/env python3
"""
Test quality of fast models for LocalCat/HotMem use cases
"""
import time
import requests
import json
from typing import Dict, List, Tuple

def test_model(model_id: str, prompt: str, max_tokens: int = 30) -> Tuple[str, float]:
    """Test a model and return response + time"""
    start = time.time()

    response = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
    )

    elapsed = (time.time() - start) * 1000

    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        return content, elapsed
    return "ERROR", elapsed

def evaluate_quality():
    """Comprehensive quality evaluation"""

    # Test cases relevant to LocalCat/HotMem
    test_suite = {
        "Intent Classification": [
            ("Hello!", "REACTION"),
            ("What's the weather like?", "PURE_QUESTION"),
            ("I work at Google as an engineer", "FACT_STATEMENT"),
            ("Actually, I meant Microsoft", "CORRECTION"),
            ("How are you doing?", "PURE_QUESTION"),
        ],

        "Entity Extraction": [
            ("My dog Max is 5 years old", ["dog", "Max", "5 years"]),
            ("Sarah works at OpenAI in San Francisco", ["Sarah", "OpenAI", "San Francisco"]),
            ("The meeting is scheduled for 3pm tomorrow", ["meeting", "3pm", "tomorrow"]),
            ("John's phone number is 555-1234", ["John", "phone number", "555-1234"]),
        ],

        "Relationship Extraction": [
            ("Sarah is married to Michael", ("Sarah", "married_to", "Michael")),
            ("John works at Google", ("John", "works_at", "Google")),
            ("Emma is Sarah's daughter", ("Emma", "daughter_of", "Sarah")),
            ("The cat belongs to Mary", ("cat", "belongs_to", "Mary")),
        ],

        "Simple QA": [
            ("Who is the CEO of OpenAI?", "Sam Altman"),
            ("What color is the sky?", "blue"),
            ("What is 2+2?", "4"),
            ("Name a programming language", ["Python", "JavaScript", "Java", "C++"]),
        ],

        "Memory Retrieval": [
            ("Find facts about Sarah", "search for Sarah"),
            ("What do we know about the meeting?", "search for meeting"),
            ("Tell me about John's work", "search for John work"),
        ]
    }

    models = [
        "llama-3.2-1b-instruct-4bit",
        "gemma-3-270m-it-mlx-8bit",
        "llama-3.2-3b-instruct-4bit"  # Baseline for quality
    ]

    print("="*70)
    print("QUALITY COMPARISON: Fast Models for LocalCat/HotMem")
    print("="*70)

    results = {}

    for model in models:
        print(f"\n\n🤖 Testing: {model}")
        print("-"*50)

        model_scores = {}
        model_times = []

        for category, tests in test_suite.items():
            print(f"\n📋 {category}:")
            correct = 0
            total = len(tests)

            for test_input, expected in tests:
                # Create appropriate prompt based on category
                if category == "Intent Classification":
                    prompt = f"Classify the intent of this text as one of: REACTION, PURE_QUESTION, FACT_STATEMENT, CORRECTION.\nText: {test_input}\nIntent:"
                elif category == "Entity Extraction":
                    prompt = f"Extract all entities from: {test_input}\nEntities:"
                elif category == "Relationship Extraction":
                    prompt = f"Extract the relationship triple (subject, predicate, object) from: {test_input}\nTriple:"
                elif category == "Simple QA":
                    prompt = test_input
                else:  # Memory Retrieval
                    prompt = f"What search query would you use to: {test_input}\nSearch query:"

                response, time_ms = test_model(model, prompt, 30)
                model_times.append(time_ms)

                # Evaluate response quality
                response_lower = response.lower()

                if category == "Intent Classification":
                    if expected.lower() in response_lower:
                        correct += 1
                        print(f"  ✓ {test_input[:30]:30} → {expected:15} [{time_ms:.0f}ms]")
                    else:
                        print(f"  ✗ {test_input[:30]:30} → Got: {response[:20]:20} [{time_ms:.0f}ms]")

                elif category == "Entity Extraction":
                    found = sum(1 for entity in expected if entity.lower() in response_lower)
                    if found >= len(expected) * 0.5:  # At least 50% entities found
                        correct += 1
                        print(f"  ✓ Found {found}/{len(expected)} entities [{time_ms:.0f}ms]")
                    else:
                        print(f"  ✗ Found {found}/{len(expected)} entities [{time_ms:.0f}ms]")

                elif category == "Relationship Extraction":
                    subj, _, obj = expected
                    if subj.lower() in response_lower and obj.lower() in response_lower:
                        correct += 1
                        print(f"  ✓ {expected} [{time_ms:.0f}ms]")
                    else:
                        print(f"  ✗ Expected: {expected}, Got: {response[:30]} [{time_ms:.0f}ms]")

                elif category == "Simple QA":
                    if isinstance(expected, list):
                        if any(e.lower() in response_lower for e in expected):
                            correct += 1
                            print(f"  ✓ {test_input[:30]:30} [{time_ms:.0f}ms]")
                        else:
                            print(f"  ✗ {test_input[:30]:30} [{time_ms:.0f}ms]")
                    else:
                        if expected.lower() in response_lower:
                            correct += 1
                            print(f"  ✓ {test_input[:30]:30} [{time_ms:.0f}ms]")
                        else:
                            print(f"  ✗ {test_input[:30]:30} [{time_ms:.0f}ms]")

                else:  # Memory Retrieval
                    if "search" in response_lower or "find" in response_lower:
                        correct += 1
                        print(f"  ✓ Reasonable search strategy [{time_ms:.0f}ms]")
                    else:
                        print(f"  ✗ Poor search strategy [{time_ms:.0f}ms]")

            accuracy = (correct / total) * 100
            model_scores[category] = accuracy
            print(f"  Score: {correct}/{total} ({accuracy:.0f}%)")

        # Calculate overall metrics
        avg_accuracy = sum(model_scores.values()) / len(model_scores)
        avg_time = sum(model_times) / len(model_times)

        results[model] = {
            "scores": model_scores,
            "avg_accuracy": avg_accuracy,
            "avg_time": avg_time
        }

    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY: Quality vs Speed Tradeoff")
    print("="*70)

    print("\n📊 Overall Results:")
    print(f"{'Model':35} {'Avg Accuracy':15} {'Avg Time':10} {'Status':10}")
    print("-"*70)

    for model, metrics in results.items():
        status = "✅ FAST" if metrics['avg_time'] < 200 else "⚠️  SLOW"
        if metrics['avg_accuracy'] < 50:
            status += " ❌ POOR"
        elif metrics['avg_accuracy'] < 70:
            status += " ⚠️  OK"
        else:
            status += " ✅ GOOD"

        print(f"{model:35} {metrics['avg_accuracy']:>14.1f}% {metrics['avg_time']:>9.0f}ms {status:10}")

    print("\n📈 Category Breakdown:")
    categories = list(test_suite.keys())
    print(f"{'Model':35}", end="")
    for cat in categories:
        print(f" {cat[:8]:>8}", end="")
    print()
    print("-"*100)

    for model, metrics in results.items():
        print(f"{model:35}", end="")
        for cat in categories:
            score = metrics['scores'][cat]
            print(f" {score:>7.0f}%", end="")
        print()

    print("\n💡 RECOMMENDATIONS:")
    print("-"*50)

    # Find best model
    fast_models = [(m, r) for m, r in results.items() if r['avg_time'] < 200]
    if fast_models:
        best_fast = max(fast_models, key=lambda x: x[1]['avg_accuracy'])
        print(f"\n✅ BEST FAST MODEL: {best_fast[0]}")
        print(f"   - Average accuracy: {best_fast[1]['avg_accuracy']:.1f}%")
        print(f"   - Average latency: {best_fast[1]['avg_time']:.0f}ms")
        print(f"   - Best for: {', '.join([cat for cat, score in best_fast[1]['scores'].items() if score >= 60])}")

    print("\n🎯 For LocalCat/HotMem Production:")
    print("   1. Use Llama-3.2-1B for general tasks (best quality under 200ms)")
    print("   2. Consider task-specific fine-tuning for better accuracy")
    print("   3. Implement fallback to larger model for complex queries")
    print("   4. Cache frequent queries to achieve <5ms for repeated patterns")

if __name__ == "__main__":
    evaluate_quality()