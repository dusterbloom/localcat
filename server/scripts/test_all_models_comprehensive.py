#!/usr/bin/env python3
"""
COMPREHENSIVE test with ALL available models in LM Studio
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.add(sys.stderr, level="INFO")


def get_all_available_models():
    """Get ALL models from LM Studio"""
    try:
        response = requests.get("http://127.0.0.1:1234/v1/models", timeout=2)
        if response.status_code == 200:
            models = [m['id'] for m in response.json()['data']]
            return models
    except:
        pass
    return []


def test_extraction_with_model(model: str, texts: List[str]) -> Dict:
    """Test extraction with specific model"""

    results = []
    total_latency = 0

    for text in texts:
        # Try different prompts to see what works
        prompts = [
            # Prompt 1: Direct JSON
            f"""Extract knowledge graph triples. Return JSON only.
Text: "{text}"
JSON: """,

            # Prompt 2: With examples
            f"""Extract triples as JSON array.
Example: "John works at Google" → [["John", "works_at", "Google"]]
Text: "{text}"
Output: """,

            # Prompt 3: Structured
            f"""Task: Extract subject-relation-object triples.
Input: {text}
Format: [["subject", "relation", "object"]]
Output: """
        ]

        best_result = None
        best_latency = float('inf')

        for prompt in prompts[:1]:  # Test first prompt for now
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 150,
                "stream": False
            }

            try:
                t0 = time.perf_counter()
                response = requests.post(
                    "http://127.0.0.1:1234/v1/chat/completions",
                    json=payload,
                    timeout=30
                )
                latency = (time.perf_counter() - t0) * 1000

                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]

                    # Try to parse triples
                    triples = []
                    try:
                        # Find JSON in response
                        if '[' in content:
                            start = content.find('[')
                            end = content.rfind(']') + 1
                            if end > start:
                                json_str = content[start:end]
                                parsed = json.loads(json_str)
                                if isinstance(parsed, list):
                                    triples = parsed
                    except:
                        # Try other formats
                        if '(' in content and ')' in content:
                            # Tuple format
                            import re
                            matches = re.findall(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', content)
                            triples = [list(m) for m in matches]

                    if triples and latency < best_latency:
                        best_result = triples
                        best_latency = latency

            except Exception as e:
                logger.debug(f"Model {model} failed with prompt: {e}")

        results.append({
            "text": text,
            "triples": best_result or [],
            "latency": best_latency if best_latency < float('inf') else 0
        })
        total_latency += best_latency if best_latency < float('inf') else 0

    return {
        "model": model,
        "results": results,
        "avg_latency": total_latency / len(texts) if texts else 0,
        "total_triples": sum(len(r["triples"]) for r in results)
    }


def compare_with_yaml_baseline(texts: List[str]) -> Dict:
    """Get YAML baseline for comparison"""
    from core.memory.extractors.yaml_extractor import YAMLExtractor

    yaml_path = "archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml"
    extractor = YAMLExtractor(yaml_path)

    results = []
    total_latency = 0

    for text in texts:
        t0 = time.perf_counter()
        _, triples, _, doc = extractor.extract(text, 'en')
        triples = extractor.refine(text, triples, doc)
        latency = (time.perf_counter() - t0) * 1000

        results.append({
            "text": text,
            "triples": [list(t) for t in triples],
            "latency": latency
        })
        total_latency += latency

    return {
        "model": "YAML_baseline",
        "results": results,
        "avg_latency": total_latency / len(texts) if texts else 0,
        "total_triples": sum(len(r["triples"]) for r in results)
    }


def main():
    """Run comprehensive test with ALL models"""

    # Test texts covering different complexity
    test_texts = [
        # Simple
        "John works at Google",
        "Alice is the CEO",
        "Bob loves Python",

        # Medium
        "The company announced a new product that will launch next month",
        "Mary founded the startup in 2020 and it now has 50 employees",

        # Complex
        "The researchers discovered that the new compound, which was synthesized last year, shows promising results in treating the disease",
        "After graduating from MIT, Sarah joined a tech company where she developed AI systems before starting her own venture",
    ]

    print("=" * 80)
    print("COMPREHENSIVE MODEL TESTING - ALL AVAILABLE MODELS")
    print("=" * 80)

    # Get all available models
    models = get_all_available_models()
    print(f"\nFound {len(models)} models in LM Studio:")
    for m in models:
        print(f"  - {m}")

    if not models:
        print("ERROR: No models found. Is LM Studio running?")
        return

    # Test each model
    all_results = []

    # First, get YAML baseline
    print("\n" + "-" * 60)
    print("Testing: YAML_baseline")
    yaml_result = compare_with_yaml_baseline(test_texts)
    all_results.append(yaml_result)
    print(f"  Avg latency: {yaml_result['avg_latency']:.0f}ms")
    print(f"  Total triples: {yaml_result['total_triples']}")

    # Test each LLM
    for model in models:
        print("\n" + "-" * 60)
        print(f"Testing: {model}")

        result = test_extraction_with_model(model, test_texts)
        all_results.append(result)

        print(f"  Avg latency: {result['avg_latency']:.0f}ms")
        print(f"  Total triples: {result['total_triples']}")

        # Show sample output
        if result['results'] and result['results'][0]['triples']:
            print(f"  Sample: {result['results'][0]['triples']}")

    # Rankings
    print("\n" + "=" * 80)
    print("PERFORMANCE RANKINGS")
    print("=" * 80)

    # Speed ranking
    speed_sorted = sorted(all_results, key=lambda x: x['avg_latency'])
    print("\n🚀 SPEED RANKING (fastest first):")
    for i, r in enumerate(speed_sorted[:10], 1):
        print(f"{i:2}. {r['model'][:30]:30} {r['avg_latency']:>8.0f}ms")

    # Extraction quality ranking
    quality_sorted = sorted(all_results, key=lambda x: x['total_triples'], reverse=True)
    print("\n📊 EXTRACTION RICHNESS (most triples):")
    for i, r in enumerate(quality_sorted[:10], 1):
        print(f"{i:2}. {r['model'][:30]:30} {r['total_triples']:>3} triples")

    # Find best balance (low latency, good extraction)
    balance_sorted = sorted(all_results,
                           key=lambda x: x['avg_latency'] / max(1, x['total_triples']))
    print("\n⚖️  BEST BALANCE (latency per triple):")
    for i, r in enumerate(balance_sorted[:10], 1):
        ratio = r['avg_latency'] / max(1, r['total_triples'])
        print(f"{i:2}. {r['model'][:30]:30} {ratio:>8.0f}ms/triple")

    # Detailed comparison for best models
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON - TOP 3 MODELS")
    print("=" * 80)

    for r in quality_sorted[:3]:
        print(f"\n📍 {r['model']}")
        print("-" * 40)
        for i, res in enumerate(r['results'][:3]):  # First 3 examples
            print(f"Text: \"{res['text']}\"")
            print(f"Extracted ({len(res['triples'])}): {res['triples']}")
            print(f"Latency: {res['latency']:.0f}ms")
            print()

    # Save full results
    output = {
        "timestamp": time.time(),
        "models_tested": len(all_results),
        "test_texts": test_texts,
        "results": all_results,
        "rankings": {
            "speed": [(r['model'], r['avg_latency']) for r in speed_sorted[:5]],
            "quality": [(r['model'], r['total_triples']) for r in quality_sorted[:5]],
            "balance": [(r['model'], r['avg_latency']/max(1, r['total_triples']))
                       for r in balance_sorted[:5]]
        }
    }

    output_path = Path("results/comprehensive_model_test.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n✅ Full results saved to: {output_path}")

    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Find best for different use cases
    fast_good = None
    for r in speed_sorted:
        if r['total_triples'] >= 10:  # At least decent extraction
            fast_good = r
            break

    if fast_good:
        print(f"\n🎯 Best for hotpath: {fast_good['model']}")
        print(f"   - Latency: {fast_good['avg_latency']:.0f}ms")
        print(f"   - Quality: {fast_good['total_triples']} triples")

    best_quality = quality_sorted[0] if quality_sorted else None
    if best_quality:
        print(f"\n🏆 Best for accuracy: {best_quality['model']}")
        print(f"   - Latency: {best_quality['avg_latency']:.0f}ms")
        print(f"   - Quality: {best_quality['total_triples']} triples")


if __name__ == "__main__":
    main()