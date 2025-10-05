#!/usr/bin/env python3
"""
Test LLM extraction with models available in LM Studio
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


def test_llm_extraction(model: str, text: str) -> Tuple[List[Tuple], float]:
    """Test extraction using LLM via LM Studio"""

    url = "http://127.0.0.1:1234/v1/chat/completions"

    prompt = f"""Extract knowledge graph triples from this text. Return JSON array only.
Format: [["subject", "relation", "object"], ...]

Text: "{text}"

Rules:
- Use verb_preposition format (e.g., "works_at", "founded_in")
- Resolve pronouns (I→you, it→antecedent)
- Keep entities capitalized

JSON output:"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
        "stream": False
    }

    t0 = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=10)
        latency_ms = (time.perf_counter() - t0) * 1000

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]

            # Extract JSON from response
            try:
                # Find JSON array
                start = content.find('[')
                end = content.rfind(']')
                if start >= 0 and end > start:
                    json_str = content[start:end+1]
                    triples_raw = json.loads(json_str)
                    triples = [tuple(t) if isinstance(t, list) else t for t in triples_raw]
                    return triples, latency_ms
            except:
                logger.warning(f"Failed to parse LLM response: {content}")
                return [], latency_ms
        else:
            logger.error(f"LLM request failed: {response.status_code}")
            return [], latency_ms
    except Exception as e:
        logger.error(f"LLM extraction error: {e}")
        return [], 0


def compare_all_methods():
    """Compare YAML, SLM, and LLMs"""

    from core.memory.extractors.yaml_extractor import YAMLExtractor
    from core.memory.extractors.hybrid_slm import YAMLWithSLMRefinement

    # Test cases
    test_cases = [
        "John works at Google",
        "Alice founded the company in 2020 which now has offices in multiple countries",
        "The CEO announced that the new product will launch next month",
        "I visited the museum yesterday and saw amazing paintings",
    ]

    # Available LLMs from LM Studio
    llm_models = [
        "llama-3.2-3b-instruct",       # Good general model
        "qwen2.5-coder-0.5b-instruct", # Small coder model
        "lfm2-350m-extract",           # Extraction-specific model!
        "google/gemma-3n-e4b",         # Google model
    ]

    yaml_path = "archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml"

    print("=" * 80)
    print("COMPREHENSIVE EXTRACTION COMPARISON")
    print("=" * 80)

    # Initialize extractors
    yaml_ext = YAMLExtractor(yaml_path)
    slm_ext = YAMLWithSLMRefinement(
        yaml_path=yaml_path,
        slm_model='mlx-community/Qwen2.5-0.5B-Instruct-4bit',
        max_refinement_ms=200
    )

    results = {}

    for text in test_cases:
        print(f"\nText: \"{text}\"")
        print("-" * 60)

        results[text] = {}

        # 1. YAML baseline
        t0 = time.perf_counter()
        _, yaml_triples, _, doc = yaml_ext.extract(text, 'en')
        yaml_triples = yaml_ext.refine(text, yaml_triples, doc)
        yaml_latency = (time.perf_counter() - t0) * 1000

        print(f"YAML ({yaml_latency:.0f}ms): {yaml_triples}")
        results[text]["yaml"] = {"triples": yaml_triples, "latency": yaml_latency}

        # 2. YAML + SLM
        t0 = time.perf_counter()
        _, slm_triples, _, _ = slm_ext.extract(text, 'en')
        slm_latency = (time.perf_counter() - t0) * 1000

        print(f"SLM ({slm_latency:.0f}ms): {slm_triples}")
        results[text]["slm"] = {"triples": slm_triples, "latency": slm_latency}

        # 3. LLMs
        for model in llm_models:
            try:
                result = test_llm_extraction(model, text)
                if result:
                    llm_triples, llm_latency = result
                    print(f"{model[:20]} ({llm_latency:.0f}ms): {llm_triples}")
                    results[text][model] = {"triples": llm_triples, "latency": llm_latency}
                else:
                    print(f"{model[:20]} (FAILED)")
            except Exception as e:
                print(f"{model[:20]} (ERROR: {e})")

    # Summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    # Calculate averages
    avg_latencies = {}
    avg_counts = {}

    for method in ["yaml", "slm"] + llm_models:
        latencies = []
        counts = []
        for text in test_cases:
            if method in results[text]:
                latencies.append(results[text][method]["latency"])
                counts.append(len(results[text][method]["triples"]))
        if latencies:
            avg_latencies[method] = sum(latencies) / len(latencies)
            avg_counts[method] = sum(counts) / len(counts)

    print(f"\n{'Method':<25} {'Avg Latency (ms)':>15} {'Avg Triple Count':>15}")
    print("-" * 60)

    for method in ["yaml", "slm"] + llm_models:
        if method in avg_latencies:
            name = method[:25]
            print(f"{name:<25} {avg_latencies[method]:>15.0f} {avg_counts[method]:>15.1f}")

    # Best method analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Speed ranking
    speed_ranking = sorted(avg_latencies.items(), key=lambda x: x[1])
    print("\n🚀 Speed Ranking:")
    for i, (method, latency) in enumerate(speed_ranking[:5], 1):
        print(f"  {i}. {method}: {latency:.0f}ms")

    # Extraction richness ranking
    richness_ranking = sorted(avg_counts.items(), key=lambda x: x[1], reverse=True)
    print("\n📊 Extraction Richness (avg triple count):")
    for i, (method, count) in enumerate(richness_ranking[:5], 1):
        print(f"  {i}. {method}: {count:.1f} triples")

    # Save results
    output_path = Path("results/llm_extraction_comparison.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Detailed results saved to: {output_path}")


if __name__ == "__main__":
    compare_all_methods()