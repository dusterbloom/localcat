#!/usr/bin/env python3
"""
Test script for new SLM and LLM models:
- SLM: qwen2.5-coder-0.5b-instruct
- LLM: openai/gpt-oss-20b
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.add(sys.stderr, level="INFO")


def test_slm_extraction():
    """Test SLM via LM Studio with two models (lfm2-350m-extract, qwen2.5-coder-0.5b-instruct)."""
    print("\n" + "=" * 60)
    print("Testing SLM via LM Studio: lfm2-350m-extract / qwen2.5-coder-0.5b-instruct")
    print("=" * 60)

    # Configure for LM Studio (OpenAI-compatible)
    os.environ["SLM_REFINEMENT_ENABLED"] = "true"
    os.environ["SLM_PROVIDER"] = "openai"
    os.environ["SLM_BASE_URL"] = os.getenv("SLM_BASE_URL", "http://127.0.0.1:1234/v1")
    os.environ["SLM_API_KEY"] = os.getenv("SLM_API_KEY", "not-needed")
    os.environ["SLM_PRIMARY_MODEL"] = os.getenv("SLM_PRIMARY_MODEL", "lfm2-350m-extract")
    os.environ["SLM_SECONDARY_MODEL"] = os.getenv("SLM_SECONDARY_MODEL", "qwen2.5-coder-0.5b-instruct")
    os.environ["SLM_MODE"] = os.getenv("SLM_MODE", "fallback")  # try primary then secondary
    os.environ["SLM_MAX_REFINEMENT_MS"] = os.getenv("SLM_MAX_REFINEMENT_MS", "200")
    os.environ["SLM_TEMP"] = os.getenv("SLM_TEMP", "0.1")
    os.environ["SLM_MAX_TOKENS"] = os.getenv("SLM_MAX_TOKENS", "120")
    os.environ["SLM_PREWARM_ON_INIT"] = os.getenv("SLM_PREWARM_ON_INIT", "true")

    from core.memory.extractors.hybrid_slm import YAMLWithSLMRefinement

    yaml_path = "archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml"
    extractor = YAMLWithSLMRefinement(
        yaml_path=yaml_path,
        slm_model=None,
        max_refinement_ms=int(os.getenv("SLM_MAX_REFINEMENT_MS", "200"))
    )

    # Test examples
    test_cases = [
        "John works at Google",
        "The CEO announced a new product that will launch next month",
        "Alice, who founded the company in 2020, is planning to expand to Europe",
    ]

    for text in test_cases:
        print(f"\nInput: {text}")
        print("-" * 40)

        t0 = time.perf_counter()
        entities, triples, neg_count, doc = extractor.extract(text, "en")
        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"Triples ({len(triples)}):")
        for s, r, d in triples:
            print(f"  ({s}, {r}, {d})")
        print(f"Latency: {latency_ms:.1f}ms")

        # Check if SLM was actually used
        metrics = extractor.get_metrics()
        print(f"SLM enabled: {metrics['slm_enabled']} | provider={metrics.get('provider')} | primary={metrics.get('primary_model')} | mode={metrics.get('mode')}")


def test_llm_extraction():
    """Test Hybrid with openai/gpt-oss-20b"""
    print("\n" + "=" * 60)
    print("Testing LLM: openai/gpt-oss-20b")
    print("=" * 60)

    # Check if LM Studio is running
    import requests
    lm_studio_url = "http://127.0.0.1:1234/v1"
    try:
        response = requests.get(f"{lm_studio_url}/models", timeout=1)
        if response.status_code == 200:
            models = response.json()
            print(f"LM Studio available. Models: {models.get('data', [])[:2]}")
        else:
            print("⚠️  LM Studio not responding")
            return
    except Exception as e:
        print(f"⚠️  Cannot connect to LM Studio: {e}")
        return

    # Configure for LLM
    os.environ["HOTMEM_LLM_ASSISTED"] = "true"
    os.environ["HOTMEM_LLM_ASSISTED_MODEL"] = "openai/gpt-oss-20b"
    os.environ["HOTMEM_LLM_ASSISTED_BASE_URL"] = lm_studio_url
    os.environ["HOTMEM_COMPLEXITY_THRESHOLD"] = "0.3"  # Use LLM for more cases

    # Note: This would need the actual hybrid extractor implementation
    # For now, we'll simulate with a basic test
    print("\nNote: Hybrid LLM extraction requires recovered_hybrid.py implementation")
    print("Would use model: openai/gpt-oss-20b via LM Studio")


def run_ab_comparison():
    """Run A/B test with both models"""
    print("\n" + "=" * 60)
    print("A/B Comparison: YAML vs SLM vs Hybrid")
    print("=" * 60)

    # Run the A/B test script
    cmd = """
    YAML_GRAPH_JUDGE=on \
    ENABLE_YAML_JUDGE=true \
    ENABLE_SLM=true \
    SLM_REFINEMENT_ENABLED=true \
    SLM_MODEL_PATH=qwen2.5-coder-0.5b-instruct \
    ENABLE_HYBRID=true \
    HOTMEM_LLM_ASSISTED=true \
    HOTMEM_LLM_ASSISTED_MODEL=openai/gpt-oss-20b \
    python scripts/eval_extraction_ab.py \
        --dataset tests/data/simple_test.json \
        --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml \
        --output results/model_comparison_$(date +%s).json
    """

    print("Run this command to compare all methods:")
    print(cmd.strip())


def main():
    """Main test runner"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["slm", "llm", "both", "ab"], default="slm")
    args = parser.parse_args()

    if args.test in ["slm", "both"]:
        test_slm_extraction()

    if args.test in ["llm", "both"]:
        test_llm_extraction()

    if args.test == "ab":
        run_ab_comparison()


if __name__ == "__main__":
    main()
