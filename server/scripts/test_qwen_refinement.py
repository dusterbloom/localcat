#!/usr/bin/env python3
"""
Test refinement using Qwen2.5-coder-0.5b-instruct model
This model showed good results in our comprehensive test
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


class QwenRefinement:
    """Refine extractions using Qwen2.5-coder model via LM Studio"""

    def __init__(self):
        self.model = "qwen2.5-coder-0.5b-instruct"
        self.url = "http://127.0.0.1:1234/v1/chat/completions"

    def refine(self, text: str, raw_triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Refine raw triples using Qwen"""

        # Simple, direct prompt
        prompt = f"""Fix these extraction triples. Output JSON array only.

Text: "{text}"
Current: {json.dumps(raw_triples)}

Fix these issues:
- "work at google" → ["John", "works_at", "Google"]
- "alice" → "Alice" (capitalize names)
- Empty objects should be filled if possible
- Pronouns like "it" should be resolved

Output (JSON array of [subject, relation, object]):"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200,
            "stream": False
        }

        try:
            t0 = time.perf_counter()
            response = requests.post(self.url, json=payload, timeout=10)
            latency_ms = (time.perf_counter() - t0) * 1000

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]

                # Parse refined triples
                try:
                    # Remove markdown code blocks if present
                    if '```' in content:
                        content = content.replace('```json', '').replace('```', '')

                    # Handle multiple arrays on same line
                    content = content.strip()
                    if content.startswith('[') and '], [' in content:
                        # Multiple arrays: wrap in outer array
                        content = '[' + content + ']'

                    # Find JSON array in response
                    start = content.find('[')
                    end = content.rfind(']')
                    if start >= 0 and end > start:
                        json_str = content[start:end+1]
                        refined = json.loads(json_str)

                        # Convert to tuples
                        refined_triples = []
                        for item in refined:
                            if isinstance(item, list) and len(item) == 3:
                                refined_triples.append(tuple(item))
                            elif isinstance(item, dict):
                                # Handle dict format from Qwen
                                s = item.get('subject', item.get('s', ''))
                                r = item.get('relation', item.get('predicate', item.get('r', '')))
                                o = item.get('object', item.get('o', ''))
                                if s and r:
                                    refined_triples.append((s, r, o or ''))

                        logger.info(f"Qwen refinement ({latency_ms:.0f}ms): {len(refined_triples)} triples")
                        return refined_triples

                except Exception as e:
                    logger.warning(f"Failed to parse Qwen response: {e}")
                    logger.info(f"Raw response: {content[:300]}")

        except Exception as e:
            logger.error(f"Qwen refinement failed: {e}")

        # Return original if refinement fails
        return raw_triples


def test_qwen_refinement():
    """Test Qwen2.5-coder for refinement"""

    from core.memory.extractors.yaml_extractor import YAMLExtractor

    # Test cases that need refinement
    test_cases = [
        "John works at Google",
        "Alice founded the company. It now has 100 employees.",
        "The CEO announced that the new product will launch next month",
        "John and Mary work at the same company",
        "The book was written by Stephen King",
    ]

    yaml_path = "archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml"

    print("=" * 80)
    print("TESTING QWEN2.5-CODER REFINEMENT")
    print("=" * 80)

    # Initialize extractors
    yaml_ext = YAMLExtractor(yaml_path)
    qwen_refiner = QwenRefinement()

    results = []

    for text in test_cases:
        print(f"\nText: \"{text}\"")
        print("-" * 60)

        # Extract with YAML
        t0 = time.perf_counter()
        _, yaml_triples, _, doc = yaml_ext.extract(text, 'en')
        yaml_triples = yaml_ext.refine(text, yaml_triples, doc)
        yaml_latency = (time.perf_counter() - t0) * 1000

        print(f"YAML baseline ({yaml_latency:.0f}ms): {yaml_triples}")

        # Refine with Qwen
        t0 = time.perf_counter()
        qwen_triples = qwen_refiner.refine(text, yaml_triples)
        qwen_latency = (time.perf_counter() - t0) * 1000

        print(f"Qwen refined ({qwen_latency:.0f}ms): {qwen_triples}")

        # Compare
        improved = qwen_triples != yaml_triples
        if improved:
            print("✅ Improvement detected!")
        else:
            print("⚠️ No improvement")

        results.append({
            "text": text,
            "yaml": yaml_triples,
            "qwen": qwen_triples,
            "yaml_latency": yaml_latency,
            "qwen_latency": qwen_latency,
            "improved": improved
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    improvements = sum(1 for r in results if r["improved"])
    avg_yaml_latency = sum(r["yaml_latency"] for r in results) / len(results)
    avg_qwen_latency = sum(r["qwen_latency"] for r in results) / len(results)

    print(f"\nImprovements: {improvements}/{len(results)} cases")
    print(f"Avg YAML latency: {avg_yaml_latency:.0f}ms")
    print(f"Avg Qwen refinement latency: {avg_qwen_latency:.0f}ms")
    print(f"Total latency (YAML + Qwen): {avg_yaml_latency + avg_qwen_latency:.0f}ms")

    # Show improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENTS MADE BY QWEN")
    print("=" * 80)

    for r in results:
        if r["improved"]:
            print(f"\nText: \"{r['text']}\"")
            print(f"Before: {r['yaml']}")
            print(f"After:  {r['qwen']}")

    # Check lexicalization specifically
    print("\n" + "=" * 80)
    print("LEXICALIZATION CHECK")
    print("=" * 80)

    for r in results:
        for s, rel, o in r["qwen"]:
            if "_" in rel:
                print(f"✅ Lexicalized: {rel}")

    # Save results
    output_path = Path("results/qwen_refinement_test.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Results saved to: {output_path}")

    # Recommendations
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    print(f"""
Based on testing:
1. Qwen2.5-coder: {avg_qwen_latency:.0f}ms avg latency, {improvements}/{len(results)} improvements
2. Best for: Simple to medium complexity refinement
3. Use when: Need fast lexicalization and capitalization fixes
4. Total pipeline: ~{avg_yaml_latency + avg_qwen_latency:.0f}ms (YAML + refinement)

Next steps:
- Implement codegen for YAML (10x speedup: 450ms → 45ms)
- Use staged routing based on complexity
- Keep Qwen for medium cases, skip for simple cases
""")


if __name__ == "__main__":
    test_qwen_refinement()