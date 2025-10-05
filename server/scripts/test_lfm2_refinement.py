#!/usr/bin/env python3
"""
Test refinement using LFM2-350M-Extract model
This model is specifically designed for extraction tasks
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


class LFM2ExtractRefinement:
    """Refine extractions using LFM2-350M-Extract model via LM Studio"""

    def __init__(self):
        self.model = "lfm2-350m-extract"
        self.url = "http://127.0.0.1:1234/v1/chat/completions"

    def refine(self, text: str, raw_triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Refine raw triples using LFM2-350M-Extract"""

        # Create focused prompt for extraction refinement with strict JSON schema
        prompt = f"""You are a triple extraction refinement system. Output ONLY a JSON array.

JSON Schema:
{{
  "type": "array",
  "items": {{
    "type": "array",
    "items": [
      {{"type": "string", "description": "subject"}},
      {{"type": "string", "description": "relation"}},
      {{"type": "string", "description": "object"}}
    ],
    "minItems": 3,
    "maxItems": 3
  }}
}}

Example:
Input: "John works at Google"
Output: [["John", "works_at", "Google"]]

Input: "Alice founded the company. It has 100 employees."
Output: [["Alice", "founded", "company"], ["company", "has", "100 employees"]]

Rules:
- Lexicalize: "work at" → "works_at", "give to" → "gives_to"
- Resolve pronouns: "it" → actual entity
- Fix empty objects
- Capitalize proper nouns

Text: "{text}"
Current: {json.dumps(raw_triples)}

Output JSON array only:"""

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
                                # Handle dict format
                                s = item.get('subject', item.get('s', ''))
                                r = item.get('relation', item.get('predicate', item.get('r', '')))
                                o = item.get('object', item.get('o', ''))
                                if s and r:
                                    refined_triples.append((s, r, o or ''))

                        logger.info(f"LFM2 refinement ({latency_ms:.0f}ms): {len(refined_triples)} triples")
                        return refined_triples

                except Exception as e:
                    logger.warning(f"Failed to parse LFM2 response: {e}")
                    logger.info(f"Raw LFM2 response: {content[:200]}")  # Show first 200 chars

        except Exception as e:
            logger.error(f"LFM2 refinement failed: {e}")

        # Return original if refinement fails
        return raw_triples


def test_lfm2_refinement():
    """Test LFM2-350M-Extract for refinement"""

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
    print("TESTING LFM2-350M-EXTRACT REFINEMENT")
    print("=" * 80)

    # Initialize extractors
    yaml_ext = YAMLExtractor(yaml_path)
    lfm2_refiner = LFM2ExtractRefinement()

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

        # Refine with LFM2
        t0 = time.perf_counter()
        lfm2_triples = lfm2_refiner.refine(text, yaml_triples)
        lfm2_latency = (time.perf_counter() - t0) * 1000

        print(f"LFM2 refined ({lfm2_latency:.0f}ms): {lfm2_triples}")

        # Compare
        improved = lfm2_triples != yaml_triples
        if improved:
            print("✅ Improvement detected!")
        else:
            print("⚠️ No improvement")

        results.append({
            "text": text,
            "yaml": yaml_triples,
            "lfm2": lfm2_triples,
            "yaml_latency": yaml_latency,
            "lfm2_latency": lfm2_latency,
            "improved": improved
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    improvements = sum(1 for r in results if r["improved"])
    avg_yaml_latency = sum(r["yaml_latency"] for r in results) / len(results)
    avg_lfm2_latency = sum(r["lfm2_latency"] for r in results) / len(results)

    print(f"\nImprovements: {improvements}/{len(results)} cases")
    print(f"Avg YAML latency: {avg_yaml_latency:.0f}ms")
    print(f"Avg LFM2 refinement latency: {avg_lfm2_latency:.0f}ms")
    print(f"Total latency (YAML + LFM2): {avg_yaml_latency + avg_lfm2_latency:.0f}ms")

    # Show improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENTS MADE BY LFM2")
    print("=" * 80)

    for r in results:
        if r["improved"]:
            print(f"\nText: \"{r['text']}\"")
            print(f"Before: {r['yaml']}")
            print(f"After:  {r['lfm2']}")

    # Save results
    output_path = Path("results/lfm2_refinement_test.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Results saved to: {output_path}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if improvements > len(results) / 2:
        print("""
✅ LFM2-350M-Extract shows significant improvements!
- Use for medium-complexity cases (0.4 < complexity < 0.7)
- Budget 300-400ms for refinement
- Focus on pronoun resolution and lexicalization
""")
    else:
        print("""
⚠️ LFM2-350M-Extract shows limited improvements.
Consider:
- Better prompt engineering
- Using qwen2.5-coder-0.5b-instruct instead (faster)
- Skip refinement for simple cases
""")


if __name__ == "__main__":
    test_lfm2_refinement()