#!/usr/bin/env python3
"""
Basic L3 sanity tests using language-specific micro rules.
Falls back to English spaCy if specific models are unavailable.
"""

import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.add(sys.stderr, level="INFO")


def extract(text: str, lang: str):
    from core.memory.extractors.yaml_extractor import YAMLExtractor
    yaml_path = "archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml"
    ext = YAMLExtractor(yaml_path)
    t0 = time.perf_counter()
    _, triples, _, doc = ext.extract(text, lang)
    triples = ext.refine(text, triples, doc)
    return triples, (time.perf_counter() - t0) * 1000


def main():
    # Ensure L3 is enabled by the YAML (language_extensions present)
    cases = []

    # Spanish: gustar-type psych verbs
    es_text = "A Juan le gusta el chocolate."
    tri, ms = extract(es_text, 'es')
    cases.append({
        "name": "ES_GUSTAR_PSYCH_VERBS",
        "text": es_text,
        "triples": tri,
        "latency": ms,
        "pass": any(r == 'like' for _, r, _ in tri)
    })

    # French: clitic pronoun climbing
    fr_text = "Je le veux."
    tri, ms = extract(fr_text, 'fr')
    cases.append({
        "name": "FR_CLITIC_PRONOUN_CLIMBING",
        "text": fr_text,
        "triples": tri,
        "latency": ms,
        "pass": any(r.startswith('vou') or r in {'vouloir', 'veux', 'veu'} for _, r, _ in tri)
    })

    # Chinese: topic-comment + like
    zh_text = "北京，我喜欢。"
    tri, ms = extract(zh_text, 'zh')
    cases.append({
        "name": "ZH_TOPIC_COMMENT_STRUCTURE",
        "text": zh_text,
        "triples": tri,
        "latency": ms,
        "pass": any(r == 'like' for _, r, _ in tri)
    })

    # Summary
    print("=" * 80)
    print("L3 PATTERN TESTS")
    print("=" * 80)
    for c in cases:
        status = "✅" if c["pass"] else "⚠️"
        print(f"\n{status} {c['name']} | {c['latency']:.0f}ms\nText: {c['text']}\nTriples: {c['triples']}")

    total = len(cases)
    passed = sum(1 for c in cases if c["pass"])
    print("\n" + "-" * 60)
    print(f"Summary: {passed}/{total} passed")

    Path("results/l3_patterns_test.json").write_text(json.dumps(cases, indent=2))
    print("\n✅ Results saved to: results/l3_patterns_test.json")


if __name__ == '__main__':
    main()
