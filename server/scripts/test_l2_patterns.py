#!/usr/bin/env python3
"""
Test key L2 micro features: coreference, discourse/temporal links, and clustering.
Outputs a simple pass/fail summary per case and saves detailed results.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.add(sys.stderr, level="INFO")


def extract_triples(text: str) -> Tuple[List[Tuple[str, str, str]], float]:
    from core.memory.extractors.yaml_extractor import YAMLExtractor
    yaml_path = "archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml"
    extractor = YAMLExtractor(yaml_path)
    t0 = time.perf_counter()
    _, triples, _, doc = extractor.extract(text, 'en')
    triples = extractor.refine(text, triples, doc)
    return triples, (time.perf_counter() - t0) * 1000


def has_triplet(triples: List[Tuple[str, str, str]], s=None, r=None, o=None, contains: bool = False) -> bool:
    def norm(x: Any) -> str:
        return (str(x or "").strip().lower())
    sn, rn, on = norm(s) if s is not None else None, norm(r) if r is not None else None, norm(o) if o is not None else None
    for ts, tr, to in triples:
        tsn, trn, ton = norm(ts), norm(tr), norm(to)
        if contains:
            ok = True
            if sn is not None and sn not in tsn:
                ok = False
            if rn is not None and rn not in trn:
                ok = False
            if on is not None and on not in ton:
                ok = False
            if ok:
                return True
        else:
            if (sn is None or sn == tsn) and (rn is None or rn == trn) and (on is None or on == ton):
                return True
    return False


def main():
    # Enable coref/discourse passes
    os.environ["YAML_COREF"] = "on"

    cases = []

    # 1) Pronoun resolution (3rd person)
    text = "Mary met John. He works at Google."
    triples, ms = extract_triples(text)
    # Expect resolution of 'He' → likely 'john'
    pass1 = any(s in {"john", "mary"} and ("work" in r) for s, r, _ in triples)
    cases.append({"name": "PRONOMINAL_3SG_RESOLUTION", "text": text, "triples": triples, "latency": ms, "pass": pass1})

    # 2) Definite NP coref
    text = "Alice bought a car. The car is red."
    triples, ms = extract_triples(text)
    # Accept either direct adjectival copula or any 'car is <adj>' style edge
    pass2 = has_triplet(triples, s="alice", r=None, o="car", contains=True) and (
        has_triplet(triples, s="car", r="be", o="red") or
        has_triplet(triples, s="car", r="is", o="red")
    )
    cases.append({"name": "DEFINITE_NP_COREFERENCE", "text": text, "triples": triples, "latency": ms, "pass": pass2})

    # 3) Discourse causal connective
    text = "John left because Mary called."
    triples, ms = extract_triples(text)
    pass3 = has_triplet(triples, s="john", r="leave_because_of", o="call")
    cases.append({"name": "DISCOURSE_CONNECTIVE_RESOLUTION", "text": text, "triples": triples, "latency": ms, "pass": pass3})

    # 4) Temporal chaining (before/after)
    text = "Before John left, Mary called."
    triples, ms = extract_triples(text)
    pass4 = has_triplet(triples, s="john", r="leave_after", o="call")
    cases.append({"name": "TEMPORAL_EVENT_CHAINING", "text": text, "triples": triples, "latency": ms, "pass": pass4})

    # 5) Entity clustering (singularize/alias)
    text = "A dog runs."
    triples, ms = extract_triples(text)
    # Accept if we see at least one intransitive 'dog' subject
    subj_set = {s for s, _, _ in triples}
    pass5 = ("dog" in subj_set)
    cases.append({"name": "ENTITY_CLUSTER_MERGING", "text": text, "triples": triples, "latency": ms, "pass": pass5})

    # Summary
    print("=" * 80)
    print("L2 PATTERN TESTS")
    print("=" * 80)
    for c in cases:
        status = "✅" if c["pass"] else "⚠️"
        print(f"\n{status} {c['name']} | {c['latency']:.0f}ms\nText: {c['text']}\nTriples: {c['triples']}")

    total = len(cases)
    passed = sum(1 for c in cases if c["pass"])
    print("\n" + "-" * 60)
    print(f"Summary: {passed}/{total} passed")

    Path("results/l2_patterns_test.json").write_text(json.dumps(cases, indent=2))
    print("\n✅ Results saved to: results/l2_patterns_test.json")


if __name__ == "__main__":
    main()
