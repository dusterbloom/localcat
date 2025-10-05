#!/usr/bin/env python3
"""
Side-by-side extraction diff:
- A = YAML extractor WITHOUT judge
- B = YAML extractor WITH judge (lite or distilled)

For each example, prints:
  - Sentence
  - Gold triples
  - A triples, per-example PRF
  - B triples, per-example PRF
  - Deltas: kept, dropped_by_judge, added_by_judge

Usage (from server/):
  YAML_NOMINALS=on SPACY_MODEL_EN=en_core_web_trf YAML_DENSITY_CAPS=off \
  .venv/bin/python -m scripts.eval_extraction_diff \
    --dataset tests/data/yaml_eval_l1_en_medium.json \
    --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml \
    --lang en --judge_model models/graph_judge.json --limit 10
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

from core.memory.extractors.yaml_extractor import YAMLExtractor
from core.memory.eval_graph import prf1


def setenv(name: str, val: str | None) -> None:
    if val is None:
        if name in os.environ:
            del os.environ[name]
    else:
        os.environ[name] = val


def triples_set(trs: List[Tuple[str, str, str]]) -> set[Tuple[str, str, str]]:
    return set((s, r, d) for s, r, d in trs)


def per_example_prf(pred: List[Tuple[str, str, str]], gold: List[Tuple[str, str, str]]) -> Tuple[float, float, float]:
    m = prf1(pred, gold)
    return (m["precision"], m["recall"], m["f1"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--judge_model", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import json
    from pathlib import Path

    data = json.loads(Path(args.dataset).read_text())

    # Build extractor once; refine path reads env per call
    ext = YAMLExtractor(args.yaml)
    ext.prewarm(args.lang)

    agg_gold: List[Tuple[str, str, str]] = []
    agg_A: List[Tuple[str, str, str]] = []
    agg_B: List[Tuple[str, str, str]] = []

    printed = 0
    md_lines: List[str] = []
    for ex in data:
        text: str = ex["text"]
        gold: List[Tuple[str, str, str]] = [(s, r, d) for s, r, d in ex["gold"]]
        lang = ex.get("lang", args.lang)

        # A: judge OFF
        setenv("YAML_GRAPH_JUDGE", "off")
        ents, trA, negA, docA = ext.extract(text, lang)
        trA = ext.refine(text, trA, docA)

        # B: judge ON
        setenv("YAML_GRAPH_JUDGE", "on")
        if args.judge_model:
            setenv("YAML_GRAPH_JUDGE_MODEL", args.judge_model)
        ents, trB, negB, docB = ext.extract(text, lang)
        trB = ext.refine(text, trB, docB)

        # Accumulate
        agg_gold.extend(gold)
        agg_A.extend(trA)
        agg_B.extend(trB)

        # Pretty print limited examples
        if args.limit and printed >= args.limit:
            continue
        printed += 1

        pA, rA, fA = per_example_prf(trA, gold)
        pB, rB, fB = per_example_prf(trB, gold)
        setA, setB = triples_set(trA), triples_set(trB)
        kept = sorted(setA & setB)
        dropped = sorted(setA - setB)
        added = sorted(setB - setA)

        block = []
        block.append("\n=== Example ===")
        block.append(f"Text: {text}")
        block.append(f"Gold: {gold}")
        block.append(f"A (YAML raw)  [{pA:.3f}/{rA:.3f}/{fA:.3f}]: {trA}")
        block.append(f"B (YAML judge)[{pB:.3f}/{rB:.3f}/{fB:.3f}]: {trB}")
        block.append(f"Kept           : {kept}")
        block.append(f"Dropped by judge: {dropped}")
        if added:
            block.append(f"Added by judge : {added}")
        print("\n".join(block))
        if args.out:
            md_lines.append(f"\n#### Example\n")
            md_lines.append(f"- Text: {text}")
            md_lines.append(f"- Gold: {gold}")
            md_lines.append(f"- A (YAML raw)  [{pA:.3f}/{rA:.3f}/{fA:.3f}]: {trA}")
            md_lines.append(f"- B (YAML judge)[{pB:.3f}/{rB:.3f}/{fB:.3f}]: {trB}")
            md_lines.append(f"- Kept: {kept}")
            md_lines.append(f"- Dropped by judge: {dropped}")
            if added:
                md_lines.append(f"- Added by judge: {added}")

    # Aggregate summary
    mA = prf1(agg_A, agg_gold)
    mB = prf1(agg_B, agg_gold)
    summary = []
    summary.append("\n=== Aggregate (strict PRF) ===")
    summary.append(f"YAML raw   : P={mA['precision']:.3f} R={mA['recall']:.3f} F1={mA['f1']:.3f}")
    summary.append(f"YAML judge : P={mB['precision']:.3f} R={mB['recall']:.3f} F1={mB['f1']:.3f}")
    print("\n".join(summary))
    if args.out:
        from pathlib import Path
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        md = ["### Extraction Diff Report", *md_lines, "\n### Aggregate", f"- YAML raw: P={mA['precision']:.3f} R={mA['recall']:.3f} F1={mA['f1']:.3f}", f"- YAML judge: P={mB['precision']:.3f} R={mB['recall']:.3f} F1={mB['f1']:.3f}"]
        outp.write_text("\n".join(md))
        print(f"\nSaved Markdown report to {outp}")


if __name__ == "__main__":
    main()
