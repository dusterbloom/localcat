#!/usr/bin/env python3
"""
Eval (Dev) – OpenIE on CaRB-style data (upper-bound option)

This is a lightweight harness to compare our extraction against a CaRB-like
test set. It does NOT implement the official CaRB matching; it computes a
simple tuple-level precision/recall/F1 with partial credit for string overlap.

Usage:
  SPACY_MODEL_EN=en_core_web_trf YAML_DENSITY_CAPS=off \
  uv run --project server --directory server -m scripts.eval_openie_carb \
    --sentences data/carb/carb_test_sentences.txt \
    --gold data/carb/carb_test.tsv \
    --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml

Expected data (simplified):
  - sentences: one sentence per line
  - gold tsv: tab-separated triples per line: sent_idx \t arg1 \t rel \t arg2

This harness is meant for quick internal comparison. For official SOTA
comparison, use the CaRB scorer on the official release.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

from core.memory.extractors.yaml_extractor import YAMLExtractor


def load_sentences(path: Path) -> List[str]:
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]


def load_gold(path: Path) -> Dict[int, List[Tuple[str, str, str]]]:
    gold: Dict[int, List[Tuple[str, str, str]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 4:
            # sent_idx, arg1, rel, arg2
            continue
        idx = int(cols[0])
        arg1, rel, arg2 = cols[1].strip().lower(), cols[2].strip().lower(), cols[3].strip().lower()
        gold.setdefault(idx, []).append((arg1, rel, arg2))
    return gold


def norm(t: str) -> str:
    return (t or "").strip().lower()


def overlap(a: str, b: str) -> float:
    # Token-level Jaccard for rough partial credit
    ta = set(norm(a).split())
    tb = set(norm(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def tuple_match(pred: Tuple[str, str, str], golds: List[Tuple[str, str, str]]) -> float:
    # Score a predicted triple against a list of gold triples by average overlap
    s, r, d = pred
    best = 0.0
    for gs, gr, gd in golds:
        sc = 0.34 * overlap(s, gs) + 0.33 * overlap(r, gr) + 0.33 * overlap(d, gd)
        if sc > best:
            best = sc
    return best


def evaluate(preds: Dict[int, List[Tuple[str, str, str]]], gold: Dict[int, List[Tuple[str, str, str]]]) -> Tuple[float, float, float]:
    # Micro-averaged PRF with soft matching
    tp = 0.0
    total_pred = 0
    total_gold = 0

    for i, gtriples in gold.items():
        ptriples = preds.get(i, [])
        total_pred += len(ptriples)
        total_gold += len(gtriples)
        for pt in ptriples:
            tp += tuple_match(pt, gtriples)

    precision = (tp / total_pred) if total_pred else 0.0
    recall = (tp / total_gold) if total_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--yaml", required=True)
    args = ap.parse_args()

    sents = load_sentences(Path(args.sentences))
    gold = load_gold(Path(args.gold))

    ext = YAMLExtractor(args.yaml)
    preds: Dict[int, List[Tuple[str, str, str]]] = {}

    for i, sent in enumerate(sents):
        ents, triples, neg, doc = ext.extract(sent, lang="en")
        triples = ext.refine(sent, triples, doc)
        preds[i] = triples

    p, r, f1 = evaluate(preds, gold)
    print("\n=== CaRB-style (soft) ===")
    print(f"Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()

