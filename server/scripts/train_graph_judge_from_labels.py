#!/usr/bin/env python3
"""
Train a distilled GraphJudge model from LLM-labeled gray-zone records.

Inputs:
  --labels: JSONL from llm_judge_labeler.py with {text, triples, labels}
  --yaml: YAML index path used by YAMLExtractor (for doc/NER features)
  --out: output JSON model with {intercept, weights{}, threshold}
  --thresh: decision threshold (default 0.5) or --keep_rate to set threshold by positive rate

Example:
  python -m scripts.train_graph_judge_from_labels \
    --labels data/judge_labels.jsonl \
    --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml \
    --out models/graph_judge.json --keep_rate 0.35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from core.memory.extractors.yaml_extractor import YAMLExtractor
from core.memory.judge import build_features


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--keep_rate", type=float, default=0.0, help="Target fraction to keep; sets threshold to match approx keep rate if > 0")
    args = ap.parse_args()

    lbls = load_jsonl(Path(args.labels))
    if not lbls:
        raise SystemExit(f"No label records at {args.labels}")

    ext = YAMLExtractor(args.yaml)

    feats_order: List[str] = []
    X_rows: List[List[float]] = []
    y_rows: List[int] = []

    for rec in lbls:
        text = rec.get("text", "")
        triples = rec.get("triples", []) or rec.get("candidates", [])
        labels = rec.get("labels", [])
        if not text or not triples or not labels or len(triples) != len(labels):
            continue
        # Build doc for NER hints
        _, _, _, doc = ext.extract(text, args.lang)
        for (s, r, d), lab in zip(triples, labels):
            try:
                f = build_features(s, r, d, doc)
            except Exception:
                continue
            if not feats_order:
                feats_order = list(f.keys())
            X_rows.append([float(f[k]) for k in feats_order])
            y_rows.append(1 if int(lab) == 1 else 0)

    if not X_rows:
        raise SystemExit("No trainable records after parsing labels.")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X, y)

    weights = {feat: float(coef) for feat, coef in zip(feats_order, clf.coef_[0])}
    intercept = float(clf.intercept_[0])

    thresh = float(args.thresh)
    if args.keep_rate > 0.0:
        probs = clf.predict_proba(X)[:, 1]
        # pick threshold closest to desired keep rate
        sorted_probs = sorted(probs, reverse=True)
        k = max(1, min(len(sorted_probs), int(args.keep_rate * len(sorted_probs))))
        thresh = float(sorted_probs[k - 1])

    out = {"intercept": intercept, "weights": weights, "threshold": thresh}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Saved distilled judge to {args.out} with threshold={thresh:.3f}")


if __name__ == "__main__":
    main()

