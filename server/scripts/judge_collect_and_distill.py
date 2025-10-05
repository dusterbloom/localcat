#!/usr/bin/env python3
"""
Collect gray-zone judge records and distill a new logistic model.

Typical usage (from server/):
  # Supervised using a labeled dataset to auto-label gray-zone examples
  .venv/bin/python -m scripts.judge_collect_and_distill \
      --log data/judge_grayzone.jsonl \
      --dataset tests/data/yaml_eval_l1_en_medium.json \
      --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml \
      --out models/graph_judge.json --auto_calibrate

If --dataset is omitted, the script will export a feature matrix and exit with a note
that labels are required (e.g., via an LLM‑Judge pass) before training.
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


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text())


def prf1(pred: List[Tuple[str, str, str]], gold: List[Tuple[str, str, str]]) -> Dict[str, float]:
    P = set(pred)
    G = set(gold)
    tp = len(P & G)
    fp = max(0, len(P) - tp)
    fn = max(0, len(G) - tp)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--auto_calibrate", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    log_path = Path(args.log)
    rows = load_jsonl(log_path)
    if not rows:
        raise SystemExit(f"No gray-zone records found at {log_path}")

    ds_map: Dict[str, List[Tuple[str, str, str]]] = {}
    if args.dataset:
        data = load_dataset(Path(args.dataset))
        for ex in data:
            text = ex["text"]
            gold = [(s, r, d) for s, r, d in ex["gold"]]
            ds_map[text] = gold

    ext = YAMLExtractor(args.yaml)

    feats_order: List[str] = []
    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    pred_all: List[Tuple[str, str, str]] = []
    gold_all: List[Tuple[str, str, str]] = []

    for rec in rows:
        text = rec.get("text", "")
        triple = rec.get("triple", ["", "", ""])  # [s, r, d]
        if not text or not triple or len(triple) != 3:
            continue
        s, r, d = triple
        # Build doc for NER hints
        _, _, _, doc = ext.extract(text, args.lang)
        f = build_features(s, r, d, doc)
        if not feats_order:
            feats_order = list(f.keys())
        X_rows.append([float(f[k]) for k in feats_order])
        pred_all.append((s, r, d))
        # Label if dataset provided
        if args.dataset and text in ds_map:
            G = set(ds_map[text])
            y_rows.append(1 if (s, r, d) in G else 0)
            gold_all.extend(ds_map[text])

    if not args.dataset:
        out_mat = Path(args.out).with_suffix(".features.json")
        out_mat.write_text(json.dumps({"features": feats_order, "X": X_rows, "triples": pred_all}, indent=2))
        print(f"Exported features to {out_mat}; supply labels to train a model.")
        return

    if not X_rows or not y_rows:
        raise SystemExit("Insufficient labeled training examples. Provide a dataset matching your logs.")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X, y)

    weights = {feat: float(coef) for feat, coef in zip(feats_order, clf.coef_[0])}
    intercept = float(clf.intercept_[0])

    thresh = args.thresh
    if args.auto_calibrate:
        probs = clf.predict_proba(X)[:, 1]
        best_f1 = -1.0
        best_t = thresh
        for t in np.linspace(0.2, 0.8, 25):
            keep = [pred for pred, p in zip(pred_all, probs) if p >= t]
            m = prf1(keep, gold_all)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_t = float(t)
        thresh = best_t

    out = {"intercept": intercept, "weights": weights, "threshold": thresh}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Saved distilled judge to {args.out} with threshold={thresh:.3f}")


if __name__ == "__main__":
    main()

