#!/usr/bin/env python3
"""
Train a distilled GraphJudge (logistic regression) on YAML extractor outputs vs gold.

Inputs:
  --dataset: JSON file with [{"text": str, "gold": [[s,r,d], ...], "lang": "en"?}]
  --yaml: YAML index path used by YAMLExtractor
  --out: output JSON file for model weights {intercept: float, weights: {feat: weight}, threshold: float}
  --thresh: decision threshold (default 0.5); if --auto-calibrate, pick threshold maximizing F1 on dev

The model uses the same feature function as YAMLExtractor GraphJudge‑lite.
Exports a tiny JSON that runtime uses for a fast dot‑product + sigmoid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from core.memory.extractors.yaml_extractor import YAMLExtractor


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text())


def features(s: str, r: str, d: str, doc: Any) -> Dict[str, float]:
    r = (r or "").lower(); d = (d or "").lower()
    feats: Dict[str, float] = {
        "bias": 1.0,
        "lexicalized": 1.0 if any(r.endswith(suf) for suf in ("_on", "_in", "_to", "_from", "_with", "_for", "_into")) else 0.0,
        "len_d": min(len(d) / 32.0, 1.5),
        "very_short_d": 1.0 if 0 < len(d) <= 3 else 0.0,
        "empty_d": 1.0 if len(d) == 0 else 0.0,
        "rel_strong": 1.0 if r in {"work_on", "focus_on", "agree_on", "agree_with", "agree_to", "result_in", "stem_from", "lead_to", "apply_to", "apply_for", "comply_with", "adhere_to", "engage_in", "engage_with", "enter_into", "consist_of", "consist_in"} else 0.0,
    }
    generic_objs = {"thing", "things", "stuff", "issue", "issues", "something", "anything", "everything"}
    feats["generic_head"] = 1.0 if any(w in generic_objs for w in d.split()) else 0.0
    feats["pron_d"] = 1.0 if d in {"it", "this", "that", "there", "something", "anything"} else 0.0
    has_loc = False; has_org = False
    try:
        for ent in getattr(doc, "ents", []) or []:
            txt = (getattr(ent, "text", "") or "").lower()
            if txt and txt in d:
                if getattr(ent, "label_", "") in {"GPE", "LOC", "FAC"}:
                    has_loc = True
                if getattr(ent, "label_", "") in {"ORG", "NORP"}:
                    has_org = True
    except Exception:
        pass
    feats["type_loc_match"] = 1.0 if ((r in {"live_in", "arrive_at", "arrive_in"}) and has_loc) else 0.0
    feats["type_org_match"] = 1.0 if ((r in {"work_at", "present_to"}) and has_org) else 0.0
    return feats


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--auto_calibrate", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    data = load_dataset(Path(args.dataset))
    ext = YAMLExtractor(args.yaml)

    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    feats_order: List[str] = []
    gold_all: List[Tuple[str, str, str]] = []
    pred_all: List[Tuple[str, str, str]] = []

    # Collect features and labels by matching extractor preds to gold
    for ex in data:
        text = ex["text"]
        lang = ex.get("lang", args.lang)
        gold = [(s, r, d) for s, r, d in ex["gold"]]
        _, triples, _, doc = ext.extract(text, lang)
        triples = ext.refine(text, triples, doc)
        G = set(gold)
        for s, r, d in triples:
            f = features(s, r, d, doc)
            if not feats_order:
                feats_order = list(f.keys())
            X_rows.append([float(f[k]) for k in feats_order])
            y_rows.append(1 if (s, r, d) in G else 0)
        gold_all.extend(gold)
        pred_all.extend(triples)

    if not X_rows:
        raise SystemExit("No training examples collected. Check dataset and extractor output.")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)

    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X, y)

    weights = {feat: float(coef) for feat, coef in zip(feats_order, clf.coef_[0])}
    intercept = float(clf.intercept_[0])

    thresh = args.thresh
    if args.auto_calibrate:
        # Simple sweep to maximize F1 on the same set (or split if needed)
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

