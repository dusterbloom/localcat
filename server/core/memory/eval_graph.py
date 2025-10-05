"""
Graph extraction evaluation utilities.

Metrics:
- Edge Precision/Recall/F1 on (s, r, d) triples after canonicalization.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Iterable
from .canonicalize import canonicalize_triple


def _canon(s: str) -> str:
    t = (s or "").strip().lower()
    for det in ("the", "a", "an", "my", "your", "his", "her", "our", "their", "its"):
        if t.startswith(det + " "):
            t = t[len(det) + 1 :]
            break
    if t.endswith("'s"):
        t = t[:-2]
    if t in {"i", "me", "my", "mine", "myself"}:
        return "you"
    return t


def _norm_triples(triples: Iterable[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    out = []
    for s, r, d in triples:
        s2, r2, d2 = _canon(s), (r or "").strip().lower(), _canon(d)
        # Apply canonicalization (verb+prep folding, copula unification)
        s3, r3, d3 = canonicalize_triple(s2, r2, d2)
        out.append((s3, r3, d3))
    return out


def prf1(pred: Iterable[Tuple[str, str, str]], gold: Iterable[Tuple[str, str, str]]) -> Dict[str, float]:
    pset = set(_norm_triples(pred))
    gset = set(_norm_triples(gold))
    tp = len(pset & gset)
    fp = len(pset - gset)
    fn = len(gset - pset)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
