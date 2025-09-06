"""
Complexity-aware confidence scoring helpers for extracted triples.

These functions are optional and only used when HOTMEM_EXTRA_CONFIDENCE=true.
They should be fast and conservative to avoid regressions on the hot path.
"""
from typing import Tuple

HEDGE_WORDS = {
    "maybe", "probably", "perhaps", "possibly", "i think", "i guess",
    "kinda", "sort of", "not sure", "i'm not sure"
}


def _doc_complexity_penalty(doc) -> float:
    """Return a penalty [0..0.4] based on structural complexity."""
    try:
        sents = list(getattr(doc, 'sents', []))
        multi_sent = 0.1 if len(sents) > 1 else 0.0
        has_sub = any(getattr(t, 'dep_', '') in {"mark", "advcl", "ccomp", "xcomp"} for t in doc)
        sub_penalty = 0.15 if has_sub else 0.0
        has_hedge = any((getattr(t, 'text', '') or '').lower() in HEDGE_WORDS for t in doc)
        hedge_penalty = 0.15 if has_hedge else 0.0
        return min(0.4, multi_sent + sub_penalty + hedge_penalty)
    except Exception:
        return 0.0


def calculate_extraction_confidence(doc, triple: Tuple[str, str, str]) -> float:
    """Return [0..1] confidence for an extracted triple given doc context."""
    try:
        base = 0.75
        s, r, d = triple
        # Simple clarity bonuses for canonical relations
        high_rel = {"name", "age", "lives_in", "works_at", "favorite_color"}
        if r in high_rel:
            base += 0.05
        # Apply complexity penalties
        pen = _doc_complexity_penalty(doc)
        conf = max(0.3, min(1.0, base - pen))
        return conf
    except Exception:
        return 0.6

