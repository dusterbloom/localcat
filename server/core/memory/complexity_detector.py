"""
Lightweight sentence complexity detector.

Purpose: provide a local, dependency-free signal to gate optional DSPy
extraction without spamming the logs when the optional package is absent.

Design goals (KISS, SOLID):
- Single responsibility: decide if a sentence is complex enough to warrant
  enhanced processing; expose simple metrics for observability.
- No heavy dependencies; operates on a spaCy Doc already computed upstream.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


class ComplexityDetector:
    """Heuristic, fast complexity detector.

    Signals "complex" for sentences likely to benefit from advanced extraction,
    such as those with multiple clauses, coordination, or deep dependency trees.
    """

    def __init__(self,
                 min_tokens: int = 18,
                 clause_weight: float = 0.6,
                 conj_weight: float = 0.3,
                 depth_weight: float = 0.1,
                 threshold: float = 0.8) -> None:
        self.min_tokens = int(min_tokens)
        self.clause_weight = float(clause_weight)
        self.conj_weight = float(conj_weight)
        self.depth_weight = float(depth_weight)
        self.threshold = float(threshold)

    def is_complex(self, doc: Any) -> Tuple[bool, Dict[str, float]]:
        """Return (is_complex, metrics) for a spaCy Doc.

        Metrics include:
        - tokens: token count
        - clauses: rough count based on clausal deps (ccomp, xcomp, advcl, acl, csubj)
        - conj: coordination count (and/or, comma-separated coordination)
        - depth: approximate dependency depth from heads
        - complexity_score: weighted composite in [0, ~2]
        """
        try:
            tokens = len(doc)
        except Exception:
            tokens = 0

        if tokens == 0:
            return False, {
                "tokens": 0,
                "clauses": 0.0,
                "conj": 0.0,
                "depth": 0.0,
                "complexity_score": 0.0,
            }

        # Clause-like dependencies
        clause_deps = {"ccomp", "xcomp", "advcl", "acl", "csubj"}
        clauses = 0
        conj = 0
        max_depth = 1

        for t in doc:
            # Rough depth via head chain length
            d = 1
            h = t.head
            # Stop excessive loops if trees are malformed
            steps = 0
            while h is not None and h != t and steps < 64:
                d += 1
                if getattr(h, "head", None) is None or h == h.head:
                    break
                h = h.head
                steps += 1
            if d > max_depth:
                max_depth = d

            if t.dep_ in clause_deps:
                clauses += 1
            if t.dep_ == "conj" or t.text.lower() in ("and", "or", ","):
                conj += 1

        # Normalize components
        clause_score = min(1.0, clauses / 2.0)  # ≥2 clauses → max
        conj_score = min(1.0, conj / 3.0)       # ≥3 coordinations → max
        depth_score = min(1.0, (max_depth - 1) / 6.0)  # depth ≥7 → max

        # Token bonus: long sentences are more likely complex
        token_bonus = 0.2 if tokens >= self.min_tokens else 0.0

        complexity = (
            clause_score * self.clause_weight
            + conj_score * self.conj_weight
            + depth_score * self.depth_weight
            + token_bonus
        )

        return complexity >= self.threshold, {
            "tokens": float(tokens),
            "clauses": float(clauses),
            "conj": float(conj),
            "depth": float(max_depth),
            "complexity_score": float(complexity),
        }

