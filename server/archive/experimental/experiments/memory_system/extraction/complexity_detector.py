"""
Sentence complexity detector for triggering enhanced extraction

Detects complex sentences that likely need LLM-based re-extraction
to capture all edges that spaCy dependency parsing might miss.
"""

from typing import Any


class ComplexityDetector:
    """Detect if a sentence is complex enough to need enhanced extraction"""

    def __init__(
        self,
        conjunction_threshold: int = 2,
        clause_threshold: int = 2,
        token_threshold: int = 15
    ):
        """
        Initialize complexity detector with thresholds

        Args:
            conjunction_threshold: Minimum coordinating conjunctions (and, or, but)
            clause_threshold: Minimum subordinate clauses (who, which, that)
            token_threshold: Minimum token count
        """
        self.conjunction_threshold = conjunction_threshold
        self.clause_threshold = clause_threshold
        self.token_threshold = token_threshold

    def is_complex(self, doc: Any) -> tuple[bool, dict]:
        """
        Determine if sentence is complex

        Args:
            doc: spaCy Doc object

        Returns:
            (is_complex, metrics_dict) where metrics contains:
            - conjunction_count: number of coordinating conjunctions
            - clause_count: number of subordinate clauses
            - token_count: total tokens
            - complexity_score: normalized score 0-1
        """
        # Count coordinating conjunctions (and, or, but)
        conjunction_count = sum(1 for token in doc if token.dep_ == "cc")

        # Count subordinate clauses and relative clauses
        clause_indicators = {"acl", "advcl", "ccomp", "relcl", "csubj", "xcomp"}
        clause_count = sum(1 for token in doc if token.dep_ in clause_indicators)

        # Count tokens (excluding punctuation)
        token_count = sum(1 for token in doc if not token.is_punct)

        # Count conjuncts (things joined by conjunctions)
        conj_count = sum(1 for token in doc if token.dep_ == "conj")

        # Count appositions (descriptive phrases like "Alice, a software engineer")
        appos_count = sum(1 for token in doc if token.dep_ == "appos")

        # Calculate complexity score (normalized 0-1)
        complexity_score = (
            (conjunction_count / max(self.conjunction_threshold, 1)) * 0.3 +
            (clause_count / max(self.clause_threshold, 1)) * 0.3 +
            (token_count / max(self.token_threshold, 1)) * 0.2 +
            (conj_count / 3) * 0.1 +
            (appos_count / 2) * 0.1
        )

        # Determine if complex based on thresholds
        is_complex = (
            conjunction_count >= self.conjunction_threshold or
            clause_count >= self.clause_threshold or
            token_count >= self.token_threshold or
            conj_count >= 3 or
            appos_count >= 2
        )

        metrics = {
            "conjunction_count": conjunction_count,
            "clause_count": clause_count,
            "token_count": token_count,
            "conj_count": conj_count,
            "appos_count": appos_count,
            "complexity_score": min(complexity_score, 1.0)
        }

        return is_complex, metrics

    def explain_complexity(self, metrics: dict) -> str:
        """Generate human-readable explanation of complexity"""
        reasons = []

        if metrics["conjunction_count"] >= self.conjunction_threshold:
            reasons.append(f"{metrics['conjunction_count']} conjunctions")

        if metrics["clause_count"] >= self.clause_threshold:
            reasons.append(f"{metrics['clause_count']} clauses")

        if metrics["token_count"] >= self.token_threshold:
            reasons.append(f"{metrics['token_count']} tokens")

        if metrics["conj_count"] >= 3:
            reasons.append(f"{metrics['conj_count']} conjuncts")

        if metrics["appos_count"] >= 2:
            reasons.append(f"{metrics['appos_count']} appositions")

        if reasons:
            return f"Complex: {', '.join(reasons)} (score={metrics['complexity_score']:.2f})"
        else:
            return f"Simple (score={metrics['complexity_score']:.2f})"


def detect_complexity(doc: Any, thresholds: dict = None) -> tuple[bool, dict]:
    """
    Convenience function for complexity detection

    Args:
        doc: spaCy Doc object
        thresholds: Optional dict of {conjunction_threshold, clause_threshold, token_threshold}

    Returns:
        (is_complex, metrics)
    """
    if thresholds is None:
        thresholds = {}

    detector = ComplexityDetector(**thresholds)
    return detector.is_complex(doc)