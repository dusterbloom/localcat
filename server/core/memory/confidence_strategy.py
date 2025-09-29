"""
Confidence scoring strategies for memory facts

This module provides pluggable confidence strategies following SOLID principles:
- Strategy Pattern: Different confidence algorithms can be swapped
- Dependency Inversion: HotMemory depends on ConfidenceStrategy interface, not implementations
- Open/Closed: New strategies can be added without modifying existing code
"""

import time
from typing import Protocol, Optional
from dataclasses import dataclass


@dataclass
class Edge:
    """Edge representation for confidence scoring"""
    src: str
    rel: str
    dst: str
    pos: int  # Positive reinforcements
    neg: int  # Negative evidence
    updated_at: int  # Timestamp in milliseconds
    id: str  # Edge ID


@dataclass
class Context:
    """Context for confidence scoring"""
    store: any  # MemoryStore instance
    text: Optional[str] = None
    session_id: Optional[str] = None
    turn_id: Optional[int] = None


class ConfidenceStrategy(Protocol):
    """
    Strategy interface for computing fact confidence

    All confidence strategies must implement this interface.
    """

    def score(self, edge: Edge, context: Context) -> float:
        """
        Compute confidence score for an edge

        Args:
            edge: The edge (fact) to score
            context: Additional context (store, text, etc.)

        Returns:
            Confidence score between 0.0 and 1.0
        """
        ...


class RelationTypeConfidence:
    """
    Baseline confidence strategy using static relation-type mapping

    This is the current implementation - assigns confidence based
    only on relation type with no learning.
    """

    def score(self, edge: Edge, context: Context) -> float:
        """
        Score confidence using static relation-type rules

        Args:
            edge: Edge to score
            context: Context (unused in baseline)

        Returns:
            Static confidence: 0.95 for names, 0.85 for verbs, 0.9 for others
        """
        if edge.rel == "name":
            return 0.95
        elif edge.rel.startswith("v:"):
            return 0.85
        else:
            return 0.9


class UsageBasedConfidence:
    """
    Learned confidence from usage patterns (structural signals)

    Uses database signals to adjust confidence:
    - Reinforcement: Facts mentioned multiple times boost confidence
    - Recency: Old facts decay over time
    - Source count: More conversation sources = higher confidence
    - Negation: Contradicted facts lose confidence

    No LLM required - pure structural learning.
    """

    def __init__(self, baseline: Optional[ConfidenceStrategy] = None):
        """
        Args:
            baseline: Base strategy for starting confidence (defaults to RelationTypeConfidence)
        """
        self.baseline = baseline or RelationTypeConfidence()

    def score(self, edge: Edge, context: Context) -> float:
        """
        Score confidence using structural signals from database

        Args:
            edge: Edge to score
            context: Context with store for querying

        Returns:
            Confidence adjusted by reinforcement, recency, and source count
        """
        # Start with relation-type baseline
        baseline_conf = self.baseline.score(edge, context)

        # Structural multipliers
        reinforcement = self._reinforcement_multiplier(edge)
        recency = self._recency_multiplier(edge)
        source_count = self._source_count_multiplier(edge, context)

        # Combine multiplicatively
        confidence = baseline_conf * reinforcement * recency * source_count

        return min(1.0, max(0.0, confidence))

    def _reinforcement_multiplier(self, edge: Edge) -> float:
        """
        Boost for validated facts, penalty for negated

        Args:
            edge: Edge with pos/neg counts

        Returns:
            Multiplier between 0.4 and 1.15
        """
        if edge.pos > 0 and edge.neg == 0:
            # Fact reinforced, never contradicted
            # Up to 15% boost for 3+ reinforcements
            return 1.0 + (0.05 * min(edge.pos, 3))
        elif edge.neg > 0:
            # Fact contradicted
            # Down to 40% penalty for 3+ negations
            return 0.7 - (0.1 * min(edge.neg, 3))
        return 1.0  # No evidence yet

    def _recency_multiplier(self, edge: Edge) -> float:
        """
        Decay old facts

        Args:
            edge: Edge with updated_at timestamp

        Returns:
            Multiplier between 0.8 and 1.0
        """
        age_days = (time.time() - edge.updated_at / 1000) / 86400

        if age_days > 90:
            return 0.8   # 20% penalty after 90 days
        elif age_days > 30:
            return 0.9   # 10% penalty after 30 days
        return 1.0

    def _source_count_multiplier(self, edge: Edge, context: Context) -> float:
        """
        Boost facts mentioned multiple times

        Args:
            edge: Edge to check
            context: Context with store for querying

        Returns:
            Multiplier between 1.0 and 1.1
        """
        if context.store is None:
            return 1.0

        # Query edge_source table for source count
        count = context.store.get_edge_sources_count(edge.id)

        if count >= 3:
            return 1.1   # 10% boost for 3+ mentions
        elif count >= 2:
            return 1.05  # 5% boost for 2 mentions
        return 1.0


def create_confidence_strategy(name: str = "relation_type") -> ConfidenceStrategy:
    """
    Factory function for creating confidence strategies

    Args:
        name: Strategy name ("relation_type", "usage_based")

    Returns:
        Confidence strategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    if name == "relation_type":
        return RelationTypeConfidence()
    elif name == "usage_based":
        return UsageBasedConfidence()
    else:
        raise ValueError(f"Unknown confidence strategy: {name}")