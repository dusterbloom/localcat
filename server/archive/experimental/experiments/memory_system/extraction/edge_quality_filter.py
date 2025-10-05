"""
Simple edge quality filter for DSPy-extracted edges

Note: This is a lightweight filter for real-time use. Full GEPA-based
offline refinement will be implemented separately as per gepa_graph_refinement_architecture.md
"""

from typing import List, Tuple, Set
from loguru import logger


class EdgeQualityFilter:
    """Filter low-quality edges from DSPy extraction"""

    def __init__(
        self,
        min_entity_length: int = 2,
        max_entity_length: int = 100,
        blacklist_relations: Set[str] = None
    ):
        """
        Initialize quality filter

        Args:
            min_entity_length: Minimum character length for entities
            max_entity_length: Maximum character length for entities
            blacklist_relations: Relations to filter out
        """
        self.min_entity_length = min_entity_length
        self.max_entity_length = max_entity_length
        self.blacklist_relations = blacklist_relations or {
            "is", "has", "does", "will", "can", "should"  # Too generic
        }

    def filter_edges(
        self,
        edges: List[Tuple[str, str, str]],
        existing_edges: List[Tuple[str, str, str]] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Filter low-quality edges

        Args:
            edges: Edges to filter
            existing_edges: Already extracted edges (to avoid duplicates)

        Returns:
            Filtered list of high-quality edges
        """
        filtered = []
        existing_set = set(existing_edges) if existing_edges else set()

        for src, rel, dst in edges:
            # Skip if already exists
            if (src, rel, dst) in existing_set:
                logger.debug(f"Skipping duplicate: ({src}, {rel}, {dst})")
                continue

            # Normalize
            src = src.strip().lower()
            rel = rel.strip().lower().replace(" ", "_")
            dst = dst.strip().lower()

            # Filter by entity length
            if len(src) < self.min_entity_length or len(src) > self.max_entity_length:
                logger.debug(f"Filtering src too short/long: {src}")
                continue

            if len(dst) < self.min_entity_length or len(dst) > self.max_entity_length:
                logger.debug(f"Filtering dst too short/long: {dst}")
                continue

            # Filter blacklisted relations
            if rel in self.blacklist_relations:
                logger.debug(f"Filtering blacklisted relation: {rel}")
                continue

            # Filter empty
            if not src or not rel or not dst:
                logger.debug("Filtering empty edge")
                continue

            # Filter if src == dst (self-loops usually errors)
            if src == dst:
                logger.debug(f"Filtering self-loop: {src}")
                continue

            # Passed all filters
            filtered.append((src, rel, dst))

        logger.debug(f"Edge quality filter: {len(edges)} → {len(filtered)} edges")
        return filtered


def filter_edges(
    edges: List[Tuple[str, str, str]],
    existing_edges: List[Tuple[str, str, str]] = None,
    **kwargs
) -> List[Tuple[str, str, str]]:
    """
    Convenience function for edge filtering

    Args:
        edges: Edges to filter
        existing_edges: Already extracted edges
        **kwargs: Optional filter configuration

    Returns:
        Filtered edges
    """
    filter_obj = EdgeQualityFilter(**kwargs)
    return filter_obj.filter_edges(edges, existing_edges)