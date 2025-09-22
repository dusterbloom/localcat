"""
Memory Retriever - Focused component for retrieving relevant memories.

This component handles the retrieval of relevant memory bullets based on
current context, entities, and conversation history. It follows the
Single Responsibility Principle by focusing solely on memory retrieval.
"""

import time
from typing import List, Tuple, Set, Dict, Optional, Any, Deque
from collections import defaultdict, deque
import heapq
from dataclasses import dataclass

from loguru import logger


@dataclass
class RecencyItem:
    """Item in recency buffer"""
    s: str  # subject
    r: str  # relation
    d: str  # destination
    text: str  # original text for context
    timestamp: int
    turn_id: int
    score: float = 1.0


class MemoryRetriever:
    """
    Focused component for retrieving relevant memories from storage
    and recency buffers. Handles entity-based lookup and context ranking.
    """

    def __init__(self, store: Any, max_recency: int = 50):
        self.store = store
        self.max_recency = max_recency

        # Hot indices (RAM)
        self.entity_index = defaultdict(set)  # entity -> set of (s,r,d) triples
        self.recency_buffer: Deque[RecencyItem] = deque(maxlen=max_recency)

        # Performance tracking
        self.metrics = defaultdict(list)
        self.max_metric_size = 1000

    def rebuild_from_store(self) -> None:
        """Rebuild RAM indices from persistent storage."""
        try:
            logger.debug("Rebuilding memory indices from store...")
            start_time = time.perf_counter()

            # Clear existing indices
            self.entity_index.clear()

            # Rebuild from all stored triples
            # This is a simplified version - in practice you'd iterate through stored data
            # For now, we'll start fresh and build indices as we go

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Memory indices rebuilt in {elapsed_ms:.1f}ms")

        except Exception as e:
            logger.warning(f"Could not rebuild from store: {e}")

    def retrieve_bullets(self, text: str, entities: List[str], turn_id: int, read_only: bool = False) -> List[str]:
        """
        Retrieve relevant memory bullets for the given context.

        Args:
            text: Current conversation text
            entities: Extracted entities from current text
            turn_id: Current turn ID
            read_only: If True, don't update recency scores

        Returns:
            List of formatted memory bullets
        """
        start_time = time.perf_counter()

        # Get relevant memories from different sources
        memories = []

        # 1. Entity-based retrieval from hot index
        entity_memories = self._retrieve_entity_memories(entities)
        memories.extend(entity_memories)

        # 2. Recency-based retrieval
        recency_memories = self._retrieve_recency_memories(text, entities)
        memories.extend(recency_memories)

        # 3. Graph-based retrieval (from persistent store)
        graph_memories = self._retrieve_graph_memories(entities)
        memories.extend(graph_memories)

        # Deduplicate and rank memories
        ranked_memories = self._rank_and_deduplicate(memories, text, entities)

        # Format as bullets
        bullets = self._format_as_bullets(ranked_memories)

        # Track performance
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.metrics['retrieval_ms'].append(elapsed_ms)
        self._cleanup_metrics()

        return bullets

    def update_indices(self, triples: List[Tuple[str, str, str]], text: str, turn_id: int) -> None:
        """
        Update memory indices with new triples.

        Args:
            triples: New triples to index
            text: Original text context
            turn_id: Current turn ID
        """
        now_ts = int(time.time() * 1000)

        # Update entity index
        for s, r, d in triples:
            self.entity_index[s].add((s, r, d))
            self.entity_index[d].add((s, r, d))

        # Update recency buffer
        for s, r, d in triples:
            self.recency_buffer.append(RecencyItem(s, r, d, text, now_ts, turn_id))

    def _retrieve_entity_memories(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Retrieve memories based on entity matches in hot index."""
        memories = []

        for entity in entities:
            if entity in self.entity_index:
                triples = list(self.entity_index[entity])
                for triple in triples:
                    memories.append({
                        'triple': triple,
                        'source': 'entity_index',
                        'relevance': 0.8,  # High relevance for direct entity matches
                        'recency': 0.5
                    })

        return memories

    def _retrieve_recency_memories(self, text: str, entities: List[str]) -> List[Dict[str, Any]]:
        """Retrieve recent memories that might be relevant."""
        memories = []

        # Look at recent triples for entity matches
        entity_set = set(entities)
        for item in self.recency_buffer:
            # Check if recency item involves any of our entities
            if item.s in entity_set or item.d in entity_set:
                memories.append({
                    'triple': (item.s, item.r, item.d),
                    'text': item.text,
                    'source': 'recency',
                    'relevance': 0.6,
                    'recency': self._calculate_recency_score(item.timestamp)
                })

        return memories

    def _retrieve_graph_memories(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Retrieve memories from persistent graph storage."""
        memories = []

        try:
            # Query persistent storage for each entity
            for entity in entities:
                # This is a simplified version - in practice you'd use the store's query methods
                # For now, we'll return empty results as the store interface may vary
                pass

        except Exception as e:
            logger.debug(f"Graph retrieval failed: {e}")

        return memories

    def _calculate_recency_score(self, timestamp: int) -> float:
        """Calculate recency score based on timestamp (0.0 = old, 1.0 = very recent)."""
        now = int(time.time() * 1000)
        age_ms = now - timestamp

        # Exponential decay: half-life of 5 minutes
        half_life_ms = 5 * 60 * 1000
        score = 2 ** (-age_ms / half_life_ms)

        return max(0.0, min(1.0, score))

    def _rank_and_deduplicate(self, memories: List[Dict[str, Any]], text: str, entities: List[str]) -> List[Dict[str, Any]]:
        """
        Rank memories by relevance and recency, removing duplicates.
        """
        # Remove duplicates based on triple
        seen_triples = set()
        unique_memories = []

        for memory in memories:
            triple = memory['triple']
            if triple not in seen_triples:
                seen_triples.add(triple)
                unique_memories.append(memory)

        # Rank by combined score
        for memory in unique_memories:
            relevance = memory.get('relevance', 0.5)
            recency = memory.get('recency', 0.5)

            # Combined score with weights
            memory['score'] = (relevance * 0.7) + (recency * 0.3)

        # Sort by score descending
        unique_memories.sort(key=lambda x: x['score'], reverse=True)

        return unique_memories

    def _format_as_bullets(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Format ranked memories as human-readable bullets."""
        bullets = []

        for memory in memories:
            triple = memory['triple']
            s, r, d = triple

            # Format based on relation type
            if r == "name":
                bullet = f"{s} is named {d}"
            elif r.startswith("v:"):  # verb relation
                verb = r[2:]  # Remove "v:" prefix
                bullet = f"{s} {verb} {d}"
            elif r == "is":
                bullet = f"{s} is {d}"
            elif r == "has":
                bullet = f"{s} has {d}"
            elif r == "belongs_to":
                bullet = f"{s} belongs to {d}"
            else:
                bullet = f"{s} {r} {d}"

            # Add context if available
            if 'text' in memory:
                # Truncate context for brevity
                context = memory['text'][:50]
                if len(memory['text']) > 50:
                    context += "..."
                bullet += f" (from: {context})"

            bullets.append(bullet)

        return bullets

    def _cleanup_metrics(self):
        """Clean up old metrics to prevent memory bloat."""
        for key in self.metrics:
            if len(self.metrics[key]) > self.max_metric_size:
                self.metrics[key] = self.metrics[key][-self.max_metric_size:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        import statistics

        result = {}
        for key, values in self.metrics.items():
            if values:
                result[key] = {
                    'mean': statistics.mean(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                    'count': len(values)
                }
            else:
                result[key] = {'count': 0}
        return result