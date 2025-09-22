"""
Memory Orchestrator - Coordinates memory processing components.

This component orchestrates the overall memory processing flow by coordinating
the FactExtractor, MemoryRetriever, and ContextFormatter components. It follows
the Single Responsibility Principle by focusing solely on orchestration.
"""

import time
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from loguru import logger

from .fact_extractor import FactExtractor
from .memory_retriever import MemoryRetriever
from .context_formatter import ContextFormatter
from .memory_store import MemoryStore


class MemoryOrchestrator:
    """
    Orchestrates memory processing by coordinating specialized components.
    Handles the overall flow: extract → retrieve → format → store.
    """

    def __init__(self, store: MemoryStore, max_recency: int = 50):
        self.store = store
        self.max_recency = max_recency

        # Initialize components
        self.extractor = FactExtractor()
        self.retriever = MemoryRetriever(store, max_recency)
        self.formatter = ContextFormatter()

        # Performance tracking
        self.metrics = defaultdict(list)
        self.max_metric_size = 1000

        # User ID (for compatibility)
        self.user_eid = "you"

    def prewarm(self, lang: str = "en") -> None:
        """Pre-load resources for all components."""
        try:
            self.extractor.prewarm(lang)
            logger.debug("Memory orchestrator prewarmed")
        except Exception as e:
            logger.warning(f"Failed to prewarm memory orchestrator: {e}")

    def process_turn(self, text: str, session_id: str, turn_id: int) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Process a conversation turn end-to-end.

        Returns:
            (memory_bullets, extracted_triples)
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Extract facts from text
            entities, raw_triples, neg_count, doc = self.extractor.extract(text)

            # Step 2: Refine extracted triples
            refined_triples = self.extractor.refine(text, raw_triples, doc)

            # Step 3: Refine entities
            entities = self.extractor.refine_entities(text, entities)

            # Step 4: Filter meaningful triples
            meaningful_triples = self._filter_meaningful_triples(refined_triples)

            # Step 5: Retrieve relevant memories
            memory_bullets = self.retriever.retrieve_bullets(text, entities, turn_id)

            # Step 6: Update storage and indices (if not a question)
            if not self._is_question(text):
                self._update_memory(meaningful_triples, text, turn_id)

            # Track performance
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.metrics['total_ms'].append(elapsed_ms)
            self._cleanup_metrics()

            if elapsed_ms > 200:
                logger.warning(f"Memory processing took {elapsed_ms:.1f}ms (budget: 200ms)")

            return memory_bullets, meaningful_triples

        except Exception as e:
            logger.error(f"Memory processing failed: {e}")
            # Return empty results on failure
            return [], []

    def _filter_meaningful_triples(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Filter triples to only meaningful facts."""
        meaningful = []
        for triple in triples:
            if self.extractor._is_meaningful_fact(*triple):
                meaningful.append(triple)
        return meaningful

    def _update_memory(self, triples: List[Tuple[str, str, str]], text: str, turn_id: int) -> None:
        """Update memory storage and indices with new triples."""
        try:
            now_ts = int(time.time() * 1000)

            # Update persistent storage
            for s, r, d in triples:
                # Determine confidence based on relation type
                if r == "name":
                    conf = 0.95
                elif r.startswith("v:"):
                    conf = 0.85
                else:
                    conf = 0.9

                # Store the triple
                self.store.observe_edge(s, r, d, conf, now_ts)

            # Update retriever indices
            self.retriever.update_indices(triples, text, turn_id)

            # Flush to ensure persistence
            self.store.flush_if_needed()

        except Exception as e:
            logger.warning(f"Memory update failed: {e}")

    def _is_question(self, text: str) -> bool:
        """Simple heuristic to detect questions."""
        question_words = {"what", "when", "where", "why", "how", "who", "which", "whose", "whom"}
        first_word = text.strip().split()[0].lower().rstrip("?,.;:!")
        return first_word in question_words or text.strip().endswith("?")

    def rebuild_from_store(self) -> None:
        """Rebuild indices from persistent storage."""
        try:
            self.retriever.rebuild_from_store()
        except Exception as e:
            logger.warning(f"Could not rebuild from store: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get combined performance metrics from all components."""
        import statistics

        # Get metrics from all components
        all_metrics = {
            'orchestrator': self._calculate_metrics(self.metrics),
            'extractor': self.extractor.get_metrics(),
            'retriever': self.retriever.get_metrics(),
        }

        return all_metrics

    def _calculate_metrics(self, metrics_dict: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate statistics for a metrics dictionary."""
        import statistics

        result = {}
        for key, values in metrics_dict.items():
            if values:
                result[key] = {
                    'mean': statistics.mean(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                    'count': len(values)
                }
            else:
                result[key] = {'count': 0}
        return result

    def _cleanup_metrics(self):
        """Clean up old metrics to prevent memory bloat."""
        for key in self.metrics:
            if len(self.metrics[key]) > self.max_metric_size:
                self.metrics[key] = self.metrics[key][-self.max_metric_size:]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            'orchestrator_metrics': self.get_metrics(),
            'store_metrics': self.store.get_metrics() if hasattr(self.store, 'get_metrics') else {},
            'retriever_stats': {
                'entity_index_size': len(self.retriever.entity_index),
                'recency_buffer_size': len(self.retriever.recency_buffer)
            }
        }