"""
MiniLM Semantic Similarity for Enhanced Entity Linking

Enhances entity linking and semantic understanding using MiniLM sentence embeddings.
Designed to improve semantic noise resistance and multi-hop reasoning.

Usage:
    # Enable via environment variable
    os.environ['USE_MINILM'] = 'true'

    # Use for semantic similarity
    minilm = MiniLMSimilarity()
    similarity_score = minilm.compute_similarity(text1, text2)
"""

import os
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available - MiniLM disabled")

@dataclass
class SimilarityResult:
    """Result of semantic similarity computation."""
    similarity_score: float
    text1_embedding: np.ndarray
    text2_embedding: np.ndarray
    computation_time_ms: float


class MiniLMSimilarity:
    """MiniLM-based semantic similarity for enhanced understanding."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize MiniLM similarity model.

        Args:
            model_name: Sentence Transformers model name
        """
        self.model_name = model_name
        self.enabled = SENTENCE_TRANSFORMERS_AVAILABLE and os.getenv('USE_MINILM', 'false').lower() in ('1', 'true', 'yes')
        self._model = None
        self._load_time_ms = 0

        if self.enabled:
            self._initialize_model()
        else:
            logger.info("🧠 MiniLM disabled (USE_MINILM=false or sentence-transformers not available)")

    def _initialize_model(self):
        """Initialize the MiniLM model with timing."""
        logger.info(f"🧠 Initializing MiniLM with model: {self.model_name}")
        start_time = time.perf_counter()

        try:
            # Initialize the sentence transformer
            self._model = SentenceTransformer(self.model_name)

            self._load_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"✅ MiniLM initialized in {self._load_time_ms:.1f}ms")

        except Exception as e:
            logger.error(f"❌ Failed to initialize MiniLM: {e}")
            self.enabled = False

    def compute_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            SimilarityResult with similarity score and embeddings
        """
        if not self.enabled or not text1 or not text2:
            return SimilarityResult(0.0, np.array([]), np.array([]), 0.0)

        start_time = time.perf_counter()

        try:
            # Generate embeddings
            embedding1 = self._model.encode(text1, convert_to_numpy=True)
            embedding2 = self._model.encode(text2, convert_to_numpy=True)

            # Compute cosine similarity
            similarity = self._model.similarity(embedding1, embedding2)[0][0]
            computation_time_ms = (time.perf_counter() - start_time) * 1000

            result = SimilarityResult(
                similarity_score=float(similarity),
                text1_embedding=embedding1,
                text2_embedding=embedding2,
                computation_time_ms=computation_time_ms
            )

            logger.debug(f"🧠 MiniLM similarity: {similarity:.3f} in {computation_time_ms:.1f}ms")
            return result

        except Exception as e:
            logger.error(f"❌ MiniLM similarity computation failed: {e}")
            return SimilarityResult(0.0, np.array([]), np.array([]), 0.0)

    def find_similar_entities(self, query_entity: str, candidate_entities: List[str],
                             threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find entities similar to a query entity.

        Args:
            query_entity: Entity to find matches for
            candidate_entities: List of candidate entities
            threshold: Similarity threshold for matches

        Returns:
            List of (entity, similarity_score) tuples above threshold
        """
        if not self.enabled or not candidate_entities:
            return []

        # Encode query entity once
        query_embedding = self._model.encode(query_entity, convert_to_numpy=True)

        # Encode all candidates
        candidate_embeddings = self._model.encode(candidate_entities, convert_to_numpy=True)

        # Compute similarities
        similarities = self._model.similarity(query_embedding, candidate_embeddings)[0]

        # Filter by threshold and sort
        results = []
        for i, (candidate, similarity) in enumerate(zip(candidate_entities, similarities)):
            if similarity >= threshold:
                results.append((candidate, float(similarity)))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"🧠 Found {len(results)} similar entities to '{query_entity}'")
        return results

    def resolve_coreference(self, pronoun: str, candidate_entities: List[str],
                           context: str = "") -> Optional[str]:
        """
        Resolve pronoun coreference using semantic similarity.

        Args:
            pronoun: Pronoun to resolve (e.g., "he", "she", "it")
            candidate_entities: List of potential referent entities
            context: Additional context for resolution

        Returns:
            Best matching entity or None
        """
        if not self.enabled or not candidate_entities:
            return None

        # Create context-enhanced queries
        pronoun_with_context = f"{context} {pronoun}".strip()

        best_match = None
        best_score = 0.0

        for entity in candidate_entities:
            result = self.compute_similarity(pronoun_with_context, entity)
            if result.similarity_score > best_score:
                best_score = result.similarity_score
                best_match = entity

        # Only return if above reasonable threshold
        if best_score > 0.5:
            logger.debug(f"🧠 Coreference: '{pronoun}' → '{best_match}' (score: {best_score:.3f})")
            return best_match

        return None

    def detect_semantic_noise(self, user_fact: str, mentioned_topics: List[str]) -> bool:
        """
        Detect if mentioned topics are semantic noise vs actual facts.

        Args:
            user_fact: Fact about the user (e.g., "I work at OpenAI")
            mentioned_topics: Topics mentioned in conversation (e.g., ["Google", "Anthropic"])

        Returns:
            True if topics appear to be semantic noise
        """
        if not self.enabled or not mentioned_topics:
            return False

        # Extract the user entity from the fact
        user_entity = self._extract_user_entity(user_fact)
        if not user_entity:
            return False

        # Compare user fact similarity with mentioned topics
        user_fact_embedding = self._model.encode(user_fact, convert_to_numpy=True)

        for topic in mentioned_topics:
            topic_embedding = self._model.encode(topic, convert_to_numpy=True)
            similarity = self._model.similarity(user_fact_embedding, topic_embedding)[0][0]

            # High similarity between user fact and mentioned topic suggests
            # the topic might be a fact, not noise
            if similarity > 0.8:
                logger.debug(f"🧠 Topic '{topic}' appears to be factual (similarity: {similarity:.3f})")
                return False

        # Low similarity suggests semantic noise
        logger.debug(f"🧠 Topics appear to be semantic noise (low similarity to user fact)")
        return True

    def _extract_user_entity(self, user_fact: str) -> Optional[str]:
        """Extract the main user entity from a fact."""
        # Simple heuristic: look for personal pronouns and extract the object
        personal_indicators = ['i work at', 'i live in', 'i am', 'i have', 'my name is', 'i like']

        fact_lower = user_fact.lower()
        for indicator in personal_indicators:
            if fact_lower.startswith(indicator):
                # Extract the object/prepositional phrase
                entity_part = user_fact[len(indicator):].strip()
                # Simple cleanup - take first few words
                words = entity_part.split()[:3]
                return ' '.join(words)

        return None

    def enhance_retrieval_ranking(self, query: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Enhance retrieval ranking using semantic similarity.

        Args:
            query: Search query
            candidates: List of (candidate_text, original_score) tuples

        Returns:
            Reranked list with semantic enhancement
        """
        if not self.enabled or len(candidates) <= 1:
            return candidates

        query_embedding = self._model.encode(query, convert_to_numpy=True)
        candidate_texts = [c[0] for c in candidates]
        candidate_embeddings = self._model.encode(candidate_texts, convert_to_numpy=True)

        # Compute semantic similarities
        similarities = self._model.similarity(query_embedding, candidate_embeddings)[0]

        # Combine original scores with semantic similarity
        enhanced_candidates = []
        for i, (text, original_score) in enumerate(candidates):
            semantic_score = similarities[i]
            # Weighted combination: 70% original + 30% semantic
            combined_score = 0.7 * original_score + 0.3 * semantic_score
            enhanced_candidates.append((text, combined_score))

        # Sort by combined score
        enhanced_candidates.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"🧠 Enhanced ranking for {len(candidates)} candidates")
        return enhanced_candidates

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the similarity model."""
        return {
            'enabled': self.enabled,
            'model_name': self.model_name,
            'load_time_ms': self._load_time_ms,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE
        }


# Singleton instance for reuse across the system
_minilm_instance: Optional[MiniLMSimilarity] = None


def get_minilm_similarity() -> MiniLMSimilarity:
    """Get singleton MiniLM similarity instance."""
    global _minilm_instance

    if _minilm_instance is None:
        _minilm_instance = MiniLMSimilarity()

    return _minilm_instance


def detect_semantic_noise_with_minilm(user_fact: str, mentioned_topics: List[str]) -> bool:
    """
    Convenience function for semantic noise detection.

    This is the main entry point for integrating MiniLM into the pipeline.

    Args:
        user_fact: Fact about the user
        mentioned_topics: Topics mentioned in conversation

    Returns:
        True if topics appear to be semantic noise
    """
    similarity = get_minilm_similarity()
    return similarity.detect_semantic_noise(user_fact, mentioned_topics)


if __name__ == "__main__":
    # Simple test
    minilm = MiniLMSimilarity()

    # Test semantic similarity
    text1 = "I work at OpenAI"
    text2 = "I'm employed by OpenAI"
    result = minilm.compute_similarity(text1, text2)
    print(f"Similarity between '{text1}' and '{text2}': {result.similarity_score:.3f}")

    # Test semantic noise detection
    user_fact = "I work at OpenAI"
    topics = ["Google projects", "Anthropic research"]
    is_noise = minilm.detect_semantic_noise(user_fact, topics)
    print(f"Are topics semantic noise? {is_noise}")

    # Test coreference resolution
    pronoun = "he"
    candidates = ["John", "Mary", "OpenAI"]
    context = "John works at"
    resolved = minilm.resolve_coreference(pronoun, candidates, context)
    print(f"Coreference: '{pronoun}' → '{resolved}'")