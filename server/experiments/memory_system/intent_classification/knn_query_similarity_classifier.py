"""
K-Nearest Neighbors Query Similarity Intent Classifier

Based on "Intent Classification on Low-Resource Languages with Query Similarity Search" (2025)
https://arxiv.org/pdf/2505.18241

Key innovation: Instead of traditional supervised learning, uses query similarity search
with k-nearest neighbors to classify intents, especially effective for low-resource domains.

This approach:
1. Stores query examples with their intents
2. For new queries, finds k most similar examples
3. Classifies based on majority vote of nearest neighbors
4. Automatically adapts as more examples are added
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from collections import Counter, defaultdict
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    import faiss  # For efficient similarity search
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS not available, using numpy for similarity search")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.info("scikit-learn not available, using manual cosine similarity")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class QueryExample:
    """A query example with its intent label"""
    query: str
    intent: str
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    timestamp: float = 0.0


class KNNQuerySimilarityClassifier:
    """
    K-Nearest Neighbors Query Similarity Classifier

    Implementation of the 2025 SOTA approach for intent classification
    using query similarity search instead of traditional supervised learning.
    """

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 k: int = 5,
                 similarity_threshold: float = 0.7,
                 max_examples_per_intent: int = 50):

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required")

        self.model_name = model_name
        self.k = k  # Number of nearest neighbors
        self.similarity_threshold = similarity_threshold
        self.max_examples_per_intent = max_examples_per_intent

        # Initialize embedding model
        self.model: Optional[SentenceTransformer] = None

        # Query examples database
        self.query_examples: List[QueryExample] = []
        self.embeddings: Optional[np.ndarray] = None

        # FAISS index for efficient similarity search (if available)
        self.faiss_index = None

        # Intent mapping (from your existing system)
        self.intent_configs = {
            "remember_fact": {"skip_memory": False, "strategy": "storage_focused"},
            "recall_query": {"skip_memory": False, "strategy": "retrieval_focused"},
            "general_chat": {"skip_memory": True, "strategy": "minimal"},
            "greeting": {"skip_memory": True, "strategy": "skip"},
            "goodbye": {"skip_memory": True, "strategy": "skip"},
            "clarification": {"skip_memory": False, "strategy": "contextual"},
            "correction": {"skip_memory": False, "strategy": "recent_context"}
        }

        self.initialized = False

    async def initialize(self):
        """Initialize the sentence transformer model"""
        try:
            logger.info(f"Initializing KNN Query Similarity classifier with {self.model_name}")
            start_time = time.perf_counter()

            # Load sentence transformer
            self.model = SentenceTransformer(self.model_name)

            # Initialize with some basic examples from your domain
            await self._add_initial_examples()

            # Build initial index
            self._build_similarity_index()

            init_time = (time.perf_counter() - start_time) * 1000
            self.initialized = True

            logger.info(f"KNN classifier initialized in {init_time:.2f}ms with {len(self.query_examples)} examples")

        except Exception as e:
            logger.error(f"Failed to initialize KNN classifier: {e}")
            raise

    async def _add_initial_examples(self):
        """Add initial query examples based on your domain"""

        # Technical/factual statements that should NOT skip memory
        factual_queries = [
            ("The payment system uses blockchain technology", "remember_fact"),
            ("Multilateral netting settles payments between agents", "remember_fact"),
            ("Agents coordinate to settle at a specific time", "remember_fact"),
            ("The purpose is to save agent liquidity", "remember_fact"),
            ("There are cycles in the payments graph", "remember_fact"),
            ("You can pay immediately or delay payment", "remember_fact"),
            ("The graph needs to be balanced", "remember_fact"),
            ("Banks have been doing netting for centuries", "remember_fact"),
            ("This avoids blockchain transaction costs", "remember_fact"),
            ("Agents get refunds from the liquidity pool", "remember_fact"),
            ("The provider manages the netting process", "remember_fact"),
            ("Liquidity is saved through coordination", "remember_fact"),
            ("Payment graphs show agent relationships", "remember_fact"),
            ("Settlement happens at regular intervals", "remember_fact")
        ]

        # Questions that should retrieve memory
        question_queries = [
            ("How does the payment system work?", "recall_query"),
            ("What is multilateral netting?", "recall_query"),
            ("How do payment graphs work?", "recall_query"),
            ("What saves agent liquidity?", "recall_query"),
            ("How does settlement work?", "recall_query"),
            ("What did I tell you about payments?", "recall_query"),
            ("Can you explain the netting process?", "recall_query"),
            ("How does the balancing work?", "recall_query"),
            ("What do you know about the system?", "recall_query"),
            ("Tell me about liquidity management", "recall_query")
        ]

        # Casual conversation that should skip memory
        casual_queries = [
            ("That's really interesting", "general_chat"),
            ("I see what you mean", "general_chat"),
            ("Thanks for the explanation", "general_chat"),
            ("That makes sense", "general_chat"),
            ("Okay I understand", "general_chat"),
            ("Let me think about that", "general_chat"),
            ("Sure, I can help with that", "general_chat"),
            ("Hmm, interesting point", "general_chat")
        ]

        # Greetings
        greeting_queries = [
            ("Hello", "greeting"),
            ("Hi there", "greeting"),
            ("Good morning", "greeting"),
            ("Hey", "greeting")
        ]

        # Goodbyes
        goodbye_queries = [
            ("Goodbye", "goodbye"),
            ("See you later", "goodbye"),
            ("Take care", "goodbye"),
            ("I gotta go now", "goodbye"),
            ("Amazing, bye", "goodbye")
        ]

        # Combine all examples
        all_examples = (factual_queries + question_queries +
                       casual_queries + greeting_queries + goodbye_queries)

        # Add examples with embeddings
        for query, intent in all_examples:
            await self.add_example(query, intent)

    async def add_example(self, query: str, intent: str, confidence: float = 1.0):
        """Add a new query example to the database"""

        if not self.model:
            logger.warning("Model not initialized")
            return

        # Create embedding
        embedding = self.model.encode([query])[0]

        # Create example
        example = QueryExample(
            query=query,
            intent=intent,
            embedding=embedding,
            confidence=confidence,
            timestamp=time.time()
        )

        self.query_examples.append(example)

        # Rebuild index if we have examples
        if len(self.query_examples) % 10 == 0:  # Rebuild every 10 additions
            self._build_similarity_index()

        logger.debug(f"Added example: '{query[:30]}...' → {intent}")

    def _build_similarity_index(self):
        """Build similarity search index"""

        if not self.query_examples:
            return

        # Extract embeddings
        embeddings = np.array([ex.embedding for ex in self.query_examples])
        self.embeddings = embeddings

        # Build FAISS index if available
        if FAISS_AVAILABLE and len(self.query_examples) > 10:
            try:
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

                # Normalize embeddings for cosine similarity
                normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                self.faiss_index.add(normalized_embeddings.astype('float32'))

                logger.debug(f"Built FAISS index with {len(self.query_examples)} examples")

            except Exception as e:
                logger.warning(f"Failed to build FAISS index: {e}")
                self.faiss_index = None

    def _find_nearest_neighbors(self, query_embedding: np.ndarray) -> List[Tuple[int, float]]:
        """Find k nearest neighbors for a query"""

        if self.faiss_index and len(self.query_examples) > 10:
            # Use FAISS for efficient search
            try:
                # Normalize query embedding
                normalized_query = query_embedding / np.linalg.norm(query_embedding)
                normalized_query = normalized_query.reshape(1, -1).astype('float32')

                # Search
                similarities, indices = self.faiss_index.search(normalized_query, self.k)

                return [(int(idx), float(sim)) for idx, sim in zip(indices[0], similarities[0])]

            except Exception as e:
                logger.warning(f"FAISS search failed: {e}, falling back to numpy")

        # Fallback to numpy-based search
        if self.embeddings is None:
            return []

        # Compute cosine similarities
        if SKLEARN_AVAILABLE:
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        else:
            # Manual cosine similarity
            normalized_query = query_embedding / np.linalg.norm(query_embedding)
            normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            similarities = np.dot(normalized_embeddings, normalized_query)

        # Get top k
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]

        return [(int(idx), float(similarities[idx])) for idx in top_k_indices]

    async def predict(self, text: str, confidence_threshold: float = None) -> Dict[str, Any]:
        """
        Predict intent using k-nearest neighbors query similarity search

        Args:
            text: Input query text
            confidence_threshold: Minimum confidence threshold

        Returns:
            Classification result
        """
        if not self.initialized:
            await self.initialize()

        threshold = confidence_threshold or self.similarity_threshold
        start_time = time.perf_counter()

        try:
            # Encode query
            query_embedding = self.model.encode([text])[0]

            # Find k nearest neighbors
            neighbors = self._find_nearest_neighbors(query_embedding)

            if not neighbors:
                # No examples available
                return self._get_fallback_result("No examples available", start_time)

            # Collect neighbor intents and similarities
            neighbor_votes = []
            neighbor_details = []

            for idx, similarity in neighbors:
                if idx < len(self.query_examples):
                    example = self.query_examples[idx]
                    neighbor_votes.append((example.intent, similarity, example.confidence))
                    neighbor_details.append({
                        "query": example.query,
                        "intent": example.intent,
                        "similarity": similarity
                    })

            # Weighted voting based on similarity and confidence
            intent_scores = defaultdict(float)
            total_weight = 0.0

            for intent, similarity, confidence in neighbor_votes:
                weight = similarity * confidence
                intent_scores[intent] += weight
                total_weight += weight

            if not intent_scores:
                return self._get_fallback_result("No valid neighbors", start_time)

            # Find best intent
            best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
            best_score = intent_scores[best_intent] / total_weight if total_weight > 0 else 0.0

            # Apply confidence threshold
            if best_score < threshold:
                final_intent = "general_chat"
                final_confidence = 0.3
                low_confidence = True
            else:
                final_intent = best_intent
                final_confidence = min(1.0, best_score)
                low_confidence = False

            # Get intent configuration
            intent_config = self.intent_configs.get(final_intent, {
                "skip_memory": True,
                "strategy": "minimal"
            })

            inference_time = (time.perf_counter() - start_time) * 1000

            return {
                "intent": final_intent,
                "confidence": final_confidence,
                "low_confidence": low_confidence,
                "skip_memory": intent_config["skip_memory"],
                "strategy": intent_config["strategy"],
                "inference_time_ms": inference_time,
                "model_label": best_intent,
                "knn_details": {
                    "k_used": len(neighbors),
                    "best_similarity": neighbors[0][1] if neighbors else 0.0,
                    "neighbor_intents": [detail["intent"] for detail in neighbor_details],
                    "neighbor_similarities": [detail["similarity"] for detail in neighbor_details],
                    "all_scores": dict(intent_scores),
                    "threshold_used": threshold
                }
            }

        except Exception as e:
            logger.error(f"KNN classification failed for '{text}': {e}")
            return self._get_fallback_result(f"Error: {str(e)}", start_time)

    def _get_fallback_result(self, reason: str, start_time: float) -> Dict[str, Any]:
        """Get fallback result for failed classifications"""
        return {
            "intent": "general_chat",
            "confidence": 0.0,
            "low_confidence": True,
            "skip_memory": True,
            "strategy": "minimal",
            "inference_time_ms": (time.perf_counter() - start_time) * 1000,
            "model_label": "fallback",
            "reason": reason
        }

    def learn_from_production(self, query: str, correct_intent: str):
        """Learn from production corrections - zero maintenance adaptation"""
        asyncio.create_task(self.add_example(query, correct_intent, confidence=0.9))

    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics"""
        intent_counts = Counter(ex.intent for ex in self.query_examples)

        return {
            "total_examples": len(self.query_examples),
            "intent_distribution": dict(intent_counts),
            "k": self.k,
            "similarity_threshold": self.similarity_threshold,
            "model_name": self.model_name,
            "faiss_enabled": self.faiss_index is not None,
            "initialized": self.initialized
        }


# Factory function
def create_knn_classifier(**kwargs) -> KNNQuerySimilarityClassifier:
    """Create KNN classifier with default settings"""
    return KNNQuerySimilarityClassifier(**kwargs)


# Test function
async def test_knn_classifier():
    """Test the KNN classifier with your production data"""

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("⚠️  sentence-transformers not available")
        return

    print("Testing KNN Query Similarity Intent Classifier")
    print("Based on 2025 SOTA research: arxiv.org/pdf/2505.18241")
    print("=" * 60)

    # Load your production test cases
    try:
        with open("intent_replay_data.json", "r") as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("❌ intent_replay_data.json not found")
        return

    classifier = create_knn_classifier(k=3, similarity_threshold=0.6)
    await classifier.initialize()

    stats = classifier.get_stats()
    print(f"✅ Initialized with {stats['total_examples']} query examples")
    print(f"📊 Intent distribution: {stats['intent_distribution']}")
    print()

    correct = 0
    total = len(test_cases)
    detailed_results = []

    for case in test_cases:
        result = await classifier.predict(case["text"])

        # Determine if this should skip memory based on content
        expected_should_skip = True  # Default
        text_lower = case["text"].lower()

        # Technical content should NOT skip memory
        if any(term in text_lower for term in ['payment', 'graph', 'agent', 'liquidity', 'netting', 'blockchain']):
            if len(case["text"]) > 30:  # Substantial technical content
                expected_should_skip = False

        actual_should_skip = result["skip_memory"]
        is_correct = (expected_should_skip == actual_should_skip)

        if is_correct:
            correct += 1

        detailed_results.append((case, result, is_correct, expected_should_skip))

        status = "✅" if is_correct else "❌"
        print(f"{status} '{case['text'][:50]}...'")
        print(f"   Predicted: {result['intent']} (conf: {result['confidence']:.3f}, skip: {actual_should_skip})")
        print(f"   Expected skip: {expected_should_skip}")

        # Show KNN details for interesting cases
        knn_details = result.get('knn_details', {})
        if knn_details.get('neighbor_intents'):
            neighbor_intents = knn_details['neighbor_intents'][:3]  # Top 3
            similarities = knn_details['neighbor_similarities'][:3]
            print(f"   Neighbors: {list(zip(neighbor_intents, [f'{s:.3f}' for s in similarities]))}")
        print()

    accuracy = (correct / total) * 100
    print(f"🎯 KNN Query Similarity Accuracy: {accuracy:.1f}% ({correct}/{total})")

    if accuracy > 70:
        print("🎉 KNN classifier performing well!")
        print("💡 Query similarity approach is working for your domain")
    else:
        print("🔧 Adding corrections to improve performance...")

        # Learn from mistakes
        for case, result, is_correct, expected_should_skip in detailed_results:
            if not is_correct:
                correct_intent = "remember_fact" if not expected_should_skip else "general_chat"
                await classifier.add_example(case["text"], correct_intent, confidence=0.8)

        print(f"📚 Added corrections, new total: {classifier.get_stats()['total_examples']} examples")

    return classifier


if __name__ == "__main__":
    asyncio.run(test_knn_classifier())