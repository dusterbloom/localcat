"""Optional embedding-based reranker for semantic similarity.

Uses SentenceTransformers models to compute semantic similarity between
query and candidate texts, providing an optional wsim component for
composite scoring.
"""

import os
import time
import math
from typing import List, Optional, Tuple
from loguru import logger


class EmbeddingReranker:
    """
    Lazy-loaded embedding reranker with optional dependencies.
    
    Only loads SentenceTransformers when enabled and available.
    Returns zeros when disabled or dependencies missing.
    """
    
    def __init__(self):
        self._model = None
        self._enabled = None
        self._model_name = None
        self._max_candidates = None
        self._warning_logged = False
        
    def _check_enabled(self) -> bool:
        """Check if embeddings are enabled via environment."""
        if self._enabled is None:
            self._enabled = os.getenv("MEMORY_RERANK_EMBEDDINGS_ENABLED", "false").lower() in ("1", "true", "yes")
            if self._enabled:
                self._model_name = os.getenv("MEMORY_RERANK_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                self._max_candidates = int(os.getenv("MEMORY_RERANK_MAX_CANDIDATES", "24"))
        return self._enabled
    
    def _load_model(self):
        """Lazy load the SentenceTransformers model."""
        if self._model is not None:
            return
            
        try:
            import sentence_transformers
            self._model = sentence_transformers.SentenceTransformer(self._model_name)
            logger.info(f"[EmbeddingReranker] Loaded model: {self._model_name}")
        except ImportError:
            if not self._warning_logged:
                logger.warning("[EmbeddingReranker] sentence_transformers not available, embeddings disabled")
                self._warning_logged = True
            self._enabled = False
        except Exception as e:
            if not self._warning_logged:
                logger.warning(f"[EmbeddingReranker] Failed to load model {self._model_name}: {e}")
                self._warning_logged = True
            self._enabled = False
    
    def similarity(self, query: str, texts: List[str]) -> List[float]:
        """
        Compute semantic similarity between query and each text.
        
        Args:
            query: Query text to embed
            texts: List of candidate texts to compare against query
            
        Returns:
            List of similarity scores (cosine similarity or inner product)
            Returns zeros if embeddings disabled or unavailable
        """
        if not self._check_enabled() or not texts:
            return [0.0] * len(texts)
        
        # Limit candidates to configured maximum
        if len(texts) > self._max_candidates:
            texts = texts[:self._max_candidates]
        
        try:
            self._load_model()
            if self._model is None:
                return [0.0] * len(texts)
            
            # Compute embeddings
            start_time = time.time()
            query_emb = self._model.encode([query], convert_to_numpy=True)
            text_embs = self._model.encode(texts, convert_to_numpy=True)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Compute cosine similarity
            similarities = []
            for text_emb in text_embs:
                # Cosine similarity: (A·B) / (|A| * |B|)
                dot_product = float(query_emb[0] @ text_emb)
                norm_query = float(math.sqrt((query_emb[0] ** 2).sum()))
                norm_text = float(math.sqrt((text_emb ** 2).sum()))
                
                if norm_query > 0 and norm_text > 0:
                    similarity = dot_product / (norm_query * norm_text)
                    # Scale to [0, 1] range from [-1, 1]
                    similarity = (similarity + 1.0) / 2.0
                else:
                    similarity = 0.0
                
                similarities.append(similarity)
            
            logger.debug(f"[EmbeddingReranker] Computed {len(similarities)} similarities in {elapsed_ms:.1f}ms")
            return similarities
            
        except Exception as e:
            logger.error(f"[EmbeddingReranker] Failed to compute similarities: {e}")
            return [0.0] * len(texts)


# Global instance for reuse
_reranker_instance = None


def get_embedding_reranker() -> EmbeddingReranker:
    """Get or create the global embedding reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = EmbeddingReranker()
    return _reranker_instance
