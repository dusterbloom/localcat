"""
Semantic Memory Sidecar (Optional, Production-Safe)

Provides semantic search capabilities using sentence embeddings and FAISS.
Designed to be strictly optional with graceful degradation if dependencies are missing.
"""

import os
import json
import hashlib
import time
from typing import List, Tuple, Dict, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

# Optional imports - will be checked at runtime
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.debug("NumPy not available - semantic memory disabled")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.debug("FAISS not available - semantic memory disabled")

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    logger.debug("xxhash not available - using fallback hash")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.debug("sentence-transformers not available - semantic memory disabled")


@dataclass
class SemanticMetadata:
    """Metadata for semantic memory entries."""
    text: str
    ts: int  # timestamp in milliseconds
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    kind: str = "unknown"  # "conversation", "summary", "graph_fact"


class SemanticMemorySidecar:
    """
    Semantic memory sidecar with FAISS index and sentence embeddings.
    
    Provides fuzzy semantic search capabilities that complement exact graph/FTS search.
    Designed to be production-safe with optional dependencies and graceful degradation.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: str = "data/semantic_index",
        similarity_threshold: float = 0.85,
        max_vectors: int = 10000
    ):
        """
        Initialize semantic memory sidecar.
        
        Args:
            model_name: Sentence transformer model name
            index_dir: Directory to store FAISS index and metadata
            similarity_threshold: Threshold for duplicate detection
            max_vectors: Maximum vectors to store in index
        """
        self.model_name = model_name
        # Expand ~ and $VARS for production bundles
        self.index_dir = Path(os.path.expanduser(os.path.expandvars(index_dir)))
        self.similarity_threshold = similarity_threshold
        self.max_vectors = max_vectors
        
        # Runtime state
        self._model = None
        self._index = None
        self._metadata: Dict[int, SemanticMetadata] = {}
        self._next_id = 1
        
        # Ensure dependencies are available
        self._check_dependencies()
        
        # Create index directory
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create index
        self._load_or_create_index()
        
        logger.info(f"SemanticMemorySidecar initialized with model: {model_name}")
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for semantic memory")
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for semantic memory")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for semantic memory")
    
    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            logger.debug(f"Loading sentence transformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.debug("Sentence transformer model loaded")
        return self._model
    
    def _load_or_create_index(self) -> None:
        """Load existing index or create a new one."""
        index_path = self.index_dir / "index.faiss"
        metadata_path = self.index_dir / "metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load existing index
                self._index = faiss.read_index(str(index_path))
                logger.debug(f"Loaded FAISS index with {self._index.ntotal} vectors")
                
                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                
                self._metadata = {}
                for id_str, meta_dict in metadata_dict.items():
                    self._metadata[int(id_str)] = SemanticMetadata(**meta_dict)
                
                # Set next ID
                if self._metadata:
                    self._next_id = max(self._metadata.keys()) + 1
                
                logger.debug(f"Loaded metadata for {len(self._metadata)} entries")
                
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        # Get embedding dimension from model
        model = self._get_model()
        dummy_embedding = model.encode(["test"], convert_to_numpy=True)
        dim = dummy_embedding.shape[1]
        
        # Create IndexIDMap2 with IndexFlatIP for cosine similarity
        base_index = faiss.IndexFlatIP(dim)
        self._index = faiss.IndexIDMap2(base_index)
        
        self._metadata = {}
        self._next_id = 1
        
        logger.debug(f"Created new FAISS index with dimension {dim}")
    
    def _generate_stable_id(self, text: str, ts: int) -> int:
        """Generate stable ID for text and timestamp."""
        if XXHASH_AVAILABLE:
            # Use xxhash for better distribution
            hash_obj = xxhash.xxh64(f"{text}_{ts}".encode('utf-8'))
            return hash_obj.intdigest()
        else:
            # Fallback to built-in hash
            combined = f"{text}_{ts}".encode('utf-8')
            return int(hashlib.sha256(combined).hexdigest(), 16) % (2**63)
    
    def _is_duplicate(self, embedding: np.ndarray, text: str) -> Optional[int]:
        """
        Check if embedding/text is too similar to existing entries.
        
        Args:
            embedding: New embedding to check
            text: New text to check (for content similarity)
            
        Returns:
            ID of duplicate entry if found, None otherwise
        """
        if self._index.ntotal == 0:
            return None
        
        # Search for similar vectors
        k = min(5, self._index.ntotal)
        similarities, ids = self._index.search(embedding.reshape(1, -1), k)
        
        # Check similarity threshold
        for i, sim in enumerate(similarities[0]):
            if sim >= self.similarity_threshold:
                existing_id = int(ids[0][i])
                # Additional text-based check to avoid false positives
                if existing_id in self._metadata:
                    existing_text = self._metadata[existing_id].text.lower()
                    new_text = text.lower()
                    
                    # Simple text similarity check
                    words_new = set(new_text.split())
                    words_existing = set(existing_text.split())
                    
                    if words_new and words_existing:
                        overlap = len(words_new.intersection(words_existing)) / len(words_new.union(words_existing))
                        if overlap >= 0.7:  # 70% word overlap threshold
                            return existing_id
        
        return None
    
    def _truncate_index_if_needed(self) -> None:
        """Truncate index if it exceeds max_vectors."""
        if self._index.ntotal <= self.max_vectors:
            return
        
        logger.info(f"Truncating semantic index from {self._index.ntotal} to {self.max_vectors}")
        
        # Get all IDs and timestamps
        all_ids = []
        all_timestamps = []
        
        for idx in range(self._index.ntotal):
            vector_id = int(self._index.id_map.at(idx))
            if vector_id in self._metadata:
                all_ids.append(vector_id)
                all_timestamps.append(self._metadata[vector_id].ts)
        
        # Sort by timestamp (oldest first)
        sorted_indices = sorted(range(len(all_ids)), key=lambda i: all_timestamps[i])
        
        # Keep only the most recent max_vectors entries
        keep_count = self.max_vectors
        remove_ids = [all_ids[i] for i in sorted_indices[:-keep_count]]
        
        # Remove old entries (this is inefficient but FAISS doesn't support easy removal)
        # We'll rebuild the index with only the entries we want to keep
        keep_ids = set(all_ids) - set(remove_ids)
        
        # Get embeddings for entries to keep
        keep_embeddings = []
        keep_vector_ids = []
        new_metadata = {}
        
        for vector_id in keep_ids:
            if vector_id in self._metadata:
                # Re-encode the text (could store embeddings, but this keeps memory usage lower)
                text = self._metadata[vector_id].text
                embedding = self._get_model().encode([text], convert_to_numpy=True)
                keep_embeddings.append(embedding[0])
                keep_vector_ids.append(vector_id)
                new_metadata[vector_id] = self._metadata[vector_id]
        
        # Recreate index
        if keep_embeddings:
            embeddings_array = np.array(keep_embeddings).astype('float32')
            vector_ids_array = np.array(keep_vector_ids)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Create new index
            base_index = faiss.IndexFlatIP(embeddings_array.shape[1])
            self._index = faiss.IndexIDMap2(base_index)
            self._index.add_with_ids(embeddings_array, vector_ids_array)
            
            self._metadata = new_metadata
            logger.info(f"Rebuilt index with {len(keep_embeddings)} entries")
    
    def add(self, text: str, metadata: SemanticMetadata) -> bool:
        """
        Add text to semantic index with duplicate detection.
        
        Args:
            text: Text to add
            metadata: Metadata for the text
            
        Returns:
            True if added, False if duplicate detected
        """
        try:
            # Generate embedding
            model = self._get_model()
            embedding = model.encode([text], convert_to_numpy=True)
            
            # Check for duplicates
            duplicate_id = self._is_duplicate(embedding[0], text)
            if duplicate_id is not None:
                logger.debug(f"Skipping duplicate semantic entry: '{text[:50]}...'")
                return False
            
            # Generate stable ID
            vector_id = self._generate_stable_id(text, metadata.ts)
            
            # Ensure ID is unique
            while vector_id in self._metadata:
                vector_id += 1
            
            # Normalize embedding for cosine similarity
            faiss.normalize_L2(embedding)
            
            # Add to index
            self._index.add_with_ids(embedding, np.array([vector_id]))
            
            # Store metadata
            self._metadata[vector_id] = metadata
            
            # Truncate if needed
            self._truncate_index_if_needed()
            
            logger.debug(f"Added semantic entry: '{text[:50]}...' (ID: {vector_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add semantic entry: {e}")
            return False
    
    def recall(
        self,
        query: str,
        k: int = 10,
        scopes: Optional[Dict[str, Any]] = None,
        token_budget: int = 100
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Recall semantically similar texts.
        
        Args:
            query: Query text
            k: Maximum number of results to return
            scopes: Optional scopes for filtering (user_id, session_id, kind)
            token_budget: Maximum estimated tokens to return
            
        Returns:
            List of (text, similarity_score, metadata_dict) tuples
        """
        if self._index.ntotal == 0:
            return []
        
        try:
            # Generate query embedding
            model = self._get_model()
            query_embedding = model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search index
            search_k = min(k * 2, self._index.ntotal)  # Get more for filtering
            similarities, ids = self._index.search(query_embedding, search_k)
            
            results = []
            used_tokens = 0
            
            for i, similarity in enumerate(similarities[0]):
                if similarity <= 0.1:  # Minimum similarity threshold
                    break
                
                vector_id = int(ids[0][i])
                if vector_id not in self._metadata:
                    continue
                
                metadata = self._metadata[vector_id]
                
                # Apply scope filtering
                if scopes:
                    if 'user_id' in scopes and metadata.user_id != scopes['user_id']:
                        continue
                    if 'session_id' in scopes and metadata.session_id != scopes['session_id']:
                        continue
                    if 'kind' in scopes and metadata.kind != scopes['kind']:
                        continue
                
                # Estimate token usage
                text_tokens = len(metadata.text) // 4
                if used_tokens + text_tokens > token_budget:
                    break
                
                # Prepare result metadata
                result_metadata = {
                    'ts': metadata.ts,
                    'user_id': metadata.user_id,
                    'session_id': metadata.session_id,
                    'kind': metadata.kind,
                    'vector_id': vector_id
                }
                
                results.append((metadata.text, float(similarity), result_metadata))
                used_tokens += text_tokens
                
                if len(results) >= k:
                    break
            
            logger.debug(f"Semantic recall: {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Semantic recall failed: {e}")
            return []
    
    def save(self) -> bool:
        """Save index and metadata to disk."""
        try:
            index_path = self.index_dir / "index.faiss"
            metadata_path = self.index_dir / "metadata.json"
            
            # Save FAISS index
            faiss.write_index(self._index, str(index_path))
            
            # Save metadata
            metadata_dict = {}
            for vector_id, metadata in self._metadata.items():
                metadata_dict[str(vector_id)] = {
                    'text': metadata.text,
                    'ts': metadata.ts,
                    'user_id': metadata.user_id,
                    'session_id': metadata.session_id,
                    'kind': metadata.kind
                }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved semantic index with {len(self._metadata)} entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save semantic index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the semantic index."""
        return {
            'total_entries': len(self._metadata),
            'index_vectors': self._index.ntotal if self._index else 0,
            'model_name': self.model_name,
            'index_dir': str(self.index_dir),
            'similarity_threshold': self.similarity_threshold,
            'max_vectors': self.max_vectors
        }


# Global singleton instance
_semantic_sidecar: Optional[SemanticMemorySidecar] = None


def get_semantic_sidecar() -> Optional[SemanticMemorySidecar]:
    """
    Get the global semantic sidecar instance.
    
    Returns:
        SemanticMemorySidecar instance if enabled and available, None otherwise
    """
    global _semantic_sidecar
    
    # Check if semantic memory is enabled
    if os.getenv("MEMORY_SEMANTIC_ENABLED", "false").lower() not in ("1", "true", "yes"):
        return None
    
    # Check dependencies once
    if not (NUMPY_AVAILABLE and FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE):
        return None
    
    if _semantic_sidecar is None:
        try:
            model_name = os.getenv("MEMORY_SEMANTIC_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            index_dir = os.getenv("MEMORY_SEMANTIC_DIR", "data/semantic_index")
            similarity_threshold = float(os.getenv("MEMORY_SEMANTIC_SIMILARITY_THRESHOLD", "0.85"))
            max_vectors = int(os.getenv("MEMORY_SEMANTIC_MAX_VECTORS", "10000"))
            
            _semantic_sidecar = SemanticMemorySidecar(
                model_name=model_name,
                index_dir=index_dir,
                similarity_threshold=similarity_threshold,
                max_vectors=max_vectors
            )
            
            logger.info("Semantic memory sidecar initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize semantic sidecar: {e}")
            _semantic_sidecar = None
    
    return _semantic_sidecar


def ingest_conversation_turn(
    text: str,
    user_id: str,
    session_id: str,
    turn_id: int,
    ts: Optional[int] = None
) -> bool:
    """
    Ingest a conversation turn into semantic memory.
    
    Args:
        text: Conversation turn text
        user_id: User identifier
        session_id: Session identifier
        turn_id: Turn number
        ts: Timestamp in milliseconds (default: current time)
        
    Returns:
        True if ingested, False if skipped or failed
    """
    sidecar = get_semantic_sidecar()
    if not sidecar:
        return False
    
    if ts is None:
        ts = int(time.time() * 1000)
    
    metadata = SemanticMetadata(
        text=text,
        ts=ts,
        user_id=user_id,
        session_id=session_id,
        kind="conversation"
    )
    
    return sidecar.add(text, metadata)


def ingest_summary(
    summary: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    ts: Optional[int] = None
) -> bool:
    """
    Ingest a summary into semantic memory.
    
    Args:
        summary: Summary text
        user_id: Optional user identifier
        session_id: Optional session identifier
        ts: Timestamp in milliseconds (default: current time)
        
    Returns:
        True if ingested, False if skipped or failed
    """
    sidecar = get_semantic_sidecar()
    if not sidecar:
        return False
    
    if ts is None:
        ts = int(time.time() * 1000)
    
    metadata = SemanticMetadata(
        text=summary,
        ts=ts,
        user_id=user_id,
        session_id=session_id,
        kind="summary"
    )
    
    return sidecar.add(summary, metadata)


def ingest_graph_fact(
    fact: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    ts: Optional[int] = None
) -> bool:
    """
    Ingest a graph fact into semantic memory.
    
    Args:
        fact: Graph fact text (humanized)
        user_id: Optional user identifier
        session_id: Optional session identifier
        ts: Timestamp in milliseconds (default: current time)
        
    Returns:
        True if ingested, False if skipped or failed
    """
    sidecar = get_semantic_sidecar()
    if not sidecar:
        return False
    
    if ts is None:
        ts = int(time.time() * 1000)
    
    metadata = SemanticMetadata(
        text=fact,
        ts=ts,
        user_id=user_id,
        session_id=session_id,
        kind="graph_fact"
    )
    
    return sidecar.add(fact, metadata)


def save_semantic_index() -> bool:
    """Save the semantic index to disk."""
    sidecar = get_semantic_sidecar()
    if sidecar:
        return sidecar.save()
    return True  # No-op if semantic sidecar is disabled
