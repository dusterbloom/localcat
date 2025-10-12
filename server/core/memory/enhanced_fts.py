"""
Enhanced FTS implementation with BM25 ranking and query expansion
SOTA full-text search for the memory system
"""

import re
import math
import time
from typing import List, Tuple, Dict, Set, Optional
from functools import lru_cache
from loguru import logger

# Import constants for cache configuration
try:
    from .memory_constants import CacheConfig
except ImportError:
    # Fallback if constants not available
    class CacheConfig:
        QUERY_EXPANSION_CACHE_SIZE = 128
        BM25_STATS_TTL_SECONDS = 60


class EnhancedFTS:
    """
    State-of-the-Art FTS with BM25 ranking, query expansion, and multi-factor scoring
    WITH CACHING for improved performance
    """

    def __init__(self, store):
        self.store = store
        self._init_enhanced_schema()

        # Query expansion dictionary
        self.expansions = {
            "live": ["reside", "dwell", "inhabit", "stay", "located"],
            "work": ["job", "career", "employment", "profession", "occupy"],
            "home": ["house", "residence", "location", "place", "dwelling"],
            "name": ["called", "known as", "named", "identity"],
            "from": ["originate", "come", "hail", "from_place"],
            "family": ["parent", "child", "sibling", "relative"],
            "friend": ["acquaintance", "colleague", "companion"],
            "school": ["education", "study", "learn", "university"],
            "food": ["eat", "meal", "cuisine", "dish"],
            "travel": ["trip", "journey", "visit", "go"],
        }

        # BM25 statistics cache (TTL-based)
        self._stats_cache: Optional[Tuple[int, float]] = None
        self._stats_cache_time: float = 0
        self._stats_ttl: float = CacheConfig.BM25_STATS_TTL_SECONDS

        # Bind cached query expansion method
        self._cached_expand_query = lru_cache(maxsize=CacheConfig.QUERY_EXPANSION_CACHE_SIZE)(
            self._expand_query_impl
        )
    
    def _init_enhanced_schema(self):
        """Initialize enhanced FTS schema with BM25 support"""
        cur = self.store.sql.cursor()
        try:
            # Create enhanced content table for better ranking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks_content(
                    rowid INTEGER PRIMARY KEY,
                    text TEXT,
                    eid TEXT,
                    ts INTEGER,
                    session_id TEXT,
                    turn_id INTEGER,
                    term_frequency REAL DEFAULT 1.0,
                    document_length INTEGER,
                    entity_boost REAL DEFAULT 1.0
                )
            """)
            
            # Create enhanced FTS table with better tokenization
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_enhanced USING fts5(
                    text,
                    eid UNINDEXED,
                    ts UNINDEXED,
                    session_id UNINDEXED,
                    turn_id UNINDEXED,
                    tokenize='unicode61 remove_diacritics 1',
                    content=chunks_content,
                    content_rowid=rowid
                )
            """)
            
            # Triggers to keep content table in sync
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_enhanced_insert AFTER INSERT ON chunks_content
                BEGIN
                    INSERT INTO chunks_fts_enhanced(rowid, text, eid, ts, session_id, turn_id)
                    VALUES (new.rowid, new.text, new.eid, new.ts, new.session_id, new.turn_id);
                END
            """)
            
            self.store.sql.commit()
            logger.debug("[EnhancedFTS] Enhanced schema initialized")
            
        except Exception as e:
            logger.warning(f"[EnhancedFTS] Schema initialization failed: {e}")
    
    def expand_query(self, query: str) -> str:
        """
        Intelligently expand query with synonyms and related terms (CACHED)

        Args:
            query: Original query string

        Returns:
            Expanded query with OR terms for better recall
        """
        # Delegate to cached implementation
        return self._cached_expand_query(query)

    def _expand_query_impl(self, query: str) -> str:
        """
        Internal implementation of query expansion (cached via LRU)

        This method should NOT be called directly - use expand_query() instead.
        """
        # Clean and normalize terms
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
        terms = cleaned.split()

        # Common stopwords to filter out
        stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                     "have", "has", "had", "do", "does", "did", "will", "would", "could",
                     "should", "may", "might", "can", "of", "at", "by", "for", "with", "to", "in", "on"}

        expanded_terms = []
        seen = set()

        # Limit to first 5 terms to prevent query explosion
        for term in terms[:5]:
            if term and term not in seen and term not in stopwords and len(term) > 2:
                # Remove quotes - FTS5 needs bareword matching, not exact phrase
                expanded_terms.append(term)
                seen.add(term)

                # Add selective synonyms (max 2 per term)
                if term in self.expansions:
                    for synonym in self.expansions[term][:2]:
                        if synonym not in seen:
                            expanded_terms.append(synonym)
                            seen.add(synonym)

        return " OR ".join(expanded_terms) if expanded_terms else ""

    def _get_collection_stats(self) -> Tuple[int, float]:
        """
        Get collection statistics with TTL-based caching

        Returns:
            Tuple of (total_docs, avg_doc_length)
        """
        now = time.time()

        # Check cache validity
        if self._stats_cache is None or (now - self._stats_cache_time) > self._stats_ttl:
            # Cache miss or expired - fetch from DB
            cur = self.store.sql.cursor()
            stats = cur.execute("SELECT COUNT(*), AVG(document_length) FROM chunks_content").fetchone()
            self._stats_cache = stats if stats else (0, 100)
            self._stats_cache_time = now
            logger.debug(f"[EnhancedFTS] BM25 stats cache refreshed: {self._stats_cache}")

        return self._stats_cache
    
    def calculate_bm25_score(self, term_freq: float, doc_length: int, avg_doc_length: float, 
                           total_docs: int, docs_with_term: int, k1: float = 1.2, b: float = 0.75) -> float:
        """
        Calculate BM25 score for document relevance
        
        Args:
            term_freq: Term frequency in document
            doc_length: Document length
            avg_doc_length: Average document length in collection
            total_docs: Total number of documents
            docs_with_term: Number of documents containing term
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            
        Returns:
            BM25 score
        """
        idf = math.log((total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5))
        tf_component = (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * doc_length / avg_doc_length))
        return idf * tf_component
    
    def enhanced_search(self, query: str, limit: int = 10, session_ids: List[str] = None, eids: List[str] = None) -> List[Tuple[float, str, str, int]]:
        """
        SOTA FTS search with BM25 ranking and multi-factor scoring (WITH CACHING)

        Args:
            query: Search query
            limit: Maximum results
            session_ids: Optional session ID filter for scoped search (preferred for convo)
            eids: Optional entity ID filter (kept for compatibility)

        Returns:
            List of (score, text, eid, timestamp) tuples
        """
        if not query.strip():
            return []

        expanded_query = self.expand_query(query)  # Uses LRU cache
        logger.debug(f"[EnhancedFTS] Expanded query: {expanded_query}")

        cur = self.store.sql.cursor()

        # Get collection statistics for BM25 (WITH TTL CACHE)
        total_docs, avg_doc_length = self._get_collection_stats()

        if total_docs == 0:
            return []
        
        try:
            # Base query with BM25
            if session_ids:
                placeholders = ','.join('?' * len(session_ids))
                sql = f"""
                    SELECT c.text AS text, c.eid, c.ts, c.term_frequency, c.document_length, c.entity_boost,
                           bm25(chunks_fts_enhanced) AS bm25_score
                    FROM chunks_fts_enhanced 
                    JOIN chunks_content c ON chunks_fts_enhanced.rowid = c.rowid
                    WHERE chunks_fts_enhanced MATCH ? AND c.session_id IN ({placeholders})
                    ORDER BY bm25_score DESC, c.ts DESC
                    LIMIT ?
                """
                params = [expanded_query] + session_ids + [limit * 2]
            elif eids:
                placeholders = ','.join('?' * len(eids))
                sql = f"""
                    SELECT c.text AS text, c.eid, c.ts, c.term_frequency, c.document_length, c.entity_boost,
                           bm25(chunks_fts_enhanced) AS bm25_score
                    FROM chunks_fts_enhanced 
                    JOIN chunks_content c ON chunks_fts_enhanced.rowid = c.rowid
                    WHERE chunks_fts_enhanced MATCH ? AND c.eid IN ({placeholders})
                    ORDER BY bm25_score DESC, c.ts DESC
                    LIMIT ?
                """
                params = [expanded_query] + eids + [limit * 2]
            else:
                sql = """
                    SELECT c.text AS text, c.eid, c.ts, c.term_frequency, c.document_length, c.entity_boost,
                           bm25(chunks_fts_enhanced) AS bm25_score
                    FROM chunks_fts_enhanced 
                    JOIN chunks_content c ON chunks_fts_enhanced.rowid = c.rowid
                    WHERE chunks_fts_enhanced MATCH ?
                    ORDER BY bm25_score DESC, c.ts DESC
                    LIMIT ?
                """
                params = [expanded_query, limit * 2]
            
            results = cur.execute(sql, params).fetchall()
            
        except Exception as e:
            logger.warning(f"[EnhancedFTS] Enhanced search failed, falling back to basic: {e}")
            return self._fallback_search(query, limit, eids)
        
        # Multi-factor relevance scoring
        scored_results = []
        current_time = int(time.time() * 1000)
        
        for text, eid, ts, term_freq, doc_length, entity_boost, bm25_score in results:
            # Recency boost (newer = higher score, 1-week decay)
            age_hours = (current_time - ts) / (1000 * 60 * 60)
            recency_boost = max(0.1, 1.0 - (age_hours / 168))
            
            # Entity importance boost
            if eid == 'conversation':
                entity_importance = 1.0  # Baseline
            elif eid == 'summary':
                entity_importance = 1.2  # Summaries get slight boost
            elif eid and eid.startswith('session:'):
                entity_importance = 1.1  # Session-specific gets small boost
            else:
                entity_importance = 1.3  # Specific entities get highest boost
            
            # Document length normalization (shorter docs often more relevant)
            length_norm = 1.0 / (1.0 + math.log(doc_length / 50 + 1))
            
            # Final composite score
            final_score = (
                bm25_score * 0.4 +           # BM25 relevance
                recency_boost * 0.2 +        # Recency
                entity_importance * 0.2 +    # Entity importance
                length_norm * 0.1 +          # Length preference
                entity_boost * 0.1           # Custom boost
            )
            
            scored_results.append((final_score, text, eid, ts))
        
        # Sort by final score and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[:limit]
    
    def _fallback_search(self, query: str, limit: int, eids: List[str] = None) -> List[Tuple[float, str, str, int]]:
        """Fallback to basic FTS if enhanced search fails"""
        try:
            if eids:
                results = self.store.search_fts_scoped(query, eids, limit)
            else:
                results = self.store.search_fts(query, limit)
            
            # Simple scoring for fallback
            current_time = int(time.time() * 1000)
            scored = []
            for text, eid, ts in results:
                age_hours = (current_time - ts) / (1000 * 60 * 60)
                recency_score = max(0.1, 1.0 - (age_hours / 168))
                scored.append((recency_score, text, eid, ts))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
            
        except Exception as e:
            logger.error(f"[EnhancedFTS] Fallback search also failed: {e}")
            return []
    
    def index_conversation(self, text: str, session_id: str, turn_id: int, timestamp: int):
        """
        Index a conversation turn with enhanced metadata
        
        Args:
            text: Conversation text
            session_id: Session identifier
            turn_id: Turn number
            timestamp: Unix timestamp in milliseconds
        """
        cur = self.store.sql.cursor()
        try:
            # Calculate basic statistics
            terms = text.lower().split()
            term_freq = len([t for t in terms if t]) / max(len(terms), 1)
            doc_length = len(text)
            
            # Insert into enhanced content table
            cur.execute("""
                INSERT OR REPLACE INTO chunks_content 
                (text, eid, ts, session_id, turn_id, term_frequency, document_length, entity_boost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (text, 'conversation', timestamp, session_id, turn_id, term_freq, doc_length, 1.0))
            
            self.store.sql.commit()
            logger.debug(f"[EnhancedFTS] Indexed conversation: {text[:30]}...")
            
        except Exception as e:
            logger.error(f"[EnhancedFTS] Failed to index conversation: {e}")
    
    def reindex_existing_data(self):
        """Reindex existing conversation data with enhanced FTS"""
        logger.info("[EnhancedFTS] Starting reindex of existing data...")
        
        cur = self.store.sql.cursor()
        try:
            # Get existing conversation turns
            conversations = cur.execute("""
                SELECT id, text, session_id, turn_id, ts 
                FROM conversation_turn 
                ORDER BY ts DESC
                LIMIT 1000
            """).fetchall()
            
            indexed_count = 0
            for conv_id, text, session_id, turn_id, ts in conversations:
                self.index_conversation(text, session_id, turn_id, ts)
                indexed_count += 1
                
                if indexed_count % 100 == 0:
                    logger.debug(f"[EnhancedFTS] Reindexed {indexed_count} conversations...")
            
            logger.info(f"[EnhancedFTS] Reindexed {indexed_count} conversations")
            
        except Exception as e:
            logger.error(f"[EnhancedFTS] Reindex failed: {e}")
