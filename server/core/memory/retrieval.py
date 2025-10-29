"""
Retrieval module (Phase 1D)

Implements the existing retrieval policy:
- Entity-first selection with relation priority and recency
- Fallback to recent facts
- Returns up to 3 bullets

This adapter is behavior-preserving: it reads from the host's indices and
store exactly as the previous implementation did.

PERFORMANCE OPTIMIZATIONS:
- P0.1: Persistent LRU cache for provenance lookups (10-30ms reduction)
"""

from typing import List, Tuple, Any, Dict, Optional, Set, NamedTuple
import time
import math
import json
import threading
from functools import lru_cache
from .memory_constants import WEIGHT_MIN_ACTIVE, RECENCY_HALF_LIFE_MS
from loguru import logger
from .quality_filter import QualityFilter

import os

# Import cache config
try:
    from .memory_constants import CacheConfig
except ImportError:
    class CacheConfig:
        PROVENANCE_CACHE_SIZE = 1024


class SessionScopedCache:
    """
    Thread-safe session-scoped cache for provenance visibility checks.

    Fixes cache pollution bug: Each session gets its own LRU cache to prevent
    cross-session data leakage. Implements aggressive cleanup to manage memory.
    """

    def __init__(self, cache_size: int = 256, max_sessions: int = 50):
        """
        Initialize session-scoped cache.

        Args:
            cache_size: Size of LRU cache per session
            max_sessions: Maximum number of session caches to keep
        """
        self._caches: Dict[str, Any] = {}  # session_id -> lru_cache function
        self._access_times: Dict[str, float] = {}  # session_id -> last_access_time
        self._lock = threading.Lock()
        self._cache_size = cache_size
        self._max_sessions = max_sessions
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_for_session(self, session_id: str, impl_func) -> Any:
        """
        Get or create LRU cache for specific session.

        Args:
            session_id: Session identifier
            impl_func: Implementation function to wrap with LRU cache

        Returns:
            Cached function for this session
        """
        with self._lock:
            # Update access time
            self._access_times[session_id] = time.time()

            # Create cache if needed
            if session_id not in self._caches:
                self._caches[session_id] = lru_cache(maxsize=self._cache_size)(impl_func)
                self._cache_misses += 1

                # Periodic cleanup
                if len(self._caches) > self._max_sessions:
                    self._cleanup_stale_sessions()
            else:
                self._cache_hits += 1

            return self._caches[session_id]

    def check_visibility(self, session_id: str, edge_id: str, user_id: Optional[str], impl_func) -> bool:
        """
        Check edge visibility with session-scoped caching.

        Args:
            session_id: Current session ID
            edge_id: Edge to check
            user_id: Current user ID
            impl_func: Implementation function

        Returns:
            True if edge is visible, False otherwise
        """
        if not session_id:
            # No session - execute without caching
            return impl_func(edge_id, user_id, session_id)

        # Get session-specific cache
        cached_func = self.get_cache_for_session(session_id, impl_func)

        # Call cached function
        return cached_func(edge_id, user_id, session_id)

    def clear_session(self, session_id: str):
        """
        Clear cache for specific session when it ends.

        Args:
            session_id: Session to clear
        """
        with self._lock:
            self._caches.pop(session_id, None)
            self._access_times.pop(session_id, None)
            logger.debug(f"[SessionScopedCache] Cleared cache for session {session_id}")

    def _cleanup_stale_sessions(self):
        """Remove least recently used sessions to manage memory."""
        if len(self._caches) <= self._max_sessions:
            return

        # Sort by access time
        sorted_sessions = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )

        # Remove oldest half
        to_remove_count = len(sorted_sessions) // 2
        for session_id, _ in sorted_sessions[:to_remove_count]:
            self._caches.pop(session_id, None)
            self._access_times.pop(session_id, None)

        logger.info(f"[SessionScopedCache] Cleaned up {to_remove_count} stale session caches")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self._lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

            return {
                "active_sessions": len(self._caches),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": hit_rate,
                "max_sessions": self._max_sessions
            }


class Candidate(NamedTuple):
    """Candidate record for composite scoring."""
    text: str
    source: str  # 'graph', 'convo', 'summary'
    score_hint: float  # BM25 for convo, unused for graph, optional for summary
    ts: int  # timestamp
    meta: Dict[str, Any]  # source-specific metadata


class Retrieval:
    def __init__(self, host: Any):
        """host must expose: entity_index, recency_buffer, store."""
        self.host = host

        # Load composite scoring weights from environment or use defaults
        self.weights = self._load_composite_weights()

        # Lazy-loaded embedding reranker
        self._embedding_reranker = None

        # P0.1: Session-scoped provenance cache to avoid repeated batch queries
        # SECURITY FIX: Each session gets its own cache to prevent cross-session data leakage
        self._provenance_cache = SessionScopedCache(
            cache_size=CacheConfig.PROVENANCE_CACHE_SIZE // 4,  # 256 per session
            max_sessions=50  # Keep caches for last 50 sessions
        )

        # Initialize quality filter for Layer 4 defense (retrieval-time filtering)
        self.quality_filter = QualityFilter()
        self._scoring_clock_ms: Optional[int] = None
        
        # Per-retrieve prosody cache to avoid repeated store hits for convo candidates
        self._prosody_cache: Dict[Tuple[str, int], Tuple[float, dict]] = {}

    def _check_edge_visibility_impl(self, edge_id: str, user_id: Optional[str], session_id: Optional[str]) -> bool:
        """
        Internal implementation for checking edge visibility (cached via LRU).

        Args:
            edge_id: Edge identifier to check
            user_id: Current user ID (or None)
            session_id: Current session ID (or None)

        Returns:
            True if edge is visible to user/session, False otherwise
        """
        try:
            # Get provenance for this edge
            prov = self.host.store.get_edge_provenance(edge_id)

            if not prov:
                return False

            # Check user ownership
            if user_id:
                # Extract unique session IDs from provenance
                session_ids = set(sess_id for (_text, sess_id, _turn, _ts) in prov)

                # Batch check if any session belongs to user
                if session_ids:
                    owned_sessions = self.host.store.are_sessions_owned_by_user_batch(list(session_ids), user_id)
                    return bool(owned_sessions)
                return False

            # Fallback: check current session
            elif session_id:
                return any(sess_id == session_id for (_text, sess_id, _turn, _ts) in prov)

            return False

        except Exception as e:
            logger.warning(f"[Retrieval] Provenance check failed for edge {edge_id}: {e}")
            return False

    def retrieve(self, query: str, entities: List[str], turn_id: int, max_bullets: int = 3, intent: Optional[Dict] = None) -> List[str]:
        """
        Retrieve memory bullets using hybrid strategy with fair budget allocation.

        Args:
            query: User query text
            entities: Extracted entities from query
            turn_id: Current conversation turn ID
            max_bullets: Maximum bullets to return
            intent: Optional intent classification result for routing

        Returns:
            List of formatted bullet strings
        """
        # Clear per-retrieve prosody cache to avoid cross-retrieve contamination
        self._prosody_cache.clear()
        
        # Detect slot from query (attribute-aware routing)
        try:
            from .slot_router import SlotRouter
            slot_id, slot_conf = SlotRouter.detect_slot(query)
        except Exception:
            slot_id, slot_conf = (None, 0.0)

        # Source control via env (defaults to graph only for backward compatibility)
        enabled_sources = [s.strip() for s in os.getenv("MEMORY_SOURCES", "graph").split(",") if s.strip()]
        logger.info(f"[Retrieval] Searching memory sources={enabled_sources} for query='{query[:50]}...'")
        logger.debug(f"[Retrieval] enabled_sources={enabled_sources} query='{query[:50]}'")

        # Strengthened intent gating for greetings - suppress memory unless name is relevant
        q = (query or "").strip().lower()
        greeting_terms = ("hello", "hi", "hey", "good morning", "good afternoon", "good evening", "top of the morning", "howdy", "greetings", "yo")
        smalltalk_terms = ("how are you", "how's it going", "what's up", "how do you do", "nice to meet you", "how are you doing")
        
        is_greeting = any(term in q for term in greeting_terms) and len(q.split()) <= 4 if q else False
        is_smalltalk = any(term in q for term in smalltalk_terms) if q else False
        
        # For greetings and smalltalk, only allow name-related memories
        if is_greeting or is_smalltalk:
            # Check if query asks about name identity
            name_indicators = ("name", "who are you", "what's your name", "what is your name", "called", "identity")
            asks_for_name = any(indicator in q for indicator in name_indicators)
            
            if not asks_for_name:
                # Suppress all memory injection for pure greetings/smalltalk
                logger.info(f"[Retrieval] Greeting/smalltalk detected - no memory context needed for: '{q}'")
                logger.debug(f"[Retrieval] Greeting/smalltalk detected, suppressing memory injection: '{q}'")
                return []
            
            relation_allowlist: Optional[Set[str]] = {"name"}
        else:
            relation_allowlist = None

        # Determine source priority based on query characteristics and intent
        source_priority = self._get_source_priority(query, intent)

        # Budget allocation strategy: Give each source a fair chance
        # This prevents graph from starving convo/summary
        budget = self._allocate_budget(max_bullets, enabled_sources)

        # Collect candidates from all enabled sources
        all_candidates: List[Candidate] = []
        seen_texts = set()

        # Observability: track per-source counts before/after slot filtering
        pre_counts = {"graph": 0, "convo": 0, "summary": 0, "semantic": 0}
        post_counts = {"graph": 0, "convo": 0, "summary": 0, "semantic": 0}

        for source in enabled_sources:
            if source == "graph" and budget.get("graph", 0) > 0:
                graph_candidates = self._graph_collect_candidates(
                    query, entities, turn_id, budget["graph"], seen_texts.copy(), relation_allowlist
                )
                pre_counts["graph"] = len(graph_candidates)
                # Slot alignment filter for graph candidates (post-humanization text)
                if slot_id:
                    try:
                        from .slot_router import SlotRouter
                        graph_candidates = [c for c in graph_candidates if SlotRouter.is_slot_aligned(c.text, slot_id)]
                    except Exception:
                        pass
                post_counts["graph"] = len(graph_candidates)
                all_candidates.extend(graph_candidates)
                logger.debug(f"[Retrieval] graph_candidates count={len(graph_candidates)}")

            elif source == "convo" and budget.get("convo", 0) > 0:
                convo_candidates = self._convo_collect_candidates(query, budget["convo"], seen_texts.copy(), slot_id=slot_id)
                pre_counts["convo"] = len(convo_candidates)
                if slot_id:
                    try:
                        from .slot_router import SlotRouter
                        convo_candidates = [c for c in convo_candidates if SlotRouter.is_slot_aligned(c.text, slot_id)]
                    except Exception:
                        pass
                post_counts["convo"] = len(convo_candidates)
                all_candidates.extend(convo_candidates)
                logger.debug(f"[Retrieval] convo_candidates count={len(convo_candidates)}")

            elif source == "summary" and budget.get("summary", 0) > 0:
                summary_candidates = self._summary_collect_candidates(budget["summary"], seen_texts.copy())
                pre_counts["summary"] = len(summary_candidates)
                # Optional: slot filtering for summaries when slot_id present (conservative)
                if slot_id:
                    try:
                        from .slot_router import SlotRouter
                        summary_candidates = [c for c in summary_candidates if SlotRouter.is_slot_aligned(c.text, slot_id)]
                    except Exception:
                        pass
                post_counts["summary"] = len(summary_candidates)
                all_candidates.extend(summary_candidates)
                logger.debug(f"[Retrieval] summary_candidates count={len(summary_candidates)}")

            elif source == "semantic" and budget.get("semantic", 0) > 0:
                semantic_candidates = self._semantic_collect_candidates(query, budget["semantic"], seen_texts.copy())
                pre_counts["semantic"] = len(semantic_candidates)
                if slot_id:
                    try:
                        from .slot_router import SlotRouter
                        semantic_candidates = [c for c in semantic_candidates if SlotRouter.is_slot_aligned(c.text, slot_id)]
                    except Exception:
                        pass
                post_counts["semantic"] = len(semantic_candidates)
                all_candidates.extend(semantic_candidates)
                logger.debug(f"[Retrieval] semantic_candidates count={len(semantic_candidates)}")

        # If a slot is detected, prefer aligned candidates only
        # (Already filtered per-source, this extra guard avoids any leakage).
        if slot_id:
            try:
                from .slot_router import SlotRouter
                all_candidates = [c for c in all_candidates if SlotRouter.is_slot_aligned(c.text, slot_id)]
            except Exception:
                pass

        # Composite re-rank all candidates
        scored_candidates: List[Tuple[float, Candidate, Dict[str, float]]] = []
        start_time = time.time()

        previous_clock = self._scoring_clock_ms
        self._scoring_clock_ms = int(time.time()) * 1000
        try:
            for candidate in all_candidates:
                total_score, components = self._composite_score(query, candidate, source_priority, all_candidates)
                scored_candidates.append((total_score, candidate, components))
        finally:
            self._scoring_clock_ms = previous_clock

        # If a slot is detected, boost best graph candidate to guarantee first bullet when present
        if scored_candidates and slot_id:
            try:
                from .slot_router import SlotRouter
                # Identify best aligned graph candidate
                best_idx = -1
                best_score = -1e9
                for i, (score, cand, comps) in enumerate(scored_candidates):
                    if cand.source == 'graph' and SlotRouter.is_slot_aligned(cand.text, slot_id):
                        if score > best_score:
                            best_score = score
                            best_idx = i
                if best_idx >= 0:
                    # Apply a large priority bonus to surface it first
                    bonus = 1000.0
                    s, c, comp = scored_candidates[best_idx]
                    scored_candidates[best_idx] = (s + bonus, c, comp)
            except Exception:
                pass

        rerank_time_ms = (time.time() - start_time) * 1000

        # INFO: slot summary for observability
        try:
            logger.info(
                f"[Retrieval] SlotRouter: slot={slot_id or '-'} conf={slot_conf:.2f}; "
                f"convo {pre_counts['convo']}→{post_counts['convo']} graph {pre_counts['graph']}→{post_counts['graph']} "
                f"summary {pre_counts['summary']}→{post_counts['summary']}"
            )
        except Exception:
            pass
        logger.debug(f"[Retrieval] Composite reranking took {rerank_time_ms:.1f}ms for {len(all_candidates)} candidates")

        # Sort by composite score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Optional: restrict to top-priority source for simpler context, useful for small LMs
        try:
            single_source = os.getenv("MEMORY_SINGLE_SOURCE", "false").lower() in ("1", "true", "yes")
        except Exception:
            single_source = False
        if single_source and scored_candidates:
            top_source = source_priority[0] if source_priority else None
            if top_source:
                scored_candidates = [(s, c, comp) for s, c, comp in scored_candidates if c.source == top_source]

        # Apply token budget enforcement and cross-source deduplication
        final_bullets, selected_candidates = self._apply_token_budget_and_deduplication(
            scored_candidates, max_bullets, query
        )

        # Update usage tracking for selected graph candidates
        current_time = int(time.time() * 1000)
        for candidate in selected_candidates:
            if candidate.source == "graph" and "edge_id" in candidate.meta:
                try:
                    self.host.store.increment_edge_usage(candidate.meta["edge_id"], current_time)
                except Exception as e:
                    logger.warning(f"[Retrieval] Failed to update edge usage: {e}")

        # Log source distribution in final results
        source_counts = {}
        for bullet in final_bullets:
            for src in ["graph", "convo", "summary"]:
                if f"[{src}]" in bullet:
                    source_counts[src] = source_counts.get(src, 0) + 1
        if not final_bullets:
            logger.info(f"[Retrieval] No memory context found for query")
        else:
            logger.info(f"[Retrieval] Returning {len(final_bullets)} memory bullets from sources: {source_counts}")
        logger.debug(f"[Retrieval] final_bullets={len(final_bullets)} source_counts={source_counts}")

        return final_bullets[:max_bullets]

    def _semantic_collect_candidates(self, query: str, max_bullets: int, seen: set) -> List[Candidate]:
        """Collect semantic candidates from the optional semantic sidecar."""
        candidates = []
        
        try:
            # Try to import and use semantic sidecar
            from .semantic_sidecar import get_semantic_sidecar
            
            semantic_sidecar = get_semantic_sidecar()
            if not semantic_sidecar:
                logger.debug("[Retrieval._semantic_collect] Semantic sidecar not available")
                return candidates
            
            # Get current user and session for namespacing
            current_user = getattr(self.host, 'current_user_id', None)
            current_session = getattr(self.host, 'current_session_id', None)
            
            # Define scopes for semantic search
            scopes = {}
            if current_user:
                scopes['user_id'] = current_user
            if current_session:
                scopes['session_id'] = current_session
            
            # Query semantic sidecar
            semantic_results = semantic_sidecar.recall(
                query=query,
                k=max_bullets * 2,  # Get more to allow for filtering
                scopes=scopes,
                token_budget=100  # Semantic budget within overall budget
            )
            
            for text, score, metadata in semantic_results:
                if not text or not text.strip():
                    continue
                    
                # Skip if already seen (cross-source dedup)
                normalized_text = self._normalize_candidate_text(text)
                if normalized_text in seen:
                    continue
                
                candidate = Candidate(
                    text=text.strip(),
                    source="semantic",
                    score_hint=score,  # Semantic similarity score
                    ts=metadata.get('ts', int(time.time() * 1000)),
                    meta={
                        'similarity_score': score,
                        'kind': metadata.get('kind', 'unknown'),
                        'user_id': metadata.get('user_id'),
                        'session_id': metadata.get('session_id')
                    }
                )
                candidates.append(candidate)
                
            logger.debug(f"[Retrieval._semantic_collect] Returning {len(candidates)} semantic candidates")
            
        except ImportError:
            logger.debug("[Retrieval._semantic_collect] Semantic sidecar module not available")
        except Exception as e:
            logger.warning(f"[Retrieval._semantic_collect] Semantic search failed: {e}")
        
        return candidates

    def _apply_token_budget_and_deduplication(
        self, 
        scored_candidates: List[Tuple[float, Candidate, Dict[str, float]]], 
        max_bullets: int,
        query: str
    ) -> Tuple[List[str], List[Candidate]]:
        """
        Apply token budget enforcement and cross-source deduplication.
        
        Args:
            scored_candidates: List of (score, candidate, components) tuples
            max_bullets: Maximum number of bullets to return
            query: Original query for greeting/intent gating
            
        Returns:
            Tuple of (final_bullets, selected_candidates)
        """
        # Load token budget from environment
        try:
            max_tokens = int(os.getenv("MEMORY_TOKEN_BUDGET", "300"))
        except (ValueError, TypeError):
            max_tokens = 300
            
        # Default bullet cap (can be overridden by max_bullets)
        bullet_cap = min(max_bullets, int(os.getenv("MEMORY_MAX_BULLETS", "2")))
        
        final_bullets = []
        selected_candidates = []
        used_tokens = 0
        seen_normalized_texts = set()  # For cross-source deduplication
        
        # Enhanced greeting/intent gating
        if self._should_suppress_memory_injection(query):
            logger.debug(f"[Retrieval] Suppressing memory injection for query: '{query[:50]}...'")
            return [], []
        
        for score, candidate, components in scored_candidates:
            # Enhanced cross-source deduplication:
            # 1. Exact match on normalized text
            normalized_text = self._normalize_candidate_text(candidate.text)

            if normalized_text in seen_normalized_texts:
                logger.debug(f"[Retrieval] Skipping exact duplicate: '{candidate.text[:50]}...'")
                continue

            # 2. Semantic similarity check against all selected candidates
            # Load similarity threshold from environment
            try:
                similarity_threshold = float(os.getenv("MEMORY_DEDUP_THRESHOLD", "0.6"))
            except (ValueError, TypeError):
                similarity_threshold = 0.6

            is_duplicate = False
            for selected in selected_candidates:
                if self._are_semantically_similar(candidate.text, selected.text, similarity_threshold):
                    logger.debug(
                        f"[Retrieval] Skipping semantic duplicate: '{candidate.text[:50]}...' "
                        f"(similar to '{selected.text[:50]}...')"
                    )
                    is_duplicate = True
                    break

            if is_duplicate:
                continue
                
            # Format the bullet based on metadata format setting (for A/B testing)
            metadata_format = os.getenv("MEMORY_METADATA_FORMAT", "technical").lower()
            injection_mode = os.getenv("MEMORY_INJECTION_MODE", "bullets").lower()
            header_expand_threshold = float(os.getenv("MEMORY_HEADER_EXPAND_THRESHOLD", "0.65"))

            if metadata_format == "emoji":
                # A/B test variant A: Emoji-based metadata
                bullet = self._format_emoji_bullet(candidate, components, score)
            elif metadata_format == "minimal":
                # A/B test variant B: Minimal symbol metadata
                bullet = self._format_minimal_bullet(candidate, components, score)
            elif injection_mode == "headers":
                # Control/Technical: Original header format with numeric metadata
                bullet = self._format_header_bullet(candidate, components, score, header_expand_threshold)
            else:
                # Legacy bullet formatting (fallback)
                if candidate.source == "graph":
                    bullet = f"• [graph] {candidate.text}{self._ago_suffix(candidate.ts)}"
                elif candidate.source == "convo":
                    bullet = f"• [convo] {self._smart_truncate(candidate.text, 120)}{self._ago_suffix(candidate.ts)}"
                elif candidate.source == "semantic":
                    bullet = f"• [semantic] {self._smart_truncate(candidate.text, 140)}{self._ago_suffix(candidate.ts)}"
                else:  # summary
                    bullet = f"• [summary] {self._smart_truncate(candidate.text, 160)}{self._ago_suffix(candidate.ts)}"
            
            # Estimate token count (heuristic: chars/4 for English)
            estimated_tokens = len(bullet) // 4
            
            # Check token budget
            if used_tokens + estimated_tokens > max_tokens:
                logger.debug(f"[Retrieval] Token budget exceeded: {used_tokens + estimated_tokens} > {max_tokens}")
                break
            
            # Check bullet cap
            if len(final_bullets) >= bullet_cap:
                break
                
            # Accept this candidate
            final_bullets.append(bullet)
            selected_candidates.append(candidate)
            seen_normalized_texts.add(normalized_text)
            used_tokens += estimated_tokens
            
            # Log top-k components for debugging (optional)
            if len(final_bullets) <= 3 and os.getenv("MEMORY_LOG_COMPONENTS", "false").lower() in ("1", "true", "yes"):
                logger.debug(f"[Retrieval] Top-{len(final_bullets)} candidate components: {components}")
        
        logger.debug(f"[Retrieval] Token budget: {used_tokens}/{max_tokens}, Bullets: {len(final_bullets)}/{bullet_cap}")
        
        return final_bullets, selected_candidates
    
    def _normalize_candidate_text(self, text: str) -> str:
        """
        Normalize candidate text for cross-source deduplication.

        Removes source tags, normalizes case, removes punctuation/underscores, and collapses whitespace.
        This creates a more aggressive normalization for better duplicate detection.
        """
        import re
        # Remove source tags like [graph], [convo], etc.
        normalized = re.sub(r'\[(graph|convo|summary|semantic)\]\s*', '', text, flags=re.IGNORECASE)
        # Normalize case
        normalized = normalized.lower()
        # Replace underscores with spaces (before other punctuation removal)
        normalized = normalized.replace('_', ' ')
        # Remove punctuation and normalize whitespace
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = ' '.join(normalized.split())
        return normalized

    def _are_semantically_similar(self, text1: str, text2: str, threshold: float = 0.6) -> bool:
        """
        Check if two texts are semantically similar using Jaccard similarity.

        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold (default 0.6 = 60% overlap)

        Returns:
            True if texts are similar enough to be considered duplicates
        """
        # Normalize both texts
        norm1 = self._normalize_candidate_text(text1)
        norm2 = self._normalize_candidate_text(text2)

        # Split into word sets
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        # Handle empty sets
        if not words1 or not words2:
            return norm1 == norm2

        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return False

        similarity = intersection / union
        return similarity >= threshold

    def _is_text_question(self, text: str) -> bool:
        """
        Detect if text is a question using text patterns (fallback when prosody unavailable).

        Args:
            text: Text to check

        Returns:
            True if text appears to be a question
        """
        text_clean = text.strip()

        # Question mark is strongest indicator
        if text_clean.endswith('?'):
            return True

        # Question word starters
        question_starters = (
            'do you', 'did you', 'can you', 'could you', 'would you', 'will you', 'should you',
            'what', 'when', 'where', 'who', 'why', 'how', 'which',
            'is it', 'are you', 'have you', 'does', 'did', 'can', 'will', 'should', 'are'
        )

        text_lower = text_clean.lower()
        return any(text_lower.startswith(q) for q in question_starters)

    def _should_suppress_memory_injection(self, query: str) -> bool:
        """
        Enhanced greeting and intent gating to suppress memory injection for inappropriate queries.

        Returns True if memory injection should be suppressed.
        """
        if not query:
            return False

        q = query.strip().lower()

        # Greeting detection (expanded)
        greeting_terms = (
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "top of the morning", "howdy", "greetings", "what's up", "sup", "yo"
        )
        is_greeting = any(term in q for term in greeting_terms) and len(q.split()) <= 5
        if is_greeting:
            # Only allow name-related memories for greetings
            name_terms = ("name", "call me", "called", "my name is")
            return not any(term in q for term in name_terms)

        # Meta-conversational queries that don't need memory
        meta_queries = (
            "how are you", "how do you work", "what can you do", "who are you",
            "what are you", "help me", "assist me", "i need help"
        )
        if any(mq in q for mq in meta_queries):
            return True

        # Very short queries that are likely conversational fillers
        if len(q.split()) <= 2:
            short_fillers = {
                "ok",
                "okay",
                "sure",
                "thanks",
                "thank you",
                "cool",
                "awesome",
                "great",
                "nice",
                "sounds good",
                "all good",
                "alright",
                "right",
                "yep",
                "yeah",
                "yup",
                "no",
                "nah",
                "wow",
                "lol",
                "hmm",
                "hm",
                "mm",
            }
            if q in short_fillers:
                return True

        # Questions about capabilities or general knowledge
        # CRITICAL FIX: Don't suppress memory recall questions!
        capability_questions = (
            "can you", "will you", "would you", "could you", "should you",
            "do you know", "are you able", "is it possible"
        )
        if any(cq in q for cq in capability_questions) and "?" in q:
            # EXCEPTION: If query contains personal memory indicators, it's memory recall
            memory_indicators = ("my", "our", "we", "i", "me", "name", "dog", "cat", "pet",
                                "favorite", "friend", "family", "parent", "sibling", "child",
                                "where", "when", "what", "who", "live", "work", "from")
            if any(indicator in q for indicator in memory_indicators):
                return False  # DON'T suppress - this is memory recall!
            return True

        return False

    def _graph_collect_candidates(self, query: str, entities: List[str], turn_id: int, max_bullets: int, seen: set, allowed_relations: Optional[Set[str]] = None) -> List[Candidate]:
        """Collect graph candidates as Candidate records with metadata."""
        candidates = []
        # Similar logic to _graph_retrieve but return Candidate records instead of formatted bullets
        
        # Identity scope
        current_user = getattr(self.host, 'current_user_id', None)
        current_session = getattr(self.host, 'current_session_id', None)
        edge_scope_cache: Dict[str, bool] = {}
        # Prefer fact bullets based on query entities
        ent_set = [e for e in entities if e]
        non_you = [e for e in ent_set if e != "you"]
        include_you = any(e == "you" for e in ent_set)
        query_entities = non_you[:4]
        if include_you:
            query_entities.append("you")

        pred_pri = {
            "lives_in": 100,
            "works_at": 95,
            "born_in": 90,
            "moved_from": 85,
            "participated_in": 80,
            "friend_of": 78,
            "name": 75,
            "has": 60,
        }

        WEIGHT_MIN = WEIGHT_MIN_ACTIVE
        REL_MIN_POS: Dict[str, int] = {
            "also_known_as": 2,
        }

        now_ms = int(time.time() * 1000)
        for entity in query_entities:
            if entity in self.host.entity_index:
                candidates_list = list(self.host.entity_index[entity])
                scored: List[Tuple[float, int, str, str, str]] = []

                # Build quick lookup for (s,r)-> dst meta once per relation
                meta_cache: Dict[Tuple[str, str], Dict[str, Tuple[float, int, int, int, int]]] = {}

                # P0.1: Use persistent LRU cache for provenance checks
                # First check cache, then batch query uncached edges
                candidate_edge_ids = [self.host.store.edge_id(s, r, d) for s, r, d in candidates_list]

                # Check which edges are already cached (across all retrieval calls)
                for edge_id in candidate_edge_ids:
                    if edge_id not in edge_scope_cache:
                        # Try session-scoped cache first (persists across retrieve() calls, isolated per session)
                        try:
                            if current_session:
                                cached_result = self._provenance_cache.check_visibility(
                                    current_session, edge_id, current_user, self._check_edge_visibility_impl
                                )
                            else:
                                cached_result = self._check_edge_visibility_impl(edge_id, current_user, current_session)
                            edge_scope_cache[edge_id] = cached_result
                        except Exception as e:
                            logger.debug(f"[Retrieval] Cache check failed for {edge_id}: {e}")
                            # Will be handled by batch query below
                            pass

                # Batch query any edges not in cache
                uncached_edge_ids = [eid for eid in candidate_edge_ids if eid not in edge_scope_cache]

                if uncached_edge_ids:
                    try:
                        prov_batch = self.host.store.get_edges_provenance_batch(uncached_edge_ids)
                    except Exception:
                        prov_batch = {}

                    all_session_ids = set()
                    for prov_list in prov_batch.values():
                        for (_text, sess_id, _turn, _ts) in prov_list:
                            all_session_ids.add(sess_id)

                    if current_user and all_session_ids:
                        try:
                            owned_sessions = self.host.store.are_sessions_owned_by_user_batch(list(all_session_ids), current_user)
                        except Exception:
                            owned_sessions = set()
                    else:
                        owned_sessions = set()

                    for edge_id in uncached_edge_ids:
                        prov = prov_batch.get(edge_id, [])
                        allowed_edge = False
                        if current_user:
                            for (_text, sess_id, _turn, _ts) in prov:
                                if sess_id in owned_sessions:
                                    allowed_edge = True
                                    break
                        elif current_session:
                            allowed_edge = any(sess_id == current_session for (_text, sess_id, _turn, _ts) in prov)
                        edge_scope_cache[edge_id] = allowed_edge

                for s, r, d in candidates_list:
                    edge_id = self.host.store.edge_id(s, r, d)
                    if not edge_scope_cache.get(edge_id, False):
                        continue

                    key = (s, r)
                    if key not in meta_cache:
                        try:
                            neigh = self.host.store.neighbors(s, r)
                            meta_cache[key] = {
                                dst: (float(w), int(nts), int(pos), int(neg), int(st))
                                for (dst, w, nts, pos, neg, st) in neigh
                            }
                        except Exception:
                            meta_cache[key] = {}

                    w, ts, pos, neg, status = 1.0, 0, 0, 0, 1
                    meta = meta_cache.get(key, {}).get(d)
                    if meta is not None:
                        w, ts, pos, neg, status = meta

                    if status <= 0 or pos <= neg or w < WEIGHT_MIN:
                        continue
                    if allowed_relations is not None and r not in allowed_relations:
                        continue
                    if r == "also_known_as" and s.lower() != "you":
                        continue
                    min_pos = REL_MIN_POS.get(r, 1)
                    if pos < min_pos:
                        if ("happen" in r or "feel" in r) and pos < 2:
                            continue
                        if r in ("quality", "quantity"):
                            continue

                    human = self._humanize_fact(s, r, d)
                    if not human:
                        continue
                    
                    if human in seen:
                        continue

                    pri = pred_pri.get(r, 50)
                    support = 1.0 + min(max(pos, 0), 5) * 0.1
                    age_ms = max(0, now_ms - int(ts)) if ts else 0
                    half_life_ms = RECENCY_HALF_LIFE_MS
                    recency_factor = (2 ** (-(age_ms / half_life_ms))) if ts else 0.8
                    
                    candidates.append(Candidate(
                        text=human,
                        source="graph",
                        score_hint=0.0,  # Not used for graph
                        ts=ts,
                        meta={
                            "edge_id": edge_id,
                            "weight": w,
                            "pos": pos,
                            "neg": neg,
                            "priority": pri,
                            "support": support
                        }
                    ))
                    
                    seen.add(human)
                    if len(candidates) >= max_bullets:
                        return candidates

        # Fallback to recency
        if allowed_relations is None:
            for item in reversed(list(self.host.recency_buffer)[-10:]):
                human = self._humanize_fact(item.s, item.r, item.d)
                if not human or human in seen:
                    continue
                    
                ts = item.timestamp if hasattr(item, 'timestamp') else 0
                candidates.append(Candidate(
                    text=human,
                    source="graph",
                    score_hint=0.0,
                    ts=ts,
                    meta={"edge_id": self.host.store.edge_id(item.s, item.r, item.d)}
                ))
                
                seen.add(human)
                if len(candidates) >= max_bullets:
                    break

        return candidates[:max_bullets]

    def _convo_collect_candidates(self, query: str, max_bullets: int, seen: set, slot_id: Optional[str] = None) -> List[Candidate]:
        """Collect conversation candidates as Candidate records with BM25 scores."""
        candidates = []
        
        try:
            # Try Enhanced FTS first
            try:
                from .enhanced_fts import EnhancedFTS
                enhanced_fts = EnhancedFTS(self.host.store)

                # User-wide scope by default: include all sessions owned by current user
                session_id = getattr(self.host, 'current_session_id', None)
                user_id = getattr(self.host, 'current_user_id', None)
                allowed_sessions = []
                try:
                    if user_id and hasattr(self.host.store, 'get_sessions_by_user'):
                        allowed_sessions = list(self.host.store.get_sessions_by_user(user_id) or [])
                except Exception:
                    allowed_sessions = []
                if not allowed_sessions and session_id:
                    # Fallback to current session if user mapping not available
                    allowed_sessions = [session_id]

                enhanced_results = enhanced_fts.enhanced_search(query, max_bullets * 2, session_ids=allowed_sessions, slot_id=slot_id)
                hits = [(score, text, eid, ts, turn_id) for score, text, eid, ts, turn_id in enhanced_results]

                logger.debug(f"[Retrieval._convo_collect] Enhanced FTS returned {len(hits)} hits")
                if not hits:
                    import re
                    sanitized = re.sub(r'[^\w\s]', ' ', query)
                    sanitized = ' '.join(sanitized.split())
                    if not sanitized.strip():
                        return []
                    if session_id and hasattr(self.host.store, 'search_fts_scoped'):
                        # Note: basic FTS scopes by eid; conversation turns are stored with eid='conversation'.
                        # As a conservative fallback, use global search to avoid scoping everything away.
                        hits = [(0.0, text, eid, ts, None) for text, eid, ts in self.host.store.search_fts(sanitized, limit=max_bullets * 2)]
                    else:
                        hits = [(0.0, text, eid, ts, None) for text, eid, ts in self.host.store.search_fts(sanitized, limit=max_bullets * 2)]
                
            except ImportError:
                logger.debug("[Retrieval._convo_collect] Enhanced FTS not available")
                import re
                sanitized = re.sub(r'[^\w\s]', ' ', query)
                sanitized = ' '.join(sanitized.split())

                if not sanitized.strip():
                    return []

                user_id = getattr(self.host, 'current_user_id', None)
                session_id = getattr(self.host, 'current_session_id', None)
                allowed = [e for e in [user_id, session_id] if e]
                if allowed and hasattr(self.host.store, 'search_fts_scoped'):
                    hits = [(0.0, text, eid, ts, None) for text, eid, ts in self.host.store.search_fts_scoped(sanitized, allowed, limit=max_bullets * 2)]
                else:
                    hits = [(0.0, text, eid, ts, None) for text, eid, ts in self.host.store.search_fts(sanitized, limit=max_bullets * 2)]
            
        except Exception as e:
            logger.warning(f"[Retrieval._convo_collect] FTS search failed: {e}")
            hits = []

        # Filter out excluded phrases
        excluded = []
        try:
            ex_raw = os.getenv("EXCLUDED_MEMORY_PHRASES", "").strip()
            fixed = os.getenv("ENROLLMENT_FIXED_PHRASE", "").strip()
            excluded = [p.strip().lower() for p in ex_raw.split("||") if p.strip()]
            if fixed:
                excluded.append(fixed.lower())
        except Exception:
            pass

        for hit in hits:
            # Handle the new (score, text, eid, ts, turn_id) format
            if len(hit) >= 5:
                bm25_score, text, eid, ts, turn_id = hit[:5]
            elif len(hit) >= 4:
                bm25_score, text, eid, ts = hit[:4]
                turn_id = None  # Fallback when turn_id not available
            else:
                continue
                
            # Only allow conversation rows; filter summaries and other eids
            if not eid or eid != "conversation":
                continue
                
            s = text.strip().replace("\n", " ")
            if not s or s in seen:
                continue
                
            if excluded:
                tl = s.lower()
                if any(p in tl for p in excluded):
                    continue

            # Apply quality filter
            if not self._is_quality_bullet(s):
                continue

            # Filter out questions (they confuse the agent and shouldn't be used as context)
            if self._is_text_question(s):
                logger.debug(f"[Retrieval._convo] Skipping question: '{s[:50]}...'")
                continue

            candidates.append(Candidate(
                text=s,
                source="convo",
                score_hint=bm25_score,
                ts=ts,
                meta={"bm25_score": bm25_score, "eid": eid, "turn_id": turn_id}
            ))

            seen.add(s)
            if len(candidates) >= max_bullets:
                break

        return candidates

    def _summary_collect_candidates(self, max_bullets: int, seen: set) -> List[Candidate]:
        """Collect summary candidates as Candidate records."""
        candidates = []
        
        try:
            rows = self.host.store.get_recent_chunks_by_eid("summary", limit=max_bullets * 2)
        except Exception as e:
            logger.warning(f"[Retrieval._summary_collect] Failed to load summary chunks: {e}")
            return candidates

        # TTL: 7 days for summaries
        current_time = int(time.time() * 1000)
        ttl_ms = 7 * 24 * 60 * 60 * 1000

        for text, ts in rows:
            if current_time - ts > ttl_ms:
                continue
                
            s = text.strip().replace("\n", " ")
            if not s or s in seen:
                continue

            if not self._is_quality_bullet(s):
                continue

            candidates.append(Candidate(
                text=s,
                source="summary",
                score_hint=0.5,  # Constant prior for summaries
                ts=ts,
                meta={}
            ))
            
            seen.add(s)
            if len(candidates) >= max_bullets:
                break

        return candidates

    def _allocate_budget(self, max_bullets: int, enabled_sources: List[str]) -> Dict[str, int]:
        """
        Allocate retrieval budget across sources to prevent starvation.

        Strategy:
        - Get MORE candidates from each source than max_bullets
        - Let re-ranking decide final selection
        - This ensures diversity without starving any source
        """
        budget = {}

        if max_bullets <= 0:
            return budget

        # Count active sources
        active_sources = [s for s in ["graph", "convo", "summary"] if s in enabled_sources]

        if not active_sources:
            return budget

        # Give each source a generous budget to ensure candidates
        # We'll let re-ranking pick the best max_bullets from all sources
        # This is key: don't limit sources too early!
        per_source_budget = max(max_bullets, 3)  # At least 3 per source, or max_bullets if higher

        for source in active_sources:
            budget[source] = per_source_budget

        return budget

    def _graph_retrieve(self, query: str, entities: List[str], turn_id: int, max_bullets: int, seen: set, allowed_relations: Optional[Set[str]] = None) -> List[str]:
        out: List[str] = []
        # Identity scope
        current_user = getattr(self.host, 'current_user_id', None)
        current_session = getattr(self.host, 'current_session_id', None)
        edge_scope_cache: Dict[str, bool] = {}
        # Prefer fact bullets based on query entities
        ent_set = [e for e in entities if e]
        non_you = [e for e in ent_set if e != "you"]
        include_you = any(e == "you" for e in ent_set)
        query_entities = non_you[:4]
        if include_you:
            query_entities.append("you")

        pred_pri = {
            "lives_in": 100,
            "works_at": 95,
            "born_in": 90,
            "moved_from": 85,
            "participated_in": 80,
            "friend_of": 78,
            "name": 75,
            "has": 60,
        }

        WEIGHT_MIN = WEIGHT_MIN_ACTIVE  # Align with status thresholding used by store
        # Stricter minimum positive support for noisy relations
        REL_MIN_POS: Dict[str, int] = {
            "also_known_as": 2,
        }

        now_ms = int(time.time() * 1000)
        for entity in query_entities:
            if entity in self.host.entity_index:
                candidates = list(self.host.entity_index[entity])
                scored: List[Tuple[float, int, str, str, str]] = []

                # Build quick lookup for (s,r)-> dst meta once per relation
                meta_cache: Dict[Tuple[str, str], Dict[str, Tuple[float, int, int, int, int]]] = {}

                # P0.1: Use persistent LRU cache for provenance checks
                # First check cache, then batch query uncached edges
                candidate_edge_ids = [self.host.store.edge_id(s, r, d) for s, r, d in candidates]

                # Check which edges are already cached (across all retrieval calls)
                for edge_id in candidate_edge_ids:
                    if edge_id not in edge_scope_cache:
                        # Try session-scoped cache first (persists across retrieve() calls, isolated per session)
                        try:
                            if current_session:
                                cached_result = self._provenance_cache.check_visibility(
                                    current_session, edge_id, current_user, self._check_edge_visibility_impl
                                )
                            else:
                                cached_result = self._check_edge_visibility_impl(edge_id, current_user, current_session)
                            edge_scope_cache[edge_id] = cached_result
                        except Exception as e:
                            logger.debug(f"[Retrieval] Cache check failed for {edge_id}: {e}")
                            # Will be handled by batch query below
                            pass

                # Batch query any edges not in cache
                uncached_edge_ids = [eid for eid in candidate_edge_ids if eid not in edge_scope_cache]

                if uncached_edge_ids:
                    # Batch query all provenance data
                    try:
                        prov_batch = self.host.store.get_edges_provenance_batch(uncached_edge_ids)
                    except Exception:
                        prov_batch = {}

                    # Collect all unique session_ids from provenance for batch ownership check
                    all_session_ids = set()
                    for prov_list in prov_batch.values():
                        for (_text, sess_id, _turn, _ts) in prov_list:
                            all_session_ids.add(sess_id)

                    # Batch query session ownership if we have a current user
                    if current_user and all_session_ids:
                        try:
                            owned_sessions = self.host.store.are_sessions_owned_by_user_batch(list(all_session_ids), current_user)
                        except Exception:
                            owned_sessions = set()
                    else:
                        owned_sessions = set()

                    # Populate edge_scope_cache from batch results
                    for edge_id in uncached_edge_ids:
                        prov = prov_batch.get(edge_id, [])
                        allowed_edge = False
                        if current_user:
                            # Check if any session belongs to user
                            for (_text, sess_id, _turn, _ts) in prov:
                                if sess_id in owned_sessions:
                                    allowed_edge = True
                                    break
                        elif current_session:
                            # Fallback: allow edges from current session only
                            allowed_edge = any(sess_id == current_session for (_text, sess_id, _turn, _ts) in prov)
                        edge_scope_cache[edge_id] = allowed_edge

                # Now filter candidates using the populated cache
                for s, r, d in candidates:
                    edge_id = self.host.store.edge_id(s, r, d)
                    if not edge_scope_cache.get(edge_id, False):
                        continue

                    # Retrieve neighbor meta for this (s,r) only once
                    key = (s, r)
                    if key not in meta_cache:
                        try:
                            neigh = self.host.store.neighbors(s, r)
                            meta_cache[key] = {
                                dst: (float(w), int(nts), int(pos), int(neg), int(st))
                                for (dst, w, nts, pos, neg, st) in neigh
                            }
                        except Exception:
                            meta_cache[key] = {}

                    w, ts, pos, neg, status = 1.0, 0, 0, 0, 1
                    meta = meta_cache.get(key, {}).get(d)
                    if meta is not None:
                        w, ts, pos, neg, status = meta

                    # Skip edges that are stale/negated/weak
                    if status <= 0:
                        continue
                    if pos <= neg:
                        continue
                    if w < WEIGHT_MIN:
                        continue
                    # Optional relation allowlist (e.g., for greetings)
                    if allowed_relations is not None and r not in allowed_relations:
                        continue
                    # Disallow AKA unless it's explicitly about the user
                    if r == "also_known_as" and s.lower() != "you":
                        continue
                    # Per‑relation support requirements
                    min_pos = REL_MIN_POS.get(r, 1)
                    if pos < min_pos:
                        # Additional guard for families of noisy relations
                        if ("happen" in r or "feel" in r) and pos < 2:
                            continue
                        if r in ("quality", "quantity"):
                            continue

                    pri = pred_pri.get(r, 50)
                    # Composite score: priority × weight × support × recency
                    support = 1.0 + min(max(pos, 0), 5) * 0.1  # dampen large pos
                    age_ms = max(0, now_ms - int(ts)) if ts else 0
                    # Recency decay
                    half_life_ms = RECENCY_HALF_LIFE_MS
                    recency_factor = (2 ** (-(age_ms / half_life_ms))) if ts else 0.8
                    score = float(pri) * float(max(w, 0.01)) * support * recency_factor
                    scored.append((score, ts, s, r, d))
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                for _score, _ts, s, r, d in scored:
                    suffix = self._ago_suffix(_ts)
                    human = self._humanize_fact(s, r, d)
                    if not human:
                        continue
                    key = human  # Dedup based on humanized fact to collapse alias/case variants
                    if key in seen:
                        continue
                    out.append(f"• [graph] {human}{suffix}")
                    seen.add(key)
                    if len(out) >= max_bullets:
                        return out

        # Fallback to recency unless an allowlist is active
        if allowed_relations is None:
            for item in reversed(list(self.host.recency_buffer)[-10:]):
                age = self._ago_suffix(item.timestamp if hasattr(item, 'timestamp') else 0)
                human = self._humanize_fact(item.s, item.r, item.d)
                if not human:
                    continue
                key = human
                if key in seen:
                    continue
                out.append(f"• [graph] {human}{age}")
                seen.add(key)
                if len(out) >= max_bullets:
                    break

        return out[:max_bullets]

    def _is_quality_bullet(self, text: str) -> bool:
        """
        LAYER 4 DEFENSE: Enhanced quality filter for retrieved conversation bullets.
        Returns True if the bullet should be shown, False if filtered out.

        REFACTORED: Delegates to QualityFilter.is_quality_for_retrieval()
        """
        return self.quality_filter.is_quality_for_retrieval(text)

    def _convo_retrieve(self, query: str, max_bullets: int, seen: set) -> List[str]:
        out: List[str] = []
        try:
            # Try Enhanced FTS first (SOTA implementation)
            try:
                from .enhanced_fts import EnhancedFTS
                enhanced_fts = EnhancedFTS(self.host.store)

                # User-wide scope by default for conversation retrieval
                session_id = getattr(self.host, 'current_session_id', None)
                user_id = getattr(self.host, 'current_user_id', None)
                allowed_sessions = []
                try:
                    if user_id and hasattr(self.host.store, 'get_sessions_by_user'):
                        allowed_sessions = list(self.host.store.get_sessions_by_user(user_id) or [])
                except Exception:
                    allowed_sessions = []
                if not allowed_sessions and session_id:
                    allowed_sessions = [session_id]

                # Use enhanced search with BM25 and query expansion, scoped by user sessions
                enhanced_results = enhanced_fts.enhanced_search(query, max_bullets * 2, session_ids=allowed_sessions)
                hits = [(text, eid, ts) for score, text, eid, ts in enhanced_results]

                logger.debug(f"[Retrieval._convo] Enhanced FTS returned {len(hits)} hits for query='{query[:30]}'")
                # If enhanced index is empty or produced no hits, fall back to basic FTS
                if not hits:
                    import re
                    sanitized = re.sub(r'[^\w\s]', ' ', query)
                    sanitized = ' '.join(sanitized.split())
                    if not sanitized.strip():
                        logger.debug(f"[Retrieval._convo] Query sanitized to empty string (enhanced fallback)")
                        return []
                    # Basic FTS scopes by eid, which doesn't match convo rows (eid='conversation').
                    # Use global FTS as a conservative fallback.
                    if hasattr(self.host.store, 'search_fts'):
                        hits = self.host.store.search_fts(sanitized, limit=max_bullets * 2)
                    else:
                        hits = self.host.store.search_fts(sanitized, limit=max_bullets * 2)
                
            except ImportError:
                logger.debug("[Retrieval._convo] Enhanced FTS not available, using basic FTS")
                # Fallback to basic FTS with sanitization
                import re
                sanitized = re.sub(r'[^\w\s]', ' ', query)  # Keep only alphanumeric and spaces
                sanitized = ' '.join(sanitized.split())  # Normalize whitespace

                if not sanitized.strip():
                    logger.debug(f"[Retrieval._convo] Query sanitized to empty string")
                    return []

                # Use global FTS for fallback to avoid scoping away convo rows
                hits = self.host.store.search_fts(sanitized, limit=max_bullets * 2)
            
            # Initialize sanitized for logging only
            sanitized = (query or '')
            logger.debug(f"[Retrieval._convo] FTS returned {len(hits)} hits for query='{sanitized[:30]}'")
        except Exception as e:
            logger.warning(f"[Retrieval._convo] FTS search failed: {e}")
            hits = []
        # Exclude enrollment/fixed phrases from retrieval context
        excluded = []
        try:
            import os
            ex_raw = os.getenv("EXCLUDED_MEMORY_PHRASES", "").strip()
            fixed = os.getenv("ENROLLMENT_FIXED_PHRASE", "").strip()
            excluded = [p.strip().lower() for p in ex_raw.split("||") if p.strip()]
            if fixed:
                excluded.append(fixed.lower())
        except Exception as e:
            logger.debug(f"[Retrieval._convo] Failed to load excluded phrases: {e}")
        for text, eid, ts in hits:
            logger.debug(f"[Retrieval._convo] Processing hit: eid='{eid}' text='{text[:40]}'")
            # Only allow conversation entries (not summary/mentions)
            if not eid or eid != "conversation":
                logger.debug(f"[Retrieval._convo] Skipping non-convo eid: {eid}")
                continue
            s = text.strip().replace("\n", " ")
            if not s:
                logger.debug(f"[Retrieval._convo] Skipping empty text")
                continue
            if excluded:
                tl = s.lower()
                if any(p in tl for p in excluded):
                    logger.debug("[Retrieval._convo] Skipping excluded phrase hit")
                    continue

            # LAYER 4 DEFENSE: Apply quality filter to retrieved bullets
            if not self._is_quality_bullet(s):
                logger.debug(f"[Retrieval._convo] Filtered low-quality bullet: '{s[:50]}...'")
                continue

            # Use smart truncation to avoid cutting mid-word
            bullet = f"• [convo] {self._smart_truncate(s, 120)}{self._ago_suffix(ts)}"
            if bullet in seen:
                logger.debug(f"[Retrieval._convo] Skipping duplicate bullet")
                continue
            seen.add(bullet)
            out.append(bullet)
            logger.debug(f"[Retrieval._convo] Added bullet: {bullet[:60]}")
            if len(out) >= max_bullets:
                break
        logger.debug(f"[Retrieval._convo] Returning {len(out)} bullets")
        return out

    def _smart_truncate(self, text: str, max_len: int = 120) -> str:
        """
        Smart truncation that respects word boundaries.

        Args:
            text: Text to truncate
            max_len: Maximum length in characters

        Returns:
            Truncated text with ellipsis if needed
        """
        if len(text) <= max_len:
            return text

        # Truncate at last space before max_len
        truncated = text[:max_len].rsplit(' ', 1)[0]

        # If no space found (single long word), hard truncate with ellipsis
        if not truncated:
            return text[:max_len] + "..."

        return truncated + "..."

    def _summary_retrieve(self, max_bullets: int, seen: set) -> List[str]:
        out: List[str] = []
        try:
            rows = self.host.store.get_recent_chunks_by_eid("summary", limit=max_bullets * 2)
        except Exception as e:
            logger.warning(f"[Retrieval._summary] Failed to load summary chunks: {e}")
            rows = []

        # TTL: 7 days for summaries
        current_time = int(time.time() * 1000)
        ttl_ms = 7 * 24 * 60 * 60 * 1000  # 7 days in milliseconds

        for text, ts in rows:
            # TTL filter: skip summaries older than 7 days
            if current_time - ts > ttl_ms:
                age_days = (current_time - ts) // (24 * 60 * 60 * 1000)
                logger.debug(f"[Retrieval._summary] Skipping stale summary (age={age_days}d)")
                continue

            s = text.strip().replace("\n", " ")
            if not s:
                continue

            # LAYER 4 DEFENSE: Apply quality filter to summaries
            if not self._is_quality_bullet(s):
                logger.debug(f"[Retrieval._summary] Filtered low-quality summary: '{s[:50]}...'")
                continue

            # Use smart truncation to avoid cutting mid-word
            bullet = f"• [summary] {self._smart_truncate(s, 160)}{self._ago_suffix(ts)}"
            if bullet in seen:
                continue
            seen.add(bullet)
            out.append(bullet)
            if len(out) >= max_bullets:
                break
        return out

    def _ago_suffix(self, ts_ms: int) -> str:
        try:
            if not ts_ms:
                return ""
            now_ms = int(time.time() * 1000)
            delta = max(0, now_ms - int(ts_ms))
            sec = delta // 1000
            if sec < 60:
                return f" ({sec}s ago)"
            mins = sec // 60
            if mins < 60:
                return f" ({mins}m ago)"
            hours = mins // 60
            if hours < 24:
                return f" ({hours}h ago)"
            days = hours // 24
            # compact format: include days and remaining hours
            rem_h = hours % 24
            if rem_h > 0:
                return f" ({days}d {rem_h}h ago)"
            return f" ({days}d ago)"
        except Exception:
            # Keep suffix optional on failure
            return ""

    def _humanize_fact(self, s: str, r: str, d: str) -> str:
        """Convert (s,r,d) to a compact English fragment.

        Applies conservative filtering for conversational/command relations and
        fixes common agreement issues for second-person subjects.
        """
        # Role-aware display mapping
        def _display(x: str) -> str:
            try:
                if x.startswith('you:'):
                    return x.split(':', 1)[1]
                if x.startswith('agent:'):
                    return x.split(':', 1)[1]
            except Exception as e:
                logger.debug(f"[Retrieval._humanize_fact] display mapping failed: {e}")
            return x

        ds = _display(s)
        dd = _display(d)

        meaningless_entities = {"it", "this", "that", "there", "here", "been", "know"}
        wh_words = {"what", "who", "when", "where", "why", "how", "which"}
        if ds.lower() in meaningless_entities or dd.lower() in meaningless_entities:
            return ""
        # Drop obvious punctuation artifacts
        if "," in ds or "," in dd:
            return ""
        if ds.lower() in wh_words or dd.lower() in wh_words:
            return ""

        stop_relations = {
            "and",
            "know",
            "remember",
            "say",
            "tell",
            "think",
            "ask",
            "quality",
            "quantity",
            "tell_about",
            "talk",
            "talk_about",
            "delete",
            "remove",
            # keep also_known_as with support gating; suppress generic variants
            "known",
            "known_as",
        }
        if r in stop_relations:
            return ""

        if r == "name":
            return f"{ds} is named {dd}"
        if r == "has":
            return f"{ds} has {dd}"
        if r == "also_known_as":
            # Only meaningful for user identity
            if ds.lower() not in ("you", _display(getattr(self.host, 'user_eid', 'you'))):
                return ""
            return f"{ds} aka {dd}"
        if r == "is":
            if ds.lower() in meaningless_entities or dd.lower().startswith("what "):
                return ""
            return f"{ds} is {dd}"
        if r.startswith("v:"):
            return f"{ds} {r[2:]} {dd}"
        # Common relation fixes (remove underscore)
        if r == "lives_in":
            return f"{ds} lives in {dd}"
        if r == "works_at":
            return f"{ds} works at {dd}"
        if r == "works_in":
            return f"{ds} works in {dd}"

        return f"{ds} {r.replace('_', ' ')} {dd}"

    def _is_temporal_query(self, query: str) -> bool:
        """Detect if query is asking about time-based information."""
        q = query.lower()
        temporal_keywords = {
            "yesterday", "today", "recently", "last week", "last month",
            "earlier", "before", "ago", "just now", "this morning",
            "this afternoon", "this evening", "last night", "past"
        }
        return any(kw in q for kw in temporal_keywords)

    def _is_semantic_query(self, query: str) -> bool:
        """Detect if query is asking about topics/concepts rather than facts."""
        q = query.lower()
        semantic_indicators = {
            "about", "related to", "regarding", "concerning",
            "hobbies", "interests", "preferences", "likes", "favorites",
            "thoughts on", "opinion", "views", "feelings"
        }
        return any(ind in q for ind in semantic_indicators)

    def _load_composite_weights(self) -> Dict[str, float]:
        """Load composite scoring weights from environment or use defaults."""
        default_weights = {
            "wsrc": 0.1,      # Source bias
            "wconf": 0.35,    # Confidence/support
            "wrec": 0.25,     # Recency
            "wuse": 0.1,      # Usage boost
            "wsim": 0.15,     # Semantic similarity
            "wdiv": 0.05      # Diversity penalty (negative weight)
        }
        
        # Allow individual source weight configuration via environment
        source_weights = {
            "MEMORY_WEIGHT_GRAPH": 0.3,
            "MEMORY_WEIGHT_CONVO": 0.4, 
            "MEMORY_WEIGHT_SUMMARY": 0.2,
            "MEMORY_WEIGHT_SEMANTIC": 0.1
        }
        
        try:
            # Load JSON composite weights first
            weights_json = os.getenv("MEMORY_RERANK_WEIGHTS")
            if weights_json:
                custom_weights = json.loads(weights_json)
                for key, value in custom_weights.items():
                    if key in default_weights and isinstance(value, (int, float)):
                        default_weights[key] = float(value)
                        
            # Load individual source weights
            for env_key, default_val in source_weights.items():
                env_val = os.getenv(env_key)
                if env_val is not None:
                    try:
                        source_weights[env_key] = float(env_val)
                    except ValueError:
                        logger.warning(f"[Retrieval] Invalid {env_key}: {env_val}, using default")
                        
            logger.debug(f"[Retrieval] Using composite weights: {default_weights}")
            logger.debug(f"[Retrieval] Using source weights: {source_weights}")
            
            # Store source weights for later use
            self.source_weights = source_weights
            
        except Exception as e:
            logger.warning(f"[Retrieval] Failed to load weights: {e}, using defaults")
            self.source_weights = source_weights
        
        return default_weights
    
    def _get_embedding_reranker(self):
        """Get or create the embedding reranker instance."""
        if self._embedding_reranker is None:
            try:
                from .rerank_embeddings import get_embedding_reranker
                self._embedding_reranker = get_embedding_reranker()
            except ImportError:
                self._embedding_reranker = None
        return self._embedding_reranker
    
    def _composite_score(self, query: str, candidate: Candidate, source_priority: List[str] = None, other_candidates: List[Candidate] = None) -> Tuple[float, Dict[str, float]]:
        """
        Compute composite score for a candidate with diversity penalty.
        
        Args:
            query: User query text
            candidate: Candidate to score
            source_priority: Priority order for sources
            other_candidates: Other candidates for diversity calculation
            
        Returns tuple of (total_score, component_scores) for debugging.
        """
        components = {}
        
        # Source bias (wsrc) - use configured source weights
        if source_priority is None:
            source_priority = self._get_source_priority(query)
        
        # Use environment-configured source weights
        source_weight_map = {
            "graph": getattr(self, 'source_weights', {}).get("MEMORY_WEIGHT_GRAPH", 0.3),
            "convo": getattr(self, 'source_weights', {}).get("MEMORY_WEIGHT_CONVO", 0.4),
            "summary": getattr(self, 'source_weights', {}).get("MEMORY_WEIGHT_SUMMARY", 0.2),
            "semantic": getattr(self, 'source_weights', {}).get("MEMORY_WEIGHT_SEMANTIC", 0.1)
        }
        
        wsrc = source_weight_map.get(candidate.source, 0.1)
        components["wsrc"] = wsrc
        
        # Confidence (wconf)
        if candidate.source == "graph":
            # For graph: combine weight and support (pos > neg)
            weight = candidate.meta.get("weight", 0.0)
            pos = candidate.meta.get("pos", 0)
            neg = candidate.meta.get("neg", 0)
            support = 1.0 + min(max(pos, 0), 5) * 0.1  # Same as in _graph_retrieve
            confidence = max(weight, 0.01) * support
        elif candidate.source == "convo":
            # For convo: use BM25 score, normalized
            confidence = candidate.meta.get("bm25_score", candidate.score_hint)
            if isinstance(confidence, (int, float)):
                confidence = min(1.0, max(0.0, confidence / 10.0))  # Normalize BM25 ~0-10 to 0-1
        elif candidate.source == "semantic":
            # For semantic: use provided score or fallback
            similarity_score = candidate.meta.get("similarity_score")
            semantic_score = candidate.meta.get("semantic_score")
            confidence = semantic_score if semantic_score is not None else (similarity_score if similarity_score is not None else candidate.score_hint)
        else:  # summary
            # For summary: use constant prior
            confidence = 0.5
        
        # Normalize confidence to [0, 1] range
        confidence = max(0.0, min(1.0, confidence))
        components["wconf"] = confidence * self.weights["wconf"]
        
        # Recency (wrec)
        now_ms = self._scoring_clock_ms or (int(time.time()) * 1000)
        age_ms = max(0, now_ms - candidate.ts) if candidate.ts else float('inf')
        recency_factor = (2 ** (-(age_ms / RECENCY_HALF_LIFE_MS))) if candidate.ts else 0.0
        components["wrec"] = recency_factor * self.weights["wrec"]
        
        # Usage boost (wuse)
        if candidate.source == "graph" and "edge_id" in candidate.meta:
            edge_id = candidate.meta["edge_id"]
            access_count, last_accessed = self.host.store.get_edge_usage(edge_id)
            # Logarithmic boost to avoid runaway popularity
            usage_boost = math.log1p(access_count) / math.log1p(10)  # Normalize to ~[0, 1] for up to 10 uses
            components["wuse"] = usage_boost * self.weights["wuse"]
        else:
            components["wuse"] = 0.0
        
        # Semantic similarity (wsim) - optional
        reranker = self._get_embedding_reranker()
        if reranker and os.getenv("MEMORY_RERANK_EMBEDDINGS_ENABLED", "false").lower() in ("1", "true", "yes"):
            similarities = reranker.similarity(query, [candidate.text])
            wsim = similarities[0] if similarities else 0.0
            components["wsim"] = wsim * self.weights["wsim"]
        else:
            components["wsim"] = 0.0
        
        # Prosody component (wpro) - for convo candidates only
        wpro = 0.0
        if candidate.source == "convo":
            # Get prosody weight from environment
            prosody_weight = float(os.getenv("MEMORY_WEIGHT_PROSODY", "0.15"))
            
            if prosody_weight > 0.0:
                wpro = self._calculate_prosody_component(candidate)
                components["wpro"] = wpro * prosody_weight
            else:
                components["wpro"] = 0.0
        else:
            components["wpro"] = 0.0
        
        # Diversity penalty (wdiv) - penalize similar candidates
        diversity_penalty = 0.0
        if other_candidates and "wdiv" in self.weights:
            diversity_penalty = self._calculate_diversity_penalty(candidate, other_candidates)
            components["wdiv"] = -diversity_penalty * self.weights["wdiv"]  # Negative penalty
        else:
            components["wdiv"] = 0.0
        
        # Total score
        total_score = sum(components.values())
        
        return total_score, components
    
    def _calculate_diversity_penalty(self, candidate: Candidate, other_candidates: List[Candidate]) -> float:
        """
        Calculate diversity penalty to avoid duplicate/redundant information.
        
        Args:
            candidate: Current candidate to penalize
            other_candidates: List of other candidates to compare against
            
        Returns:
            Diversity penalty score [0, 1] where higher means more penalty
        """
        if not other_candidates:
            return 0.0
            
        # Normalize candidate text for comparison
        candidate_norm = self._normalize_for_diversity(candidate.text)
        max_similarity = 0.0
        
        for other in other_candidates:
            if other is candidate:
                continue  # Skip self-comparison

            # REMOVED SOURCE RESTRICTION: Now compares across all sources
            # This enables cross-source diversity penalty to catch duplicates
                
            other_norm = self._normalize_for_diversity(other.text)
            
            # Simple similarity metrics
            # 1. Jaccard similarity of word sets
            candidate_words = set(candidate_norm.split())
            other_words = set(other_norm.split())
            
            if candidate_words and other_words:
                jaccard = len(candidate_words & other_words) / len(candidate_words | other_words)
                max_similarity = max(max_similarity, jaccard)
            
            # 2. Exact string overlap (for short texts)
            if len(candidate_norm) < 50 and len(other_norm) < 50:
                overlap = len(set(candidate_norm) & set(other_norm)) / max(len(set(candidate_norm)), len(set(other_norm)))
                max_similarity = max(max_similarity, overlap)
        
        return min(1.0, max_similarity)
    
    def _normalize_for_diversity(self, text: str) -> str:
        """Normalize text for diversity comparison."""
        # Lowercase, remove punctuation, normalize spaces
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        return text

    def _calculate_prosody_component(self, candidate: Candidate) -> float:
        """
        Calculate prosody component for convo candidates.
        
        Args:
            candidate: Convo candidate to score
            
        Returns:
            Prosody certainty value [0, 1]
        """
        # Get session_id and turn_id from candidate metadata or host
        session_id = getattr(self.host, 'current_session_id', None)
        turn_id = candidate.meta.get('turn_id')
        
        if not session_id or turn_id is None:
            return 0.0
        
        # Check cache first
        cache_key = (session_id, turn_id)
        if cache_key in self._prosody_cache:
            certainty, _ = self._prosody_cache[cache_key]
            return certainty
        
        # Retrieve from store
        try:
            certainty, meta = self.host.store.get_turn_prosody(session_id, turn_id)
            
            # Cache the result
            self._prosody_cache[cache_key] = (certainty, meta)
            
            return certainty
            
        except Exception as e:
            logger.warning(f"[Retrieval] Failed to retrieve prosody for session={session_id}, turn={turn_id}: {e}")
            return 0.0

    def _format_emoji_bullet(self, candidate: Candidate, components: Dict[str, float], total_score: float) -> str:
        """
        Format candidate with intuitive emoji metadata (A/B test variant A).

        Emojis represent confidence, recency, and importance without technical jargon.
        Research shows LLMs understand emojis semantically and won't quote them as technical data.

        Args:
            candidate: Candidate to format
            components: Scoring components from _composite_score
            total_score: Combined score for the candidate

        Returns:
            Formatted bullet with emoji metadata
        """
        # Extract metrics
        confidence = components.get("wconf", 0.0) / self.weights.get("wconf", 1.0)
        recency = components.get("wrec", 0.0) / self.weights.get("wrec", 1.0)
        prosody = components.get("wpro", 0.0)
        prosody_weight = float(os.getenv("MEMORY_WEIGHT_PROSODY", "0.15"))
        if prosody_weight > 0:
            prosody = prosody / prosody_weight

        # Map confidence to stars
        if confidence >= 0.7:
            conf_emoji = "⭐⭐⭐"  # High confidence
        elif confidence >= 0.4:
            conf_emoji = "⭐⭐"    # Medium confidence
        else:
            conf_emoji = "⭐"      # Low confidence

        # Map recency to time emoji (based on timestamp age)
        now_ms = int(time.time() * 1000)
        age_ms = max(0, now_ms - candidate.ts) if candidate.ts else float('inf')
        age_hours = age_ms / (1000 * 60 * 60)

        if age_hours < 1:
            rec_emoji = "🆕"  # Very recent (<1 hour)
        elif age_hours < 24:
            rec_emoji = "⏰"  # Recent (<1 day)
        else:
            rec_emoji = "📅"  # Older (>1 day)

        # Source/importance emoji
        if candidate.source == "graph":
            source_emoji = "📌"  # Established fact
        elif candidate.source == "convo":
            source_emoji = "💬"  # From conversation
        elif candidate.source == "semantic":
            source_emoji = "🔍"  # Semantic search result
        else:
            source_emoji = "📝"  # Summary

        # Build compact emoji prefix
        emoji_prefix = f"{conf_emoji}{rec_emoji}{source_emoji}"

        # Get cleaned text
        if candidate.source == "convo":
            cleaned = self._strip_directive_scaffolding(candidate.text)
            text = self._smart_truncate(cleaned, 100)
            text = self._normalize_perspective_to_second_person(text)
        else:
            text = self._smart_truncate(candidate.text, 100)

        return f"• {emoji_prefix} {text}"

    def _format_minimal_bullet(self, candidate: Candidate, components: Dict[str, float], total_score: float) -> str:
        """
        Format candidate with minimal symbol metadata (A/B test variant B).

        Uses ultra-simple 3-character symbols for confidence and short text for recency.
        Designed to be unambiguous and impossible to confuse with factual content.

        Args:
            candidate: Candidate to format
            components: Scoring components from _composite_score
            total_score: Combined score for the candidate

        Returns:
            Formatted bullet with minimal metadata
        """
        # Extract metrics
        confidence = components.get("wconf", 0.0) / self.weights.get("wconf", 1.0)

        # Map confidence to simple symbols
        if confidence >= 0.7:
            conf_symbol = "+++"  # High confidence
        elif confidence >= 0.4:
            conf_symbol = "++-"  # Medium confidence
        else:
            conf_symbol = "+--"  # Low confidence

        # Map recency to short text
        now_ms = int(time.time() * 1000)
        age_ms = max(0, now_ms - candidate.ts) if candidate.ts else float('inf')
        age_hours = age_ms / (1000 * 60 * 60)

        if age_hours < 1:
            rec_symbol = "now"  # Very recent
        elif age_hours < 24:
            rec_symbol = "day"  # Today
        else:
            rec_symbol = "old"  # Older

        # Get cleaned text
        if candidate.source == "convo":
            cleaned = self._strip_directive_scaffolding(candidate.text)
            text = self._smart_truncate(cleaned, 100)
            text = self._normalize_perspective_to_second_person(text)
        else:
            text = self._smart_truncate(candidate.text, 100)

        return f"• {conf_symbol} {rec_symbol}: {text}"

    def _format_header_bullet(self, candidate: Candidate, components: Dict[str, float], total_score: float, expand_threshold: float) -> str:
        """
        Format candidate as compact header with automatic expansion based on score.

        Args:
            candidate: Candidate to format
            components: Scoring components from _composite_score
            total_score: Combined score for the candidate
            expand_threshold: Score below which to expand to full text

        Returns:
            Formatted header (or expanded header + full text)
        """
        # Extract small scalars for bracket block
        confidence = components.get("wconf", 0.0) / self.weights.get("wconf", 1.0)  # Normalize back to [0,1]
        recency = components.get("wrec", 0.0) / self.weights.get("wrec", 1.0)  # Normalize back to [0,1]
        usage = components.get("wuse", 0.0) / self.weights.get("wuse", 1.0)  # Normalize back to [0,1]
        prosody = components.get("wpro", 0.0)
        prosody_weight = float(os.getenv("MEMORY_WEIGHT_PROSODY", "0.15"))
        if prosody_weight > 0:
            prosody = prosody / prosody_weight  # Normalize back to [0,1]
        
        # Format based on source type
        if candidate.source == "graph":
            # Graph: relation-typed header
            # Extract relation from candidate text by simple parsing
            text = candidate.text
            
            # Try to extract relation and entity for compact header
            # Example: "Alice lives in NYC" -> "lives_in: Alice [conf=0.95 rec=0.85]"
            if " is " in text and " named " in text:
                # "Alice is named Smith" -> "name: Alice [conf=0.95 rec=0.85]"
                parts = text.split(" is named ")
                if len(parts) == 2:
                    entity, value = parts[0].strip(), parts[1].strip()
                    header = f"name: {entity} [conf={confidence:.2f} rec={recency:.2f} use={usage:.0f}]"
                else:
                    header = f"fact: {self._smart_truncate(text, 100)} [conf={confidence:.2f} rec={recency:.2f}]"
            elif " has " in text:
                # "Alice has cat" -> "has: Alice [conf=0.95 rec=0.85]"
                parts = text.split(" has ")
                if len(parts) == 2:
                    entity, value = parts[0].strip(), parts[1].strip()
                    header = f"has: {entity} [conf={confidence:.2f} rec={recency:.2f} use={usage:.0f}]"
                else:
                    header = f"has: {self._smart_truncate(text, 100)} [conf={confidence:.2f} rec={recency:.2f}]"
            else:
                # Fallback: generic relation header (INCREASED from 40 → 100 chars)
                header = f"fact: {self._smart_truncate(text, 100)} [conf={confidence:.2f} rec={recency:.2f}]"

        elif candidate.source == "convo":
            # Convo: short gist with scalars (INCREASED from 60 → 150 chars)
            # Clean directive scaffolding and normalize perspective for user addressing
            cleaned = self._strip_directive_scaffolding(candidate.text)
            gist = self._smart_truncate(cleaned, 150)
            gist = self._normalize_perspective_to_second_person(gist)
            # Optionally include emotion/arousal if stored for this turn
            emo_suffix = ""
            try:
                turn_id = candidate.meta.get("turn_id") if isinstance(candidate.meta, dict) else None
                session_id = getattr(self.host, 'current_session_id', None)
                if session_id and isinstance(turn_id, int) and turn_id >= 0:
                    certainty, meta = self.host.store.get_turn_prosody(session_id, turn_id)
                    emotion = (meta or {}).get("emotion")
                    arousal = (meta or {}).get("arousal")
                    # Keep compact: include only if available
                    if emotion is not None:
                        emo_suffix += f" emo={emotion}"
                    if isinstance(arousal, (int, float)):
                        emo_suffix += f" ar={float(arousal):.2f}"
            except Exception:
                pass
            header = f"convo: {gist} [conf={confidence:.2f} pro={prosody:.2f} rec={recency:.2f}{emo_suffix}]"
        
        elif candidate.source == "semantic":
            # Semantic: similarity-based header (INCREASED from 50 → 120 chars)
            similarity_score = candidate.meta.get("similarity_score")
            semantic_score = candidate.meta.get("semantic_score")
            similarity = semantic_score if semantic_score is not None else (similarity_score if similarity_score is not None else candidate.score_hint)
            gist = self._smart_truncate(candidate.text, 120)
            header = f"semantic: {gist} [sim={similarity:.2f} conf={confidence:.2f}]"

        else:  # summary
            # Summary: gist with recency (INCREASED from 50 → 120 chars)
            gist = self._smart_truncate(candidate.text, 120)
            header = f"summary: {gist} [conf={confidence:.2f} rec={recency:.2f}]"
        
        # Auto-expand rule: expand only for low-score AND low-confidence
        # Avoid mixing normalized gist with contradictory raw text unless necessary
        if total_score < expand_threshold and confidence < 0.30:
            # Add cleaned, normalized full text after header
            full_text = candidate.text.replace("\n", " ").strip()
            full_text = self._strip_directive_scaffolding(full_text)
            full_text = self._normalize_perspective_to_second_person(full_text)
            return f"• {header} -> {full_text}"
        else:
            # Compact header only
            return f"• {header}"

    def _get_source_priority(self, query: str, intent: Optional[Dict] = None) -> List[str]:
        """
        Determine source priority order based on query characteristics and intent.

        Returns list of sources in priority order: ['graph', 'convo', 'summary']
        """
        # Intent-based routing (highest priority)
        if intent and not intent.get('fallback', False):
            intent_type = intent.get('intent', '')

            # Memory lookup intents should prioritize conversation history
            if intent_type in ['lookup_memory', 'recall_information', 'retrieve_information']:
                return ['convo', 'summary', 'graph']

            # Update/delete intents should prioritize graph for accuracy
            if intent_type in ['memory_update', 'delete_memory', 'store_information']:
                return ['graph', 'convo', 'summary']

        # Query pattern-based routing (medium priority)
        if self._is_temporal_query(query):
            # Temporal queries benefit most from conversation history
            return ['convo', 'semantic', 'summary', 'graph']

        if self._is_semantic_query(query):
            # Semantic queries benefit from semantic, summaries and conversation
            return ['semantic', 'summary', 'convo', 'graph']

        # Default: factual queries work best with graph-first
        return ['graph', 'convo', 'semantic', 'summary']

    def _normalize_perspective_to_second_person(self, text: str) -> str:
        """
        Convert common first-person phrases to second-person for user addressing.

        This is a conservative, general-purpose normalization applied to retrieved
        conversation gists so responses use "you/your" rather than "I/my".
        """
        import re
        if not text:
            return text

        # Apply targeted replacements in order to avoid over-substitution
        rules = [
            (r"\bI'm\b", "You're"),
            (r"\bI am\b", "you are"),
            (r"\bI have\b", "you have"),
            (r"\bI've\b", "You've"),
            (r"\bI like\b", "you like"),
            (r"\bI love\b", "you love"),
            (r"\bI prefer\b", "you prefer"),
            (r"\bmy\b", "your"),
            (r"\bmine\b", "yours"),
            (r"\bI\b", "you"),
        ]

        out = text
        for pattern, repl in rules:
            out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

        # Clean double spaces if any
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def _strip_directive_scaffolding(self, text: str) -> str:
        """Remove common directive scaffolding from conversation text.

        Examples removed: "(so )?please remember", "remember that", "note that",
        "let me remind you", "I want you to remember", etc.
        """
        import re
        if not isinstance(text, str):
            return text
        t = text.strip()
        tl = t.lower()
        patterns = [
            r"^\s*(so\s+)?please\s+remember\s*",
            r"^\s*remember\s+that\s*",
            r"^\s*remember\s*",
            r"^\s*note\s+that\s*",
            r"^\s*let\s+me\s+remind\s+you\s*",
            r"^\s*i\s+want\s+you\s+to\s+remember\s*",
            r"^\s*kindly\s+remember\s*",
        ]
        for p in patterns:
            m = re.match(p, tl)
            if m:
                idx = m.end()
                t = t[idx:].lstrip()
                tl = t.lower()
                break
        t = re.sub(r"\s+", " ", t)
        return t.strip()
