"""
MemoryRetriever: Dedicated Context Retrieval Service
==================================================

Extracted from HotMemory monolith - now focused solely on:
- MMR (Maximal Marginal Relevance) algorithm for diverse memory selection
- Entity expansion and relationship discovery
- LEANN semantic search integration
- FTS (Full-Text Search) retrieval fusion
- Context bullet formatting
"""

import os
import time
import math
import re
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

from loguru import logger


@dataclass
class RetrievalResult:
    """Result of memory retrieval operation"""
    bullets: List[str]
    relevant_triples: List[Tuple[str, str, str]]
    query_entities: List[str]
    expanded_entities: List[str]
    retrieval_stats: Dict[str, Any]


class MemoryRetriever:
    """
    Dedicated retrieval service for memory context and relevant facts.
    Handles MMR algorithm, entity expansion, and semantic search.
    """
    
    def __init__(self, store, entity_index: Dict[str, Set], config: Dict[str, Any]):
        """Initialize retriever with storage and configuration"""
        self.store = store
        self.entity_index = entity_index
        
        # Configuration
        self.use_leann = config.get('use_leann', True)
        self.leann_index_path = config.get('leann_index_path')
        self.leann_complexity = config.get('leann_complexity', 16)
        self.retrieval_fusion = config.get('retrieval_fusion', True)
        self.use_leann_summaries = config.get('use_leann_summaries', True)
        
        # LEANN search (lazy loaded)
        self._leann_searcher = None
        self._leann_loaded_mtime = 0.0
        
        # Performance tracking
        self.metrics = defaultdict(list)
        
        # Bullet formatter (lazy import to avoid circular imports)
        self.bullet_formatter = None
        
        # Edge metadata (for scoring)
        self.edge_meta = {}
        
        # Verbose retrieval debugging (prints full candidates/selection/bullets)
        self.debug = os.getenv('HOTMEM_RETRIEVAL_DEBUG', 'false').lower() in ('1', 'true', 'yes')
        # Feature gates and thresholds (env configurable)
        self.graph_enabled = os.getenv('HOTMEM_GRAPH_ENABLED', 'true').lower() in ('1', 'true', 'yes')
        self.fts_only_summary = os.getenv('HOTMEM_FTS_ONLY_SUMMARY', 'true').lower() in ('1', 'true', 'yes')
        try:
            self.fts_min_overlap = float(os.getenv('HOTMEM_FTS_MIN_OVERLAP', '0.1'))
        except Exception:
            self.fts_min_overlap = 0.05
        # Pin a top KG candidate that matches query intent (e.g., lives_in for where/live queries)
        self.pin_intent_match = os.getenv('HOTMEM_PIN_INTENT_MATCH', 'true').lower() in ('1', 'true', 'yes')

        logger.info(f"[MemoryRetriever] Initialized with LEANN={'✓' if self.use_leann else '✗'}, fusion={'✓' if self.retrieval_fusion else '✗'}")
    
    def retrieve_context(self, query: str, entities: List[str], turn_id: int, intent=None) -> RetrievalResult:
        """
        Main retrieval entry point - retrieves relevant memory context
        """
        start = time.perf_counter()
        
        try:
            # Expand entities with aliases and relationships
            expanded_entities = self._expand_query_entities(entities, query)
            logger.debug(f"[MemoryRetriever] Retrieval context: query='{query[:50]}...', entities={entities}, total_edges={sum(len(triples) for triples in self.entity_index.values())}")
            
            # Get candidate triples through multiple strategies
            candidate_triples = self._gather_candidate_triples(query, expanded_entities, intent)
            if self.debug:
                self._debug_log_candidates(candidate_triples)
            
            # Apply MMR algorithm for diverse selection
            bullets = self._apply_mmr_selection(query, candidate_triples, turn_id)
            if self.debug:
                logger.info(f"[Retrieval DEBUG] bullets_selected={len(bullets)}")
                for i, b in enumerate(bullets):
                    logger.info(f"[Retrieval DEBUG] bullet[{i}]: {b}")
            
            # Track performance
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics['retrieval_ms'].append(elapsed_ms)
            
            stats = {
                'elapsed_ms': elapsed_ms,
                'candidates': len(candidate_triples),
                'selected': len(bullets),
                'expanded_entities': len(expanded_entities)
            }
            
            return RetrievalResult(
                bullets=bullets,
                relevant_triples=candidate_triples,
                query_entities=entities,
                expanded_entities=expanded_entities,
                retrieval_stats=stats
            )
            
        except Exception as e:
            logger.error(f"[MemoryRetriever] Retrieval failed: {e}")
            return RetrievalResult([], [], entities, entities, {'error': str(e)})

    def _debug_log_candidates(self, candidates: List[Tuple[float, int, str, Any]]):
        """Verbose dump of retrieval candidates before selection."""
        try:
            logger.info(f"[Retrieval DEBUG] candidates_total={len(candidates)}")
            for i, (sc, ts, kind, payload) in enumerate(candidates):
                if kind == 'kg' and isinstance(payload, (tuple, list)) and len(payload) >= 3:
                    s, r, d = payload[:3]
                    logger.info(f"[Retrieval DEBUG] cand[{i}] kind=kg score={sc:.3f} ts={ts} triple=({s}, {r}, {d})")
                else:
                    logger.info(f"[Retrieval DEBUG] cand[{i}] kind={kind} score={sc:.3f} ts={ts} text={str(payload)}")
        except Exception:
            pass
    
    def _expand_query_entities(self, entities: List[str], query: str) -> List[str]:
        """Expand query entities with aliases and relationships"""
        expanded = set(entities)
        
        # Add "you" if query contains first-person pronouns
        t = (query or "").lower()
        if any(p in t for p in [" i ", " my ", " me ", "i'm", "i've"]):
            expanded.add("you")
        
        # Expand aliases using also_known_as relationships
        for ent in list(entities):
            if ent in self.entity_index:
                for item in self.entity_index[ent]:
                    # Ensure item is a proper tuple before unpacking (type safety!)
                    if isinstance(item, (tuple, list)) and len(item) >= 3:
                        s2, r2, d2 = item[:3]  # Take first 3 elements to be safe
                        if r2 == 'also_known_as' and d2 == ent:
                            expanded.add(s2)
                    else:
                        # Skip malformed entries during alias expansion
                        logger.warning(f"[MemoryRetriever] Malformed entry in entity_index for '{ent}': {item}")
                        continue
        
        # Multi-hop entity expansion for richer context
        expanded = self._multi_hop_expansion(expanded, query)
        
        return list(expanded)[:12]  # Limit to prevent explosion
    
    def _multi_hop_expansion(self, base_entities: Set[str], query: str) -> Set[str]:
        """Perform multi-hop graph expansion to find related entities"""
        expanded = set(base_entities)
        
        # Add fuzzy matches for entity names
        for entity in base_entities:
            fuzzy_matches = self._find_fuzzy_entity_matches(entity, expanded)
            expanded.update(fuzzy_matches)
        
        # 1-hop and 2-hop graph traversal
        for entity in base_entities:
            if entity in self.entity_index:
                # 1-hop: Direct connections
                direct_connections = set()
                for item in self.entity_index[entity]:
                    # Type safety check
                    if isinstance(item, (tuple, list)) and len(item) >= 3:
                        s, r, d = item[:3]  # Take first 3 elements to be safe
                        if d == entity and s not in expanded:
                            direct_connections.add(s)
                        if s == entity and d not in expanded:
                            direct_connections.add(d)
                    else:
                        logger.warning(f"[MemoryRetriever] Malformed entry for '{entity}': {item}")
                        continue
                
                # Add direct connections
                expanded.update(direct_connections)
                
                # 2-hop: Friends-of-friends (limit to prevent explosion)
                if len(expanded) < 12:
                    for connected_entity in direct_connections:
                        if connected_entity in self.entity_index:
                            for item in self.entity_index[connected_entity]:
                                # Type safety check
                                if isinstance(item, (tuple, list)) and len(item) >= 3:
                                    s2, r2, d2 = item[:3]  # Take first 3 elements to be safe
                                    # Prioritize high-value 2-hop connections
                                    if r2 in {"lives_in", "works_at", "teach_at", "married_to", "has", "also_known_as"}:
                                        if d2 == connected_entity and s2 not in expanded:
                                            expanded.add(s2)
                                        if s2 == connected_entity and d2 not in expanded:
                                            expanded.add(d2)
                                else:
                                    logger.warning(f"[MemoryRetriever] Malformed entry for '{connected_entity}': {item}")
                                    continue
        
        return expanded
    
    def _gather_candidate_triples(self, query: str, entities: List[str], intent=None) -> List[Tuple[float, int, str, Any]]:
        """Gather candidate triples using multiple strategies"""
        candidates = []
        now_ms = int(time.time() * 1000)
        recency_T_ms = 7 * 24 * 60 * 60 * 1000  # 7 days
        
        # Strategy 1: Entity-based retrieval (gated)
        if self.graph_enabled:
            for entity in entities:
                if entity in self.entity_index:
                    candidates.extend(self._score_entity_triples(entity, query, now_ms, recency_T_ms))
        
        # Strategy 2: LEANN semantic enhancement (if enabled)
        if self.use_leann and self.retrieval_fusion:
            leann_enhanced = self._retrieve_with_leann_enhancement(query, entities)
            candidates.extend(leann_enhanced)
        
        # Strategy 3: FTS summary search (if fusion enabled)
        if self.retrieval_fusion and query:
            fts_results = self._search_fts_summaries(query)
            candidates.extend(fts_results)
        
        return candidates
    
    def _score_entity_triples(self, entity: str, query: str, now_ms: int, recency_T_ms: int) -> List[Tuple[float, int, str, Any]]:
        """Score and return triples for a given entity"""
        candidates = []
        
        if entity not in self.entity_index:
            return candidates
            
        # Simple lexical similarity for query tokens
        qtok = set(self._tokenize_query(query))
        # Query-intent hints
        is_where_q = ('where' in qtok) or ('live' in qtok) or ('address' in qtok)
        is_work_q = ('work' in qtok) or ('company' in qtok) or ('job' in qtok) or ('employ' in qtok)
        is_name_q = ('name' in qtok) or ('call' in qtok)

        for item in self.entity_index[entity]:
            # Type safety check
            if not isinstance(item, (tuple, list)) or len(item) < 3:
                logger.warning(f"[MemoryRetriever] Malformed triple for '{entity}': {item}")
                continue
                
            s, r, d = item[:3]  # Take first 3 elements to be safe
            
            # Skip pronoun-like subjects
            pronoun_skip = {"he", "she", "it", "they", "we", "who", "what", "when", "where", "how", "why", "that"}
            if s in pronoun_skip:
                continue
            
            # Down-weight low-value relations
            low_value_rel = {"say", "tell", "feel", "do", "is", "and"}
            if r in low_value_rel:
                base_penalty = 0.1
            else:
                base_penalty = 0.0

            # Get metadata
            meta = self.edge_meta.get((s, r, d), {})
            ts = int(meta.get('ts', 0))
            age = max(0, now_ms - ts)
            rec = math.exp(-age / max(1, recency_T_ms)) if ts > 0 else 0.0
            
            # Semantic similarity (simple lexical for now)
            sem = 0.0
            stok = self._tokenize_query(f"{s} {r} {d}")
            if qtok and stok:
                inter = len(qtok & stok)
                union = len(qtok | stok)
                sem = inter / union if union else 0.0
            
            # Weight from extraction confidence
            w = float(meta.get('weight', 0.3))
            
            # Relation/query intent boost
            relation_boost = 0.0
            if is_where_q and r in {"lives_in", "live_in", "resides_in", "address", "born_in", "moved_to", "moved_from"}:
                relation_boost += 0.4
            if is_work_q and r in {"works_at", "employed_by", "work_at"}:
                relation_boost += 0.3
            if is_name_q and r in {"name", "also_known_as", "called"}:
                relation_boost += 0.3
            if s == 'you':
                relation_boost += 0.2

            # Composite scoring (clamped)
            score = max(0.0, 0.4 * rec + 0.4 * sem + 0.2 * w + relation_boost - base_penalty)
            candidates.append((score, ts, 'kg', (s, r, d)))
        
        return candidates
    
    def _apply_mmr_selection(self, query: str, candidates: List[Tuple[float, int, str, Any]], turn_id: int) -> List[str]:
        """Apply MMR algorithm for diverse selection"""
        if not candidates:
            return []
        
        # Sort by score
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Calculate threshold (75th percentile)
        scores_only = [s for (s, _ts, _k, _p) in candidates]
        idx = max(0, int(len(scores_only) * 0.75) - 1)
        tau = scores_only[idx] if scores_only else 0.0
        eps = 0.05
        
        # Filter by threshold
        pool = [(sc, ts, k, p) for (sc, ts, k, p) in candidates if sc >= max(0.0, tau - eps)]
        
        # Optional: pin a best-matching KG candidate (e.g., lives_in for where/live queries)
        selected: List[Tuple[float, int, str, Any]] = []
        seen_triples: Set[Tuple[str, str, str]] = set()
        if self.pin_intent_match and pool:
            qtok = self._tokenize_query(query)
            is_where_q = ('where' in qtok) or ('live' in qtok) or ('address' in qtok)
            is_work_q = ('work' in qtok) or ('company' in qtok) or ('job' in qtok) or ('employ' in qtok)
            is_name_q = ('name' in qtok) or ('call' in qtok)
            intent_rels = set()
            if is_where_q:
                intent_rels.update({'lives_in', 'live_in', 'resides_in', 'address', 'born_in', 'moved_to', 'moved_from'})
            if is_work_q:
                intent_rels.update({'works_at', 'employed_by', 'work_at'})
            if is_name_q:
                intent_rels.update({'name', 'also_known_as', 'called'})
            if intent_rels:
                pinned = []
                for i, (sc, ts, k, p) in enumerate(pool):
                    if k == 'kg' and isinstance(p, (tuple, list)) and len(p) >= 3:
                        s, r, d = p[:3]
                        if r in intent_rels and s == 'you':
                            pinned.append((i, sc, ts, k, p))
                if pinned:
                    # Pick the highest score among pinned
                    pinned.sort(key=lambda x: x[1], reverse=True)
                    i, sc, ts, k, p = pinned[0]
                    selected.append((sc, ts, k, p))
                    s, r, d = p[:3]
                    seen_triples.add((s, r, d))
                    pool.pop(i)
                    if self.debug:
                        logger.info(f"[Retrieval DEBUG] pinned_top kind=kg triple=({s}, {r}, {d}) score={sc:.3f}")
        if self.debug:
            logger.info(f"[Retrieval DEBUG] MMR pool size={len(pool)} (tau={tau:.3f}, eps={eps})")
            for i, (sc, ts, k, p) in enumerate(pool):
                if k == 'kg' and isinstance(p, (tuple, list)) and len(p) >= 3:
                    s, r, d = p[:3]
                    logger.info(f"[Retrieval DEBUG] pool[{i}] kind=kg score={sc:.3f} ts={ts} triple=({s}, {r}, {d})")
                else:
                    logger.info(f"[Retrieval DEBUG] pool[{i}] kind={k} score={sc:.3f} ts={ts} text={str(p)}")
        
        # MMR selection
        lambda_rel = 0.2  # Balance relevance vs diversity
        K_max = min(15, len(pool))  # Limit results
        
        while pool and len(selected) < K_max:
            best_idx = -1
            best_mmr = -1.0
            
            for i, (sc, ts, k, p) in enumerate(pool):
                if k == 'kg':
                    # Type safety check for MMR
                    if isinstance(p, (tuple, list)) and len(p) >= 3:
                        s, r, d = p[:3]  # Take first 3 elements to be safe
                        if (s, r, d) in seen_triples:
                            continue
                    else:
                        # Skip malformed entries during MMR scoring
                        logger.warning(f"[MemoryRetriever] Malformed entry in MMR pool: {p}")
                        continue
                
                # Calculate maximum similarity to selected items
                max_sim = 0.0
                for (_sc2, _ts2, k2, p2) in selected:
                    max_sim = max(max_sim, self._calculate_similarity((k, p), (k2, p2)))
                
                # MMR score
                mmr = lambda_rel * sc - (1 - lambda_rel) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(pool[best_idx])
                # Add to seen triples if valid KG entry
                _sc, _ts, k, p = pool[best_idx]
                if k == 'kg' and isinstance(p, (tuple, list)) and len(p) >= 3:
                    s, r, d = p[:3]
                    seen_triples.add((s, r, d))
                pool.pop(best_idx)
            else:
                # No selectable item found (all malformed or duplicates) → avoid infinite loop
                logger.warning("[MemoryRetriever] No selectable candidate in MMR pool; breaking to avoid hang")
                break
        
        # Format selected items as bullets
        bullets = []
        if self.debug:
            logger.info(f"[Retrieval DEBUG] selected_items={len(selected)}")
            for i, (_sc, _ts, k, p) in enumerate(selected):
                if k == 'kg' and isinstance(p, (tuple, list)) and len(p) >= 3:
                    s, r, d = p[:3]
                    logger.info(f"[Retrieval DEBUG] selected[{i}] kind=kg triple=({s}, {r}, {d})")
                else:
                    logger.info(f"[Retrieval DEBUG] selected[{i}] kind={k} text={str(p)}")
        for _sc, _ts, k, p in selected:
            if k == 'kg':
                # Type safety check for bullet formatting
                if isinstance(p, (tuple, list)) and len(p) >= 3:
                    s, r, d = p[:3]  # Take first 3 elements to be safe
                    bullets.append(self._format_memory_bullet(s, r, d))
                else:
                    logger.warning(f"[MemoryRetriever] Malformed entry for bullet formatting: {p}")
            else:
                # Handle non-KG entries (summaries, etc.)
                if isinstance(p, str):
                    bullets.append(f"• {p}")
                else:
                    logger.warning(f"[MemoryRetriever] Non-KG entry with unexpected type: {type(p)}")
        
        return bullets[:5]  # Final limit
    
    def _calculate_similarity(self, item1: Tuple[str, Any], item2: Tuple[str, Any]) -> float:
        """Calculate similarity between two retrieval items"""
        kind1, pay1 = item1
        kind2, pay2 = item2
        
        # KG vs KG: prefer different subject/relation
        if kind1 == 'kg' and kind2 == 'kg':
            if not (isinstance(pay1, (tuple, list)) and len(pay1) >= 3 and 
                   isinstance(pay2, (tuple, list)) and len(pay2) >= 3):
                return 0.0
                
            s1, r1, d1 = pay1[:3]
            s2, r2, d2 = pay2[:3]
            sim = 0.0
            if s1 == s2:
                sim += 0.6
            if r1 == r2:
                sim += 0.3
            if d1 == d2:
                sim += 0.1
            return sim
        
        # Summary vs Summary: token overlap
        if kind1 != 'kg' and kind2 != 'kg':
            t1 = self._tokenize_query(str(pay1))
            t2 = self._tokenize_query(str(pay2))
            if not t1 or not t2:
                return 0.0
            inter = len(t1 & t2)
            union = len(t1 | t2)
            return 0.6 * (inter / union)
        
        # Cross-type: light similarity
        return 0.1
    
    def _tokenize_query(self, text: str) -> Set[str]:
        """Tokenize and normalize query text (strip punctuation, min length)."""
        if not text:
            return set()
        tokens: Set[str] = set()
        for w in str(text).lower().split():
            # Remove punctuation to improve FTS prefix matches (e.g., 'live?' -> 'live')
            w2 = re.sub(r"[^\w]", "", w)
            if len(w2) >= 3:
                tokens.add(w2)
        return tokens
    
    def _find_fuzzy_entity_matches(self, query_entity: str, all_entities: set) -> set:
        """Find fuzzy matches for entity names"""
        matches = set()
        query_lower = query_entity.lower()
        
        for entity in all_entities:
            entity_lower = entity.lower()
            if (query_lower in entity_lower or entity_lower in query_lower) and query_entity != entity:
                matches.add(entity)
        
        return matches
    
    def _search_fts_summaries(self, query: str, limit: int = 12) -> List[Tuple[float, int, str, str]]:
        """Search FTS summaries with robust fallback tokenization.

        Returns a list of (score, ts, kind, payload) where kind='fts' and payload is text.
        """
        if not hasattr(self.store, 'search_fts_detailed'):
            return []

        # Build candidate queries: prefer tokenized OR prefix form first for recall
        queries: List[str] = []
        toks = list(self._tokenize_query(query))
        if toks:
            queries.append(" OR ".join(f"{t}*" for t in toks))
        # Raw query as a secondary attempt
        if query:
            queries.append(query)

        fts_results: List[Tuple[str, str, int]] = []
        for q in queries:
            try:
                fts_results = self.store.search_fts_detailed(q, limit=limit)
                if fts_results:
                    if self.debug:
                        logger.info(f"[Retrieval DEBUG] FTS MATCH '{q}' hits={len(fts_results)}")
                    break
            except Exception:
                # Try next variant
                continue

        results: List[Tuple[float, int, str, str]] = []
        qtok = self._tokenize_query(query)
        for (text_fts, eid_fts, ts_fts) in fts_results:
            if not text_fts:
                continue
            # Prefer summaries
            is_summary = isinstance(eid_fts, str) and (eid_fts.startswith('summary:') or eid_fts.startswith('session:'))
            pri = 0.50 if is_summary else 0.40

            rec = 0.0
            if ts_fts and ts_fts > 0:
                age = max(0, int(time.time() * 1000) - int(ts_fts))
                rec = math.exp(-age / (7 * 24 * 60 * 60 * 1000))  # 7 days

            # Simple token overlap with query to down-rank irrelevant text
            ttok = self._tokenize_query(text_fts)
            sem = 0.0
            if qtok and ttok:
                inter = len(qtok & ttok)
                union = len(qtok | ttok)
                sem = inter / union if union else 0.0
            w = 0.3

            sc = 0.3 * pri + 0.4 * rec + 0.3 * sem + 0.1 * w
            results.append((sc, int(ts_fts or 0), 'fts', text_fts))

        if self.debug:
            logger.info(f"[Retrieval DEBUG] FTS candidates_built={len(results)}")
            for i, (sc, ts, k, txt) in enumerate(results):
                logger.info(f"[Retrieval DEBUG] fts[{i}] score={sc:.3f} ts={ts} text={txt}")

        return results
    
    def _retrieve_with_leann_enhancement(self, query: str, entity_triples: List[Tuple[str, str, str]], top_k: int = 32) -> List[Tuple[float, int, str, Any]]:
        """LEANN semantic search enhancement - TEMPORARILY DISABLED"""
        # Disabled to prevent hanging - method calls non-existent _leann_query_scores
        return []
    
    def _format_memory_bullet(self, s: str, r: str, d: str) -> str:
        """Format a triple as a memory bullet"""
        # Lazy import bullet formatter
        if self.bullet_formatter is None:
            try:
                from services.enhanced_bullet_formatter import EnhancedBulletFormatter
                self.bullet_formatter = EnhancedBulletFormatter()
            except Exception:
                # Fallback to simple formatting
                def simple_formatter(s, r, d):
                    return f"• {s} {r} {d}"
                self.bullet_formatter = type('SimpleFormatter', (), {'format_bullet': simple_formatter})()
        
        try:
            return self.bullet_formatter.format_bullet(s, r, d)
        except Exception:
            return f"• {s} {r} {d}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retrieval performance metrics"""
        return dict(self.metrics)


logger.info("🎯 MemoryRetriever initialized - dedicated retrieval service")
logger.info("📊 Features: MMR algorithm, entity expansion, LEANN integration, FTS fusion")
