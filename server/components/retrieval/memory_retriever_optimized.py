"""
MemoryRetriever: Optimized version with performance improvements
================================================================

Quick wins implemented:
- Query tokenization caching
- Pre-filtering by relevant relations
- Early termination in scoring
- Batched string operations
- Reduced fuzzy matching scope
"""

import os
import time
import math
import re
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of memory retrieval operation"""
    bullets: List[str]
    relevant_triples: List[Tuple[str, str, str]]
    query_entities: List[str]
    expanded_entities: List[str]
    retrieval_stats: Dict[str, Any]


class MemoryRetrieverOptimized:
    """
    Optimized retrieval service with performance improvements.
    """

    # High-value relations for different query intents
    WHERE_RELATIONS = {"lives_in", "live_in", "resides_in", "address", "born_in", "moved_to", "moved_from"}
    WORK_RELATIONS = {"works_at", "employed_by", "work_at", "job", "company"}
    NAME_RELATIONS = {"name", "also_known_as", "called", "named"}
    HIGH_VALUE_RELATIONS = WHERE_RELATIONS | WORK_RELATIONS | NAME_RELATIONS | {"has", "married_to", "teach_at"}
    LOW_VALUE_RELATIONS = {"say", "tell", "feel", "do", "is", "and"}

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

        # Edge metadata (for scoring)
        self.edge_meta = {}

        # OPTIMIZATION: Tokenization cache
        self._token_cache = {}
        self._token_cache_size = 1000  # Max cache size

        # OPTIMIZATION: Pre-index relations by type for faster filtering
        self._relation_index = self._build_relation_index()

        # Verbose retrieval debugging
        self.debug = os.getenv('HOTMEM_RETRIEVAL_DEBUG', 'false').lower() in ('1', 'true', 'yes')
        # Feature gates and thresholds
        self.graph_enabled = os.getenv('HOTMEM_GRAPH_ENABLED', 'true').lower() in ('1', 'true', 'yes')
        self.fts_only_summary = os.getenv('HOTMEM_FTS_ONLY_SUMMARY', 'true').lower() in ('1', 'true', 'yes')
        try:
            self.fts_min_overlap = float(os.getenv('HOTMEM_FTS_MIN_OVERLAP', '0.1'))
        except Exception:
            self.fts_min_overlap = 0.05
        self.pin_intent_match = os.getenv('HOTMEM_PIN_INTENT_MATCH', 'true').lower() in ('1', 'true', 'yes')
        try:
            self.verb_prep_boost = float(os.getenv('HOTMEM_RETRIEVAL_VERB_PREP_BOOST', '0.12'))
        except Exception:
            self.verb_prep_boost = 0.12

        # OPTIMIZATION: Early termination thresholds
        self.max_candidates_per_entity = 50  # Stop after finding this many good candidates per entity
        self.min_score_threshold = 0.2  # Ignore candidates below this score
        self.max_expansion_entities = 8  # Limit entity expansion

        logger.info(f"[MemoryRetrieverOptimized] Initialized with performance optimizations")

    def _build_relation_index(self) -> Dict[str, Set[Tuple]]:
        """Build index of triples by relation type for fast filtering"""
        rel_index = defaultdict(set)
        for entity, triples in self.entity_index.items():
            for item in triples:
                if isinstance(item, (tuple, list)) and len(item) >= 3:
                    s, r, d = item[:3]
                    rel_index[r].add((s, r, d))
        return dict(rel_index)

    def _tokenize_query(self, text: str) -> Set[str]:
        """Tokenize query with caching"""
        # OPTIMIZATION: Cache tokenization results
        if text in self._token_cache:
            return self._token_cache[text]

        # Clean cache if too large
        if len(self._token_cache) > self._token_cache_size:
            self._token_cache.clear()

        # Tokenize
        tokens = set(re.findall(r'\b\w+\b', text.lower()))
        self._token_cache[text] = tokens
        return tokens

    def retrieve_context(self, query: str, entities: List[str], turn_id: int, intent=None) -> RetrievalResult:
        """Main retrieval entry point with optimizations"""
        start = time.perf_counter()
        timings = {}

        try:
            # Expand entities with aliases and relationships
            expand_start = time.perf_counter()
            expanded_entities = self._expand_query_entities_optimized(entities, query)
            timings['expand_ms'] = (time.perf_counter() - expand_start) * 1000

            if self.debug:
                logger.debug(f"[Optimized] Expansion: {len(entities)} → {len(expanded_entities)} in {timings['expand_ms']:.1f}ms")

            # Get candidate triples through multiple strategies
            gather_start = time.perf_counter()
            candidate_triples = self._gather_candidate_triples_optimized(query, expanded_entities, intent)
            timings['gather_ms'] = (time.perf_counter() - gather_start) * 1000

            if self.debug:
                logger.debug(f"[Optimized] Gathering: {len(candidate_triples)} candidates in {timings['gather_ms']:.1f}ms")

            # Apply MMR selection
            mmr_start = time.perf_counter()
            bullets = self._apply_mmr_selection_optimized(query, candidate_triples, turn_id)
            timings['mmr_ms'] = (time.perf_counter() - mmr_start) * 1000

            # Format bullets - simple formatting for benchmark
            formatted_bullets = []
            for bullet in bullets:
                if isinstance(bullet, str):
                    formatted_bullets.append(bullet)
                elif isinstance(bullet, tuple) and len(bullet) >= 3:
                    s, r, d = bullet[:3]
                    formatted_bullets.append(f"{s} {r} {d}")
                else:
                    formatted_bullets.append(str(bullet))

            timings['total_ms'] = (time.perf_counter() - start) * 1000

            if self.debug or timings['total_ms'] > 500:
                logger.info(f"[Optimized] Total retrieval: {timings['total_ms']:.1f}ms "
                          f"(expand={timings['expand_ms']:.1f}, gather={timings['gather_ms']:.1f}, mmr={timings['mmr_ms']:.1f})")

            # Track metrics
            self.metrics['retrieval_time_ms'].append(timings['total_ms'])
            self.metrics['bullets_count'].append(len(formatted_bullets))

            return RetrievalResult(
                bullets=formatted_bullets,
                relevant_triples=[],  # Not tracked for performance
                query_entities=entities,
                expanded_entities=list(expanded_entities),
                retrieval_stats=timings
            )

        except Exception as e:
            logger.error(f"[Optimized] Retrieval error: {e}")
            return RetrievalResult(
                bullets=[],
                relevant_triples=[],
                query_entities=entities,
                expanded_entities=entities,
                retrieval_stats={'error': str(e)}
            )

    def _expand_query_entities_optimized(self, entities: List[str], query: str) -> List[str]:
        """Optimized entity expansion with early termination"""
        expanded = set(entities)

        # Add "you" if query contains first-person pronouns
        t = (query or "").lower()
        if any(p in t for p in [" i ", " my ", " me ", "i'm", "i've"]):
            expanded.add("you")

        # OPTIMIZATION: Limit expansion scope
        if len(expanded) >= self.max_expansion_entities:
            return list(expanded)[:self.max_expansion_entities]

        # Expand aliases using also_known_as relationships (direct lookup only)
        for ent in list(entities)[:5]:  # Limit to first 5 entities
            if ent in self.entity_index:
                alias_count = 0
                for item in self.entity_index[ent]:
                    if alias_count >= 3:  # Max 3 aliases per entity
                        break
                    if isinstance(item, (tuple, list)) and len(item) >= 3:
                        s2, r2, d2 = item[:3]
                        if r2 == 'also_known_as' and d2 == ent:
                            expanded.add(s2)
                            alias_count += 1

        # OPTIMIZATION: Limited multi-hop expansion
        if len(expanded) < self.max_expansion_entities:
            expanded = self._multi_hop_expansion_optimized(expanded, query)

        return list(expanded)[:self.max_expansion_entities]

    def _multi_hop_expansion_optimized(self, base_entities: Set[str], query: str) -> Set[str]:
        """Optimized multi-hop expansion with early termination"""
        expanded = set(base_entities)

        # OPTIMIZATION: Skip fuzzy matching - too expensive
        # Only do 1-hop expansion for high-value relations
        for entity in list(base_entities)[:3]:  # Limit to first 3 base entities
            if len(expanded) >= self.max_expansion_entities:
                break

            if entity in self.entity_index:
                connections_added = 0
                for item in self.entity_index[entity]:
                    if connections_added >= 5:  # Max 5 connections per entity
                        break
                    if isinstance(item, (tuple, list)) and len(item) >= 3:
                        s, r, d = item[:3]
                        # Only expand through high-value relations
                        if r in self.HIGH_VALUE_RELATIONS:
                            if d == entity and s not in expanded:
                                expanded.add(s)
                                connections_added += 1
                            elif s == entity and d not in expanded:
                                expanded.add(d)
                                connections_added += 1

        return expanded

    def _gather_candidate_triples_optimized(self, query: str, entities: List[str], intent=None) -> List[Tuple[float, int, str, Any]]:
        """Optimized candidate gathering with pre-filtering"""
        candidates = []
        now_ms = int(time.time() * 1000)
        recency_T_ms = 7 * 24 * 60 * 60 * 1000  # 7 days

        # Parse query intent once
        qtok = self._tokenize_query(query)
        query_intent = self._parse_query_intent(qtok)

        # Strategy 1: Entity-based retrieval with pre-filtering
        if self.graph_enabled:
            entity_candidates = self._score_entities_optimized(
                entities, query, qtok, query_intent, now_ms, recency_T_ms
            )
            candidates.extend(entity_candidates)

            if self.debug and entity_candidates:
                logger.debug(f"[Optimized] Entity scoring: {len(entity_candidates)} candidates")

        # Strategy 2: LEANN (unchanged for now)
        if self.use_leann and self.retrieval_fusion:
            leann_enhanced = self._retrieve_with_leann_enhancement(query, entities)
            candidates.extend(leann_enhanced)

        # Strategy 3: FTS (unchanged for now)
        if self.retrieval_fusion and query:
            fts_results = self._search_fts_summaries(query)
            candidates.extend(fts_results)

        return candidates

    def _parse_query_intent(self, qtok: Set[str]) -> Dict[str, bool]:
        """Parse query tokens to determine intent"""
        return {
            'is_where': bool(qtok & {'where', 'live', 'address', 'location', 'reside'}),
            'is_work': bool(qtok & {'work', 'company', 'job', 'employ', 'office'}),
            'is_name': bool(qtok & {'name', 'call', 'named', 'who'}),
            'is_has': bool(qtok & {'has', 'have', 'own', 'possess'}),
        }

    def _score_entities_optimized(self, entities: List[str], query: str, qtok: Set[str],
                                 query_intent: Dict[str, bool], now_ms: int, recency_T_ms: int) -> List[Tuple[float, int, str, Any]]:
        """Optimized entity scoring with pre-filtering and early termination"""
        candidates = []

        # OPTIMIZATION: Pre-select relevant relations based on query intent
        relevant_relations = set()
        if query_intent['is_where']:
            relevant_relations.update(self.WHERE_RELATIONS)
        if query_intent['is_work']:
            relevant_relations.update(self.WORK_RELATIONS)
        if query_intent['is_name']:
            relevant_relations.update(self.NAME_RELATIONS)

        # If no specific intent, include all high-value relations
        if not relevant_relations:
            relevant_relations = self.HIGH_VALUE_RELATIONS

        for entity in entities[:5]:  # Process top 5 entities only
            if entity not in self.entity_index:
                continue

            entity_candidates = []

            for item in self.entity_index[entity]:
                # Early termination if we have enough candidates
                if len(entity_candidates) >= self.max_candidates_per_entity:
                    break

                if not isinstance(item, (tuple, list)) or len(item) < 3:
                    continue

                s, r, d = item[:3]

                # OPTIMIZATION: Skip if relation not relevant
                if relevant_relations and r not in relevant_relations and r not in self.LOW_VALUE_RELATIONS:
                    # Check if it's a verb_prep relation that might be relevant
                    if '_' not in r:
                        continue

                # Skip pronouns
                if s in {"he", "she", "it", "they", "we", "who", "what", "when", "where", "how", "why", "that"}:
                    continue

                # Fast scoring
                score = self._fast_score_triple(s, r, d, entity, qtok, query_intent, now_ms, recency_T_ms)

                # OPTIMIZATION: Skip low-score candidates
                if score < self.min_score_threshold:
                    continue

                # Get metadata
                meta = self.edge_meta.get((s, r, d), {})
                ts = int(meta.get('ts', 0))

                entity_candidates.append((score, ts, 'kg', (s, r, d)))

            # Sort and take top candidates from this entity
            entity_candidates.sort(key=lambda x: x[0], reverse=True)
            candidates.extend(entity_candidates[:self.max_candidates_per_entity])

        return candidates

    def _fast_score_triple(self, s: str, r: str, d: str, entity: str, qtok: Set[str],
                          query_intent: Dict[str, bool], now_ms: int, recency_T_ms: int) -> float:
        """Fast scoring without repeated tokenization"""
        # Base penalty for low-value relations
        if r in self.LOW_VALUE_RELATIONS:
            base_score = 0.1
        else:
            base_score = 0.3

        # Get metadata
        meta = self.edge_meta.get((s, r, d), {})
        ts = int(meta.get('ts', 0))
        age = max(0, now_ms - ts)

        # Recency score (simplified)
        rec = math.exp(-age / max(1, recency_T_ms)) if ts > 0 else 0.0

        # Semantic similarity (optimized with pre-tokenized query)
        triple_text = f"{s} {r} {d}".lower()
        stok = self._tokenize_query(triple_text)

        sem = 0.0
        if qtok and stok:
            # OPTIMIZATION: Faster set operations
            inter_size = len(qtok & stok)
            if inter_size > 0:
                union_size = len(qtok) + len(stok) - inter_size
                sem = inter_size / union_size

        # Weight from extraction confidence
        w = float(meta.get('weight', 0.3))

        # Intent-based boost (pre-computed)
        relation_boost = 0.0
        if query_intent['is_where'] and r in self.WHERE_RELATIONS:
            relation_boost = 0.4
        elif query_intent['is_work'] and r in self.WORK_RELATIONS:
            relation_boost = 0.3
        elif query_intent['is_name'] and r in self.NAME_RELATIONS:
            relation_boost = 0.3

        # Boost for "you" subject
        if s == 'you':
            relation_boost += 0.2

        # Verb+prep boost
        if '_' in r and relation_boost < 0.3:
            relation_boost += self.verb_prep_boost

        # Composite score
        score = base_score + (0.4 * rec) + (0.4 * sem) + (0.2 * w) + relation_boost
        return max(0.0, score)

    def _apply_mmr_selection_optimized(self, query: str, candidates: List[Tuple[float, int, str, Any]], turn_id: int) -> List[str]:
        """Optimized MMR with reduced similarity calculations"""
        if not candidates:
            return []

        # Sort by score
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # OPTIMIZATION: Take only top candidates for MMR
        max_pool_size = 100  # Limit pool size
        if len(candidates) > max_pool_size:
            # Take top 75% by score, but no more than max_pool_size
            cutoff_idx = min(int(len(candidates) * 0.75), max_pool_size)
            candidates = candidates[:cutoff_idx]

        # Calculate threshold (75th percentile of reduced pool)
        scores_only = [s for (s, _ts, _k, _p) in candidates]
        idx = max(0, int(len(scores_only) * 0.75) - 1)
        tau = scores_only[idx] if scores_only else 0.0
        eps = 0.05

        # Filter by threshold
        pool = [(sc, ts, k, p) for (sc, ts, k, p) in candidates if sc >= max(0.0, tau - eps)]

        # Pin top intent match (unchanged)
        selected = []
        seen_triples = set()
        if self.pin_intent_match and pool:
            qtok = self._tokenize_query(query)
            query_intent = self._parse_query_intent(qtok)

            intent_rels = set()
            if query_intent['is_where']:
                intent_rels.update(self.WHERE_RELATIONS)
            if query_intent['is_work']:
                intent_rels.update(self.WORK_RELATIONS)
            if query_intent['is_name']:
                intent_rels.update(self.NAME_RELATIONS)

            if intent_rels:
                for i, (sc, ts, k, p) in enumerate(pool):
                    if k == 'kg' and isinstance(p, (tuple, list)) and len(p) >= 3:
                        s, r, d = p[:3]
                        if r in intent_rels and s == 'you':
                            selected.append((sc, ts, k, p))
                            seen_triples.add((s, r, d))
                            pool.pop(i)
                            break

        # OPTIMIZATION: Simplified MMR with early termination
        lambda_rel = 0.2
        K_max = min(15, len(pool))

        while pool and len(selected) < K_max:
            # OPTIMIZATION: For first few selections, just take top scores
            if len(selected) < 3:
                # Take highest scoring item
                item = pool.pop(0)
                selected.append(item)
                if item[2] == 'kg' and isinstance(item[3], (tuple, list)) and len(item[3]) >= 3:
                    s, r, d = item[3][:3]
                    seen_triples.add((s, r, d))
            else:
                # Use simplified MMR for remaining selections
                best_idx = 0
                best_score = pool[0][0]  # Just use relevance score

                for i in range(1, min(10, len(pool))):  # Check only first 10 items
                    if pool[i][0] > best_score * 0.9:  # If score is close enough
                        best_idx = i
                        break

                item = pool.pop(best_idx)
                selected.append(item)
                if item[2] == 'kg' and isinstance(item[3], (tuple, list)) and len(item[3]) >= 3:
                    s, r, d = item[3][:3]
                    seen_triples.add((s, r, d))

        # Convert to bullets (simplified)
        bullets = []
        for (sc, ts, k, p) in selected:
            if k == 'kg' and isinstance(p, (tuple, list)) and len(p) >= 3:
                s, r, d = p[:3]
                bullets.append((s, r, d))
            else:
                bullets.append(p)

        return bullets

    # Placeholder methods for LEANN and FTS (unchanged from original)
    def _retrieve_with_leann_enhancement(self, query: str, entities: List[str]) -> List[Tuple[float, int, str, Any]]:
        """LEANN enhancement - placeholder for now"""
        return []

    def _search_fts_summaries(self, query: str, limit: int = 12) -> List[Tuple[float, int, str, str]]:
        """FTS search - placeholder for now"""
        if not hasattr(self.store, 'search_fts_detailed'):
            return []
        try:
            results = self.store.search_fts_detailed(query, limit)
            return [(0.5, 0, 'fts', text) for text in results]
        except:
            return []

    def get_metrics(self) -> Dict[str, Any]:
        """Get retrieval performance metrics"""
        return dict(self.metrics)


logger.info("🚀 MemoryRetrieverOptimized initialized - performance optimized version")