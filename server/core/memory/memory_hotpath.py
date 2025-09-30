"""
HotMem: Ultra-fast local memory for voice agents
Full USGS Grammar-to-Graph 27 dependency pattern implementation
Target: <200ms p95 extraction + retrieval
"""

import os
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict, deque
import heapq
from dataclasses import dataclass
import statistics

from loguru import logger
import spacy
from spacy.tokens import Token

from .memory_store import MemoryStore
from .extractors.ud import UDExtractor
from .retrieval import Retrieval
from .confidence_strategy import (
    ConfidenceStrategy,
    RelationTypeConfidence,
    Edge,
    Context
)
from .processors.coreference import CoreferenceProcessor
from .config import MemoryConfig

# Try to import language detection
try:
    import pycld3
    PYCLD3_AVAILABLE = True
except ImportError:
    PYCLD3_AVAILABLE = False
    logger.info("pycld3 not available, defaulting to English")

# DEPRECATED: Legacy NLP loading - migrating to SharedNLPManager
# This function is kept for backward compatibility during migration
def _load_nlp(lang: str = "en"):
    """
    DEPRECATED: Load spaCy model (cached singleton)

    This function is deprecated in favor of SharedNLPManager.
    Use get_nlp_model(lang) instead for new code.
    """
    from .nlp_manager import get_nlp_model
    logger.debug(f"Using SharedNLPManager for language: {lang}")
    return get_nlp_model(lang)

def _norm(text: str) -> str:
    """Fast normalization"""
    return text.lower().strip() if text else ""

_DET_WORDS = {
    "the", "a", "an",
    "my", "your", "his", "her", "their", "our", "its"
}

_PRON_YOU = {"i", "me", "my", "mine", "myself"}

def _strip_leading_dets(text: str) -> str:
    t = _norm(text)
    # Remove leading possessives/determiners
    for det in list(_DET_WORDS):
        if t.startswith(det + " "):
            t = t[len(det) + 1 :]
            break
    # Remove trailing possessive suffix "'s"
    if t.endswith("'s"):
        t = t[:-2]
    return t.strip()

def _canon_entity_text(text: str) -> str:
    t = _norm(text)
    if t in _PRON_YOU:
        return "you"
    t = _strip_leading_dets(t)
    return t

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


class HotMemory:
    """
    Ultra-fast memory with USGS 27 dependency patterns
    All operations target <200ms p95
    """
    
    def __init__(self, store: MemoryStore, max_recency: int = 50,
                 confidence_strategy: Optional[ConfidenceStrategy] = None,
                 enable_dspy_extraction: bool = None):
        self.store = store
        self.user_eid = "you"

        # Confidence scoring strategy (dependency injection)
        self.confidence = confidence_strategy or RelationTypeConfidence()

        # DSPy-enhanced extraction for complex sentences
        self.enable_dspy_extraction = enable_dspy_extraction if enable_dspy_extraction is not None else \
            os.getenv("ENABLE_DSPY_EXTRACTION", "false").lower() in ("true", "1", "yes")

        # Lazy load DSPy extractor (only if enabled)
        self._dspy_extractor = None
        self._complexity_detector = None

        # Hot indices (RAM)
        self.entity_index = defaultdict(set)  # entity -> set of (s,r,d) triples
        self.recency_buffer = deque(maxlen=max_recency)  # Recent interactions
        self.entity_cache = {}  # Canonical entity mapping

        # Performance tracking
        self.metrics = defaultdict(list)
        self.max_metric_size = 1000
        # Extractor (Phase 1C): adapter to existing implementation
        self.extractor = UDExtractor(self)
        self.retriever = Retrieval(self)

        # Coreference resolution (SOLID refactored component)
        config = MemoryConfig.from_env()
        self.coref_processor = CoreferenceProcessor(
            timeout_ms=config.coreference.timeout_ms,
            min_text_length=config.coreference.min_text_length,
            lang=config.coreference.lang
        ) if config.coreference.enabled else None

    def prewarm(self, lang: str = "en") -> None:
        """Load NLP resources up-front to avoid first-turn latency."""
        try:
            _load_nlp(lang)
        except Exception:
            pass

    def process_turn(self, text: str, session_id: str, turn_id: int, focus: str = 'standard') -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Process a conversation turn
        Returns: (memory_bullets, extracted_triples)
        """
        start = time.perf_counter()

        # Log the focus strategy being used (for debugging)
        if focus != 'standard':
            logger.debug(f"[HotMem] Using focus strategy: {focus}")
        
        # Language detection
        lang = self._detect_language(text) if PYCLD3_AVAILABLE else "en"
        
        # Stage 1: Extract entities and relations (via extractor seam)
        # NOTE: Coreference resolution exists but needs proper spacy-coref integration
        # TODO: Implement proper coref that resolves pronouns in doc before extraction
        extract_start = time.perf_counter()
        entities, triples, neg_count, doc, entity_aliases = self.extractor.extract(text, lang)
        self.metrics['extraction_ms'].append((time.perf_counter() - extract_start) * 1000)
        # Store aliases for dual registration in hot index
        self._entity_aliases = entity_aliases
        logger.debug(f"[HotMem] Extracted {len(triples)} raw triples from '{text[:50]}...'")
        if triples:
            logger.debug(f"[HotMem] Raw triples (first 3): {triples[:3]}")

        # Stage 1.5: DSPy-enhanced extraction for complex sentences
        if self.enable_dspy_extraction and doc:
            dspy_start = time.perf_counter()
            additional_triples = self._extract_with_dspy(text, triples, doc)
            if additional_triples:
                triples.extend(additional_triples)
                logger.debug(f"[HotMem] DSPy added {len(additional_triples)} edges")
            self.metrics['dspy_extraction_ms'].append((time.perf_counter() - dspy_start) * 1000)

        # Stage 2: Refine triples and update memory with new facts (skip writes for questions)
        refine_start = time.perf_counter()
        triples = self.extractor.refine(text, triples, doc)
        logger.debug(f"[HotMem] After refinement: {len(triples)} triples")
        # Rebuild entities from refined triples + text context
        ent_from_triples: Set[str] = set()
        for s, r, d in triples:
            ent_from_triples.add(s)
            ent_from_triples.add(d)
        entities = self.extractor.refine_entities(text, list(ent_from_triples))

        # Ensure base aliases (e.g., "swimming") are present alongside enriched forms
        # (e.g., "swimming in the sea") so retrieval can fan out on both keys.
        if self._entity_aliases:
            base_entities: Set[str] = set(self._entity_aliases.values())
            seen_entities: Set[str] = set(entities)
            for base_entity in base_entities:
                if base_entity not in seen_entities:
                    entities.append(base_entity)
                    seen_entities.add(base_entity)

        # Filter noisy triples before storing/retrieving
        triples_before_filter = len(triples)
        triples = [t for t in triples if self._is_meaningful_fact(*t)]
        logger.debug(f"[HotMem] After filtering: {len(triples)} triples (removed {triples_before_filter - len(triples)})")
        if triples:
            logger.debug(f"[HotMem] Filtered triples (first 3): {triples[:3]}")

        update_start = time.perf_counter()
        now_ts = int(time.time() * 1000)

        # Store the conversation turn FIRST (before edge extraction) for provenance
        turn_id_hash = self.store.enqueue_turn(text, session_id, turn_id, now_ts)

        is_question = self._is_question(text)
        logger.debug(f"[HotMem] Text classified as question: {is_question}")

        if not is_question:
            for s, r, d in triples:
                # Demote conflicting facts before observing new evidence
                conflicting = [fact for fact in list(self.entity_index.get(s, set())) if fact[1] == r and fact[2] != d]
                for _s, _r, old_d in conflicting:
                    try:
                        self.store.negate_edge(s, r, old_d, conf=0.6, now_ts=now_ts)
                    except Exception as e:
                        logger.warning(f"HotMem demotion failed for ({s}, {r}, {old_d}): {e}")
                    self.entity_index[s].discard((_s, _r, old_d))
                    if old_d in self.entity_index:
                        self.entity_index[old_d].discard((_s, _r, old_d))
                    self._prune_recency_item(s, r, old_d)

                # Compute confidence using injected strategy
                edge_id = self.store.edge_id(s, r, d)

                # Get edge data for confidence calculation
                # (pos/neg come from existing edge if it exists)
                cur = self.store.sql.cursor()
                edge_data = cur.execute(
                    "SELECT pos, neg, updated_at FROM edge WHERE id = ?",
                    (edge_id,)
                ).fetchone()

                if edge_data:
                    pos, neg, updated_at = edge_data
                else:
                    pos, neg, updated_at = 0, 0, now_ts

                # Create Edge object for strategy
                edge_obj = Edge(
                    src=s, rel=r, dst=d,
                    pos=pos, neg=neg,
                    updated_at=updated_at,
                    id=edge_id
                )
                context_obj = Context(
                    store=self.store,
                    text=text,
                    session_id=session_id,
                    turn_id=turn_id
                )

                # Score confidence
                conf = self.confidence.score(edge_obj, context_obj)

                # Apply negation to verb-based relations when negation detected
                if neg_count > 0:
                    try:
                        self.store.negate_edge(s, r, d, conf=0.6, now_ts=now_ts)
                        logger.debug(f"[HotMem] Negated: ({s}, {r}, {d})")
                    except Exception as e:
                        logger.warning(f"HotMem negation failed for ({s}, {r}, {d}): {e}")
                else:
                    self.store.observe_edge(s, r, d, conf, now_ts)

                # Link edge to conversation turn (provenance)
                edge_id = self.store.edge_id(s, r, d)
                self.store.enqueue_edge_source(edge_id, turn_id_hash, now_ts)

                # Update hot indices
                self.entity_index[s].add((s, r, d))
                self.entity_index[d].add((s, r, d))

                # Dual registration: If dst was enriched, also index under base form
                # This enables queries like "swimming" to find "swimming in the sea"
                base_d = self._entity_aliases.get(d, d)
                if base_d != d:
                    self.entity_index[base_d].add((s, r, d))
        self.metrics['update_ms'].append((time.perf_counter() - update_start) * 1000)
        
        # Stage 3: Retrieve relevant memories
        retrieve_start = time.perf_counter()
        bullets = self._retrieve_context(text, entities, turn_id)
        self.metrics['retrieval_ms'].append((time.perf_counter() - retrieve_start) * 1000)
        
        # Update recency with extracted triples
        for s, r, d in triples:
            self.recency_buffer.append(RecencyItem(s, r, d, text, now_ts, turn_id))
        
        # Track overall performance
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics['total_ms'].append(elapsed_ms)
        self._cleanup_metrics()
        
        if elapsed_ms > 200:
            logger.warning(f"Hot path took {elapsed_ms:.1f}ms (budget: 200ms)")
        
        return bullets, triples
    
    def _extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any, Dict[str, str]]:
        """
        Extract entities and relations using USGS 27-pattern approach
        Returns: (entities, triples, negation_count, doc, entity_aliases)
        """
        nlp = _load_nlp(lang)

        if not nlp:
            return [], [], 0, None, {}

        doc = nlp(text)
        entities = set()
        triples = []
        neg_count = 0

        # Initialize per-extraction tracking
        self._entity_aliases = {}  # enriched -> base mapping
        self._enriched_entities = set()  # base entities that were enriched
        
        # Stage 1: Build entity map
        entity_map = self._build_entity_map(doc, entities)
        
        # Stage 2: Process all 27 dependency types
        for token in doc:
            dep = token.dep_
            
            # Core grammatical relations
            if dep in {"nsubj", "nsubjpass"}:
                self._extract_subject(token, entity_map, triples, entities)
            elif dep in {"dobj", "obj"}:
                self._extract_object(token, entity_map, triples, entities)
            elif dep == "iobj":
                self._extract_indirect_object(token, entity_map, triples, entities)
            elif dep == "attr":
                self._extract_attribute(token, entity_map, triples, entities)
            elif dep == "acomp":
                self._extract_acomp(token, entity_map, triples, entities)
            
            # Modifiers
            elif dep == "amod":
                self._extract_amod(token, entity_map, triples, entities)
            elif dep == "advmod":
                self._extract_advmod(token, entity_map, triples, entities)
            elif dep == "nummod":
                self._extract_nummod(token, entity_map, triples, entities)
            elif dep == "nmod":
                self._extract_nmod(token, entity_map, triples, entities)
            
            # Structural
            elif dep == "compound":
                self._extract_compound(token, entity_map, triples, entities)
            elif dep == "poss":
                self._extract_possessive(token, entity_map, triples, entities)
            elif dep == "appos":
                self._extract_appos(token, entity_map, triples, entities)
            elif dep == "conj":
                self._extract_conj(token, entity_map, triples, entities)
            elif dep == "prep":
                self._extract_prep(token, entity_map, triples, entities)
            elif dep == "pobj":
                pass  # Handled by prep
            
            # Clausal
            elif dep == "acl":
                self._extract_acl(token, entity_map, triples, entities)
            elif dep == "advcl":
                self._extract_advcl(token, entity_map, triples, entities)
            elif dep == "ccomp":
                self._extract_ccomp(token, entity_map, triples, entities)
            elif dep == "csubj":
                self._extract_csubj(token, entity_map, triples, entities)
            elif dep == "xcomp":
                self._extract_xcomp(token, entity_map, triples, entities)
            
            # Special
            elif dep == "agent":
                self._extract_agent(token, entity_map, triples, entities)
            elif dep == "oprd":
                self._extract_oprd(token, entity_map, triples, entities)
            
            # Count negations
            elif dep == "neg":
                neg_count += 1

        return list(entities), triples, neg_count, doc, self._entity_aliases
    
    def _build_entity_map(self, doc, entities: Set[str]) -> Dict[int, str]:
        """Build entity map from document"""
        entity_map = {}
        
        # Named entities
        for ent in doc.ents:
            norm_text = _canon_entity_text(ent.text)
            entities.add(norm_text)
            for token in ent:
                entity_map[token.i] = norm_text
        
        # Noun chunks
        for chunk in doc.noun_chunks:
            chunk_text = _canon_entity_text(chunk.text)
            entities.add(chunk_text)
            entity_map[chunk.root.i] = chunk_text
        
        # Individual tokens
        for token in doc:
            if token.i not in entity_map:
                if token.pos_ in {"NOUN", "PROPN", "PRON"}:
                    entity_text = _canon_entity_text(token.text)
                    # Canonicalize pronouns
                    if entity_text in _PRON_YOU:
                        entity_text = self.user_eid
                    entities.add(entity_text)
                    entity_map[token.i] = entity_text
        
        return entity_map
    
    # === 27 Dependency Handlers ===
    
    def _get_entity(self, token, entity_map) -> str:
        """Get entity for token"""
        return entity_map.get(token.i, _norm(token.text))

    def _extract_base_entity(self, entity: str) -> str:
        """
        Heuristic to extract base from enriched form.

        Examples:
        - "swimming in the sea" -> "swimming" (first word)
        - "red car" -> "car" (last word if multiple)
        - "machine learning" -> "learning" (last word)

        Strategy: If multi-word, check if contains prepositions (in, on, at, with, for).
        If yes, base is first word. Otherwise, base is last word (compound pattern).
        """
        words = entity.split()
        if len(words) <= 1:
            return entity

        # Check for prep pattern: "X in Y", "X on Y"
        preps = {"in", "on", "at", "with", "for", "from", "to", "by"}
        if any(w in preps for w in words[1:]):
            return words[0]  # "swimming" from "swimming in sea"

        # Otherwise assume compound: "machine learning" -> "learning"
        return words[-1]

    def _get_entity_with_context(self, token, entity_map, max_length: int = 50) -> tuple[str, str]:
        """
        Get entity with full contextual modifiers.

        Returns:
            (root_entity, enriched_entity) tuple for dual registration

        CRITICAL: root stays untouched (canonical base form), enriched gets modifiers.
        This ensures entity_index["car"] finds "red car" edges.

        Includes:
        1. Prepositional phrases (location, time, manner)
        2. Adjectival modifiers (attributes)
        3. Compound nouns (multi-word concepts)

        Examples:
        - root="swimming", enriched="swimming in the sea"       ← prep
        - root="car", enriched="red car"                        ← amod
        - root="learning", enriched="machine learning"          ← compound
        - root="meeting", enriched="meeting on tuesday"         ← prep
        """
        import time
        start = time.perf_counter()

        # Get root entity (canonical form) - NEVER MODIFIED
        # IMPORTANT: entity_map may already contain full noun chunks ("red car"),
        # so derive the true head from the token itself before consulting the map.
        raw_root = token.lemma_ or token.text
        root = _canon_entity_text(raw_root)  # Apply canonical form immediately

        # If entity_map carries a chunk-alias (e.g., "red car"), record it now so
        # we can index the enriched edge under both the chunk and the canonical root.
        chunk_alias = entity_map.get(token.i)
        if chunk_alias and _canon_entity_text(chunk_alias) != root:
            self._entity_aliases[_canon_entity_text(chunk_alias)] = root

        # Start building enriched from root
        enriched = root

        # Phase 3: Collect compound nouns (comes before root)
        # Cap at 3 compounds to prevent pathological cases
        # IMPORTANT: Sort by token.i to preserve left-to-right order ("machine learning", not "learning machine")
        compounds = []
        for child in sorted(token.children, key=lambda t: t.i):
            if child.dep_ == "compound" and len(compounds) < 3:
                compounds.append(_canon_entity_text(child.text))

        # Build enriched with compounds (root stays untouched)
        if compounds:
            enriched = " ".join(compounds + [enriched])

        # Phase 2: Collect adjectives
        # Cap at 5 adjectives to prevent pathological cases
        # IMPORTANT: Sort by token.i to preserve natural order ("big blue house", not "blue big house")
        adjectives = []
        for child in sorted(token.children, key=lambda t: t.i):
            if child.dep_ == "amod" and len(adjectives) < 5:
                adjectives.append(_canon_entity_text(child.text))

        # Add adjectives before enriched (root stays untouched)
        if adjectives:
            enriched = " ".join(adjectives + [enriched])

        # Phase 1: Collect prepositional phrases
        # Cap at 3 prep phrases to prevent pathological cases
        prep_parts = []
        for child in token.children:
            if child.dep_ == "prep" and len(prep_parts) < 3:
                prep_text = child.text.lower()
                # Get prepositional object
                for pobj_child in child.children:
                    if pobj_child.dep_ == "pobj":
                        pobj_text = self._get_entity(pobj_child, entity_map)
                        pobj_text = _canon_entity_text(pobj_text)
                        prep_parts.append(f"{prep_text} {pobj_text}")
                        break  # Only first pobj per prep

        # Combine with prep phrases (root stays untouched)
        if prep_parts:
            enriched = f"{enriched} {' '.join(prep_parts)}"

        # Cap length to prevent pathological cases
        if len(enriched) > max_length:
            truncated = enriched[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
            # Fallback if truncation produces empty string
            if truncated:
                enriched = truncated
            else:
                enriched = enriched[:max_length]  # Hard cut as last resort
            # Track truncations for monitoring
            if hasattr(self, '_metrics'):
                self._metrics['enrichment_truncations'] = self._metrics.get('enrichment_truncations', 0) + 1

        # Performance monitoring
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Only log slow-path warning if debug logging enabled to avoid log noise
        if elapsed_ms > 1.0 and logger.level <= 10:  # DEBUG = 10
            logger.debug(f"Slow entity enrichment: {elapsed_ms:.2f}ms for '{enriched}'")

        # Track metrics
        if hasattr(self, '_metrics'):
            self._metrics.setdefault('entity_enrichment_times_ms', []).append(elapsed_ms)
            self._metrics.setdefault('enriched_lengths', []).append(len(enriched))

        # Store alias mapping if enriched differs from root
        if enriched != root:
            self._entity_aliases[enriched] = root

        return root, enriched

    def _extract_subject(self, token, entity_map, triples, entities):
        """nsubj, nsubjpass - nominal subject"""
        subj = self._get_entity(token, entity_map)
        head = token.head
        
        # Passive: "My son is named Jake"
        if token.dep_ == "nsubjpass" and head.pos_ == "VERB":
            verb = head.lemma_.lower()
            if verb in {"name", "call"}:
                for child in head.children:
                    if child.dep_ == "oprd":
                        name = self._get_entity(child, entity_map)
                        triples.append((subj, "name", name))
                        entities.add(name)
                        # Check for possessive
                        for gc in token.children:
                            if gc.dep_ == "poss" and gc.text.lower() in {"my", "mine"}:
                                triples.append((self.user_eid, "has", subj))
                        return
        
        # Copula: X is Y
        if head.pos_ == "AUX" or any(c.dep_ == "cop" for c in head.children):
            for child in head.children:
                if child.dep_ == "attr":
                    root_obj, enriched_obj = self._get_entity_with_context(child, entity_map)
                    # Special: "My name is X"
                    if token.text.lower() == "name":
                        for gc in token.children:
                            if gc.dep_ == "poss" and gc.text.lower() in {"my", "mine"}:
                                triples.append((self.user_eid, "name", enriched_obj))
                                entities.add(root_obj)
                                self._enriched_entities.add(root_obj)
                                return
                    triples.append((subj, "is", enriched_obj))
                    entities.add(root_obj)
                    self._enriched_entities.add(root_obj)
        
        # Active verb: X verbs Y
        elif head.pos_ == "VERB":
            verb = head.lemma_.lower()
            
            # Direct object
            for child in head.children:
                if child.dep_ in {"dobj", "obj"}:
                    root_obj, enriched_obj = self._get_entity_with_context(child, entity_map)
                    pred = "has" if verb in {"have", "has", "had", "own"} else verb
                    triples.append((subj, pred, enriched_obj))
                    entities.add(root_obj)
                    self._enriched_entities.add(root_obj)
            
            # Prepositional complement
            for child in head.children:
                if child.dep_ == "prep":
                    prep = child.text.lower()
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            obj = self._get_entity(gc, entity_map)
                            # Special patterns
                            if verb == "live" and prep == "in":
                                triples.append((subj, "lives_in", obj))
                            elif verb == "work" and prep in {"at", "for"}:
                                triples.append((subj, "works_at", obj))
                            elif verb in {"go", "went"} and prep == "to":
                                triples.append((subj, "went_to", obj))
                            elif verb in {"move", "moved"} and prep == "from":
                                triples.append((subj, "moved_from", obj))
                            elif verb in {"participate", "participated"} and prep == "in":
                                triples.append((subj, "participated_in", obj))
                            elif verb in {"born", "bear"} and prep == "in":
                                triples.append((subj, "born_in", obj))
                            elif verb in {"paint", "painted"}:
                                triples.append((subj, "painted", obj))
                                if prep == "in":  # temporal
                                    continue
                            elif verb in {"read"}:
                                triples.append((subj, "read", obj))
                            else:
                                triples.append((subj, f"{verb}_{prep}", obj))
                            entities.add(obj)

            # Conjoined verbs (inherit subject unless explicit)
            for v2 in [c for c in head.children if c.dep_ == "conj" and c.pos_ == "VERB"]:
                # Prefer explicit subject on the conj verb, else inherit
                subj2 = None
                for c2 in v2.children:
                    if c2.dep_ in {"nsubj", "nsubjpass"}:
                        subj2 = self._get_entity(c2, entity_map)
                        break
                subj2 = subj2 or subj
                verb2 = v2.lemma_.lower()

                # Objects of the conj verb
                for ch in v2.children:
                    if ch.dep_ in {"dobj", "obj"}:
                        root_obj, enriched_obj = self._get_entity_with_context(ch, entity_map)
                        pred = "has" if verb2 in {"have", "has", "had", "own"} else verb2
                        triples.append((subj2, pred, enriched_obj))
                        entities.add(root_obj)
                        self._enriched_entities.add(root_obj)

                # Prepositional complements of the conj verb
                for ch in v2.children:
                    if ch.dep_ == "prep":
                        prep = ch.text.lower()
                        for gc in ch.children:
                            if gc.dep_ == "pobj":
                                obj = self._get_entity(gc, entity_map)
                                if verb2 == "live" and prep == "in":
                                    triples.append((subj2, "lives_in", obj))
                                elif verb2 == "work" and prep in {"at", "for"}:
                                    triples.append((subj2, "works_at", obj))
                                elif verb2 in {"go", "went"} and prep == "to":
                                    triples.append((subj2, "went_to", obj))
                                elif verb2 in {"move", "moved"} and prep == "from":
                                    triples.append((subj2, "moved_from", obj))
                                elif verb2 in {"participate", "participated"} and prep == "in":
                                    triples.append((subj2, "participated_in", obj))
                                elif verb2 in {"born", "bear"} and prep == "in":
                                    triples.append((subj2, "born_in", obj))
                                else:
                                    triples.append((subj2, f"{verb2}_{prep}", obj))
                                entities.add(obj)
    
    def _extract_object(self, token, entity_map, triples, entities):
        """dobj, obj - direct object"""
        root_obj, enriched_obj = self._get_entity_with_context(token, entity_map)
        head = token.head

        if head.pos_ == "VERB":
            for child in head.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subj = self._get_entity(child, entity_map)
                    verb = head.lemma_.lower()
                    pred = "has" if verb in {"have", "has", "had"} else verb
                    triples.append((subj, pred, enriched_obj))
                    entities.add(root_obj)
                    self._enriched_entities.add(root_obj)
                    break

    def _extract_indirect_object(self, token, entity_map, triples, entities):
        """iobj - indirect object"""
        root_iobj, enriched_iobj = self._get_entity_with_context(token, entity_map)
        head = token.head

        # Find subject
        for child in head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj = self._get_entity(child, entity_map)
                triples.append((subj, f"gave_to", enriched_iobj))
                entities.add(root_iobj)
                self._enriched_entities.add(root_iobj)
                break

    def _extract_attribute(self, token, entity_map, triples, entities):
        """attr - attribute (copula complement)"""
        root_attr, enriched_attr = self._get_entity_with_context(token, entity_map)

        for child in token.head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj = self._get_entity(child, entity_map)
                triples.append((subj, "is", enriched_attr))
                entities.add(root_attr)
                self._enriched_entities.add(root_attr)
                break

    def _extract_acomp(self, token, entity_map, triples, entities):
        """acomp - adjectival complement (copula complement)"""
        # Handle patterns like "Caroline is single"
        root_adj, enriched_adj = self._get_entity_with_context(token, entity_map)
        head = token.head

        # Find subject of copula
        for child in head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj = self._get_entity(child, entity_map)
                triples.append((subj, "is", enriched_adj))
                entities.add(root_adj)
                self._enriched_entities.add(root_adj)
                break
    
    def _extract_amod(self, token, entity_map, triples, entities):
        """amod - adjectival modifier"""
        # Only extract quality triple if parent wasn't enriched
        # (Enriched objects already contain adjectives)
        head_entity = self._get_entity(token.head, entity_map)
        head_base = _canon_entity_text(token.head.lemma_ or token.head.text)

        if head_base not in self._enriched_entities:
            adj = token.text.lower()
            triples.append((head_entity, "quality", adj))
    
    def _extract_advmod(self, token, entity_map, triples, entities):
        """advmod - adverbial modifier"""
        # Usually modifies actions, skip for entities
        pass
    
    def _extract_nummod(self, token, entity_map, triples, entities):
        """nummod - numeric modifier"""
        num = token.text
        head_entity = self._get_entity(token.head, entity_map)
        triples.append((head_entity, "quantity", num))
    
    def _extract_nmod(self, token, entity_map, triples, entities):
        """nmod - nominal modifier"""
        mod = self._get_entity(token, entity_map)
        head_entity = self._get_entity(token.head, entity_map)
        triples.append((head_entity, "modified_by", mod))
    
    def _extract_compound(self, token, entity_map, triples, entities):
        """compound - multiword expression"""
        part = self._get_entity(token, entity_map)
        whole = self._get_entity(token.head, entity_map)
        # Usually forms a single entity, already in entity_map
        pass
    
    def _extract_possessive(self, token, entity_map, triples, entities):
        """poss - possessive"""
        possessor = self._get_entity(token, entity_map)
        possessed = self._get_entity(token.head, entity_map)
        
        if possessor in {"my", "mine"}:
            possessor = self.user_eid
        
        triples.append((possessor, "has", possessed))
        entities.add(possessed)
    
    def _extract_appos(self, token, entity_map, triples, entities):
        """appos - apposition"""
        entity1 = self._get_entity(token.head, entity_map)
        entity2 = self._get_entity(token, entity_map)
        triples.append((entity1, "also_known_as", entity2))
    
    def _extract_conj(self, token, entity_map, triples, entities):
        """conj - conjunction"""
        # Skip verb-verb conjunctions (e.g., "live and work") to reduce noise
        if token.head.pos_ == "VERB" and token.pos_ == "VERB":
            return
        item1 = self._get_entity(token.head, entity_map)
        item2 = self._get_entity(token, entity_map)
        triples.append((item1, "and", item2))
    
    def _extract_prep(self, token, entity_map, triples, entities):
        """prep - preposition (handled in subject extraction)"""
        pass
    
    def _extract_acl(self, token, entity_map, triples, entities):
        """acl - adnominal clause"""
        # Complex clausal relation
        pass
    
    def _extract_advcl(self, token, entity_map, triples, entities):
        """advcl - adverbial clause"""
        # Complex clausal relation
        pass
    
    def _extract_ccomp(self, token, entity_map, triples, entities):
        """ccomp - clausal complement"""
        # Handle patterns like "Melanie has read [Nothing is Impossible]"
        head = token.head
        if head.pos_ == "VERB":
            verb = head.lemma_.lower()
            if verb in {"read", "write", "say", "think", "know"}:
                # Find subject of main verb
                for child in head.children:
                    if child.dep_ in {"nsubj", "nsubjpass"}:
                        subj = self._get_entity(child, entity_map)
                        # Extract the clause as object
                        obj_tokens = []
                        for desc in token.subtree:
                            obj_tokens.append(desc.text)
                        obj = " ".join(obj_tokens).strip()
                        if obj:
                            obj = _canon_entity_text(obj)
                            triples.append((subj, verb, obj))
                            entities.add(obj)
                        break
    
    def _extract_csubj(self, token, entity_map, triples, entities):
        """csubj - clausal subject"""
        # Complex clausal relation
        pass
    
    def _extract_xcomp(self, token, entity_map, triples, entities):
        """xcomp - open clausal complement"""
        # Handle patterns like "likes reading", "wants to go"
        head = token.head
        if head.pos_ == "VERB":
            # Find subject of main verb (Caroline is nsubj of "likes")
            subj = None
            for child in head.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subj = self._get_entity(child, entity_map)
                    break

            if subj:
                root_obj, enriched_obj = self._get_entity_with_context(token, entity_map)
                verb = head.lemma_.lower()
                triples.append((subj, verb, enriched_obj))
                entities.add(root_obj)
                self._enriched_entities.add(root_obj)
    
    def _extract_agent(self, token, entity_map, triples, entities):
        """agent - agent (by-phrase in passive)"""
        agent = self._get_entity(token, entity_map)
        action = token.head.lemma_.lower()
        triples.append((agent, "performed", action))
    
    def _extract_oprd(self, token, entity_map, triples, entities):
        """oprd - object predicate"""
        # In passive constructions like "is named X"
        oprd = self._get_entity(token, entity_map)
        for child in token.head.children:
            if child.dep_ in {"nsubjpass"}:
                subj = self._get_entity(child, entity_map)
                if token.head.lemma_ in {"name", "call"}:
                    triples.append((subj, "name", oprd))
                    entities.add(oprd)
                break
    
    def _retrieve_context(self, query: str, entities: List[str], turn_id: int) -> List[str]:
        """Compatibility shim: delegate to retriever (no behavior change)."""
        try:
            return self.retriever.retrieve(query, entities, turn_id)
        except Exception:
            return []
    
    def _detect_language(self, text: str) -> str:
        """Detect language using env override or pycld3"""
        hint = os.getenv("HOTMEM_LANG")
        if hint and len(hint) >= 2:
            return hint[:2]
        if PYCLD3_AVAILABLE:
            try:
                result = pycld3.get_language(text)
                return result.language[:2] if result.is_reliable else "en"
            except Exception:
                return "en"
        return "en"

    def _get_dspy_extractor(self):
        """Lazy load DSPy extractor"""
        if self._dspy_extractor is None:
            try:
                from .dspy_extractor import create_dspy_extractor
                self._dspy_extractor = create_dspy_extractor()
                logger.info("DSPy extractor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize DSPy extractor: {e}")
                self._dspy_extractor = False  # Mark as failed
        return self._dspy_extractor if self._dspy_extractor is not False else None

    def _get_complexity_detector(self):
        """Lazy load complexity detector"""
        if self._complexity_detector is None:
            try:
                from .complexity_detector import ComplexityDetector
                self._complexity_detector = ComplexityDetector()
                logger.info("Complexity detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize complexity detector: {e}")
                self._complexity_detector = False
        return self._complexity_detector if self._complexity_detector is not False else None

    def _extract_with_dspy(self, text: str, existing_triples: List[Tuple[str, str, str]], doc: Any) -> List[Tuple[str, str, str]]:
        """
        Extract additional edges using DSPy for complex sentences

        IMPORTANT: Each DSPy call should be isolated to avoid context pollution.
        The extractor is cached but each extraction is independent.

        Args:
            text: Original text
            existing_triples: Edges already extracted by spaCy
            doc: spaCy Doc object

        Returns:
            Additional edges found by DSPy
        """
        try:
            # Check complexity
            detector = self._get_complexity_detector()
            if not detector:
                return []

            is_complex, metrics = detector.is_complex(doc)

            if not is_complex:
                logger.debug(f"[HotMem] Sentence not complex (score={metrics['complexity_score']:.2f}), skipping DSPy")
                return []

            logger.debug(f"[HotMem] Complex sentence detected (score={metrics['complexity_score']:.2f}), using DSPy")

            # Get DSPy extractor (cached, lazy loaded)
            extractor = self._get_dspy_extractor()
            if not extractor:
                return []

            # Extract missing edges (each call is isolated - no session pollution)
            missing_edges = extractor.extract_missing_edges(text, existing_triples)

            if not missing_edges:
                return []

            # Filter quality
            from .edge_quality_filter import filter_edges
            filtered_edges = filter_edges(missing_edges, existing_triples)

            logger.info(
                f"[HotMem] DSPy extraction: {len(missing_edges)} raw → {len(filtered_edges)} filtered "
                f"(complexity={metrics['complexity_score']:.2f})"
            )

            return filtered_edges

        except Exception as e:
            logger.error(f"[HotMem] DSPy extraction failed: {e}")
            return []
    
    def _cleanup_metrics(self):
        """Keep metrics bounded"""
        for key in self.metrics:
            if len(self.metrics[key]) > self.max_metric_size:
                self.metrics[key] = self.metrics[key][-self.max_metric_size:]

    def _prune_recency_item(self, s: str, r: str, d: str) -> None:
        """Remove an existing triple from the recency buffer if present."""
        try:
            filtered = [item for item in self.recency_buffer if not (item.s == s and item.r == r and item.d == d)]
            self.recency_buffer = deque(filtered, maxlen=self.recency_buffer.maxlen)
        except Exception as e:
            logger.warning(f"Failed pruning recency buffer for ({s}, {r}, {d}): {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        results = {}
        for key, values in self.metrics.items():
            if values:
                results[key] = {
                    'mean': statistics.mean(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) > 20 else max(values),
                    'count': len(values)
                }
        
        results['entities'] = len(self.entity_index)
        results['recency_buffer'] = len(self.recency_buffer)
        
        return results
    
    def rebuild_from_store(self):
        """Rebuild hot indices from persistent store"""
        start = time.perf_counter()
        count = 0
        
        logger.info("Rebuilding hot memory from store...")
        
        # Get all edges from store
        edges = self.store.get_all_edges()
        
        for s, r, d, conf in edges:
            if conf > 0.1:  # Only active edges
                self.entity_index[s].add((s, r, d))
                self.entity_index[d].add((s, r, d))

                # Also index under base form if dst looks enriched
                base_d = self._extract_base_entity(d)
                if base_d != d:
                    self.entity_index[base_d].add((s, r, d))

                count += 1
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Rebuilt {count} edges in {elapsed_ms:.1f}ms")

    # ---------- Non-mutating helpers for verification ----------
    def preview_bullets(self, text: str, lang: str = "en") -> Dict[str, Any]:
        """Return entities detected and bullets that would be injected, without updating store.

        Useful for validating retrieval independently of writes.
        """
        try:
            entities, _, _, _, _ = self.extractor.extract(text, lang)
            entities = self.extractor.refine_entities(text, entities)
        except Exception:
            entities = []
        bullets = self.retriever.retrieve(text, entities, turn_id=-1)
        return {"entities": entities, "bullets": bullets}

    # Phase 0: unified retrieval entry point (read-only or normal)
    def retrieve_bullets(self, text: str, read_only: bool = True, lang: str = "en") -> List[str]:
        """
        Retrieve bullets for the given text.

        - read_only=True: does not perform any store updates or recency changes; uses extraction + retrieval only.
        - read_only=False: behaves like normal retrieval path after extraction/persist (callers should have persisted if needed).
        """
        if read_only:
            try:
                entities, _, _, _, _ = self.extractor.extract(text, lang)
                entities = self.extractor.refine_entities(text, entities)
            except Exception:
                entities = []
            return self.retriever.retrieve(text, entities, turn_id=-1)
        else:
            # Non read-only: reuse preview path for now; callers may have called process_turn before this.
            try:
                entities, _, _, _, _ = self.extractor.extract(text, lang)
                entities = self.extractor.refine_entities(text, entities)
            except Exception:
                entities = []
            return self.retriever.retrieve(text, entities, turn_id=-1)

    # ---------- Refinement helpers (quality without large perf cost) ----------
    def _is_question(self, text: str) -> bool:
        """Conservative question detector.

        Treat as a question if the final punctuation is a question mark,
        or if it starts with a wh-word. Allows facts in multi-sentence inputs
        that contain a question earlier but end with a statement.
        """
        t = (text or "").strip().lower()
        if not t:
            return False
        # Ends with a question mark → question
        if t.endswith("?"):
            return True
        # Starts with wh-word → likely a question
        wh = ("who", "what", "when", "where", "why", "how")
        return any(t.startswith(w + " ") for w in wh)

    def _is_meaningful_fact(self, s: str, r: str, d: str) -> bool:
        """Filter out conversational junk so we only persist actionable facts."""
        s_norm = (s or "").strip().lower()
        r_norm = (r or "").strip().lower()
        d_norm = (d or "").strip().lower()

        if not s_norm or not d_norm or len(d_norm) < 2:
            return False

        # Ignore filler subjects/objects
        stop_entities = {"it", "this", "that", "there", "here", "been"}
        if s_norm in stop_entities or d_norm in stop_entities:
            return False

        # Ignore filler relations
        stop_relations = {
            "and",
            "know",
            "remember",
            "say",
            "tell",
            "think",
            "ask",
            "quality",
            "tell_about",
        }
        if r_norm in stop_relations:
            return False

        # Guard generic "is" facts unless subject is meaningful
        if r_norm == "is" and (s_norm in stop_entities or d_norm.startswith("what ")):
            return False

        return True

    def _refine_entities_from_text(self, text: str, entities: List[str]) -> List[str]:
        # Canonicalize and drop noisy scaffolding like 'my name'
        out = []
        for e in entities:
            ce = _canon_entity_text(e)
            if ce and ce not in {"my name", "my dog's name", "name"}:
                out.append(ce)
        # Ensure 'you' if text is self-referential
        t = (text or "").lower()
        if any(p in t for p in [" i ", " my ", " me "]) and "you" not in out:
            out.append("you")
        # Unique, preserve order
        seen = set()
        uniq = []
        for e in out:
            if e not in seen:
                uniq.append(e)
                seen.add(e)
        return uniq

    def _refine_triples(self, text: str, triples: List[Tuple[str, str, str]], doc) -> List[Tuple[str, str, str]]:
        t = (text or "").lower()
        refined: List[Tuple[str, str, str]] = []

        # Name patterns from raw text (more reliable than dep combos alone)
        # 1) My name is X
        m = None
        try:
            m = __import__("re").search(r"\bmy name is\s+([^,.!?]+)", t)
        except Exception:
            m = None
        if m:
            name = _canon_entity_text(m.group(1))
            refined.append(("you", "name", name))

        # 2) My dog's name is X
        md = None
        try:
            md = __import__("re").search(r"\bmy dog'?s name is\s+([^,.!?]+)", t)
        except Exception:
            md = None
        if md:
            dname = _canon_entity_text(md.group(1))
            refined.append(("dog", "name", dname))
            refined.append(("you", "has", "dog"))

        # 3) My son is named X
        ms = None
        try:
            ms = __import__("re").search(r"\bmy son is named\s+([^,.!?]+)", t)
        except Exception:
            ms = None
        if ms:
            sname = _canon_entity_text(ms.group(1))
            refined.append(("son", "name", sname))
            refined.append(("you", "has", "son"))

        # 4) Favorite color is X → favorite_color
        fc = None
        try:
            fc = __import__("re").search(r"\bfavorite color is\s+([^,.!?]+)", t)
        except Exception:
            fc = None
        if fc:
            fav = _canon_entity_text(fc.group(1))
            refined.append(("you", "favorite_color", fav))

        for s, r, d in triples:
            cs = _canon_entity_text(s)
            cd = _canon_entity_text(d)

            # Pronouns → you
            if cs in _PRON_YOU:
                cs = "you"
            if cd in _PRON_YOU:
                cd = "you"

            rr = r
            # Fix generic preposition rels if the verb is inferable from surface text
            if r in {"_in", "_at", "_from", "_to"}:
                if " live" in t or t.startswith("live") or " living" in t:
                    rr = "lives_in" if r == "_in" else "lives_at"
                elif " born" in t or t.startswith("born"):
                    rr = "born_in"
                elif " work" in t or t.startswith("work"):
                    rr = "works_at" if r == "_at" else "works_in"
                elif (" move" in t or " moved" in t) and r == "_from":
                    rr = "moved_from"
                elif (" participate" in t or " participated" in t) and r == "_in":
                    rr = "participated_in"
                elif (" go" in t or " went" in t) and r == "_to":
                    rr = "went_to"

            # Normalize belongs_to ownership
            if (r in {"_to", "belong_to", "belongs_to"}) and ("belong" in t):
                rr = "belongs_to"
                if cd in {"me", "you"}:
                    # Flip to (you, owns, s)
                    refined.append(("you", "owns", cs))
                    continue

            # Drop scaffolding nodes around name if canonical name fact present
            scaff_names = {"my name", "my dog's name", "name", "dog's name"}
            if (cs in scaff_names or cd in scaff_names) and rr in {"is", "has"}:
                continue

            # Drop "you has name" scaffolding
            if cs == "you" and rr == "has" and cd == "name":
                continue

            # Keep
            refined.append((cs, rr, cd))

        # Derive symmetric friend_of from patterns
        friends_pairs: List[Tuple[str, str]] = []
        names = set()
        for (a, r, b) in refined:
            if r == "and":
                friends_pairs.append((a, b))
            if r == "is" and b == "friends":
                names.add(a)
        for (a, b) in friends_pairs:
            if a in names:
                refined.append((a, "friend_of", b))
                refined.append((b, "friend_of", a))

        # Collect temporal signals (language-agnostic via UD roles)
        years: List[str] = []
        durations: List[str] = []
        try:
            for tok in (doc or []):
                # Year candidates anywhere
                if tok.like_num:
                    try:
                        val = int(tok.text)
                        if 1900 <= val <= 2100 and len(tok.text) == 4:
                            years.append(tok.text)
                    except Exception:
                        pass

                # Duration attached to verbs as oblique or nominal modifier
                if tok.dep_ in {"obl", "nmod"} and tok.head and tok.head.pos_ in {"VERB", "AUX"}:
                    # Look for numeric child modifying this token
                    num_child = None
                    for ch in tok.children:
                        if ch.dep_ == "nummod" and ch.like_num:
                            num_child = ch
                            break
                    # Or the token itself is numeric (e.g., year)
                    if num_child is not None:
                        # Use the subtree span text as duration phrase
                        left = min([t.left_edge.i for t in [tok, num_child]])
                        right = max([t.right_edge.i for t in [tok, num_child]])
                        span = tok.doc[left:right+1]
                        durations.append(_canon_entity_text(span.text))
        except Exception:
            pass

        # Derive symmetric friend_of from patterns
        friends_pairs: List[Tuple[str, str]] = []
        names = set()
        for (a, r, b) in refined:
            if r == "and":
                friends_pairs.append((a, b))
            if r == "is" and b == "friends":
                names.add(a)
        for (a, b) in friends_pairs:
            if a in names:
                refined.append((a, "friend_of", b))
                refined.append((b, "friend_of", a))

        # Attach temporal info to the most relevant event triple
        def is_event_rel(rel: str) -> bool:
            return rel not in {"has", "name", "favorite_color", "friend_of", "quality", "quantity", "is", "owns"}

        anchor: Optional[Tuple[str, str, str]] = None
        for tr in refined:
            if is_event_rel(tr[1]):
                anchor = tr
                break
        if anchor is None and refined:
            anchor = refined[0]

        if anchor is not None:
            s_anchor, r_anchor, _ = anchor
            for y in years:
                refined.append((s_anchor, "time", y))
            for dur in durations:
                refined.append((s_anchor, "duration", dur))

        # De-duplicate while preserving order
        seen = set()
        uniq: List[Tuple[str, str, str]] = []
        for tr in refined:
            if tr not in seen and all(tr):
                uniq.append(tr)
                seen.add(tr)
        return uniq

    
