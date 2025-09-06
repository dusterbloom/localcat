"""
HotMem: Ultra-fast local memory for voice agents
Full USGS Grammar-to-Graph 27 dependency pattern implementation
Target: <200ms p95 extraction + retrieval
"""

import os
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict, deque
import math
import heapq
from dataclasses import dataclass
import statistics

from loguru import logger
import spacy
from spacy.tokens import Token

from memory_store import MemoryStore
from memory_intent import get_intent_classifier, get_quality_filter, IntentType
try:
    from memory_decomposer import decompose as _decompose_clauses  # type: ignore
except Exception:
    _decompose_clauses = None  # Optional
try:
    from memory_quality import calculate_extraction_confidence as _extra_confidence  # type: ignore
except Exception:
    _extra_confidence = None  # Optional

# Try to import language detection
try:
    import pycld3
    PYCLD3_AVAILABLE = True
except ImportError:
    PYCLD3_AVAILABLE = False
    logger.info("pycld3 not available, defaulting to English")

# Singleton NLP model cache
_nlp_cache = {}

def _load_nlp(lang: str = "en"):
    """Load spaCy model (cached singleton)"""
    if lang not in _nlp_cache:
        try:
            # Keep lemmatizer+parser across all supported langs; disable only NER/textcat for speed
            if lang == "en":
                nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
            else:
                nlp = spacy.load(f"{lang}_core_news_sm", disable=["ner", "textcat"])
            _nlp_cache[lang] = nlp
            logger.info(f"Loaded spaCy model {lang}_core_web_sm")
        except:
            _nlp_cache[lang] = None
            logger.warning(f"Could not load spaCy model for {lang}")
    return _nlp_cache[lang]

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
    
    def __init__(self, store: MemoryStore, max_recency: int = 50):
        self.store = store
        self.user_eid = "you"
        
        # Hot indices (RAM)
        self.entity_index = defaultdict(set)  # entity -> set of (s,r,d) triples
        self.recency_buffer = deque(maxlen=max_recency)  # Recent interactions
        self.entity_cache = {}  # Canonical entity mapping
        self.edge_meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}  # (s,r,d) -> {ts, weight}
        # Optional LEANN semantic search
        self.use_leann = os.getenv("HOTMEM_USE_LEANN", "false").lower() in ("1", "true", "yes")
        self.leann_index_path = os.getenv("LEANN_INDEX_PATH", os.path.join(os.path.dirname(__file__), '..', 'data', 'memory_vectors.leann'))
        if self.leann_index_path and not os.path.isabs(self.leann_index_path):
            self.leann_index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.leann_index_path))
        self.leann_complexity = int(os.getenv("HOTMEM_LEANN_COMPLEXITY", "16"))
        self._leann_searcher = None
        self._leann_loaded_mtime = 0.0
        
        # Performance tracking
        self.metrics = defaultdict(list)
        self.max_metric_size = 1000

    def prewarm(self, lang: str = "en") -> None:
        """Load NLP resources up-front to avoid first-turn latency."""
        try:
            _load_nlp(lang)
        except Exception:
            pass

    def process_turn(self, text: str, session_id: str, turn_id: int) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Process a conversation turn with intelligent intent analysis
        Returns: (memory_bullets, extracted_triples)
        """
        start = time.perf_counter()
        
        # Language detection first (needed for intent analysis)
        lang = self._detect_language(text) if PYCLD3_AVAILABLE else "en"
        
        # Stage 0: Intent classification for quality guidance
        intent_start = time.perf_counter()
        intent_classifier = get_intent_classifier()
        quality_filter = get_quality_filter()
        intent = intent_classifier.analyze(text, lang)
        self.metrics['intent_ms'].append((time.perf_counter() - intent_start) * 1000)
        
        # Early exit for reactions and pure questions (no fact extraction)
        if intent.intent in {IntentType.REACTION, IntentType.PURE_QUESTION}:
            logger.debug(f"Skipping extraction for {intent.intent.value}: {text[:50]}...")
            # Still retrieve context for responses
            retrieve_start = time.perf_counter()
            entities = self._extract_entities_light(text)
            bullets = self._retrieve_context(text, entities, turn_id, intent=intent)
            self.metrics['retrieval_ms'].append((time.perf_counter() - retrieve_start) * 1000)
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics['total_ms'].append(elapsed_ms)
            return bullets, []
        
        # Language already detected above
        
        # Stage 1: Extract entities and relations
        extract_start = time.perf_counter()
        entities, triples, neg_count, doc = self._extract(text, lang)
        self.metrics['extraction_ms'].append((time.perf_counter() - extract_start) * 1000)
        
        # Stage 2: Refine triples with intent-aware processing
        refine_start = time.perf_counter()
        triples = self._refine_triples(text, triples, doc, intent, lang)
        # Apply light coreference resolution for third-person pronouns
        try:
            triples = self._apply_coref_lite(triples, doc)
        except Exception:
            pass
        # Rebuild entities from refined triples + text context
        ent_from_triples: Set[str] = set()
        for s, r, d in triples:
            ent_from_triples.add(s)
            ent_from_triples.add(d)
        entities = self._refine_entities_from_text(text, list(ent_from_triples))

        # Stage 3: Quality filtering and storage
        update_start = time.perf_counter()
        now_ts = int(time.time() * 1000)
        
        # Filter and store facts based on quality and intent
        stored_triples = []
        # Hedge/negation adjustments
        t_lower = (text or "").lower()
        hedge_terms = ("maybe", "i think", "probably", "kinda", "sort of", "not sure", "perhaps", "possibly")
        hedged = any(ht in t_lower for ht in hedge_terms)

        conf_thresh = float(os.getenv("HOTMEM_CONFIDENCE_THRESHOLD", "0.3"))
        use_extra_conf = os.getenv("HOTMEM_EXTRA_CONFIDENCE", "false").lower() in ("1", "true", "yes")

        for s, r, d in triples:
            should_store, confidence = quality_filter.should_store_fact(s, r, d, intent)
            # Optional conservative complexity confidence
            if use_extra_conf and _extra_confidence is not None:
                try:
                    confidence = min(confidence, float(_extra_confidence(doc, (s, r, d))))
                except Exception:
                    pass
            # Adjust confidence for hedging/negation context
            if hedged:
                confidence -= 0.2
            if neg_count > 0 and intent.intent != IntentType.CORRECTION:
                confidence -= 0.2
            confidence = max(0.0, min(1.0, confidence))
            if confidence < conf_thresh:
                should_store = False
            if should_store:
                # Handle corrections by demoting old facts
                if intent.intent == IntentType.CORRECTION:
                    self._handle_fact_correction(s, r, d, confidence, now_ts)
                else:
                    self.store.observe_edge(s, r, d, confidence, now_ts)
                    
                # Update hot indices
                self.entity_index[s].add((s, r, d))
                self.entity_index[d].add((s, r, d))
                stored_triples.append((s, r, d))
            else:
                logger.debug(f"Filtered low-quality fact: ({s}, {r}, {d}) confidence={confidence:.2f}")
                
        self.metrics['update_ms'].append((time.perf_counter() - update_start) * 1000)
        
        # Stage 3: Retrieve relevant memories
        retrieve_start = time.perf_counter()
        bullets = self._retrieve_context(text, entities, turn_id, intent=intent)
        self.metrics['retrieval_ms'].append((time.perf_counter() - retrieve_start) * 1000)
        
        # Update recency with stored triples only
        for s, r, d in stored_triples:
            self.recency_buffer.append(RecencyItem(s, r, d, text, now_ts, turn_id))
            # Update hot edge metadata for recency/weight (weight approx if unknown)
            self.edge_meta[(s, r, d)] = {
                'ts': now_ts,
                'weight': max(0.3, 0.7)  # optimistic default
            }
        
        # Track overall performance
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics['total_ms'].append(elapsed_ms)
        self._cleanup_metrics()
        
        if elapsed_ms > 200:
            logger.warning(f"Hot path took {elapsed_ms:.1f}ms (budget: 200ms) - intent: {intent.intent.value}")
        
        return bullets, stored_triples
    
    def _extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """
        Extract entities and relations using USGS 27-pattern approach.
        Optionally decompose complex sentences into clause spans when enabled.
        Returns: (entities, triples, negation_count, original_doc)
        """
        nlp = _load_nlp(lang)
        if not nlp:
            return [], [], 0, None

        doc = nlp(text)

        use_decomp = os.getenv("HOTMEM_DECOMPOSE_CLAUSES", "false").lower() in ("1", "true", "yes")
        if use_decomp and _decompose_clauses is not None:
            try:
                spans = _decompose_clauses(doc)
            except Exception:
                spans = []
            if spans:
                ents_all: Set[str] = set()
                trips_all: List[Tuple[str, str, str]] = []
                neg_total = 0
                for sp in spans:
                    try:
                        subdoc = sp.as_doc()
                    except Exception:
                        subdoc = doc
                    ents, trips, negc = self._extract_from_doc(subdoc)
                    ents_all.update(ents)
                    trips_all.extend(trips)
                    neg_total += negc
                return list(ents_all), trips_all, neg_total, doc

        ents, trips, negc = self._extract_from_doc(doc)
        return ents, trips, negc, doc

    def _extract_from_doc(self, doc) -> Tuple[List[str], List[Tuple[str, str, str]], int]:
        """Internal: 27-pattern extraction over a spaCy Doc."""
        entities = set()
        triples: List[Tuple[str, str, str]] = []
        neg_count = 0

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

        return list(entities), triples, neg_count
    
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
                    obj = self._get_entity(child, entity_map)
                    # Special: "My name is X"
                    if token.text.lower() == "name":
                        for gc in token.children:
                            if gc.dep_ == "poss" and gc.text.lower() in {"my", "mine"}:
                                triples.append((self.user_eid, "name", obj))
                                entities.add(obj)
                                return
                    triples.append((subj, "is", obj))
                    entities.add(obj)
        
        # Active verb: X verbs Y
        elif head.pos_ == "VERB":
            verb = head.lemma_.lower()
            
            # Direct object
            for child in head.children:
                if child.dep_ in {"dobj", "obj"}:
                    obj = self._get_entity(child, entity_map)
                    pred = "has" if verb in {"have", "has", "had", "own"} else verb
                    triples.append((subj, pred, obj))
                    entities.add(obj)
            
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
                        obj = self._get_entity(ch, entity_map)
                        pred = "has" if verb2 in {"have", "has", "had", "own"} else verb2
                        triples.append((subj2, pred, obj))
                        entities.add(obj)

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
        obj = self._get_entity(token, entity_map)
        head = token.head
        
        if head.pos_ == "VERB":
            for child in head.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subj = self._get_entity(child, entity_map)
                    verb = head.lemma_.lower()
                    pred = "has" if verb in {"have", "has", "had"} else verb
                    triples.append((subj, pred, obj))
                    break
    
    def _extract_indirect_object(self, token, entity_map, triples, entities):
        """iobj - indirect object"""
        iobj = self._get_entity(token, entity_map)
        head = token.head
        
        # Find subject
        for child in head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj = self._get_entity(child, entity_map)
                triples.append((subj, f"gave_to", iobj))
                break
    
    def _extract_attribute(self, token, entity_map, triples, entities):
        """attr - attribute (copula complement)"""
        attr = self._get_entity(token, entity_map)
        
        for child in token.head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj_token = child
                subj = self._get_entity(child, entity_map)
                
                # Special handling for possessive copula: "My X is Y" -> (you, X, Y)
                possessive_owner = None
                for poss_child in subj_token.children:
                    if poss_child.dep_ == "poss" and poss_child.text.lower() in {"my", "mine"}:
                        possessive_owner = "you"
                        break
                
                if possessive_owner:
                    # Convert "favorite number" to "favorite_number" relation
                    relation = subj.lower().replace(" ", "_")
                    triples.append((possessive_owner, relation, attr))
                    entities.add(attr)
                    # Skip the normal subject-is-attribute extraction since we handled it
                    return
                else:
                    triples.append((subj, "is", attr))
                break
    
    def _extract_acomp(self, token, entity_map, triples, entities):
        """acomp - adjectival complement (copula complement)"""
        # Handle patterns like "Caroline is single"
        adj = self._get_entity(token, entity_map)
        head = token.head
        
        # Find subject of copula
        for child in head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj = self._get_entity(child, entity_map)
                triples.append((subj, "is", adj))
                entities.add(adj)
                break
    
    def _extract_amod(self, token, entity_map, triples, entities):
        """amod - adjectival modifier"""
        adj = token.text.lower()
        head_entity = self._get_entity(token.head, entity_map)
        
        # Skip quality extraction for possessive copula patterns ("My favorite X is Y")
        # Check if the head is subject of a copula and has possessive
        head_token = token.head
        if head_token.dep_ == "nsubj" and head_token.head.lemma_ in {"be", "is", "am", "are", "was", "were"}:
            # Check if head has possessive child and copula has attribute
            has_possessive = any(child.dep_ == "poss" and child.text.lower() in {"my", "mine"} 
                               for child in head_token.children)
            has_attribute = any(child.dep_ in {"attr", "acomp"} 
                              for child in head_token.head.children)
            if has_possessive and has_attribute:
                return  # Skip meaningless quality extraction
        
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
        
        # Skip possessive extraction if this is part of a copula pattern ("My X is Y")
        # Check if the possessed noun is the subject of a copula with an attribute
        head_token = token.head
        if head_token.dep_ == "nsubj" and head_token.head.lemma_ in {"be", "is", "am", "are", "was", "were"}:
            # Check if there's an attribute - if so, skip this possessive extraction
            for sibling in head_token.head.children:
                if sibling.dep_ in {"attr", "acomp"}:
                    return  # Skip extraction, will be handled by attribute extraction
        
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
                obj = self._get_entity(token, entity_map)
                verb = head.lemma_.lower()
                triples.append((subj, verb, obj))
                entities.add(obj)
    
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
    
    def _format_memory_bullet(self, s: str, r: str, d: str) -> str:
        """Format a memory triple into a human-readable bullet point"""
        # Handle pronouns and possessives more naturally
        if s == "you":
            if r == "name":
                return f"• Your name is {d}"
            elif r == "age":
                # Avoid duplicating "years old" if already in destination
                if "years old" in d.lower():
                    return f"• You are {d}"
                else:
                    return f"• You are {d} years old"
            elif r == "favorite_number":
                return f"• Your favorite number is {d}"
            elif r == "has":
                return f"• You have {d}"
            elif r == "is":
                return f"• You are {d}"
            elif r == "lives_in":
                return f"• You live in {d}"
            elif r == "works_at":
                return f"• You work at {d}"
            elif r == "born_in":
                return f"• You were born in {d}"
            elif r == "friend_of":
                return f"• You are a friend of {d}"
            elif r.startswith("v:"):
                verb = r[2:]
                # Fix verb conjugation for "you"
                if verb.endswith("s") and len(verb) > 2:
                    verb = verb[:-1]  # "enjoys" -> "enjoy"
                return f"• You {verb} {d}"
            else:
                relation = r.replace('_', ' ')
                if relation in ["has", "is"]:
                    return f"• You {relation} {d}"
                else:
                    return f"• Your {relation} is {d}"
        else:
            if r == "name":
                return f"• {s.title()}'s name is {d}"
            elif r == "age":
                # Avoid duplicating "years old" if already in destination
                if "years old" in d.lower():
                    return f"• {s.title()} is {d}"
                else:
                    return f"• {s.title()} is {d} years old"
            elif r == "favorite_number":
                return f"• {s.title()}'s favorite number is {d}"
            elif r == "has":
                return f"• {s.title()} has {d}"
            elif r == "is":
                return f"• {s.title()} is {d}"
            elif r == "lives_in":
                return f"• {s.title()} lives in {d}"
            elif r == "works_at":
                return f"• {s.title()} works at {d}"
            elif r == "born_in":
                return f"• {s.title()} was born in {d}"
            elif r == "friend_of":
                return f"• {s.title()} is a friend of {d}"
            elif r.startswith("v:"):
                verb = r[2:]
                return f"• {s.title()} {verb} {d}"
            elif "_color" in r:
                # Handle color relations more naturally
                color_type = r.replace("_color", "")
                return f"• {s.title()}'s {color_type} color is {d}"
            else:
                # Generic fallback with better formatting
                relation = r.replace('_', ' ')
                if relation in ["has", "is"]:
                    return f"• {s.title()} {relation} {d}"
                else:
                    return f"• {s.title()}'s {relation} is {d}"
    
    def _retrieve_context(self, query: str, entities: List[str], turn_id: int, intent=None) -> List[str]:
        """
        Retrieve relevant memory bullets for context
        Returns up to 5 most relevant memories based on scoring
        """
        bullets: List[str] = []
        seen_triples: Set[Tuple[str, str, str]] = set()

        # Tokenize query for lexical overlap (simple, fast)
        def _tokens(t: str) -> Set[str]:
            out = set()
            for w in (t or "").lower().split():
                w = ''.join(ch for ch in w if ch.isalnum())
                if len(w) >= 3:
                    out.add(w)
            return out
        qtok = _tokens(query)

        # Optional: LEANN similarity scores map for this query
        leann_scores: Dict[Tuple[str, str, str], float] = {}
        if self.use_leann and query and len(query.strip()) >= 2:
            try:
                leann_scores = self._leann_query_scores(query, top_k=64)
            except Exception:
                leann_scores = {}

        # 1) Prefer fact bullets based on query entities
        #    Put non-'you' entities first; then 'you' if present.
        ent_set = [e for e in entities if e]
        non_you = [e for e in ent_set if e != "you"]
        include_you = any(e == "you" for e in ent_set)
        query_entities = non_you[:4]
        if include_you:
            query_entities.append("you")

        # Gating: avoid injecting memory for generic questions that only mention "you/I"
        # unless the query contains attribute keywords or an explicit remember request.
        t = (query or "").lower()
        attr_keywords = (
            "name", "age", "live", "lives", "hometown", "from",
            "work", "works", "job", "company", "employer",
            "favorite", "colour", "color", "pet", "dog", "cat",
            "spouse", "partner", "wife", "husband", "child", "kid", "kids"
        )
        remember_request = ("remember" in t) or ("save this" in t) or ("note this" in t)
        has_attr = any(kw in t for kw in attr_keywords)
        if include_you and not non_you:
            # you-only reference
            if not has_attr and not remember_request:
                return []

        # Predicate priority (normalized 0..1)
        pred_pri = {
            "lives_in": 1.00,
            "works_at": 0.95,
            "born_in": 0.90,
            "moved_from": 0.85,
            "participated_in": 0.80,
            "friend_of": 0.78,
            "name": 0.75,
            "favorite_color": 0.70,
            "favorite_number": 0.70,
            "has": 0.60,
            "is": 0.55,
        }

        # Weights (env tunable later)
        alpha, beta, gamma, delta = 0.4, 0.2, 0.3, 0.1
        recency_T_ms = 3 * 24 * 3600 * 1000  # 3 days

        # Intent-aware K limit
        K_max = 5
        try:
            from memory_intent import IntentType  # type: ignore
            if intent and getattr(intent, 'intent', None) in {IntentType.REACTION, IntentType.PURE_QUESTION}:
                K_max = 2
        except Exception:
            pass

        scored_all: List[Tuple[float, int, str, str, str]] = []
        now_ms = int(time.time() * 1000)
        # Only inject from a safe canonical relation set; exclude question scaffolding/noise
        allowed_rels = {
            "name", "age", "favorite_color",
            "lives_in", "works_at", "born_in", "moved_from",
            "participated_in", "went_to",
            "friend_of", "owns", "has", "favorite_number",
        }
        blocked_tokens = {"system prompt", "what time", "no-"}

        for entity in query_entities:
            if entity in self.entity_index:
                candidates = list(self.entity_index[entity])
                for s, r, d in candidates:
                    # Filter out low-value relations and question scaffolding for injection
                    if r not in allowed_rels:
                        continue
                    sd_text = f"{s} {d}".lower()
                    if any(bt in sd_text for bt in blocked_tokens):
                        continue
                    meta = self.edge_meta.get((s, r, d), {})
                    ts = int(meta.get('ts', 0))
                    age = max(0, now_ms - ts)
                    rec = math.exp(-age / max(1, recency_T_ms)) if ts > 0 else 0.0
                    pri = pred_pri.get(r, 0.5)
                    # Similarity (LEANN or lexical)
                    sem = 0.0
                    if leann_scores:
                        sem = float(leann_scores.get((s, r, d), 0.0))
                    else:
                        stok = _tokens(s) | _tokens(r) | _tokens(d)
                        if qtok and stok:
                            inter = len(qtok & stok)
                            union = len(qtok | stok)
                            sem = inter / union if union else 0.0
                    w = float(meta.get('weight', 0.3))
                    score = alpha * pri + beta * rec + gamma * sem + delta * w
                    scored_all.append((score, ts, s, r, d))

        # Threshold (τ) and diversity
        if scored_all:
            scored_all.sort(key=lambda x: (x[0], x[1]), reverse=True)
            scores_only = [s for (s, _ts, _s, _r, _d) in scored_all]
            idx = max(0, int(len(scores_only) * 0.75) - 1)
            tau = scores_only[idx]
            eps = 0.05
            by_rel: Dict[str, int] = defaultdict(int)
            for score, _ts, s, r, d in scored_all:
                if (s, r, d) in seen_triples:
                    continue
                # diversity: allow 1 per relation, or 2 if high-scoring
                if by_rel[r] >= 1 and not (by_rel[r] < 2 and score >= (tau + eps)):
                    continue
                if score < tau and len(bullets) >= max(1, K_max // 2):
                    continue
                bullets.append(self._format_memory_bullet(s, r, d))
                seen_triples.add((s, r, d))
                by_rel[r] += 1
                if len(bullets) >= K_max:
                    return bullets

        # No recency fallback: only inject facts relevant to current turn
        return bullets[:K_max]

    # ---- LEANN helpers ----
    def _ensure_leann_searcher(self):
        if not self.use_leann:
            return None
        try:
            import os as _os
            mtime = _os.path.getmtime(self.leann_index_path)
        except Exception:
            return None
        try:
            if (self._leann_searcher is None) or (mtime > self._leann_loaded_mtime):
                from leann.api import LeannSearcher  # type: ignore
                self._leann_searcher = LeannSearcher(self.leann_index_path)
                self._leann_loaded_mtime = mtime
        except Exception:
            self._leann_searcher = None
        return self._leann_searcher

    def _leann_query_scores(self, query: str, top_k: int = 32) -> Dict[Tuple[str, str, str], float]:
        scores: Dict[Tuple[str, str, str], float] = {}
        searcher = self._ensure_leann_searcher()
        if not searcher:
            return scores
        try:
            results = searcher.search(query, top_k=top_k, complexity=self.leann_complexity)
        except Exception:
            return scores
        # Map by metadata if available; else fallback to rank-based score
        for rank, res in enumerate(results or []):
            try:
                meta = getattr(res, 'metadata', None) or {}
                s = meta.get('src'); r = meta.get('rel'); d = meta.get('dst')
                if s and r and d:
                    sc = getattr(res, 'score', None)
                    if sc is None:
                        sc = getattr(res, 'similarity', None)
                    if sc is None:
                        sc = 1.0 - (rank / float(top_k + 1))
                    scores[(str(s), str(r), str(d))] = float(sc)
            except Exception:
                continue
        return scores
    
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
    
    def _cleanup_metrics(self):
        """Keep metrics bounded"""
        for key in self.metrics:
            if len(self.metrics[key]) > self.max_metric_size:
                self.metrics[key] = self.metrics[key][-self.max_metric_size:]
    
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
                # Seed meta with weight; ts unknown (0)
                self.edge_meta[(s, r, d)] = {'ts': 0, 'weight': float(conf)}
                count += 1
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Rebuilt {count} edges in {elapsed_ms:.1f}ms")

    # ---------- Non-mutating helpers for verification ----------
    def preview_bullets(self, text: str, lang: str = "en") -> Dict[str, Any]:
        """Return entities detected and bullets that would be injected, without updating store.

        Useful for validating retrieval independently of writes.
        """
        try:
            entities, _, _, _ = self._extract(text, lang)
            entities = self._refine_entities_from_text(text, entities)
        except Exception:
            entities = []
        bullets = self._retrieve_context(text, entities, turn_id=-1, intent=None)
        return {"entities": entities, "bullets": bullets}

    # ---------- Refinement helpers (quality without large perf cost) ----------
    def _is_question(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        if "?" in t:
            return True
        wh = ("who", "what", "when", "where", "why", "how")
        return any(t.startswith(w + " ") for w in wh)

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

    def _refine_triples(self, text: str, triples: List[Tuple[str, str, str]], doc, intent=None, lang: str = "en") -> List[Tuple[str, str, str]]:
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

        # 5) Favorite number is X → favorite_number
        fn = None
        try:
            fn = __import__("re").search(r"\bfavorite number is\s+([^,.!?]+)", t)
        except Exception:
            fn = None
        if fn:
            favn = _canon_entity_text(fn.group(1))
            refined.append(("you", "favorite_number", favn))

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

                # Multilingual normalization (simple keyword checks)
                l = (lang or "en").lower()
                try:
                    if l == 'es':
                        if (" vive" in t or t.startswith("vive")) and r == "_in":
                            rr = "lives_in"
                        elif (" trabaja" in t or t.startswith("trabaja")) and r in {"_at", "_in"}:
                            rr = "works_at"
                        elif (" nació" in t or t.startswith("nació")) and r == "_in":
                            rr = "born_in"
                        elif (" mudó" in t or " se mudó" in t) and r == "_from":
                            rr = "moved_from"
                        elif (" particip" in t) and r == "_in":
                            rr = "participated_in"
                        elif (" fue" in t or " ir " in t or t.startswith("fue") or t.startswith("ir ")) and r == "_to":
                            rr = "went_to"
                    elif l == 'fr':
                        if (" habite" in t or t.startswith("habite")) and r in {"_in", "_at"}:
                            rr = "lives_in"
                        elif (" travaille" in t or t.startswith("travaille")) and r in {"_at", "_in"}:
                            rr = "works_at"
                        elif (" né" in t or " née" in t) and r == "_in":
                            rr = "born_in"
                        elif (" déménagé" in t) and r == "_from":
                            rr = "moved_from"
                        elif (" participé" in t) and r == "_in":
                            rr = "participated_in"
                        elif (" allé" in t) and r == "_to":
                            rr = "went_to"
                    elif l == 'de':
                        if (" wohnt" in t or t.startswith("wohnt")) and r == "_in":
                            rr = "lives_in"
                        elif (" arbeitet" in t or t.startswith("arbeitet")) and r in {"_at", "_in"}:
                            rr = "works_at"
                        elif (" geboren" in t) and r == "_in":
                            rr = "born_in"
                        elif (" zog" in t or " umgezogen" in t) and r == "_from":
                            rr = "moved_from"
                        elif (" teilgenommen" in t) and r == "_in":
                            rr = "participated_in"
                        elif (" ging" in t or " gegangen" in t) and r == "_to":
                            rr = "went_to"
                    elif l == 'it':
                        if (" vive" in t or t.startswith("vive")) and r == "_in":
                            rr = "lives_in"
                        elif (" lavora" in t or t.startswith("lavora")) and r in {"_at", "_in"}:
                            rr = "works_at"
                        elif (" nato" in t or " nata" in t) and r == "_in":
                            rr = "born_in"
                        elif (" trasferit" in t) and r == "_from":
                            rr = "moved_from"
                        elif (" partecipat" in t) and r == "_in":
                            rr = "participated_in"
                        elif (" andat" in t) and r == "_to":
                            rr = "went_to"
                except Exception:
                    pass

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

    # --- Coref-lite: resolve third-person pronouns to recent mentions ---
    def _build_mention_stack(self, doc) -> List[str]:
        stack: List[str] = []
        try:
            # Noun chunks and proper nouns in order of appearance
            if doc is not None:
                for chunk in getattr(doc, 'noun_chunks', []):
                    txt = _canon_entity_text(chunk.text)
                    if txt and txt not in stack:
                        stack.append(txt)
                for tok in doc:
                    if tok.pos_ == 'PROPN':
                        txt = _canon_entity_text(tok.text)
                        if txt and txt not in stack:
                            stack.append(txt)
        except Exception:
            pass
        # Add recent entities from recency_buffer
        try:
            for item in list(self.recency_buffer)[-5:]:
                for ent in (item.s, item.d):
                    if ent and ent not in stack and ent != 'you':
                        stack.append(ent)
        except Exception:
            pass
        return stack

    def _resolve_pronoun(self, token_text: str, stack: List[str]) -> Optional[str]:
        p = (token_text or '').strip().lower()
        if not p:
            return None
        third = {'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs', 'it', 'this', 'that'}
        if p in third:
            # Return most recent stack entry not 'you'
            for cand in reversed(stack):
                if cand != 'you':
                    return cand
        return None

    def _apply_coref_lite(self, triples: List[Tuple[str, str, str]], doc) -> List[Tuple[str, str, str]]:
        stack = self._build_mention_stack(doc)
        out: List[Tuple[str, str, str]] = []
        for s, r, d in triples:
            rs = s
            rd = d
            if s not in {'you'}:
                rs2 = self._resolve_pronoun(s, stack)
                if rs2:
                    rs = rs2
            if d not in {'you'}:
                rd2 = self._resolve_pronoun(d, stack)
                if rd2:
                    rd = rd2
            out.append((rs, r, rd))
        return out

    def _extract_entities_light(self, text: str) -> List[str]:
        """Lightweight entity extraction for reactions/questions (no full NLP)"""
        # Simple pattern-based extraction for performance
        entities = []
        words = text.split()
        
        # Look for capitalized words (likely names/places)
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                clean_word = _canon_entity_text(word)
                if clean_word and clean_word not in entities:
                    entities.append(clean_word)
        
        # Also look for lowercase words that exist in our entity index
        # This catches common nouns like "dog", "cat", "car" that are stored entities
        for word in words:
            clean_word = _canon_entity_text(word)
            if clean_word and len(clean_word) > 2 and clean_word not in entities:
                # Check if this entity exists in our knowledge base
                if clean_word in self.entity_index:
                    entities.append(clean_word)
        
        # Add 'you' if self-referential
        if any(p in text.lower() for p in [" i ", " my ", " me "]):
            entities.append("you")
            
        return entities
        
    def _handle_fact_correction(self, s: str, r: str, d: str, confidence: float, now_ts: int):
        """Handle fact corrections by demoting conflicting facts"""
        # For functional relations like 'name' or 'age', demote old values
        functional_relations = {"name", "age", "favorite_color", "lives_in", "works_at"}
        
        if r in functional_relations:
            # Find existing facts with same subject and relation
            existing_edges = self.entity_index.get(s, set())
            for existing_s, existing_r, existing_d in existing_edges.copy():
                if existing_s == s and existing_r == r and existing_d != d:
                    # Demote the old fact
                    self.store.negate_edge(s, r, existing_d, confidence, now_ts)
                    logger.debug(f"Corrected fact: ({s}, {r}) {existing_d} -> {d}")
                    # Remove from hot index
                    self.entity_index[s].discard((existing_s, existing_r, existing_d))
                    self.entity_index[existing_d].discard((existing_s, existing_r, existing_d))
        
        # Store the new corrected fact
        self.store.observe_edge(s, r, d, confidence, now_ts)

    
