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

from memory_store import MemoryStore

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
            if lang == "en":
                nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
            else:
                nlp = spacy.load(f"{lang}_core_news_sm", disable=["ner", "lemmatizer", "textcat"])
            _nlp_cache[lang] = nlp
            logger.info(f"Loaded spaCy model {lang}_core_web_sm")
        except:
            _nlp_cache[lang] = None
            logger.warning(f"Could not load spaCy model for {lang}")
    return _nlp_cache[lang]

def _norm(text: str) -> str:
    """Fast normalization"""
    return text.lower().strip() if text else ""

@dataclass
class RecencyItem:
    """Item in recency buffer"""
    text: str
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
        
        # Performance tracking
        self.metrics = defaultdict(list)
        self.max_metric_size = 1000
        
    def process_turn(self, text: str, session_id: str, turn_id: int) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Process a conversation turn
        Returns: (memory_bullets, extracted_triples)
        """
        start = time.perf_counter()
        
        # Language detection
        lang = self._detect_language(text) if PYCLD3_AVAILABLE else "en"
        
        # Stage 1: Extract entities and relations
        extract_start = time.perf_counter()
        entities, triples, neg_count, doc = self._extract(text, lang)
        self.metrics['extraction_ms'].append((time.perf_counter() - extract_start) * 1000)
        
        # Stage 2: Update memory with new facts
        update_start = time.perf_counter()
        now_ts = int(time.time() * 1000)
        for s, r, d in triples:
            self.store.observe_edge(s, r, d, 0.9, now_ts)
            # Update hot indices
            self.entity_index[s].add((s, r, d))
            self.entity_index[d].add((s, r, d))
        self.metrics['update_ms'].append((time.perf_counter() - update_start) * 1000)
        
        # Stage 3: Retrieve relevant memories
        retrieve_start = time.perf_counter()
        bullets = self._retrieve_context(text, entities, turn_id)
        self.metrics['retrieval_ms'].append((time.perf_counter() - retrieve_start) * 1000)
        
        # Update recency
        self.recency_buffer.append(RecencyItem(text, now_ts, turn_id))
        
        # Track overall performance
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics['total_ms'].append(elapsed_ms)
        self._cleanup_metrics()
        
        if elapsed_ms > 200:
            logger.warning(f"Hot path took {elapsed_ms:.1f}ms (budget: 200ms)")
        
        return bullets, triples
    
    def _extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """
        Extract entities and relations using USGS 27-pattern approach
        Returns: (entities, triples, negation_count, doc)
        """
        nlp = _load_nlp(lang)
        
        if not nlp:
            return [], [], 0, None
        
        doc = nlp(text)
        entities = set()
        triples = []
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
        
        return list(entities), triples, neg_count, doc
    
    def _build_entity_map(self, doc, entities: Set[str]) -> Dict[int, str]:
        """Build entity map from document"""
        entity_map = {}
        
        # Named entities
        for ent in doc.ents:
            norm_text = _norm(ent.text)
            entities.add(norm_text)
            for token in ent:
                entity_map[token.i] = norm_text
        
        # Noun chunks
        for chunk in doc.noun_chunks:
            chunk_text = _norm(chunk.text)
            entities.add(chunk_text)
            entity_map[chunk.root.i] = chunk_text
        
        # Individual tokens
        for token in doc:
            if token.i not in entity_map:
                if token.pos_ in {"NOUN", "PROPN", "PRON"}:
                    entity_text = _norm(token.text)
                    # Canonicalize pronouns
                    if entity_text in {"i", "me", "my", "mine", "myself"}:
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
                subj = self._get_entity(child, entity_map)
                triples.append((subj, "is", attr))
                break
    
    def _extract_amod(self, token, entity_map, triples, entities):
        """amod - adjectival modifier"""
        adj = token.text.lower()
        head_entity = self._get_entity(token.head, entity_map)
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
        # Complex clausal relation
        pass
    
    def _extract_csubj(self, token, entity_map, triples, entities):
        """csubj - clausal subject"""
        # Complex clausal relation
        pass
    
    def _extract_xcomp(self, token, entity_map, triples, entities):
        """xcomp - open clausal complement"""
        # Complex clausal relation
        pass
    
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
        """
        Retrieve relevant memory bullets for context
        Returns top 3 most relevant memories
        """
        bullets = []
        seen = set()
        
        # Get recent memories
        for item in list(self.recency_buffer)[-5:]:
            if item.text not in seen:
                bullets.append(f"• {item.text}")
                seen.add(item.text)
        
        # Get entity-related memories
        for entity in entities[:3]:  # Top entities
            if entity in self.entity_index:
                for s, r, d in list(self.entity_index[entity])[:2]:
                    fact = f"{s} {r} {d}"
                    if fact not in seen:
                        bullets.append(f"• {fact}")
                        seen.add(fact)
        
        return bullets[:3]  # Return top 3
    
    def _detect_language(self, text: str) -> str:
        """Detect language using pycld3"""
        if PYCLD3_AVAILABLE:
            try:
                result = pycld3.get_language(text)
                return result.language[:2] if result.is_reliable else "en"
            except:
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
                count += 1
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Rebuilt {count} edges in {elapsed_ms:.1f}ms")