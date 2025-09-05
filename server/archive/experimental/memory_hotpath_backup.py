"""
LocalCat: Hot-path memory — UD extractor + compiled RAM retriever (no LLMs)
Multilingual, list-free (uses Universal Dependencies structures)
"""

import os
import time
import re
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set, Any
from loguru import logger

# Optional language detection
try:
    import pycld3
    HAS_PYCLD3 = True
except ImportError:
    HAS_PYCLD3 = False
    logger.info("pycld3 not available, defaulting to English")

# Cache for spaCy models
SPACY_DISABLE = os.getenv("SPACY_DISABLE_MODELS", "false").lower() == "true"
_nlp_cache: Dict[str, object] = {}


def _detect_lang(text: str) -> str:
    """Detect language of text"""
    if not text:
        return "en"
    
    if HAS_PYCLD3:
        try:
            r = pycld3.get_language(text[:512])
            return r.language if r and r.is_reliable else "en"
        except:
            return "en"
    return "en"


def _load_nlp(lang: str):
    """Lazy load spaCy model for language"""
    if SPACY_DISABLE:
        return None
    
    if lang in _nlp_cache:
        return _nlp_cache[lang]
    
    try:
        import spacy
        
        # Map language codes to model names
        model_map = {
            "en": "en_core_web_sm",
            "it": "it_core_news_sm", 
            "es": "es_core_news_sm",
            "fr": "fr_core_news_sm",
            "de": "de_core_news_sm",
        }
        
        model_name = model_map.get(lang, "en_core_web_sm")
        
        try:
            # Try to load the model
            nlp = spacy.load(model_name, exclude=["textcat", "lemmatizer"])
            logger.info(f"Loaded spaCy model {model_name}")
        except OSError:
            # Model not installed, create blank pipeline
            logger.warning(f"spaCy model {model_name} not found, using blank pipeline")
            nlp = spacy.blank(lang if lang in {"en", "it", "es", "fr", "de"} else "en")
        
        # Add sentencizer if not present
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        
        _nlp_cache[lang] = nlp
        return nlp
        
    except ImportError:
        logger.warning("spaCy not available, extraction will use regex fallback")
        return None


def _norm(s: str) -> str:
    """Normalize string for matching"""
    return re.sub(r"\s+", " ", s.strip().lower())


@dataclass
class Factlet:
    """A single fact/memory with metadata"""
    s: str  # subject
    r: str  # relation
    d: str  # object (destination)
    text: str  # original text
    ts: float  # timestamp
    sid: str  # session_id
    tid: int  # turn_id


class HotMemory:
    """
    Ultra-fast memory with:
    - UD-based extraction (copula/apposition/SVO/obl) → triples
    - Compiled RAM indices: adjacency, recency
    - Retrieval: adjacency first, then BM25
    - Injection: ≤3 bullets from recent factlets
    """
    
    def __init__(self, store, ring_size: int = 200, max_entities: int = 10000):
        self.store = store
        self.recency: deque = deque(maxlen=ring_size)
        
        # Bounded adjacency map to prevent memory leaks
        self.adj_out: Dict[str, Dict[str, List[Tuple[str, float, int]]]] = defaultdict(lambda: defaultdict(list))
        self.entity_access: Dict[str, float] = {}  # Track last access for cleanup
        self.max_entities = max_entities
        
        # Canonical user/speaker ID
        self.user_eid = "you"
        
        # Entity canonicalization cache
        self.entity_cache: Dict[str, str] = {}
        
        # Performance tracking
        self.metrics = defaultdict(list)
    
    # ---------- Extraction ----------
    def process_turn(self, text: str, session_id: str, turn_id: int) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Main entry point: extract facts and retrieve relevant memories
        Returns: (bullets, triples)
        """
        start = time.perf_counter()
        
        # Language detection
        lang = _detect_lang(text)
        
        # Extract entities and relations
        with self._time_operation("extraction"):
            entities, triples, neg_count, doc = self._extract(text, lang)
        
        # Update memory
        with self._time_operation("update"):
            self._update_memory(entities, triples, neg_count, text, session_id, turn_id)
        
        # Retrieve relevant facts
        with self._time_operation("retrieval"):
            bullets = self._retrieve_and_format(entities, k=3)
        
        # Track overall performance
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics['total_ms'].append(elapsed_ms)
        self._cleanup_metrics()
        
        if elapsed_ms > 200:
            logger.warning(f"Hot path took {elapsed_ms:.1f}ms (budget: 200ms)")
        
        return bullets, triples
    
    def _extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """
        Extract entities and relations using UD or regex fallback
        Returns: (entities, triples, negation_count, doc)
        """
        nlp = _load_nlp(lang)
        
        if nlp and not SPACY_DISABLE:
            return self._ud_extract(text, nlp)
        else:
            return self._regex_extract(text)
    
    def _ud_extract(self, text: str, nlp) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """Extract using USGS Grammar-to-Graph 27 dependency patterns"""
        doc = nlp(text)
        entities: Set[str] = set()
        triples: List[Tuple[str, str, str]] = []
        neg_count = 0
        
        # Stage 1: Extract all entities (NER, chunks, nouns, pronouns)
        entity_map = self._build_entity_map(doc, entities)
        
        # Stage 2: Process all 27 dependency types systematically
        for token in doc:
            dep = token.dep_
            
            # Count negations
            if dep == "neg":
                neg_count += 1
            
            # Core grammatical relations (6 types)
            if dep in {"nsubj", "nsubjpass"}:
                self._extract_subject(token, entity_map, triples, entities)
            elif dep in {"dobj", "obj"}:
                self._extract_object(token, entity_map, triples, entities)
            elif dep == "iobj":
                self._extract_indirect_object(token, entity_map, triples, entities)
            elif dep == "attr":
                self._extract_attribute(token, entity_map, triples, entities)
            
            # Modifier relations (4 types)
            elif dep == "amod":
                self._extract_adjectival_mod(token, entity_map, triples, entities)
            elif dep == "advmod":
                self._extract_adverbial_mod(token, entity_map, triples, entities)
            elif dep == "nummod":
                self._extract_numeric_mod(token, entity_map, triples, entities)
            elif dep == "nmod":
                self._extract_nominal_mod(token, entity_map, triples, entities)
            
            # Structural relations (6 types)
            elif dep == "compound":
                self._extract_compound(token, entity_map, triples, entities)
            elif dep == "poss":
                self._extract_possessive(token, entity_map, triples, entities)
            elif dep == "appos":
                self._extract_apposition(token, entity_map, triples, entities)
            elif dep == "conj":
                self._extract_conjunction(token, entity_map, triples, entities)
            elif dep == "prep":
                self._extract_preposition(token, entity_map, triples, entities)
            elif dep == "pobj":
                self._extract_prep_object(token, entity_map, triples, entities)
            
            # Clausal relations (5 types)
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
            
            # Special relations (2 types)
            elif dep == "agent":
                self._extract_agent(token, entity_map, triples, entities)
            elif dep == "oprd":
                self._extract_oprd(token, entity_map, triples, entities)
            
            # Function words (4 types) - usually don't generate triples
            # aux, auxpass, cop, det, case, mark, cc, neg - handled elsewhere or ignored
        
        return list(entities), triples, neg_count, doc
    
    def _build_entity_map(self, doc, entities: Set[str]) -> Dict[int, str]:
        """Build comprehensive entity map"""
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
    
    # === 27 Dependency Handler Methods (USGS Grammar-to-Graph) ===
    
    def _extract_subject(self, token, entity_map, triples, entities):
            root = sent.root
            
            # Check for negation
            if any(child.dep_ == "neg" for child in root.children):
                neg_count += 1
            
            # Pattern 1: Copula constructions (X is Y)
            # Handle both cop dependency and ROOT AUX (spaCy quirk)
            is_copula = (root.pos_ == "AUX" and root.lemma_ in {"be", "is", "am", "are", "was", "were"}) or \
                       any(child.dep_ == "cop" for child in root.children)
            
            if is_copula or root.pos_ == "AUX":  # Always check AUX roots
                subj = self._find_child(root, {"nsubj", "nsubj:pass"})
                attr = self._find_child(root, {"attr", "acomp", "amod"})
                
                # For ROOT AUX, attr might be direct child
                if not attr and root.pos_ == "AUX":
                    for child in root.children:
                        if child.pos_ in {"PROPN", "NOUN", "ADJ"}:
                            attr = child
                            break
                
                if subj and attr:
                    # Handle possessive chains (e.g., "My dog's name is Potola")
                    # We need to extract both the possession and the attribute
                    
                    # Check for possessive modifiers on the subject
                    possessive_chain = []
                    current = subj
                    while current:
                        poss_child = self._find_child(current, {"poss"})
                        if poss_child:
                            possessive_chain.append(poss_child)
                            current = poss_child
                        else:
                            break
                    
                    # If we have a possessive chain, extract relationships
                    if possessive_chain:
                        # Example: "My dog's name" -> My=you, dog, name
                        # Extract: (you, has, dog) and (dog, name, Potola)
                        
                        # Convert "My" to canonical "you"
                        first_poss = possessive_chain[-1]  # Reverse order
                        owner_text = first_poss.text.lower()
                        owner = self.user_eid if owner_text in {"my", "mine"} else self._get_entity_id(first_poss)
                        
                        # If subject is "name" and we have possessors, build the chain
                        if subj.text.lower() in {"name", "nome", "nombre"}:
                            # Find the actual possessed entity (e.g., "dog" in "dog's name")
                            possessed = None
                            for child in subj.children:
                                if child.dep_ == "poss" and child.text.lower() not in {"my", "mine"}:
                                    possessed = self._get_entity_id(child)
                                    # Add possession relation
                                    if owner and possessed:
                                        triples.append((owner, "has", possessed))
                                        entities.add(possessed)
                                    # Add name relation
                                    if possessed:
                                        attr_eid = entity_map.get(attr.i, _norm(attr.text))
                                        if attr_eid:
                                            triples.append((possessed, "name", attr_eid))
                                            entities.add(attr_eid)
                                    break
                            
                            # Simple "My name is X" without intermediate entity
                            if not possessed and owner == self.user_eid:
                                attr_eid = entity_map.get(attr.i, _norm(attr.text))
                                if attr_eid:
                                    triples.append((owner, "name", attr_eid))
                                    entities.add(attr_eid)
                    else:
                        # Simple copula without possessive chain
                        subj_eid = entity_map.get(subj.i, _norm(subj.text))
                        attr_eid = entity_map.get(attr.i, _norm(attr.text))
                        
                        # Handle "My name is X" (possessive on name)
                        if subj and subj.text.lower() == "name":
                            poss = self._find_child(subj, {"poss"})
                            if poss and poss.text.lower() in {"my", "mine"}:
                                if attr_eid:
                                    triples.append((self.user_eid, "name", attr_eid))
                                    entities.add(attr_eid)
                                    # Also add fallback extraction
                            else:
                                # No possessive found but it's still "name is X"
                                if attr_eid:
                                    triples.append((subj_eid, "is", attr_eid))
                                    entities.add(attr_eid)
                        # Handle "My favorite X is Y"
                        elif any(c.dep_ == "poss" and c.text.lower() in {"my", "mine"} for c in subj.children):
                            if "favorite" in subj.text.lower() or "favourite" in subj.text.lower():
                                subj_type = subj.text.lower().replace("favorite", "").replace("favourite", "").strip()
                                if attr_eid:
                                    triples.append((self.user_eid, f"favorite_{subj_type}", attr_eid))
                                    entities.add(attr_eid)
                        # Regular copula
                        elif subj_eid and attr_eid:
                            rel = "name" if self._contains_lemma(sent, {"name", "nome", "nombre", "nom"}) else "is"
                            triples.append((subj_eid, rel, attr_eid))
            
            # Apposition (X, Y)
            for token in sent:
                if token.dep_ == "appos" and token.head.pos_ in {"NOUN", "PROPN"} and token.pos_ == "PROPN":
                    head_eid = self._get_entity_id(token.head)
                    appos_eid = self._get_entity_id(token)
                    if head_eid and appos_eid:
                        triples.append((head_eid, "alias", appos_eid))
            
            # SVO constructions and verb patterns
            if root.pos_ == "VERB":
                subj = self._find_child(root, {"nsubj", "nsubj:pass", "nsubjpass"})
                obj = self._find_child(root, {"obj", "dobj", "iobj"})
                
                # Handle passive voice: "My son is named Jake"
                if root.lemma_ in {"name", "call"} and any(c.dep_ in {"auxpass", "aux:pass"} for c in root.children):
                    name = self._find_child(root, {"oprd", "xcomp", "attr"})
                    if subj and name:
                        # Check for possessive on subject
                        poss = self._find_child(subj, {"poss"})
                        if poss and poss.text.lower() in {"my", "mine"}:
                            subj_eid = self._get_entity_id(subj)
                            name_eid = self._get_entity_id(name)
                            if subj_eid and name_eid:
                                triples.append((self.user_eid, "has", subj_eid))
                                triples.append((subj_eid, "name", name_eid))
                                entities.add(subj_eid)
                                entities.add(name_eid)
                # Regular SVO
                elif subj and obj:
                    subj_eid = self._get_entity_id(subj)
                    obj_eid = self._get_entity_id(obj)
                    if subj_eid and obj_eid:
                        if root.lemma_ in {"have", "has", "own"}:
                            triples.append((subj_eid, "has", obj_eid))
                        else:
                            triples.append((subj_eid, f"v:{root.lemma_}", obj_eid))
                
                # Prepositional patterns: "I live in Seattle", "I work at Microsoft"
                if subj:
                    subj_eid = entity_map.get(subj.i, _norm(subj.text))
                    for child in root.children:
                        if child.dep_ == "prep":
                            pobj = self._find_child(child, {"pobj"})
                            if pobj and pobj.pos_ in {"PROPN", "NOUN"}:
                                pobj_eid = entity_map.get(pobj.i, _norm(pobj.text))
                                if subj_eid and pobj_eid:
                                    prep = child.text.lower()
                                    # Special patterns
                                    if root.lemma_ == "live" and prep == "in":
                                        triples.append((subj_eid, "lives_in", pobj_eid))
                                        entities.add(pobj_eid)
                                    elif root.lemma_ == "work" and prep in {"at", "for"}:
                                        triples.append((subj_eid, "works_at", pobj_eid))
                                        entities.add(pobj_eid)
                                    elif root.lemma_ in {"bear", "born"} and prep == "in":
                                        triples.append((subj_eid, "born_in", pobj_eid))
                                        entities.add(pobj_eid)
                
                # Oblique/prepositional relations (fallback)
                for obl in root.children:
                    if obl.dep_ in {"obl", "nmod"}:
                        case = self._find_child(obl, {"case", "mark"})
                        if case and subj:
                            subj_eid = self._get_entity_id(subj)
                            obl_eid = self._get_entity_id(obl)
                            if subj_eid and obl_eid:
                                triples.append((subj_eid, f"obl:{root.lemma_}_{case.text}", obl_eid))
        
        return list(entities), triples, neg_count, doc
    
    def _regex_extract(self, text: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, None]:
        """Fallback extraction using regex patterns"""
        tnorm = _norm(text)
        entities: Set[str] = set()
        triples: List[Tuple[str, str, str]] = []
        neg_count = 1 if any(neg in tnorm for neg in ["not", "don't", "doesn't", "no", "never"]) else 0
        
        # Extract capitalized words as entities
        for match in re.finditer(r'\b[A-Z][\w\-]{2,}\b', text):
            entities.add(_norm(match.group()))
        
        # Common patterns
        patterns = [
            # Name patterns
            (r"my (\w+)'s name is ([A-Z][\w\-']+)", lambda m: (self.user_eid, "has", _norm(m[1])), (_norm(m[1]), "name", _norm(m[2]))),
            (r"(\w+)'s name is ([A-Z][\w\-']+)", lambda m: (_norm(m[1]), "name", _norm(m[2]))),
            (r"i'm ([A-Z][\w\-']+)", lambda m: (self.user_eid, "name", _norm(m[1]))),
            (r"my name is ([A-Z][\w\-']+)", lambda m: (self.user_eid, "name", _norm(m[1]))),
            
            # Location patterns
            (r"i live in ([A-Z][\w\-' ]+)", lambda m: (self.user_eid, "lives_in", _norm(m[1]))),
            (r"i'm from ([A-Z][\w\-' ]+)", lambda m: (self.user_eid, "from", _norm(m[1]))),
            (r"i work at ([A-Z][\w\-' ]+)", lambda m: (self.user_eid, "works_at", _norm(m[1]))),
            
            # Possession patterns  
            (r"i have (?:a |an )?(\w+)", lambda m: (self.user_eid, "has", _norm(m[1]))),
            (r"my (\w+) is ([A-Z][\w\-']+)", lambda m: (self.user_eid, "has", _norm(m[1])), (_norm(m[1]), "name", _norm(m[2]))),
            
            # Preference patterns
            (r"i (?:like|love|enjoy) (\w+)", lambda m: (self.user_eid, "likes", _norm(m[1]))),
            (r"i (?:hate|dislike) (\w+)", lambda m: (self.user_eid, "dislikes", _norm(m[1]))),
        ]
        
        for pattern, extractor in patterns:
            for match in re.finditer(pattern, tnorm):
                result = extractor(match)
                if isinstance(result, tuple) and len(result) == 3:
                    triples.append(result)
                    entities.update([result[0], result[2]])
                elif isinstance(result, tuple) and len(result) == 2:
                    # Multiple triples returned
                    for triple in result:
                        if isinstance(triple, tuple) and len(triple) == 3:
                            triples.append(triple)
                            entities.update([triple[0], triple[2]])
        
        # Special handling for "my X" patterns
        if "my dog" in tnorm:
            triples.append((self.user_eid, "has_pet", "dog"))
            entities.add("dog")
        if "my cat" in tnorm:
            triples.append((self.user_eid, "has_pet", "cat"))
            entities.add("cat")
        
        return list(entities), triples, neg_count, None
    
    def _update_memory(self, entities: List[str], triples: List[Tuple[str, str, str]], 
                      neg_count: int, text: str, session_id: str, turn_id: int):
        """Update memory store and RAM indices"""
        now = int(time.time())
        
        for s, r, d in triples:
            # Canonicalize entities
            s = self._canonicalize(s)
            d = self._canonicalize(d)
            
            # Determine confidence
            conf = 0.9 if r in {"name", "alias"} else (0.75 if r.startswith("v:") else 0.6)
            
            # Handle negation
            if neg_count > 0 and r.startswith("v:"):
                self.store.negate_edge(s, r, d, conf=0.6, now_ts=now)
            else:
                self.store.observe_edge(s, r, d, conf=conf, now_ts=now)
            
            # Update RAM indices
            self.adj_out[s][r].append((d, conf, now))
            self.entity_access[s] = time.time()
            self.entity_access[d] = time.time()
            
            # Add to recency ring
            self.recency.append(Factlet(s, r, d, text[:200], now, session_id, turn_id))
            
            # Store mention
            self.store.enqueue_mention(s, text, now, session_id, turn_id)
        
        # Flush to disk if needed
        self.store.flush_if_needed()
        
        # Cleanup old entities if needed
        self._cleanup_entities()
    
    def _retrieve_and_format(self, entities: List[str], k: int = 3) -> List[str]:
        """Retrieve relevant facts and format as bullets"""
        candidates = []
        now = int(time.time())
        
        # Look up adjacency for detected entities
        for eid in entities or [self.user_eid]:
            eid = self._canonicalize(eid)
            if eid in self.adj_out:
                for rel, neighbors in self.adj_out[eid].items():
                    for dst, weight, ts in neighbors:
                        # Time decay: half-life of 30 minutes
                        decay = 0.5 ** ((now - ts) / 1800.0)
                        score = decay * weight
                        candidates.append((score, eid, rel, dst, ts))
        
        # Also check recency for entity mentions
        if len(candidates) < k:
            for factlet in reversed(self.recency):
                if not entities or factlet.s in entities or factlet.d in entities:
                    candidates.append((0.95, factlet.s, factlet.r, factlet.d, factlet.ts))
                    if len(candidates) >= k * 2:
                        break
        
        # Sort by score and take top k
        candidates.sort(key=lambda x: -x[0])
        
        # Format as bullets
        bullets = []
        seen = set()
        for score, s, r, d, ts in candidates[:k*2]:
            # Deduplicate
            key = f"{s}|{r}|{d}"
            if key in seen:
                continue
            seen.add(key)
            
            # Find best snippet from recency
            snippet = self._find_snippet(s, r, d)
            if snippet:
                bullets.append(f"• {snippet}")
            else:
                # Format relation nicely
                r_display = r.replace("_", " ").replace("v:", "").replace("obl:", "")
                if r == "name":
                    bullets.append(f"• {s}'s name is {d}")
                elif r == "has_pet":
                    bullets.append(f"• You have a {d}")
                elif r.startswith("likes"):
                    bullets.append(f"• You like {d}")
                else:
                    bullets.append(f"• {s} {r_display} {d}")
            
            if len(bullets) >= k:
                break
        
        return bullets
    
    # ---------- Helper methods ----------
    def _find_child(self, head, dep_set):
        """Find child with specific dependency relation"""
        if isinstance(dep_set, str):
            dep_set = {dep_set}
        for child in head.children:
            if child.dep_ in dep_set:
                return child
        return None
    
    def _contains_lemma(self, span, lemmas: Set[str]) -> bool:
        """Check if span contains any of the lemmas"""
        for token in span:
            try:
                if token.lemma_.lower() in lemmas:
                    return True
            except:
                pass
        return False
    
    def _get_entity_id(self, token) -> str:
        """Get normalized entity ID from token"""
        if token:
            return _norm(token.text)
        return ""
    
    def _canonicalize(self, eid: str) -> str:
        """Canonicalize entity ID (future: fuzzy matching)"""
        if eid in self.entity_cache:
            return self.entity_cache[eid]
        
        # For now, simple normalization
        # Future: use rapidfuzz for fuzzy matching
        canonical = _norm(eid)
        self.entity_cache[eid] = canonical
        return canonical
    
    def _find_snippet(self, s: str, r: str, d: str) -> Optional[str]:
        """Find best snippet from recency buffer"""
        for factlet in reversed(self.recency):
            if factlet.s == s and factlet.r == r and factlet.d == d:
                # Return concise version of original text
                text = re.sub(r"\s+", " ", factlet.text).strip()
                if len(text) <= 100:
                    return text
                return text[:97] + "..."
        return None
    
    def _cleanup_entities(self):
        """Remove stale entities to prevent memory leaks"""
        if len(self.entity_access) > self.max_entities:
            # Remove 20% of oldest entities
            cutoff_time = time.time() - 86400  # 24 hours
            to_remove = []
            
            for eid, last_access in self.entity_access.items():
                if last_access < cutoff_time:
                    to_remove.append(eid)
            
            for eid in to_remove:
                del self.entity_access[eid]
                if eid in self.adj_out:
                    del self.adj_out[eid]
    
    def _cleanup_metrics(self):
        """Keep metrics bounded"""
        for key in self.metrics:
            if len(self.metrics[key]) > 100:
                self.metrics[key] = self.metrics[key][-100:]
    
    def _time_operation(self, op_name: str):
        """Context manager for timing operations"""
        class Timer:
            def __init__(self, memory, name):
                self.memory = memory
                self.name = name
                self.start = None
            
            def __enter__(self):
                self.start = time.perf_counter()
                return self
            
            def __exit__(self, *args):
                elapsed_ms = (time.perf_counter() - self.start) * 1000
                self.memory.metrics[f"{self.name}_ms"].append(elapsed_ms)
        
        return Timer(self, op_name)
    
    def rebuild_from_store(self):
        """Rebuild RAM indices from persistent store on startup"""
        logger.info("Rebuilding hot memory from store...")
        start = time.perf_counter()
        
        cur = self.store.sql.cursor()
        count = 0
        
        for (s, r, d, w, ts, status) in cur.execute(
            "SELECT src, rel, dst, weight, updated_at, status FROM edge WHERE status = 1 ORDER BY updated_at DESC LIMIT 1000"
        ):
            self.adj_out[s][r].append((d, float(w), int(ts)))
            self.entity_access[s] = float(ts)
            self.entity_access[d] = float(ts)
            count += 1
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Rebuilt {count} edges in {elapsed_ms:.1f}ms")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {}
        for key, values in self.metrics.items():
            if values:
                import statistics
                metrics[key] = {
                    'mean': statistics.mean(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) > 20 else max(values),
                    'count': len(values)
                }
        
        # Add memory usage
        metrics['entities'] = len(self.entity_access)
        metrics['recency_buffer'] = len(self.recency)
        
        return metrics