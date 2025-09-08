#!/usr/bin/env python3
"""
Enhanced HotMem Relation Extractor
Designed to integrate seamlessly with HotMem's existing extraction pipeline.
Focuses on extracting high-quality relations using enhanced dependency patterns.
"""

import os
import time
from typing import List, Tuple, Optional, Dict, Set
from loguru import logger
import spacy
from spacy.tokens import Doc, Token

# Try to import sentence transformers for cross-lingual similarity
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


class EnhancedHotMemExtractor:
    """
    Enhanced HotMem extractor that works with HotMem's pipeline.
    Focuses on target relations using enhanced dependency patterns.
    """
    
    # Target relations for HotMem extraction
    TARGET_RELATIONS = {
        "lives_in", "works_at", "teach_at", "born_in", "moved_from",
        "went_to", "married_to", "also_known_as", "name", "age", 
        "has", "owns", "founded", "invented", "discovered"
    }
    
    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        """Initialize with HotMem-compatible interface."""
        self.model_id = model_id or "enhanced-hotmem-extractor"
        self.device = device
        self._ready = False
        
        # Load cross-lingual embeddings if available
        self.embed_model = None
        if SBERT_AVAILABLE:
            try:
                self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("[Enhanced HotMem] Loaded multilingual embeddings")
            except:
                pass
    
    def extract(self, text: str) -> List[Tuple[str, str, str, float]]:
        """
        Extract relations from text. Compatible with ReLiK interface.
        Returns: List of (subject, relation, object, confidence) tuples.
        
        This method is designed to be called AFTER HotMem's main extraction,
        to add additional high-confidence relations that UD patterns might miss.
        """
        if not text or len(text) > 480:  # Respect ReLiK's text limit
            return []
        
        # Detect language
        lang = self._detect_language(text)
        
        # Get spaCy doc (reuse HotMem's model loading)
        try:
            from components.memory.memory_hotpath import _load_nlp
            nlp = _load_nlp(lang)
        except:
            # Fallback
            try:
                nlp = spacy.load(f"{lang}_core_web_sm" if lang != "en" else "en_core_web_sm")
            except:
                nlp = spacy.load("en_core_web_sm")
        
        if not nlp:
            return []
        
        doc = nlp(text)
        triples = []
        
        # Strategy 1: Enhanced passive voice patterns
        triples.extend(self._extract_passive_patterns(doc))
        
        # Strategy 2: Complex conjunctions
        triples.extend(self._extract_conjunction_patterns(doc))
        
        # Strategy 3: Copula with role descriptions
        triples.extend(self._extract_copula_roles(doc))
        
        # Strategy 4: Cross-lingual verb mapping
        if self.embed_model:
            triples.extend(self._extract_crosslingual_patterns(doc, lang))
        
        # Deduplicate and filter
        seen = set()
        result = []
        for s, r, o, conf in triples:
            key = (s.lower().strip(), r, o.lower().strip())
            if key not in seen and r in self.TARGET_RELATIONS:
                seen.add(key)
                result.append((s, r, o, conf))
        
        return result
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # Check for common non-English words
        words = text.lower().split()
        
        if any(w in ["le", "la", "les", "est", "sont", "dans"] for w in words):
            return "fr"
        elif any(w in ["el", "la", "los", "es", "son", "en"] for w in words):
            return "es"
        elif any(w in ["der", "die", "das", "ist", "sind", "bei"] for w in words):
            return "de"
        elif any(w in ["il", "la", "gli", "è", "sono", "nel"] for w in words):
            return "it"
        
        return "en"
    
    def _extract_passive_patterns(self, doc: Doc) -> List[Tuple[str, str, str, float]]:
        """Extract from passive constructions like 'X was founded by Y'."""
        triples = []
        
        for token in doc:
            # Look for passive auxiliaries
            if token.dep_ == "auxpass" and token.head.pos_ == "VERB":
                verb = token.head
                
                # Find subject (which is the object in passive)
                obj = None
                for child in verb.children:
                    if child.dep_ in {"nsubjpass", "nsubj"}:
                        obj = self._get_entity_text(child)
                        break
                
                # Find agent (the real subject)
                agents = []
                for child in verb.children:
                    if child.dep_ == "agent" or (child.dep_ == "prep" and child.text.lower() == "by"):
                        for gc in child.children:
                            if gc.dep_ in {"pobj", "obj"}:
                                agents.append(self._get_entity_text(gc))
                                # Check for conjunctions
                                for conj in gc.children:
                                    if conj.dep_ == "conj":
                                        agents.append(self._get_entity_text(conj))
                
                # Map verb to relation
                if obj and agents:
                    relation = self._map_verb_to_relation(verb.lemma_)
                    if relation:
                        for agent in agents:
                            triples.append((agent, relation, obj, 0.85))
        
        return triples
    
    def _extract_conjunction_patterns(self, doc: Doc) -> List[Tuple[str, str, str, float]]:
        """Extract from compound sentences with conjunctions."""
        triples = []
        
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                # Find coordinated verbs
                for conj in token.children:
                    if conj.dep_ == "conj" and conj.pos_ == "VERB":
                        # The subject is usually shared
                        subj = None
                        for child in token.children:
                            if child.dep_ in {"nsubj", "nsubjpass"}:
                                subj = self._get_entity_text(child)
                                break
                        
                        # Find object of conjunct verb
                        if subj:
                            for child in conj.children:
                                if child.dep_ == "prep":
                                    prep = child.text.lower()
                                    for gc in child.children:
                                        if gc.dep_ in {"pobj", "obj"}:
                                            obj = self._get_entity_text(gc)
                                            relation = self._map_verb_prep_to_relation(conj.lemma_, prep)
                                            if relation:
                                                triples.append((subj, relation, obj, 0.8))
        
        return triples
    
    def _extract_copula_roles(self, doc: Doc) -> List[Tuple[str, str, str, float]]:
        """Extract from 'X is the CEO of Y' patterns."""
        triples = []
        
        for token in doc:
            # Handle multiple languages' copula
            if token.lemma_ in {"be", "être", "ser", "estar", "sein", "essere"} and token.dep_ == "ROOT":
                subj = None
                for child in token.children:
                    if child.dep_ in {"nsubj", "nsubjpass"}:
                        subj = self._get_entity_text(child)
                        break
                
                if subj:
                    for child in token.children:
                        if child.dep_ == "attr":
                            attr_text = child.text.lower()
                            
                            # Look for "of/de/von" pattern
                            for gc in child.children:
                                if gc.dep_ == "prep" and gc.text.lower() in ["of", "de", "von", "di"]:
                                    for ggc in gc.children:
                                        if ggc.dep_ in {"pobj", "obj"}:
                                            org = self._get_entity_text(ggc)
                                            
                                            # Map role to relation
                                            if any(role in attr_text for role in ["ceo", "founder", "director", "president"]):
                                                triples.append((subj, "works_at", org, 0.75))
                                            elif any(role in attr_text for role in ["capital", "capitale", "hauptstadt"]):
                                                triples.append((org, "capital", subj, 0.75))
        
        return triples
    
    def _extract_crosslingual_patterns(self, doc: Doc, lang: str) -> List[Tuple[str, str, str, float]]:
        """Use embeddings to match cross-lingual verb patterns."""
        if not self.embed_model:
            return []
        
        triples = []
        
        # Define target patterns with descriptions
        patterns = {
            "founded": ["founded company", "created organization", "established business"],
            "invented": ["invented technology", "discovered element", "created innovation"],
            "works_at": ["works at company", "employed by organization"],
            "lives_in": ["lives in city", "resides in location"],
        }
        
        for token in doc:
            if token.pos_ == "VERB":
                # Get verb phrase
                verb_phrase = token.text
                for child in token.children:
                    if child.dep_ == "prep":
                        verb_phrase = f"{token.text} {child.text}"
                        break
                
                # Find best matching pattern
                try:
                    verb_emb = self.embed_model.encode(verb_phrase)
                    best_relation = None
                    best_score = 0.0
                    
                    for rel, descriptions in patterns.items():
                        pattern_embs = self.embed_model.encode(descriptions)
                        scores = self.embed_model.similarity(verb_emb, pattern_embs)
                        max_score = float(scores.max())
                        
                        if max_score > best_score and max_score > 0.6:
                            best_score = max_score
                            best_relation = rel
                    
                    # If found a match, extract subject and object
                    if best_relation:
                        subj = None
                        obj = None
                        
                        for child in token.children:
                            if child.dep_ in {"nsubj", "nsubjpass"}:
                                subj = self._get_entity_text(child)
                            elif child.dep_ in {"dobj", "obj"}:
                                obj = self._get_entity_text(child)
                            elif child.dep_ == "prep":
                                for gc in child.children:
                                    if gc.dep_ in {"pobj", "obj"}:
                                        obj = self._get_entity_text(gc)
                        
                        if subj and obj:
                            triples.append((subj, best_relation, obj, best_score))
                
                except Exception as e:
                    logger.debug(f"Cross-lingual extraction failed: {e}")
        
        return triples
    
    def _get_entity_text(self, token: Token) -> str:
        """Get full entity text from token."""
        # Try to get the full noun chunk
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                return chunk.text
        
        # Otherwise get subtree
        subtree = sorted(token.subtree, key=lambda t: t.i)
        return " ".join(t.text for t in subtree)
    
    def _map_verb_to_relation(self, verb_lemma: str) -> Optional[str]:
        """Map verb lemma to target relation."""
        verb = verb_lemma.lower()
        
        # Direct mappings
        mappings = {
            "found": "founded",
            "founded": "founded",
            "establish": "founded",
            "create": "founded",
            "start": "founded",
            "invent": "invented",
            "discover": "discovered",
            "develop": "invented",
            "marry": "married_to",
            "wed": "married_to",
        }
        
        return mappings.get(verb)
    
    def _map_verb_prep_to_relation(self, verb_lemma: str, prep: str) -> Optional[str]:
        """Map verb + preposition to relation."""
        verb = verb_lemma.lower()
        prep = prep.lower()
        
        patterns = {
            ("live", "in"): "lives_in",
            ("lives", "in"): "lives_in",
            ("work", "at"): "works_at",
            ("works", "at"): "works_at",
            ("work", "for"): "works_at",
            ("teach", "at"): "teach_at",
            ("teaches", "at"): "teach_at",
            ("born", "in"): "born_in",
            ("move", "from"): "moved_from",
            ("moved", "from"): "moved_from",
            ("go", "to"): "went_to",
            ("went", "to"): "went_to",
        }
        
        return patterns.get((verb, prep))


def test_extractor():
    """Test the enhanced extractor."""
    extractor = EnhancedHotMemExtractor()
    
    test_cases = [
        "My brother Tom lives in Portland and teaches at Reed College.",
        "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino in 1976.",
        "Marie Curie discovered radium and polonium with her husband Pierre.",
        "Barcelona es la capital de Cataluña.",
        "Elon Musk is the CEO of Tesla and SpaceX.",
    ]
    
    print("=== Enhanced ReLiK Replacement Test ===\n")
    
    for text in test_cases:
        print(f"Text: {text}")
        
        start = time.perf_counter()
        relations = extractor.extract(text)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"Time: {elapsed:.1f}ms")
        print("Relations:")
        for s, r, o, conf in relations:
            print(f"  ({s}, {r}, {o}) conf={conf:.2f}")
        print()


if __name__ == "__main__":
    test_extractor()