#!/usr/bin/env python3
"""
Lightweight multilingual relation extractor as ReLiK replacement.
Built on spaCy dependency parsing with targeted patterns for HotMem relations.
"""

from typing import List, Tuple, Optional, Dict, Set
import spacy
from spacy.tokens import Doc, Token
from loguru import logger
import time


class LightweightRelationExtractor:
    """
    Drop-in replacement for ReLiK using enhanced spaCy dependency parsing.
    Targets the 12 specific relations HotMem needs with multilingual support.
    """
    
    # Target relations from ReLiK extractor
    TARGET_RELATIONS = {
        "lives_in", "works_at", "teach_at", "born_in", "moved_from", 
        "went_to", "married_to", "also_known_as", "name", "age", "has", "owns"
    }
    
    # Multilingual verb patterns mapped to target relations
    VERB_PATTERNS = {
        "lives_in": {
            "en": ["live", "reside", "dwell", "stay", "inhabit"],
            "es": ["vivir", "residir", "habitar", "morar"],
            "fr": ["vivre", "habiter", "résider", "demeurer"],
            "de": ["wohnen", "leben", "residieren"],
            "it": ["vivere", "abitare", "risiedere", "dimorare"]
        },
        "works_at": {
            "en": ["work", "employ", "job"],
            "es": ["trabajar", "emplear", "laborar"],
            "fr": ["travailler", "employer", "bosser"],
            "de": ["arbeiten", "beschäftigen", "jobben"],
            "it": ["lavorare", "impiegare", "operare"]
        },
        "teach_at": {
            "en": ["teach", "lecture", "instruct", "educate"],
            "es": ["enseñar", "instruir", "educar", "impartir"],
            "fr": ["enseigner", "instruire", "éduquer"],
            "de": ["lehren", "unterrichten", "dozieren"],
            "it": ["insegnare", "istruire", "educare"]
        },
        "born_in": {
            "en": ["born", "birth"],
            "es": ["nacer", "nacido"],
            "fr": ["naître", "né"],
            "de": ["geboren", "geburt"],
            "it": ["nascere", "nato"]
        },
        "married_to": {
            "en": ["marry", "married", "wed", "spouse", "husband", "wife"],
            "es": ["casar", "casado", "esposo", "esposa", "marido", "mujer"],
            "fr": ["marier", "marié", "époux", "épouse", "mari", "femme"],
            "de": ["heiraten", "verheiratet", "ehemann", "ehefrau"],
            "it": ["sposare", "sposato", "sposo", "sposa", "marito", "moglie"]
        },
        "has": {
            "en": ["have", "has", "own", "possess", "got"],
            "es": ["tener", "poseer", "haber"],
            "fr": ["avoir", "posséder"],
            "de": ["haben", "besitzen"],
            "it": ["avere", "possedere"]
        },
        "owns": {
            "en": ["own", "possess", "belong"],
            "es": ["poseer", "pertenecer", "ser dueño"],
            "fr": ["posséder", "appartenir", "être propriétaire"],
            "de": ["besitzen", "gehören", "eigentümer"],
            "it": ["possedere", "appartenere", "proprietario"]
        }
    }
    
    # Preposition patterns for location relations
    PREP_PATTERNS = {
        "lives_in": ["in", "at", "on", "en", "à", "dans", "auf", "bei", "a", "in"],
        "works_at": ["at", "for", "in", "en", "para", "chez", "à", "bei", "für", "presso", "per"],
        "teach_at": ["at", "in", "en", "à", "bei", "a", "presso"],
        "born_in": ["in", "at", "en", "à", "dans", "in", "a"]
    }
    
    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        """Initialize with ReLiK-compatible interface."""
        self.model_id = model_id or "lightweight-spacy"
        self.device = device
        self.nlp_cache: Dict[str, spacy.Language] = {}
        self._ready = False
        
    def _get_nlp(self, lang: str = "en") -> spacy.Language:
        """Get or cache spaCy model for language."""
        if lang not in self.nlp_cache:
            model_map = {
                "en": "en_core_web_sm",
                "es": "es_core_web_sm", 
                "fr": "fr_core_web_sm",
                "de": "de_core_news_sm",
                "it": "it_core_news_sm"
            }
            model_name = model_map.get(lang, "en_core_web_sm")
            try:
                self.nlp_cache[lang] = spacy.load(model_name)
                logger.info(f"Loaded spaCy model {model_name} for {lang}")
            except:
                # Fallback to English
                self.nlp_cache[lang] = spacy.load("en_core_web_sm")
                logger.warning(f"Failed to load {model_name}, using English")
        return self.nlp_cache[lang]
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words."""
        # This is a simplified version - in production use pycld3 or langdetect
        lang_indicators = {
            "es": ["el", "la", "de", "que", "en", "es", "por"],
            "fr": ["le", "la", "de", "que", "dans", "est", "pour"],
            "de": ["der", "die", "das", "und", "ist", "für", "mit"],
            "it": ["il", "la", "di", "che", "è", "per", "con"]
        }
        
        words = text.lower().split()
        for lang, indicators in lang_indicators.items():
            if sum(1 for w in words if w in indicators) >= 2:
                return lang
        return "en"
    
    def _map_verb_to_relation(self, verb_lemma: str, lang: str = "en") -> Optional[str]:
        """Map verb lemma to target relation."""
        verb_lower = verb_lemma.lower()
        
        for relation, patterns in self.VERB_PATTERNS.items():
            lang_patterns = patterns.get(lang, patterns.get("en", []))
            if verb_lower in lang_patterns:
                return relation
        
        # Check if verb contains relation keywords
        if "found" in verb_lower or "establish" in verb_lower:
            return "found"
        if "discover" in verb_lower:
            return "discover"
            
        return None
    
    def _extract_subject(self, token: Token) -> Optional[str]:
        """Extract subject from verb token."""
        subjects = [t for t in token.children if t.dep_ in ("nsubj", "nsubjpass")]
        if subjects:
            # Get full noun phrase
            subj = subjects[0]
            # Include compounds and modifiers
            subtree = list(subj.subtree)
            return " ".join(t.text for t in sorted(subtree, key=lambda x: x.i))
        
        # For passive constructions, look for agent (by-phrase)
        for child in token.children:
            if child.dep_ == "agent" or (child.dep_ == "prep" and child.text.lower() == "by"):
                pobjs = [t for t in child.children if t.dep_ in ("pobj", "obj")]
                if pobjs:
                    subtree = list(pobjs[0].subtree)
                    return " ".join(t.text for t in sorted(subtree, key=lambda x: x.i))
        
        return None
    
    def _extract_object_with_prep(self, token: Token) -> Tuple[Optional[str], Optional[str]]:
        """Extract object and preposition from verb token."""
        # Direct objects
        dobjs = [t for t in token.children if t.dep_ in ("dobj", "obj")]
        if dobjs:
            obj = dobjs[0]
            subtree = list(obj.subtree)
            obj_text = " ".join(t.text for t in sorted(subtree, key=lambda x: x.i))
            return obj_text, None
            
        # Prepositional objects
        preps = [t for t in token.children if t.dep_ == "prep"]
        for prep in preps:
            pobjs = [t for t in prep.children if t.dep_ in ("pobj", "obj")]
            if pobjs:
                obj = pobjs[0]
                subtree = list(obj.subtree)
                obj_text = " ".join(t.text for t in sorted(subtree, key=lambda x: x.i))
                return obj_text, prep.text
                
        # Check for phrasal verbs (particle + prep)
        prts = [t for t in token.children if t.dep_ == "prt"]
        for prt in prts:
            for sibling in token.children:
                if sibling.dep_ == "prep":
                    pobjs = [t for t in sibling.children if t.dep_ in ("pobj", "obj")]
                    if pobjs:
                        obj = pobjs[0]
                        subtree = list(obj.subtree)
                        obj_text = " ".join(t.text for t in sorted(subtree, key=lambda x: x.i))
                        return obj_text, sibling.text
                        
        return None, None
    
    def _extract_from_dependencies(self, doc: Doc, lang: str = "en") -> List[Tuple[str, str, str, float]]:
        """Extract relations using dependency parsing."""
        triples = []
        
        for token in doc:
            if token.pos_ == "VERB":
                subject = self._extract_subject(token)
                obj, prep = self._extract_object_with_prep(token)
                
                if subject and obj:
                    # Map verb to relation
                    relation = self._map_verb_to_relation(token.lemma_, lang)
                    
                    # If no direct mapping, check preposition patterns
                    if not relation and prep:
                        for rel, preps in self.PREP_PATTERNS.items():
                            if prep.lower() in preps:
                                relation = rel
                                break
                    
                    # Special handling for common patterns
                    verb_lower = token.lemma_.lower()
                    if not relation:
                        # Handle founding/creation patterns
                        if verb_lower in ["found", "establish", "create", "start", "fundar", "créer", "gründen", "fondare"]:
                            relation = "found"
                        # Handle discovery patterns  
                        elif verb_lower in ["discover", "invent", "develop", "descubrir", "découvrir", "entdecken", "scoprire"]:
                            relation = "discover"
                        # Handle teaching patterns with preposition
                        elif verb_lower in ["teach", "lecture", "enseñar", "enseigner", "lehren", "insegnare"] and prep in ["at", "in"]:
                            relation = "teach_at"
                        # Handle generic "use" patterns
                        elif verb_lower in ["use", "utilize", "employ", "usar", "utiliser", "verwenden", "usare"]:
                            relation = "use"
                    
                    # Add triple even if not in target relations for broader coverage
                    if relation:
                        confidence = 0.8 if relation in self.TARGET_RELATIONS else 0.6
                        triples.append((subject, relation, obj, confidence))
                    else:
                        # Keep verb as-is for unmapped relations
                        triples.append((subject, token.lemma_, obj, 0.5))
                    
        return triples
    
    def _extract_noun_modifier_relations(self, doc: Doc) -> List[Tuple[str, str, str, float]]:
        """Extract relations from noun modifiers (e.g., 'CEO of Apple')."""
        triples = []
        
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ == "NOUN":
                # Pattern: NOUN + prep + NOUN (e.g., "CEO of Apple")
                pobjs = [t for t in token.children if t.dep_ in ("pobj", "obj")]
                if pobjs:
                    head_subtree = list(token.head.subtree)
                    head_text = " ".join(t.text for t in sorted(head_subtree, key=lambda x: x.i) if t.i < token.i)
                    
                    obj_subtree = list(pobjs[0].subtree)
                    obj_text = " ".join(t.text for t in sorted(obj_subtree, key=lambda x: x.i))
                    
                    # Map preposition to relation
                    if token.text.lower() in ["of", "at", "in", "for"]:
                        if "ceo" in head_text.lower() or "director" in head_text.lower():
                            triples.append((head_text, "works_at", obj_text, 0.7))
                        elif "brother" in head_text.lower() or "sister" in head_text.lower():
                            triples.append((head_text, "related_to", obj_text, 0.6))
                            
        return triples
    
    def _extract_copula_relations(self, doc: Doc) -> List[Tuple[str, str, str, float]]:
        """Extract relations from copula constructions (e.g., 'X is Y')."""
        triples = []
        
        for token in doc:
            if token.lemma_ == "be" and token.pos_ == "AUX":
                # Find subject
                subjects = [t for t in token.children if t.dep_ in ("nsubj", "nsubjpass")]
                # Find attribute or object
                attrs = [t for t in token.children if t.dep_ in ("attr", "acomp")]
                
                if subjects and attrs:
                    subj = subjects[0]
                    attr = attrs[0]
                    
                    subj_text = " ".join(t.text for t in sorted(subj.subtree, key=lambda x: x.i))
                    attr_text = " ".join(t.text for t in sorted(attr.subtree, key=lambda x: x.i))
                    
                    # Determine relation based on attribute
                    if attr.pos_ == "PROPN" or (attr.pos_ == "NOUN" and attr.ent_type_ == "PERSON"):
                        triples.append((subj_text, "also_known_as", attr_text, 0.7))
                    elif "year" in attr_text.lower() or "old" in attr_text.lower():
                        triples.append((subj_text, "age", attr_text, 0.6))
                    elif attr.pos_ == "NOUN":
                        triples.append((subj_text, "is", attr_text, 0.5))
                        
        return triples
    
    def extract(self, text: str, lang: Optional[str] = None) -> List[Tuple[str, str, str, float]]:
        """
        Extract relations from text. Returns list of (subject, relation, object, confidence).
        Compatible with ReLiK interface.
        """
        if not text:
            return []
            
        # Auto-detect language if not provided
        if not lang:
            lang = self._detect_language(text)
            
        # Get appropriate spaCy model
        nlp = self._get_nlp(lang)
        doc = nlp(text)
        
        # Extract using multiple strategies
        triples = []
        triples.extend(self._extract_from_dependencies(doc, lang))
        triples.extend(self._extract_noun_modifier_relations(doc))
        triples.extend(self._extract_copula_relations(doc))
        
        # Deduplicate while preserving highest confidence
        seen = {}
        for s, r, o, conf in triples:
            key = (s.lower(), r, o.lower())
            if key not in seen or seen[key][3] < conf:
                seen[key] = (s, r, o, conf)
                
        return list(seen.values())
    
    async def extract_async(self, text: str, lang: Optional[str] = None) -> List[Tuple[str, str, str, float]]:
        """Async wrapper for compatibility."""
        return self.extract(text, lang)


def test_extractor():
    """Test the lightweight extractor with various sentences."""
    extractor = LightweightRelationExtractor()
    
    test_cases = [
        ("My brother Tom lives in Portland and teaches at Reed College.", "en"),
        ("Apple was founded by Steve Jobs and Steve Wozniak in Cupertino in 1976.", "en"),
        ("Marie Curie discovered radium and polonium with her husband Pierre.", "en"),
        ("Barcelona es la capital de Cataluña y la segunda ciudad más grande de España.", "es"),
        ("The transformer architecture uses self-attention mechanisms.", "en"),
    ]
    
    for text, lang in test_cases:
        print(f"\nText: {text}")
        print(f"Language: {lang}")
        
        start = time.perf_counter()
        relations = extractor.extract(text, lang)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"Time: {elapsed:.1f}ms")
        print("Relations:")
        for s, r, o, conf in relations:
            print(f"  ({s}, {r}, {o}) conf={conf:.2f}")


if __name__ == "__main__":
    test_extractor()