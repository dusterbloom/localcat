"""
Fact Extractor - Focused component for extracting facts from text.

This component handles entity extraction and triple generation using
dependency parsing patterns. It follows the Single Responsibility Principle
by focusing solely on fact extraction from text.
"""

import os
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict
import statistics

from loguru import logger
import spacy
from spacy.tokens import Token

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
                nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
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


class FactExtractor:
    """
    Focused component for extracting factual triples from text using
    dependency parsing patterns. Handles entity recognition and relation extraction.
    """

    def __init__(self):
        self.metrics = defaultdict(list)
        self.max_metric_size = 1000

    def prewarm(self, lang: str = "en") -> None:
        """Pre-load NLP resources to avoid first-turn latency."""
        try:
            _load_nlp(lang)
        except Exception:
            pass

    def extract(self, text: str, lang: str = "en") -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """
        Extract entities and relations using USGS 27 dependency patterns.

        Returns:
            (entities, triples, negation_count, doc)
        """
        start_time = time.perf_counter()

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

        # Track performance
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.metrics['extraction_ms'].append(elapsed_ms)
        self._cleanup_metrics()

        return list(entities), triples, neg_count, doc

    def refine(self, text: str, triples: List[Tuple[str, str, str]], doc: Any) -> List[Tuple[str, str, str]]:
        """
        Refine extracted triples by filtering noise and improving quality.
        """
        # Filter noisy triples before storing/retrieving
        meaningful_triples = [t for t in triples if self._is_meaningful_fact(*t)]

        # Additional refinement logic can be added here
        return meaningful_triples

    def refine_entities(self, text: str, entities: List[str]) -> List[str]:
        """
        Refine entity list based on text context and extracted triples.
        """
        # Basic refinement - can be extended with more sophisticated logic
        return entities

    def _detect_language(self, text: str) -> str:
        """Detect language of input text."""
        if PYCLD3_AVAILABLE:
            try:
                detector = pycld3.get_cld3()
                result = detector.find_language(text)
                if result and result.language and result.is_reliable:
                    return result.language[:2].lower()
            except Exception:
                pass
        return "en"

    def _is_meaningful_fact(self, subject: str, relation: str, obj: str) -> bool:
        """
        Determine if a triple represents a meaningful fact worth storing.
        """
        # Skip trivial relations
        if relation in {"be", "have", "do", "make", "get", "take", "give"}:
            return False

        # Skip if subject or object is too short or generic
        if len(subject.strip()) < 2 or len(obj.strip()) < 2:
            return False

        # Skip pronouns as subjects (unless they're "you")
        if subject.lower() in {"i", "he", "she", "it", "we", "they"}:
            return False

        return True

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

        return entity_map

    # Dependency pattern extraction methods
    def _extract_subject(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract subject-verb-object triples from nsubj dependencies."""
        if token.i in entity_map and token.head.pos_ == "VERB":
            subject = entity_map[token.i]
            verb = token.head.lemma_
            # Look for direct object
            for child in token.head.children:
                if child.dep_ in {"dobj", "obj"} and child.i in entity_map:
                    obj = entity_map[child.i]
                    triples.append((subject, f"v:{verb}", obj))
                    break

    def _extract_object(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract direct object relations."""
        if token.i in entity_map and token.head.pos_ == "VERB":
            obj = entity_map[token.i]
            verb = token.head.lemma_
            # Look for subject
            for child in token.head.children:
                if child.dep_ in {"nsubj", "nsubjpass"} and child.i in entity_map:
                    subject = entity_map[child.i]
                    triples.append((subject, f"v:{verb}", obj))
                    break

    def _extract_indirect_object(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract indirect object relations."""
        if token.i in entity_map:
            obj = entity_map[token.i]
            # Look for verb and direct object
            verb = None
            direct_obj = None
            for child in token.head.children:
                if child.pos_ == "VERB":
                    verb = child.lemma_
                elif child.dep_ in {"dobj", "obj"} and child.i in entity_map:
                    direct_obj = entity_map[child.i]
            if verb and direct_obj:
                triples.append((obj, f"v:{verb}", direct_obj))

    def _extract_attribute(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract attribute relations (noun is attribute of subject)."""
        if token.i in entity_map:
            attr = entity_map[token.i]
            # Look for subject of the copula verb
            for child in token.head.children:
                if child.dep_ in {"nsubj", "nsubjpass"} and child.i in entity_map:
                    subject = entity_map[child.i]
                    triples.append((subject, "is", attr))
                    break

    def _extract_acomp(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract adjectival complement relations."""
        if token.pos_ == "ADJ":
            adj = token.lemma_
            # Look for subject
            for child in token.head.children:
                if child.dep_ in {"nsubj", "nsubjpass"} and child.i in entity_map:
                    subject = entity_map[child.i]
                    triples.append((subject, "is", adj))
                    break

    def _extract_amod(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract adjective modifier relations."""
        if token.pos_ == "ADJ" and token.head.i in entity_map:
            noun = entity_map[token.head.i]
            adj = token.lemma_
            triples.append((noun, "is", adj))

    def _extract_advmod(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract adverb modifier relations."""
        if token.pos_ == "ADV" and token.head.pos_ == "ADJ" and token.head.head.i in entity_map:
            noun = entity_map[token.head.head.i]
            adv = token.lemma_
            triples.append((noun, "is", f"{adv} {token.head.lemma_}"))

    def _extract_nummod(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract numeric modifier relations."""
        if token.pos_ == "NUM" and token.head.i in entity_map:
            noun = entity_map[token.head.i]
            num = token.text
            triples.append((noun, "has", num))

    def _extract_nmod(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract nominal modifier relations."""
        if token.i in entity_map and token.head.i in entity_map:
            modifier = entity_map[token.i]
            noun = entity_map[token.head.i]
            triples.append((noun, "has", modifier))

    def _extract_compound(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract compound noun relations."""
        if token.i in entity_map and token.head.i in entity_map:
            modifier = entity_map[token.i]
            head = entity_map[token.head.i]
            triples.append((head, "has", modifier))

    def _extract_possessive(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract possessive relations."""
        if token.i in entity_map and token.head.i in entity_map:
            possessor = entity_map[token.i]
            possessed = entity_map[token.head.i]
            triples.append((possessed, "belongs_to", possessor))

    def _extract_appos(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract appositional relations."""
        if token.i in entity_map and token.head.i in entity_map:
            entity1 = entity_map[token.head.i]
            entity2 = entity_map[token.i]
            triples.append((entity1, "is", entity2))

    def _extract_conj(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract conjunction relations."""
        if token.i in entity_map and token.head.i in entity_map:
            entity1 = entity_map[token.head.i]
            entity2 = entity_map[token.i]
            triples.append((entity1, "related_to", entity2))

    def _extract_prep(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract prepositional relations."""
        if token.text in entity_map and token.head.i in entity_map:
            prep = token.text
            obj = entity_map[token.i]
            # Look for subject
            for child in token.head.children:
                if child.dep_ in {"nsubj", "nsubjpass"} and child.i in entity_map:
                    subject = entity_map[child.i]
                    triples.append((subject, f"{prep}", obj))
                    break

    def _extract_acl(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract adjectival clause relations."""
        if token.pos_ == "VERB" and token.head.i in entity_map:
            noun = entity_map[token.head.i]
            verb = token.lemma_
            triples.append((noun, f"v:{verb}", "something"))

    def _extract_advcl(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract adverbial clause relations."""
        if token.pos_ == "VERB":
            verb = token.lemma_
            # Look for subject of adverbial clause
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"} and child.i in entity_map:
                    subject = entity_map[child.i]
                    triples.append((subject, f"v:{verb}", "something"))
                    break

    def _extract_ccomp(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract clausal complement relations."""
        if token.pos_ == "VERB":
            verb = token.lemma_
            # Look for subject
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"} and child.i in entity_map:
                    subject = entity_map[child.i]
                    triples.append((subject, f"v:{verb}", "something"))
                    break

    def _extract_csubj(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract clausal subject relations."""
        if token.pos_ == "VERB" and token.head.i in entity_map:
            verb = token.lemma_
            subject_verb = token.lemma_
            triples.append(("someone", f"v:{verb}", "something"))

    def _extract_xcomp(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract open clausal complement relations."""
        if token.pos_ == "VERB":
            verb = token.lemma_
            # Look for subject
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"} and child.i in entity_map:
                    subject = entity_map[child.i]
                    triples.append((subject, f"v:{verb}", "something"))
                    break

    def _extract_agent(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract agent relations (passive voice)."""
        if token.i in entity_map:
            agent = entity_map[token.i]
            # Look for verb
            for child in token.head.children:
                if child.pos_ == "VERB":
                    verb = child.lemma_
                    triples.append((agent, f"v:{verb}", "something"))
                    break

    def _extract_oprd(self, token: Token, entity_map: Dict[int, str], triples: List[Tuple[str, str, str]], entities: Set[str]):
        """Extract object predicate relations."""
        if token.i in entity_map and token.head.i in entity_map:
            entity1 = entity_map[token.head.i]
            entity2 = entity_map[token.i]
            triples.append((entity1, "is", entity2))

    def _cleanup_metrics(self):
        """Clean up old metrics to prevent memory bloat."""
        for key in self.metrics:
            if len(self.metrics[key]) > self.max_metric_size:
                self.metrics[key] = self.metrics[key][-self.max_metric_size:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        result = {}
        for key, values in self.metrics.items():
            if values:
                result[key] = {
                    'mean': statistics.mean(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                    'count': len(values)
                }
            else:
                result[key] = {'count': 0}
        return result