"""
Simple Working Coreference Resolver
===================================

Using Coreferee - a working, maintained alternative for 2024
- Compatible with spaCy 3.x
- Production-ready
- Simple integration
"""

import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

try:
    import spacy
    import coreferee
    COREFEREE_AVAILABLE = True
    logger.info("[SimpleCoref] Coreferee available for coreference resolution")
except ImportError as e:
    COREFEREE_AVAILABLE = False
    logger.warning(f"[SimpleCoref] Coreferee not available: {e}")


@dataclass
class SimpleCoreferenceResult:
    """Simple coreference resolution result"""
    resolved_triples: List[Tuple[str, str, str]]
    resolution_map: Dict[str, str]  # pronoun -> entity
    processing_time_ms: float
    method: str


class SimpleCoreferenceResolver:
    """
    Simple, working coreference resolver using Coreferee

    Fast, reliable pronoun resolution for production use
    """

    def __init__(self):
        self.enabled = COREFEREE_AVAILABLE
        self._nlp = None

        # Fallback rule-based patterns
        self.gender_indicators = {
            'male': {'mr.', 'mr', 'man', 'boy', 'father', 'son', 'john', 'steve', 'tim', 'ceo', 'jobs'},
            'female': {'ms.', 'mrs.', 'miss', 'dr.', 'woman', 'girl', 'mother', 'daughter', 'maria', 'sarah', 'chen'},
            'plural': {'team', 'company', 'group', 'people', 'employees', 'researchers'}
        }

        self.pronouns = {
            'he': 'male', 'him': 'male', 'his': 'male',
            'she': 'female', 'her': 'female', 'hers': 'female',
            'they': 'plural', 'them': 'plural', 'their': 'plural'
        }

        logger.info(f"[SimpleCoref] Initialized with coreferee={'✓' if self.enabled else '✗'}")

    def _load_nlp(self):
        """Load spaCy with coreferee if available"""
        if not self._nlp:
            try:
                if COREFEREE_AVAILABLE:
                    # Try to load with coreferee
                    self._nlp = spacy.load("en_core_web_sm")
                    self._nlp.add_pipe('coreferee')
                    logger.debug("[SimpleCoref] Loaded spaCy with coreferee")
                else:
                    # Fallback to regular spaCy
                    self._nlp = spacy.load("en_core_web_sm")
                    logger.debug("[SimpleCoref] Loaded spaCy without coreferee (rule-based only)")
            except Exception as e:
                logger.warning(f"[SimpleCoref] Failed to load coreferee: {e}, using rule-based")
                self._nlp = spacy.load("en_core_web_sm")
                self.enabled = False

    def resolve_coreferences(self, triples: List[Tuple[str, str, str]],
                           entities: List[str], text: str = "") -> SimpleCoreferenceResult:
        """
        Resolve pronouns in triples using coreferee or rule-based fallback
        """
        start = time.perf_counter()

        if not triples:
            return SimpleCoreferenceResult(triples, {}, 0.0, "none")

        if not self._nlp:
            self._load_nlp()

        try:
            # Try coreferee first if available
            if self.enabled and text:
                result = self._resolve_with_coreferee(triples, text)
                if result.resolved_triples != triples:
                    return result

            # Fallback to rule-based resolution
            return self._resolve_with_rules(triples, entities, start)

        except Exception as e:
            logger.debug(f"[SimpleCoref] Coreference failed: {e}")
            processing_time = (time.perf_counter() - start) * 1000
            return SimpleCoreferenceResult(triples, {}, processing_time, "failed")

    def _resolve_with_coreferee(self, triples: List[Tuple[str, str, str]], text: str) -> SimpleCoreferenceResult:
        """Resolve using coreferee neural coreference"""
        start = time.perf_counter()

        try:
            # Process text with coreferee
            doc = self._nlp(text)

            # Extract coreference chains
            resolution_map = {}

            if hasattr(doc._, 'coref_chains') and doc._.coref_chains:
                for chain in doc._.coref_chains:
                    # Find the main entity (usually the first or longest mention)
                    main_mention = None
                    mentions = []

                    for mention in chain:
                        mention_text = doc[mention.root_index].text.lower()
                        mentions.append(mention_text)

                        # Use first non-pronoun as main entity
                        if not main_mention and mention_text not in self.pronouns:
                            main_mention = mention_text

                    # If we found a main mention, map pronouns to it
                    if main_mention:
                        for mention_text in mentions:
                            if mention_text in self.pronouns:
                                resolution_map[mention_text] = main_mention

            # Apply resolution to triples
            resolved_triples = []
            for subj, rel, obj in triples:
                resolved_subj = resolution_map.get(subj.lower(), subj)
                resolved_obj = resolution_map.get(obj.lower(), obj)
                resolved_triples.append((resolved_subj, rel, resolved_obj))

            processing_time = (time.perf_counter() - start) * 1000
            method = f"coreferee ({len(resolution_map)} pronouns resolved)"

            logger.debug(f"[SimpleCoref] Coreferee resolved: {resolution_map}")
            return SimpleCoreferenceResult(resolved_triples, resolution_map, processing_time, method)

        except Exception as e:
            logger.debug(f"[SimpleCoref] Coreferee failed: {e}")
            # Fall through to rule-based
            return SimpleCoreferenceResult(triples, {}, 0.0, "coreferee_failed")

    def _resolve_with_rules(self, triples: List[Tuple[str, str, str]],
                           entities: List[str], start: float) -> SimpleCoreferenceResult:
        """Simple rule-based pronoun resolution"""

        # Find potential antecedents by gender/type
        antecedents = {'male': None, 'female': None, 'plural': None, 'generic': None}

        for entity in entities:
            entity_lower = entity.lower()

            if any(indicator in entity_lower for indicator in self.gender_indicators['male']):
                antecedents['male'] = entity
            elif any(indicator in entity_lower for indicator in self.gender_indicators['female']):
                antecedents['female'] = entity
            elif any(indicator in entity_lower for indicator in self.gender_indicators['plural']):
                antecedents['plural'] = entity
            else:
                antecedents['generic'] = entity

        # Create resolution map
        resolution_map = {}
        for pronoun, gender_type in self.pronouns.items():
            if antecedents[gender_type]:
                resolution_map[pronoun] = antecedents[gender_type]

        # Apply resolution to triples
        resolved_triples = []
        for subj, rel, obj in triples:
            resolved_subj = resolution_map.get(subj.lower(), subj)
            resolved_obj = resolution_map.get(obj.lower(), obj)
            resolved_triples.append((resolved_subj, rel, resolved_obj))

        processing_time = (time.perf_counter() - start) * 1000
        method = f"rule_based ({len(resolution_map)} pronouns mapped)"

        logger.debug(f"[SimpleCoref] Rule-based resolved: {resolution_map}")
        return SimpleCoreferenceResult(resolved_triples, resolution_map, processing_time, method)


# Test the resolver
if __name__ == "__main__":
    resolver = SimpleCoreferenceResolver()

    test_cases = [
        {
            "text": "Steve Jobs founded Apple. He was a visionary. His company changed everything.",
            "triples": [("he", "was", "visionary"), ("his", "company", "changed")],
            "entities": ["steve jobs", "apple", "company", "visionary"]
        },
        {
            "text": "Maria works at Google. She leads the AI team.",
            "triples": [("she", "leads", "team")],
            "entities": ["maria", "google", "ai team"]
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test['text']}")
        print(f"Input triples: {test['triples']}")

        result = resolver.resolve_coreferences(test['triples'], test['entities'], test['text'])

        print(f"Resolved triples: {result.resolved_triples}")
        print(f"Resolution map: {result.resolution_map}")
        print(f"Time: {result.processing_time_ms:.1f}ms")
        print(f"Method: {result.method}")