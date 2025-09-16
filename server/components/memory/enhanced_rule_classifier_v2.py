"""
Enhanced Rule-Based Intent Classifier V2
Universal Dependencies-based for multilingual support with <1ms inference
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from loguru import logger

# FORCE SPACY DISABLED FOR MAXIMUM SPEED (<1ms)
SPACY_AVAILABLE = False
Doc = None

class IntentType(Enum):
    """Core intent types that matter most"""
    QUESTION = "question"
    FACT = "fact"
    GREETING = "greeting"
    ACKNOWLEDGMENT = "acknowledgment"
    REACTION = "reaction"
    CORRECTION = "correction"
    COMMAND = "command"
    REQUEST = "request"
    FAREWELL = "farewell"
    UNKNOWN = "unknown"

@dataclass
class IntentClassification:
    """Simplified classification result"""
    primary_intent: IntentType
    confidence: float
    requires_memory: bool
    requires_retrieval: bool
    metadata: Dict

class EnhancedRuleClassifierV2:
    """
    Universal Dependencies-based classifier for multilingual support.
    Uses spaCy's POS tags and dependency parsing for language-agnostic patterns.
    Falls back to English patterns when spaCy is unavailable.
    """

    def __init__(self, model_name: str = None):
        self._nlp = None
        self._model_name = model_name

        # Language-agnostic patterns using Universal POS tags
        # These work across languages!

        # Greeting patterns (multilingual)
        self.greeting_lemmas = {
            'hello', 'hi', 'hey', 'howdy' 'greetings', 'hola', 'bonjour', 'ciao',
            'salut', 'hallo', 'aloha', 'namaste', 'konnichiwa', 'calimera'
            'night', 'morning', 'afternoon', 'evening'  # Time-based greetings
        }

        # Acknowledgment lemmas (multilingual)
        self.ack_lemmas = {
            'ok', 'okay', 'yes', 'yeah', 'yep', 'sure', 'right', 'correct',
            'understand', 'got', 'thanks', 'thank', 'gracias', 'merci',
            'danke', 'grazie', 'arigato', 'roger', 'copy', 'acknowledge'
        }

        # Reaction/emotion lemmas
        self.reaction_lemmas = {
            'wow', 'amazing', 'awesome', 'great', 'excellent', 'wonderful',
            'terrible', 'awful', 'horrible', 'interesting', 'incredible',
            'fantastic', 'oh', 'ah', 'hmm', 'really', 'seriously'
        }

        # Command verbs (imperative mood, cross-linguistic)
        self.command_lemmas = {
            'show', 'display', 'list', 'find', 'search', 'get', 'give',
            'calculate', 'create', 'make', 'delete', 'remove', 'update',
            'change', 'tell', 'explain', 'help', 'describe', 'write'
        }

        # Farewell lemmas
        self.farewell_lemmas = {
            'bye', 'goodbye', 'farewell', 'later', 'adios', 'ciao',
            'auf wiedersehen', 'sayonara', 'see', 'night'
        }

        # Correction markers (often language-specific but patterns are similar)
        self.correction_lemmas = {
            'no', 'not', 'actually', 'mean', 'correction', 'wrong',
            'mistake', 'sorry', 'wait', 'rather', 'instead'
        }

        # Request patterns using auxiliaries (language-agnostic)
        self.request_aux = {'can', 'could', 'would', 'will', 'may', 'might', 'should'}

        # Temporal markers (for fact detection)
        self.temporal_pos = {'DATE', 'TIME'}  # Universal POS tags for dates/times

    def _get_nlp(self, lang: str = "en"):
        """Get or create spaCy model (cached)"""
        if not SPACY_AVAILABLE:
            return None

        if self._nlp is None:
            try:
                # Try to load the specified model or language-specific model
                if self._model_name:
                    self._nlp = spacy.load(self._model_name, disable=["ner", "textcat", "lemmatizer"])
                else:
                    # Map language codes to lightweight models
                    model_map = {
                        "en": "en_core_web_sm",
                        "es": "es_core_news_sm",
                        "fr": "fr_core_news_sm",
                        "de": "de_core_news_sm",
                        "it": "it_core_news_sm",
                        "pt": "pt_core_news_sm",
                        "nl": "nl_core_news_sm",
                        "multi": "xx_ent_wiki_sm",  # Multilingual model
                    }
                    model = model_map.get(lang, "en_core_web_sm")
                    self._nlp = spacy.load(model, disable=["ner", "textcat"])

                # Pre-compile for speed
                self._nlp.max_length = 1000  # Short texts only

            except Exception as e:
                logger.debug(f"Could not load spaCy model: {e}")
                return None

        return self._nlp

    def classify(self, text: str, context: Optional[List[str]] = None, lang: str = "en") -> IntentClassification:
        """
        Classify using Universal Dependencies for language-agnostic patterns.
        Falls back to string patterns when spaCy unavailable.
        """
        if not text:
            return self._create_result(IntentType.UNKNOWN, 0.0)

        text_lower = text.lower().strip()

        # Try spaCy-based classification first (language-agnostic)
        nlp = self._get_nlp(lang)
        if nlp:
            return self._classify_with_ud(text, text_lower, nlp)
        else:
            # Fallback to pattern-based classification
            return self._classify_with_patterns(text, text_lower)

    def _classify_with_ud(self, text: str, text_lower: str, nlp) -> IntentClassification:
        """Classify using Universal Dependencies (fast, multilingual)"""

        # Process with spaCy (this is the only slow part, ~5-10ms)
        doc = nlp(text_lower)

        # Quick checks using token attributes (very fast)
        tokens = list(doc)
        if not tokens:
            return self._create_result(IntentType.UNKNOWN, 0.0)

        first_token = tokens[0]
        lemmas = {token.lemma_ for token in tokens}
        pos_tags = {token.pos_ for token in tokens}
        dep_labels = {token.dep_ for token in tokens}

        # === PRIORITY 1: Check strong patterns using UD ===

        # Greeting: Check lemmas and common greeting patterns
        if (first_token.lemma_ in self.greeting_lemmas or
            lemmas & self.greeting_lemmas):
            return self._create_result(IntentType.GREETING, 0.9, retrieve=False, store=False)

        # Farewell: Check lemmas
        if lemmas & self.farewell_lemmas:
            return self._create_result(IntentType.FAREWELL, 0.9, retrieve=False, store=False)

        # Acknowledgment: Short utterance with ack lemmas
        if len(tokens) <= 4 and lemmas & self.ack_lemmas:
            return self._create_result(IntentType.ACKNOWLEDGMENT, 0.85, retrieve=False, store=False)

        # Additional greeting patterns: time-based greetings
        if (lemmas & {'good', 'morning', 'afternoon', 'evening', 'night', 'day'} and
            len(tokens) <= 3):
            return self._create_result(IntentType.GREETING, 0.9, retrieve=False, store=False)

        # Reaction: Short utterance with emotion/reaction lemmas + exclamation
        if (len(tokens) <= 5 and lemmas & self.reaction_lemmas and
            ('INTJ' in pos_tags or '!' in text)):
            return self._create_result(IntentType.REACTION, 0.85, retrieve=False, store=False)

        # Correction: Negation at start or correction lemmas + temporal corrections
        correction_patterns = [
            r'\bno\b.*\bi?\s*(mean|meant|said)',
            r'\bactually\b.*(not|wrong|different)',
            r'\bwait\b.*(no|i?\s*(mean|meant))',
            r'\bsorry\b.*(wrong|mistake)',
            r'\bi\s*(think|believe)\s+(you|it)\s*(have|got)\s*(wrong|incorrect)',
            r'\bcorrection\b', r'\bcorrect\s+(that|me)',
            r'\bno(t\s+(really|actually))?\b.*\b(is|are|was|were)\b',  # Temporal corrections like "No, I was there since 2020"
        ]

        if any(re.search(pat, text_lower, re.IGNORECASE) for pat in correction_patterns):
            return self._create_result(IntentType.CORRECTION, 0.9, retrieve=True, store=True)

        # Legacy negation check as fallback
        if (first_token.lemma_ in self.correction_lemmas or
            (first_token.dep_ == 'neg' or first_token.pos_ == 'PART')):
            # Check for negation + assertion pattern
            has_verb = 'VERB' in pos_tags or 'AUX' in pos_tags
            if has_verb:
                return self._create_result(IntentType.CORRECTION, 0.9, retrieve=True, store=True)

        # === PRIORITY 2: Commands and Requests using UD ===

        # Command: Imperative mood (verb at start without subject)
        if first_token.pos_ == 'VERB' and 'nsubj' not in dep_labels:
            # Imperative sentence pattern
            return self._create_result(IntentType.COMMAND, 0.85, retrieve=True, store=False)

        # Command: Command verb lemmas at start
        if first_token.lemma_ in self.command_lemmas:
            return self._create_result(IntentType.COMMAND, 0.85, retrieve=True, store=False)

        # Request: Modal auxiliary + verb pattern
        has_modal = bool(lemmas & self.request_aux)
        has_verb = 'VERB' in pos_tags
        if has_modal and has_verb:
            # Check for question pattern (aux before subject)
            for i, token in enumerate(tokens[:-1]):
                if token.pos_ == 'AUX' and tokens[i+1].dep_ == 'nsubj':
                    return self._create_result(IntentType.REQUEST, 0.85, retrieve=True, store=False)

        # === PRIORITY 3: Questions vs Statements using UD ===

        # Question detection using Universal Dependencies
        is_question = False

        # Method 1: Question mark (universal)
        if '?' in text:
            is_question = True

        # Method 2: WH-word detection (language-agnostic using POS tags)
        elif first_token.tag_ in {'WDT', 'WP', 'WP$', 'WRB'}:  # WH-determiners, pronouns, adverbs
            is_question = True

        # Method 3: Auxiliary inversion (aux before subject)
        else:
            for token in tokens:
                if token.dep_ == 'aux' and token.i == 0:  # Auxiliary at start
                    is_question = True
                    break

        if is_question:
            return self._create_result(IntentType.QUESTION, 0.8, retrieve=True, store=False)

        # === PRIORITY 4: Facts and Statements ===

        # Temporal fact: Has DATE/TIME entities + explicit temporal queries
        has_temporal = bool(pos_tags & self.temporal_pos)

        # Temporal queries: When/How long/Since patterns
        temporal_query_patterns = [
            r'\b(when|how long|since|until|before|after)\b.*\?(?:\s|$)',
            r'\bwhat\s+(time|date|day|year)\b',
            r'\b(is|are|was|were)\s+(it|he|she|they|this|that)\s+(since|from|starting|beginning)\b',
        ]

        is_temporal_query = any(re.search(pat, text_lower, re.IGNORECASE) for pat in temporal_query_patterns)

        # Factual statement: Has subject + verb/copula + object/complement
        has_subject = 'nsubj' in dep_labels or 'nsubjpass' in dep_labels
        has_verb = 'VERB' in pos_tags or 'AUX' in pos_tags
        has_object = bool({'obj', 'iobj', 'attr', 'acomp', 'xcomp'} & dep_labels)

        # Personal fact: Possessive pronouns
        has_possessive = 'PRON' in pos_tags and any(t.tag_ == 'PRP$' for t in tokens)

        # Enhanced fact detection with temporal
        if (has_temporal and (has_subject and has_verb)) or is_temporal_query or has_possessive:
            # Temporal facts need retrieval for context (e.g., "since 2020" links to employment)
            requires_retrieve = is_temporal_query or has_temporal
            return self._create_result(IntentType.FACT, 0.8 if has_temporal else 0.75, retrieve=requires_retrieve, store=True)

        # Basic fact detection: Simple declarative sentence structure
        if has_subject and has_verb and (has_object or len(tokens) > 3):
            return self._create_result(IntentType.FACT, 0.7, retrieve=False, store=True)

        # === DEFAULT ===
        return self._create_result(IntentType.UNKNOWN, 0.3, retrieve=True, store=False)

    def _classify_with_patterns(self, text: str, text_lower: str) -> IntentClassification:
        """Fallback pattern-based classification when spaCy unavailable"""

        words = text_lower.split()
        first_word = words[0] if words else ""
        words_set = set(words)

        # Quick pattern checks (similar to original V2 but simplified)

        # Greetings
        if (first_word in {'hello', 'hi', 'hey', 'greetings'} or
            'how are you' in text_lower or
            any(greeting in text_lower for greeting in ['good morning', 'good afternoon', 'good evening'])):
            return self._create_result(IntentType.GREETING, 0.9, retrieve=False, store=False)

        # Farewells
        if any(w in text_lower for w in ['bye', 'goodbye', 'see you', 'farewell']):
            return self._create_result(IntentType.FAREWELL, 0.9, retrieve=False, store=False)

        # Acknowledgments
        if len(words) <= 4 and words_set & {'ok', 'okay', 'thanks', 'thank', 'got', 'understood'}:
            return self._create_result(IntentType.ACKNOWLEDGMENT, 0.85, retrieve=False, store=False)

        # Reactions
        if len(words) <= 5 and '!' in text and words_set & {'wow', 'amazing', 'great', 'oh'}:
            return self._create_result(IntentType.REACTION, 0.85, retrieve=False, store=False)

        # Corrections
        if text_lower.startswith(('no,', 'actually', 'i mean', 'correction')):
            return self._create_result(IntentType.CORRECTION, 0.9, retrieve=True, store=True)

        # Commands
        if first_word in {'show', 'display', 'list', 'find', 'tell', 'explain'}:
            return self._create_result(IntentType.COMMAND, 0.85, retrieve=True, store=False)

        # Requests
        if any(p in text_lower for p in ['can you', 'could you', 'would you', 'please']):
            return self._create_result(IntentType.REQUEST, 0.85, retrieve=True, store=False)

        # Facts first (higher priority for first-person statements)
        if (first_word in {'i', 'my', 'we', 'our'} or
            any(w in words_set for w in {'is', 'are', 'was', 'were', 'have', 'has', 'am'})):
            # Additional check to avoid false positives on questions starting with these words
            if not text.endswith('?') and first_word not in {'is', 'are', 'do', 'does', 'did', 'can', 'could', 'will', 'would'}:
                return self._create_result(IntentType.FACT, 0.75, retrieve=False, store=True)

        # Questions
        if text.endswith('?') or first_word in {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'do', 'does', 'did', 'can', 'could', 'will', 'would'}:
            return self._create_result(IntentType.QUESTION, 0.8, retrieve=True, store=False)

        # Unknown
        return self._create_result(IntentType.UNKNOWN, 0.3, retrieve=True, store=False)

    def _create_result(self, intent: IntentType, confidence: float,
                       retrieve: bool = None, store: bool = None) -> IntentClassification:
        """Create result with explicit retrieval/storage decisions"""

        # Use provided values or defaults based on intent
        if retrieve is None:
            retrieve = intent in {IntentType.QUESTION, IntentType.COMMAND,
                                 IntentType.REQUEST, IntentType.CORRECTION,
                                 IntentType.UNKNOWN}

        if store is None:
            store = intent in {IntentType.FACT, IntentType.CORRECTION}

        return IntentClassification(
            primary_intent=intent,
            confidence=min(confidence, 0.95),
            requires_memory=store,
            requires_retrieval=retrieve,
            metadata={'method': 'enhanced_rules_v2', 'version': '2.0'}
        )


# Test the classifier
if __name__ == "__main__":
    classifier = EnhancedRuleClassifierV2()

    test_cases = [
        ("What is the capital of France?", True, False),
        ("My dog's name is Potola", False, True),
        ("Hello, how are you?", False, False),
        ("OK, got it", False, False),
        ("No, actually her name is Sarah", True, True),
        ("Yesterday I went to the park", False, True),
        ("Wow, that's amazing!", False, False),
        ("Can you help me with Python?", True, False),
        ("What if robots took over?", True, False),
        ("Remember when we discussed the project?", True, False),
        ("The meeting is at 3pm tomorrow", False, True),
        ("Thanks for your help", False, False),
        ("Show me the latest results", True, False),
        ("I think we should try a different approach", False, True),
        ("See you later!", False, False),
    ]

    # Test multilingual if spaCy available
    multilingual_tests = [
        ("Bonjour, comment allez-vous?", False, False),  # French greeting
        ("Mi perro se llama Max", False, True),  # Spanish fact
        ("Danke schön", False, False),  # German thanks
    ]

    print("Enhanced Rule-Based Classifier V2 Test")
    print("=" * 50)

    correct = 0
    for text, expect_retrieve, expect_store in test_cases:
        result = classifier.classify(text)

        retrieve_correct = result.requires_retrieval == expect_retrieve
        store_correct = result.requires_memory == expect_store
        both_correct = retrieve_correct and store_correct

        if both_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        print(f"{status} '{text[:40]}...'")
        print(f"   Intent: {result.primary_intent.value:15s} | R:{result.requires_retrieval} (want {expect_retrieve}) | S:{result.requires_memory} (want {expect_store})")

    accuracy = (correct / len(test_cases)) * 100
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{len(test_cases)} correct)")

    if SPACY_AVAILABLE and accuracy >= 70:
        print("\n📍 Testing multilingual support...")
        for text, expect_retrieve, expect_store in multilingual_tests:
            result = classifier.classify(text, lang="multi")
            print(f"   '{text}' → {result.primary_intent.value}")

    if accuracy >= 70:
        print("\n✅ Classifier achieves target accuracy!")
    else:
        print("\n⚠️ Classifier needs tuning")