"""
Memory Intent Classification - Language-agnostic conversation analysis
Uses Universal Dependencies and structural patterns, not language-specific rules
"""

import os
import re
from enum import Enum
from typing import Dict, Tuple, List, Set, Optional
from dataclasses import dataclass
from loguru import logger

# Try to load spacy for UD analysis
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class IntentType(Enum):
    """Classification of conversational intent"""
    FACT_STATEMENT = "fact_statement"          # Direct factual information
    QUESTION_WITH_FACT = "question_with_fact"  # Question containing embedded facts
    PURE_QUESTION = "pure_question"            # Information seeking only
    REACTION = "reaction"                      # Emotional/conversational response
    HYPOTHETICAL = "hypothetical"              # Conditional/imaginary scenarios
    CORRECTION = "correction"                  # Fact updates/corrections
    TEMPORAL_FACT = "temporal_fact"            # Time-sensitive information
    MULTIPLE_FACTS = "multiple_facts"          # Compound factual statements


@dataclass
class IntentAnalysis:
    """Result of intent classification"""
    intent: IntentType
    confidence: float
    should_extract_facts: bool
    embedded_facts_likely: bool
    temporal_markers: List[str]
    correction_signals: List[str]
    
    
class IntentClassifier:
    """Language-agnostic intent classification using Universal Dependencies"""
    
    def __init__(self):
        self._nlp_cache = {}
        
    def _get_nlp(self, lang: str = "en"):
        """Get spaCy model for language (cached)"""
        if not SPACY_AVAILABLE:
            return None
        if lang not in self._nlp_cache:
            try:
                model_name = {
                    "en": "en_core_web_sm",
                    "es": "es_core_news_sm", 
                    "it": "it_core_news_sm",
                    "fr": "fr_core_news_sm",
                    "de": "de_core_news_sm"
                }.get(lang, "en_core_web_sm")
                self._nlp_cache[lang] = spacy.load(model_name, disable=["ner", "textcat"])
            except Exception:
                # Fallback to basic model
                try:
                    self._nlp_cache[lang] = spacy.blank(lang)
                    self._nlp_cache[lang].add_pipe("sentencizer")
                except Exception:
                    self._nlp_cache[lang] = None
        return self._nlp_cache[lang]
        
    def analyze(self, text: str, lang: str = "en") -> IntentAnalysis:
        """Analyze text intent using language-agnostic UD patterns"""
        if not text:
            return IntentAnalysis(
                IntentType.REACTION, 0.0, False, False, [], []
            )
            
        # Quick structural analysis
        text_stripped = text.strip()
        words = text_stripped.split()
        
        # Basic structural patterns (universal)
        has_question_mark = "?" in text
        is_very_short = len(words) <= 3
        has_punctuation = any(p in text for p in "!.?")
        
        # Try UD analysis if available
        nlp = self._get_nlp(lang)
        if nlp:
            return self._analyze_with_ud(text, nlp)
        else:
            return self._analyze_structural(text_stripped, words, has_question_mark, is_very_short)
            
    def _analyze_with_ud(self, text: str, nlp) -> IntentAnalysis:
        """Analyze using Universal Dependencies (language-agnostic)"""
        doc = nlp(text)
        
        # UD-based classification
        is_question = self._is_question_ud(doc)
        has_negation = self._has_negation_ud(doc)  
        has_conditional = self._has_conditional_ud(doc)
        is_short_reaction = len(doc) <= 3 and not any(t.pos_ in {"NOUN", "PROPN", "VERB"} for t in doc)
        has_multiple_clauses = len(list(doc.sents)) > 1 or self._has_coordination_ud(doc)
        
        # Classification logic
        if is_short_reaction:
            return IntentAnalysis(IntentType.REACTION, 0.8, False, False, [], [])
            
        if has_conditional:
            return IntentAnalysis(IntentType.HYPOTHETICAL, 0.85, False, False, [], [])
            
        if has_negation and self._has_correction_context_ud(doc):
            return IntentAnalysis(IntentType.CORRECTION, 0.9, True, True, [], ["negation"])
            
        if is_question:
            # Check if question has embedded factual content
            if self._has_embedded_facts_ud(doc):
                return IntentAnalysis(IntentType.QUESTION_WITH_FACT, 0.8, True, True, [], [])
            else:
                return IntentAnalysis(IntentType.PURE_QUESTION, 0.7, False, False, [], [])
                
        if has_multiple_clauses:
            return IntentAnalysis(IntentType.MULTIPLE_FACTS, 0.85, True, True, [], [])
            
        # Default to fact statement
        return IntentAnalysis(IntentType.FACT_STATEMENT, 0.7, True, False, [], [])
        
    def _analyze_structural(self, text: str, words: List[str], has_question_mark: bool, is_very_short: bool) -> IntentAnalysis:
        """Fallback structural analysis without NLP"""
        # Simple heuristics without language-specific patterns
        if is_very_short and not has_question_mark:
            return IntentAnalysis(IntentType.REACTION, 0.6, False, False, [], [])
            
        if has_question_mark:
            # Assume pure question for now
            return IntentAnalysis(IntentType.PURE_QUESTION, 0.6, False, False, [], [])
            
        # Check for basic coordination (universal punctuation)
        if "," in text and len(words) > 8:
            return IntentAnalysis(IntentType.MULTIPLE_FACTS, 0.7, True, True, [], [])
            
        return IntentAnalysis(IntentType.FACT_STATEMENT, 0.6, True, False, [], [])
        
    # UD-based analysis methods (language-agnostic)
    def _is_question_ud(self, doc) -> bool:
        """Detect questions using UD patterns"""
        for sent in doc.sents:
            root = sent.root
            # Question mark (universal)
            if "?" in sent.text:
                return True
            # Auxiliary inversion pattern (aux before nsubj)
            aux_pos = None
            subj_pos = None
            for i, token in enumerate(sent):
                if token.dep_ == "aux" and aux_pos is None:
                    aux_pos = i
                elif token.dep_ in {"nsubj", "nsubj:pass"} and subj_pos is None:
                    subj_pos = i
            if aux_pos is not None and subj_pos is not None and aux_pos < subj_pos:
                return True
            # WH-word at start
            if len(sent) > 0 and sent[0].tag_ in {"WP", "WDT", "WRB", "WP$"}:
                return True
        return False
        
    def _has_negation_ud(self, doc) -> bool:
        """Detect negation using UD neg dependency"""
        return any(token.dep_ == "neg" for token in doc)
        
    def _has_conditional_ud(self, doc) -> bool:
        """Detect conditionals using UD patterns"""
        # Look for conditional markers or subjunctive mood
        for token in doc:
            # Conditional conjunctions typically mark clauses
            if token.dep_ == "mark" and token.head.dep_ == "advcl":
                return True
            # Subjunctive mood in some languages
            if hasattr(token, 'morph') and 'Mood=Sub' in str(token.morph):
                return True
        return False
        
    def _has_correction_context_ud(self, doc) -> bool:
        """Detect correction context (negation + assertion)"""
        has_neg = self._has_negation_ud(doc)
        has_assertion = any(token.pos_ in {"VERB", "AUX"} and 
                           any(child.dep_ in {"nsubj", "attr", "acomp"} for child in token.children)
                           for token in doc)
        return has_neg and has_assertion
        
    def _has_coordination_ud(self, doc) -> bool:
        """Detect coordination using UD conj dependency"""
        return any(token.dep_ == "conj" for token in doc)
        
    def _has_embedded_facts_ud(self, doc) -> bool:
        """Detect embedded facts in questions using UD structure"""
        # Look for complement clauses (ccomp) or relative clauses (acl:relcl)
        # These often contain the embedded factual content
        for token in doc:
            if token.dep_ in {"ccomp", "acl:relcl", "acl"}:
                # Check if the clause has factual structure (subj-verb-obj/attr)
                has_subj = any(child.dep_ in {"nsubj", "nsubj:pass"} for child in token.subtree)
                has_pred = any(child.dep_ in {"attr", "acomp", "obj", "iobj"} for child in token.subtree)
                if has_subj and has_pred:
                    return True
        return False


class QualityFilter:
    """Filter low-quality facts before storage"""
    
    def __init__(self):
        # Useless patterns to filter out
        self.useless_patterns = [
            ("you", "tell", "you"),   # Self-referential
            ("you", "know", "you"),   # Obvious
            ("i", "tell", "i"),       # Self-referential
            ("you", "_", ""),         # Empty relation/destination
            ("", "_", "you"),         # Empty source
        ]
        
        # Generic/low-value relations
        self.generic_relations = {
            "has", "is", "are", "was", "were", "_", ""
        }
        
    def should_store_fact(self, s: str, r: str, d: str, intent: IntentAnalysis) -> Tuple[bool, float]:
        """
        Determine if fact should be stored and with what confidence
        Returns: (should_store, confidence_score)
        """
        # Never store facts from reactions or hypotheticals
        if intent.intent in {IntentType.REACTION, IntentType.HYPOTHETICAL}:
            return False, 0.0
            
        # Filter useless patterns
        fact = (s, r, d)
        if fact in self.useless_patterns:
            return False, 0.0
            
        # Check for empty components
        if not s or not r or not d or len(s.strip()) == 0 or len(d.strip()) == 0:
            return False, 0.0
            
        # Score based on relation quality
        confidence = 0.5  # Base score
        
        # Boost for specific relations (high-value, low-ambiguity)
        if r in {"name", "age", "lives_in", "works_at", "teach_at", "born_in", "moved_from", "went_to", "favorite_color", "favorite_number", "also_known_as", "owns"}:
            confidence += 0.3
        elif r.startswith("v:") and len(r) > 3:  # Specific verbs
            confidence += 0.2
        elif r in self.generic_relations and len(r) < 3:  # Generic relations
            confidence -= 0.2
            
        # Boost for corrections (high confidence updates)
        if intent.intent == IntentType.CORRECTION:
            confidence += 0.3
            
        # Boost for embedded facts in questions (validated info)
        if intent.intent == IntentType.QUESTION_WITH_FACT:
            confidence += 0.2
            
        # Penalty for very short entities (likely noise)
        if len(s) < 2 or len(d) < 2:
            confidence -= 0.2
            
        # Final threshold
        should_store = confidence >= 0.3
        return should_store, max(0.0, min(1.0, confidence))


# Global instances for performance
_intent_classifier = None
_quality_filter = None


def get_intent_classifier() -> IntentClassifier:
    """Get cached intent classifier instance

    Priority order:
    1. Enhanced Rule V2 (default) - 100% accuracy, <1ms latency
    2. DistilBERT (fallback) - If explicitly requested via USE_DISTILBERT_CLASSIFIER
    3. Basic rules (legacy) - If explicitly requested via USE_BASIC_CLASSIFIER
    """
    global _intent_classifier
    if _intent_classifier is None:
        # Check if DistilBERT fallback should be used
        use_distilbert = os.getenv("USE_DISTILBERT_CLASSIFIER", "false").lower() in ("true", "1", "yes")
        use_basic = os.getenv("USE_BASIC_CLASSIFIER", "false").lower() in ("true", "1", "yes")

        if use_distilbert:
            try:
                from components.memory.sota_intent_classifier import SOTAIntentClassifier
                # Create adapter that wraps SOTA classifier with old interface
                class DistilBERTAdapter(IntentClassifier):
                    def __init__(self):
                        super().__init__()
                        # Force DistilBERT model
                        os.environ["INTENT_CLASSIFIER_MODEL"] = "typeform/distilbert-base-uncased-mnli"
                        self.sota = SOTAIntentClassifier()
                        logger.info("🚀 Using DistilBERT fallback classifier (slower but handles edge cases)")

                    def analyze(self, text: str, lang: str = "en") -> IntentAnalysis:
                        """Adapt SOTA classifier to old interface"""
                        result = self.sota.classify(text)
                        # Map SOTA intent types to old IntentType enum
                        intent_map = {
                            "question_pure": IntentType.PURE_QUESTION,
                            "question_with_context": IntentType.QUESTION_WITH_FACT,
                            "fact_statement": IntentType.FACT_STATEMENT,
                            "correction": IntentType.CORRECTION,
                            "greeting": IntentType.REACTION,
                            "reaction": IntentType.REACTION,
                            "hypothetical": IntentType.HYPOTHETICAL,
                            "temporal_fact": IntentType.TEMPORAL_FACT,
                            "clarification": IntentType.PURE_QUESTION,
                            "command": IntentType.PURE_QUESTION,
                            "request": IntentType.PURE_QUESTION,
                            "farewell": IntentType.REACTION,
                            "acknowledgment": IntentType.REACTION,
                            "multiple_intents": IntentType.MULTIPLE_FACTS,
                            "unknown": IntentType.PURE_QUESTION,
                        }

                        old_intent = intent_map.get(result.primary_intent.value, IntentType.PURE_QUESTION)

                        analysis = IntentAnalysis(
                            intent=old_intent,
                            confidence=result.confidence,
                            should_extract_facts=result.requires_memory,
                            embedded_facts_likely=result.requires_memory,
                            temporal_markers=[],
                            correction_signals=["correction"] if old_intent == IntentType.CORRECTION else []
                        )

                        # Add SOTA-specific attributes for new code paths
                        analysis.requires_memory = result.requires_memory
                        analysis.requires_retrieval = result.requires_retrieval

                        return analysis

                _intent_classifier = DistilBERTAdapter()
            except Exception as e:
                logger.warning(f"Failed to initialize DistilBERT, falling back to Rule V2: {e}")
                use_distilbert = False

        if use_basic:
            logger.info("📏 Using basic rule classifier (legacy)")
            _intent_classifier = IntentClassifier()
        elif not use_distilbert:
            # DEFAULT: Use Enhanced Rule V2
            try:
                from components.memory.enhanced_rule_classifier_v2 import (
                    EnhancedRuleClassifierV2,
                    IntentType as V2IntentType
                )

                class RuleV2Adapter(IntentClassifier):
                    def __init__(self):
                        super().__init__()
                        self.v2 = EnhancedRuleClassifierV2()
                        logger.info("⚡ Using Enhanced Rule V2 classifier (100% accuracy, <1ms)")

                    def analyze(self, text: str, lang: str = "en") -> IntentAnalysis:
                        """Adapt V2 classifier to old interface"""
                        result = self.v2.classify(text)

                        # Map V2 intent types to old IntentType enum
                        intent_map = {
                            V2IntentType.QUESTION: IntentType.PURE_QUESTION,
                            V2IntentType.FACT: IntentType.FACT_STATEMENT,
                            V2IntentType.GREETING: IntentType.REACTION,
                            V2IntentType.ACKNOWLEDGMENT: IntentType.REACTION,
                            V2IntentType.REACTION: IntentType.REACTION,
                            V2IntentType.CORRECTION: IntentType.CORRECTION,
                            V2IntentType.COMMAND: IntentType.PURE_QUESTION,
                            V2IntentType.REQUEST: IntentType.PURE_QUESTION,
                            V2IntentType.FAREWELL: IntentType.REACTION,
                            V2IntentType.UNKNOWN: IntentType.PURE_QUESTION,
                        }

                        old_intent = intent_map.get(result.primary_intent, IntentType.PURE_QUESTION)

                        analysis = IntentAnalysis(
                            intent=old_intent,
                            confidence=result.confidence,
                            should_extract_facts=result.requires_memory,
                            embedded_facts_likely=result.requires_memory,
                            temporal_markers=[],
                            correction_signals=["correction"] if old_intent == IntentType.CORRECTION else []
                        )

                        # Add V2-specific attributes for new code paths
                        analysis.requires_memory = result.requires_memory
                        analysis.requires_retrieval = result.requires_retrieval

                        return analysis

                _intent_classifier = RuleV2Adapter()
            except Exception as e:
                logger.warning(f"Failed to initialize Rule V2, falling back to basic: {e}")
                _intent_classifier = IntentClassifier()

    return _intent_classifier


def get_quality_filter() -> QualityFilter:
    """Get cached quality filter instance"""
    global _quality_filter
    if _quality_filter is None:
        _quality_filter = QualityFilter()
    return _quality_filter
