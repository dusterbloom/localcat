#!/usr/bin/env python3
"""
Language-Agnostic Memory Correction System
==========================================

Uses Universal Dependencies (UD) patterns to detect and process correction commands
across multiple languages. Integrates with HotMemory for real-time fact updates.

Supported correction patterns (language agnostic via UD):
- Negation + correction: "No, I work at Google, not Microsoft"  
- Explicit correction: "Actually, my dog is 5 years old"
- Command-based: "/correct my age is 30" (English fallback)
- Fact replacement: "I moved to Seattle from Portland"

Core design:
- UD dependency parsing for cross-language support
- Intent detection via linguistic patterns
- Immediate fact update with temporal prioritization  
- Transparent feedback to user about changes made
"""

import os
import time
import re
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from loguru import logger

try:
    import spacy
    from spacy.tokens import Doc, Token
except ImportError:
    logger.warning("spaCy not available - correction system will be disabled")
    spacy = None

class CorrectionType(Enum):
    """Types of corrections detected"""
    NEGATION_CORRECTION = "negation_correction"    # "No, I work at X"
    EXPLICIT_CORRECTION = "explicit_correction"    # "Actually, I am Y"  
    COMMAND_CORRECTION = "command_correction"      # "/correct my name is Z"
    FACT_REPLACEMENT = "fact_replacement"         # "I moved to A from B"
    CONTRADICTION = "contradiction"               # "I don't work at X anymore"

@dataclass  
class CorrectionInstruction:
    """Parsed correction instruction"""
    correction_type: CorrectionType
    old_facts: List[Tuple[str, str, str]]  # Facts to demote/remove
    new_facts: List[Tuple[str, str, str]]  # Facts to promote/add
    confidence: float                       # Confidence in correction
    language: str                          # Detected language
    raw_text: str                          # Original text
    explanation: str                       # Human-readable explanation

class LanguageAgnosticCorrector:
    """
    Language-agnostic correction system using UD patterns.
    
    Detects correction intent across languages using Universal Dependencies,
    then extracts old/new facts for memory system updates.
    """
    
    def __init__(self):
        self.nlp_models = {}  # Cache for spaCy models by language
        self.correction_patterns = self._init_correction_patterns()
        
    def _init_correction_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize UD-based correction patterns for multiple languages"""
        return {
            # Negation patterns - universal across languages
            "negation": [
                {
                    "description": "Direct negation correction",
                    "pattern": [
                        {"DEP": {"IN": ["neg", "advmod"]}, "LOWER": {"IN": ["no", "not", "non", "nie", "不", "nein", "नहीं"]}},
                        {"DEP": {"IN": ["punct"]}, "OP": "?"},
                        {"POS": {"IN": ["PRON", "NOUN", "PROPN"]}},
                        {"DEP": {"IN": ["cop", "aux", "ROOT"]}},
                        {"DEP": {"IN": ["attr", "dobj", "nmod", "obl"]}}
                    ],
                    "correction_type": CorrectionType.NEGATION_CORRECTION
                }
            ],
            
            # Explicit correction patterns  
            "explicit": [
                {
                    "description": "Actually/correction adverbs",
                    "triggers": ["actually", "correction", "rectification", "eigentlich", "реально", "实际上", "वास्तव में"],
                    "pattern": [
                        {"LOWER": {"IN": ["actually", "correction", "eigentlich", "réellement", "실제로", "実際に"]}},
                        {"DEP": {"IN": ["punct"]}, "OP": "?"},
                        {"POS": {"IN": ["PRON", "NOUN", "PROPN"]}},
                        {"DEP": {"IN": ["cop", "aux", "ROOT"]}},
                        {"DEP": {"IN": ["attr", "dobj", "nmod", "obl"]}}
                    ],
                    "correction_type": CorrectionType.EXPLICIT_CORRECTION
                }
            ],
            
            # Contradiction patterns
            "contradiction": [
                {
                    "description": "No longer/anymore patterns", 
                    "pattern": [
                        {"POS": {"IN": ["PRON", "NOUN", "PROPN"]}},
                        {"DEP": {"IN": ["aux", "ROOT"]}},
                        {"DEP": {"IN": ["neg"]}, "LOWER": {"IN": ["not", "no", "nie", "不", "नहीं"]}},
                        {"LOWER": {"IN": ["anymore", "longer", "mehr", "plus", "了", "अब"]}},
                    ],
                    "correction_type": CorrectionType.CONTRADICTION
                }
            ],
            
            # Replacement patterns (moved from X to Y, changed from A to B)
            "replacement": [
                {
                    "description": "Movement/change patterns",
                    "pattern": [
                        {"LOWER": {"IN": ["moved", "changed", "switched", "transferred", "gewechselt", "déménagé", "이사했다", "引っ越した"]}},
                        {"LOWER": {"IN": ["to", "into", "at", "zu", "à", "に", "को"]}, "OP": "?"},
                        {"POS": {"IN": ["NOUN", "PROPN"]}},  # New location/value
                        {"LOWER": {"IN": ["from", "out", "of", "von", "de", "から", "से"]}}, 
                        {"POS": {"IN": ["NOUN", "PROPN"]}}   # Old location/value
                    ],
                    "correction_type": CorrectionType.FACT_REPLACEMENT
                }
            ]
        }
    
    def _load_nlp_model(self, language: str = "en") -> Optional[Any]:
        """Load spaCy model for language, with fallback to English"""
        if not spacy:
            return None
            
        if language in self.nlp_models:
            return self.nlp_models[language]
            
        # Map language codes to spaCy model names
        model_map = {
            "en": "en_core_web_sm",
            "es": "es_core_news_sm", 
            "de": "de_core_news_sm",
            "fr": "fr_core_news_sm",
            "zh": "zh_core_web_sm",
            "ja": "ja_core_news_sm",
            "hi": "xx_ent_wiki_sm"  # Multilingual fallback
        }
        
        model_name = model_map.get(language, "en_core_web_sm")
        
        try:
            nlp = spacy.load(model_name)
            self.nlp_models[language] = nlp
            return nlp
        except OSError:
            # Fallback to English if specific model not available
            if language != "en":
                logger.warning(f"spaCy model '{model_name}' not found, falling back to English")
                return self._load_nlp_model("en")
            logger.error("English spaCy model not available - correction system disabled")
            return None
    
    def detect_language(self, text: str) -> str:
        """Detect language from text, with fallback to English"""
        try:
            # Use pycld3 if available for accurate language detection
            import pycld3
            result = pycld3.get_language(text)
            if result and result.probability > 0.7:
                return result.language
        except ImportError:
            pass
            
        # Simple heuristic fallback
        if any(char in text for char in "áéíóúñü"):
            return "es"
        elif any(char in text for char in "äöüß"):  
            return "de"
        elif any(char in text for char in "àéèêôç"):
            return "fr"
        elif any(ord(char) > 0x4e00 for char in text):  # Chinese characters
            return "zh"
        elif any(ord(char) > 0x3040 for char in text):  # Japanese characters  
            return "ja"
        elif any(ord(char) > 0x900 for char in text):   # Hindi/Devanagari
            return "hi"
            
        return "en"  # Default fallback
    
    def detect_correction_intent(self, text: str, language: str = None) -> Optional[CorrectionInstruction]:
        """
        Detect correction intent in text using UD patterns.
        
        Args:
            text: Input text to analyze
            language: Language code (auto-detected if None)
            
        Returns:
            CorrectionInstruction if correction detected, None otherwise
        """
        if not text or not text.strip():
            return None
            
        if language is None:
            language = self.detect_language(text)
            
        nlp = self._load_nlp_model(language)
        if not nlp:
            return None
            
        doc = nlp(text)
        
        # Check for explicit command-based corrections first
        command_correction = self._detect_command_correction(text, doc, language)
        if command_correction:
            return command_correction
            
        # Check UD-based patterns
        for pattern_type, patterns in self.correction_patterns.items():
            for pattern_config in patterns:
                correction = self._match_ud_pattern(doc, pattern_config, text, language)
                if correction:
                    return correction
                    
        return None
    
    def _detect_command_correction(self, text: str, doc: Doc, language: str) -> Optional[CorrectionInstruction]:
        """Detect explicit command-based corrections like /correct, /fix, etc."""
        text_lower = text.lower().strip()
        
        # Command prefixes in multiple languages
        command_prefixes = {
            "en": ["/correct", "/fix", "/update", "/change"],
            "es": ["/corregir", "/arreglar", "/actualizar"],  
            "de": ["/korrigieren", "/reparieren", "/aktualisieren"],
            "fr": ["/corriger", "/réparer", "/mettre à jour"],
            "zh": ["/纠正", "/修复", "/更新"],
            "ja": ["/修正", "/修復", "/更新"],
            "hi": ["/सुधार", "/ठीक", "/अपडेट"]
        }
        
        all_prefixes = []
        for prefixes in command_prefixes.values():
            all_prefixes.extend(prefixes)
            
        # Check if text starts with correction command
        matching_prefix = None
        for prefix in all_prefixes:
            if text_lower.startswith(prefix):
                matching_prefix = prefix
                break
                
        if not matching_prefix:
            return None
            
        # Extract content after command
        content = text[len(matching_prefix):].strip()
        if not content:
            return None
            
        # Parse the correction content to extract facts
        try:
            new_facts = self._extract_facts_from_text(content, doc, language)
            return CorrectionInstruction(
                correction_type=CorrectionType.COMMAND_CORRECTION,
                old_facts=[],  # Commands typically don't specify old facts
                new_facts=new_facts,
                confidence=0.9,  # High confidence for explicit commands
                language=language,
                raw_text=text,
                explanation=f"Command correction: {matching_prefix} → {len(new_facts)} new facts"
            )
        except Exception as e:
            logger.warning(f"Failed to parse command correction: {e}")
            return None
    
    def _match_ud_pattern(self, doc: Doc, pattern_config: Dict, text: str, language: str) -> Optional[CorrectionInstruction]:
        """Match UD-based correction patterns in parsed document"""
        try:
            # Simplified pattern matching - look for key linguistic markers
            correction_type = pattern_config["correction_type"]
            
            if correction_type == CorrectionType.NEGATION_CORRECTION:
                return self._detect_negation_correction(doc, text, language)
            elif correction_type == CorrectionType.EXPLICIT_CORRECTION:
                return self._detect_explicit_correction(doc, text, language)
            elif correction_type == CorrectionType.CONTRADICTION:
                return self._detect_contradiction(doc, text, language) 
            elif correction_type == CorrectionType.FACT_REPLACEMENT:
                return self._detect_replacement(doc, text, language)
                
        except Exception as e:
            logger.warning(f"Pattern matching failed: {e}")
            
        return None
    
    def _detect_negation_correction(self, doc: Doc, text: str, language: str) -> Optional[CorrectionInstruction]:
        """Detect negation-based corrections: 'No, I work at X, not Y'"""
        text_lower = text.lower()
        
        # Universal negation markers
        negation_markers = ["no,", "not", "nein,", "non,", "nie", "不是", "नहीं,"]
        
        has_negation = any(marker in text_lower for marker in negation_markers)
        if not has_negation:
            return None
            
        # Look for correction pattern: negation + new info + "not" + old info
        not_phrases = ["not", "nie", "pas", "不是", "नहीं"]
        contrast_found = any(phrase in text_lower for phrase in not_phrases)
        
        if has_negation and contrast_found:
            # Extract facts before and after negation
            new_facts = self._extract_facts_from_text(text, doc, language)
            
            return CorrectionInstruction(
                correction_type=CorrectionType.NEGATION_CORRECTION,
                old_facts=[],  # Would need more sophisticated extraction
                new_facts=new_facts,
                confidence=0.75,
                language=language,
                raw_text=text,
                explanation=f"Negation correction detected with {len(new_facts)} new facts"
            )
            
        return None
    
    def _detect_explicit_correction(self, doc: Doc, text: str, language: str) -> Optional[CorrectionInstruction]:
        """Detect explicit correction markers: 'Actually, I live in X'"""
        text_lower = text.lower()
        
        # Universal correction markers
        correction_markers = [
            "actually", "correction", "eigentlich", "réellement", "実際に", "वास्तव में",
            "let me correct", "to clarify", "i should clarify", "just to be clear"
        ]
        
        has_correction_marker = any(marker in text_lower for marker in correction_markers)
        if not has_correction_marker:
            return None
            
        # Extract facts from corrected statement
        new_facts = self._extract_facts_from_text(text, doc, language)
        
        return CorrectionInstruction(
            correction_type=CorrectionType.EXPLICIT_CORRECTION,
            old_facts=[],
            new_facts=new_facts,
            confidence=0.8,
            language=language,
            raw_text=text,
            explanation=f"Explicit correction with {len(new_facts)} new facts"
        )
    
    def _detect_contradiction(self, doc: Doc, text: str, language: str) -> Optional[CorrectionInstruction]:
        """Detect contradictions: 'I don't work there anymore'"""
        text_lower = text.lower()
        
        # Universal contradiction patterns
        contradiction_patterns = [
            "don't.*anymore", "no longer", "not.*anymore", "nie mehr", "plus", "もう.*ない", "अब नहीं"
        ]
        
        has_contradiction = any(re.search(pattern, text_lower) for pattern in contradiction_patterns)
        if not has_contradiction:
            return None
            
        # This indicates old facts should be demoted/removed
        return CorrectionInstruction(
            correction_type=CorrectionType.CONTRADICTION,
            old_facts=[],  # Would extract from context
            new_facts=[],
            confidence=0.7,
            language=language,
            raw_text=text,
            explanation="Contradiction detected - old facts should be demoted"
        )
    
    def _detect_replacement(self, doc: Doc, text: str, language: str) -> Optional[CorrectionInstruction]:
        """Detect replacement patterns: 'I moved to X from Y'"""
        text_lower = text.lower()
        
        # Universal replacement patterns
        replacement_verbs = ["moved", "changed", "switched", "transferred", "gewechselt", "déménagé", "이사", "引っ越し"]
        
        has_replacement = any(verb in text_lower for verb in replacement_verbs)
        if not has_replacement:
            return None
            
        # Extract old and new values
        new_facts = self._extract_facts_from_text(text, doc, language)
        
        return CorrectionInstruction(
            correction_type=CorrectionType.FACT_REPLACEMENT,
            old_facts=[],  # Would need to extract "from" part
            new_facts=new_facts,
            confidence=0.8,
            language=language,
            raw_text=text,
            explanation=f"Replacement correction with {len(new_facts)} new facts"
        )
    
    def _extract_facts_from_text(self, text: str, doc: Doc, language: str) -> List[Tuple[str, str, str]]:
        """Extract facts from corrected text using UD patterns and text patterns"""
        facts = []
        text_lower = text.lower().strip()
        
        try:
            # Command-based extraction (more reliable)
            if text_lower.startswith(('/correct', '/corregir', '/korrigieren')):
                content = text[text.find(' '):].strip()
                return self._extract_from_command_content(content, doc)
            
            # Pattern-based extraction for common correction formats
            facts.extend(self._extract_name_facts(text_lower, doc))
            facts.extend(self._extract_work_facts(text_lower, doc))
            facts.extend(self._extract_location_facts(text_lower, doc))
            facts.extend(self._extract_general_facts(text_lower, doc))
                            
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            
        return facts
    
    def _extract_from_command_content(self, content: str, doc: Doc) -> List[Tuple[str, str, str]]:
        """Extract facts from command content like '/correct my name is Sarah'"""
        content_lower = content.lower()
        facts = []
        
        # Name patterns
        if "my name is" in content_lower or "name is" in content_lower:
            import re
            match = re.search(r'(?:my )?name is\s+(\w+)', content_lower)
            if match:
                name = match.group(1)
                facts.append(("you", "name", name))
        
        # Work patterns  
        if "i work at" in content_lower or "work at" in content_lower:
            import re
            match = re.search(r'(?:i )?work at\s+(\w+)', content_lower)
            if match:
                company = match.group(1)
                facts.append(("you", "works_at", company))
        
        # Location patterns
        if "i live in" in content_lower or "live in" in content_lower:
            import re  
            match = re.search(r'(?:i )?live in\s+(\w+)', content_lower)
            if match:
                location = match.group(1)
                facts.append(("you", "lives_in", location))
                
        return facts
    
    def _extract_name_facts(self, text_lower: str, doc: Doc) -> List[Tuple[str, str, str]]:
        """Extract name-related facts"""
        facts = []
        
        # "My name is X" patterns
        import re
        patterns = [
            r'my name is\s+(\w+)',
            r'i am\s+(\w+)',  
            r'call me\s+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1)
                facts.append(("you", "name", name))
                break
                
        return facts
    
    def _extract_work_facts(self, text_lower: str, doc: Doc) -> List[Tuple[str, str, str]]:
        """Extract work-related facts"""
        facts = []
        
        # Work patterns
        import re
        work_patterns = [
            r'i work at\s+(\w+)',
            r'work at\s+(\w+)',
            r'employed at\s+(\w+)',
            r'job at\s+(\w+)'
        ]
        
        for pattern in work_patterns:
            match = re.search(pattern, text_lower)
            if match:
                company = match.group(1)
                facts.append(("you", "works_at", company))
                break
                
        return facts
    
    def _extract_location_facts(self, text_lower: str, doc: Doc) -> List[Tuple[str, str, str]]:
        """Extract location-related facts"""  
        facts = []
        
        # Location patterns
        import re
        location_patterns = [
            r'i live in\s+(\w+)',
            r'live in\s+(\w+)',
            r'from\s+(\w+)',
            r'moved to\s+(\w+)',
            r'living in\s+(\w+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text_lower)
            if match:
                location = match.group(1)
                facts.append(("you", "lives_in", location))
                break
                
        return facts
    
    def _extract_general_facts(self, text_lower: str, doc: Doc) -> List[Tuple[str, str, str]]:
        """Extract general facts using UD parsing as fallback"""
        facts = []
        
        try:
            # Simplified UD extraction for remaining cases
            for token in doc:
                if token.dep_ == "nsubj" and token.head.pos_ in ["VERB", "AUX"]:
                    subj = token.text.lower()
                    if subj == "i":
                        subj = "you"
                        
                    verb = token.head.lemma_.lower()
                    
                    # Find direct objects or attributes
                    for child in token.head.children:
                        if child.dep_ in ["dobj", "attr"]:
                            obj = child.text.lower()
                            
                            # Map common verbs to relations
                            if verb in ["be", "am", "is", "are"]:
                                relation = "is"
                            elif verb == "have":
                                relation = "has" 
                            else:
                                relation = verb
                                
                            facts.append((subj, relation, obj))
                            
        except Exception as e:
            logger.debug(f"UD extraction failed: {e}")
            
        return facts
    
    def apply_correction(self, instruction: CorrectionInstruction, hot_memory) -> Dict[str, Any]:
        """
        Apply correction instruction to HotMemory system.
        
        Args:
            instruction: Parsed correction instruction
            hot_memory: HotMemory instance to update
            
        Returns:
            Results of correction application
        """
        try:
            start_time = time.perf_counter()
            
            # Demote old facts if specified
            demoted_count = 0
            for old_fact in instruction.old_facts:
                s, r, d = old_fact
                # Reduce confidence/weight of old fact
                if (s, r, d) in hot_memory.edge_meta:
                    hot_memory.edge_meta[(s, r, d)]['weight'] *= 0.1  # Demote significantly
                    demoted_count += 1
            
            # Promote new facts
            promoted_count = 0 
            now_ts = int(time.time() * 1000)
            
            for new_fact in instruction.new_facts:
                s, r, d = new_fact
                # Store with high confidence and fresh timestamp
                hot_memory.store.observe_edge(s, r, d, instruction.confidence, now_ts)
                
                # Update hot indices
                hot_memory.entity_index[s].add((s, r, d))
                hot_memory.entity_index[d].add((s, r, d))
                
                # Add to recency buffer
                from memory_hotpath import RecencyItem
                hot_memory.recency_buffer.append(RecencyItem(s, r, d, instruction.raw_text, now_ts, -1))
                
                # Update edge metadata
                hot_memory.edge_meta[(s, r, d)] = {
                    'ts': now_ts,
                    'weight': 0.9  # High weight for corrections
                }
                
                promoted_count += 1
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            result = {
                'success': True,
                'correction_type': instruction.correction_type.value,
                'language': instruction.language,
                'demoted_facts': demoted_count,
                'promoted_facts': promoted_count, 
                'processing_time_ms': processing_time,
                'explanation': instruction.explanation
            }
            
            logger.info(f"Applied correction: {instruction.correction_type.value} → demoted {demoted_count}, promoted {promoted_count}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply correction: {e}")
            return {
                'success': False,
                'error': str(e),
                'correction_type': instruction.correction_type.value if instruction else None
            }

# Global corrector instance
_corrector_instance = None

def get_corrector() -> LanguageAgnosticCorrector:
    """Get global corrector instance (singleton)"""
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = LanguageAgnosticCorrector()
    return _corrector_instance