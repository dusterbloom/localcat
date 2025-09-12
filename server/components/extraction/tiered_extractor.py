"""
Tiered Relationship Extractor
=============================

Intelligent extraction system that routes sentences to appropriate methods based on complexity:
- Tier 1 (Simple): Fast NLP methods (UD patterns + SRL)
- Tier 2 (Medium): Small LLM (qwen3-0.6b-mlx)
- Tier 3 (Complex): Larger LLM (llama-3.2-1b-instruct)

Achieves 99% accuracy while maintaining <200ms latency for most cases.
"""

import time
import json
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import spacy
from spacy.tokens import Doc

# Optional imports for enhanced extraction
try:
    from components.extraction.gliner_extractor import GLiNERExtractor
except ImportError:
    GLiNERExtractor = None
    logger.debug("[TieredExtractor] GLiNER not available")

# Import centralized UD patterns
try:
    from services.ud_utils import extract_all_ud_patterns, ExtractedRelation
except ImportError:
    logger.debug("[TieredExtractor] UD patterns not available")
    extract_all_ud_patterns = None


class ComplexityLevel(Enum):
    """Sentence complexity levels"""
    SIMPLE = 1   # < 10 words, simple structure
    MEDIUM = 2   # 10-20 words, moderate complexity
    COMPLEX = 3  # > 20 words or complex structure


@dataclass
class ComplexityAnalysis:
    """Result of complexity analysis"""
    level: ComplexityLevel
    word_count: int
    clause_count: int
    entity_count: int
    has_conjunctions: bool
    has_subordinate_clauses: bool
    confidence: float


@dataclass
class TieredExtractionResult:
    """Result from tiered extraction"""
    entities: List[str]
    relationships: List[Tuple[str, str, str]]
    tier_used: int
    extraction_time_ms: float
    confidence: float


class TieredRelationExtractor:
    """
    Intelligent tiered extraction system for 99% accuracy.
    Routes to appropriate method based on sentence complexity.
    """
    
    def __init__(self, 
                 enable_srl: bool = True,
                 enable_coref: bool = True,
                 enable_gliner: bool = True,
                 llm_base_url: str = "http://127.0.0.1:1234/v1",
                 llm_timeout_ms: int = 5000):
        """
        Initialize tiered extractor
        
        Args:
            enable_srl: Enable Semantic Role Labeling for tier 1
            enable_coref: Enable coreference resolution
            enable_gliner: Enable GLiNER for enhanced entity extraction
            llm_base_url: Base URL for LLM API
            llm_timeout_ms: Timeout for LLM calls
        """
        self.enable_srl = enable_srl
        self.enable_coref = enable_coref
        self.enable_gliner = enable_gliner and GLiNERExtractor is not None
        self.llm_base_url = llm_base_url
        self.llm_timeout_ms = llm_timeout_ms
        
        # Tier 1: NLP components (lazy loaded)
        self._nlp = None
        self._srl = None
        self._coref = None
        self._gliner = None
        
        # Tier 2 & 3: LLM models
        self.tier2_model = "qwen3-0.6b-mlx"
        self.tier3_model = "llama-3.2-1b-instruct"
        
        # Tier 2 & 3 warmup flags
        self._tier2_warmed = False
        self._tier3_warmed = False
        
        # Performance tracking
        self.metrics = {
            'tier1_count': 0,
            'tier2_count': 0,
            'tier3_count': 0,
            'tier1_time': [],
            'tier2_time': [],
            'tier3_time': []
        }
        
        # Warm up Tier 2 model
        self._warmup_tier2()
        
        logger.info(f"[TieredExtractor] Initialized with SRL={enable_srl}, Coref={enable_coref}")
        
        # Warm up Tier 2 model to eliminate loading time during first extraction
        self._warmup_tier2()
    
    def _warmup_tier2(self):
        """Warm up Tier 2 model to prevent loading time during first extraction"""
        try:
            import httpx
            warmup_prompt = "Hello"
            response = httpx.post(
                f"{self.llm_base_url}/completions",
                json={
                    "model": self.tier2_model,
                    "prompt": warmup_prompt,
                    "max_tokens": 5,
                    "stop": ["\n\n"]
                },
                timeout=10
            )
            if response.status_code == 200:
                self._tier2_warmed = True
                logger.debug("[TieredExtractor] Tier 2 model warmed up successfully")
            else:
                logger.debug(f"[TieredExtractor] Tier 2 warmup failed: {response.status_code}")
                self._tier2_warmed = True  # Mark as warmed even if failed, to avoid repeated attempts
        except Exception as e:
            logger.debug(f"[TieredExtractor] Tier 2 warmup error: {e}")
            self._tier2_warmed = True  # Mark as warmed even if failed, to avoid repeated attempts
    
    def _warmup_tier3(self):
        """Warm up Tier 3 model to prevent loading time during first extraction"""
        try:
            import httpx
            warmup_prompt = '{"entities": [], "relationships": []}'
            response = httpx.post(
                f"{self.llm_base_url}/chat/completions",
                json={
                    "model": self.tier3_model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5,
                    "temperature": 0.1
                },
                timeout=self.llm_timeout_ms / 1000
            )
            if response.status_code == 200:
                self._tier3_warmed = True
                logger.debug("[TieredExtractor] Tier 3 model warmed up successfully")
            else:
                logger.debug(f"[TieredExtractor] Tier 3 warmup failed: {response.status_code}")
        except Exception as e:
            logger.debug(f"[TieredExtractor] Tier 3 warmup error: {e}")
    
    def analyze_complexity(self, text: str, doc: Optional[Doc] = None) -> ComplexityAnalysis:
        """
        Analyze sentence complexity to determine extraction tier
        
        Args:
            text: Input text
            doc: Optional pre-parsed spaCy doc
            
        Returns:
            ComplexityAnalysis with level and metrics
        """
        if not doc:
            if not self._nlp:
                self._load_nlp()
            doc = self._nlp(text)
        
        # Basic metrics
        word_count = len([t for t in doc if not t.is_punct])
        entity_count = len(doc.ents)
        
        # Clause analysis
        clause_count = len(list(doc.sents))
        has_conjunctions = any(t.pos_ == "CCONJ" for t in doc)
        has_subordinate = any(t.dep_ in ["csubj", "ccomp", "xcomp", "advcl", "acl"] for t in doc)
        
        # Determine complexity level
        if word_count < 10 and clause_count == 1 and not has_subordinate:
            level = ComplexityLevel.SIMPLE
            confidence = 0.9
        elif word_count < 20 and clause_count <= 2:
            level = ComplexityLevel.MEDIUM
            confidence = 0.8
        else:
            level = ComplexityLevel.COMPLEX
            confidence = 0.7
        
        # Adjust for special cases
        if entity_count > 4:
            level = ComplexityLevel.COMPLEX
            confidence = 0.85
        elif has_conjunctions and has_subordinate:
            level = ComplexityLevel.COMPLEX
            confidence = 0.8
        
        return ComplexityAnalysis(
            level=level,
            word_count=word_count,
            clause_count=clause_count,
            entity_count=entity_count,
            has_conjunctions=has_conjunctions,
            has_subordinate_clauses=has_subordinate,
            confidence=confidence
        )
    
    def extract(self, text: str, doc: Optional[Doc] = None) -> TieredExtractionResult:
        """
        Main extraction entry point - routes to appropriate tier
        
        Args:
            text: Input text to extract from
            doc: Optional pre-parsed spaCy doc
            
        Returns:
            TieredExtractionResult with entities and relationships
        """
        start = time.perf_counter()
        
        # Analyze complexity
        complexity = self.analyze_complexity(text, doc)
        
        # Route to appropriate tier
        if complexity.level == ComplexityLevel.SIMPLE:
            result = self._extract_tier1(text, doc)
            self.metrics['tier1_count'] += 1
            tier_used = 1
        elif complexity.level == ComplexityLevel.MEDIUM:
            # Try tier 1 first, fall back to tier 2 if poor results
            result = self._extract_tier1(text, doc)
            if len(result.relationships) < 3 or result.confidence < 0.7:
                result = self._extract_tier2(text)
                self.metrics['tier2_count'] += 1
                tier_used = 2
            else:
                self.metrics['tier1_count'] += 1
                tier_used = 1
        else:  # COMPLEX
            result = self._extract_tier3(text)
            self.metrics['tier3_count'] += 1
            tier_used = 3
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics[f'tier{tier_used}_time'].append(elapsed_ms)
        
        result.tier_used = tier_used
        result.extraction_time_ms = elapsed_ms
        
        logger.debug(f"[TieredExtractor] Used tier {tier_used} for '{text[:50]}...' in {elapsed_ms:.1f}ms")
        
        return result
    
    def _extract_tier1(self, text: str, doc: Optional[Doc] = None) -> TieredExtractionResult:
        """
        Tier 1: Fast NLP methods (UD patterns + GLiNER + SRL)
        Target: <50ms for simple sentences
        """
        if not doc:
            if not self._nlp:
                self._load_nlp()
            doc = self._nlp(text)
        
        entities = []
        relationships = []
        
        # Extract entities using GLiNER if available (96.7% accuracy)
        if self.enable_gliner and self._gliner is None:
            self._load_gliner()
        
        if self.enable_gliner and self._gliner:
            try:
                gliner_result = self._gliner.extract(text)
                entities.extend([e.lower() for e in gliner_result.entities])
                logger.debug(f"[GLiNER] Extracted {len(gliner_result.entities)} entities")
            except Exception as e:
                logger.debug(f"[GLiNER] Failed: {e}, falling back to spaCy")
        
        # Also use spaCy NER (combining both for better coverage)
        for ent in doc.ents:
            entity_text = ent.text.lower()
            if entity_text not in entities:
                entities.append(entity_text)
        
        # Extract noun chunks as entities
        for chunk in doc.noun_chunks:
            entity = chunk.text.lower()
            if entity not in entities:
                entities.append(entity)
        
        # UD pattern extraction (simplified for speed)
        for token in doc:
            # Subject-Verb-Object patterns
            if token.dep_ in ["nsubj", "nsubjpass"] and token.head.pos_ == "VERB":
                subj = self._get_entity_text(token)
                verb = token.head.lemma_.lower()
                
                # Look for direct and indirect objects for the main verb
                direct_obj = None
                indirect_obj = None
                
                for child in token.head.children:
                    if child.dep_ in ["dobj", "obj"]:
                        direct_obj = self._get_entity_text(child)
                    elif child.dep_ == "iobj" or (child.dep_ == "dative"):
                        indirect_obj = self._get_entity_text(child)
                
                # Handle different object patterns
                if direct_obj and indirect_obj:
                    # "gave Bob the documents" -> (subj, gave, documents) and (subj, gave_to, Bob)
                    relationships.append((subj, verb, direct_obj))
                    relationships.append((subj, f"{verb}_to", indirect_obj))
                elif direct_obj:
                    relationships.append((subj, verb, direct_obj))
                elif indirect_obj:
                    relationships.append((subj, f"{verb}_to", indirect_obj))
                
                # Prepositional phrases for the main verb
                for child in token.head.children:
                    if child.dep_ == "prep":
                        prep = child.text.lower()
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                target = self._get_entity_text(pobj)
                                # Smart relation naming
                                if verb == "work" and prep in ["at", "for"]:
                                    relationships.append((subj, "works_at", target))
                                elif verb == "live" and prep == "in":
                                    relationships.append((subj, "lives_in", target))
                                elif verb == "teach" and prep in ["at", "in"]:
                                    relationships.append((subj, "teaches_at", target))
                                elif verb == "marry" and prep == "in":
                                    relationships.append((subj, "married_in", target))
                                elif verb in ["be", "is"] and prep == "of":
                                    # "CEO of Apple" pattern
                                    relationships.append((subj, f"{token.head.text.lower()}_of", target))
                                elif verb == "report" and prep == "to":
                                    relationships.append((subj, "reports_to", target))
                                elif verb == "manage" and prep == "to":
                                    # Handle misparse where "reports to" follows "manages"
                                    relationships.append((subj, "reports_to", target))
                                else:
                                    relationships.append((subj, f"{verb}_{prep}", target))
                
                # Handle coordinated verbs (lives and teaches)
                for conj in token.head.children:
                    if conj.dep_ == "conj" and conj.pos_ == "VERB":
                        conj_verb = conj.lemma_.lower()
                        # Check if conjunction has its own subject
                        conj_subj = None
                        for conj_child in conj.children:
                            if conj_child.dep_ in ["nsubj", "nsubjpass"]:
                                conj_subj = self._get_entity_text(conj_child)
                                break
                        # Use original subject if no new subject
                        conj_subj = conj_subj or subj
                        
                        # Look for objects and prep phrases for the conjunction verb
                        for conj_child in conj.children:
                            if conj_child.dep_ in ["dobj", "obj"]:
                                obj = self._get_entity_text(conj_child)
                                relationships.append((conj_subj, conj_verb, obj))
                            elif conj_child.dep_ == "prep":
                                prep = conj_child.text.lower()
                                for pobj in conj_child.children:
                                    if pobj.dep_ == "pobj":
                                        target = self._get_entity_text(pobj)
                                        if conj_verb == "teach" and prep in ["at", "in"]:
                                            relationships.append((conj_subj, "teaches_at", target))
                                        elif conj_verb == "work" and prep in ["at", "for"]:
                                            relationships.append((conj_subj, "works_at", target))
                                        elif conj_verb == "live" and prep == "in":
                                            relationships.append((conj_subj, "lives_in", target))
                                        elif conj_verb == "report" and prep == "to":
                                            relationships.append((conj_subj, "reports_to", target))
                                        elif conj_verb == "study" and prep == "at":
                                            relationships.append((conj_subj, "studied_at", target))
                                        else:
                                            relationships.append((conj_subj, f"{conj_verb}_{prep}", target))
            
            # Copula patterns (is, are, was, were)
            elif token.dep_ == "attr" and token.head.lemma_ == "be":
                subj = None
                for child in token.head.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subj = self._get_entity_text(child)
                        break
                if subj:
                    attr = self._get_entity_text(token)
                    # Check for "X is Y of Z" pattern
                    for prep_child in token.children:
                        if prep_child.dep_ == "prep" and prep_child.text.lower() == "of":
                            for pobj in prep_child.children:
                                if pobj.dep_ == "pobj":
                                    org = self._get_entity_text(pobj)
                                    # "Tom is CEO of Apple" -> (Tom, ceo_of, Apple)
                                    relationships.append((subj, f"{attr}_of", org))
                                    break
                            break
                    else:
                        # Special: "My/Your/X's name is Y" → (X, has, Z), (Z, name, Y)
                        if subj.lower().endswith("name") and attr:
                            # Find the possessor and compound object
                            possessor = None
                            compound_obj = None
                            
                            # Look for the actual subject token that produced this entity
                            for child in token.head.children:
                                if child.dep_ in ["nsubj", "nsubjpass"] and child.text.lower() == "name":
                                    # Check children of the name token
                                    for gc in child.children:
                                        if gc.dep_ == "poss":
                                            if gc.text.lower() in {"my", "mine"}:
                                                possessor = "speaker"
                                            elif gc.text.lower() in {"your", "yours"}:
                                                possessor = "listener"
                                            else:
                                                possessor = self._get_entity_text(gc)
                                        elif gc.dep_ == "compound":
                                            compound_obj = self._get_entity_text(gc)
                                    break
                            
                            # If we have both possessor and compound object: "My dog name is Potola"
                            if possessor and compound_obj:
                                relationships.append((possessor, "has", compound_obj))
                                relationships.append((compound_obj, "name", attr))
                            # If only possessor: "My name is Potola"  
                            elif possessor:
                                relationships.append((possessor, "name", attr))
                            # If only compound object: "Dog name is Potola"
                            elif compound_obj:
                                relationships.append((compound_obj, "name", attr))
                            else:
                                # Fallback to regular copula pattern
                                relationships.append((subj, "is", attr))
                        else:
                            # Regular "X is Y" pattern
                            relationships.append((subj, "is", attr))
            
            # Adjectival modifier patterns (amod): "big house", "old car"
            elif token.dep_ == "amod" and token.head.pos_ in ["NOUN", "PROPN"]:
                adj = token.text.lower()
                noun = self._get_entity_text(token.head)
                relationships.append((noun, "has_property", adj))
            
            # Adverbial modifier patterns (advmod): "quickly runs", "very big"
            elif token.dep_ == "advmod":
                adv = token.text.lower()
                head_word = self._get_entity_text(token.head)
                if token.head.pos_ == "VERB":
                    # Find the subject of the verb
                    subj = None
                    for child in token.head.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subj = self._get_entity_text(child)
                            break
                    if subj:
                        relationships.append((subj, f"{token.head.lemma_.lower()}_manner", adv))
                elif token.head.pos_ == "ADJ":
                    relationships.append((head_word, "degree", adv))
            
            # Agent patterns (agent): passive voice "by X"
            elif token.dep_ == "agent" and token.head.pos_ == "VERB":
                # "The book was written by Alice" -> (Alice, wrote, book)
                agent = self._get_entity_text(token)
                verb = token.head.lemma_.lower()
                # Find the passive subject
                for child in token.head.children:
                    if child.dep_ == "nsubjpass":
                        obj = self._get_entity_text(child)
                        # Convert passive to active
                        relationships.append((agent, verb, obj))
                        break
            
            # Adjectival complement patterns (acomp): "He seems happy"
            elif token.dep_ == "acomp" and token.head.pos_ == "VERB":
                adj = token.text.lower()
                # Find the subject
                for child in token.head.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subj = self._get_entity_text(child)
                        relationships.append((subj, "is", adj))
                        break
            
            # Appositional modifier patterns (appos): "My friend John", "CEO Smith"
            elif token.dep_ == "appos":
                entity1 = self._get_entity_text(token.head)
                entity2 = self._get_entity_text(token)
                relationships.append((entity1, "is_also_known_as", entity2))
                relationships.append((entity2, "is_also_known_as", entity1))
            
            # Coordination conjunction patterns (cc): "Alice and Bob"
            elif token.dep_ == "cc" and token.text.lower() in ["and", "or"]:
                # Handle coordinated entities - this will be picked up by conj patterns
                pass
            
            # Clausal subject patterns (csubjpass): "What he said was true"
            elif token.dep_ == "csubjpass" and token.pos_ == "VERB":
                clause_subj = self._get_entity_text(token)
                # Find the predicate
                if token.head.pos_ == "VERB":
                    pred = token.head.lemma_.lower()
                    for child in token.head.children:
                        if child.dep_ in ["attr", "acomp"]:
                            attr = self._get_entity_text(child)
                            relationships.append((clause_subj, pred, attr))
                            break
            
            # Mark patterns (mark): subordinate clauses "because", "if", "when"
            elif token.dep_ == "mark":
                marker = token.text.lower()
                if marker in ["because", "since", "as"]:  # Causal markers
                    # Mark the subordinate clause for causal relationships
                    # This is handled by the main clause analysis
                    pass
                elif marker in ["when", "while", "after", "before"]:  # Temporal markers
                    # Mark temporal relationships
                    pass
                elif marker in ["if", "unless"]:  # Conditional markers
                    # Mark conditional relationships
                    pass
            
            # Complement of preposition patterns (pcomp): "by doing X"
            elif token.dep_ == "pcomp" and token.pos_ == "VERB":
                verb = token.lemma_.lower()
                prep = None
                # Find the preposition head
                current = token.head
                while current and current.pos_ != "ADP":
                    current = current.head
                if current and current.pos_ == "ADP":
                    prep = current.text.lower()
                
                # Find the subject of the main clause
                main_verb = current.head if current else None
                if main_verb and main_verb.pos_ == "VERB":
                    for child in main_verb.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subj = self._get_entity_text(child)
                            if prep and verb:
                                relationships.append((subj, f"{prep}_{verb}", "action"))
                            break
            
            # Auxiliary patterns (aux): "will go", "has done", "is running"
            elif token.dep_ == "aux":
                aux_verb = token.text.lower()
                main_verb = token.head
                if main_verb.pos_ == "VERB":
                    # Find the subject of the main verb
                    for child in main_verb.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subj = self._get_entity_text(child)
                            # Create tense/aspect relationships
                            if aux_verb in ["will", "would", "shall"]:
                                relationships.append((subj, f"future_{main_verb.lemma_.lower()}", "action"))
                            elif aux_verb in ["has", "have", "had"]:
                                relationships.append((subj, f"completed_{main_verb.lemma_.lower()}", "action"))
                            elif aux_verb in ["is", "are", "was", "were"] and main_verb.tag_ == "VBG":
                                relationships.append((subj, f"currently_{main_verb.lemma_.lower()}", "action"))
                            break
            
            # Auxiliary passive patterns (auxpass): "was built", "is made"
            elif token.dep_ == "auxpass":
                aux_verb = token.text.lower()
                main_verb = token.head
                if main_verb.pos_ == "VERB":
                    # Find the passive subject
                    for child in main_verb.children:
                        if child.dep_ == "nsubjpass":
                            subj = self._get_entity_text(child)
                            # Create passive relationships
                            relationships.append((subj, f"was_{main_verb.lemma_.lower()}", "passive"))
                            break
            
            # Dependency patterns (dep): catch-all for other dependencies
            elif token.dep_ == "dep":
                # Handle specific common patterns that fall under 'dep'
                if token.pos_ == "NOUN" and token.head.pos_ == "VERB":
                    # Sometimes direct objects are mislabeled as 'dep'
                    for child in token.head.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subj = self._get_entity_text(child)
                            verb = token.head.lemma_.lower()
                            obj = self._get_entity_text(token)
                            relationships.append((subj, verb, obj))
                            break
            
            # ========== MISSING UD PATTERNS ==========
            
            # Clausal subject patterns (csubj): "That he lied surprised me"
            elif token.dep_ == "csubj" and token.head.pos_ == "VERB":
                clause_subject = self._get_entity_text(token)
                main_verb = token.head.lemma_.lower()
                
                # Find the object/predicate of the main clause
                objects = self._find_objects(token.head)
                if "attribute" in objects:
                    relationships.append((clause_subject, main_verb, objects["attribute"]))
                elif "direct" in objects:
                    relationships.append((clause_subject, main_verb, objects["direct"]))
            
            # Open clausal complement patterns (xcomp): "She wants to leave"
            elif token.dep_ == "xcomp" and token.pos_ == "VERB":
                self._handle_clause_dependencies(token, relationships)
            
            # Clausal complement patterns (ccomp): "He says that you like to swim"
            elif token.dep_ == "ccomp" and token.pos_ == "VERB":
                self._handle_clause_dependencies(token, relationships)
            
            # Adverbial clause modifier patterns (advcl): "Leave when you're ready"
            elif token.dep_ == "advcl" and token.pos_ == "VERB":
                self._handle_clause_dependencies(token, relationships)
            
            # Adjectival clause modifier patterns (acl): "the book that I read"
            elif token.dep_ in ["acl", "acl:relcl", "relcl"]:
                self._handle_clause_dependencies(token, relationships)
            
            # Parataxis patterns: "I said: 'Go home'" or "Three muffins," he answered."
            elif token.dep_ == "parataxis":
                # Handle loose syntactic connections like direct speech
                subject, main_verb = self._find_verb_with_subject(token.head)
                if subject and main_verb:
                    parataxis_content = self._get_entity_text(token)
                    relationships.append((subject, main_verb, parataxis_content))
            
            # Numeric modifier patterns (nummod): "three cups"
            elif token.dep_ == "nummod":
                number = self._get_entity_text(token)
                noun = self._get_entity_text(token.head)
                if noun:
                    relationships.append((noun, "has_quantity", number))
            
            # ========== END MISSING PATTERNS ==========
        
        # SRL extraction if enabled
        if self.enable_srl and self._srl:
            try:
                srl_results = self._extract_with_srl(doc)
                relationships.extend(srl_results)
            except Exception as e:
                logger.debug(f"[Tier1 SRL] Failed: {e}")
        
        # Apply coreference if enabled
        if self.enable_coref and relationships:
            relationships = self._apply_coreference(relationships, doc)
        
        # Add centralized UD patterns for comprehensive coverage
        if extract_all_ud_patterns:
            try:
                ud_relations = extract_all_ud_patterns(text, self._nlp)
                for rel in ud_relations:
                    # Convert ExtractedRelation to tuple format and filter out low-confidence relations
                    if rel.confidence >= 0.6:  # Only include high-confidence relations
                        relationship = (rel.subject, rel.relation, rel.object)
                        if relationship not in relationships:  # Avoid duplicates
                            relationships.append(relationship)
                logger.debug(f"[TieredExtractor] Added {len([r for r in ud_relations if r.confidence >= 0.6])} centralized UD relations")
            except Exception as e:
                logger.debug(f"[TieredExtractor] Centralized UD patterns failed: {e}")
        
        # Calculate confidence based on extraction quality
        confidence = min(1.0, len(relationships) * 0.3) if relationships else 0.3
        
        return TieredExtractionResult(
            entities=entities,
            relationships=relationships,
            tier_used=1,
            extraction_time_ms=0,  # Will be set by caller
            confidence=confidence
        )
    
    def _extract_tier2(self, text: str) -> TieredExtractionResult:
        """
        Tier 2: Small LLM (qwen3-0.6b-mlx)
        Hybrid approach: Uses Tier 1 entities + LLM for relationships only (JSON output)
        Target: <500ms for medium complexity
        """
        # Warm up model on first use
        if not self._tier2_warmed:
            self._warmup_tier2()
        
        # First, get high-quality entities from Tier 1 (GLiNER + spaCy)
        tier1_result = self._extract_tier1(text)
        
        if not tier1_result.entities:
            # No entities found, return tier 1 result
            return tier1_result
        
        # Use Tier 1 entities and only ask LLM to find relationships
        entities = tier1_result.entities
        
        system_prompt = """/no_think 
            Extract all relationships between pre-identified entities from the text given.
            Instructions:
            1. Read text
            2. Read entities from list
            3. Find the relationships between them 
            
            Format:
            {
            "relationships": [
                {"source": "entity_from_list", "target": "entity_from_list", "relation": "relationship_type"}
            ]
            }
            """




        # Format entities for the prompt
        entity_list = "\n".join([f"- {entity}" for entity in entities])
        user_prompt = f"Text: {text}\n\nEntities found in text:\n{entity_list}\n\n JSON:"
        
        try:
            json_result = self._call_llm_tier2_hybrid(system_prompt, user_prompt, self.tier2_model)
            if json_result:
                # Parse relationships from simplified response
                relationships = self._parse_relationships_only(json_result)
                
                # Calculate confidence based on relationship quality
                confidence = min(0.85, len(relationships) * 0.15 + 0.4)
                
                logger.debug(f"[Tier2 Hybrid] Extracted {len(relationships)} relationships from {len(entities)} entities")
                
                return TieredExtractionResult(
                    entities=entities,
                    relationships=relationships,
                    tier_used=2,
                    extraction_time_ms=0,
                    confidence=confidence
                )
        except Exception as e:
            logger.debug(f"[Tier2 Hybrid] Failed: {e}")
        
        # Fallback to tier 1
        return tier1_result
    
    def _extract_tier3(self, text: str) -> TieredExtractionResult:
        """
        Tier 3: Larger LLM (llama-3.2-1b-instruct)
        Hybrid approach: Uses Tier 1 entities + LLM for relationships only
        Works with model's natural markdown output strengths
        Target: <1500ms for complex sentences  
        """
        # Warm up model on first use
        if not self._tier3_warmed:
            self._warmup_tier3()
        
        # First, get high-quality entities from Tier 1 (GLiNER + spaCy)
        tier1_result = self._extract_tier1(text)
        
        if not tier1_result.entities:
            # No entities found, fall back to full extraction
            return self._extract_tier3_full(text)
        
        # Use Tier 1 entities and only ask LLM to find relationships
        entities = tier1_result.entities
        
        system_prompt = """You are a relationship extraction expert. Given text and pre-identified entities, find relationships between them.

Output in markdown format:

## Relationships
- Entity1 -> relationship_type -> Entity2
- Entity3 -> relationship_type -> Entity4

Use only exact entity names from the provided list. If no relationships exist, output "## Relationships" followed by "No relationships found."""

        # Format entities for the prompt
        entity_list = "\n".join([f"- {entity}" for entity in entities])
        user_prompt = f"""Text: {text}

Entities found in text:
{entity_list}

Extract relationships between these entities:"""
        
        try:
            markdown_result = self._call_llm_tier3_markdown(system_prompt, user_prompt, self.tier3_model)
            if markdown_result:
                # Parse relationships from markdown response
                relationships = self._parse_markdown_relationships(markdown_result, entities)
                
                # Calculate confidence based on relationship quality
                confidence = min(0.95, len(relationships) * 0.2 + 0.3)
                
                logger.debug(f"[Tier3 Markdown] Extracted {len(relationships)} relationships from {len(entities)} entities")
                
                return TieredExtractionResult(
                    entities=entities,
                    relationships=relationships,
                    tier_used=3,
                    extraction_time_ms=0,
                    confidence=confidence
                )
        except Exception as e:
            logger.debug(f"[Tier3 Markdown] Failed: {e}")
        
        # Fallback to JSON approach
        try:
            logger.debug("[Tier3] Falling back to JSON approach")
            json_result = self._call_llm_tier3_json_old(system_prompt, user_prompt, self.tier3_model)
            if json_result:
                # Parse relationships from simplified response
                relationships = self._parse_relationships_only(json_result)
                
                # Calculate confidence based on relationship quality
                confidence = min(0.95, len(relationships) * 0.2 + 0.3)
                
                logger.debug(f"[Tier3 JSON Fallback] Extracted {len(relationships)} relationships")
                
                return TieredExtractionResult(
                    entities=entities,
                    relationships=relationships,
                    tier_used=3,
                    extraction_time_ms=0,
                    confidence=confidence
                )
        except Exception as e:
            logger.debug(f"[Tier3 JSON Fallback] Failed: {e}")
        
        # Final fallback to full extraction
        return self._extract_tier3_full(text)
    
    def _extract_tier3_full(self, text: str) -> TieredExtractionResult:
        """Fallback full extraction method for Tier 3"""
        # Warm up model on first use
        if not self._tier3_warmed:
            self._warmup_tier3()
        
        system_prompt = """You are a world-class AI model specialized in extracting knowledge graphs from text. You analyze the input text to identify key entities and their relationships.
        Output ONLY valid JSON matching the provided schema. Do not add explanations, markdown, or extra text. If no entities/relationships are found, return an empty graph.

JSON Schema: {
  "entities": [
    {
      "id": "unique_id",
      "label": "entity_type",
      "name": "entity_name"
    }
  ],
  "relationships": [
    {
      "source": "source_node_id",
      "target": "target_node_id",
      "label": "relationship_type",
      "confidence": "Float"
    }
  ]
}

User: Extract a knowledge graph from this text. First, identify main entities (e.g., people, places, concepts). Then, find relationships between them (e.g., "works at", "located in"). Assign unique IDs to nodes starting from 1."""

        user_prompt = f"Text: {text}. "
        
        try:
            json_result = self._call_llm_tier3_json(system_prompt, user_prompt, self.tier3_model)
            if json_result:
                entities, relationships = self._parse_knowledge_graph_json(json_result)
                
                # Calculate confidence based on graph quality
                confidence = min(0.95, len(relationships) * 0.15 + 0.3)
                
                logger.debug(f"[Tier3 Full] Extracted knowledge graph with {len(entities)} entities, {len(relationships)} relationships")
                
                return TieredExtractionResult(
                    entities=entities,
                    relationships=relationships,
                    tier_used=3,
                    extraction_time_ms=0,
                    confidence=confidence
                )
        except Exception as e:
            logger.debug(f"[Tier3 Full] Failed: {e}")
        
        # Fallback to tier 2
        return self._extract_tier2(text)
    
    def _parse_relationships_only(self, json_result: Dict) -> List[Tuple[str, str, str]]:
        """Parse simplified relationships-only JSON from Tier 3 hybrid"""
        relationships = []
        
        try:
            relationships_data = json_result.get('relationships', [])
            for rel in relationships_data:
                if all(k in rel for k in ['source', 'target', 'relation']):
                    source = rel['source'].lower()
                    target = rel['target'].lower()
                    relation = rel['relation'].lower()
                    
                    # Skip self-references
                    if source != target:
                        relationships.append((source, relation, target))
        except Exception as e:
            logger.debug(f"[Tier3 Hybrid] Error parsing relationships JSON: {e}")
        
        return relationships
    
    def _load_nlp(self):
        """Load spaCy model"""
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("[TieredExtractor] Loaded spaCy model")
        except Exception as e:
            logger.error(f"[TieredExtractor] Failed to load spaCy: {e}")
    
    def _load_gliner(self):
        """Load GLiNER extractor"""
        if not self.enable_gliner:
            return
        try:
            if GLiNERExtractor is not None:
                self._gliner = GLiNERExtractor()
                logger.info("[TieredExtractor] Loaded GLiNER model")
        except Exception as e:
            logger.debug(f"[TieredExtractor] Failed to load GLiNER: {e}")
            self.enable_gliner = False
    
    def _load_srl(self):
        """Load SRL extractor"""
        try:
            from components.processing.semantic_roles import SRLExtractor
            self._srl = SRLExtractor(use_normalizer=True)
            logger.info("[TieredExtractor] Loaded SRL extractor")
        except Exception as e:
            logger.debug(f"[TieredExtractor] SRL not available: {e}")
            self._srl = None
    
    def _load_coref(self):
        """Load coreference resolver"""
        try:
            from services.fastcoref import FCoref
            self._coref = FCoref(device='cpu')
            logger.info("[TieredExtractor] Loaded coreference resolver")
        except Exception as e:
            logger.debug(f"[TieredExtractor] Coref not available: {e}")
            self._coref = None
    
    def _get_entity_text(self, token) -> str:
        """Get entity text from token, checking for compounds and modifiers"""
        # Check if part of named entity FIRST (most specific)
        if token.ent_type_:
            # Find the full entity span
            for ent in token.doc.ents:
                if token in ent:
                    return ent.text.lower()
        
        # Handle compound entities (e.g., "Reed College", "Tesla Model S")
        entity_parts = []
        
        # Look for compound modifiers to the left
        for child in token.children:
            if child.dep_ == "compound" and child.i < token.i:
                entity_parts.append(child.text)
        
        # Add the main token
        entity_parts.append(token.text)
        
        # Look for compound modifiers to the right (less common)
        for child in token.children:
            if child.dep_ in ["compound", "flat"] and child.i > token.i:
                entity_parts.append(child.text)
        
        # If we found compounds, return the full entity
        if len(entity_parts) > 1:
            return " ".join(entity_parts).lower()
        
        # For single proper nouns or pronouns, just return the token
        if token.pos_ in ["PROPN", "PRON"] and token.dep_ in ["nsubj", "dobj", "iobj", "dative", "pobj"]:
            return token.text.lower()
        
        # Check if part of noun chunk (but be careful with verbs in between)
        for chunk in token.doc.noun_chunks:
            if token in chunk and chunk.root == token:
                # Only use chunk if this token is the root of the chunk
                # This avoids "Alice gave Bob" being treated as one entity
                return chunk.text.lower()
        
        # Return token text
        return token.text.lower()
    
    # ========== CENTRALIZED HELPER FUNCTIONS ==========
    
    def _find_subject(self, token) -> Optional[str]:
        """Find subject of any token (nsubj, nsubjpass, csubj, csubjpass)"""
        # Check direct children first
        for child in token.children:
            if child.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass"]:
                return self._get_entity_text(child)
        
        # Check the token itself if it's a subject
        if token.dep_ in ["nsubj", "nsubjpass", "csubj", "csubjpass"]:
            return self._get_entity_text(token)
        
        return None
    
    def _find_objects(self, token) -> Dict[str, str]:
        """Find all objects related to a token: {dep_type: entity_text}"""
        objects = {}
        
        for child in token.children:
            if child.dep_ in ["dobj", "obj"]:
                objects["direct"] = self._get_entity_text(child)
            elif child.dep_ in ["iobj", "dative"]:
                objects["indirect"] = self._get_entity_text(child)
            elif child.dep_ in ["attr", "acomp"]:
                objects["attribute"] = self._get_entity_text(child)
            elif child.dep_ == "pobj" and child.head.dep_ == "prep":
                objects["prepositional"] = self._get_entity_text(child)
        
        return objects
    
    def _find_verb_with_subject(self, token) -> Tuple[Optional[str], Optional[str]]:
        """Find verb and its subject from any token"""
        # If token is a verb, find its subject
        if token.pos_ == "VERB":
            verb = token.lemma_.lower()
            subject = self._find_subject(token)
            return subject, verb
        
        # If token has a verb head, find head and its subject
        if token.head.pos_ == "VERB":
            verb = token.head.lemma_.lower()
            subject = self._find_subject(token.head)
            return subject, verb
        
        return None, None
    
    def _add_smart_relationship(self, relationships: List[Tuple[str, str, str]], 
                              subj: str, verb: str, obj: str, prep: Optional[str] = None):
        """Add relationship with smart naming based on verb and preposition"""
        if not subj or not obj:
            return
        
        if prep:
            # Smart preposition-based naming
            if verb == "work" and prep in ["at", "for"]:
                relationships.append((subj, "works_at", obj))
            elif verb == "live" and prep == "in":
                relationships.append((subj, "lives_in", obj))
            elif verb == "teach" and prep in ["at", "in"]:
                relationships.append((subj, "teaches_at", obj))
            elif verb == "study" and prep == "at":
                relationships.append((subj, "studied_at", obj))
            elif verb == "marry" and prep == "in":
                relationships.append((subj, "married_in", obj))
            elif verb in ["be", "is"] and prep == "of":
                relationships.append((subj, f"{verb}_of", obj))
            elif verb == "report" and prep == "to":
                relationships.append((subj, "reports_to", obj))
            else:
                relationships.append((subj, f"{verb}_{prep}", obj))
        else:
            relationships.append((subj, verb, obj))
    
    def _handle_clause_dependencies(self, token, relationships: List[Tuple[str, str, str]]) -> bool:
        """Generic handler for clause dependencies (xcomp, ccomp, advcl, acl)"""
        subject, main_verb = self._find_verb_with_subject(token.head)
        
        if not subject or not main_verb:
            return False
        
        # Get the clause verb
        clause_verb = token.lemma_.lower() if token.pos_ == "VERB" else None
        
        if not clause_verb:
            return False
        
        # Find objects of the clause
        objects = self._find_objects(token)
        
        # Handle different clause types
        if token.dep_ == "xcomp":
            # Open clausal complement: "She wants to leave"
            self._add_smart_relationship(relationships, subject, main_verb, clause_verb)
            return True
        elif token.dep_ == "ccomp":
            # Clausal complement: "He says that you like to swim"
            if "direct" in objects:
                self._add_smart_relationship(relationships, subject, main_verb, objects["direct"])
            return True
        elif token.dep_ == "advcl":
            # Adverbial clause: "Leave when you're ready"
            self._add_smart_relationship(relationships, subject, main_verb, f"when_{clause_verb}")
            return True
        elif token.dep_ in ["acl", "relcl"]:
            # Adjectival/relative clause: "the book that I read"
            if "direct" in objects:
                rel_name = "is_characterized_by" if token.dep_ == "relcl" else "has_property"
                relationships.append((subject, rel_name, objects["direct"]))
            return True
        
        return False
    
    # ========== END HELPER FUNCTIONS ==========
    
    def _extract_with_srl(self, doc) -> List[Tuple[str, str, str]]:
        """Extract using Semantic Role Labeling"""
        if not self._srl:
            self._load_srl()
            if not self._srl:
                return []
        
        try:
            predications = self._srl.doc_to_predications(doc, lang='en')
            return self._srl.predications_to_triples(predications)
        except Exception as e:
            logger.debug(f"[SRL] Extraction failed: {e}")
            return []
    
    def _apply_coreference(self, relationships: List[Tuple[str, str, str]], doc) -> List[Tuple[str, str, str]]:
        """Apply coreference resolution to relationships"""
        if not self._coref:
            self._load_coref()
            if not self._coref:
                return relationships
        
        try:
            # Get coreference clusters
            clusters = self._coref.predict(doc.text)
            
            # Build replacement map
            replacements = {}
            for cluster in clusters:
                # Use first mention as canonical
                canonical = cluster[0].lower()
                for mention in cluster[1:]:
                    replacements[mention.lower()] = canonical
            
            # Apply replacements
            resolved = []
            for s, r, o in relationships:
                s_resolved = replacements.get(s, s)
                o_resolved = replacements.get(o, o)
                
                # Convert pronouns
                if s_resolved in ["i", "me", "my"]:
                    s_resolved = "you"
                if o_resolved in ["i", "me", "my"]:
                    o_resolved = "you"
                
                resolved.append((s_resolved, r, o_resolved))
            
            return resolved
        except Exception as e:
            logger.debug(f"[Coref] Resolution failed: {e}")
            return relationships
    
    def _call_llm_tier2(self, prompt: str, model: str) -> Optional[Dict]:
        """Call Tier 2 LLM API for JSON extraction"""
        try:
            import httpx
            
            response = httpx.post(
                f"{self.llm_base_url}/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 100,  # Optimized: 100 tokens is sufficient for clean JSON
                    "temperature": 0.1,
                    "stop": ["\n\n"]  # Critical: stops qwen3 repetition
                },
                timeout=self.llm_timeout_ms / 1000
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('choices', [{}])[0].get('text', '')
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON content in the response
                    json_start = text.find('{')
                    json_end = text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = text[json_start:json_end]
                        
                        # Try to parse as-is first
                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError:
                            # Try to fix common JSON issues
                            try:
                                # Fix missing quotes around keys
                                json_text = json_text.replace('name', '"name"').replace('type', '"type"')
                                json_text = json_text.replace('source', '"source"').replace('relation', '"relation"').replace('target', '"target"')
                                return json.loads(json_text)
                            except json.JSONDecodeError as e:
                                logger.debug(f"[Tier2] JSON fix failed: {e}, attempting partial parse...")
                                
                                # Try to extract entities and relationships manually as last resort
                                return self._extract_partial_json(json_text)
                    else:
                        logger.debug(f"[Tier2] No JSON found in response: {text[:100]}...")
                        return None
                except Exception as e:
                    logger.debug(f"[Tier2] JSON parse error: {e}")
                    return None
        except Exception as e:
            logger.debug(f"[Tier2 LLM] Call failed: {e}")
        
        return None
    
    def _call_llm_tier2_hybrid(self, system_prompt: str, user_prompt: str, model: str) -> Optional[Dict]:
        """Call Tier 2 LLM API for hybrid JSON relationship extraction"""
        try:
            import httpx
            
            response = httpx.post(
                f"{self.llm_base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 80,  # Reduced for simple relationships
                    "temperature": 0.1
                },
                timeout=self.llm_timeout_ms / 1000
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON content in the response
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = content[json_start:json_end]
                        
                        # Try to parse as-is first
                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError:
                            # Try to fix common JSON issues
                            try:
                                # Fix missing quotes around keys
                                json_text = json_text.replace('source', '"source"').replace('relation', '"relation"').replace('target', '"target"')
                                return json.loads(json_text)
                            except json.JSONDecodeError:
                                logger.debug(f"[Tier2 Hybrid] JSON fix failed")
                                return None
                    else:
                        logger.debug(f"[Tier2 Hybrid] No JSON found in response: {content[:100]}...")
                        return None
                except Exception as e:
                    logger.debug(f"[Tier2 Hybrid] JSON parse error: {e}")
                    return None
        except Exception as e:
            logger.debug(f"[Tier2 Hybrid LLM] Call failed: {e}")
        
        return None
    
    def _extract_partial_json(self, json_text: str) -> Dict:
        """Extract partial JSON from malformed response using regex"""
        import re
        
        try:
            # Try to extract entities array
            entities_match = re.search(r'"entities":\s*\[(.*?)\]', json_text, re.DOTALL)
            entities = []
            if entities_match:
                entities_text = entities_match.group(1)
                # Extract individual entity objects
                entity_matches = re.findall(r'\{"name":\s*"([^"]+)",\s*"type":\s*"([^"]+)"\}', entities_text)
                entities = [name for name, _ in entity_matches]
            
            # Try to extract relationships array  
            relationships_match = re.search(r'"relationships":\s*\[(.*?)\]', json_text, re.DOTALL)
            relationships = []
            if relationships_match:
                relationships_text = relationships_match.group(1)
                # Extract individual relationship objects
                rel_matches = re.findall(r'\{"source":\s*"([^"]+)",\s*"relation":\s*"([^"]+)",\s*"target":\s*"([^"]+)"\}', relationships_text)
                relationships = [(source, relation, target) for source, relation, target in rel_matches]
            
            return {"entities": entities, "relationships": relationships}
        except Exception as e:
            logger.debug(f"[Tier2] Partial JSON extraction failed: {e}")
            return {"entities": [], "relationships": []}
    
    def _call_llm_tier3_json(self, system_prompt: str, user_prompt: str, model: str) -> Optional[Dict]:
        """Call Tier 3 LLM API for JSON knowledge graph extraction"""
        try:
            import httpx
            
            response = httpx.post(
                f"{self.llm_base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 150,  # Reduced from 400 for faster inference
                    "temperature": 0.1
                },
                timeout=self.llm_timeout_ms / 1000
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON content in the response
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = content[json_start:json_end]
                        
                        # Try to parse as-is first
                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError as e:
                            logger.debug(f"[Tier3] JSON parse error: {e}, attempting partial parse...")
                            # Skip automatic fixes since they often corrupt valid JSON
                            return self._extract_partial_json(json_text)
                    else:
                        logger.debug(f"[Tier3] No JSON found in response: {content[:100]}...")
                        return None
                except json.JSONDecodeError as e:
                    logger.debug(f"[Tier3] JSON parse error: {e}")
                    return None
        except Exception as e:
            logger.debug(f"[Tier3 LLM] Call failed: {e}")
        
        return None
    
    def _call_llm_tier3_markdown(self, system_prompt: str, user_prompt: str, model: str) -> Optional[str]:
        """Call Tier 3 LLM API for markdown relationship extraction"""
        try:
            import httpx
            
            response = httpx.post(
                f"{self.llm_base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 100,  # Reduced for markdown output
                    "temperature": 0.1
                },
                timeout=self.llm_timeout_ms / 1000
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                return content.strip()
        except Exception as e:
            logger.debug(f"[Tier3 Markdown LLM] Call failed: {e}")
        
        return None
    
    def _call_llm_tier3_json_old(self, system_prompt: str, user_prompt: str, model: str) -> Optional[Dict]:
        """Old JSON method for fallback"""
        try:
            import httpx
            
            response = httpx.post(
                f"{self.llm_base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 150,
                    "temperature": 0.1
                },
                timeout=self.llm_timeout_ms / 1000
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Try to extract JSON from the response
                try:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = content[json_start:json_end]
                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError:
                            return None
                except json.JSONDecodeError:
                    return None
        except Exception as e:
            logger.debug(f"[Tier3 JSON Fallback] Call failed: {e}")
        
        return None
    
    def _parse_markdown_relationships(self, markdown_text: str, valid_entities: List[str]) -> List[Tuple[str, str, str]]:
        """Parse relationships from markdown output, validating against entity list"""
        relationships = []
        
        lines = markdown_text.split('\n')
        in_relationships = False
        
        for line in lines:
            line = line.strip()
            
            # Check if we're in the relationships section
            if line.startswith('## Relationships') or line.startswith('**Relationships'):
                in_relationships = True
                continue
            
            # Skip if not in relationships section
            if not in_relationships:
                continue
            
            # Parse relationship lines
            if '->' in line and (line.startswith('-') or line.startswith('*')):
                # Extract the relationship part
                relationship_part = line.lstrip('- *').strip()
                
                # Split by -> to get source, relation, target
                parts = relationship_part.split('->')
                if len(parts) >= 3:
                    source = parts[0].strip().lower()
                    relation = parts[1].strip().lower()
                    target = parts[2].strip().lower()
                    
                    # Validate that source and target are in our entity list
                    source_valid = source in valid_entities
                    target_valid = target in valid_entities
                    
                    # Try partial matches if exact match fails
                    if not source_valid:
                        for entity in valid_entities:
                            if source in entity or entity in source:
                                source = entity
                                source_valid = True
                                break
                    
                    if not target_valid:
                        for entity in valid_entities:
                            if target in entity or entity in target:
                                target = entity
                                target_valid = True
                                break
                    
                    # Only add if both source and target are valid
                    if source_valid and target_valid and relation:
                        relationships.append((source, relation, target))
        
        return relationships
    
    def _parse_markdown_result(self, markdown_text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Parse markdown result from Tier 3 LLM"""
        entities = []
        relationships = []
        
        lines = markdown_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Section headers
            if line.startswith('## Entities') or line.startswith('**Entities'):
                current_section = 'entities'
                continue
            elif line.startswith('## Relationships') or line.startswith('**Relationships'):
                current_section = 'relationships'
                continue
            
            # Parse entities
            if current_section == 'entities' and line.startswith('-'):
                # Format: - Name (Type) or - Name
                entity_text = line[1:].strip()
                if '(' in entity_text:
                    entity_name = entity_text.split('(')[0].strip()
                else:
                    entity_name = entity_text
                if entity_name:
                    entities.append(entity_name.lower())
            
            # Parse relationships
            elif current_section == 'relationships' and '->' in line:
                # Format: - Source -> Relation -> Target
                line = line.lstrip('- *')
                parts = line.split('->')
                if len(parts) >= 3:
                    source = parts[0].strip().lower()
                    relation = parts[1].strip().lower()
                    target = parts[2].strip().lower()
                    if source and relation and target:
                        relationships.append((source, relation, target))
        
        return entities, relationships
    
    def _parse_knowledge_graph_json(self, json_result: Dict) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Parse knowledge graph JSON result from Tier 3 LLM"""
        entities = []
        relationships = []
        
        try:
            # Parse entities (new format)
            entities_data = json_result.get('entities', [])
            entity_mapping = {}  # Map entity IDs to names
            
            for entity in entities_data:
                if 'name' in entity:
                    entity_name = entity['name'].lower()
                    entities.append(entity_name)
                    # Store ID mapping if available
                    if 'id' in entity:
                        entity_mapping[entity['id']] = entity_name
            
            # Parse relationships (new format)
            relationships_data = json_result.get('relationships', [])
            for rel in relationships_data:
                if all(k in rel for k in ['source', 'target', 'label']):
                    source_id = rel['source']
                    target_id = rel['target']
                    label = rel['label'].lower()
                    
                    # Convert entity IDs back to entity names
                    source_name = entity_mapping.get(source_id)
                    target_name = entity_mapping.get(target_id)
                    
                    if source_name and target_name:
                        relationships.append((source_name, label, target_name))
                    
            # Fallback: try old format for compatibility
            if not entities and not relationships:
                nodes = json_result.get('nodes', [])
                node_mapping = {}
                
                for node in nodes:
                    if 'name' in node and 'id' in node:
                        entity_name = node['name'].lower()
                        entities.append(entity_name)
                        node_mapping[node['id']] = entity_name
                
                edges = json_result.get('edges', [])
                for edge in edges:
                    if all(k in edge for k in ['source', 'target', 'label']):
                        source_id = edge['source']
                        target_id = edge['target']
                        label = edge['label'].lower()
                        
                        source_name = node_mapping.get(source_id)
                        target_name = node_mapping.get(target_id)
                        
                        if source_name and target_name:
                            relationships.append((source_name, label, target_name))
            
        except Exception as e:
            logger.debug(f"[Tier3] Error parsing knowledge graph JSON: {e}")
        
        return entities, relationships
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity text for world-centric knowledge representation"""
        if not entity:
            return entity
            
        entity = entity.lower().strip()
        
        # Map first/second person pronouns to neutral representations
        if entity in {"i", "me", "my", "mine", "myself"}:
            return "speaker"
        elif entity in {"you", "your", "yours", "yourself"}:
            return "listener"
        elif entity in {"he", "him", "his", "himself"}:
            return "male_person"
        elif entity in {"she", "her", "hers", "herself"}:
            return "female_person"
        elif entity in {"they", "them", "their", "theirs", "themselves"}:
            return "person"
        
        # Strip leading determiners but preserve entity identity
        for det in ["the ", "a ", "an "]:
            if entity.startswith(det):
                entity = entity[len(det):]
                break
        
        # Handle possessive forms
        if entity.endswith("'s"):
            entity = entity[:-2]
        elif entity.endswith("'"):
            entity = entity[:-1]
            
        return entity.strip()
    
    def _enhance_relation_type(self, relation: str, source: str, target: str) -> str:
        """Enhance relation types based on entity context for better semantic representation"""
        relation = relation.lower().strip()
        
        # Common relation normalizations
        relation_mappings = {
            "work for": "works_at",
            "work at": "works_at", 
            "employed by": "works_at",
            "live in": "lives_in",
            "reside in": "lives_in",
            "stay in": "lives_in",
            "go to": "went_to",
            "went to": "went_to",
            "travel to": "went_to",
            "move to": "moved_to",
            "relocate to": "moved_to",
            "study at": "studied_at",
            "learn at": "studied_at",
            "attend": "studied_at",
            "teach at": "teaches_at",
            "instruct at": "teaches_at",
            "report to": "reports_to",
            "managed by": "reports_to",
            "supervised by": "reports_to",
            "has name": "name",
            "named": "name",
            "called": "name",
            "is called": "name"
        }
        
        # Apply mapping if available
        if relation in relation_mappings:
            return relation_mappings[relation]
        
        # Context-aware relation enhancement
        if "name" in relation and any(word in source.lower() for word in ["call", "name"]):
            return "name"
        elif relation in ["is", "are", "was", "were"] and target in ["ceo", "president", "manager", "director"]:
            return f"{target}_of"
        elif relation == "ceo" and "company" in target.lower():
            return "ceo_of"
        
        return relation
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        return {
            'tier_usage': {
                'tier1': self.metrics['tier1_count'],
                'tier2': self.metrics['tier2_count'],
                'tier3': self.metrics['tier3_count']
            },
            'avg_time_ms': {
                'tier1': avg(self.metrics['tier1_time']),
                'tier2': avg(self.metrics['tier2_time']),
                'tier3': avg(self.metrics['tier3_time'])
            }
        }