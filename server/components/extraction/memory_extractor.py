"""
MemoryExtractor: Dedicated Entity and Relation Extraction Service
=================================================================

Extracted from HotMemory monolith - now focused solely on extraction:
- Entity recognition and mapping
- Dependency pattern extraction  
- Multiple extraction strategies (UD, SRL, ONNX, ReLiK)
- Light entity extraction for retrieval
"""

import os
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from loguru import logger
import spacy
from spacy.tokens import Token, Doc

# Optional extraction components
try:
    from components.processing.semantic_roles import SRLExtractor  # type: ignore
except Exception:
    SRLExtractor = None
try:
    from services.onnx_nlp import OnnxTokenNER, OnnxSRLTagger  # type: ignore
except Exception:
    OnnxTokenNER = None

# Import centralized UD patterns
try:
    from services.ud_utils import extract_all_ud_patterns, ExtractedRelation
except ImportError:
    extract_all_ud_patterns = None
    logger.debug("[MemoryExtractor] UD patterns not available")
    OnnxSRLTagger = None
try:
    from components.extraction.hotmem_extractor import HotMemExtractor  # type: ignore
except Exception:
    HotMemExtractor = None
try:
    from components.extraction.enhanced_hotmem_extractor import EnhancedHotMemExtractor  # type: ignore
except Exception:
    EnhancedHotMemExtractor = None
try:
    from components.extraction.hybrid_spacy_llm_extractor import HybridRelationExtractor  # type: ignore
except Exception:
    HybridRelationExtractor = None
try:
    from components.extraction.gliner_extractor import GLiNERExtractor  # type: ignore
except Exception:
    GLiNERExtractor = None
try:
    from components.extraction.tiered_extractor import TieredRelationExtractor  # type: ignore
except Exception:
    TieredRelationExtractor = None


@dataclass
class ExtractionResult:
    """Result of extraction operation"""
    entities: List[str]
    triples: List[Tuple[str, str, str]]
    negation_count: int
    doc: Optional[Any] = None


class MemoryExtractor:
    """
    Dedicated extraction service for entities and relations.
    Handles all extraction strategies and patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize extractor with configuration"""
        self.config = config
        
        # Extraction strategy flags
        self.use_srl = config.get('use_srl', False)
        self.use_onnx_ner = config.get('use_onnx_ner', False)
        self.use_onnx_srl = config.get('use_onnx_srl', False)
        self.use_relik = config.get('use_relik', False)
        self.use_dspy = config.get('use_dspy', False)
        self.use_gliner = config.get('use_gliner', True)  # Enable GLiNER by default
        
        # Optional extractors (lazy loaded)
        self._srl: Optional[Any] = None
        self._onnx_ner = None
        self._onnx_srl = None
        self._relik = None
        self._dspy_extractor = None
        self._gliner = None
        self._tiered_extractor = None
        
        # Performance tracking
        self.metrics = defaultdict(list)
        
    def extract(self, text: str, lang: str = "en") -> ExtractionResult:
        """
        Main extraction entry point - extracts entities and relations from text
        """
        start = time.perf_counter()
        
        try:
            # Load language model
            doc = _load_nlp(lang)(text) if text else None
            if not doc:
                return ExtractionResult([], [], 0, None)
                
            # Stage 1: Extract using multiple strategies
            entities, triples, neg_count = self._extract_strategies(doc, text)
            
            # Track performance
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics['extraction_ms'].append(elapsed_ms)
            
            return ExtractionResult(entities, triples, neg_count, doc)
            
        except Exception as e:
            logger.error(f"[MemoryExtractor] Extraction failed: {e}")
            return ExtractionResult([], [], 0, None)
    
    def extract_entities_light(self, text: str, entity_index: Optional[Set[str]] = None) -> List[str]:
        """
        Light entity extraction for retrieval context - prioritizes GLiNER for accuracy
        """
        try:
            # Use GLiNER if available for superior entity extraction (96.7% accuracy)
            if self.use_gliner and GLiNERExtractor is not None:
                if self._gliner is None:
                    self._gliner = GLiNERExtractor()
                try:
                    gliner_result = self._gliner.extract(text, entity_index)
                    if gliner_result.entities:
                        logger.debug(f"[GLiNER Light] Extracted {len(gliner_result.entities)} entities")
                        return gliner_result.entities
                except Exception as e:
                    logger.debug(f"[GLiNER Light] Failed: {e}")
            
            # Fallback to spaCy if GLiNER unavailable
            try:
                nlp = _load_nlp("en")
                doc = nlp(text)
                entities = [_canon_entity_text(ent.text) for ent in doc.ents]
                if entities:
                    return entities
            except Exception:
                pass
            
            # Final fallback to pattern matching
            return self._extract_entities_light_fallback(text)
                
        except Exception as e:
            logger.debug(f"[MemoryExtractor] Light entity extraction failed: {e}")
            return self._extract_entities_light_fallback(text)
    
    def _extract_entities_light_fallback(self, text: str) -> List[str]:
        """Fallback entity extraction using simple patterns"""
        import re
        
        # Simple pattern-based extraction
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        words = text.split()
        for word in words:
            # Clean word from punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                entities.append(clean_word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_strategies(self, doc: Doc, text: str) -> Tuple[List[str], List[Tuple[str, str, str]], int]:
        """Extract using multiple complementary strategies"""
        all_entities = set()
        all_triples = []
        neg_count = 0
        
        # Strategy 0: GLiNER entity extraction (if enabled)
        if self.use_gliner and GLiNERExtractor is not None:
            if self._gliner is None:
                self._gliner = GLiNERExtractor()
            try:
                gliner_result = self._gliner.extract(text)
                all_entities.update(gliner_result.entities)
                logger.debug(f"[GLiNER] Extracted {len(gliner_result.entities)} entities")
            except Exception as e:
                logger.debug(f"[GLiNER] Extraction failed: {e}")
        
        # Strategy 1: Tiered extraction (best accuracy)
        if TieredRelationExtractor is not None:
            if self._tiered_extractor is None:
                # Initialize with coref and SRL enabled
                self._tiered_extractor = TieredRelationExtractor(
                    enable_srl=self.use_srl,
                    enable_coref=self.config.get('use_coref', True),
                    llm_base_url=self.config.get('llm_base_url', 'http://127.0.0.1:1234/v1')
                )
            try:
                tiered_result = self._tiered_extractor.extract(text, doc)
                all_entities.update(tiered_result.entities)
                all_triples.extend(tiered_result.relationships)
                logger.debug(f"[Tiered] Extracted {len(tiered_result.relationships)} relationships using tier {tiered_result.tier_used}")
            except Exception as e:
                logger.debug(f"[Tiered] Extraction failed: {e}")
                # Fallback to UD patterns
                entities, triples, neg_count = self._extract_from_doc(doc)
                all_entities.update(entities)
                all_triples.extend(triples)
        else:
            # Fallback: Universal Dependencies extraction (base)
            entities, triples, neg_count = self._extract_from_doc(doc)
            all_entities.update(entities)
            all_triples.extend(triples)
        
        # Strategy 2: SRL enhancement (if enabled)
        if self.use_srl and SRLExtractor is not None and self._srl is None:
            try:
                self._srl = SRLExtractor(use_normalizer=True)
            except Exception:
                pass
                
        if self.use_srl and self._srl:
            try:
                srl_trips = self._srl.extract(doc)
                for (s, r, d) in srl_trips:
                    all_entities.add(_canon_entity_text(s))
                    all_entities.add(_canon_entity_text(d))
                all_triples.extend(srl_trips)
            except Exception:
                pass
        
        # Strategy 3: ONNX enhancement (if enabled)
        if self.use_onnx_ner and self._onnx_ner is None:
            self._init_onnx_ner()
        if self.use_onnx_srl and self._onnx_srl is None:
            self._init_onnx_srl()
            
        # Strategy 4: ReLiK enhancement (if enabled)
        if self.use_relik and self._relik is None:
            self._init_relik()
            
        return list(all_entities), all_triples, neg_count
    
    def _extract_from_doc(self, doc) -> Tuple[List[str], List[Tuple[str, str, str]], int]:
        """Extract entities and triples from spaCy doc using UD patterns"""
        # Use a compact, high-signal subset of the 27 patterns
        # Optimized for speed and low allocation
        ents_set: Set[str] = set()
        triples_list: List[Tuple[str, str, str]] = []
        triples_seen: Set[Tuple[str, str, str]] = set()
        neg_count = 0

        # Build entity map from doc.ents (cheap) and any previously found entities
        entity_map = self._build_entity_map(doc, set())

        def add_triple(s: str, r: str, d: str) -> None:
            if not s or not d or not r:
                return
            t = (s, r, d)
            if t in triples_seen:
                return
            triples_seen.add(t)
            triples_list.append(t)
            ents_set.add(s)
            ents_set.add(d)

        def canon(text: str) -> str:
            try:
                t = (text or "").strip().lower()
            except Exception:
                t = ""
            # Strip simple leading determiners/possessives
            for det in ("the ", "a ", "an ", "my ", "your ", "his ", "her ", "their ", "our ", "its "):
                if t.startswith(det):
                    t = t[len(det):]
                    break
            if t.endswith("'s"):
                t = t[:-2]
            return t

        def get_entity(token: Token) -> str:
            return canon(entity_map.get(token.i, token.text))

        # Core pass: walk tokens and apply compact UD patterns
        for tok in doc:
            dep = tok.dep_

            # Subject-driven extraction: handles active verbs, copulas, and simple preps
            if dep in {"nsubj", "nsubjpass"}:
                subj = get_entity(tok)
                head = tok.head

                # Copular: X is Y (attr/acomp)
                if head.pos_ == "AUX" or any(c.dep_ == "cop" for c in head.children):
                    for ch in head.children:
                        if ch.dep_ in {"attr", "acomp"}:
                            obj = get_entity(ch)
                            # Special: "My/Your/X's name is Y" → (X, has, Z), (Z, name, Y)
                            if tok.text.lower() == "name":
                                print(f"DEBUG: Found 'name' as subject, checking children...")
                                # Find the possessor through possessive relationships
                                possessor = None
                                compound_obj = None
                                
                                print(f"DEBUG: 'name' token children: {[f'{c.text} ({c.dep_})' for c in tok.children]}")
                                for gc in tok.children:
                                    print(f"DEBUG: Checking child '{gc.text}' with dep '{gc.dep_}'")
                                    if gc.dep_ == "poss":
                                        if gc.text.lower() in {"my", "mine"}:
                                            possessor = "speaker"
                                        elif gc.text.lower() in {"your", "yours"}:
                                            possessor = "listener"
                                        else:
                                            possessor = get_entity(gc)
                                        print(f"DEBUG: Found possessor: {possessor}")
                                    elif gc.dep_ == "compound":
                                        compound_obj = get_entity(gc)
                                        print(f"DEBUG: Found compound: {compound_obj}")
                                
                                print(f"DEBUG: possessor={possessor}, compound_obj={compound_obj}, obj={obj}")
                                # If we have both possessor and compound object: "My dog name is Potola"
                                if possessor and compound_obj:
                                    print(f"DEBUG: Adding triples: ({possessor}, has, {compound_obj}) and ({compound_obj}, name, {obj})")
                                    add_triple(possessor, "has", compound_obj)
                                    add_triple(compound_obj, "name", obj)
                                # If only possessor: "My name is Potola"  
                                elif possessor:
                                    print(f"DEBUG: Adding triple: ({possessor}, name, {obj})")
                                    add_triple(possessor, "name", obj)
                                # If only compound object: "Dog name is Potola"
                                elif compound_obj:
                                    print(f"DEBUG: Adding triple: ({compound_obj}, name, {obj})")
                                    add_triple(compound_obj, "name", obj)
                                else:
                                    print(f"DEBUG: Fallback: adding ({subj}, is, {obj})")
                                    # Fallback to regular copula pattern
                                    add_triple(subj, "is", obj)
                            else:
                                add_triple(subj, "is", obj)

                # Active verb with direct object
                elif head.pos_ == "VERB":
                    v_lemma = head.lemma_.lower()

                    # Object relations (dobj/obj)
                    for ch in head.children:
                        if ch.dep_ in {"dobj", "obj"}:
                            obj = get_entity(ch)
                            pred = "has" if v_lemma in {"have", "has", "had", "own"} else v_lemma
                            add_triple(subj, pred, obj)

                    # Prepositional complements: live in, work at, go to, moved from, etc.
                    for ch in head.children:
                        if ch.dep_ == "prep":
                            prep = ch.text.lower()
                            pobj = None
                            for gc in ch.children:
                                if gc.dep_ == "pobj":
                                    pobj = get_entity(gc)
                                    break
                            if not pobj:
                                continue
                            if v_lemma == "live" and prep == "in":
                                add_triple(subj, "lives_in", pobj)
                            elif v_lemma == "work" and prep in {"at", "for"}:
                                add_triple(subj, "works_at", pobj)
                            elif v_lemma in {"go", "went"} and prep == "to":
                                add_triple(subj, "went_to", pobj)
                            elif v_lemma in {"move", "moved"} and prep == "from":
                                add_triple(subj, "moved_from", pobj)
                            elif v_lemma in {"participate", "participated"} and prep == "in":
                                add_triple(subj, "participated_in", pobj)
                            elif v_lemma in {"born", "bear"} and prep == "in":
                                add_triple(subj, "born_in", pobj)
                            else:
                                add_triple(subj, f"{v_lemma}_{prep}", pobj)

                    # Conjoined verbs inherit subject when not explicit
                    for v2 in (c for c in head.children if c.dep_ == "conj" and c.pos_ == "VERB"):
                        subj2 = None
                        for c2 in v2.children:
                            if c2.dep_ in {"nsubj", "nsubjpass"}:
                                subj2 = get_entity(c2)
                                break
                        subj2 = subj2 or subj
                        v2_lemma = v2.lemma_.lower()

                        for ch in v2.children:
                            if ch.dep_ in {"dobj", "obj"}:
                                obj = get_entity(ch)
                                pred = "has" if v2_lemma in {"have", "has", "had", "own"} else v2_lemma
                                add_triple(subj2, pred, obj)
                        for ch in v2.children:
                            if ch.dep_ == "prep":
                                prep = ch.text.lower()
                                pobj = None
                                for gc in ch.children:
                                    if gc.dep_ == "pobj":
                                        pobj = get_entity(gc)
                                        break
                                if not pobj:
                                    continue
                                if v2_lemma == "live" and prep == "in":
                                    add_triple(subj2, "lives_in", pobj)
                                elif v2_lemma == "work" and prep in {"at", "for"}:
                                    add_triple(subj2, "works_at", pobj)
                                elif v2_lemma in {"go", "went"} and prep == "to":
                                    add_triple(subj2, "went_to", pobj)
                                elif v2_lemma in {"move", "moved"} and prep == "from":
                                    add_triple(subj2, "moved_from", pobj)
                                elif v2_lemma in {"participate", "participated"} and prep == "in":
                                    add_triple(subj2, "participated_in", pobj)
                                elif v2_lemma in {"born", "bear"} and prep == "in":
                                    add_triple(subj2, "born_in", pobj)
                                else:
                                    add_triple(subj2, f"{v2_lemma}_{prep}", pobj)

            # Object-driven fallback: if we see an object first
            elif dep in {"dobj", "obj"} and tok.head.pos_ == "VERB":
                obj = get_entity(tok)
                head = tok.head
                subj = None
                for ch in head.children:
                    if ch.dep_ in {"nsubj", "nsubjpass"}:
                        subj = get_entity(ch)
                        break
                if subj:
                    v_lemma = head.lemma_.lower()
                    pred = "has" if v_lemma in {"have", "has", "had", "own"} else v_lemma
                    add_triple(subj, pred, obj)

            # Copular attributes encountered independently
            elif dep in {"attr", "acomp"}:
                attr = get_entity(tok)
                head = tok.head
                for ch in head.children:
                    if ch.dep_ in {"nsubj", "nsubjpass"}:
                        subj = get_entity(ch)
                        add_triple(subj, "is", attr)
                        break

            # Possessives: X's Y → (X, has, Y)
            elif dep == "poss":
                possessor = get_entity(tok)
                possessed = get_entity(tok.head)
                # Map 'my'/'mine' to 'speaker' for world-centric representation
                if possessor in {"my", "mine"}:
                    possessor = "speaker"
                add_triple(possessor, "has", possessed)

            # Adjectival modifiers: nice car → (car, quality, nice)
            elif dep == "amod":
                adj = tok.text.lower()
                head_entity = get_entity(tok.head)
                add_triple(head_entity, "quality", adj)

            # Numeric modifiers: 3 cats → (cats, quantity, 3)
            elif dep == "nummod":
                head_entity = get_entity(tok.head)
                add_triple(head_entity, "quantity", tok.text)

            # Appositions: Jake, my son → (jake, also_known_as, my son)
            elif dep == "appos":
                e1 = get_entity(tok.head)
                e2 = get_entity(tok)
                add_triple(e1, "also_known_as", e2)

            # Conjunctions: A and B → (a, and, b); skip verb-verb
            elif dep == "conj":
                if tok.head.pos_ == "VERB" and tok.pos_ == "VERB":
                    continue
                e1 = get_entity(tok.head)
                e2 = get_entity(tok)
                add_triple(e1, "and", e2)

        # === Missing USGS Pattern Implementations ===
        
        # Adverbial clause modifier (advcl): "He left when I arrived"
        for tok in doc:
            if tok.dep_ == "advcl":
                head_entity = get_entity(tok.head)
                clause_entity = get_entity(tok)
                if head_entity and clause_entity:
                    add_triple(head_entity, "happened_when", clause_entity)
        
        # Clausal complement (ccomp): "I think that he is right"
        for tok in doc:
            if tok.dep_ == "ccomp":
                head_entity = get_entity(tok.head)
                complement_entity = get_entity(tok)
                if head_entity and complement_entity:
                    add_triple(head_entity, "believes", complement_entity)
        
        # Clausal subject (csubj): "That he came surprised me"
        for tok in doc:
            if tok.dep_ == "csubj":
                head_entity = get_entity(tok.head)
                subject_entity = get_entity(tok)
                if head_entity and subject_entity:
                    add_triple(subject_entity, "causes", head_entity)
        
        # Open clausal complement (xcomp): "She wants to go"
        for tok in doc:
            if tok.dep_ == "xcomp":
                head_entity = get_entity(tok.head)
                complement_entity = get_entity(tok)
                if head_entity and complement_entity:
                    add_triple(head_entity, "wants", complement_entity)
        
        # Adverbial modifier (advmod): "He runs quickly"
        for tok in doc:
            if tok.dep_ == "advmod":
                head_entity = get_entity(tok.head)
                adv_entity = get_entity(tok)
                if head_entity and adv_entity:
                    add_triple(head_entity, "manner", adv_entity)
        
        # Nominal modifier (nmod): "the book cover"
        for tok in doc:
            if tok.dep_ == "nmod":
                head_entity = get_entity(tok.head)
                mod_entity = get_entity(tok)
                if head_entity and mod_entity:
                    add_triple(head_entity, "modified_by", mod_entity)
        
        # Agent (agent): "The cat was chased by the dog"
        for tok in doc:
            if tok.dep_ == "agent":
                head_entity = get_entity(tok.head)
                agent_entity = get_entity(tok)
                if head_entity and agent_entity:
                    add_triple(agent_entity, "acted_on", head_entity)
        
        # Object predicate (oprd): "She considers him smart"
        for tok in doc:
            if tok.dep_ == "oprd":
                head_entity = get_entity(tok.head)
                pred_entity = get_entity(tok)
                if head_entity and pred_entity:
                    add_triple(head_entity, "considers", pred_entity)
        
        # Prepositional object (pobj): "in the house"
        for tok in doc:
            if tok.dep_ == "pobj":
                prep_token = None
                for parent in tok.ancestors():
                    if parent.dep_ == "prep":
                        prep_token = parent
                        break
                if prep_token:
                    obj_entity = get_entity(tok)
                    if obj_entity:
                        add_triple("location", prep_token.text, obj_entity)
        
        # Preposition (prep): Handle prepositional phrases
        for tok in doc:
            if tok.dep_ == "prep":
                head_entity = get_entity(tok.head)
                prep_text = tok.text.lower()
                for child in tok.children:
                    if child.dep_ == "pobj":
                        obj_entity = get_entity(child)
                        if head_entity and obj_entity:
                            add_triple(head_entity, f"relation_{prep_text}", obj_entity)
        
        # Root: Main predicate
        for tok in doc:
            if tok.dep_ == "root" and tok.pos_ == "VERB":
                root_entity = get_entity(tok)
                if root_entity:
                    # Find subjects and objects connected to root
                    for child in tok.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subj_entity = get_entity(child)
                            if subj_entity and root_entity:
                                add_triple(subj_entity, "performs_action", root_entity)
                        elif child.dep_ in ["dobj", "obj"]:
                            obj_entity = get_entity(child)
                            if obj_entity and root_entity:
                                add_triple("action", root_entity, obj_entity)
        
        return list(ents_set), triples_list, neg_count
    
    def _build_entity_map(self, doc, entities: Set[str]) -> Dict[int, str]:
        """Build token index to entity mapping"""
        entity_map = {}
        
        # Add pre-defined entities
        for ent in getattr(doc, 'ents', []):
            for token_idx in range(ent.start, ent.end):
                entity_map[token_idx] = ent.text
        
        # Add entities from extraction
        for entity in entities:
            entity_text = entity.lower()
            for token in doc:
                if token.text.lower() == entity_text:
                    entity_map[token.i] = entity_text
                    
        return entity_map
    
    def _init_onnx_ner(self):
        """Initialize ONNX NER model"""
        try:
            ner_model = os.getenv("HOTMEM_ONNX_NER_MODEL", "")
            ner_labels = os.getenv("HOTMEM_ONNX_NER_LABELS", "")
            base_dir = os.path.dirname(__file__)
            
            if ner_model and not os.path.isabs(ner_model):
                ner_model = os.path.abspath(os.path.join(base_dir, ner_model))
            if ner_labels and not os.path.isabs(ner_labels):
                ner_labels = os.path.abspath(os.path.join(base_dir, ner_labels))
                
            ner_tok = os.getenv("HOTMEM_ONNX_NER_TOKENIZER", "bert-base-cased")
            self._onnx_ner = OnnxTokenNER(ner_model, ner_labels, tokenizer_name=ner_tok)
            logger.info("[MemoryExtractor ONNX] NER ready")
        except Exception as e:
            logger.warning(f"[MemoryExtractor ONNX] NER unavailable: {e}")
            self._onnx_ner = None
    
    def _init_onnx_srl(self):
        """Initialize ONNX SRL model"""
        try:
            srl_model = os.getenv("HOTMEM_ONNX_SRL_MODEL", "")
            srl_labels = os.getenv("HOTMEM_ONNX_SRL_LABELS", "")
            srl_tok = os.getenv("HOTMEM_ONNX_SRL_TOKENIZER", "bert-base-cased")
            self._onnx_srl = OnnxSRLTagger(srl_model, srl_labels, tokenizer_name=srl_tok)
            logger.info("[MemoryExtractor ONNX] SRL ready")
        except Exception as e:
            logger.warning(f"[MemoryExtractor ONNX] SRL unavailable: {e}")
            self._onnx_srl = None
    
    def _init_relik(self):
        """Initialize ReLiK extractor"""
        try:
            # Try hybrid extractor first (best quality)
            if HybridRelationExtractor is not None:
                relik_dev = os.getenv("HOTMEM_RELIK_DEVICE", "cpu")
                self._relik = HybridRelationExtractor(device=relik_dev)
                logger.info("[MemoryExtractor ReLiK] Using hybrid spaCy+LLM extractor")
            # Try enhanced replacement second
            elif EnhancedHotMemExtractor is not None:
                relik_dev = os.getenv("HOTMEM_RELIK_DEVICE", "cpu")
                self._relik = EnhancedHotMemExtractor(device=relik_dev)
                logger.info("[MemoryExtractor] Using enhanced HotMem extractor")
            elif HotMemExtractor is not None:
                relik_id = os.getenv("HOTMEM_RELIK_MODEL_ID", "relik-ie/relik-relation-extraction-small")
                relik_dev = os.getenv("HOTMEM_RELIK_DEVICE", "cpu")
                self._relik = HotMemExtractor(model_id=relik_id, device=relik_dev)
                logger.info(f"[MemoryExtractor ReLiK] ready: {relik_id}")
            else:
                logger.warning("[MemoryExtractor ReLiK] No extractor available")
                self._relik = None
        except Exception as e:
            logger.warning(f"[MemoryExtractor ReLiK] unavailable: {e}")
            self._relik = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get extraction performance metrics"""
        return dict(self.metrics)


# Helper functions (extracted from original)
def _canon_entity_text(text: str) -> str:
    """Canonicalize entity text"""
    return text.strip().lower()

def _load_nlp(lang: str = "en"):
    """Load spaCy model with caching"""
    # Simple spaCy model loading with basic error handling
    try:
        # Try to load the English model
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except Exception:
        try:
            # Fallback to loading any available model
            nlp = spacy.load("en_core_web_md")
            return nlp
        except Exception:
            try:
                # Final fallback - create a blank model
                nlp = spacy.blank("en")
                return nlp
            except Exception:
                # If all else fails, return None
                return None

def _extract_entities_light_fallback(self, text: str) -> List[str]:
    """Fallback light entity extraction"""
    # Implementation will be extracted from original
    pass


logger.info("🎯 MemoryExtractor initialized - dedicated extraction service")
logger.info("📊 Strategies: UD, SRL, ONNX, ReLiK, DSPy")
