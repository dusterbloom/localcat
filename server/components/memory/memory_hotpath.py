"""
HotMem: Ultra-fast local memory for voice agents
Full USGS Grammar-to-Graph 27 dependency pattern implementation
Target: <200ms p95 extraction + retrieval
"""

import os
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict, deque
import math
import heapq
from dataclasses import dataclass
import statistics
import json
import urllib.request
import urllib.error

from loguru import logger
import spacy
from spacy.tokens import Token
try:
    from unidecode import unidecode  # type: ignore
except Exception:
    def unidecode(s: str) -> str:  # type: ignore
        return s

from components.memory.memory_store import MemoryStore
from components.memory.memory_intent import get_intent_classifier, get_quality_filter, IntentType
try:
    from components.processing.semantic_roles import SRLExtractor  # type: ignore
except Exception:
    SRLExtractor = None  # Optional SRL layer
try:
    from services.onnx_nlp import OnnxTokenNER, OnnxSRLTagger  # type: ignore
except Exception:
    OnnxTokenNER = None
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
    from services.enhanced_bullet_formatter import EnhancedBulletFormatter  # type: ignore
except Exception:
    EnhancedBulletFormatter = None
try:
    from components.extraction.improved_ud_extractor import ImprovedUDExtractor  # type: ignore
except Exception:
    ImprovedUDExtractor = None
try:
    from fastcoref import FCoref  # type: ignore
except Exception:
    FCoref = None
try:
    from components.memory.memory_decomposer import decompose as _decompose_clauses  # type: ignore
except Exception:
    _decompose_clauses = None  # Optional
try:
    from components.memory.memory_quality import calculate_extraction_confidence as _extra_confidence  # type: ignore
except Exception:
    _extra_confidence = None  # Optional

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
            # Keep lemmatizer+parser across all supported langs; disable only NER/textcat for speed
            if lang == "en":
                nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
            else:
                nlp = spacy.load(f"{lang}_core_news_sm", disable=["ner", "textcat"])
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

# Map first/second person pronouns/determiners to canonical 'you'
_PRON_YOU = {"i", "me", "my", "mine", "myself", "your", "yours", "yourself"}

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

@dataclass
class RecencyItem:
    """Item in recency buffer"""
    s: str  # subject
    r: str  # relation
    d: str  # destination
    text: str  # original text for context
    timestamp: int
    turn_id: int
    score: float = 1.0


class HotMemory:
    """
    Ultra-fast memory with USGS 27 dependency patterns
    All operations target <200ms p95
    """
    
    def __init__(self, store: MemoryStore, max_recency: int = 50):
        self.store = store
        self.user_eid = "you"
        
        # Hot indices (RAM)
        self.entity_index = defaultdict(set)  # entity -> set of (s,r,d) triples
        self.recency_buffer = deque(maxlen=max_recency)  # Recent interactions
        self.entity_cache = {}  # Canonical entity mapping
        self.edge_meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}  # (s,r,d) -> {ts, weight}
        # Enhanced bullet formatter (lazy import to avoid circular imports)
        self.bullet_formatter = None
        
        # Improved UD extractor for quality filtering
        self.ud_processor = ImprovedUDExtractor() if ImprovedUDExtractor else None
        
        # Optional LEANN semantic search
        self.use_leann = os.getenv("HOTMEM_USE_LEANN", "false").lower() in ("1", "true", "yes")
        self.leann_index_path = os.getenv("LEANN_INDEX_PATH", os.path.join(os.path.dirname(__file__), '..', 'data', 'memory_vectors.leann'))
        if self.leann_index_path and not os.path.isabs(self.leann_index_path):
            self.leann_index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.leann_index_path))
        self.leann_complexity = int(os.getenv("HOTMEM_LEANN_COMPLEXITY", "16"))
        self._leann_searcher = None
        self._leann_loaded_mtime = 0.0
        
        # Optional LLM-assisted micro-refiner (tiny model) for hard extractions
        self.assisted_enabled = os.getenv("HOTMEM_LLM_ASSISTED", "false").lower() in ("1","true","yes")
        self.assisted_model = os.getenv("HOTMEM_LLM_ASSISTED_MODEL", "google/gemma-3-270m")
        # Prefer explicit assisted base URL; else reuse summarizer base (LM Studio) if set; else default to LM Studio
        self.assisted_base_url = (
            os.getenv("HOTMEM_LLM_ASSISTED_BASE_URL")
            or os.getenv("SUMMARIZER_BASE_URL")
            or "http://127.0.0.1:1234/v1"
        )
        try:
            self.assisted_timeout_ms = int(os.getenv("HOTMEM_LLM_ASSISTED_TIMEOUT_MS", "120"))
        except Exception:
            self.assisted_timeout_ms = 120
        try:
            self.assisted_max_triples = int(os.getenv("HOTMEM_LLM_ASSISTED_MAX_TRIPLES", "3"))
        except Exception:
            self.assisted_max_triples = 3
        # Retrieval fusion flags
        self.retrieval_fusion = os.getenv("HOTMEM_RETRIEVAL_FUSION", "false").lower() in ("1", "true", "yes")
        self.use_leann_summaries = os.getenv("HOTMEM_USE_LEANN_SUMMARIES", "false").lower() in ("1", "true", "yes")
        
        # Performance tracking
        self.metrics = defaultdict(list)
        self.max_metric_size = 1000
        self._assisted_calls = 0
        self._assisted_success = 0
        self._pending_edge_props = {}
        if self.assisted_enabled:
            try:
                logger.info(f"[HotMem Assisted] enabled model={self.assisted_model} base={self.assisted_base_url}")
            except Exception:
                pass

        # SRL integration (optional, off by default)
        self.use_srl = os.getenv("HOTMEM_USE_SRL", "false").lower() in ("1", "true", "yes")
        self._srl: Optional[Any] = None
        # ONNX NER/SRL (optional, off by default)
        self.use_onnx_ner = os.getenv("HOTMEM_USE_ONNX_NER", "false").lower() in ("1", "true", "yes") and (OnnxTokenNER is not None)
        self.use_onnx_srl = os.getenv("HOTMEM_USE_ONNX_SRL", "false").lower() in ("1", "true", "yes") and (OnnxSRLTagger is not None)
        self._onnx_ner = None
        self._onnx_srl = None
        # ReLiK (optional)
        self.use_relik = os.getenv("HOTMEM_USE_RELIK", "false").lower() in ("1", "true", "yes") and (HotMemExtractor is not None)
        self._relik = None
        # Neural coreference (optional)
        self.use_coref = os.getenv("HOTMEM_USE_COREF", "false").lower() in ("1", "true", "yes") and (FCoref is not None)
        
        # Smart coreference configuration
        self.coref_max_entities = int(os.getenv("HOTMEM_COREF_MAX_ENTITIES", "24"))  # Performance guard
        self._coref_model = None  # Lazy loaded
        self._coref_cache = {}  # Cache resolved text
        
        # DSPy framework integration
        self.use_dspy = os.getenv("HOTMEM_USE_DSPY", "false").lower() in ("1", "true", "yes")
        self._dspy_extractor = None  # Lazy loaded
        
        # Classifier result caching
        self._classifier_cache = {}  # Cache for classifier results
        self._cache_max_size = int(os.getenv("HOTMEM_CACHE_SIZE", "1000"))

    def prewarm(self, lang: str = "en") -> None:
        """Load NLP resources up-front to avoid first-turn latency."""
        try:
            _load_nlp(lang)
            if self.use_srl and SRLExtractor is not None:
                # Initialize SRL and preload relation normalizer embeddings
                if self._srl is None:
                    self._srl = SRLExtractor(use_normalizer=True)
                try:
                    # Touch the normalizer once to load the model
                    if getattr(self._srl, 'normalizer', None) is not None:
                        self._srl.normalizer._ensure_model()
                except Exception:
                    pass
            # ONNX NER prewarm
            if self.use_onnx_ner and self._onnx_ner is None:
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
                    logger.info("[HotMem ONNX] NER ready")
                except Exception as e:
                    logger.warning(f"[HotMem ONNX] NER unavailable: {e}")
                    self._onnx_ner = None
            # ONNX SRL prewarm
            if self.use_onnx_srl and self._onnx_srl is None:
                try:
                    srl_model = os.getenv("HOTMEM_ONNX_SRL_MODEL", "")
                    srl_labels = os.getenv("HOTMEM_ONNX_SRL_LABELS", "")
                    srl_tok = os.getenv("HOTMEM_ONNX_SRL_TOKENIZER", "bert-base-cased")
                    self._onnx_srl = OnnxSRLTagger(srl_model, srl_labels, tokenizer_name=srl_tok)
                    logger.info("[HotMem ONNX] SRL ready")
                except Exception as e:
                    logger.warning(f"[HotMem ONNX] SRL unavailable: {e}")
                    self._onnx_srl = None
            # ReLiK prewarm - Try hybrid extractor first, then enhanced, then original
            if self.use_relik and self._relik is None:
                try:
                    # Try hybrid spaCy+LLM extractor first (best quality)
                    if HybridRelationExtractor is not None:
                        relik_dev = os.getenv("HOTMEM_RELIK_DEVICE", "cpu")
                        self._relik = HybridRelationExtractor(device=relik_dev)
                        logger.info(f"[HotMem ReLiK] Using hybrid spaCy+LLM extractor")
                    # Try enhanced replacement second
                    elif EnhancedHotMemExtractor is not None:
                        relik_dev = os.getenv("HOTMEM_RELIK_DEVICE", "cpu")
                        self._relik = EnhancedHotMemExtractor(device=relik_dev)
                        logger.info(f"[HotMem] Using enhanced HotMem extractor")
                    elif HotMemExtractor is not None:
                        # Fallback to original ReLiK
                        # Encourage MPS fallback to avoid hard crashes
                        try:
                            if os.getenv('HOTMEM_RELIK_DEVICE', '').lower() == 'mps':
                                os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
                        except Exception:
                            pass
                        relik_id = os.getenv("HOTMEM_RELIK_MODEL_ID", "relik-ie/relik-relation-extraction-small")
                        relik_dev = os.getenv("HOTMEM_RELIK_DEVICE", "cpu")
                        self._relik = HotMemExtractor(model_id=relik_id, device=relik_dev)
                        logger.info(f"[HotMem ReLiK] ready: {relik_id}")
                    else:
                        logger.warning("[HotMem ReLiK] No extractor available")
                        self._relik = None
                except Exception as e:
                    logger.warning(f"[HotMem ReLiK] unavailable: {e}")
                    self._relik = None
            # Neural coref prewarm
            if self.use_coref and self._coref_model is None:
                try:
                    device = os.getenv("HOTMEM_COREF_DEVICE", "cpu")
                    self._coref_model = FCoref(device=device)
                    logger.info("[HotMem Coref] Neural coref ready")
                except Exception as e:
                    logger.warning(f"[HotMem Coref] unavailable: {e}")
                    self._coref_model = None
        except Exception:
            pass

    def process_turn(self, text: str, session_id: str, turn_id: int) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Process a conversation turn with intelligent intent analysis
        Returns: (memory_bullets, extracted_triples)
        """
        start = time.perf_counter()
        
        # Language detection first (needed for intent analysis)
        lang = self._detect_language(text) if PYCLD3_AVAILABLE else "en"
        
        # Stage 0: Intent classification for quality guidance
        intent_start = time.perf_counter()
        intent_classifier = get_intent_classifier()
        quality_filter = get_quality_filter()
        intent = intent_classifier.analyze(text, lang)
        self.metrics['intent_ms'].append((time.perf_counter() - intent_start) * 1000)
        
        # Early exit for reactions and pure questions (no fact extraction)
        if intent.intent in {IntentType.REACTION, IntentType.PURE_QUESTION}:
            logger.debug(f"Skipping extraction for {intent.intent.value}: {text[:50]}...")
            # Still retrieve context for responses
            retrieve_start = time.perf_counter()
            entities = self._extract_entities_light(text)
            bullets = self._retrieve_context(text, entities, turn_id, intent=intent)
            self.metrics['retrieval_ms'].append((time.perf_counter() - retrieve_start) * 1000)
            
            # Extract triples even for pure questions (for testing and analysis)
            triples = self._extract(text, lang)[1] if lang else []
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics['total_ms'].append(elapsed_ms)
            return bullets, triples
        
        # Language already detected above
        
        # Stage 0: Apply smart coreference resolution if enabled
        if self.use_coref:
            text = self._apply_coref_smart(text, lang)
        
        # Stage 1: Extract entities and relations
        extract_start = time.perf_counter()
        entities, triples, neg_count, doc = self._extract(text, lang)
        self.metrics['extraction_ms'].append((time.perf_counter() - extract_start) * 1000)
        
        # Stage 1b: Optional LLM-assisted micro-refiner (only for hard cases)
        if self._should_assist(text, triples, doc):
            assist_start = time.perf_counter()
            assisted = []
            try:
                # Provide UD entities to the micro-refiner for relation linking only
                assisted = self._assist_extract(text, entities, triples, session_id=session_id)
            except Exception as e:
                logger.debug(f"[HotMem Assisted] error: {e}")
                assisted = []
            ms = (time.perf_counter() - assist_start) * 1000
            self.metrics['assisted_ms'].append(ms)
            self._assisted_calls += 1
            if assisted:
                self._assisted_success += 1
                # Merge and de-dup
                try:
                    seen = set(map(tuple, triples))
                    for tr in assisted:
                        if tuple(tr) not in seen:
                            triples.append(tuple(tr))
                            seen.add(tuple(tr))
                except Exception:
                    pass
                logger.info(f"[HotMem Assisted] triggered (ms={ms:.0f}, triples={len(assisted)})")
            else:
                logger.info(f"[HotMem Assisted] triggered (ms={ms:.0f}, triples=0)")

        # Stage 2: Refine triples with intent-aware processing
        refine_start = time.perf_counter()
        triples = self._refine_triples(text, triples, doc, intent, lang)
        # Apply neural coreference if enabled, otherwise light coref
        if self.use_coref and self._coref_model is not None and doc is not None and len(triples) <= 24:
            try:
                triples = self._apply_coref_neural(triples, doc)
            except Exception:
                try:
                    triples = self._apply_coref_lite(triples, doc)
                except Exception:
                    pass
        else:
            try:
                triples = self._apply_coref_lite(triples, doc)
            except Exception:
                pass
        # Rebuild entities from refined triples + text context
        ent_from_triples: Set[str] = set()
        for s, r, d in triples:
            ent_from_triples.add(s)
            ent_from_triples.add(d)
        entities = self._refine_entities_from_text(text, list(ent_from_triples))

        # Stage 3: Quality filtering and storage
        update_start = time.perf_counter()
        now_ts = int(time.time() * 1000)
        
        # Filter and store facts based on quality and intent
        stored_triples = []
        # Coarse provenance tag for this turn
        prov_tag = 'ud_only'
        if self.use_srl:
            prov_tag = 'srl_ud'
        if getattr(self, 'use_onnx_srl', False):
            prov_tag = 'onnx_srl_ud'
        # Hedge/negation adjustments
        t_lower = (text or "").lower()
        hedge_terms = ("maybe", "i think", "probably", "kinda", "sort of", "not sure", "perhaps", "possibly")
        hedged = any(ht in t_lower for ht in hedge_terms)

        conf_thresh = float(os.getenv("HOTMEM_CONFIDENCE_THRESHOLD", "0.3"))
        use_extra_conf = os.getenv("HOTMEM_EXTRA_CONFIDENCE", "false").lower() in ("1", "true", "yes")

        # Confidence gating helpers
        def _is_basic_fact(subj: str, rel: str, obj: str) -> bool:
            subj = (subj or '').lower().strip()
            rel = (rel or '').lower().strip()
            obj = (obj or '').lower().strip()
            # Avoid storing basic facts when subject is an unresolved pronoun placeholder
            pronoun_like = {"he","she","they","him","her","them","who","whom","whose","which","that"}
            if subj in pronoun_like and subj != 'you':
                return False
            basic_rels = {
                'owns', 'name', 'age', 'lives_in', 'works_at', 'teach_at',
                'born_in', 'moved_from', 'went_to',
                'favorite_color', 'favorite_number', 'also_known_as'
            }
            pet_terms = {'dog', 'cat', 'puppy', 'kitten', 'pet'}
            family_terms = {'son','daughter','child','kids','wife','husband','spouse','partner','brother','sister'}
            asset_terms = {'car','bike','tesla'}
            if rel in basic_rels:
                return True
            # Heuristic: (you, has, <pet/family>)
            if subj == 'you' and rel == 'has' and (obj in pet_terms or obj in family_terms or obj in asset_terms):
                return True
            return False

        bypass_basic = os.getenv('HOTMEM_BYPASS_CONFIDENCE_FOR_BASIC', 'true').lower() in ("1","true","yes")
        basic_floor = float(os.getenv('HOTMEM_CONFIDENCE_FLOOR_BASIC', '0.6'))

        for s, r, d in triples:
            should_store, confidence = quality_filter.should_store_fact(s, r, d, intent)
            # Optional conservative complexity confidence
            if use_extra_conf and _extra_confidence is not None:
                try:
                    confidence = min(confidence, float(_extra_confidence(doc, (s, r, d))))
                except Exception:
                    pass
            # Adjust confidence for hedging/negation context
            if hedged:
                confidence -= 0.2
            if neg_count > 0 and intent.intent != IntentType.CORRECTION:
                confidence -= 0.2
            confidence = max(0.0, min(1.0, confidence))
            if confidence < conf_thresh:
                should_store = False

            # Basic-fact override: store core personal facts even if quality filter is conservative
            if bypass_basic and not should_store:
                if _is_basic_fact(s, r, d) and not hedged and (neg_count == 0 or intent.intent == IntentType.CORRECTION):
                    should_store = True
                    confidence = max(confidence, basic_floor)
                    try:
                        logger.debug(f"[HotMem] Basic-fact override: storing ({s}, {r}, {d}) with confidence {confidence:.2f}")
                    except Exception:
                        pass
            if should_store:
                # Handle corrections by demoting old facts
                if intent.intent == IntentType.CORRECTION:
                    self._handle_fact_correction(s, r, d, confidence, now_ts)
                else:
                    self.store.observe_edge(s, r, d, confidence, now_ts)
                # Persist edge metadata (provenance, language, props)
                try:
                    props = self._pending_edge_props.get((s, r, d)) or {}
                    self.store.enqueue_edge_meta(s, r, d, prov=prov_tag, lang=lang, span=None, props=props, ts=now_ts)
                except Exception:
                    pass
                # Alias expansion: when we learn (you, name, X), map X -> you for retrieval
                try:
                    if (s == self.user_eid) and (r == 'name') and d:
                        self.store.enqueue_alias(str(d), self.user_eid)
                        # Also persist an explicit alias edge for immediate in-memory expansion
                        self.store.observe_edge(self.user_eid, 'also_known_as', str(d), max(confidence, 0.6), now_ts)
                        self.entity_index[self.user_eid].add((self.user_eid, 'also_known_as', str(d)))
                        self.entity_index[str(d)].add((self.user_eid, 'also_known_as', str(d)))
                        self.store.flush_if_needed()
                except Exception:
                    pass
                # Cross-lingual transliteration aliasing (fully local)
                try:
                    if r in {'name', 'also_known_as'} and d:
                        d_txt = str(d)
                        d_ascii = unidecode(d_txt)
                        if d_ascii and d_ascii != d_txt and len(d_ascii) >= 3:
                            self.store.enqueue_alias(d_ascii, d_txt)
                except Exception:
                    pass
                    
                # Update hot indices
                self.entity_index[s].add((s, r, d))
                self.entity_index[d].add((s, r, d))
                stored_triples.append((s, r, d))
            else:
                logger.debug(f"Filtered low-quality fact: ({s}, {r}, {d}) confidence={confidence:.2f}")
                
        self.metrics['update_ms'].append((time.perf_counter() - update_start) * 1000)
        
        # Stage 3: Retrieve relevant memories
        retrieve_start = time.perf_counter()
        bullets = self._retrieve_context(text, entities, turn_id, intent=intent)
        self.metrics['retrieval_ms'].append((time.perf_counter() - retrieve_start) * 1000)
        
        # Update recency with stored triples only
        for s, r, d in stored_triples:
            self.recency_buffer.append(RecencyItem(s, r, d, text, now_ts, turn_id))
            # Update hot edge metadata for recency/weight (weight approx if unknown)
            em = self.edge_meta.get((s, r, d), {})
            em['ts'] = now_ts
            em['weight'] = max(0.3, 0.7)
            # Preserve provenance/props if set earlier in the turn
            if 'prov' not in em:
                em['prov'] = prov_tag
            props = self._pending_edge_props.get((s, r, d)) or {}
            if props:
                try:
                    if 'props' in em and isinstance(em['props'], dict):
                        em['props'].update(props)
                    else:
                        em['props'] = dict(props)
                except Exception:
                    em['props'] = props
            self.edge_meta[(s, r, d)] = em
        
        # Track overall performance
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics['total_ms'].append(elapsed_ms)
        self._cleanup_metrics()
        
        if elapsed_ms > 200:
            logger.warning(f"Hot path took {elapsed_ms:.1f}ms (budget: 200ms) - intent: {intent.intent.value}")
        
        return bullets, stored_triples
    
    def _extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """
        Extract entities and relations.
        - If HOTMEM_USE_SRL=true and SRLExtractor available, run SRL-first and fuse with UD 27 patterns.
        - Otherwise, run UD 27 patterns only.
        Returns: (entities, triples, negation_count, original_doc)
        """
        nlp = _load_nlp(lang)
        if not nlp:
            return [], [], 0, None

        doc = nlp(text)

        # Collect negation count once
        neg_total = sum(1 for t in doc if t.dep_ == "neg")

        ents_all: Set[str] = set()
        trips_all: List[Tuple[str, str, str]] = []

        # ReLiK-first path (if enabled)
        if self.use_relik and self._relik is not None:
            try:
                # Gating: short texts only (intent not available in this scope)
                max_chars = int(os.getenv('HOTMEM_RELIK_MAX_CHARS', '480'))
                if len(text or '') > max_chars:
                    raise RuntimeError('ReLiK gated: text length')
                _t0 = time.perf_counter()
                relik_triples = self._relik.extract(text) or []
                relik_ms = (time.perf_counter() - _t0) * 1000
                try:
                    self.metrics['relik_ms'].append(relik_ms)
                    if len(self.metrics['relik_ms']) > self.max_metric_size:
                        self.metrics['relik_ms'] = self.metrics['relik_ms'][-self.max_metric_size:]
                except Exception:
                    pass
                added = 0
                for (s, r, d, c) in relik_triples:
                    if s and r and d:
                        trips_all.append((s, r, d))
                        ents_all.add(s); ents_all.add(d)
                        # Seed edge_meta with provenance and weight
                        try:
                            em = self.edge_meta.get((s, r, d), {})
                            em['ts'] = em.get('ts', 0)
                            em['weight'] = float(c)
                            em['prov'] = 'relik'
                            self.edge_meta[(s, r, d)] = em
                        except Exception:
                            pass
                        added += 1
                try:
                    logger.info(f"[ReLiK] ms={relik_ms:.1f} triples={len(relik_triples)} added={added}")
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"[HotMem ReLiK] skipped/fail: {e}")

        # ONNX-SRL-first path (if enabled)
        if self.use_onnx_srl and self._onnx_srl is not None:
            try:
                roles_list = self._onnx_srl.extract(text)
                for roles in roles_list:
                    pred = roles.get('predicate', '')
                    s = _canon_entity_text(roles.get('agent', ''))
                    o = _canon_entity_text(roles.get('patient', '') or roles.get('destination', '') or roles.get('location', ''))
                    if s and o:
                        rel = pred.lower().strip()
                        # Heuristic normalization (reuse SRL normalizer if available)
                        if self.use_srl and self._srl is not None:
                            try:
                                rel = self._srl.normalizer.normalize(rel, roles, None)
                            except Exception:
                                pass
                        trips_all.append((s, rel, o))
                        ents_all.add(s); ents_all.add(o)
            except Exception as e:
                logger.debug(f"[HotMem ONNX SRL] failed: {e}")

        # SRL-first path (semantic_roles.py)
        if self.use_srl and SRLExtractor is not None:
            try:
                if self._srl is None:
                    self._srl = SRLExtractor(use_normalizer=True)
                preds = self._srl.doc_to_predications(doc, lang=lang)
                srl_trips = self._srl.predications_to_triples(preds)
                try:
                    logger.debug(f"[HotMem SRL] predications={len(preds)} triples={len(srl_trips)}")
                except Exception:
                    pass
                # Build entities from SRL triples
                for (s, r, d) in srl_trips:
                    ents_all.add(_canon_entity_text(s))
                    ents_all.add(_canon_entity_text(d))
                trips_all.extend(srl_trips)
            except Exception as e:
                logger.debug(f"[HotMem SRL] failed; falling back to UD only: {e}")

        # Always include UD extraction and fuse
        ents_doc, trips_doc, _neg_doc = self._extract_from_doc(doc)
        
        # Apply improved UD processing to raw UD triples before merging
        if self.ud_processor and trips_doc:
            try:
                processed_triples = self.ud_processor.process_triples(trips_doc)
                # Extract just the triples (without confidence for now)
                trips_doc = [(s, r, o) for s, r, o, _ in processed_triples]
                logger.debug(f"[Improved UD] Processed {len(trips_doc)} UD triples from raw extraction")
            except Exception as e:
                logger.debug(f"[Improved UD] Processing failed, using raw UD triples: {e}")
        
        for e in ents_doc:
            ents_all.add(e)
        for t in trips_doc:
            if t not in trips_all:
                trips_all.append(t)

        # Optionally add clause spans (extractions merged/deduped)
        use_decomp = os.getenv("HOTMEM_DECOMPOSE_CLAUSES", "false").lower() in ("1", "true", "yes")
        if use_decomp and _decompose_clauses is not None:
            try:
                spans = _decompose_clauses(doc)
            except Exception:
                spans = []
            for sp in spans or []:
                try:
                    subdoc = sp.as_doc()
                except Exception:
                    subdoc = doc
                ents, trips, _negc = self._extract_from_doc(subdoc)
                ents_all.update(ents)
                for tt in trips:
                    if tt not in trips_all:
                        trips_all.append(tt)

        return list(ents_all), trips_all, neg_total, doc

    def _extract_from_doc(self, doc) -> Tuple[List[str], List[Tuple[str, str, str]], int]:
        """Internal: 27-pattern extraction over a spaCy Doc."""
        entities = set()
        triples: List[Tuple[str, str, str]] = []
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

        return list(entities), triples, neg_count
    
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
        
        # ONNX NER enrichment (add spans + map tokens)
        try:
            if self.use_onnx_ner and self._onnx_ner is not None and doc is not None:
                text = doc.text
                ents_onnx = self._onnx_ner.extract(text)
                for (span_text, label, score, (cs, ce)) in ents_onnx:
                    ent_txt = _canon_entity_text(span_text)
                    if not ent_txt:
                        continue
                    entities.add(ent_txt)
                    # Map spaCy tokens overlapping this char span to the entity
                    try:
                        span = doc.char_span(cs, ce, alignment_mode='expand')
                        if span is not None:
                            for t in span:
                                entity_map[t.i] = ent_txt
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"[HotMem ONNX NER] enrichment failed: {e}")

        # Individual tokens
        for token in doc:
            if token.i not in entity_map:
                if token.pos_ in {"NOUN", "PROPN", "PRON"}:
                    entity_text = _canon_entity_text(token.text)
                    # Canonicalize pronouns
                    if entity_text in _PRON_YOU:
                        entity_text = self.user_eid
                    entities.add(entity_text)
                    entity_map[token.i] = entity_text
        
        return entity_map
    
    # === 27 Dependency Handlers ===
    
    def _get_entity(self, token, entity_map) -> str:
        """Get entity for token"""
        return entity_map.get(token.i, _norm(token.text))

    def _apply_coref_neural(self, triples: List[Tuple[str, str, str]], doc):
        """Resolve pronouns using a neural coref model (fastcoref).
        Conservative resolution: only replace with clear non-pronoun antecedents; keep 'you'.
        """
        if not self._coref_model:
            return triples
        text = doc.text
        try:
            pred = self._coref_model.predict(texts=[text], min_cluster_size=2)
            clusters = pred.get_clusters(as_strings=False)[0] if hasattr(pred, 'get_clusters') else []
        except Exception:
            return triples

        # Build representative mention mapping
        rep_map: Dict[Tuple[int, int], str] = {}
        try:
            for cluster in clusters:
                best_txt = None
                best_len = -1
                for (s_idx, e_idx) in cluster:
                    span = doc[s_idx:e_idx]
                    span_txt = _canon_entity_text(span.text)
                    if span_txt in {"i","me","my","mine","your","yours","yourself","you"}:
                        continue
                    if len(span) == 1 and span[0].pos_ == 'PRON':
                        continue
                    if len(span_txt) > best_len:
                        best_txt = span_txt
                        best_len = len(span_txt)
                if not best_txt and cluster:
                    (s0, e0) = cluster[0]
                    best_txt = _canon_entity_text(doc[s0:e0].text)
                for (s_idx, e_idx) in cluster:
                    rep_map[(s_idx, e_idx)] = best_txt or ""
        except Exception:
            pass

        def resolve_token_string(tok_str: str) -> Optional[str]:
            tnorm = (tok_str or '').strip().lower()
            if tnorm in {"you","i","me","my","mine","your","yours","yourself"}:
                return "you"
            try:
                for token in doc:
                    if _canon_entity_text(token.text) == tnorm:
                        for (s_idx, e_idx), rep in rep_map.items():
                            if s_idx <= token.i < e_idx and rep and rep not in {"you","i","me"}:
                                return rep
                        break
            except Exception:
                pass
            return None

        out: List[Tuple[str, str, str]] = []
        for (s, r, d) in triples:
            rs, rd = s, d
            rep = resolve_token_string(rs)
            if rep:
                rs = rep
            rep = resolve_token_string(rd)
            if rep:
                rd = rep
            out.append((rs, r, rd))
        return out
    
    def _extract_subject(self, token, entity_map, triples, entities):
        """nsubj, nsubjpass - nominal subject"""
        subj = self._get_entity(token, entity_map)
        head = token.head
        
        # Passive: "My son is named Jake"
        if token.dep_ == "nsubjpass" and head.pos_ == "VERB":
            verb = head.lemma_.lower()
            if verb in {"name", "call"}:
                for child in head.children:
                    if child.dep_ == "oprd":
                        name = self._get_entity(child, entity_map)
                        triples.append((subj, "name", name))
                        entities.add(name)
                        # Check for possessive
                        for gc in token.children:
                            if gc.dep_ == "poss" and gc.text.lower() in {"my", "mine"}:
                                triples.append((self.user_eid, "has", subj))
                        return
        
        # Copula: X is Y
        if head.pos_ == "AUX" or any(c.dep_ == "cop" for c in head.children):
            for child in head.children:
                if child.dep_ == "attr":
                    obj = self._get_entity(child, entity_map)
                    # Special: "My name is X"
                    if token.text.lower() == "name":
                        for gc in token.children:
                            if gc.dep_ == "poss" and gc.text.lower() in {"my", "mine"}:
                                triples.append((self.user_eid, "name", obj))
                                entities.add(obj)
                                return
                    triples.append((subj, "is", obj))
                    entities.add(obj)
        
        # Active verb: X verbs Y
        elif head.pos_ == "VERB":
            verb = head.lemma_.lower()
            
            # Direct object
            for child in head.children:
                if child.dep_ in {"dobj", "obj"}:
                    obj = self._get_entity(child, entity_map)
                    pred = "has" if verb in {"have", "has", "had", "own"} else verb
                    triples.append((subj, pred, obj))
                    entities.add(obj)
            
            # Prepositional complement
            for child in head.children:
                if child.dep_ == "prep":
                    prep = child.text.lower()
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            obj = self._get_entity(gc, entity_map)
                            # Special patterns
                            if verb == "live" and prep == "in":
                                triples.append((subj, "lives_in", obj))
                            elif verb == "work" and prep in {"at", "for"}:
                                triples.append((subj, "works_at", obj))
                            elif verb in {"teach", "teaches", "taught", "teaching"} and prep == "at":
                                triples.append((subj, "teach_at", obj))
                            elif verb in {"go", "went"} and prep == "to":
                                triples.append((subj, "went_to", obj))
                            elif verb in {"move", "moved"} and prep == "from":
                                triples.append((subj, "moved_from", obj))
                            elif verb in {"participate", "participated"} and prep == "in":
                                triples.append((subj, "participated_in", obj))
                            elif verb in {"born", "bear"} and prep == "in":
                                triples.append((subj, "born_in", obj))
                            elif verb in {"paint", "painted"}:
                                triples.append((subj, "painted", obj))
                                if prep == "in":  # temporal
                                    continue
                            elif verb in {"read"}:
                                triples.append((subj, "read", obj))
                            else:
                                triples.append((subj, f"{verb}_{prep}", obj))
                            entities.add(obj)

            # Conjoined verbs (inherit subject unless explicit)
            for v2 in [c for c in head.children if c.dep_ == "conj" and c.pos_ == "VERB"]:
                # Prefer explicit subject on the conj verb, else inherit
                subj2 = None
                for c2 in v2.children:
                    if c2.dep_ in {"nsubj", "nsubjpass"}:
                        subj2 = self._get_entity(c2, entity_map)
                        break
                subj2 = subj2 or subj
                verb2 = v2.lemma_.lower()

                # Objects of the conj verb
                for ch in v2.children:
                    if ch.dep_ in {"dobj", "obj"}:
                        obj = self._get_entity(ch, entity_map)
                        pred = "has" if verb2 in {"have", "has", "had", "own"} else verb2
                        triples.append((subj2, pred, obj))
                        entities.add(obj)

                # Prepositional complements of the conj verb
                for ch in v2.children:
                    if ch.dep_ == "prep":
                        prep = ch.text.lower()
                        for gc in ch.children:
                            if gc.dep_ == "pobj":
                                obj = self._get_entity(gc, entity_map)
                                if verb2 == "live" and prep == "in":
                                    triples.append((subj2, "lives_in", obj))
                                elif verb2 == "work" and prep in {"at", "for"}:
                                    triples.append((subj2, "works_at", obj))
                                elif verb2 in {"go", "went"} and prep == "to":
                                    triples.append((subj2, "went_to", obj))
                                elif verb2 in {"move", "moved"} and prep == "from":
                                    triples.append((subj2, "moved_from", obj))
                                elif verb2 in {"participate", "participated"} and prep == "in":
                                    triples.append((subj2, "participated_in", obj))
                                elif verb2 in {"born", "bear"} and prep == "in":
                                    triples.append((subj2, "born_in", obj))
                                else:
                                    triples.append((subj2, f"{verb2}_{prep}", obj))
                                entities.add(obj)
    
    def _extract_object(self, token, entity_map, triples, entities):
        """dobj, obj - direct object"""
        obj = self._get_entity(token, entity_map)
        head = token.head
        
        if head.pos_ == "VERB":
            for child in head.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subj = self._get_entity(child, entity_map)
                    verb = head.lemma_.lower()
                    pred = "has" if verb in {"have", "has", "had"} else verb
                    triples.append((subj, pred, obj))
                    break
    
    def _extract_indirect_object(self, token, entity_map, triples, entities):
        """iobj - indirect object"""
        iobj = self._get_entity(token, entity_map)
        head = token.head
        
        # Find subject
        for child in head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj = self._get_entity(child, entity_map)
                triples.append((subj, f"gave_to", iobj))
                break
    
    def _extract_attribute(self, token, entity_map, triples, entities):
        """attr - attribute (copula complement)"""
        attr = self._get_entity(token, entity_map)
        
        for child in token.head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj_token = child
                subj = self._get_entity(child, entity_map)
                
                # Special handling for possessive copula: "My X is Y" -> (you, X, Y)
                possessive_owner = None
                for poss_child in subj_token.children:
                    if poss_child.dep_ == "poss" and poss_child.text.lower() in {"my", "mine"}:
                        possessive_owner = "you"
                        break
                
                if possessive_owner:
                    # Convert "favorite number" to "favorite_number" relation
                    relation = subj.lower().replace(" ", "_")
                    triples.append((possessive_owner, relation, attr))
                    entities.add(attr)
                    # Skip the normal subject-is-attribute extraction since we handled it
                    return
                else:
                    triples.append((subj, "is", attr))
                break
    
    def _extract_acomp(self, token, entity_map, triples, entities):
        """acomp - adjectival complement (copula complement)"""
        # Handle patterns like "Caroline is single"
        adj = self._get_entity(token, entity_map)
        head = token.head
        
        # Find subject of copula
        for child in head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj = self._get_entity(child, entity_map)
                triples.append((subj, "is", adj))
                entities.add(adj)
                break
    
    def _extract_amod(self, token, entity_map, triples, entities):
        """amod - adjectival modifier"""
        adj = token.text.lower()
        head_entity = self._get_entity(token.head, entity_map)
        
        # Skip quality extraction for possessive copula patterns ("My favorite X is Y")
        # Check if the head is subject of a copula and has possessive
        head_token = token.head
        if head_token.dep_ == "nsubj" and head_token.head.lemma_ in {"be", "is", "am", "are", "was", "were"}:
            # Check if head has possessive child and copula has attribute
            has_possessive = any(child.dep_ == "poss" and child.text.lower() in {"my", "mine"} 
                               for child in head_token.children)
            has_attribute = any(child.dep_ in {"attr", "acomp"} 
                              for child in head_token.head.children)
            if has_possessive and has_attribute:
                return  # Skip meaningless quality extraction
        
        triples.append((head_entity, "quality", adj))
    
    def _extract_advmod(self, token, entity_map, triples, entities):
        """advmod - adverbial modifier"""
        # Usually modifies actions, skip for entities
        pass
    
    def _extract_nummod(self, token, entity_map, triples, entities):
        """nummod - numeric modifier"""
        num = token.text
        head_entity = self._get_entity(token.head, entity_map)
        triples.append((head_entity, "quantity", num))
    
    def _extract_nmod(self, token, entity_map, triples, entities):
        """nmod - nominal modifier"""
        mod = self._get_entity(token, entity_map)
        head_entity = self._get_entity(token.head, entity_map)
        triples.append((head_entity, "modified_by", mod))
    
    def _extract_compound(self, token, entity_map, triples, entities):
        """compound - multiword expression"""
        part = self._get_entity(token, entity_map)
        whole = self._get_entity(token.head, entity_map)
        # Usually forms a single entity, already in entity_map
        pass
    
    def _extract_possessive(self, token, entity_map, triples, entities):
        """poss - possessive"""
        possessor = self._get_entity(token, entity_map)
        possessed = self._get_entity(token.head, entity_map)
        
        if possessor in {"my", "mine"}:
            possessor = self.user_eid
        
        # Skip possessive extraction if this is part of a copula pattern ("My X is Y")
        # Check if the possessed noun is the subject of a copula with an attribute
        head_token = token.head
        if head_token.dep_ == "nsubj" and head_token.head.lemma_ in {"be", "is", "am", "are", "was", "were"}:
            # Check if there's an attribute - if so, skip this possessive extraction
            for sibling in head_token.head.children:
                if sibling.dep_ in {"attr", "acomp"}:
                    return  # Skip extraction, will be handled by attribute extraction
        
        triples.append((possessor, "has", possessed))
        entities.add(possessed)
    
    def _extract_appos(self, token, entity_map, triples, entities):
        """appos - apposition"""
        entity1 = self._get_entity(token.head, entity_map)
        entity2 = self._get_entity(token, entity_map)
        triples.append((entity1, "also_known_as", entity2))
    
    def _extract_conj(self, token, entity_map, triples, entities):
        """conj - conjunction"""
        # Skip verb-verb conjunctions (e.g., "live and work") to reduce noise
        if token.head.pos_ == "VERB" and token.pos_ == "VERB":
            return
        item1 = self._get_entity(token.head, entity_map)
        item2 = self._get_entity(token, entity_map)
        triples.append((item1, "and", item2))
    
    def _extract_prep(self, token, entity_map, triples, entities):
        """prep - preposition (handled in subject extraction)"""
        pass
    
    def _extract_acl(self, token, entity_map, triples, entities):
        """acl - adnominal clause"""
        # Complex clausal relation
        pass
    
    def _extract_advcl(self, token, entity_map, triples, entities):
        """advcl - adverbial clause"""
        # Complex clausal relation
        pass
    
    def _extract_ccomp(self, token, entity_map, triples, entities):
        """ccomp - clausal complement"""
        # Handle patterns like "Melanie has read [Nothing is Impossible]"
        head = token.head
        if head.pos_ == "VERB":
            verb = head.lemma_.lower()
            if verb in {"read", "write", "say", "think", "know"}:
                # Find subject of main verb
                for child in head.children:
                    if child.dep_ in {"nsubj", "nsubjpass"}:
                        subj = self._get_entity(child, entity_map)
                        # Extract the clause as object
                        obj_tokens = []
                        for desc in token.subtree:
                            obj_tokens.append(desc.text)
                        obj = " ".join(obj_tokens).strip()
                        if obj:
                            obj = _canon_entity_text(obj)
                            triples.append((subj, verb, obj))
                            entities.add(obj)
                        break
    
    def _extract_csubj(self, token, entity_map, triples, entities):
        """csubj - clausal subject"""
        # Complex clausal relation
        pass
    
    def _extract_xcomp(self, token, entity_map, triples, entities):
        """xcomp - open clausal complement"""
        # Handle patterns like "likes reading", "wants to go"
        head = token.head
        if head.pos_ == "VERB":
            # Find subject of main verb (Caroline is nsubj of "likes")
            subj = None
            for child in head.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subj = self._get_entity(child, entity_map)
                    break
            
            if subj:
                obj = self._get_entity(token, entity_map)
                verb = head.lemma_.lower()
                triples.append((subj, verb, obj))
                entities.add(obj)
    
    def _extract_agent(self, token, entity_map, triples, entities):
        """agent - agent (by-phrase in passive)

        Improve from (by, performed, action) → (agent_noun, verb, passive_subject)
        Examples:
          "presentation was delivered by Sarah" → (sarah, deliver, presentation)
          "who was praised by her colleagues" → (colleagues, praise, who)
        Coref later resolves 'who' to the person.
        """
        head = token.head
        verb = head.lemma_.lower() if head is not None else ""
        # Find the actual agent noun (pobj of 'by')
        agent_ent = None
        for ch in token.children:
            if ch.dep_ in {"pobj", "pcomp", "obl"}:
                agent_ent = self._get_entity(ch, entity_map)
                break
        agent_ent = agent_ent or self._get_entity(token, entity_map)
        # Find the passive subject of the verb
        patient = None
        if head is not None:
            for ch in head.children:
                if ch.dep_ == "nsubjpass":
                    patient = self._get_entity(ch, entity_map)
                    break
        # Build triple
        if agent_ent and verb and patient:
            triples.append((agent_ent, verb, patient))
            entities.add(agent_ent)
            entities.add(patient)
        else:
            # Fallback to previous generic form
            action = verb or (head.text.lower() if head is not None else "")
            triples.append((agent_ent, "performed", action))
    
    def _extract_oprd(self, token, entity_map, triples, entities):
        """oprd - object predicate"""
        # In passive constructions like "is named X"
        oprd = self._get_entity(token, entity_map)
        for child in token.head.children:
            if child.dep_ in {"nsubjpass"}:
                subj = self._get_entity(child, entity_map)
                if token.head.lemma_ in {"name", "call"}:
                    triples.append((subj, "name", oprd))
                    entities.add(oprd)
                break
    
    def _format_memory_bullet(self, s: str, r: str, d: str) -> str:
        """Format a memory triple into a human-readable bullet point"""
        # Lazy import and use enhanced formatter if available
        if self.bullet_formatter is None:
            try:
                from services.enhanced_bullet_formatter import EnhancedBulletFormatter
                self.bullet_formatter = EnhancedBulletFormatter()
            except ImportError:
                self.bullet_formatter = False  # Mark as unavailable
        
        if self.bullet_formatter:
            return self.bullet_formatter.format_bullet(s, r, d)
        
        # Fallback to original formatting
        # Handle pronouns and possessives more naturally
        if s == "you":
            if r == "name":
                # Clean up redundant text in name extraction
                clean_d = d
                if "and i work at" in d.lower():
                    # Extract just the name part before " and I work at"
                    name_part = d.split(" and ")[0].strip()
                    if name_part and len(name_part) > 2:
                        clean_d = name_part
                return f"• Your name is {clean_d}"
            elif r == "age":
                # Avoid duplicating "years old" if already in destination
                if "years old" in d.lower():
                    return f"• You are {d}"
                else:
                    return f"• You are {d} years old"
            elif r == "favorite_number":
                return f"• Your favorite number is {d}"
            elif r == "has":
                # Fix grammar issues with quantities
                if "two children" in d.lower():
                    return f"• You have two children"
                elif "three children" in d.lower():
                    return f"• You have three children"
                elif d.startswith("a ") and d.count(" ") == 1:
                    # "a tesla model" -> "a Tesla Model"
                    return f"• You have {d.title()}"
                return f"• You have {d}"
            elif r == "is":
                return f"• You are {d}"
            elif r == "lives_in":
                return f"• You live in {d}"
            elif r == "works_at":
                return f"• You work at {d}"
            elif r == "work_as":
                return f"• You work as a {d}"
            elif r == "born_in":
                return f"• You were born in {d}"
            elif r == "friend_of":
                return f"• You are a friend of {d}"
            elif r == "husband":
                return f"• Your husband is {d}"
            elif r == "wife":  
                return f"• Your wife is {d}"
            elif r == "married_to":
                return f"• You are married to {d}"
            elif r == "spouse":
                return f"• Your spouse is {d}"
            elif r == "partner":
                return f"• Your partner is {d}"
            elif r.startswith("v:"):
                verb = r[2:]
                # Fix verb conjugation for "you"
                if verb.endswith("s") and len(verb) > 2:
                    verb = verb[:-1]  # "enjoys" -> "enjoy"
                return f"• You {verb} {d}"
            else:
                relation = r.replace('_', ' ')
                if relation in ["has", "is"]:
                    return f"• You {relation} {d}"
                else:
                    return f"• Your {relation} is {d}"
        else:
            if r == "name":
                return f"• {s.title()}'s name is {d}"
            elif r == "age":
                # Avoid duplicating "years old" if already in destination
                if "years old" in d.lower():
                    return f"• {s.title()} is {d}"
                else:
                    return f"• {s.title()} is {d} years old"
            elif r == "favorite_number":
                return f"• {s.title()}'s favorite number is {d}"
            elif r == "has":
                dl = d.lower(); sl = s.lower()
                if dl.startswith(sl + "'s ") or dl.startswith(sl + "’s "):
                    pretty = d[0].upper() + d[1:]
                    return f"• {pretty}"
                return f"• {s.title()} has {d}"
            elif r == "is":
                return f"• {s.title()} is {d}"
            elif r == "lives_in":
                return f"• {s.title()} lives in {d}"
            elif r == "works_at":
                return f"• {s.title()} works at {d}"
            elif r == "born_in":
                return f"• {s.title()} was born in {d}"
            elif r == "teach_at":
                return f"• {s.title()} teaches at {d}"
            elif r == "friend_of":
                return f"• {s.title()} is a friend of {d}"
            elif r.startswith("v:"):
                verb = r[2:]
                return f"• {s.title()} {verb} {d}"
            elif "_color" in r:
                # Handle color relations more naturally
                color_type = r.replace("_color", "")
                return f"• {s.title()}'s {color_type} color is {d}"
            # Common verb phrasing templates
            elif r in {"deliver", "delivered"}:
                return f"• {s.title()} delivered {d}"
            elif r in {"praise", "praised"}:
                return f"• {s.title()} praised {d}"
            elif r in {"represent", "represents"}:
                return f"• {s.title()} represents {d}"
            elif r in {"win", "won"}:
                return f"• {s.title()} won {d}"
            elif r in {"contain", "contains"}:
                return f"• {s.title()} contains {d}"
            elif r in {"lead_to", "leads_to", "led_to"}:
                return f"• {s.title()} leads to {d}"
            elif r in {"accept_for", "accepted_for"}:
                return f"• {s.title()} was accepted for {d}"
            elif r in {"join_as", "joined_as"}:
                return f"• {s.title()} joined as {d}"
            else:
                # Generic fallback with better formatting
                relation = r.replace('_', ' ')
                if relation in ["has", "is"]:
                    return f"• {s.title()} {relation} {d}"
                else:
                    return f"• {s.title()}'s {relation} is {d}"
    
    def _retrieve_context(self, query: str, entities: List[str], turn_id: int, intent=None) -> List[str]:
        """
        Retrieve relevant memory bullets for context
        Returns up to 5 most relevant memories based on scoring
        """
        bullets: List[str] = []
        seen_triples: Set[Tuple[str, str, str]] = set()

        # Tokenize query for lexical overlap (simple, fast)
        def _tokens(t: str) -> Set[str]:
            out = set()
            for w in (t or "").lower().split():
                w = ''.join(ch for ch in w if ch.isalnum())
                if len(w) >= 3:
                    out.add(w)
            return out
        qtok = _tokens(query)
        # Query synonym expansion to improve lexical recall
        try:
            ql = (query or '').lower()
            if any(w in qtok for w in {'drive', 'drives', 'driving'}):
                qtok.add('has')  # driving often implies possession in our KB
            if any(w in qtok for w in {'teach', 'teaches', 'teaching'}):
                qtok.add('teach_at')
            if 'work' in qtok or 'works' in qtok:
                qtok.add('works_at')
        except Exception:
            pass

        # Optional: LEANN similarity scores map for this query (KG triples only)
        leann_scores: Dict[Tuple[str, str, str], float] = {}
        if self.use_leann and query and len(query.strip()) >= 2:
            try:
                # Use enhanced LEANN search
                entity_triples = list(all_edges)  # Convert set to list for LEANN
                leann_score_triples = self._retrieve_with_leann_enhancement(query, entity_triples, top_k=64)
                
                # Convert back to scores format for existing logic
                leann_scores = {}
                for triple in leann_score_triples:
                    leann_scores[triple] = 1.0  # Placeholder score
            except Exception:
                leann_scores = {}

        # 1) Prefer fact bullets based on query entities
        #    Put non-'you' entities first; then 'you' if present.
        ent_set = [e for e in entities if e]
        non_you = [e for e in ent_set if e != "you"]
        include_you = any(e == "you" for e in ent_set)
        query_entities = non_you[:4]
        if include_you:
            query_entities.append("you")
        # Expand aliases: if query mentions an alias (also_known_as), include its subject
        try:
            expanded = set(query_entities)
            for ent in list(query_entities):
                if ent in self.entity_index:
                    for (s2, r2, d2) in self.entity_index[ent]:
                        if r2 == 'also_known_as' and d2 == ent:
                            expanded.add(s2)
                # LMDB alias map (fast) → canonical ID
                try:
                    canon = self.store.resolve_alias(ent)
                    if canon:
                        expanded.add(canon)
                except Exception:
                    pass
            query_entities = list(expanded)[:6]
        except Exception:
            pass

        # Gating: avoid injecting memory for generic questions that only mention "you/I"
        # unless the query contains attribute keywords or an explicit remember request.
        t = (query or "").lower()
        attr_keywords = (
            "name", "age", "live", "lives", "hometown", "from",
            "work", "works", "job", "company", "employer",
            "favorite", "colour", "color", "pet", "dog", "cat",
            "spouse", "partner", "wife", "husband", "child", "kid", "kids"
        )
        remember_request = ("remember" in t) or ("save this" in t) or ("note this" in t)
        has_attr = any(kw in t for kw in attr_keywords)
        if include_you and not non_you:
            # you-only reference
            if not has_attr and not remember_request:
                return []

        # Predicate priority (normalized 0..1)
        pred_pri = {
            "name": 1.00,  # Personal identity is highest priority
            "lives_in": 0.98,
            "works_at": 0.95,
            "work_as": 0.93,
            # Family relationships - very high priority
            "husband": 0.92, "wife": 0.92, "married_to": 0.90, 
            "spouse": 0.90, "partner": 0.88,
            "teach_at": 0.87,
            "born_in": 0.85,
            "moved_from": 0.82,
            "participated_in": 0.80,
            "friend_of": 0.78,
            # Identity and equivalence  
            "also_known_as": 0.75, "is": 0.70,
            # Professional relationships
            "founded": 0.68, "ceo_of": 0.65, "manages": 0.62,
            "favorite_color": 0.60, "favorite_number": 0.60,
            "has": 0.55, "owns": 0.55,
        }
        # Contextual boost for possession if query asks about cars/driving
        if any(tok in qtok for tok in {"car", "drive", "drives", "driving"}):
            pred_pri["has"] = max(pred_pri.get("has", 0.6), 0.88)

        # Weights (env tunable later)
        # Temporal-first: recency heavily dominates to ensure fresh facts win
        alpha, beta, gamma, delta = 0.15, 0.60, 0.20, 0.05
        recency_T_ms = 3 * 24 * 3600 * 1000  # 3 days

        # Intent-aware K limit
        K_max = 5
        try:
            from components.memory.memory_intent import IntentType  # type: ignore
            if intent and getattr(intent, 'intent', None) in {IntentType.REACTION, IntentType.PURE_QUESTION}:
                K_max = 2
        except Exception:
            pass

        # Candidate pool: tuples of (score, ts, kind, payload)
        # kind='kg' payload=(s,r,d) | kind='fts' payload=(text) | kind='sem_summary' payload=(text)
        scored_all: List[Tuple[float, int, str, Any]] = []
        now_ms = int(time.time() * 1000)
        # Only inject from a safe canonical relation set; exclude question scaffolding/noise  
        allowed_rels = {
            "name", "age", "favorite_color",
            "lives_in", "works_at", "work_as", "teach_at", "born_in", "moved_from",
            "participated_in", "went_to",
            "friend_of", "owns", "has", "favorite_number",
            # Family and relationship relations
            "husband", "wife", "married_to", "spouse", "partner", 
            "is", "also_known_as",  # Identity and equivalence relations
            # Professional relations
            "founded", "ceo_of", "manages", "reports_to",
        }
        # If the query is explicitly temporal, allow time/duration injections
        if any(tok in qtok for tok in {"when", "year", "date", "time"}):
            allowed_rels = set(allowed_rels) | {"time", "duration"}
        # If the query is causal, allow cause injection
        if any(tok in qtok for tok in {"why", "because", "reason", "cause"}):
            allowed_rels = set(allowed_rels) | {"because_of"}
        blocked_tokens = {"system prompt", "what time", "no-"}

        pronoun_skip = {"he","she","they","him","her","them","who","whom","whose","which"}

        # Expand query entities with related entities from the knowledge graph
        expanded_entities = set(query_entities)
        for entity in query_entities:
            # Add entities that are directly related to this entity
            if entity in self.entity_index:
                for s, r, d in self.entity_index[entity]:
                    # If this entity is mentioned as a destination, add the subject as a related entity
                    if d == entity and s not in expanded_entities:
                        expanded_entities.add(s)
                    # If this entity is the subject, add the destination as a related entity  
                    if s == entity and d not in expanded_entities:
                        expanded_entities.add(d)
        
        # Use expanded entities for retrieval
        for entity in expanded_entities:
            if entity in self.entity_index:
                candidates = list(self.entity_index[entity])
                for s, r, d in candidates:
                    # Skip bullets where subject is a pronoun-like placeholder
                    if s in pronoun_skip:
                        continue
                    # Filter out low-value relations and question scaffolding for injection
                    if r not in allowed_rels:
                        continue
                    sd_text = f"{s} {d}".lower()
                    if any(bt in sd_text for bt in blocked_tokens):
                        continue
                    meta = self.edge_meta.get((s, r, d), {})
                    ts = int(meta.get('ts', 0))
                    age = max(0, now_ms - ts)
                    rec = math.exp(-age / max(1, recency_T_ms)) if ts > 0 else 0.0
                    pri = pred_pri.get(r, 0.5)
                    # Similarity (LEANN or lexical)
                    sem = 0.0
                    if leann_scores:
                        sem = float(leann_scores.get((s, r, d), 0.0))
                    else:
                        stok = _tokens(s) | _tokens(r) | _tokens(d)
                        if qtok and stok:
                            inter = len(qtok & stok)
                            union = len(qtok | stok)
                            sem = inter / union if union else 0.0
                    # Heuristic: if query is about cars/driving and this is a possession that looks like a vehicle, boost semantic
                    if r == 'has' and any(tok in qtok for tok in {'car','drive','drives','driving'}):
                        dl = (d or '').lower()
                        vehicle_cues = {
                            'tesla','model','bmw','audi','ford','toyota','honda','subaru','mercedes','volkswagen','vw',
                            'volvo','kia','hyundai','jeep','chevy','chevrolet','porsche','mustang','civic','accord','camry','prius','sedan','suv','hatchback','coupe'
                        }
                        if any(cue in dl for cue in vehicle_cues):
                            sem = min(1.0, sem + 0.20)
                    w = float(meta.get('weight', 0.3))
                    # Provenance-aware boost (quality-first)
                    prov = str(meta.get('prov', ''))
                    prov_boost_map = {'onnx_srl_ud': 1.00, 'srl_ud': 0.85, 'ud_only': 0.60, 'assisted': -0.30, '': 0.0}
                    prov_boost = prov_boost_map.get(prov, 0.0)
                    # Entity relevance boost: prefer facts that directly involve original query entities
                    entity_rel_boost = 0.0
                    if entity in query_entities:  # Direct match to original query entity
                        entity_rel_boost = 0.15
                    elif s in query_entities or d in query_entities:  # Subject or destination matches original query
                        entity_rel_boost = 0.10
                    eps_prov = 0.05
                    score = alpha * pri + beta * rec + gamma * sem + delta * w + eps_prov * prov_boost + entity_rel_boost
                    scored_all.append((score, ts, 'kg', (s, r, d)))

        # Retrieval fusion: include FTS summary hits and (optional) LEANN summary hits
        if self.retrieval_fusion and query and len(bullets) < K_max:
            # FTS lexical search over summaries/mentions
            try:
                fts_results = self.store.search_fts_detailed(query, limit=12)
            except Exception:
                fts_results = []
            # Tokenizer reused
            for (text_fts, eid_fts, ts_fts) in fts_results:
                if not text_fts:
                    continue
                # Prefer summaries
                is_summary = isinstance(eid_fts, str) and (eid_fts.startswith('summary:') or eid_fts.startswith('session:'))
                pri = 0.50 if is_summary else 0.40
                rec = 0.0
                if ts_fts and ts_fts > 0:
                    age = max(0, now_ms - int(ts_fts))
                    rec = math.exp(-age / max(1, recency_T_ms))
                sem = 0.0
                stok = _tokens(text_fts)
                if qtok and stok:
                    inter = len(qtok & stok)
                    union = len(qtok | stok)
                    sem = inter / union if union else 0.0
                w = 0.3
                sc = alpha * pri + beta * rec + gamma * sem + delta * w
                scored_all.append((sc, int(ts_fts or 0), 'fts', text_fts))

            # LEANN semantic search over summaries if enabled
            if self.use_leann and self.use_leann_summaries and len(scored_all) < 32:
                try:
                    for (text_sem, sc_sem) in self._leann_summary_texts(query, top_k=8):
                        if not text_sem:
                            continue
                        # Use the semantic score directly with small pri/rec bias
                        pri = 0.45
                        rec = 0.0
                        w = 0.3
                        sem = float(sc_sem)
                        sc = alpha * pri + beta * rec + gamma * sem + delta * w
                        scored_all.append((sc, 0, 'sem_summary', text_sem))
                except Exception:
                    pass

        # Threshold (τ) and diversity
        if scored_all:
            scored_all.sort(key=lambda x: (x[0], x[1]), reverse=True)
            scores_only = [s for (s, _ts, _k, _p) in scored_all]
            idx = max(0, int(len(scores_only) * 0.75) - 1)
            tau = scores_only[idx]
            eps = 0.05

            # MMR-based selection for diversity
            lambda_rel = 0.75
            selected: List[Tuple[float, int, str, Any]] = []

            def _tokset(t: str) -> Set[str]:
                return {w for w in (t or '').lower().split() if len(''.join(ch for ch in w if ch.isalnum())) >= 3}

            def _sim(a: Tuple[str, Any], b: Tuple[str, Any]) -> float:
                kind_a, pay_a = a
                kind_b, pay_b = b
                # KG vs KG: prefer different subject/relation
                if kind_a == 'kg' and kind_b == 'kg':
                    (s1, r1, d1) = pay_a
                    (s2, r2, d2) = pay_b
                    sim = 0.0
                    if s1 == s2:
                        sim += 0.6
                    if r1 == r2:
                        sim += 0.3
                    if d1 == d2:
                        sim += 0.1
                    return sim
                # Summary vs Summary: token overlap
                if kind_a != 'kg' and kind_b != 'kg':
                    ta = _tokset(str(pay_a))
                    tb = _tokset(str(pay_b))
                    if not ta or not tb:
                        return 0.0
                    inter = len(ta & tb)
                    union = len(ta | tb)
                    return 0.6 * (inter / union)
                # Cross-type: light similarity based on token overlap of s r d vs text
                if kind_a == 'kg' and kind_b != 'kg':
                    (s1, r1, d1) = pay_a
                    tb = _tokset(str(pay_b))
                    ta = _tokset(f"{s1} {r1} {d1}")
                else:
                    (s1, r1, d1) = pay_b
                    ta = _tokset(str(pay_a))
                    tb = _tokset(f"{s1} {r1} {d1}")
                if not ta or not tb:
                    return 0.0
                inter = len(ta & tb)
                union = len(ta | tb)
                return 0.4 * (inter / union)

            pool = [(sc, ts, k, p) for (sc, ts, k, p) in scored_all if sc >= max(0.0, tau - eps)]
            while pool and len(selected) < K_max:
                best_idx = -1
                best_mmr = -1.0
                for i, (sc, ts, k, p) in enumerate(pool):
                    if k == 'kg':
                        (s, r, d) = p
                        if (s, r, d) in seen_triples:
                            continue
                    # novelty penalty across types
                    max_sim = 0.0
                    for (_sc2, _ts2, k2, p2) in selected:
                        max_sim = max(max_sim, _sim((k, p), (k2, p2)))
                    mmr = lambda_rel * sc - (1 - lambda_rel) * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = i
                if best_idx == -1:
                    break
                chosen = pool.pop(best_idx)
                selected.append(chosen)
                _sc, _ts, k, p = chosen
                if k == 'kg':
                    s, r, d = p
                    bullets.append(self._format_memory_bullet(s, r, d))
                    seen_triples.add((s, r, d))
                else:
                    # Clean one-line snippet for summaries
                    txt = str(p).replace('\n', ' ').strip()
                    if len(txt) > 220:
                        txt = txt[:220].rstrip() + '…'
                    bullets.append(f"• {txt}")
                if len(bullets) >= K_max:
                    return bullets

        # No recency fallback: only inject facts relevant to current turn
        return bullets[:K_max]

    # ---- LEANN helpers ----
    def _ensure_leann_searcher(self):
        if not self.use_leann:
            return None
        try:
            import os as _os
            mtime = _os.path.getmtime(self.leann_index_path)
        except Exception:
            return None
        try:
            if (self._leann_searcher is None) or (mtime > self._leann_loaded_mtime):
                from leann.api import LeannSearcher  # type: ignore
                self._leann_searcher = LeannSearcher(self.leann_index_path)
                self._leann_loaded_mtime = mtime
        except Exception:
            self._leann_searcher = None
        return self._leann_searcher

    def _leann_query_scores(self, query: str, top_k: int = 32) -> Dict[Tuple[str, str, str], float]:
        scores: Dict[Tuple[str, str, str], float] = {}
        searcher = self._ensure_leann_searcher()
        if not searcher:
            return scores
        try:
            results = searcher.search(query, top_k=top_k, complexity=self.leann_complexity)
        except Exception:
            return scores
        # Map by metadata if available; else fallback to rank-based score
        for rank, res in enumerate(results or []):
            try:
                meta = getattr(res, 'metadata', None) or {}
                s = meta.get('src'); r = meta.get('rel'); d = meta.get('dst')
                if s and r and d:
                    sc = getattr(res, 'score', None)
                    if sc is None:
                        sc = getattr(res, 'similarity', None)
                    if sc is None:
                        sc = 1.0 - (rank / float(top_k + 1))
                    scores[(str(s), str(r), str(d))] = float(sc)
            except Exception:
                continue
        return scores

    def _leann_summary_texts(self, query: str, top_k: int = 8) -> List[Tuple[str, float]]:
        """Return (text, score) for summary-like entries from LEANN index.

        Requires the index to contain such entries (see utils/rebuild_leann.py with --include-summaries).
        """
        out: List[Tuple[str, float]] = []
        searcher = self._ensure_leann_searcher()
        if not searcher:
            return out
        try:
            results = searcher.search(query, top_k=top_k, complexity=self.leann_complexity)
        except Exception:
            return out
        for rank, res in enumerate(results or []):
            try:
                meta = getattr(res, 'metadata', None) or {}
                s = meta.get('src'); r = meta.get('rel'); d = meta.get('dst')
                # Skip KG-encoded items; only return free-text summaries/mentions
                if s and r and d:
                    continue
                text = getattr(res, 'text', None)
                if not text:
                    continue
                sc = getattr(res, 'score', None)
                if sc is None:
                    sc = getattr(res, 'similarity', None)
                if sc is None:
                    sc = 1.0 - (rank / float(top_k + 1))
                out.append((str(text), float(sc)))
            except Exception:
                continue
        return out
    
    def _detect_language(self, text: str) -> str:
        """Detect language using env override or pycld3"""
        hint = os.getenv("HOTMEM_LANG")
        if hint and len(hint) >= 2:
            return hint[:2]
        if PYCLD3_AVAILABLE:
            try:
                result = pycld3.get_language(text)
                return result.language[:2] if result.is_reliable else "en"
            except Exception:
                return "en"
        return "en"
    
    def _retrieve_with_leann_enhancement(self, query: str, entity_triples: List[Tuple[str, str, str]], top_k: int = 32) -> List[Tuple[str, str, str]]:
        """Enhanced LEANN retrieval with performance tracking"""
        if not self.use_leann or not query:
            return []
        
        start = time.perf_counter()
        leann_score_triples = []
        
        try:
            # Use existing LEANN search implementation
            leann_scores = self._leann_query_scores(query, top_k=top_k)
            
            # Convert scores to triple list with all_edges check
            all_edges = set(entity_triples)
            leann_score_triples = [
                (s, r, d) for (s, r, d), _ in leann_scores.items()
                if (s, r, d) in all_edges
            ]
            
            # Performance tracking
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics['leann_ms'] = self.metrics.get('leann_ms', [])
            self.metrics['leann_ms'].append(elapsed_ms)
            
            # Success tracking
            if leann_score_triples:
                self.metrics['leann_success_rate'] = self.metrics.get('leann_success_rate', [])
                self.metrics['leann_success_rate'].append(1)
                self.metrics['leann_improvements'] = self.metrics.get('leann_improvements', 0) + 1
            else:
                self.metrics['leann_success_rate'] = self.metrics.get('leann_success_rate', [])
                self.metrics['leann_success_rate'].append(0)
                
            logger.debug(f"[HotMem] LEANN search: {len(leann_score_triples)} results in {elapsed_ms:.1f}ms")
            
        except Exception as e:
            logger.debug(f"[HotMem] LEANN search failed: {e}")
            self.metrics['leann_errors'] = self.metrics.get('leann_errors', 0) + 1
        
        return leann_score_triples

    def _apply_coref_smart(self, text: str, lang: str) -> str:
        """Smart coreference resolution with performance guards"""
        if not self.use_coref or not text:
            return text
        
        # Cache check
        cache_key = hash(text)
        if cache_key in self._coref_cache:
            return self._coref_cache[cache_key]
        
        # Early exit: no pronouns detected
        nlp = _load_nlp(lang)
        if nlp:
            doc = nlp(text)
            pronouns = [t.text.lower() for t in doc if t.pos_ in ['PRON', 'DET'] and t.text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their', 'theirs']]
            if not pronouns:
                self._coref_cache[cache_key] = text
                return text
            
            # Fast path for simple cases - only 1-2 pronouns
            if len(pronouns) <= 2 and len(text.split()) <= 15:
                # Try simple rule-based resolution first
                simple_resolved = self._try_simple_coref(text, doc, pronouns)
                if simple_resolved != text:
                    logger.debug(f"[HotMem] Fast coref: '{text}' -> '{simple_resolved}'")
                    self._coref_cache[cache_key] = simple_resolved
                    return simple_resolved
            
            # Complexity guard: limit FCoref to reasonable sentence lengths
            if len(text.split()) > self.coref_max_entities:
                logger.debug(f"[HotMem] Coref skipped - too complex ({len(text.split())} words)")
                return text
        
        # Lazy load FCoref model
        if self._coref_model is None:
            try:
                from fastcoref import FCoref
                self._coref_model = FCoref()
                logger.info("[HotMem] FCoref model loaded")
            except Exception as e:
                logger.warning(f"[HotMem] FCoref load failed: {e}")
                self.use_coref = False  # Disable on failure
                return text
        
        # Apply coreference resolution
        try:
            # FCoref.predict() returns a list of CorefResult objects
            coref_results = self._coref_model.predict([text])
            if coref_results and len(coref_results) > 0:
                resolved_text = self._resolve_pronouns_with_clusters(text, coref_results[0])
            else:
                resolved_text = text
            
            # Cache result (with size limit)
            if len(self._coref_cache) < 100:  # Limit cache size
                self._coref_cache[cache_key] = resolved_text
            
            logger.debug(f"[HotMem] Coref: '{text}' -> '{resolved_text}'")
            return resolved_text
            
        except Exception as e:
            logger.debug(f"[HotMem] FCoref fallback: {e}")
            return text
    
    def _resolve_pronouns_with_clusters(self, text: str, coref_result) -> str:
        """Resolve pronouns using FCoref cluster information"""
        try:
            # Get coreference clusters as character spans
            clusters = coref_result.get_clusters(as_strings=False)
            if not clusters:
                return text
            
            # Build replacement map: find best representative for each cluster
            replacements = []  # List of (start, end, replacement_text) tuples
            
            for cluster in clusters:
                if len(cluster) < 2:  # Need at least 2 mentions for coreference
                    continue
                    
                # Find the best representative mention (longest non-pronoun)
                best_mention = None
                best_span = None
                best_length = 0
                
                for span_start, span_end in cluster:
                    mention_text = text[span_start:span_end].strip()
                    # Skip if it's a pronoun
                    if mention_text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their', 'theirs']:
                        continue
                    # Prefer longer, more descriptive mentions
                    if len(mention_text) > best_length:
                        best_mention = mention_text
                        best_span = (span_start, span_end)
                        best_length = len(mention_text)
                
                if not best_mention:
                    continue
                    
                # Replace pronouns in this cluster with the best mention
                for span_start, span_end in cluster:
                    mention_text = text[span_start:span_end].strip()
                    if mention_text.lower() in ['he', 'she', 'it', 'they']:
                        replacements.append((span_start, span_end, best_mention))
                    elif mention_text.lower() in ['him', 'her', 'them']:
                        replacements.append((span_start, span_end, best_mention))
                    elif mention_text.lower() in ['his', 'hers', 'its', 'their', 'theirs']:
                        replacements.append((span_start, span_end, f"{best_mention}'s"))
            
            # Apply replacements in reverse order to maintain character positions
            replacements.sort(key=lambda x: x[0], reverse=True)
            resolved_text = text
            
            for start, end, replacement in replacements:
                resolved_text = resolved_text[:start] + replacement + resolved_text[end:]
            
            return resolved_text
            
        except Exception as e:
            logger.debug(f"[HotMem] Pronoun resolution error: {e}")
            return text
    
    def _try_simple_coref(self, text: str, doc, pronouns: List[str]) -> str:
        """Fast rule-based coreference for simple cases"""
        try:
            # Look for named entities in the text 
            entities = [(ent.text, ent.start, ent.end) for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
            if not entities:
                # Look for proper nouns
                entities = [(token.text, token.i, token.i+1) for token in doc if token.pos_ == 'PROPN']
                
            if not entities:
                return text
                
            # Simple replacement: use the first/most recent entity for pronouns
            result = text
            primary_entity = entities[0][0]  # Use first entity as primary
            
            # Replace common pronouns with the primary entity
            replacements = {
                'he': primary_entity if any('he' in pronouns) else None,
                'she': primary_entity if any('she' in pronouns) else None, 
                'it': primary_entity if any('it' in pronouns) else None,
                'they': primary_entity if any('they' in pronouns) else None,
            }
            
            # Apply replacements carefully to maintain sentence structure
            for pronoun, replacement in replacements.items():
                if replacement and pronoun in result.lower():
                    # Only replace if it's a standalone word
                    import re
                    pattern = r'\b' + re.escape(pronoun) + r'\b'
                    result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                    
            return result
            
        except Exception:
            return text
    
    def _extract_with_dspy_fallback(self, text: str, entities: List[str], session_id: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Use DSPy framework for complex extraction cases"""
        
        # Lazy load DSPy extractor
        if self._dspy_extractor is None:
            try:
                from components.ai.dspy_modules import get_dspy_extractor
                self._dspy_extractor = get_dspy_extractor()
                if self._dspy_extractor:
                    logger.info("[HotMem] DSPy extractor loaded")
                else:
                    self.use_dspy = False  # Disable on failure
                    return []
            except Exception as e:
                logger.warning(f"[HotMem] DSPy load failed: {e}")
                self.use_dspy = False  # Disable on failure
                return []
        
        # Use DSPy for extraction
        try:
            dspy_triples = self._dspy_extractor.extract_relationships(text, entities, session_id)
            
            # Filter and validate results
            valid_triples = []
            for subj, pred, obj in dspy_triples:
                if subj and pred and obj and len(pred) > 0:
                    valid_triples.append((str(subj).strip(), str(pred).strip(), str(obj).strip()))
            
            return valid_triples
            
        except Exception as e:
            logger.debug(f"[HotMem] DSPy extraction failed: {e}")
            return []
    
    def _cache_result(self, cache_key: str, result: List[Tuple[str, str, str]]) -> None:
        """Cache extraction result with size management"""
        if len(self._classifier_cache) >= self._cache_max_size:
            # Remove oldest entries (simple FIFO eviction)
            keys_to_remove = list(self._classifier_cache.keys())[:self._cache_max_size // 4]
            for key in keys_to_remove:
                self._classifier_cache.pop(key, None)
            
            logger.debug(f"[HotMem] Cache evicted {len(keys_to_remove)} entries")
        
        self._classifier_cache[cache_key] = result
        
        # Track cache performance
        self.metrics['cache_size'] = len(self._classifier_cache)
        self.metrics['cache_misses'] = self.metrics.get('cache_misses', 0) + 1

    def _cleanup_metrics(self):
        """Keep metrics bounded"""
        for key in self.metrics:
            # Only clean up list-type metrics (time series), skip counters
            if isinstance(self.metrics[key], list) and len(self.metrics[key]) > self.max_metric_size:
                self.metrics[key] = self.metrics[key][-self.max_metric_size:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        results = {}
        for key, values in self.metrics.items():
            if values:
                results[key] = {
                    'mean': statistics.mean(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) > 20 else max(values),
                    'count': len(values)
                }
        
        results['entities'] = len(self.entity_index)
        results['recency_buffer'] = len(self.recency_buffer)
        results['assisted_calls'] = self._assisted_calls
        results['assisted_success'] = self._assisted_success
        
        return results

    # ---------- Assisted Extraction (tiny LLM) ----------
    def _should_assist(self, text: str, ud_triples: List[Tuple[str, str, str]], doc) -> bool:
        if not self.assisted_enabled:
            return False
        try:
            t = (text or '').strip()
            if not t:
                return False
            # Trigger if very few UD triples OR sentence complexity indicators present OR long text
            if len(ud_triples) <= 1:
                return True
            if len(t) > 180:
                return True
            if doc is not None:
                for tok in doc:
                    if tok.dep_ in {"ccomp", "xcomp", "advcl", "mark"}:
                        return True
        except Exception:
            return False
        return False

    def _assist_extract(self, text: str, entities: List[str], ud_triples: List[Tuple[str, str, str]], session_id: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Call a tiny LLM to link relations only between provided entities. Returns list of (s,r,d)."""
        model = self.assisted_model
        base_url = self.assisted_base_url.rstrip('/') + '/chat/completions'
        
        # Determine if this is a classifier model
        is_classifier = 'classifier' in model.lower()
        
        ents = [ _canon_entity_text(e) for e in (entities or []) if _canon_entity_text(e) ]
        ents_unique = []
        seen_e = set()
        for e in ents:
            if e not in seen_e:
                ents_unique.append(e)
                seen_e.add(e)
        
        # Step 1: Run existing classifier/extractor logic
        if is_classifier:
            base_result = self._assist_extract_classifier(text, ents_unique, base_url, model, session_id)
        else:
            base_result = self._assist_extract_json(text, ents_unique, base_url, model, session_id)
        
        # Step 2: DSPy enhancement for low-yield cases
        if self.use_dspy and len(base_result) < 2:  # Low extraction yield threshold
            try:
                dspy_triples = self._extract_with_dspy_fallback(text, entities, session_id)
                logger.debug(f"[HotMem] DSPy fallback added {len(dspy_triples)} triples")
                return base_result + dspy_triples
            except Exception as e:
                logger.debug(f"[HotMem] DSPy enhancement failed: {e}")
        
        return base_result

    def _assist_extract_classifier(self, text: str, entities: List[str], base_url: str, model: str, session_id: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Cached classifier mode: test each entity pair individually for maximum speed."""
        # Check cache first
        cache_key = f"{hash(text)}_{hash('_'.join(sorted(entities)))}"
        
        if cache_key in self._classifier_cache:
            self.metrics['cache_hits'] = self.metrics.get('cache_hits', 0) + 1
            logger.debug(f"[HotMem] Classifier cache hit")
            return self._classifier_cache[cache_key]
        
        out: List[Tuple[str, str, str]] = []
        
        if len(entities) < 2:
            # Cache empty result
            self._cache_result(cache_key, out)
            return out
            
        # Generate all possible entity pairs
        entity_pairs = []
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i != j:
                    entity_pairs.append((entities[i], entities[j]))
        
        # Classifier system message - exact format that works with hotmem-relation-classifier-mlx
        sys_msg = """You are a relation classifier. Given two entities and context, output ONLY the relation type.
If no relation exists, output 'none'.
Valid relations are suggested by the context. Anything can work. Small sample examples: works_at, CEO_of, lives_in, develops, friend_of, etc."""
        
        # Test each entity pair
        for subject, obj in entity_pairs[:10]:  # Limit to prevent too many calls
            try:
                user_msg = f"""Subject: {subject}
Context: {text}
Object: {obj}

What is the relation type?"""
                
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg}
                ]
                
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": 20,
                    "temperature": 0.1
                }
                
                if session_id:
                    payload["session_id"] = f"assist_{session_id}"
                
                data = json.dumps(payload).encode('utf-8')
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','')}"}
                req = urllib.request.Request(base_url, data=data, headers=headers, method='POST')
                timeout = max(0.05, self.assisted_timeout_ms / 1000.0)
                
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode('utf-8')
                    obj = json.loads(body)
                    choice = (obj.get('choices') or [{}])[0]
                content = ((choice.get('message') or {}).get('content') or '').strip()
                    
                if content and content.lower() != 'none':
                        # Clean and normalize relation
                        relation = content.lower().strip()
                        if ',' in relation:
                            relation = relation.split(',')[0]  # Take first if multiple
                        
                        # Remove common prefixes
                        for prefix in ['is_', 'was_', 'the_', 'a_', 'an_']:
                            if relation.startswith(prefix):
                                relation = relation[len(prefix):]
                        
                        # Normalize underscores and spaces
                        relation = relation.replace(' ', '_').replace('-', '_')
                        
                        # Basic validation
                        if relation and len(relation) > 1 and relation.isalnum():
                            out.append((subject, relation, obj))
                            
            except Exception as e:
                if os.getenv('HOTMEM_ASSISTED_LOG_REQUEST','false').lower() in ('1','true','yes'):
                    try:
                        logger.info(f"[HotMem Classifier] Pair ({subject}, {obj}) failed: {e}")
                    except Exception:
                        pass
                continue
        
        # Cache result before returning
        result = out[:self.assisted_max_triples]
        self._cache_result(cache_key, result)
        return result

    def _assist_extract_json(self, text: str, entities: List[str], base_url: str, model: str, session_id: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Traditional JSON extraction mode."""
        # Strict instructions for structured output (enum-free; relation chosen from text)
        sys_msg = (
            "You are a relation linker. Link relations only between the provided ENTITIES based on the USER message.\n"
            "Return exactly ONE JSON object with a top-level array field 'triples'.\n"
            "Each triple item must be an object with keys: s (subject), r (relation), d (destination).\n"
            "Rules:\n"
            "- Use ONLY ENTITIES for s and d (exact match). Do not invent or alter entities. If you cannot match, skip.\n"
            "- r must be a short phrase present in the USER text (1–3 tokens), lowercase, spaces → underscores (e.g., 'teaches at' → 'teaches_at').\n"
            "- If the text clearly states an age like '3 years old' for an entity, you may output r='age' and d='3 years old'.\n"
            "- If no valid relations are found, return {\"triples\": []}.\n"
            "- Output only the JSON object. No extra text. No code fences. No trailing commas.\n"
            "Checklist: s ∈ ENTITIES; d ∈ ENTITIES; r from text (lowercase, underscores); ≤3 triples; JSON parses."
        )
        # Minimal examples
        ex1 = (
            "ENTITIES: [\"dog\",\"luna\"]\n"
            "USER: My dog Luna is 3 years old and loves hiking.\n"
            "ASSISTANT: {\"triples\":[{\"s\":\"dog\",\"r\":\"also_known_as\",\"d\":\"luna\"},{\"s\":\"dog\",\"r\":\"age\",\"d\":\"3 years old\"}]}"
        )
        ex2 = (
            "ENTITIES: [\"tom\",\"reed college\",\"portland\"]\n"
            "USER: Tom lives in Portland and teaches at Reed College.\n"
            "ASSISTANT: {\"triples\":[{\"s\":\"tom\",\"r\":\"lives_in\",\"d\":\"portland\"},{\"s\":\"tom\",\"r\":\"teaches_at\",\"d\":\"reed college\"}]}"
        )
        ents_json = json.dumps(entities)
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "system", "content": ex1},
            {"role": "system", "content": ex2},
            {"role": "system", "content": f"ENTITIES: {ents_json}"},
            {"role": "user", "content": text},
        ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 128,
            "temperature": 0.0,
        }
        # Encourage JSON output in servers that support it
        if os.getenv('HOTMEM_ASSISTED_JSON_MODE','true').lower() in ('1','true','yes'):
            payload["response_format"] = {"type": "json_object"}
        if session_id:
            payload["session_id"] = f"assist_{session_id}"
        data = json.dumps(payload).encode('utf-8')
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','')}"}
        req = urllib.request.Request(base_url, data=data, headers=headers, method='POST')
        timeout = max(0.05, self.assisted_timeout_ms / 1000.0)
        try:
            if os.getenv('HOTMEM_ASSISTED_LOG_REQUEST','false').lower() in ('1','true','yes'):
                try:
                    logger.info(
                        f"[HotMem Assisted] POST {base_url} model={model} session_id={'assist_'+session_id if session_id else ''} "
                        f"ents={len(entities)} timeout_ms={int(self.assisted_timeout_ms)}"
                    )
                    if entities:
                        logger.info(f"[HotMem Assisted] ENTITIES sample: {entities[:6]}")
                except Exception:
                    pass
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode('utf-8')
                obj = json.loads(body)
                choice = (obj.get('choices') or [{}])[0]
                content = ((choice.get('message') or {}).get('content') or '').strip()
                # Some servers may return structured content directly
                if not content and isinstance((choice.get('message') or {}).get('parsed'), dict):
                    parsed = (choice.get('message') or {}).get('parsed')
                    content = json.dumps(parsed)
        except Exception as e:
            try:
                logger.info(f"[HotMem Assisted] HTTP error: {e}")
            except Exception:
                pass
            return []
        if not content:
            return []
        # Optional: log returned content (longer snippet)
        if os.getenv('HOTMEM_ASSISTED_LOG_RAW','false').lower() in ('1','true','yes'):
            try:
                snippet = content if len(content) <= 4000 else (content[:4000] + '…')
                logger.info(f"[HotMem Assisted] raw content: {snippet}")
            except Exception:
                pass
        # Parse JSON object `{ "triples": [ ... ] }` or JSONL fallback
        out: List[Tuple[str, str, str]] = []
        entity_set = set(entities)
        pronouns = {"it","this","that","he","she","they","we"}

        def _match_entity(cand: str) -> Optional[str]:
            c = (cand or '').strip().lower()
            if not c:
                return None
            if c in entity_set:
                return c
            for ent in entities:
                if c and (c in ent) and (len(ent) >= len(c)):
                    return ent
            for ent in entities:
                if ent in c:
                    return ent
            return None
        # Try JSON object first
        try:
            if content.lstrip().startswith('{'):
                obj = json.loads(content)
                arr = obj.get('triples') or []
                if isinstance(arr, list):
                    for rec in arr:
                        try:
                            s = _canon_entity_text(str(rec.get('s','') or ''))
                            r = str(rec.get('r','') or '').strip().lower().replace(' ', '_')
                            d = _canon_entity_text(str(rec.get('d','') or ''))
                            if s in pronouns:
                                continue
                            ms = _match_entity(s)
                            md = _match_entity(d)
                            if ms and md:
                                if r == 'age' and not d.endswith('years old'):
                                    dd = ''.join(ch for ch in d if ch.isdigit())
                                    if dd:
                                        d = f"{dd} years old"
                                out.append((ms,r,md))
                        except Exception:
                            continue
                        if len(out) >= self.assisted_max_triples:
                            break
        except Exception:
            out = []
        # Fallback: JSONL lines
        if not out:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    s = _canon_entity_text(str(rec.get('s','') or ''))
                    r = str(rec.get('r','') or '').strip().lower().replace(' ', '_')
                    d = _canon_entity_text(str(rec.get('d','') or ''))
                    if s in pronouns:
                        continue
                    ms = _match_entity(s)
                    md = _match_entity(d)
                    if ms and md:
                        if r == 'age' and not d.endswith('years old'):
                            # normalize if number only
                            dd = ''.join(ch for ch in d if ch.isdigit())
                            if dd:
                                d = f"{dd} years old"
                        out.append((ms,r,md))
                except Exception:
                    continue
                if len(out) >= self.assisted_max_triples:
                    break
        # If still nothing, note it (already logged raw content above)
        return out
    
    def rebuild_from_store(self):
        """Rebuild hot indices from persistent store"""
        start = time.perf_counter()
        count = 0
        
        logger.info("Rebuilding hot memory from store...")
        
        # Get all edges from store
        edges = self.store.get_all_edges()
        
        for s, r, d, conf in edges:
            if conf > 0.1:  # Only active edges
                self.entity_index[s].add((s, r, d))
                self.entity_index[d].add((s, r, d))
                # Seed meta with weight; ts unknown (0)
                self.edge_meta[(s, r, d)] = {'ts': 0, 'weight': float(conf)}
                count += 1

        # Merge persisted edge metadata if available
        try:
            metas = self.store.get_all_edge_meta()
        except Exception:
            metas = []
        for (s, r, d, meta) in metas:
            try:
                em = self.edge_meta.get((s, r, d), {'ts': 0, 'weight': 0.3})
                for k, v in meta.items():
                    em[k] = v
                self.edge_meta[(s, r, d)] = em
            except Exception:
                continue
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Rebuilt {count} edges in {elapsed_ms:.1f}ms")

    # ---------- Non-mutating helpers for verification ----------
    def preview_bullets(self, text: str, lang: str = "en") -> Dict[str, Any]:
        """Return entities detected and bullets that would be injected, without updating store.

        Useful for validating retrieval independently of writes.
        """
        try:
            entities, _, _, _ = self._extract(text, lang)
            entities = self._refine_entities_from_text(text, entities)
        except Exception:
            entities = []
        bullets = self._retrieve_context(text, entities, turn_id=-1, intent=None)
        return {"entities": entities, "bullets": bullets}

    # ---------- Refinement helpers (quality without large perf cost) ----------
    def _is_question(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        if "?" in t:
            return True
        wh = ("who", "what", "when", "where", "why", "how")
        return any(t.startswith(w + " ") for w in wh)

    def _refine_entities_from_text(self, text: str, entities: List[str]) -> List[str]:
        # Canonicalize and drop noisy scaffolding like 'my name'
        out = []
        for e in entities:
            ce = _canon_entity_text(e)
            if ce and ce not in {"my name", "my dog's name", "name"}:
                out.append(ce)
        
        # Ensure 'you' if text is self-referential
        t = (text or "").lower()
        if any(p in t for p in [" i ", " my ", " me "]) and "you" not in out:
            out.append("you")
        
        # Graph-aware entity expansion: find related entities in knowledge graph
        expanded = list(out)  # Start with original entities
        for entity in out:
            related = self._find_related_entities(entity)
            expanded.extend(related)
        
        # Unique, preserve order
        seen = set()
        uniq = []
        for e in expanded:
            if e not in seen:
                uniq.append(e)
                seen.add(e)
        return uniq
    
    def _find_related_entities(self, entity: str) -> List[str]:
        """Find entities related to the given entity through stored relationships"""
        related = []
        try:
            # Look through stored edges to find connections
            for s, r, d in self.entity_index.get(entity, []):
                # If entity is the subject, add the destination as related
                if s == entity:
                    related.append(d)
                # If entity is mentioned in destination, add the subject as related  
                elif entity.lower() in d.lower():
                    related.append(s)
            
            # Also check reverse - where entity appears as destination
            for edge_set in self.entity_index.values():
                for s, r, d in edge_set:
                    if d == entity or entity.lower() in d.lower():
                        related.append(s)
                        # If this connects to 'you', add 'you' as highly relevant
                        if s == "you":
                            related.insert(0, "you")  # Prioritize 'you'
        except Exception:
            pass
        
        # Clean and return unique related entities
        clean_related = []
        for e in related:
            ce = _canon_entity_text(e)
            if ce and ce != entity and ce not in clean_related:
                clean_related.append(ce)
        
        return clean_related[:3]  # Limit to top 3 related entities to avoid explosion

    def _refine_triples(self, text: str, triples: List[Tuple[str, str, str]], doc, intent=None, lang: str = "en") -> List[Tuple[str, str, str]]:
        t = (text or "").lower()
        orig = text or ""
        refined: List[Tuple[str, str, str]] = []

        # Name patterns from raw text (more reliable than dep combos alone)
        # 1) My name is X
        m = None
        try:
            m = __import__("re").search(r"\bmy name is\s+([^,.!?]+)", t)
        except Exception:
            m = None
        if m:
            name = _canon_entity_text(m.group(1))
            refined.append(("you", "name", name))

        # 1b) I'm X / I am X → treat X as a name (fallback without NER)
        m_iam = None
        try:
            m_iam = __import__("re").search(r"\bI(?:'m| am)\s+([A-Z][A-Za-z\-']{1,30})\b", orig)
        except Exception:
            m_iam = None
        if m_iam:
            nm = _canon_entity_text(m_iam.group(1))
            if nm and nm not in {"i", "am"}:
                refined.append(("you", "name", nm))

        # 2) My dog's name is X
        md = None
        try:
            md = __import__("re").search(r"\bmy dog'?s name is\s+([^,.!?]+)", t)
        except Exception:
            md = None
        if md:
            dname = _canon_entity_text(md.group(1))
            refined.append(("dog", "name", dname))
            refined.append(("you", "has", "dog"))

        # 3) My son is named X
        ms = None
        try:
            ms = __import__("re").search(r"\bmy son is named\s+([^,.!?]+)", t)
        except Exception:
            ms = None
        if ms:
            sname = _canon_entity_text(ms.group(1))
            refined.append(("son", "name", sname))
            refined.append(("you", "has", "son"))

        # 4) Favorite color is X → favorite_color
        fc = None
        try:
            fc = __import__("re").search(r"\bfavorite color is\s+([^,.!?]+)", t)
        except Exception:
            fc = None
        if fc:
            fav = _canon_entity_text(fc.group(1))
            refined.append(("you", "favorite_color", fav))

        # 5) Favorite number is X → favorite_number
        fn = None
        try:
            fn = __import__("re").search(r"\bfavorite number is\s+([^,.!?]+)", t)
        except Exception:
            fn = None
        if fn:
            favn = _canon_entity_text(fn.group(1))
            refined.append(("you", "favorite_number", favn))

        for s, r, d in triples:
            cs = _canon_entity_text(s)
            cd = _canon_entity_text(d)

            # Pronouns → you
            if cs in _PRON_YOU:
                cs = "you"
            if cd in _PRON_YOU:
                cd = "you"

            rr = r
            # Canonicalize minor relation variants
            if rr in {"teaches_at", "teached_at"}:
                rr = "teach_at"
            # Strip relative scaffolding prefixes from entities for readability
            for pref in ("whose ", "which ", "who ", "that "):
                if cs.startswith(pref):
                    cs = cs[len(pref):].strip()
                if cd.startswith(pref):
                    cd = cd[len(pref):].strip()

            # Normalize common patterns for better retrieval and matching
            t_low = t  # lower-cased full text
            # Name via copula: "I'm Sarah" -> (you, name, sarah) when 'Sarah' is PERSON
            if cs == 'you' and rr == 'is' and cd:
                try:
                    if doc is not None:
                        for ent in getattr(doc, 'ents', []) or []:
                            if getattr(ent, 'label_', '') == 'PERSON' and _canon_entity_text(ent.text) == cd:
                                rr = 'name'
                                break
                except Exception:
                    pass
            # Drive/own → treat as possession for user
            if rr in {"drive", "drives", "drove"} and cs == "you":
                rr = "has"
            # Teaching normalization: prefer (X, teach_at, ORG)
            if rr in {"teach", "teaches", "teaching"}:
                chosen_org: Optional[str] = None
                try:
                    # Prefer an ORG entity from the doc if present
                    if doc is not None:
                        for ent in getattr(doc, 'ents', []) or []:
                            if getattr(ent, 'label_', '') in {"ORG"}:
                                chosen_org = _canon_entity_text(ent.text)
                                break
                except Exception:
                    chosen_org = None
                # If surface contains "teach at" and we have an ORG, promote to teach_at
                if (" teach at " in t_low or " at " in t_low) and chosen_org:
                    rr = "teach_at"
                    cd = chosen_org
                elif " teach at " in t_low:
                    # Fallback to teach_at if pattern explicit, but avoid subjects like 'philosophy'
                    bad_subjects = {"philosophy", "math", "mathematics", "physics", "chemistry", "history", "biology", "english"}
                    if cd not in bad_subjects:
                        rr = "teach_at"
            # Age normalization: "X is N years old"
            try:
                import re as _re
                m_age = _re.search(r"\b(\d{1,2})\s+years?\s+old\b", t_low)

                def _is_age_subject_candidate(subj: str, _doc) -> bool:
                    subj = (subj or '').strip().lower()
                    if not subj:
                        return False
                    # Obvious candidates: you/person/pet/common kinship nouns
                    if subj in {"you","he","she","they","son","daughter","kid","child","boy","girl","luna","whiskers","tom","sarah","dog","cat","puppy","kitten","pet"}:
                        return True
                    # Reject clearly generic/abstract subjects that caused false ages in batch text
                    generic_tokens = {"analysis","approach","tuning","years","is","assistant","number","such","those","what","combination","coats","eyes","results","models","foundation"}
                    if any(tok in subj for tok in generic_tokens):
                        return False
                    try:
                        # Prefer PERSON entities for age
                        for ent in getattr(_doc, 'ents', []) or []:
                            if _canon_entity_text(ent.text) == subj and getattr(ent, 'label_', '') == 'PERSON':
                                return True
                    except Exception:
                        pass
                    # Fallback: accept short nouns (1-2 words) that appear in text
                    return len(subj.split()) <= 2 and (f" {subj} " in t_low)

                if m_age and rr in {"is", "age"} and _is_age_subject_candidate(cs, doc):
                    rr = "age"
                    cd = f"{m_age.group(1)} years old"
            except Exception:
                pass
            # Fix generic preposition rels if the verb is inferable from surface text
            if r in {"_in", "_at", "_from", "_to"}:
                if " live" in t or t.startswith("live") or " living" in t:
                    rr = "lives_in" if r == "_in" else "lives_at"
                elif " born" in t or t.startswith("born"):
                    rr = "born_in"
                elif " work" in t or t.startswith("work"):
                    rr = "works_at" if r == "_at" else "works_in"
                elif (" move" in t or " moved" in t) and r == "_from":
                    rr = "moved_from"
                elif (" participate" in t or " participated" in t) and r == "_in":
                    rr = "participated_in"
                elif (" go" in t or " went" in t) and r == "_to":
                    rr = "went_to"

                # Multilingual normalization (simple keyword checks)
                l = (lang or "en").lower()
                try:
                    if l == 'es':
                        if (" vive" in t or t.startswith("vive")) and r == "_in":
                            rr = "lives_in"
                        elif (" trabaja" in t or t.startswith("trabaja")) and r in {"_at", "_in"}:
                            rr = "works_at"
                        elif (" nació" in t or t.startswith("nació")) and r == "_in":
                            rr = "born_in"
                        elif (" mudó" in t or " se mudó" in t) and r == "_from":
                            rr = "moved_from"
                        elif (" particip" in t) and r == "_in":
                            rr = "participated_in"
                        elif (" fue" in t or " ir " in t or t.startswith("fue") or t.startswith("ir ")) and r == "_to":
                            rr = "went_to"
                    elif l == 'fr':
                        if (" habite" in t or t.startswith("habite")) and r in {"_in", "_at"}:
                            rr = "lives_in"
                        elif (" travaille" in t or t.startswith("travaille")) and r in {"_at", "_in"}:
                            rr = "works_at"
                        elif (" né" in t or " née" in t) and r == "_in":
                            rr = "born_in"
                        elif (" déménagé" in t) and r == "_from":
                            rr = "moved_from"
                        elif (" participé" in t) and r == "_in":
                            rr = "participated_in"
                        elif (" allé" in t) and r == "_to":
                            rr = "went_to"
                    elif l == 'de':
                        if (" wohnt" in t or t.startswith("wohnt")) and r == "_in":
                            rr = "lives_in"
                        elif (" arbeitet" in t or t.startswith("arbeitet")) and r in {"_at", "_in"}:
                            rr = "works_at"
                        elif (" geboren" in t) and r == "_in":
                            rr = "born_in"
                        elif (" zog" in t or " umgezogen" in t) and r == "_from":
                            rr = "moved_from"
                        elif (" teilgenommen" in t) and r == "_in":
                            rr = "participated_in"
                        elif (" ging" in t or " gegangen" in t) and r == "_to":
                            rr = "went_to"
                    elif l == 'it':
                        if (" vive" in t or t.startswith("vive")) and r == "_in":
                            rr = "lives_in"
                        elif (" lavora" in t or t.startswith("lavora")) and r in {"_at", "_in"}:
                            rr = "works_at"
                        elif (" nato" in t or " nata" in t) and r == "_in":
                            rr = "born_in"
                        elif (" trasferit" in t) and r == "_from":
                            rr = "moved_from"
                        elif (" partecipat" in t) and r == "_in":
                            rr = "participated_in"
                        elif (" andat" in t) and r == "_to":
                            rr = "went_to"
                except Exception:
                    pass

            # Normalize belongs_to ownership
            if (r in {"_to", "belong_to", "belongs_to"}) and ("belong" in t):
                rr = "belongs_to"
                if cd in {"me", "you"}:
                    # Flip to (you, owns, s)
                    refined.append(("you", "owns", cs))
                    continue

            # Drop scaffolding nodes around name if canonical name fact present
            scaff_names = {"my name", "my dog's name", "name", "dog's name"}
            if (cs in scaff_names or cd in scaff_names) and rr in {"is", "has"}:
                continue

            # Drop "you has name" scaffolding
            if cs == "you" and rr == "has" and cd == "name":
                continue

            # Keep
            refined.append((cs, rr, cd))

        # Derive symmetric friend_of from patterns
        friends_pairs: List[Tuple[str, str]] = []
        names = set()
        for (a, r, b) in refined:
            if r == "and":
                friends_pairs.append((a, b))
            if r == "is" and b == "friends":
                names.add(a)
        for (a, b) in friends_pairs:
            if a in names:
                refined.append((a, "friend_of", b))
                refined.append((b, "friend_of", a))

        # Collect temporal signals (language-agnostic via UD roles)
        years: List[str] = []
        durations: List[str] = []
        try:
            for tok in (doc or []):
                # Year candidates anywhere
                if tok.like_num:
                    try:
                        val = int(tok.text)
                        if 1900 <= val <= 2100 and len(tok.text) == 4:
                            years.append(tok.text)
                    except Exception:
                        pass

                # Duration attached to verbs as oblique or nominal modifier
                if tok.dep_ in {"obl", "nmod"} and tok.head and tok.head.pos_ in {"VERB", "AUX"}:
                    # Look for numeric child modifying this token
                    num_child = None
                    for ch in tok.children:
                        if ch.dep_ == "nummod" and ch.like_num:
                            num_child = ch
                            break
                    # Or the token itself is numeric (e.g., year)
                    if num_child is not None:
                        # Use the subtree span text as duration phrase
                        left = min([t.left_edge.i for t in [tok, num_child]])
                        right = max([t.right_edge.i for t in [tok, num_child]])
                        span = tok.doc[left:right+1]
                        durations.append(_canon_entity_text(span.text))
        except Exception:
            pass

        # Derive symmetric friend_of from patterns
        friends_pairs: List[Tuple[str, str]] = []
        names = set()
        for (a, r, b) in refined:
            if r == "and":
                friends_pairs.append((a, b))
            if r == "is" and b == "friends":
                names.add(a)
        for (a, b) in friends_pairs:
            if a in names:
                refined.append((a, "friend_of", b))
                refined.append((b, "friend_of", a))

        # Attach temporal info to the most relevant event triple
        def is_event_rel(rel: str) -> bool:
            return rel not in {"has", "name", "favorite_color", "friend_of", "quality", "quantity", "is", "owns"}

        anchor: Optional[Tuple[str, str, str]] = None
        for tr in refined:
            if is_event_rel(tr[1]):
                anchor = tr
                break
        if anchor is None and refined:
            anchor = refined[0]

        if anchor is not None:
            s_anchor, r_anchor, d_anchor = anchor
            for y in years:
                refined.append((s_anchor, "time", y))
            for dur in durations:
                refined.append((s_anchor, "duration", dur))
            # Record temporal props to annotate the anchor edge when stored
            try:
                props = {}
                if years:
                    props['time'] = list(dict.fromkeys(years))
                if durations:
                    props['duration'] = list(dict.fromkeys(durations))
                if props:
                    self._pending_edge_props[(s_anchor, r_anchor, d_anchor)] = props
            except Exception:
                pass

        # De-duplicate while preserving order
        seen = set()
        uniq: List[Tuple[str, str, str]] = []
        for tr in refined:
            if tr not in seen and all(tr):
                uniq.append(tr)
                seen.add(tr)
        return uniq

    # --- Coref-lite: resolve third-person pronouns to recent mentions ---
    def _build_mention_stack(self, doc) -> List[str]:
        stack: List[str] = []
        try:
            # Prefer named entities in order of appearance; avoid per-token PROPN noise.
            if doc is not None:
                for ent in getattr(doc, 'ents', []) or []:
                    txt = _canon_entity_text(ent.text)
                    if txt and txt not in stack:
                        stack.append(txt)
        except Exception:
            pass
        # Add recent entities from recency_buffer
        try:
            for item in list(self.recency_buffer)[-5:]:
                for ent in (item.s, item.d):
                    if ent and ent not in stack and ent != 'you':
                        stack.append(ent)
        except Exception:
            pass
        return stack

    def _resolve_pronoun(self, token_text: str, stack: List[str]) -> Optional[str]:
        p = (token_text or '').strip().lower()
        if not p:
            return None
        third = {
            'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs',
            'it', 'this', 'that', 'who', 'whom', 'which', 'whose'
        }
        if p in third:
            # Return most recent stack entry not 'you'
            for cand in reversed(stack):
                if cand != 'you':
                    return cand
        return None

    def _apply_coref_lite(self, triples: List[Tuple[str, str, str]], doc) -> List[Tuple[str, str, str]]:
        stack = self._build_mention_stack(doc)
        # Build PERSON set for preference when resolving he/she/who/whose
        person_set: Set[str] = set()
        try:
            for ent in getattr(doc, 'ents', []) or []:
                if getattr(ent, 'label_', '') == 'PERSON':
                    person_set.add(_canon_entity_text(ent.text))
        except Exception:
            pass

        out: List[Tuple[str, str, str]] = []
        last_entity: Optional[str] = None
        last_person: Optional[str] = None

        def prefer_recent_person() -> Optional[str]:
            if last_person:
                return last_person
            for cand in reversed(stack):
                if cand in person_set and cand != 'you':
                    return cand
            return None

        def nearest_person_before(token_text: str) -> Optional[str]:
            """Find nearest PERSON mention occurring before the first token match of token_text."""
            try:
                target = None
                tnorm = _canon_entity_text(token_text)
                for tok in doc:  # type: ignore[attr-defined]
                    if _canon_entity_text(tok.text) == tnorm:
                        target = tok.i
                        break
                if target is None:
                    return None
                cand = None
                for ent in getattr(doc, 'ents', []) or []:
                    if getattr(ent, 'label_', '') == 'PERSON' and ent.end <= target:
                        cand = _canon_entity_text(ent.text)
                return cand
            except Exception:
                return None

        # A/B detection mode for pronouns: 'string' (default), 'ud', 'hybrid'
        pron_mode = os.getenv('HOTMEM_PRONOUN_DETECT', 'string').lower()

        def is_pronoun_like_string(tok: str) -> bool:
            p = (tok or '').strip().lower()
            return p in {
                'he','him','his','she','her','hers','they','them','their','theirs',
                'it','this','that','who','whom','which','whose'
            }

        def pronoun_kind_string(tok: str) -> str:
            p = (tok or '').strip().lower()
            if p in {'he','him','his','she','her','hers','who','whom','whose','they','them','their','theirs'}:
                return 'person'
            return 'thing'

        def _find_token_index_for_text(token_text: str) -> Optional[int]:
            try:
                tnorm = _canon_entity_text(token_text)
                for tok in doc:  # type: ignore[attr-defined]
                    if _canon_entity_text(tok.text) == tnorm:
                        return int(tok.i)
            except Exception:
                return None
            return None

        def is_pronoun_like_idx_ud(idx: Optional[int]) -> bool:
            try:
                if idx is None:
                    return False
                tok = doc[idx]  # type: ignore[index]
                if tok.pos_ not in {'PRON', 'DET'}:
                    return False
                m = tok.morph.to_dict()
                pt = m.get('PronType')
                return pt in {'Rel', 'Prs', 'Dem', 'Int'}
            except Exception:
                return False

        def pronoun_kind_idx_ud(idx: Optional[int]) -> str:
            """Return 'person' or 'thing' heuristic based on morph.
            Personal pronouns → person; Relative/demonstrative → prefer person if left PERSON anchor exists.
            """
            try:
                if idx is None:
                    return 'thing'
                tok = doc[idx]  # type: ignore[index]
                m = tok.morph.to_dict()
                pt = m.get('PronType')
                if pt == 'Prs':
                    return 'person'
                if pt in {'Rel', 'Dem', 'Int'}:
                    # Guess person if there is a left PERSON anchor in the sentence
                    anchor = _sentence_person_anchor(idx)
                    return 'person' if anchor else 'thing'
            except Exception:
                pass
            return 'thing'

        def is_pronoun_like(tok: str) -> bool:
            if pron_mode == 'string':
                return is_pronoun_like_string(tok)
            if pron_mode == 'ud':
                return is_pronoun_like_idx_ud(_find_token_index_for_text(tok))
            # hybrid
            return is_pronoun_like_string(tok) or is_pronoun_like_idx_ud(_find_token_index_for_text(tok))

        def pronoun_kind(tok: str) -> str:
            if pron_mode == 'string':
                return pronoun_kind_string(tok)
            if pron_mode == 'ud':
                return pronoun_kind_idx_ud(_find_token_index_for_text(tok))
            # hybrid
            if is_pronoun_like_string(tok):
                return pronoun_kind_string(tok)
            return pronoun_kind_idx_ud(_find_token_index_for_text(tok))

        # --- Deterministic pronoun anchoring helpers ---
        from collections import defaultdict, deque as _deque
        pronoun_positions: Dict[str, _deque] = defaultdict(_deque)
        try:
            for i, tok in enumerate(doc):  # type: ignore[attr-defined]
                t = tok.text.lower()
                if t in {'he','him','his','she','her','hers','they','them','their','theirs','it','this','that','who','whom','which','whose'}:
                    pronoun_positions[t].append(i)
        except Exception:
            pass

        def _next_idx_for(token_text: str) -> Optional[int]:
            try:
                dq = pronoun_positions.get((token_text or '').lower())
                if dq and len(dq) > 0:
                    return int(dq.popleft())
            except Exception:
                return None
            return None

        def _sentence_person_anchor(idx: int) -> Optional[str]:
            try:
                if idx is None:
                    return None
                sent = doc[idx].sent  # type: ignore[attr-defined]
                # Prefer PERSON entities ending before idx (closest to the left)
                best = None
                best_end = -1
                for ent in getattr(sent, 'ents', []) or []:
                    if getattr(ent, 'label_', '') == 'PERSON' and ent.end <= idx and ent.end > best_end:
                        best = ent
                        best_end = ent.end
                if best is not None:
                    return _canon_entity_text(best.text)
                # Fallback: any PERSON in the sentence (leftmost)
                for ent in getattr(sent, 'ents', []) or []:
                    if getattr(ent, 'label_', '') == 'PERSON':
                        return _canon_entity_text(ent.text)
            except Exception:
                return None
            return None

        def _sentence_noun_chunk_anchor(idx: int) -> Optional[str]:
            try:
                if idx is None:
                    return None
                sent = doc[idx].sent  # type: ignore[attr-defined]
                best = None
                best_end = -1
                for chunk in getattr(sent, 'noun_chunks', []) or []:
                    if chunk.end <= idx and chunk.end > best_end:
                        best = chunk
                        best_end = chunk.end
                if best is not None:
                    return _canon_entity_text(best.text)
            except Exception:
                return None
            return None

        def _relative_anchor(idx: Optional[int]) -> Optional[str]:
            """Anchor a relative pronoun to its head noun/proper if available.

            Strategy: climb up to a few ancestors to find a NOUN/PROPN in-sentence;
            if found, return its noun chunk span text; else fallback to left noun chunk.
            """
            try:
                if idx is None:
                    return None
                tok = doc[idx]  # type: ignore[index]
                sent = tok.sent
                cur = tok.head
                hops = 0
                while cur is not None and hops < 4:
                    if cur.i < sent.start or cur.i >= sent.end:
                        break
                    if cur.pos_ in {'NOUN', 'PROPN'}:
                        # Find noun chunk whose root is cur
                        for chunk in getattr(sent, 'noun_chunks', []) or []:
                            if chunk.root.i == cur.i:
                                return _canon_entity_text(chunk.text)
                        return _canon_entity_text(cur.text)
                    nxt = cur.head
                    if nxt is cur:
                        break
                    cur = nxt
                    hops += 1
                # Fallback to nearest noun chunk to the left
                return _sentence_noun_chunk_anchor(idx)
            except Exception:
                return None

        def _is_person_like(name: str) -> bool:
            if not name or name == 'you':
                return False
            if name in person_set:
                return True
            try:
                ln = _canon_entity_text(name)
                # Only treat as person-like if NER labels it PERSON;
                # do NOT fallback to generic PROPN to avoid ORG/LOC pollution (e.g., Microsoft, Seattle)
                for ent in getattr(doc, 'ents', []) or []:  # type: ignore[attr-defined]
                    if _canon_entity_text(ent.text) == ln and getattr(ent, 'label_', '') == 'PERSON':
                        return True
                return False
            except Exception:
                return False

        for s, r, d in triples:
            rs = s
            rd = d
            # Resolve subjects (only when pronoun-like)
            if s not in {'you'} and is_pronoun_like(s):
                kind = pronoun_kind(s)
                pron_idx = _find_token_index_for_text(s)
                cand = None
                # UD-relative: anchor to head noun if available
                if pron_mode in {'ud', 'hybrid'} and pron_idx is not None:
                    try:
                        tok = doc[pron_idx]
                        if 'Rel' in tok.morph.get('PronType'):
                            cand = _relative_anchor(pron_idx)
                    except Exception:
                        pass
                if cand is None:
                    if kind == 'person':
                        cand = _sentence_person_anchor(pron_idx) or nearest_person_before(s) or prefer_recent_person() or (last_entity if last_entity and last_entity != 'you' else None)
                    else:
                        cand = _sentence_noun_chunk_anchor(pron_idx) or (last_entity if last_entity and last_entity != 'you' else None)
                if cand:
                    rs = cand
            # Resolve objects (only when pronoun-like)
            if d not in {'you'} and is_pronoun_like(d):
                kind = pronoun_kind(d)
                pron_idx = _find_token_index_for_text(d)
                cand = None
                # UD-relative: anchor to head noun if available
                if pron_mode in {'ud', 'hybrid'} and pron_idx is not None:
                    try:
                        tok = doc[pron_idx]
                        if 'Rel' in tok.morph.get('PronType'):
                            cand = _relative_anchor(pron_idx)
                    except Exception:
                        pass
                if cand is None:
                    if kind == 'person':
                        cand = _sentence_person_anchor(pron_idx) or nearest_person_before(d) or prefer_recent_person() or (last_entity if last_entity and last_entity != 'you' else None)
                    else:
                        cand = _sentence_noun_chunk_anchor(pron_idx) or (last_entity if last_entity and last_entity != 'you' else None)
                if cand:
                    rd = cand

            out.append((rs, r, rd))

            # Update trackers after resolution
            if rs and rs != 'you':
                last_entity = rs
                if _is_person_like(rs):
                    last_person = rs
            # Do not update last_person from destination objects; they often are ORG/LOC.

        return out

    def _extract_entities_light(self, text: str) -> List[str]:
        """GLiNER-based entity extraction for superior compound entity detection"""
        # Lazy load GLiNER model
        if not hasattr(self, '_gliner_model'):
            try:
                from gliner import GLiNER
                self._gliner_model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
                logger.info("[HotMem] GLiNER model loaded for entity extraction")
            except Exception as e:
                logger.warning(f"[HotMem] GLiNER load failed, falling back: {e}")
                return self._extract_entities_light_fallback(text)
        
        # Define entity types relevant for memory queries
        labels = ["person", "product", "application", "software", "organization", 
                  "place", "event", "brand", "object", "animal", "food"]
        
        try:
            entities = self._gliner_model.predict_entities(text, labels, threshold=0.4)
            extracted = []
            
            for entity in entities:
                clean_entity = _canon_entity_text(entity["text"])
                if clean_entity and clean_entity not in extracted:
                    extracted.append(clean_entity)
            
            # Also check entity index for known entities not detected by GLiNER
            words = text.lower().split()
            for word in words:
                clean_word = _canon_entity_text(word)
                if clean_word in self.entity_index and clean_word not in extracted:
                    extracted.append(clean_word)
            
            # Add 'you' for self-referential queries
            if any(p in text.lower() for p in [" i ", " my ", " me ", "i'm", "i've"]):
                if "you" not in extracted:
                    extracted.append("you")
            
            return extracted
            
        except Exception as e:
            logger.warning(f"[HotMem] GLiNER extraction failed: {e}")
            return self._extract_entities_light_fallback(text)

    def _extract_entities_light_fallback(self, text: str) -> List[str]:
        """Original word-based extraction as fallback"""
        # Simple pattern-based extraction for performance
        entities = []
        words = text.split()
        
        # Look for capitalized words (likely names/places)
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                clean_word = _canon_entity_text(word)
                if clean_word and clean_word not in entities:
                    entities.append(clean_word)
        
        # Also look for lowercase words that exist in our entity index
        # This catches common nouns like "dog", "cat", "car" that are stored entities
        for word in words:
            clean_word = _canon_entity_text(word)
            if clean_word and len(clean_word) > 2 and clean_word not in entities:
                # Check if this entity exists in our knowledge base
                if clean_word in self.entity_index:
                    entities.append(clean_word)
        
        # Add 'you' if self-referential
        if any(p in text.lower() for p in [" i ", " my ", " me "]):
            entities.append("you")
            
        return entities
        
    def _handle_fact_correction(self, s: str, r: str, d: str, confidence: float, now_ts: int):
        """Handle fact corrections by demoting conflicting facts"""
        # For functional relations like 'name' or 'age', demote old values
        functional_relations = {"name", "age", "favorite_color", "lives_in", "works_at"}
        
        if r in functional_relations:
            # Find existing facts with same subject and relation
            existing_edges = self.entity_index.get(s, set())
            for existing_s, existing_r, existing_d in existing_edges.copy():
                if existing_s == s and existing_r == r and existing_d != d:
                    # Demote the old fact
                    self.store.negate_edge(s, r, existing_d, confidence, now_ts)
                    logger.debug(f"Corrected fact: ({s}, {r}) {existing_d} -> {d}")
                    # Remove from hot index
                    self.entity_index[s].discard((existing_s, existing_r, existing_d))
                    self.entity_index[existing_d].discard((existing_s, existing_r, existing_d))
        
        # Store the new corrected fact
        self.store.observe_edge(s, r, d, confidence, now_ts)
        # If correcting the user's name, update alias mapping (X -> you)
        try:
            if (s == self.user_eid) and (r == 'name') and d:
                self.store.enqueue_alias(str(d), self.user_eid)
                self.store.flush_if_needed()
        except Exception:
            pass

    
