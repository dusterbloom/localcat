"""
HotMemoryFacade: Backward Compatibility Layer
===========================================

Provides the same interface as the original HotMemory class
while using the new extracted services internally.

This ensures no breaking changes during the refactoring.
"""

import os
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass

from loguru import logger

from components.memory.memory_store import MemoryStore
from components.memory.memory_intent import get_intent_classifier, get_quality_filter, IntentType
from components.memory.memory_quality import MemoryQuality
from components.memory.config import create_config
from components.extraction.memory_extractor import MemoryExtractor, ExtractionResult
from components.retrieval.memory_retriever import MemoryRetriever, RetrievalResult
from components.coreference.coreference_resolver import CoreferenceResolver, CoreferenceResult
from components.extraction.assisted_extractor import AssistedExtractor, AssistedExtractionResult
from components.session.session_store import get_session_store, SessionMessage

# Import components that still need to be extracted
try:
    from components.processing.semantic_roles import SRLExtractor
except Exception:
    SRLExtractor = None
try:
    from services.onnx_nlp import OnnxTokenNER, OnnxSRLTagger
except Exception:
    OnnxTokenNER = None
    OnnxSRLTagger = None
try:
    from components.extraction.hotmem_extractor import HotMemExtractor
except Exception:
    HotMemExtractor = None
try:
    from components.extraction.enhanced_hotmem_extractor import EnhancedHotMemExtractor
except Exception:
    EnhancedHotMemExtractor = None
try:
    from components.extraction.hybrid_spacy_llm_extractor import HybridRelationExtractor
except Exception:
    HybridRelationExtractor = None
try:
    from services.fastcoref import FCoref
except Exception:
    FCoref = None


@dataclass
class RecencyItem:
    """Data class for tracking recent interactions"""
    s: str
    r: str
    d: str
    text: str
    ts: int
    turn_id: int


class HotMemoryFacade:
    """
    Facade that maintains backward compatibility with original HotMemory interface
    while using new extracted services internally.
    """
    
    def __init__(self, store: MemoryStore, max_recency: int = 50):
        """Initialize with same interface as original"""
        self.store = store
        self.user_eid = "you"
        
        # Load configuration
        self.config = create_config()
        self.config.max_recency = max_recency
        
        # Initialize extracted services
        self.extractor = MemoryExtractor(self.config.get_extractor_config())
        self.retriever = MemoryRetriever(store, defaultdict(set), self.config.get_retriever_config())
        self.coreference_resolver = CoreferenceResolver(self.config.get_coreference_config())
        self.assisted_extractor = AssistedExtractor(self.config.get_assisted_config())
        
        # Initialize session store for comprehensive session management
        self.session_store = get_session_store()
        
        # Hot indices (RAM) - keeping for backward compatibility
        self.entity_index = defaultdict(set)  # entity -> set of (s,r,d) triples
        self.recency_buffer = deque(maxlen=max_recency)  # Recent interactions
        self.entity_cache = {}  # Canonical entity mapping
        self.edge_meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}  # (s,r,d) -> {ts, weight}
        
        # Update retriever with current entity_index
        self.retriever.entity_index = self.entity_index
        self.retriever.edge_meta = self.edge_meta
        
        # Enhanced bullet formatter (lazy import)
        self.bullet_formatter = None
        
        # Performance tracking
        self.metrics = defaultdict(list)
        self.max_metric_size = 1000
        
        # Legacy components that haven't been extracted yet
        self._initialize_legacy_components()
        
        # Log configuration
        self.config.log_configuration()
        
        # Rich quality filter for triple validation
        self.quality = MemoryQuality({
            'min_confidence': max(0.3, self.config.confidence_threshold),
        })
    
    def _initialize_legacy_components(self):
        """Initialize components that haven't been extracted yet"""
        # Improved UD extractor for quality filtering
        try:
            from components.processing.hotpath_processor import ImprovedUDExtractor
            self.ud_processor = ImprovedUDExtractor() if ImprovedUDExtractor else None
        except Exception:
            self.ud_processor = None
        
        # LEANN semantic search
        self.use_leann = self.config.features.use_leann
        self.leann_index_path = self.config.leann_index_path
        self.leann_complexity = self.config.leann_complexity
        self._leann_searcher = None
        self._leann_loaded_mtime = 0.0
        
        # Assisted extraction
        self.assisted_enabled = self.config.features.assisted_enabled
        self.assisted_model = self.config.assisted_model
        self._assisted_calls = 0
        self._assisted_success = 0
        
        # Retrieval fusion
        self.retrieval_fusion = self.config.features.retrieval_fusion
        self.use_leann_summaries = self.config.use_leann_summaries
        
        # SRL integration
        self.use_srl = self.config.features.use_srl
        self._srl: Optional[Any] = None
        
        # ONNX integration
        self.use_onnx_ner = self.config.features.use_onnx_ner
        self.use_onnx_srl = self.config.features.use_onnx_srl
        self._onnx_ner = None
        self._onnx_srl = None
        
        # ReLiK integration
        self.use_relik = self.config.features.use_relik
        self._relik = None
        
        # Coreference resolution
        self.use_coref = self.config.features.use_coref
        self.coref_max_entities = self.config.coref_max_entities
        self._coref_model = None
        self._coref_cache = {}
        
        # DSPy integration
        self.use_dspy = self.config.features.use_dspy
        self._dspy_extractor = None
        
        # Classifier caching
        self._classifier_cache = {}
        self._cache_max_size = self.config.cache_size
        
        # Pending edge properties
        self._pending_edge_props = {}
    
    def process_turn(self, text: str, session_id: str, turn_id: int) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Process a conversation turn - same interface as original
        """
        start = time.perf_counter()
        
        # Store user message verbatim
        self.session_store.add_message(session_id, "user", text, turn_id)
        
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
            extraction_result = self.extractor.extract(text, lang)
            _ = extraction_result.triples  # not persisted when pure question
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics['total_ms'].append(elapsed_ms)
            # Return no stored triples in this branch
            return bullets, []
        
        # Stage 1: Extract entities and relations using new extractor
        extract_start = time.perf_counter()
        extraction_result = self.extractor.extract(text, lang)
        entities = extraction_result.entities
        triples = extraction_result.triples
        neg_count = extraction_result.negation_count
        doc = extraction_result.doc
        self.metrics['extraction_ms'].append((time.perf_counter() - extract_start) * 1000)
        
        # Stage 1b: Optional LLM-assisted micro-refiner
        if self.assisted_extractor.should_assist(text, triples, doc):
            assist_start = time.perf_counter()
            assisted_result = self.assisted_extractor.extract_assisted(text, entities, triples, session_id=session_id)
            ms = (time.perf_counter() - assist_start) * 1000
            self.metrics['assisted_ms'].append(ms)
            
            if assisted_result.triples:
                # Merge and de-dup
                seen = set(map(tuple, triples))
                for tr in assisted_result.triples:
                    if tuple(tr) not in seen:
                        triples.append(tuple(tr))
                        seen.add(tuple(tr))
                logger.info(f"[HotMem Assisted] triggered (ms={ms:.0f}, triples={len(assisted_result.triples)})")
            else:
                logger.info(f"[HotMem Assisted] triggered (ms={ms:.0f}, triples=0)")
        
        # Stage 2: Refine triples with intent-aware processing
        refine_start = time.perf_counter()
        triples = self._refine_triples(text, triples, doc, intent, lang)
        
        # Apply coreference if enabled
        if self.config.features.use_coref:
            coreference_result = self.coreference_resolver.resolve_coreferences(triples, doc, text)
            triples = coreference_result.resolved_triples
            logger.debug(f"[HotMem] Coreference resolved: {len(coreference_result.resolved_triples)} triples")
        
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
        prov_tag = 'ud_only'
        if self.use_srl:
            prov_tag = 'srl_ud'
        if getattr(self, 'use_onnx_srl', False):
            prov_tag = 'onnx_srl_ud'
        
        # Apply rich quality filtering first, then intent gating
        conf_thresh = max(0.3, self.config.confidence_threshold)
        try:
            filtered = self.quality.filter_triples(triples, context={'conversation_text': text})
        except Exception:
            # Fallback to unfiltered triples if quality module fails
            filtered = [(s, r, d, 0.5) for (s, r, d) in triples]
        
        for s, r, d, q_conf in filtered:
            # Intent-based allowlist
            should_store, legacy_conf = quality_filter.should_store_fact(s, r, d, intent)
            confidence = float(max(q_conf, legacy_conf))
            if should_store and confidence >= conf_thresh:
                stored_triples.append((s, r, d))
            
            # Always update hot indices for retrieval, regardless of storage
            self.entity_index[s].add((s, r, d))
            self.entity_index[d].add((s, r, d))
            self.edge_meta[(s, r, d)] = {'ts': now_ts, 'weight': confidence}
        
        # Update recency with stored triples only
        for s, r, d in stored_triples:
            self.recency_buffer.append(RecencyItem(s, r, d, text, now_ts, turn_id))
        
        # Stage 4: Context retrieval using new MemoryRetriever
        retrieve_start = time.perf_counter()
        retrieval_result = self.retriever.retrieve_context(text, entities, turn_id, intent=intent)
        bullets = retrieval_result.bullets
        self.metrics['retrieval_ms'].append((time.perf_counter() - retrieve_start) * 1000)
        
        # Store final triples and link to session
        if stored_triples:
            self.store.update_session_triples(session_id, stored_triples)
            
            # Link extracted knowledge to session
            for s, r, d in stored_triples:
                edge_id = self.store.edge_id(s, r, d)
                self.session_store.link_knowledge_to_session(session_id, edge_id, "extracted", 0.8)
        
        # Track performance
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics['total_ms'].append(elapsed_ms)
        
        # Memory summary line
        logger.info(f"[HotMem] Summary: saved={len(stored_triples)}, pending_bullets={len(bullets)}, turn={turn_id}")
        
        return bullets, stored_triples
    
    def prewarm(self, lang: str = "en") -> None:
        """Load NLP resources up-front to avoid first-turn latency"""
        # Prewarm extractor
        try:
            self.extractor.extract("Test", lang)
        except Exception:
            pass
        
        # Prewarm legacy components
        if self.use_srl and SRLExtractor is not None:
            if self._srl is None:
                self._srl = SRLExtractor(use_normalizer=True)
            try:
                if getattr(self._srl, 'normalizer', None) is not None:
                    self._srl.normalizer._ensure_model()
            except Exception:
                pass
        
        # Additional prewarm logic from original (needs extraction)
        self._prewarm_legacy_components(lang)
    
    def store_assistant_response(self, session_id: str, response: str, turn_id: int):
        """Store assistant response and generate session summary if needed"""
        # Store assistant message verbatim
        self.session_store.add_message(session_id, "assistant", response, turn_id)
        
        # Generate session summary every few turns or at session end
        conversation = self.session_store.get_session_conversation(session_id)
        if len(conversation) >= 4 or turn_id >= 10:  # Summary after 4 messages or 10 turns
            self._generate_session_summary(session_id, conversation)
    
    def _generate_session_summary(self, session_id: str, conversation: List[SessionMessage]):
        """Generate and store session summary"""
        try:
            # Use the summarizer service if available
            from services.summarizer import periodic_summarizer
            
            # Format conversation for summarization
            conversation_text = "\n".join([
                f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                for msg in conversation[-8:]  # Use last 8 messages for summary
            ])
            
            # Generate summary
            summary = periodic_summarizer.summarize_text(conversation_text, session_id)
            
            if summary:
                self.session_store.add_session_summary(session_id, summary, "auto")
                logger.info(f"📝 Generated summary for session {session_id}: {len(summary)} chars")
                
        except Exception as e:
            logger.warning(f"Failed to generate session summary: {e}")
    
    def get_session_context(self, session_id: str) -> str:
        """Get session context for retrieval"""
        return self.session_store.get_conversation_context(session_id, max_messages=10)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = dict(self.metrics)
        metrics.update(self.extractor.get_metrics())
        metrics.update(self.retriever.get_metrics())
        
        # Add session metrics
        session_stats = self.session_store.get_stats()
        metrics.update(session_stats)
        
        return metrics
    
    def _extract_entities_light(self, text: str) -> List[str]:
        """Light entity extraction for retrieval context"""
        return self.extractor.extract_entities_light(text)
    
    # Language detection (keep simple implementation for now)
    def _detect_language(self, text: str) -> str:
        """Detect language - simple implementation"""
        return "en"
    
    # Entity refinement (keep for now)
    def _refine_entities_from_text(self, text: str, entities: List[str]) -> List[str]:
        """Refine entities from text - simple implementation"""
        return entities
    
    # Triple refinement (keep for now)
    def _refine_triples(self, text: str, triples: List[Tuple[str, str, str]], doc, intent, lang: str) -> List[Tuple[str, str, str]]:
        """Refine triples - simple implementation"""
        return triples
    
    # Legacy method for backward compatibility
    def _retrieve_context(self, query: str, entities: List[str], turn_id: int, intent=None) -> List[str]:
        """Legacy method for backward compatibility"""
        result = self.retriever.retrieve_context(query, entities, turn_id, intent=intent)
        return result.bullets
    
    # Legacy prewarm method
    def _prewarm_legacy_components(self, lang: str):
        """Legacy prewarm method - now uses service prewarm"""
        # Prewarm individual services to avoid recursion
        self.extractor.extract("test", lang)
        self.coreference_resolver.prewarm()
        # Other services are prewarmed as needed
    
    # Legacy method for backward compatibility
    def rebuild_from_store(self):
        """Rebuild in‑memory indices (entity_index, edge_meta) from persistent store.

        Called at startup to avoid a cold start for retrieval quality.
        """
        try:
            # Clear current hot indices
            self.entity_index.clear()
            self.edge_meta.clear()

            edge_count = 0
            # Load edges (src, rel, dst, weight)
            try:
                edges = self.store.get_all_edges()
            except Exception:
                edges = []

            for s, r, d, w in edges:
                self.entity_index[s].add((s, r, d))
                self.entity_index[d].add((s, r, d))
                # Initialize meta with known fields
                self.edge_meta[(s, r, d)] = {'ts': 0, 'weight': float(w)}
                edge_count += 1

            # Merge any stored metadata ( provenance/lang/span/props )
            try:
                metas = self.store.get_all_edge_meta()
            except Exception:
                metas = []
            for s, r, d, meta in metas:
                key = (s, r, d)
                base = self.edge_meta.get(key, {'ts': 0, 'weight': 1.0})
                try:
                    base.update(meta or {})
                except Exception:
                    pass
                self.edge_meta[key] = base

            # Ensure retriever sees the rebuilt indices
            self.retriever.entity_index = self.entity_index
            self.retriever.edge_meta = self.edge_meta

            logger.info(f"[HotMem] Rebuilt indices from store: entities={len(self.entity_index)}, edges={edge_count}")
        except Exception as e:
            logger.warning(f"[HotMem] Rebuild from store failed: {e}")


# Global flag for language detection
PYCLD3_AVAILABLE = False
try:
    import pycld3
    PYCLD3_AVAILABLE = True
except Exception:
    pass


logger.info("🎭 HotMemoryFacade initialized - backward compatibility maintained")
logger.info("🔄 Using extracted services internally while preserving original interface")
