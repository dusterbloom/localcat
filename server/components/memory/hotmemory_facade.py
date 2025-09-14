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

# Preferred extraction path: strategy registry
try:
    from components.extraction.extraction_registry import get_registry  # type: ignore
except Exception:
    get_registry = None  # type: ignore

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
# Lazy import fastcoref only when coref is enabled to avoid startup downloads
FCoref = None

# Layer 3: Relationship refinement
try:
    from components.semantic.semantic_filter import SemanticRelationshipFilter
except Exception:
    SemanticRelationshipFilter = None

# Layer 3: Temporal context (alias actual class name)
try:
    from components.temporal.temporal_extractor import TemporalContextExtractor as TemporalExtractor
except Exception:
    TemporalExtractor = None

# Layer 4: Graph optimization (alias actual class name)
try:
    from components.graph.graph_analyzer import KnowledgeGraphAnalyzer as GraphAnalyzer
except Exception:
    GraphAnalyzer = None


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
        # Canonical user entity id; link to USER_ID if provided for easy aliasing
        self.user_eid = "you"
        try:
            user_id_env = os.getenv("USER_ID", "").strip()
            if user_id_env and user_id_env.lower() != self.user_eid:
                # Map USER_ID alias to canonical 'you'
                store.enqueue_alias(user_id_env, self.user_eid)
        except Exception:
            pass
        
        # Load configuration
        self.config = create_config()
        self.config.max_recency = max_recency
        
        # Initialize extracted services
        self.extractor = MemoryExtractor(self.config.get_extractor_config())
        self.retriever = MemoryRetriever(store, defaultdict(set), self.config.get_retriever_config())
        self.coreference_resolver = CoreferenceResolver(self.config.get_coreference_config())
        self.assisted_extractor = AssistedExtractor(self.config.get_assisted_config())

        # Initialize extraction registry (preferred orchestration)
        self._extraction_registry = get_registry() if get_registry else None
        # Track last registry strategy for observability
        self._last_registry_strategy_used: Optional[str] = None

        # Initialize Layer 3: Relationship refinement
        semantic_config = {
            'semantic_filtering_enabled': self.config.features.use_semantic_filter,
            'semantic_similarity_threshold': 0.7
        }
        self.semantic_filter = SemanticRelationshipFilter(semantic_config) if SemanticRelationshipFilter else None

        temporal_config = {
            'temporal_extraction_enabled': self.config.features.use_temporal_extraction,
            # Quick Win #1: default to optimized spaCy temporal pipeline (override via env)
            'temporal_optimized_pipeline': os.getenv('TEMPORAL_OPTIMIZED_PIPELINE', 'true').lower() in ("1", "true", "yes"),
            'temporal_model': os.getenv('TEMPORAL_MODEL', 'en_core_web_sm')
        }
        self.temporal_extractor = TemporalExtractor(temporal_config) if TemporalExtractor else None

        # Initialize Layer 4: Graph optimization
        graph_config = {
            'graph_analysis_enabled': self.config.features.use_graph_analysis,
            'community_detection_enabled': True
        }
        self.graph_analyzer = GraphAnalyzer(graph_config) if GraphAnalyzer else None
        
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
        # Index user text into FTS for retrieval fusion (session-scoped)
        try:
            now_ts = int(time.time() * 1000)
            self.store.enqueue_mention(f"session:{session_id}", text[:500], now_ts, session_id, turn_id)
            self.store.flush_if_needed()
        except Exception:
            pass
        
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
            extraction_result = self._extract_with_registry(text, lang)
            _ = extraction_result.triples  # not persisted when pure question
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics['total_ms'].append(elapsed_ms)
            # Return no stored triples in this branch
            return bullets, []
        
        # Stage 1: Extract entities and relations using new extractor
        extract_start = time.perf_counter()
        extraction_result = self._extract_with_registry(text, lang)
        entities = extraction_result.entities
        triples = extraction_result.triples
        neg_count = extraction_result.negation_count
        doc = extraction_result.doc
        embeddings = getattr(extraction_result, 'embeddings', {}) or {}
        self.metrics['extraction_ms'].append((time.perf_counter() - extract_start) * 1000)
        # Debug logging for extraction junk
        logger.info(f"[DEBUG Extraction Raw] Triples: {[(s, r, d) for s, r, d in triples[:5]]}... (total {len(triples)})")
        logger.info(f"[DEBUG Extraction Raw] Entities: {entities[:5]}... (total {len(entities)})")
        if triples:
            logger.info(f"[DEBUG Extraction Sources] Sample deps if doc: {[(t.text, t.dep_) for t in doc[:5]] if doc else 'No doc'}")
        
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
            # Lazy import of fastcoref at first use
            global FCoref
            if FCoref is None:
                try:
                    from fastcoref import FCoref  # type: ignore
                except Exception:
                    FCoref = None
            coreference_result = self.coreference_resolver.resolve_coreferences(triples, doc, text)
            triples = coreference_result.resolved_triples
            logger.debug(f"[HotMem] Coreference resolved: {len(coreference_result.resolved_triples)} triples")

        # Layer 3: Apply semantic filtering if enabled
        if self.config.features.use_semantic_filter and self.semantic_filter:
            semantic_start = time.perf_counter()
            semantic_result = self.semantic_filter.filter_relationships(triples, text)
            triples = semantic_result.filtered_triples
            semantic_ms = (time.perf_counter() - semantic_start) * 1000
            logger.debug(f"[HotMem] Semantic filtering: {len(semantic_result.filtered_triples)} triples (removed: {len(semantic_result.removed_triples)}, time: {semantic_ms:.1f}ms)")

        # Layer 3: Apply temporal context extraction if enabled
        if self.config.features.use_temporal_extraction and self.temporal_extractor:
            temporal_start = time.perf_counter()
            temporal_result = self.temporal_extractor.extract_temporal_context(triples, text)
            # We keep triples unchanged; temporal context is tracked in result
            temporal_ms = (time.perf_counter() - temporal_start) * 1000
            ctx_count = temporal_result.extraction_stats.get('triples_with_context', 0)
            logger.debug(f"[HotMem] Temporal extraction: context on {ctx_count} triples (time: {temporal_ms:.1f}ms)")
        
        # Rebuild entities from refined triples + text context
        ent_from_triples: Set[str] = set()
        for s, r, d in triples:
            ent_from_triples.add(s)
            ent_from_triples.add(d)
        entities = self._refine_entities_from_text(text, list(ent_from_triples))

        # Layer 4: Apply graph analysis if enabled
        if self.config.features.use_graph_analysis and self.graph_analyzer:
            graph_start = time.perf_counter()
            graph_result = self.graph_analyzer.analyze_knowledge_graph(triples)
            graph_ms = (time.perf_counter() - graph_start) * 1000
            comms = len(graph_result.communities)
            stats = graph_result.graph_stats or {}
            logger.debug(f"[HotMem] Graph analysis: communities={comms}, stats={stats} (time: {graph_ms:.1f}ms)")

        # Stage 3: Quality filtering and storage (Top-K gating)
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
            # If using Enhanced Level3 via registry, prefer extractor confidences directly
            if self._last_registry_strategy_used == 'enhanced_level3':
                tmp = []
                for (s, r, d) in triples:
                    props = self._pending_edge_props.get((s, r, d)) or {}
                    try:
                        c = float(props.get('confidence', 0.0) or 0.0)
                    except Exception:
                        c = 0.0
                    # Use a safe default if extractor didn't provide
                    if c <= 0.0:
                        c = 0.7
                    tmp.append((s, r, d, c))
                filtered = tmp
            else:
                filtered = self.quality.filter_triples(triples, context={'conversation_text': text})
        except Exception:
            # Fallback to unfiltered triples if quality module fails
            filtered = [(s, r, d, 0.5) for (s, r, d) in triples]
        
        candidates = []
        for s, r, d, q_conf in filtered:
            # Intent-based allowlist
            should_store, legacy_conf = quality_filter.should_store_fact(s, r, d, intent)
            base_conf = float(max(q_conf, legacy_conf))
            # Incorporate extractor-provided confidence when available
            props_from_extractor = self._pending_edge_props.get((s, r, d)) or {}
            try:
                ext_conf = float(props_from_extractor.get('confidence', 0.0) or 0.0)
                base_conf = max(base_conf, ext_conf)
            except Exception:
                pass
            # Penalize generic UD-only predicates; prefer semantic relations
            generic = (r in {'subject_of', 'determined_by', 'prepositional_object_of'}) or r.startswith('verb:')
            factor = 0.4 if generic else 1.0
            score = base_conf * factor
            # Always update hot indices for retrieval, regardless of storage
            self.entity_index[s].add((s, r, d))
            self.entity_index[d].add((s, r, d))

            # Store edge metadata with embeddings if available
            edge_metadata = {'ts': now_ts, 'weight': score}
            if embeddings and (s, r, d) in embeddings:
                edge_metadata.update(embeddings[(s, r, d)])
                logger.debug(f"[HotMem] Added embedding for triple: {s} {r} {d}")
            # Add extractor props if present (verb, prep, normalized_relation, confidence)
            if props_from_extractor:
                try:
                    if 'props' in edge_metadata and isinstance(edge_metadata['props'], dict):
                        edge_metadata['props'].update(props_from_extractor)
                    else:
                        edge_metadata['props'] = dict(props_from_extractor)
                except Exception:
                    edge_metadata['props'] = props_from_extractor

            self.edge_meta[(s, r, d)] = edge_metadata
            # Enhanced Level3 override: store high-confidence semantic relations
            if self._last_registry_strategy_used == 'enhanced_level3' and not generic and base_conf >= conf_thresh:
                should_store = True

            # Enhanced Level3: force-candidate when extractor confidence meets min-edge gate
            try:
                min_edge_conf = float(os.getenv('HOTMEM_MIN_EDGE_CONFIDENCE', '0.8'))
            except Exception:
                min_edge_conf = 0.8
            if self._last_registry_strategy_used == 'enhanced_level3' and not generic and base_conf >= min_edge_conf:
                candidates.append((score, (s, r, d)))
            elif should_store and base_conf >= conf_thresh:
                candidates.append((score, (s, r, d)))

        # Top-K gating to prevent flooding the graph
        try:
            top_k = int(os.getenv('HOTMEM_STORE_TOPK', '8'))
        except Exception:
            top_k = 8
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, triple in candidates[:top_k]:
            stored_triples.append(triple)
        
        # Update recency with stored triples only
        for s, r, d in stored_triples:
            self.recency_buffer.append(RecencyItem(s, r, d, text, now_ts, turn_id))
        
        # Stage 4: Context retrieval using new MemoryRetriever
        retrieve_start = time.perf_counter()
        retrieval_result = self.retriever.retrieve_context(text, entities, turn_id, intent=intent)
        bullets = retrieval_result.bullets
        self.metrics['retrieval_ms'].append((time.perf_counter() - retrieve_start) * 1000)
        
        # Store final triples and link to session (with per-triple confidence)
        if stored_triples:
            for i, (s, r, d) in enumerate(stored_triples):
                # Confidence from extractor props (fallback to 0.8)
                props = self._pending_edge_props.get((s, r, d)) or {}
                try:
                    conf = float(props.get('confidence', 0.8) or 0.8)
                except Exception:
                    conf = 0.8
                # Persist edge with observed confidence
                self.store.observe_edge(s, r, d, conf, now_ts + i)
                # Mentions for FTS
                self.store.enqueue_mention(s, f"{s} {r} {d}", now_ts + i, session_id, i)
                self.store.enqueue_mention(d, f"{s} {r} {d}", now_ts + i, session_id, i)
                # Persist edge metadata with extractor props
                if props:
                    self.store.enqueue_edge_meta(s, r, d, prov=prov_tag, lang=lang, span=None, props=props, ts=now_ts + i)
                # Link extracted knowledge to session
                edge_id = self.store.edge_id(s, r, d)
                self.session_store.link_knowledge_to_session(session_id, edge_id, "extracted", conf)
            # Ensure persistence
            self.store.flush()
        
        # Track performance
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics['total_ms'].append(elapsed_ms)
        
        # Memory summary line
        logger.info(f"[HotMem] Summary: saved={len(stored_triples)}, pending_bullets={len(bullets)}, turn={turn_id}")
        
        return bullets, stored_triples

    def _extract_with_registry(self, text: str, lang: str) -> ExtractionResult:
        """Extract via strategy registry using default/fallback strategies.
        Returns an ExtractionResult compatible with legacy consumers.
        """
        try:
            if not self._extraction_registry:
                return ExtractionResult([], [], 0, None)

            default_name = os.getenv('DEFAULT_EXTRACTION_STRATEGY', 'asi1')
            fallback_name = os.getenv('FALLBACK_EXTRACTION_STRATEGY', 'asi2')

            def run(name: str):
                try:
                    strat = self._extraction_registry.get_strategy(name)
                    if not strat:
                        return [], None
                    triples = strat.extract(text, lang) or []
                    return triples, strat
                except Exception:
                    return [], None

            used_strategy = None
            used_strat_obj = None
            triples, strat_obj = run(default_name)
            if triples:
                used_strategy = default_name
                used_strat_obj = strat_obj
            elif fallback_name and fallback_name != default_name:
                triples, strat_obj = run(fallback_name)
                if triples:
                    used_strategy = fallback_name
                    used_strat_obj = strat_obj

            # Record and log strategy selection
            self._last_registry_strategy_used = used_strategy
            try:
                logger.debug(f"[Registry] strategy={used_strategy or 'none'} triples={len(triples)}")
            except Exception:
                pass

            # Capture per-triple props (confidence, verb, prep) if provided by strategy
            try:
                if used_strategy == 'enhanced_level3' and used_strat_obj is not None:
                    last_map = getattr(used_strat_obj, 'last_props_map', None)
                    if isinstance(last_map, dict):
                        for (s, r, d), props in last_map.items():
                            if s and r and d and isinstance(props, dict):
                                self._pending_edge_props[(s, r, d)] = dict(props)
            except Exception:
                pass

            # Derive lightweight entities from triples (subject/object set)
            entities: List[str] = []
            if triples:
                try:
                    ents = set()
                    for s, r, d in triples:
                        if s:
                            ents.add(str(s))
                        if d:
                            ents.add(str(d))
                    entities = list(ents)[:50]
                except Exception:
                    entities = []

            return ExtractionResult(entities, triples, 0, None)
        except Exception:
            return ExtractionResult([], [], 0, None)
    
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
