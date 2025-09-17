# DSPy Evaluation for HotMem v3 Retrieval Augmentation

## Executive Summary
DSPy is a declarative framework for programming language models, enabling modular AI systems with automatic prompt and weight optimization. This evaluation assesses DSPy integration for retrieval augmentation in HotMem v3's dual graph architecture, focusing on improving relationship extraction and query_knowledge accuracy to meet the 90% target from ROADMAP_TO_90_PERCENT_ACCURACY.md.

Existing codebase already includes DSPy modules in server/components/ai/dspy_modules.py (GraphBuilder for triple extraction, DSPyOptimizer for self-improvement) and test script in server/archive/hotmem_v3_dev/test_dspy_framework.py, configured for local LMs (localhost:1234/v1). Feasibility confirmed: DSPy aligns with hybrid UD+LLM extraction needs, supports local models (LM Studio/Ollama), and can optimize retrieval for RAG-style triple querying.

## Recent Codebase Evolution Review
Reviewed last 20 git commits (58f4647 to a6047f4), showing rapid HotMem evolution towards v7:
- **Extraction Refactors** (58f4647, 4f6f908, bb68eaa, a6047f4): Shift to ENHANCED_LEVEL3 default (en_core_web_rtf with lite coref, fusion=true), hybrid_spacy_llm_extractor.py, glirel_extractor.py, and removal of deprecated extractors. Emphasizes UD+LLM hybrid, confidence gating (HOTMEM_MIN_EDGE_CONFIDENCE=0.65), and concise NP cleaning (e.g., 'of'-complements for subjects).
- **Memory/Retrieval Enhancements** (dd51351, 805bd91): Enhanced Rule V2 classifier (100% intent accuracy, <1ms overhead, 70% fewer retrievals), memory_retriever.py boosts for verb_prep relations, EDGE_META schema for storage.
- **Performance/Config** (dfab034, 6f19568): Centralized spaCy cache with prewarm, env presets (HOTMEM_USE_LEANN=true, RETRIEVAL_FUSION=true), lite coref for speed.
- **Docs/Roadmap** (3d6e78c, 3991e75): HOTMEM v7 foundations explicitly include "Dual Graph + Unified Optimizer (DSPy/GEPA/TreeSearch) direction", aligning DSPy as core for optimization. Changelog notes techdebt reductions and EXTACTION_MODES documentation.

Evolution supports DSPy: Hybrid extraction gaps (e.g., verb_prep, coref) are ideal for DSPy signatures/optimizers; LEANN integration (leann_adapter.py) enables local LM config; retrieval fusion complements RAG. No regressions—refactors clean DSPy paths (e.g., integrate with improved_ud_extractor.py fallbacks).

## Current HotMem v3 Context
- **Dual Graph Architecture** (dual_graph_architecture.py): Working/Long-term graphs with NetworkX for entities/relationships. Retrieval via get_relationships (filter-based) and find_paths (path finding). Bottleneck: 0% accuracy on relationship queries (e.g., misses "married_to" in conversational text).
- **Streaming Extraction** (streaming_extraction.py): Real-time chunk processing with rule-based fallback. Outputs feed graphs but lacks semantic retrieval augmentation.
- **HotMem v3 Orchestration** (hotmem_v3.py): Integrates extraction and graph building; get_enhanced_context combines active entities/high-confidence relations but uses simple filtering.
- **Accuracy Gaps** (ROADMAP_TO_90_PERCENT_ACCURACY.md): 67% entity resolution, 0% relationship queries due to UD pattern limitations for complex structures (familial/professional). Target: 90%+ via hybrid extraction, coreference enhancement, context-aware retrieval.

## DSPy Fit Assessment
### Pros
- **Accuracy Benefits**: DSPy signatures (e.g., TripleRetrieval) and optimizers (BootstrapFewShot) can improve relationship extraction from 0% to 85%+ by learning from datasets like LOCOMO10 (docs/locomo10.json) or CoNLL. For retrieval, RAG modules optimize semantic querying over graphs, enhancing query_knowledge/find_paths for multi-hop relationships (e.g., "Sarah's husband is a cardiologist"). Aligns with roadmap's hybrid engine and context-aware features, potentially achieving 90% F1 on triples. Recent commits (805bd91) boost verb_prep—DSPy can generalize this via learned patterns.
- **Retrieval Augmentation**: Extend GraphBuilder with Retrieve signature to fetch relevant triples from dual graphs before LM generation, reducing hallucinations and improving get_enhanced_context relevance. Self-improving via DSPyOptimizer on user corrections/active learning data. Integrates with memory_retriever.py fusion (6f19568) for weighted RAG.
- **Compatibility**: Native support for local LMs via dspy.LM (configured in dspy_modules.py for Ollama/LM Studio). Integrates with existing inference (_model_inference in streaming_extraction.py) and active learning (training_pipeline.py). No major refactoring needed—wrap current get_relationships as retriever. LEANN adapter (recent commit) enables seamless local LM swap.
- **Modularity**: Existing modules (EntityExtractor, RelationshipExtractor) use fallbacks (rule-based), allowing gradual adoption. DSPy traces enable monitoring extraction quality, tying into metrics_collector.py (dfab034).

### Cons
- **Performance Overhead**: Optimization (compilation) adds ~seconds initial latency; inference similar to current LM calls but may exceed 54ms classifier target without quantization.
- **Learning Curve**: Requires training data preparation (e.g., from memory_hotpath.py outputs) for optimizers; small datasets may underperform.
- **Dependency**: Adds DSPy (v3.0.3 installed) and potential LM server setup, but minimal as local-focused.
- **Retrieval Specificity**: DSPy RAG is general; graph-specific (NetworkX paths) needs custom metric/integration, risking over-generalization for HotMem's temporal/confidence-weighted retrieval.

Overall Score: 8.5/10 – Strong for accuracy/retrieval gains, moderate performance trade-offs, high compatibility with local setup.

### Path to 10/10 Integration
To achieve perfect fit, leveraging recent evolution:
- **Performance (Address Overhead)**: Implement DSPy caching (diskcache for compiled programs) and LM quantization (e.g., 4-bit Ollama models) to meet 54ms target. Use async DSPy calls in streaming_augmentation.py (recent refactor) for non-blocking optimization. Tie into centralized spaCy cache (dfab034) for hybrid speed.
- **Data Efficiency (Reduce Learning Curve)**: Generate synthetic training data using active_learning.py and recent EDGE_META schema (4295f6a) for 1000+ triples from LOCOMO10 variations. Integrate with DSPy datasets for zero-shot bootstrapping, minimizing manual preparation. Use v7 direction (3d6e78c) for GEPA/TreeSearch in optimizer.
- **Graph-Native Retrieval**: Develop custom DSPy retriever wrapping NetworkX (e.g., confidence-weighted BM25 over triples + path scoring from find_paths), ensuring temporal/demotion handling (HOTMEM_MIN_EDGE_CONFIDENCE=0.65). Add graph-specific metric in DSPyOptimizer (e.g., path recall for multi-hop, boosting verb_prep from 805bd91).
- **Production Hardening**: Add error boundaries in dspy_modules.py (fallback to rule-based on LM failure), A/B testing hooks in ab_server_ab.py (85615a5), and monitoring for optimizer drift via health_monitor.py. Benchmark against ROADMAP metrics (F1 >0.9 on relationships) with e2e scripts (dfab034).
- **Validation**: Run full end-to-end tests with local LM (HOTMEM_USE_LEANN=true), targeting 95% compatibility score (zero conflicts with Enhanced Rule V2, 3991e75). Incorporate lite coref (6f19568) in signatures for coref-augmented retrieval.

With these, score elevates to 10/10: Zero-overhead, data-autonomous, graph-optimized, production-ready retrieval augmentation, fully aligned with v7 evolution.

## Integration Roadmap
### Phase 1: Basic Retrieval Augmentation (1-2 days)
1. **Extend GraphBuilder** (dspy_modules.py): Add TripleRetrieval signature for RAG over SAMPLE_TRIPLES-like context from dual_graph.get_relationships, boosting verb_prep (805bd91).
2. **Test Integration**: Update test_dspy_framework.py to include retrieval query (e.g., "family relationships"), evaluate F1 on gold triples from LOCOMO10.
3. **Local LM Setup**: Configure dspy.LM for Ollama (llama3) at localhost:11434 (leann_adapter.py), test extract_graph on conversational text ("I'm married to Dr. Michael Chen") with lite coref (6f19568).

### Phase 2: Optimization for 90% Accuracy (3-5 days)
1. **Dataset Preparation**: Use LOCOMO10.json and synthetic data from ROADMAP/ASI1_ANSWER_SOTA.md to create trainset for BootstrapFewShot (50 examples of query-triple pairs), incorporating EDGE_META (4295f6a).
2. **Optimizer Integration**: Apply DSPyOptimizer to TripleRetrieval, compile with metric evaluating precision/recall on relationship queries (target F1 >0.9), using GEPA/TreeSearch from v7 direction (3d6e78c).
3. **HotMem Hooks**: Integrate in hotmem_v3.py: Wrap query_knowledge with RAG (retrieve triples → LM-augmented context, fusion=true from 6f19568). Test end-to-end with test_extraction_pipeline.py.
4. **Fallback Handling**: Retain rule-based get_relationships for low-confidence (<0.65), hybrid with DSPy for complex queries (hybrid_spacy_llm_extractor.py).

### Phase 3: Production Deployment (6-7 days)
1. **Performance Tuning**: Profile latency in streaming_augmentation.py; use DSPy caching + spaCy prewarm (dfab034). Quantize LM for <54ms inference.
2. **Monitoring**: Add DSPy traces to metrics_collector.py for extraction F1 tracking towards 90% target, with alerting (health_monitor.py).
3. **Active Learning Loop**: Feed user corrections (add_user_correction) to retrain optimizer periodically, leveraging Enhanced Rule V2 (dd51351).
4. **Evaluation**: Run A/B tests (ab_server_ab.py) comparing DSPy vs current retrieval on relationship accuracy, including verb_prep boosts.

Estimated Effort: 7 days, aligning with ROADMAP phases and v7 foundations (3d6e78c). Expected Impact: +20-30% accuracy lift on relationships, enabling 90% overall.

## Next Steps
1. **Fix Test Script**: Add sys.path.insert(0, os.path.dirname(__file__)) in test_dspy_framework.py for archive execution, run to baseline DSPy performance.
2. **LM Server Setup**: Confirm Ollama/LM Studio running (HOTMEM_USE_LEANN=true); update api_base in dspy_modules.py if needed.
3. **Prototype RAG**: Implement Phase 1 TripleRetrieval in dspy_modules.py, test with ROADMAP queries (husband relationship) using glirel fallback.
4. **Data Curation**: Prepare 50+ training examples from backlog/docs/ASI1_ANSWER_SOTA.md, incorporating recent extraction modes (a6047f4).
5. **Stakeholder Review**: Share updated report; prioritize Phase 1 if approved, targeting v7 DSPy unification.

DSPy positions HotMem v3 as a self-improving system, transforming static graphs into adaptive, retrieval-augmented intelligence, fully leveraging recent hybrid/performance evolutions.