# Industry Memory Evaluation Frameworks

Complete analysis of 14 frameworks found via parallel GitHub search (Oct 2025).

## 🏆 Top Tier: Production-Ready Frameworks

### 1. RAGAS ⭐ 11,244 (RECOMMENDED)

**Repository:** https://github.com/explodinggradients/ragas
**Status:** Very Active (last updated Oct 29, 2025)
**License:** Apache 2.0

**What It Tests:**
- Context precision: Are retrieved facts relevant to the query?
- Context recall: Were all relevant facts retrieved?
- Faithfulness: Does the answer stick to retrieved context? (hallucination detection)
- Answer relevance: Does the answer address the question?
- Aspect critique: Custom evaluation dimensions

**Why It's Best for LocalCat:**
- ✅ Designed specifically for RAG systems
- ✅ Works with any LLM (OpenAI, Claude, local LM Studio)
- ✅ Reference-free evaluation (no ground truth needed)
- ✅ Production-ready with extensive community
- ✅ Simple integration: `pip install ragas`

**Integration Complexity:** LOW (1 day)

**Metrics Format:**
```python
{
  "context_precision": 0.85,      # 0-1, higher better
  "context_recall": 0.92,         # 0-1, higher better
  "faithfulness": 0.88,           # 0-1, higher better
  "answer_relevance": 0.91        # 0-1, higher better
}
```

**Best For:** Your memory retrieval quality evaluation

---

### 2. BEIR ⭐ 1,984 (INDUSTRY STANDARD)

**Repository:** https://github.com/beir-cellar/beir
**Website:** http://beir.ai
**Status:** Active (last updated Oct 28, 2025)
**License:** Apache 2.0

**What It Tests:**
- Retrieval quality across 15+ diverse datasets
- Standard IR metrics: NDCG, MAP, Recall@k, MRR, P@k
- Cross-domain retrieval performance

**15+ Benchmark Datasets:**
- TREC, MS MARCO, Natural Questions
- HotpotQA, FiQA, SciFact, BioASQ
- NFCorpus, DBpedia, and more

**Why It's Best for LocalCat:**
- ✅ Industry-standard retrieval benchmarks
- ✅ Compare your slot-aware retrieval against baselines
- ✅ Well-established evaluation protocols
- ✅ Widely cited in academic research

**Integration Complexity:** MEDIUM (2-3 days)

**Metrics Format:**
```python
{
  "NDCG@10": 0.72,    # Ranking quality (0-1)
  "MAP": 0.68,        # Mean average precision (0-1)
  "Recall@100": 0.89, # Coverage (0-1)
  "MRR": 0.75,        # Mean reciprocal rank (0-1)
  "P@10": 0.82        # Precision at 10 (0-1)
}
```

**Best For:** Benchmarking your retrieval against standards

---

### 3. RagaAI Catalyst ⭐ 16,043 (AGENT-FOCUSED)

**Repository:** https://github.com/raga-ai-hub/RagaAI-Catalyst
**Status:** Very Active (last updated Oct 22, 2025)
**License:** Apache 2.0

**What It Tests:**
- Multi-agent system debugging and tracing
- Agent-to-tool interaction quality
- LLM and memory performance
- Context relevance and retrieval quality
- Execution graph visualization

**Why It's Best for LocalCat:**
- ✅ Specifically designed for agentic AI systems
- ✅ Traces your HotPathMemoryProcessor calls
- ✅ Visualizes memory retrieval in Pipecat pipeline
- ✅ Self-hosted dashboard (privacy-focused)
- ✅ Timeline-based debugging

**Integration Complexity:** MEDIUM-HIGH (3-4 days)

**Features:**
- Agent, LLM, and tool call tracing
- Execution graph timeline visualization
- Advanced analytics for memory interactions
- Self-hosted observability dashboard

**Best For:** Understanding memory behavior in your voice pipeline

---

### 4. DeepEval ⭐ 11,890

**Repository:** https://github.com/confident-ai/deepeval
**Status:** Very Active (last updated Oct 30, 2025)
**License:** Apache 2.0

**What It Tests:**
- RAG-specific metrics
- Hallucination detection
- Summarization quality
- Toxicity, bias
- Custom metrics framework

**Why Consider:**
- ✅ 10+ pre-built metrics
- ✅ Works with any LLM
- ✅ CI/CD ready
- ✅ REST API for integration
- ✅ Live dashboard and monitoring

**Integration Complexity:** MEDIUM (2-3 days)

**Best For:** Comprehensive LLM evaluation beyond just RAG

---

### 5. FlashRAG ⭐ 3,090 (ACADEMIC)

**Repository:** https://github.com/RUC-NLPIR/FlashRAG
**Status:** Active (last updated Oct 29, 2025)
**License:** MIT
**Published:** WWW2025 Resource Track

**What It Tests:**
- End-to-end RAG performance
- Multi-hop reasoning evaluation
- Efficiency metrics (speed, memory)
- Dataset-specific benchmarks

**Why Consider:**
- ✅ Academic-grade benchmarks
- ✅ Multiple RAG methods support
- ✅ Built-in dataset management
- ✅ Efficiency profiling

**Integration Complexity:** MEDIUM-HIGH (3-4 days)

**Best For:** Academic research or comprehensive benchmarking

---

## 🎯 Memory-Specific Frameworks

### 6. GoodAI Long-Term Memory ⭐ 41

**Repository:** https://github.com/GoodAI/goodai-ltm
**Status:** Active (Sept 2025)

**Focus:**
- Conversational memory persistence
- Autonomous learning agents
- Long-term context management

**Best For:** Long-term memory pattern research

---

### 7. Context-Keeper ⭐ 78

**Repository:** https://github.com/redleaves/context-keeper
**Status:** Active (Oct 2025)

**Focus:**
- LLM-driven memory management
- Persistent memory with RAG
- Vector search integration
- Cursor/VSCode integration

**Best For:** Understanding memory management patterns

---

## 📊 Retrieval & Ranking Frameworks

### 8. Rankify ⭐ 517

**Repository:** https://github.com/DataScienceUIBK/Rankify
**Status:** Active (Oct 2025)

**Features:**
- 40 pre-retrieved benchmark datasets
- 7+ retrieval techniques
- 24+ state-of-the-art re-ranking models
- Multiple RAG evaluation methods

**Best For:** Re-ranking experiments

---

### 9. Awesome-GraphRAG ⭐ 1,744

**Repository:** https://github.com/DEEP-PolyU/Awesome-GraphRAG
**Status:** Very Active (Oct 2025)

**Content:**
- Curated list of GraphRAG resources
- Benchmark datasets
- Open-source project implementations
- Research surveys

**Best For:** Understanding graph-based RAG approaches

---

## 🧪 Multi-Hop Reasoning Benchmarks

### 10. PropRAG ⭐ 17

**Repository:** https://github.com/ReLink-Inc/PropRAG
**Benchmarks:** MuSiQue, HotpotQA, 2Wiki

**Focus:** Multi-hop reasoning with context-rich propositions

---

### 11. GraphRAG Benchmarking

**Repository:** https://github.com/Baltsat/GraphRAG_benchmarking
**Benchmarks:** MuSiQue, HotpotQA, BabiLong QA1

**Focus:** Graph-based retrieval with local models (Ollama)

---

### 12. StepGame ⭐ 32

**Repository:** https://github.com/ShiZhengyan/StepGame
**Published:** AAAI 2022 (Oral)

**Focus:** Multi-hop spatial reasoning benchmark

---

## 🤖 Agent Memory Frameworks

### 13. MakerAi ⭐ 84

**Repository:** https://github.com/gustavoeenriquez/MakerAi
**Status:** Very Active (Oct 2025)

**Features:**
- RAG 2.0 with semantic memory
- Autonomous agents
- Visual workflow orchestration
- Multi-LLM support (OpenAI, Claude, Gemini, Ollama)

**Best For:** Agent framework patterns

---

### 14. Ditto (CrewRiz)

**Repository:** https://github.com/CrewRiz/Ditto
**Status:** Active (Oct 2025)

**Features:**
- Knowledge graph agents (ArangoDB)
- GPU acceleration (cuGraph)
- Semantic memory
- Interactive Streamlit GUI

**Best For:** Knowledge graph integration patterns

---

## 📈 Comparison Matrix

| Framework | Stars | Retrieval | RAG/Generation | Agent Memory | Ease of Integration | Maintenance |
|-----------|-------|-----------|----------------|--------------|---------------------|-------------|
| **RAGAS** | 11.2k | ✓✓✓ | ✓✓✓ | ✓✓ | ⭐⭐⭐⭐⭐ | Very Active |
| **BEIR** | 2k | ✓✓✓ | - | - | ⭐⭐⭐⭐ | Active |
| **RagaAI** | 16k | ✓✓ | ✓✓ | ✓✓✓ | ⭐⭐⭐ | Very Active |
| **DeepEval** | 11.9k | ✓✓ | ✓✓✓ | ✓✓ | ⭐⭐⭐⭐ | Very Active |
| **FlashRAG** | 3.1k | ✓✓✓ | ✓✓✓ | ✓ | ⭐⭐⭐ | Active |

---

## 🎯 Recommendation for LocalCat

### Immediate (1 Day):
**Start with RAGAS** - Fastest path to production metrics

### Short-term (3-4 Days):
**Add BEIR** - Benchmark your retrieval quality

### Medium-term (1 Week):
**Add RagaAI Catalyst** - Trace memory interactions in pipeline

### Long-term (2 Weeks):
**Experiment with others** - Multi-hop benchmarks, agent frameworks

---

## 📚 Key Benchmarks Identified

**Multi-Hop Reasoning:**
- HotpotQA
- MuSiQue
- 2Wiki
- BabiLong

**Standard Retrieval:**
- BEIR datasets (15+)
- MS MARCO
- Natural Questions
- TriviaQA

**Agent Evaluation:**
- Custom conversation traces
- Tool interaction logs
- Memory access patterns

---

## 🔗 Resources

- [RAGAS Docs](https://docs.ragas.io/)
- [BEIR Website](http://beir.ai)
- [RagaAI Docs](https://docs.raga.ai/)
- [FlashRAG Paper](https://arxiv.org/abs/2405.13576)

---

Next: Read [02-quick-start-ragas.md](./02-quick-start-ragas.md) for immediate implementation.
