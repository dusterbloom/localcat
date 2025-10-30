# LocalCat Memory System Evaluation

Industry-standard evaluation frameworks and strategies for localcat's sophisticated memory system.

## 🚀 Quick Start (Fastest Path)

**Want to start evaluating immediately?**

```bash
cd evals/scripts
./setup_ragas.sh
```

Then read: [02-quick-start-ragas.md](./02-quick-start-ragas.md)

## 📚 Documentation

### [01-industry-frameworks.md](./01-industry-frameworks.md)
Complete analysis of 14 industry frameworks found on GitHub:
- **RAGAS** (11k⭐) - RAG evaluation standard
- **BEIR** (2k⭐) - Retrieval benchmarking
- **RagaAI Catalyst** (16k⭐) - Agent memory evaluation
- And 11 more frameworks ranked by relevance

### [02-quick-start-ragas.md](./02-quick-start-ragas.md)
**Start here if you want to evaluate NOW:**
- 1-day integration guide
- Works with your MLX/LM Studio setup
- Metrics: Context precision, faithfulness, answer relevance
- Production-ready evaluation

### [03-comprehensive-strategy.md](./03-comprehensive-strategy.md)
**Multi-framework approach:**
- RAGAS + BEIR + RagaAI Catalyst integration
- Complete evaluation coverage
- Benchmarking against industry standards
- 1-week implementation plan

## 🎯 What Gets Evaluated

Your sophisticated memory system features:
- ✅ Multi-source retrieval (graph, convo, summary, semantic)
- ✅ 6-dimensional composite scoring
- ✅ Slot-aware routing (unique feature)
- ✅ Prosody integration
- ✅ Cross-session persistence

Industry frameworks test:
- **RAGAS**: Context relevance, faithfulness, answer quality
- **BEIR**: Retrieval precision, recall, ranking (NDCG, MAP, MRR)
- **RagaAI**: Agent-memory interactions, execution traces

## 📊 Recommended Approach

### Option 1: Quick Win (1 day)
Start with RAGAS only → Get immediate metrics on memory quality

### Option 2: Balanced (3-4 days)
RAGAS + BEIR → RAG quality + retrieval benchmarks

### Option 3: Comprehensive (1 week)
All 3 frameworks → Complete industry-standard evaluation

## 🔧 Requirements

- Python 3.12+
- Your existing localcat environment
- OpenAI API key (for RAGAS LLM-as-judge)
- Or use your local LM Studio endpoint

## 📈 Metrics You'll Get

**Context Quality:**
- Precision: Are retrieved facts relevant?
- Recall: Did we get all relevant facts?
- Faithfulness: No hallucinations?

**Retrieval Performance:**
- NDCG@10: Ranking quality
- MAP: Average precision
- MRR: First relevant result

**System Performance:**
- Latency per retrieval
- Memory usage
- Token consumption

## 🎬 Next Steps

1. Read [02-quick-start-ragas.md](./02-quick-start-ragas.md)
2. Run `scripts/setup_ragas.sh`
3. Evaluate your memory system
4. Track improvements over time

## 🔗 Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [BEIR Benchmark](http://beir.ai)
- [RagaAI Catalyst](https://github.com/raga-ai-hub/RagaAI-Catalyst)

---

**Questions?** Check the detailed docs in this folder or refer to the comprehensive strategy guide.
