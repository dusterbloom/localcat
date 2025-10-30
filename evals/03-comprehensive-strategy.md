# Comprehensive Memory Evaluation Strategy

**Implementation Time:** 1 week
**Frameworks:** RAGAS + BEIR + RagaAI Catalyst
**Goal:** Industry-standard, multi-dimensional evaluation of LocalCat's memory system

---

## Overview

This strategy combines three complementary evaluation frameworks to provide complete coverage of LocalCat's sophisticated memory system:

1. **RAGAS** (Days 1-2): RAG quality metrics (context precision, recall, faithfulness)
2. **BEIR** (Days 3-4): Retrieval benchmarks (NDCG, MAP, MRR) against industry standards
3. **RagaAI Catalyst** (Days 5-7): Agent-memory interaction tracing and visualization

Together, these frameworks test:
- ✅ Retrieval quality (BEIR)
- ✅ RAG generation quality (RAGAS)
- ✅ Agent behavior and memory interactions (RagaAI)
- ✅ Slot-aware routing (all frameworks)
- ✅ Cross-session persistence (all frameworks)
- ✅ Composite scoring effectiveness (BEIR + RAGAS)

---

## Week-by-Week Implementation Plan

### Days 1-2: RAGAS Foundation

**Goal**: Establish baseline RAG quality metrics

**Tasks**:
1. Install RAGAS and dependencies
2. Create adapter for LocalCat's memory retrieval
3. Build test dataset from production logs
4. Run initial evaluation
5. Document baseline metrics

**Deliverables**:
- `evals/scripts/evaluate_ragas.py`
- `evals/outputs/ragas/baseline_metrics.json`
- Baseline scores documented

**See**: [02-quick-start-ragas.md](./02-quick-start-ragas.md) for detailed steps

---

### Days 3-4: BEIR Retrieval Benchmarks

**Goal**: Benchmark LocalCat's retrieval against industry standards

#### What BEIR Tests

BEIR (Benchmarking IR) provides 15+ diverse datasets to test retrieval quality:
- **MS MARCO**: Web search queries
- **Natural Questions**: Question answering
- **HotpotQA**: Multi-hop reasoning
- **FiQA**: Financial QA
- **SciFact**: Scientific fact verification
- **TriviaQA**: Trivia questions
- And more...

#### Installation

```bash
cd /Users/peppi/Dev/localcat/server
source .venv/bin/activate
pip install beir sentence-transformers
```

#### LocalCat Adapter for BEIR

Create `evals/scripts/evaluate_beir.py`:

```python
#!/usr/bin/env python3
"""
BEIR evaluation adapter for LocalCat's memory system
"""

import sys
sys.path.insert(0, '/Users/peppi/Dev/localcat/server')

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

from core.memory.memory_store import MemoryStore
from core.memory.retrieval import MemoryRetrieval
import numpy as np


class LocalCatBEIRRetriever:
    """
    Adapter that makes LocalCat's retrieval compatible with BEIR
    """

    def __init__(self, db_path, session_id="beir_eval"):
        self.memory_store = MemoryStore(db_path=db_path, session_id=session_id)
        self.retrieval = MemoryRetrieval(self.memory_store)

        # Store corpus in LocalCat memory
        self.corpus_map = {}  # BEIR corpus_id -> LocalCat memory

    def load_corpus(self, corpus):
        """Load BEIR corpus into LocalCat memory"""
        print(f"📥 Loading {len(corpus)} documents into LocalCat memory...")

        for doc_id, doc_data in corpus.items():
            # Store document text as memory
            text = doc_data.get('title', '') + ' ' + doc_data.get('text', '')

            # Use frame processor to extract memories
            # For now, simple storage
            memory = {
                'text': text,
                'source': 'beir_corpus',
                'doc_id': doc_id
            }

            self.corpus_map[doc_id] = memory
            # TODO: Actually store in memory_store

        print(f"✅ Loaded {len(self.corpus_map)} documents")

    def search(self, corpus, queries, top_k, score_function="dot"):
        """
        BEIR-compatible search interface

        Args:
            corpus: Dict of doc_id -> {title, text}
            queries: Dict of query_id -> query_text
            top_k: Number of results to return
            score_function: Scoring function (ignored, we use composite scoring)

        Returns:
            Dict of query_id -> {doc_id: score}
        """
        results = {}

        for query_id, query_text in queries.items():
            # Use LocalCat's retrieval
            retrieved = self.retrieval.retrieve_relevant(
                query=query_text,
                slot="beir_eval",
                k=top_k,
                session_id="beir_eval"
            )

            # Convert to BEIR format
            query_results = {}
            for i, memory in enumerate(retrieved):
                doc_id = memory.get('doc_id', f'doc_{i}')
                score = memory.get('score', 1.0 / (i + 1))  # Use composite score
                query_results[doc_id] = score

            results[query_id] = query_results

        return results


def evaluate_on_beir_dataset(
    dataset_name="nfcorpus",  # Start with small dataset
    db_path="/tmp/localcat_beir.db"
):
    """
    Evaluate LocalCat on a BEIR dataset

    Args:
        dataset_name: One of BEIR's datasets (nfcorpus, scifact, fiqa, etc.)
        db_path: Path to temporary database for BEIR evaluation
    """

    print(f"\n{'='*60}")
    print(f"Evaluating LocalCat on BEIR: {dataset_name}")
    print(f"{'='*60}\n")

    # Download and load BEIR dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, out_dir="./datasets")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    print(f"📊 Dataset Statistics:")
    print(f"  Corpus size: {len(corpus)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Relevance judgments: {len(qrels)}")

    # Initialize LocalCat retriever
    retriever = LocalCatBEIRRetriever(db_path=db_path)
    retriever.load_corpus(corpus)

    # Run evaluation
    results = retriever.search(
        corpus=corpus,
        queries=queries,
        top_k=100,
        score_function="dot"
    )

    # Compute BEIR metrics
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels=qrels,
        results=results,
        k_values=[1, 3, 5, 10, 100]
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"📊 BEIR Evaluation Results: {dataset_name}")
    print(f"{'='*60}\n")

    print("NDCG (Normalized Discounted Cumulative Gain):")
    for k, score in ndcg.items():
        print(f"  NDCG@{k:3d}: {score:.4f}")

    print("\nMAP (Mean Average Precision):")
    for k, score in _map.items():
        print(f"  MAP@{k:3d}: {score:.4f}")

    print("\nRecall:")
    for k, score in recall.items():
        print(f"  Recall@{k:3d}: {score:.4f}")

    print("\nPrecision:")
    for k, score in precision.items():
        print(f"  P@{k:3d}: {score:.4f}")

    return {
        'ndcg': ndcg,
        'map': _map,
        'recall': recall,
        'precision': precision
    }


# Benchmark on multiple datasets
BEIR_DATASETS = [
    'nfcorpus',    # Small dataset (3.6K docs) - good for testing
    'scifact',     # Scientific fact checking (5K docs)
    'fiqa',        # Financial QA (57K docs)
    'nq',          # Natural Questions (2.7M docs) - larger
]


if __name__ == "__main__":
    # Start with small dataset
    results = evaluate_on_beir_dataset("nfcorpus")

    # Save results
    import json
    with open('./outputs/beir/nfcorpus_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to ./outputs/beir/nfcorpus_results.json")
```

#### Running BEIR Evaluation

```bash
cd /Users/peppi/Dev/localcat/evals/scripts
python evaluate_beir.py
```

#### Understanding BEIR Metrics

**NDCG@k (Normalized Discounted Cumulative Gain)**
- Measures ranking quality considering position
- Higher is better (0-1 scale)
- NDCG@10 > 0.6 is strong for most datasets

**MAP@k (Mean Average Precision)**
- Average precision across all queries
- Higher is better (0-1 scale)
- MAP@100 > 0.3 is competitive

**Recall@k**
- Percentage of relevant docs retrieved in top k
- Recall@100 > 0.8 means good coverage

**MRR (Mean Reciprocal Rank)**
- Position of first relevant result
- MRR > 0.5 means first result often relevant

**Deliverables**:
- `evals/scripts/evaluate_beir.py`
- `evals/outputs/beir/results_*.json` for each dataset
- Comparison table vs. baseline systems

---

### Days 5-7: RagaAI Catalyst Agent Tracing

**Goal**: Trace and visualize memory interactions in Pipecat pipeline

#### What RagaAI Catalyst Provides

- **Agent Tracing**: See every memory retrieval call in real-time
- **Execution Timeline**: Visualize memory operations during conversation
- **Performance Analytics**: Latency breakdown for each retrieval
- **Debugging Tools**: Identify why specific memories were/weren't retrieved

#### Installation

```bash
cd /Users/peppi/Dev/localcat/server
source .venv/bin/activate
pip install ragaai-catalyst
```

#### LocalCat Integration

RagaAI Catalyst uses decorators to instrument your code:

```python
# In core/memory/retrieval.py

from ragaai_catalyst import trace_retrieval, trace_agent

class MemoryRetrieval:
    @trace_retrieval(
        name="localcat_memory_retrieval",
        metadata={"system": "localcat", "component": "memory"}
    )
    def retrieve_relevant(
        self,
        query: str,
        slot: str,
        k: int = 10,
        session_id: Optional[str] = None
    ):
        """
        Retrieve relevant memories with RagaAI tracing
        """
        # Your existing retrieval logic
        results = self._multi_source_retrieve(query, slot, k)

        # RagaAI automatically logs:
        # - Query text
        # - Retrieved context
        # - Retrieval latency
        # - Scores and metadata

        return results
```

#### Instrumenting HotPathMemoryProcessor

```python
# In core/memory/memory_hotpath.py

from ragaai_catalyst import trace_agent_action

class HotPathMemoryProcessor:

    @trace_agent_action(
        name="hotpath_memory_process",
        action_type="memory_extraction"
    )
    async def process_frame(self, frame):
        """
        Process frame with RagaAI tracing
        """
        # RagaAI logs:
        # - Input frame
        # - Extracted memories
        # - Processing time
        # - Slot routing decision

        memories = await self._extract_memories(frame)
        return memories
```

#### Launching RagaAI Dashboard

```bash
# Start self-hosted dashboard
ragaai-catalyst serve --port 8080

# Dashboard available at http://localhost:8080
```

#### Dashboard Features

1. **Execution Timeline**: See memory operations in sequence
2. **Agent Graph**: Visualize HotPathMemoryProcessor → Retrieval → LLM flow
3. **Performance Metrics**:
   - Memory retrieval latency
   - Token usage per retrieval
   - Cache hit rates
4. **Debugging Tools**:
   - Why was memory X retrieved?
   - Why was memory Y not retrieved?
   - Slot routing decisions

#### Creating Evaluation Scenarios

Create `evals/scripts/evaluate_ragaai.py`:

```python
#!/usr/bin/env python3
"""
RagaAI Catalyst evaluation scenarios for LocalCat
"""

from ragaai_catalyst import RagaAIClient, Scenario

client = RagaAIClient(api_url="http://localhost:8080")

# Scenario 1: Slot-aware retrieval test
slot_scenario = Scenario(
    name="slot_routing_test",
    description="Test that slot-aware routing prevents cross-contamination",
    steps=[
        {
            "action": "memorize",
            "slot": "favorite_color",
            "text": "My favorite color is yellow"
        },
        {
            "action": "memorize",
            "slot": "favorite_number",
            "text": "My favorite number is 42"
        },
        {
            "action": "query",
            "slot": "favorite_color",
            "question": "What is my favorite color?",
            "expected_retrieval_from": ["favorite_color"],
            "expected_no_retrieval_from": ["favorite_number"]
        }
    ]
)

# Run scenario and trace
results = client.run_scenario(slot_scenario)

# Analyze trace
print(f"Slot routing successful: {results.slot_routing_correct}")
print(f"Cross-contamination detected: {results.cross_slot_retrieval}")
print(f"Retrieval latency: {results.avg_latency_ms}ms")
```

**Deliverables**:
- Instrumented LocalCat code with RagaAI decorators
- Dashboard running at localhost:8080
- `evals/scenarios/` folder with test scenarios
- Performance report from dashboard

---

## Integrated Evaluation Pipeline

Once all three frameworks are set up, create unified evaluation:

### Master Evaluation Script

Create `evals/scripts/run_full_evaluation.py`:

```python
#!/usr/bin/env python3
"""
Run complete LocalCat memory evaluation across all frameworks
"""

import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/Users/peppi/Dev/localcat/server')

# Import all evaluators
from evaluate_ragas import evaluate_localcat_memory as eval_ragas
from evaluate_beir import evaluate_on_beir_dataset as eval_beir
from evaluate_ragaai import run_agent_scenarios as eval_ragaai


def run_full_evaluation(output_dir="./outputs/comprehensive"):
    """
    Run evaluation across RAGAS, BEIR, and RagaAI Catalyst
    """

    timestamp = datetime.now().isoformat()
    results = {'timestamp': timestamp}

    print("\n" + "="*60)
    print("🚀 Running Full LocalCat Memory Evaluation")
    print("="*60 + "\n")

    # 1. RAGAS: RAG quality metrics
    print("📊 [1/3] Running RAGAS evaluation...")
    ragas_results = eval_ragas(test_queries=PRODUCTION_QUERIES)
    results['ragas'] = dict(ragas_results)
    print("✅ RAGAS complete\n")

    # 2. BEIR: Retrieval benchmarks
    print("📊 [2/3] Running BEIR evaluation...")
    beir_results = eval_beir(dataset_name="nfcorpus")
    results['beir'] = beir_results
    print("✅ BEIR complete\n")

    # 3. RagaAI: Agent tracing
    print("📊 [3/3] Running RagaAI evaluation...")
    ragaai_results = eval_ragaai(scenarios=AGENT_SCENARIOS)
    results['ragaai'] = ragaai_results
    print("✅ RagaAI complete\n")

    # Save comprehensive report
    output_path = Path(output_dir) / f"evaluation_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("📋 Comprehensive Evaluation Report")
    print("="*60 + "\n")

    print("RAGAS (RAG Quality):")
    print(f"  Context Precision:  {results['ragas']['context_precision']:.3f}")
    print(f"  Context Recall:     {results['ragas']['context_recall']:.3f}")
    print(f"  Faithfulness:       {results['ragas']['faithfulness']:.3f}")
    print(f"  Answer Relevance:   {results['ragas']['answer_relevance']:.3f}")

    print("\nBEIR (Retrieval Benchmarks):")
    print(f"  NDCG@10:  {results['beir']['ndcg']['NDCG@10']:.3f}")
    print(f"  MAP@100:  {results['beir']['map']['MAP@100']:.3f}")
    print(f"  Recall@100: {results['beir']['recall']['Recall@100']:.3f}")

    print("\nRagaAI (Agent Performance):")
    print(f"  Avg Retrieval Latency: {results['ragaai']['avg_latency_ms']:.1f}ms")
    print(f"  Slot Routing Accuracy: {results['ragaai']['slot_accuracy']:.1%}")
    print(f"  Cross-session Success: {results['ragaai']['cross_session_success']:.1%}")

    print(f"\n✅ Full report saved to {output_path}")
    print(f"🌐 View RagaAI dashboard at http://localhost:8080\n")

    return results


if __name__ == "__main__":
    results = run_full_evaluation()
```

### Running Complete Evaluation

```bash
cd /Users/peppi/Dev/localcat/evals/scripts
python run_full_evaluation.py
```

---

## Metrics Dashboard

Create a simple web dashboard to visualize all metrics:

### Dashboard Structure

```
evals/
├── dashboard/
│   ├── index.html          # Main dashboard
│   ├── styles.css
│   ├── dashboard.js        # Load and display metrics
│   └── charts/
│       ├── ragas.js        # RAGAS charts
│       ├── beir.js         # BEIR charts
│       └── ragaai.js       # RagaAI charts
```

### Simple HTML Dashboard

```html
<!-- evals/dashboard/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>LocalCat Memory Evaluation Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <h1>LocalCat Memory System Evaluation</h1>

    <div class="metrics-grid">
        <!-- RAGAS Metrics -->
        <div class="metric-card">
            <h2>RAGAS: RAG Quality</h2>
            <canvas id="ragas-chart"></canvas>
            <div id="ragas-scores"></div>
        </div>

        <!-- BEIR Metrics -->
        <div class="metric-card">
            <h2>BEIR: Retrieval Benchmarks</h2>
            <canvas id="beir-chart"></canvas>
            <div id="beir-scores"></div>
        </div>

        <!-- RagaAI Metrics -->
        <div class="metric-card">
            <h2>RagaAI: Agent Performance</h2>
            <canvas id="ragaai-chart"></canvas>
            <div id="ragaai-scores"></div>
        </div>

        <!-- Historical Trends -->
        <div class="metric-card full-width">
            <h2>Performance Over Time</h2>
            <canvas id="history-chart"></canvas>
        </div>
    </div>

    <script src="dashboard.js"></script>
</body>
</html>
```

### Launch Dashboard

```bash
cd /Users/peppi/Dev/localcat/evals/dashboard
python -m http.server 8000

# Open http://localhost:8000
```

---

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/memory-evaluation.yml`:

```yaml
name: Memory Evaluation

on:
  push:
    branches: [main, develop]
  pull_request:
  schedule:
    - cron: '0 2 * * *'  # Run nightly at 2 AM

jobs:
  evaluate:
    runs-on: macos-latest  # Need macOS for MLX

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          cd server
          pip install -r requirements.txt
          pip install ragas beir ragaai-catalyst

      - name: Run RAGAS evaluation
        run: |
          cd evals/scripts
          python evaluate_ragas.py --save-results

      - name: Run BEIR benchmarks
        run: |
          cd evals/scripts
          python evaluate_beir.py --dataset nfcorpus

      - name: Check metric thresholds
        run: |
          cd evals/scripts
          python check_thresholds.py

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-results
          path: evals/outputs/
```

### Metric Threshold Checker

Create `evals/scripts/check_thresholds.py`:

```python
#!/usr/bin/env python3
"""
Check that evaluation metrics meet minimum thresholds
"""

import json
import sys
from pathlib import Path

THRESHOLDS = {
    'ragas': {
        'context_precision': 0.7,
        'context_recall': 0.7,
        'faithfulness': 0.85,
        'answer_relevance': 0.75
    },
    'beir': {
        'NDCG@10': 0.4,
        'MAP@100': 0.2,
        'Recall@100': 0.6
    }
}

def check_thresholds(results_path="./outputs/comprehensive/latest.json"):
    """Check if evaluation results meet thresholds"""

    with open(results_path) as f:
        results = json.load(f)

    failures = []

    # Check RAGAS
    for metric, threshold in THRESHOLDS['ragas'].items():
        value = results['ragas'][metric]
        if value < threshold:
            failures.append(f"RAGAS {metric}: {value:.3f} < {threshold:.3f}")

    # Check BEIR
    for metric, threshold in THRESHOLDS['beir'].items():
        category, k = metric.split('@')
        value = results['beir'][category.lower()][metric]
        if value < threshold:
            failures.append(f"BEIR {metric}: {value:.3f} < {threshold:.3f}")

    if failures:
        print("❌ Evaluation failed - metrics below threshold:\n")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print("✅ All metrics meet thresholds")
        sys.exit(0)

if __name__ == "__main__":
    check_thresholds()
```

---

## Production Deployment

### Continuous Monitoring

Once evaluation is solid, deploy continuous monitoring:

```python
# In server/bot.py - add evaluation hooks

from core.memory.evaluation import EvaluationMonitor

monitor = EvaluationMonitor(
    ragas_enabled=True,
    beir_enabled=False,  # Too expensive for production
    sample_rate=0.01  # Sample 1% of queries
)

# Hook into pipeline
@pipeline.on_memory_retrieval
async def log_retrieval_quality(query, retrieved, response):
    """Continuously evaluate memory quality"""
    await monitor.evaluate_retrieval(query, retrieved, response)

# Periodic summary
@scheduler.every_hour
async def report_metrics():
    summary = monitor.get_hourly_summary()
    logger.info(f"Memory quality: {summary}")
```

---

## Success Metrics

After 1 week implementation, you should have:

### Quantitative Metrics
- ✅ RAGAS scores (4 metrics tracked over time)
- ✅ BEIR benchmarks (compared against baselines)
- ✅ RagaAI performance analytics (latency, accuracy)

### Qualitative Insights
- ✅ Understanding of which memories are well-retrieved
- ✅ Identification of retrieval failures
- ✅ Visibility into slot routing effectiveness
- ✅ Cross-session behavior validation

### Infrastructure
- ✅ Automated evaluation pipeline
- ✅ Metrics dashboard
- ✅ CI/CD integration
- ✅ Continuous monitoring in production

---

## Next Steps

1. **Week 1**: Implement this comprehensive strategy
2. **Week 2**: Iterate based on findings
   - Tune composite scoring weights
   - Improve slot routing
   - Optimize retrieval latency
3. **Week 3**: Add custom benchmarks
   - Voice-specific evaluation
   - Prosody-aware metrics
   - Real-time performance tests
4. **Week 4+**: Production optimization
   - A/B test memory improvements
   - User feedback integration
   - Continuous refinement

---

## Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [BEIR Benchmark](http://beir.ai)
- [BEIR Leaderboard](https://eval.ai/web/challenges/challenge-page/1897/leaderboard)
- [RagaAI Catalyst Docs](https://docs.raga.ai/)
- [RagaAI GitHub](https://github.com/raga-ai-hub/RagaAI-Catalyst)
- [LocalCat Memory Architecture](../docs/02-architecture/memory-system-map.md)

---

**Questions or issues?** Check the troubleshooting sections in each framework's documentation, or refer to the [01-industry-frameworks.md](./01-industry-frameworks.md) comparison matrix.
