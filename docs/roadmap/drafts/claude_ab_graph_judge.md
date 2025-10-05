# A/B Testing Framework: Hybrid Extraction with GraphJudge
*Comparative evaluation of rule-based, hybrid, and LLM extraction methods*

## Executive Summary

This document outlines a comprehensive A/B testing framework for evaluating multiple graph extraction approaches in LocalCat. By combining your existing YAML patterns, recovered hybrid SpaCy+LLM extractor, new SLM refinement layers, and GraphJudge quality scoring, we can identify the optimal extraction strategy that balances accuracy (<80% F1) with latency (<200ms).

## 🎯 Objectives

1. **Quantify tradeoffs** between speed and accuracy across extraction methods
2. **Identify optimal routing** based on sentence complexity
3. **Validate GraphJudge** as a quality filter
4. **Establish baselines** for GEPA optimization
5. **Ensure production readiness** with <200ms p95 latency

## 📊 Methods Under Test

### Method A: Pure YAML (Baseline)
```python
# server/core/memory/extractors/yaml_runtime.py
extractor = YAMLExtractor("ASI1_index_v0_9.yaml")
```
- **Profile**: Ultra-fast rule-based extraction
- **Latency**: 30-50ms
- **Expected F1**: 0.60 (broad), 0.85 (focused)
- **Use Case**: Simple sentences, real-time voice

### Method B: YAML + GraphJudge
```python
# With distilled judge filter
os.environ["YAML_GRAPH_JUDGE"] = "on"
os.environ["YAML_GRAPH_JUDGE_MODEL"] = "models/graph_judge.json"
```
- **Profile**: YAML with quality filtering
- **Latency**: 40-60ms
- **Expected F1**: 0.65-0.70 (precision boost)
- **Use Case**: Reducing false positives

### Method C: Hybrid SpaCy+LLM (Recovered)
```python
# server/core/memory/extractors/recovered_hybrid.py
# Complexity-adaptive extraction
extractor = HybridSpacyLLMExtractor(
    llm_model="qwen2.5-coder:3b",
    complexity_threshold=0.6
)
```
- **Profile**: Adaptive complexity routing
- **Latency**: 50ms (simple) to 500ms (complex)
- **Expected F1**: 0.75-0.85
- **Use Case**: Mixed complexity conversations

### Method D: YAML + SLM Refinement
```python
# server/core/memory/extractors/hybrid_slm.py
extractor = YAMLWithSLMRefinement(
    base_extractor=YAMLExtractor(),
    slm_model="mlx-community/Qwen2.5-0.5B-4bit"
)
```
- **Profile**: Two-stage extraction with error correction
- **Latency**: 150-200ms
- **Expected F1**: 0.70-0.80
- **Use Case**: Balance speed/accuracy for voice

### Method E: DSPy Enhanced
```python
# server/archive/experimental/.../dspy_extractor.py
extractor = DSPyEdgeExtractor(
    model="openai/gpt-4o-mini",
    base_url=os.getenv("DSPY_BASE_URL")
)
```
- **Profile**: LLM-based with few-shot learning
- **Latency**: 300-800ms
- **Expected F1**: 0.85-0.95
- **Use Case**: Offline reprocessing, high accuracy needs

## 🧪 Test Design

### Dataset Structure
```python
# server/tests/data/complexity_test_set.json
{
  "simple": [
    {
      "id": "simple_001",
      "text": "John works at Google",
      "complexity_score": 0.2,
      "gold": [["John", "works_at", "Google"]],
      "expected_methods": ["yaml", "yaml_judge"]
    }
  ],
  "medium": [
    {
      "id": "medium_001",
      "text": "After meeting Sarah at MIT, John founded TechCo",
      "complexity_score": 0.5,
      "gold": [
        ["John", "met", "Sarah"],
        ["meeting_location", "is", "MIT"],
        ["John", "founded", "TechCo"]
      ],
      "expected_methods": ["hybrid", "yaml_slm"]
    }
  ],
  "complex": [
    {
      "id": "complex_001",
      "text": "The CEO, who had previously worked at three Fortune 500 companies before starting his own venture which was later acquired by Google for an undisclosed amount, announced quarterly earnings",
      "complexity_score": 0.8,
      "gold": [...],
      "expected_methods": ["dspy", "hybrid"]
    }
  ]
}
```

### Evaluation Metrics
```python
class ExtractionMetrics:
    """Comprehensive metrics for extraction evaluation"""

    @staticmethod
    def evaluate(predictions, gold, latency_ms):
        # Core metrics
        tp, fp, fn = calculate_confusion(predictions, gold)

        return {
            # Accuracy metrics
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "f1": calculate_f1(precision, recall),

            # Performance metrics
            "latency_ms": latency_ms,
            "latency_category": categorize_latency(latency_ms),

            # Quality metrics (via GraphJudge)
            "judge_score": graph_judge_score(predictions),
            "confidence_avg": average_confidence(predictions),

            # Error analysis
            "error_types": categorize_errors(predictions, gold),
            "complexity_handling": assess_complexity_performance()
        }
```

## 🔧 Implementation Architecture

### A/B Test Harness
```python
# server/scripts/eval_extraction_ab.py
class ExtractionABTestHarness:
    """A/B testing framework for extraction methods"""

    def __init__(self):
        self.extractors = self._initialize_extractors()
        self.datasets = self._load_datasets()
        self.judge = self._load_graph_judge()

    def _initialize_extractors(self):
        return {
            "yaml": YAMLExtractor(self.yaml_path),
            "yaml_judge": YAMLWithJudge(self.yaml_path),
            "hybrid_spacy": HybridSpacyLLMExtractor(),
            "yaml_slm": YAMLWithSLMRefinement(),
            "dspy": DSPyEdgeExtractor()
        }

    async def run_test(self, dataset_name="all"):
        """Run A/B test across all methods"""
        results = defaultdict(list)

        for example in self.datasets[dataset_name]:
            # Run all extractors in parallel
            tasks = []
            for name, extractor in self.extractors.items():
                tasks.append(self._test_extractor(
                    name, extractor, example
                ))

            method_results = await asyncio.gather(*tasks)

            # Aggregate results
            for result in method_results:
                results[result["method"]].append(result)

        return self._analyze_results(results)

    async def _test_extractor(self, name, extractor, example):
        """Test single extractor with timeout"""
        try:
            start = time.perf_counter()

            # Extract with timeout
            predictions = await asyncio.wait_for(
                self._extract(extractor, example["text"]),
                timeout=1.0  # 1 second max
            )

            latency = (time.perf_counter() - start) * 1000

            # Evaluate
            metrics = ExtractionMetrics.evaluate(
                predictions,
                example["gold"],
                latency
            )

            return {
                "method": name,
                "example_id": example["id"],
                **metrics
            }

        except asyncio.TimeoutError:
            return {
                "method": name,
                "example_id": example["id"],
                "timeout": True,
                "latency_ms": 1000
            }
```

### Complexity Assessment
```python
class ComplexityAnalyzer:
    """Assess sentence complexity for routing decisions"""

    def assess(self, text: str, doc=None) -> float:
        """Return complexity score 0-1"""

        if doc is None:
            doc = self.nlp(text)

        features = {
            "clause_count": self._count_clauses(doc),
            "conjunction_count": self._count_conjunctions(doc),
            "entity_count": len(doc.ents),
            "depth": self._parse_tree_depth(doc),
            "length": len(doc),
            "passive": self._has_passive(doc),
            "relative_clauses": self._count_relative_clauses(doc)
        }

        # Weighted scoring
        score = 0.0
        score += min(features["clause_count"] * 0.15, 0.3)
        score += min(features["conjunction_count"] * 0.1, 0.2)
        score += min(features["depth"] / 10, 0.2)
        score += min(features["length"] / 50, 0.15)
        score += 0.1 if features["passive"] else 0
        score += min(features["relative_clauses"] * 0.05, 0.15)

        return min(score, 1.0)
```

### Smart Routing Strategy
```python
class ExtractionRouter:
    """Route to optimal extractor based on context"""

    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.performance_history = defaultdict(list)

    def route(self, text: str, context: dict = None) -> str:
        """Select best extraction method"""

        # Assess complexity
        complexity = self.complexity_analyzer.assess(text)

        # Check context hints
        require_speed = context.get("realtime", False)
        require_accuracy = context.get("high_accuracy", False)

        # Routing logic
        if require_speed and complexity < 0.4:
            return "yaml"  # Fastest for simple

        if require_accuracy and complexity > 0.7:
            return "dspy"  # Most accurate for complex

        if complexity < 0.3:
            return "yaml_judge"  # Fast with quality filter

        if complexity < 0.6:
            return "yaml_slm"  # Balanced approach

        if complexity < 0.8:
            return "hybrid_spacy"  # Adaptive routing

        return "dspy"  # Complex sentences

    def update_history(self, method: str, performance: dict):
        """Track performance for adaptive routing"""
        self.performance_history[method].append({
            "f1": performance["f1"],
            "latency": performance["latency_ms"],
            "timestamp": time.time()
        })
```

## 📈 Benchmarking Protocol

### Test Execution
```bash
# 1. Baseline YAML
YAML_DENSITY_CAPS=off \
YAML_GRAPH_JUDGE=off \
python server/scripts/eval_extraction_ab.py \
  --method yaml \
  --dataset server/tests/data/yaml_eval_l1_en_medium.json

# 2. YAML with GraphJudge
YAML_GRAPH_JUDGE=on \
YAML_GRAPH_JUDGE_MODEL=models/graph_judge.json \
python server/scripts/eval_extraction_ab.py \
  --method yaml_judge \
  --dataset server/tests/data/yaml_eval_l1_en_medium.json

# 3. Hybrid SpaCy+LLM
HOTMEM_LLM_ASSISTED=true \
HOTMEM_LLM_ASSISTED_MODEL=qwen2.5-coder:3b \
HOTMEM_COMPLEXITY_THRESHOLD=0.6 \
python server/scripts/eval_extraction_ab.py \
  --method hybrid_spacy \
  --dataset server/tests/data/complexity_test_set.json

# 4. YAML + SLM Refinement
EXTRACTION_METHOD=yaml_slm \
SLM_MODEL_PATH=mlx-community/Qwen2.5-0.5B-4bit \
python server/scripts/eval_extraction_ab.py \
  --method yaml_slm \
  --dataset server/tests/data/complexity_test_set.json

# 5. Full A/B comparison
python server/scripts/eval_extraction_ab.py \
  --methods all \
  --dataset all \
  --output results/ab_test_results.json
```

### Results Visualization
```python
# server/scripts/visualize_ab_results.py
def generate_report(results_path: str):
    """Generate comprehensive A/B test report"""

    results = json.loads(Path(results_path).read_text())

    # Performance matrix
    print("\n=== Performance Matrix ===")
    print("Method         | Simple F1 | Medium F1 | Complex F1 | Avg Latency")
    print("---------------|-----------|-----------|------------|------------")
    for method in results:
        print(f"{method:14} | {simple_f1:9.3f} | ...")

    # Pareto frontier (F1 vs Latency)
    plot_pareto_frontier(results)

    # Error analysis
    print("\n=== Error Analysis ===")
    for method in results:
        print(f"\n{method}:")
        print(f"  Most common errors: {results[method]['error_types']}")

    # Recommendations
    print("\n=== Recommendations ===")
    print(recommend_production_config(results))
```

## 🎯 Expected Outcomes

### Performance Targets
| Method | Simple (F1) | Medium (F1) | Complex (F1) | P95 Latency | Cost |
|--------|------------|-------------|--------------|-------------|------|
| YAML | 0.85 | 0.60 | 0.40 | 50ms | $0 |
| YAML+Judge | 0.80 | 0.65 | 0.45 | 60ms | $0 |
| Hybrid SpaCy | 0.88 | 0.75 | 0.70 | 300ms* | $0.001 |
| YAML+SLM | 0.82 | 0.70 | 0.65 | 200ms | $0 |
| DSPy | 0.92 | 0.88 | 0.85 | 800ms | $0.005 |

*Adaptive: 50ms simple, 500ms complex

### Decision Matrix
```python
def select_production_config(requirements):
    """Select optimal configuration based on requirements"""

    if requirements["use_case"] == "realtime_voice":
        if requirements["accuracy_priority"] == "high":
            return "yaml_slm"  # Best balance
        else:
            return "yaml_judge"  # Fastest with quality

    elif requirements["use_case"] == "batch_processing":
        return "dspy"  # Maximum accuracy

    elif requirements["use_case"] == "adaptive":
        return "hybrid_spacy"  # Complexity-aware

    else:
        return "yaml"  # Safe default
```

## 🚀 Integration Plan

### Phase 1: Setup (Day 1)
- [ ] Recover `hybrid_spacy_llm_extractor.py` from git
- [ ] Create `hybrid_slm.py` with Qwen2.5-0.5B
- [ ] Set up `complexity_test_set.json`
- [ ] Update `.env` with A/B test configs

### Phase 2: Testing (Day 2-3)
- [ ] Run individual method benchmarks
- [ ] Execute full A/B comparison
- [ ] Analyze results with GraphJudge
- [ ] Generate performance report

### Phase 3: Optimization (Day 4-5)
- [ ] Fine-tune complexity thresholds
- [ ] Optimize SLM prompts
- [ ] Train custom GraphJudge model
- [ ] Implement production routing

### Phase 4: Deployment (Day 6-7)
- [ ] Update `memory_hotpath.py` with router
- [ ] Add monitoring/telemetry
- [ ] Document configuration
- [ ] Create migration guide

## 📊 Monitoring & Iteration

### Runtime Metrics
```python
# Track in production
EXTRACTION_METRICS = {
    "method_used": str,
    "complexity_score": float,
    "extraction_latency_ms": float,
    "triples_extracted": int,
    "judge_score": float,
    "routing_reason": str
}
```

### Continuous Improvement
1. **Weekly A/B tests** on new conversation data
2. **GEPA optimization** using test results
3. **GraphJudge retraining** on false positives
4. **Complexity model updates** based on errors

## 🔬 Research Opportunities

### Future Enhancements
1. **Ensemble voting**: Combine multiple methods
2. **Confidence calibration**: Better uncertainty estimates
3. **Cross-lingual transfer**: Share patterns across languages
4. **Active learning**: Request labels for uncertain cases

### Publication Targets
- "Adaptive Extraction Routing for Real-time Voice Agents"
- "GraphJudge: Distilled Quality Scoring for Knowledge Graphs"
- "Hybrid Rule-LLM Systems: Best of Both Worlds"

## 📚 References

### Internal Documents
- `docs/implementation/ASI1_progress_plan.md`
- `backlog/drafts/gepa_self_improving_extraction.md`
- `server/scripts/eval_extraction.py`

### External Resources
- [GraphJudge Dataset](https://github.com/hhy-huang/GraphJudge)
- [DSPy Framework](https://dspy.ai)
- [Qwen2.5 Models](https://huggingface.co/Qwen)

---

*This A/B testing framework enables data-driven decisions about extraction methods, ensuring LocalCat achieves the optimal balance of speed and accuracy for real-time voice interactions.*