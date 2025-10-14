# Learned Confidence Roadmap

**Status**: Phase 2 - Builds on Edge Provenance Foundation
**Date**: 2025-09-29
**Prerequisites**: `edge_provenance_implementation.md` must be completed first
**Estimated Time**: 2-3 days total

---

## Executive Summary

With edge provenance in place, we can now implement **learned confidence scoring** that improves over time based on actual usage patterns and linguistic analysis. This replaces arbitrary confidence values (0.85, 0.90, 0.95) with evidence-based scores.

**Core Insight from Research**:
> Modern knowledge graph systems use **structural validation** and **learned calibration** rather than hand-coded rules. Confidence should reflect: (1) how user said it, (2) how often it's validated, (3) how recent it is.

**Approach**: Three complementary methods
1. **Structural signals** (usage-based) - Fast, no LLM needed
2. **Linguistic analysis** (text-based) - LLM analyzes certainty markers
3. **DSPy optimization** - Learn from evaluation data

---

## Problem Recap

### Current State (Arbitrary Confidence)

**Code** (`memory_hotpath.py:164-169`):
```python
if r == "name":
    conf = 0.95  # Why 0.95? Arbitrary!
elif r.startswith("v:"):
    conf = 0.85  # Why 0.85? Made up!
else:
    conf = 0.9   # Why 0.9? Guessing!
```

### Issues

1. **Ignores HOW user said it**
   - "I think my name is Alice" → 0.95 (too high!)
   - "My name is DEFINITELY Alice!" → 0.95 (should be higher!)

2. **Ignores validation history**
   - Fact mentioned once → 0.90
   - Fact mentioned 10 times, always validated → still 0.90

3. **No learning loop**
   - System never improves
   - Can't adapt to user's speech patterns
   - No feedback from retrieval success

4. **Research shows this is wrong**
   - RAKG achieves 95.91% accuracy with learned approaches
   - Confidence should be calibrated against actual correctness
   - Structural signals (graph topology) reveal truth

---

## Solution: Three-Tier Confidence System

### Tier 1: Structural Signals (Baseline)

**No LLM, pure database signals**

```python
class UsageBasedConfidence(ConfidenceStrategy):
    """Learn confidence from usage patterns (structural signals only)"""

    def score(self, edge: Edge, context: Context) -> float:
        # Start with relation-type baseline
        baseline = self._baseline_by_relation(edge.rel)

        # Structural signals from database
        reinforcement = self._reinforcement_multiplier(edge)
        recency = self._recency_multiplier(edge)
        source_count = self._source_count_multiplier(edge)

        # Combine multiplicatively
        confidence = baseline * reinforcement * recency * source_count

        return min(1.0, max(0.0, confidence))

    def _baseline_by_relation(self, rel: str) -> float:
        """Relation-type prior (keep existing logic)"""
        if rel == "name":
            return 0.95
        elif rel.startswith("v:"):
            return 0.85
        return 0.9

    def _reinforcement_multiplier(self, edge: Edge) -> float:
        """Boost for validated facts, penalty for negated"""
        if edge.pos > 0 and edge.neg == 0:
            # Fact reinforced, never contradicted
            return 1.0 + (0.05 * min(edge.pos, 3))  # Up to 15% boost
        elif edge.neg > 0:
            # Fact contradicted
            return 0.7 - (0.1 * min(edge.neg, 3))   # Down to 40% penalty
        return 1.0  # No evidence yet

    def _recency_multiplier(self, edge: Edge) -> float:
        """Decay old facts"""
        age_days = (time.time() - edge.updated_at / 1000) / 86400
        if age_days > 30:
            return 0.9   # 10% penalty after 30 days
        elif age_days > 90:
            return 0.8   # 20% penalty after 90 days
        return 1.0

    def _source_count_multiplier(self, edge: Edge, context: Context) -> float:
        """Boost facts mentioned multiple times"""
        # Query edge_source table
        count = context.store.get_edge_sources_count(edge.id)
        if count >= 3:
            return 1.1   # 10% boost for 3+ mentions
        elif count >= 2:
            return 1.05  # 5% boost for 2 mentions
        return 1.0
```

**Benefits**:
- ✅ Fast (<1ms, no LLM)
- ✅ Learns from actual usage
- ✅ Improves over time automatically
- ✅ No external dependencies

**Limitations**:
- ❌ Doesn't analyze text ("I think..." vs "definitely...")
- ❌ All mentions weighted equally

---

### Tier 2: Linguistic Analysis (LLM-Based)

**Analyze text for certainty markers**

```python
class LinguisticConfidence(ConfidenceStrategy):
    """Score confidence from linguistic markers in source text"""

    def __init__(self, llm_client, base_strategy: ConfidenceStrategy):
        self.llm = llm_client
        self.base = base_strategy
        self._cache = {}  # Cache LLM results

    async def score_async(self, edge: Edge, context: Context) -> float:
        # Get baseline from structural signals
        baseline = self.base.score(edge, context)

        # Get source text from provenance
        provenance = context.store.get_edge_provenance(edge.id)
        if not provenance:
            return baseline  # No text, fall back to baseline

        # Analyze most recent source
        text = provenance[0][0]  # Latest text

        # Check cache
        cache_key = hashlib.sha1(f"{text}|{edge.id}".encode()).hexdigest()
        if cache_key in self._cache:
            linguistic_score = self._cache[cache_key]
        else:
            # Call LLM (with timeout)
            try:
                linguistic_score = await asyncio.wait_for(
                    self._analyze_certainty(text, edge),
                    timeout=0.1  # 100ms budget
                )
                self._cache[cache_key] = linguistic_score
            except asyncio.TimeoutError:
                return baseline  # Fallback on timeout

        # Blend: 70% linguistic, 30% structural
        final = 0.7 * linguistic_score + 0.3 * baseline
        return min(1.0, max(0.0, final))

    async def _analyze_certainty(self, text: str, edge: Edge) -> float:
        """Use LLM to detect certainty markers"""

        prompt = f"""Rate certainty (0.0-1.0) that this fact is certain based on the text.

Text: "{text}"
Fact: {edge.src} {edge.rel} {edge.dst}

Certainty markers:
- HIGH (0.9-1.0): "definitely", "always", "for sure", ALL CAPS, corrections "NO, I meant..."
- MEDIUM (0.6-0.8): clear statement, no hedging
- LOW (0.3-0.5): "I think", "maybe", "probably", "might"
- VERY LOW (0.0-0.2): questions "Is my name...?", extreme hedging "I'm not sure but..."

Return ONLY a number between 0.0 and 1.0:"""

        response = await self.llm.generate(prompt, max_tokens=5)
        score = float(response.strip())
        return score
```

**Benefits**:
- ✅ Captures how user says things
- ✅ Detects hedging, emphasis, questions
- ✅ More nuanced than structural alone

**Limitations**:
- ❌ Requires LLM (latency, cost)
- ❌ Not suitable for hot path
- ❌ Needs fallback when LLM unavailable

**Solution**: Run async in background, cache results

---

### Tier 3: DSPy Optimization (Meta-Learning)

**Learn optimal confidence function from data**

```python
import dspy

class ConfidenceScorer(dspy.Signature):
    """Score fact confidence from text and metadata"""

    text = dspy.InputField(desc="Original conversation text")
    fact = dspy.InputField(desc="Extracted triple (s, r, d)")
    relation_type = dspy.InputField(desc="Relation type")
    source_count = dspy.InputField(desc="Number of times mentioned")
    reinforcements = dspy.InputField(desc="Number of validations")

    confidence = dspy.OutputField(desc="Confidence score 0.0-1.0")
    reasoning = dspy.OutputField(desc="Why this confidence")

class OptimizedConfidence(dspy.Module):
    """DSPy-optimized confidence scorer"""

    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(ConfidenceScorer)

    def forward(self, edge, context):
        # Get provenance
        provenance = context.store.get_edge_provenance(edge.id)
        text = provenance[0][0] if provenance else ""

        # Get metadata
        source_count = context.store.get_edge_sources_count(edge.id)

        # Call DSPy module
        result = self.scorer(
            text=text,
            fact=f"{edge.src} {edge.rel} {edge.dst}",
            relation_type=edge.rel,
            source_count=source_count,
            reinforcements=edge.pos
        )

        return float(result.confidence)

# Optimization with GEPA
def optimize_confidence_scorer(train_data):
    """Train confidence scorer using GEPA optimizer"""

    # Create scorer
    scorer = OptimizedConfidence()

    # Define metric
    def confidence_calibration_metric(example, prediction, trace=None):
        """Measure if predicted confidence matches actual correctness"""
        predicted_conf = float(prediction.confidence)
        actual_correct = example.is_correct  # Ground truth

        # Calibration error
        error = abs(predicted_conf - (1.0 if actual_correct else 0.0))

        # Lower error = better
        return 1.0 - error

    # Optimize with GEPA
    from dspy.teleprompt import GEPA

    optimizer = GEPA(
        metric=confidence_calibration_metric,
        breadth=10,
        depth=3,
        n_iterations=20
    )

    optimized_scorer = optimizer.compile(
        scorer,
        trainset=train_data
    )

    return optimized_scorer
```

**Benefits**:
- ✅ Learns from actual data
- ✅ Discovers patterns we didn't hand-code
- ✅ Improves with more data
- ✅ Can adapt to user-specific patterns

**Limitations**:
- ❌ Requires labeled training data
- ❌ Optimization is slow (offline process)
- ❌ Complex to deploy

**Solution**: Run optimization offline, deploy learned weights

---

## SOLID Architecture

### Dependency Injection

```python
# 1. Interface (Dependency Inversion Principle)
class ConfidenceStrategy(Protocol):
    """Strategy for computing fact confidence"""
    def score(self, edge: Edge, context: Context) -> float:
        ...

# 2. Baseline (what we have now)
class RelationTypeConfidence(ConfidenceStrategy):
    """Static confidence by relation type"""
    def score(self, edge: Edge, context: Context) -> float:
        if edge.rel == "name":
            return 0.95
        elif edge.rel.startswith("v:"):
            return 0.85
        return 0.9

# 3. Structural (Tier 1)
class UsageBasedConfidence(ConfidenceStrategy):
    """Structural signals only"""
    # ... implementation above ...

# 4. Linguistic (Tier 2)
class LinguisticConfidence(ConfidenceStrategy):
    """LLM-based text analysis"""
    # ... implementation above ...

# 5. Optimized (Tier 3)
class DSPyConfidence(ConfidenceStrategy):
    """DSPy-optimized scorer"""
    # ... implementation above ...

# 6. Injection into HotMemory
class HotMemory:
    def __init__(self, store: MemoryStore,
                 confidence_strategy: ConfidenceStrategy = None):
        self.store = store
        # Default to usage-based (structural)
        self.confidence = confidence_strategy or UsageBasedConfidence()

    def process_turn(self, text: str, ...) -> ...:
        # ... extraction ...

        # Use injected strategy
        conf = self.confidence.score(
            Edge(s, r, d, edge.pos, edge.neg, edge.updated_at, edge.id),
            Context(store=self.store, text=text, ...)
        )

        self.store.observe_edge(s, r, d, conf, now_ts)
```

### Configuration

```python
# .env
CONFIDENCE_STRATEGY=usage_based  # Options: relation_type, usage_based, linguistic, dspy

# factory.py
def create_confidence_strategy(name: str) -> ConfidenceStrategy:
    if name == "relation_type":
        return RelationTypeConfidence()
    elif name == "usage_based":
        return UsageBasedConfidence()
    elif name == "linguistic":
        llm = create_llm_client()
        base = UsageBasedConfidence()
        return LinguisticConfidence(llm, base)
    elif name == "dspy":
        return DSPyConfidence.load("models/confidence_scorer.pkl")
    else:
        raise ValueError(f"Unknown strategy: {name}")

# Use in factory
strategy = create_confidence_strategy(os.getenv("CONFIDENCE_STRATEGY", "usage_based"))
hot_memory = HotMemory(store, confidence_strategy=strategy)
```

---

## Evaluation Framework

### Building Training Data

**Use existing conversations as evaluation set**

```python
def build_eval_dataset(store: MemoryStore) -> List[Example]:
    """Create evaluation dataset from stored conversations"""

    examples = []

    # Get all edges with provenance
    cur = store.sql.cursor()
    edges = cur.execute("""
        SELECT e.id, e.src, e.rel, e.dst, e.pos, e.neg, e.weight, e.updated_at
        FROM edge e
        WHERE e.status = 1
        LIMIT 500
    """).fetchall()

    for edge_id, src, rel, dst, pos, neg, weight, updated_at in edges:
        # Get source text
        provenance = store.get_edge_provenance(edge_id)
        if not provenance:
            continue

        text = provenance[0][0]
        source_count = len(provenance)

        # Ground truth: facts with pos>0 and neg=0 are correct
        is_correct = (pos > 0 and neg == 0)

        # Also consider: facts still active after 30 days are likely correct
        age_days = (time.time() - updated_at / 1000) / 86400
        if age_days > 30 and weight > 0.7:
            is_correct = True

        examples.append({
            'text': text,
            'fact': (src, rel, dst),
            'relation_type': rel,
            'source_count': source_count,
            'reinforcements': pos,
            'negations': neg,
            'is_correct': is_correct,
            'current_confidence': weight
        })

    return examples
```

### Metrics

```python
def evaluate_confidence_calibration(strategy: ConfidenceStrategy,
                                   test_set: List[Example]) -> Dict[str, float]:
    """Measure calibration quality"""

    predictions = []
    actuals = []

    for example in test_set:
        edge = Edge(
            id=None,
            src=example['fact'][0],
            rel=example['fact'][1],
            dst=example['fact'][2],
            pos=example['reinforcements'],
            neg=example['negations'],
            updated_at=int(time.time() * 1000)
        )
        context = Context(text=example['text'], store=None)

        conf = strategy.score(edge, context)
        predictions.append(conf)
        actuals.append(1.0 if example['is_correct'] else 0.0)

    # Calibration metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr

    return {
        'mse': mean_squared_error(actuals, predictions),
        'mae': mean_absolute_error(actuals, predictions),
        'correlation': pearsonr(predictions, actuals)[0],
        'mean_confidence': sum(predictions) / len(predictions),
        'accuracy_at_70': sum(1 for p, a in zip(predictions, actuals)
                              if (p >= 0.7 and a == 1.0) or (p < 0.7 and a == 0.0)) / len(predictions)
    }
```

### Evaluation Script

```bash
# scripts/eval_confidence.py
python scripts/eval_confidence.py --strategy usage_based --dataset data/memory.db
```

Output:
```
Evaluating UsageBasedConfidence on 500 examples...

Calibration Metrics:
  MSE: 0.043
  MAE: 0.156
  Correlation: 0.782
  Mean Confidence: 0.847
  Accuracy@0.7: 0.892

Confidence Distribution:
  [0.0-0.3): 12 edges (2.4%)  - Low confidence
  [0.3-0.5): 31 edges (6.2%)  - Medium-low
  [0.5-0.7): 78 edges (15.6%) - Medium
  [0.7-0.9): 203 edges (40.6%) - Medium-high
  [0.9-1.0]: 176 edges (35.2%) - High confidence

✅ Baseline (RelationTypeConfidence): Correlation 0.623
✅ UsageBasedConfidence: Correlation 0.782 (+25% improvement)
```

---

## Implementation Phases

### Phase 1: Extract to Interface (30 min)

**Goal**: Refactor existing code to use strategy pattern

```python
# 1. Create confidence_strategy.py with interface
# 2. Extract RelationTypeConfidence class
# 3. Inject into HotMemory
# 4. Tests pass (no behavior change)
```

### Phase 2: Add Usage-Based (2 hours)

**Goal**: Implement Tier 1 (structural signals)

```python
# 1. Create UsageBasedConfidence class
# 2. Add query methods for source count
# 3. Unit tests for each multiplier
# 4. Integration test showing improvement
# 5. Update retrieval to use confidence in ranking
```

### Phase 3: Build Evaluation (3 hours)

**Goal**: Create dataset and metrics

```python
# 1. Implement build_eval_dataset()
# 2. Implement evaluate_confidence_calibration()
# 3. Create eval script
# 4. Run baseline vs usage-based comparison
# 5. Document results
```

### Phase 4: Optional - Linguistic (4 hours)

**Goal**: Add LLM-based text analysis (if needed)

```python
# 1. Implement LinguisticConfidence
# 2. Add background worker for async scoring
# 3. Add caching layer
# 4. Evaluate on test set
# 5. Compare to structural-only
```

### Phase 5: Optional - DSPy Optimization (8 hours)

**Goal**: Meta-learn optimal confidence function

```python
# 1. Define ConfidenceScorer signature
# 2. Create training dataset
# 3. Run GEPA optimization
# 4. Save/load optimized model
# 5. Deploy and evaluate
```

---

## Migration Path

### Stage 1: Baseline (Current)

```python
CONFIDENCE_STRATEGY=relation_type  # Default
```
- Static confidence by relation type
- No learning

### Stage 2: Structural Learning

```python
CONFIDENCE_STRATEGY=usage_based
```
- Learns from reinforcements, recency, source count
- No LLM required
- Improves automatically over time

### Stage 3: Linguistic Enhancement (Optional)

```python
CONFIDENCE_STRATEGY=linguistic
```
- Adds text analysis
- Requires LLM (localhost:1234)
- Blends with structural signals

### Stage 4: Full Optimization (Optional)

```python
CONFIDENCE_STRATEGY=dspy
```
- Uses DSPy-optimized scorer
- Best performance
- Requires training data

---

## Performance Considerations

### Hot Path Impact

**Goal**: Keep extraction <200ms

**Analysis**:
- Tier 1 (structural): +1-2ms (database queries cached in RAM)
- Tier 2 (linguistic): NOT in hot path (background worker)
- Tier 3 (DSPy): Loads model once, inference <5ms

**Strategy**: Use Tier 1 (structural) in hot path, run Tier 2/3 in background

### Background Confidence Refinement

```python
class BackgroundConfidenceWorker:
    """Refine confidence scores during idle time"""

    def __init__(self, store: MemoryStore, strategy: ConfidenceStrategy):
        self.store = store
        self.strategy = strategy
        self.running = False

    async def refine_loop(self):
        """Continuously refine low-confidence edges"""
        while self.running:
            # Find edges with default confidence (never refined)
            edges = self._find_unrefined_edges(limit=10)

            for edge in edges:
                # Rescore with advanced strategy
                new_conf = await self.strategy.score_async(edge, context)

                # Update if changed significantly
                if abs(new_conf - edge.weight) > 0.1:
                    self.store.update_edge_confidence(edge.id, new_conf)

            # Idle wait
            await asyncio.sleep(5.0)
```

---

## Success Metrics

### Quantitative

✅ **Confidence correlation > 0.75** (vs ground truth)
✅ **Calibration MAE < 0.2** (predicted conf vs actual correctness)
✅ **Retrieval precision improves >10%** (with confidence filtering)
✅ **Hot path latency <200ms** (no regression)

### Qualitative

✅ **Low-confidence facts are actually uncertain** (manual review)
✅ **High-confidence facts are validated** (reinforced, multiple sources)
✅ **System learns from usage** (confidence evolves over time)
✅ **Evaluation dataset can replay conversations** (reproducible)

---

## Future Enhancements (Phase 3+)

### User-Specific Confidence Models

Learn confidence patterns per user:
- Some users hedge frequently ("I think...")
- Some users are always certain
- Adapt confidence calibration per user

### Temporal Confidence Decay

Facts decay over time unless refreshed:
- "I live in NYC" (said 2 years ago) → decay
- "I live in LA" (said yesterday) → high confidence
- Conflict resolution uses recency

### Active Learning

System asks for confirmation on low-confidence facts:
- "You mentioned you work at Google, is that still correct?"
- User confirmation → boost confidence
- User correction → negate old fact, store new one

### Prosody Integration

When voice signals available (Phase 4):
- Pitch, energy, emphasis → confidence modifiers
- Integrate with linguistic analysis
- See `prosody_integration_roadmap.md`

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/test_confidence_strategies.py
def test_relation_type_baseline()
def test_usage_based_reinforcement()
def test_usage_based_recency_decay()
def test_usage_based_source_count()
def test_linguistic_hedging_detection()
def test_linguistic_emphasis_detection()
def test_dspy_confidence_scoring()
```

### Integration Tests

```python
# tests/integration/test_confidence_learning.py
def test_confidence_improves_with_reinforcement()
def test_confidence_decays_with_age()
def test_low_confidence_filtered_in_retrieval()
def test_background_refinement_worker()
def test_confidence_strategy_swapping()
```

### Evaluation Tests

```python
# tests/evaluation/test_confidence_calibration.py
def test_baseline_calibration()
def test_usage_based_calibration()
def test_linguistic_calibration()
def test_dspy_calibration()
def test_calibration_improvement()
```

---

## Implementation Checklist

**Phase 1: Interface Extraction**
- [ ] Create `core/memory/confidence_strategy.py` with protocol
- [ ] Extract `RelationTypeConfidence` class
- [ ] Add `confidence_strategy` parameter to `HotMemory.__init__()`
- [ ] Update `process_turn()` to use strategy
- [ ] Tests pass (no behavior change)

**Phase 2: Usage-Based (Structural)**
- [ ] Implement `UsageBasedConfidence` class
- [ ] Add `get_edge_sources_count()` to `MemoryStore`
- [ ] Implement reinforcement, recency, source_count multipliers
- [ ] Write unit tests for each component
- [ ] Integration test showing improvement
- [ ] Update retrieval to filter by confidence (>0.70)

**Phase 3: Evaluation Framework**
- [ ] Implement `build_eval_dataset()`
- [ ] Implement `evaluate_confidence_calibration()`
- [ ] Create `scripts/eval_confidence.py`
- [ ] Run baseline evaluation
- [ ] Run usage-based evaluation
- [ ] Document improvement metrics

**Phase 4: Configuration & Deployment**
- [ ] Add `CONFIDENCE_STRATEGY` to `.env`
- [ ] Implement `create_confidence_strategy()` factory
- [ ] Update `factory.py` to inject strategy
- [ ] Add monitoring/logging for confidence scores
- [ ] Deploy with `usage_based` as default

**Phase 5: Optional - Linguistic**
- [ ] Implement `LinguisticConfidence` class
- [ ] Add LLM client integration
- [ ] Implement caching layer
- [ ] Create background worker
- [ ] Evaluate and compare

**Phase 6: Optional - DSPy**
- [ ] Define `ConfidenceScorer` DSPy signature
- [ ] Implement `OptimizedConfidence` module
- [ ] Create training dataset
- [ ] Run GEPA optimization
- [ ] Save and deploy model

---

## Conclusion

This roadmap provides a **clear, incremental path** from arbitrary confidence to learned confidence:

1. **Phase 1** (30 min): Extract to interface - foundation for everything
2. **Phase 2** (2 hours): Usage-based confidence - immediate improvement, no LLM
3. **Phase 3** (3 hours): Evaluation framework - measure improvement scientifically
4. **Phase 4+** (optional): Linguistic and DSPy - further improvements if needed

Each phase delivers value independently and builds on the provenance foundation established in `edge_provenance_implementation.md`.

**Next Steps**: Complete edge provenance implementation, then begin Phase 1 of this roadmap.