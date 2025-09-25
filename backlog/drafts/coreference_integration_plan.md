# Coreference Resolution Integration Plan

## Overview
Integrate `spacy-coref` library to enhance memory extraction accuracy from 70-85% to 85-95% while maintaining <200ms hot-path latency.

## Phase 1: Preparation (Read-Only Analysis)

### 1.1 Dependency Analysis
- **Library**: `spacy-coref` (talmago/spacy-coref)
- **Model**: `talmago/allennlp-coref-onnx-mMiniLMv2-L12-H384-distilled-from-XLMR-Large`
- **Requirements**: ONNX Runtime, spaCy integration
- **Size**: ~50MB distilled model (much smaller than full transformer)

### 1.2 Current Architecture Review
**Hot Path Flow** (must stay <200ms):
```
Input Text → spaCy NLP → UD Extraction → Triple Refinement → Memory Store → Retrieval → Bullets
```

**Integration Point**: Insert coreference resolution between spaCy NLP and UD Extraction:
```
Input Text → spaCy NLP → [COREFERENCE RESOLUTION] → UD Extraction → ...
```

### 1.3 Latency Budget Analysis
- **Current hot path**: <200ms p95
- **spaCy coref estimate**: 10-30ms (ONNX inference)
- **Safety margin**: 50ms max coref latency
- **Fallback strategy**: Skip coref if >50ms, use current system

## Phase 2: Implementation Plan

### 2.1 Component Architecture
```python
class CoreferenceResolver:
    def __init__(self, model_name: str, timeout_ms: int = 50):
        self.resolver = CoreferenceResolver.from_pretrained(model_name)
        self.timeout_ms = timeout_ms
        self.metrics = defaultdict(list)

    def resolve_text(self, doc: spacy.Doc) -> str:
        """Resolve coreferences in spaCy doc, return resolved text"""
        start = time.perf_counter()

        try:
            # Get coreference clusters
            clusters = doc._.coref_clusters

            # Apply resolution with timeout
            resolved = self._apply_resolution(doc, clusters)

            latency = (time.perf_counter() - start) * 1000
            self.metrics['latency_ms'].append(latency)

            if latency > self.timeout_ms:
                logger.warning(f"Coref timeout: {latency:.1f}ms")
                return doc.text  # Fallback

            return resolved

        except Exception as e:
            logger.warning(f"Coref failed: {e}")
            return doc.text  # Fallback
```

### 2.2 Integration Points

**File: `server/core/memory/hotpath_processor.py`**
- Add coreference resolver initialization
- Modify `process_turn()` to include coref step
- Add latency monitoring

**File: `server/core/memory/extractors/ud.py`**
- Update `extract()` method to accept pre-resolved text
- Maintain backward compatibility

**File: `server/pyproject.toml`**
- Add `spacy-coref` dependency
- Version constraints for compatibility

### 2.3 Configuration Changes

**File: `server/config/settings.py`**
```python
@dataclass
class MemoryConfig:
    enable_coreference: bool = True
    coref_model: str = "talmago/allennlp-coref-onnx-mMiniLMv2-L12-H384-distilled-from-XLMR-Large"
    coref_timeout_ms: int = 50
    coref_min_text_length: int = 10  # Only run on longer texts
```

### 2.4 Test Cases for Validation

**File: `server/tests/unit/test_coreference_integration.py`**
```python
test_cases = [
    {
        "input": "John went to the store. He bought milk.",
        "expected_entities": ["john", "store", "milk"],
        "expected_relations": [("john", "went_to", "store"), ("john", "bought", "milk")]
    },
    {
        "input": "My wife Sarah and I went shopping. She bought a dress.",
        "expected_entities": ["you", "sarah", "dress"],
        "expected_relations": [("you", "has", "sarah"), ("sarah", "bought", "dress")]
    }
]
```

## Phase 3: Rollout Strategy

### 3.1 Gradual Rollout
- **Week 1**: 10% of conversations with coreference
- **Week 2**: 25% with A/B accuracy monitoring
- **Week 3**: 50% if accuracy improves >5%
- **Week 4**: 100% if latency stays <200ms p95

### 3.2 Monitoring & Metrics
- **Accuracy**: Triple extraction precision/recall
- **Latency**: Hot path p95, coref component timing
- **Errors**: Coref failure rate, fallback usage
- **User Impact**: Conversation coherence scores

### 3.3 Rollback Plan
- Feature flag to disable coreference instantly
- Automatic rollback if latency >220ms p95 for 5 minutes
- Alert thresholds: accuracy drop >10%, error rate >5%

## Phase 4: Expected Outcomes

### 4.1 Performance Improvements
- **Accuracy**: 70-85% → 85-95% (15% improvement)
- **Entity Linking**: Better cross-sentence relationships
- **Memory Quality**: More coherent conversation history

### 4.2 Latency Impact
- **Best Case**: +10-20ms (still <200ms total)
- **Worst Case**: +50ms with fallbacks (acceptable)
- **Optimization**: Model warm-up, caching, selective activation

### 4.3 Risk Mitigation
- **Fallback System**: Current extraction always available
- **Timeout Protection**: Hard 50ms limit per text
- **Model Resilience**: Graceful degradation on failures

## Implementation Checklist

### Pre-Implementation
- [ ] Analyze current test suite coverage
- [ ] Benchmark current latency baselines
- [ ] Review spaCy version compatibility
- [ ] Plan model download/caching strategy

### Core Implementation
- [ ] Add spacy-coref dependency
- [ ] Create CoreferenceResolver component
- [ ] Integrate into hot path pipeline
- [ ] Add configuration options
- [ ] Implement timeout/fallback logic

### Testing & Validation
- [ ] Unit tests for coreference resolver
- [ ] Integration tests with memory system
- [ ] Performance benchmarks (latency, accuracy)
- [ ] A/B test framework setup

### Deployment & Monitoring
- [ ] Feature flag implementation
- [ ] Gradual rollout plan
- [ ] Monitoring dashboards
- [ ] Alert configuration
- [ ] Rollback procedures

This plan provides a safe, incremental approach to adding coreference resolution while preserving the critical <200ms latency requirement. The `spacy-coref` library's lightweight design makes this feasible within the existing architecture.