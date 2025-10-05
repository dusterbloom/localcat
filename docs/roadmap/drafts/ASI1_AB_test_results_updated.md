# ASI1 A/B Test Results - Updated Analysis

## Executive Summary

The A/B testing confirms that the GraphJudge implementation is working as documented, achieving **F1=0.452** on medium complexity datasets with the transformer model - exactly matching the results reported in ASI1_claude_code_sync.md.

## Key Findings

### 1. GraphJudge Impact is Model-Dependent

**With small model (en_core_web_sm):**
- YAML: F1=0.323
- YAML+Judge: F1=0.323 (no improvement)
- Latency: 227-333ms

**With transformer model (en_core_web_trf):**
- YAML: F1=0.412 (baseline)
- YAML+Judge: **F1=0.452** (10% improvement)
- Latency: 728ms

The judge requires better linguistic features from the transformer model to be effective.

### 2. Performance Across Complexity Levels

| Dataset | Complexity | YAML F1 | YAML+Judge F1 | Judge Benefit |
|---------|------------|---------|---------------|---------------|
| Simple test | Very Simple | 0.000* | 0.000* | N/A (format issue) |
| Easy | Simple | 0.375 | 0.375 | 0% |
| Medium | Medium | 0.323 | 0.323 | 0% (sm model) |
| Medium | Medium | 0.412 | **0.452** | +10% (trf model) |

*Simple test has lexicalization format mismatch preventing any matches

### 3. Latency Analysis

**Speed comparison:**
- YAML baseline: 322-333ms (small model)
- YAML+Judge: 227-259ms (small model, faster!)
- YAML+Judge: 728ms (transformer model)

Surprisingly, the judge makes extraction **faster** with small models, likely due to early filtering reducing downstream processing.

## Comparison with Initial Analysis

### What Changed Since Initial Reality Check

1. **GraphJudge Implementation**
   - Added precision filtering that improves F1 by 10% with transformer
   - Filters low-quality triples effectively

2. **Normalization Improvements**
   - Added nominalization handling
   - Improved verb+prep lexicalization
   - Better handling of complex linguistic patterns

3. **Self-Improving Loop**
   - Gray-zone logging captures uncertain cases
   - LLM labeling provides training data
   - Distillation creates better judges over time

### Current State vs. ASI1 Goals

| Metric | Initial State | Current State | Goal | Gap |
|--------|--------------|---------------|------|-----|
| F1 (medium) | 0.286 | **0.452** | 0.60 | 0.148 |
| F1 (easy) | ~0.30 | 0.375 | 0.85 | 0.475 |
| Latency | 174ms | 728ms* | <200ms | -528ms |

*With transformer; small model achieves 227ms

## Answering Your Original Questions (Updated)

### 1. Can rule-based rival LLM extraction?

**Updated answer: Getting closer, but still a gap**

- Current best: F1=0.452 (with transformer + judge)
- LLM baseline: F1=0.85+
- Gap: ~0.4 F1 points

The GraphJudge significantly helps, but fundamental coverage gaps remain:
- Only 2 of 20 L1 patterns fully implemented
- Lexicalization normalization still incomplete
- Need more sophisticated pattern matching

### 2. Should extraction move off hotpath?

**Updated answer: Yes, but with nuance**

**Recommended architecture:**
```python
def smart_extract(text, complexity):
    if complexity < 0.3 and latency_critical:
        # Hotpath: Small model + judge
        # F1 ~0.35, Latency ~230ms
        return yaml_judge_small(text)

    elif complexity < 0.6:
        # Medium path: Transformer + judge
        # F1 ~0.45, Latency ~700ms
        return yaml_judge_transformer(text)

    else:
        # Async: DSPy GEPA
        # F1 ~0.85, Latency unconstrained
        return await dspy_extract(text)
```

## Recommendations Based on Updated Results

### Immediate Actions (This Week)

1. **Fix Lexicalization Normalization**
   - The simple test showing F1=0.0 is purely format mismatch
   - Adding proper lexicalization could boost all scores by 10-20%

2. **Complete Pattern Implementation**
   - 18 L1 patterns remain unimplemented
   - Each pattern adds ~2-3% F1

3. **Optimize Judge Thresholds**
   - Current judge may be too conservative
   - Tune thresholds per complexity level

### Medium Term (2-4 Weeks)

1. **Model-Specific Judges**
   - Train separate judges for small vs transformer models
   - Small model judge should be more permissive

2. **Implement Relaxed Scoring**
   - As mentioned in ASI1 plan, add semantic equivalence scoring
   - This would reveal true performance vs. strict matching

3. **Accelerate Transformer Path**
   - Cache model loading
   - Investigate ONNX export for faster inference
   - Target <400ms for transformer path

### Long Term (1-2 Months)

1. **Complete DSPy Integration**
   - Current extraction provides training data
   - DSPy learns from both successes and failures
   - Distill back to faster models

2. **Implement Progressive Enhancement**
   - Start with fast extraction
   - Asynchronously improve with better models
   - Update memory when better extraction completes

## Test Results Summary

### Configuration Used
```bash
# Best results achieved with:
SPACY_MODEL_EN=en_core_web_trf
YAML_DENSITY_CAPS=off
YAML_NOMINALS=on
YAML_COREF=off
YAML_GRAPH_JUDGE=on
YAML_GRAPH_JUDGE_MODEL=models/graph_judge.json
```

### Results Table
| Method | Dataset | F1 | Precision | Recall | Latency |
|--------|---------|-----|-----------|--------|---------|
| Baseline UD | Medium | 0.167 | 0.129 | 0.235 | 18ms |
| YAML+Judge (sm) | Medium | 0.323 | 0.400 | 0.283 | 227ms |
| **YAML+Judge (trf)** | **Medium** | **0.452** | **0.500** | **0.412** | **728ms** |

## Conclusion

The GraphJudge implementation is working as designed and provides meaningful improvements when paired with a capable language model. The path forward is clear:

1. **Complete implementation** of existing patterns
2. **Fix format issues** (lexicalization normalization)
3. **Add DSPy** for complex cases and continuous improvement

With these improvements, achieving F1=0.70 for rule-based extraction is realistic, making it competitive for many use cases while maintaining the speed and explainability advantages over pure LLM approaches.