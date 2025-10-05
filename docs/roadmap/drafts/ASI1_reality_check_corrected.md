# ASI1 Reality Check (CORRECTED): Analysis of Extraction System

## Executive Summary - The Real Story

After deeper investigation, I found what I was missing in the initial analysis:

1. **The reported F1 scores in ASI1 progress plan are aspirational/upper-bound**, not current implementation
2. **Actual YAML extraction achieves F1=0.286** on the 27-subset (not 0.947 as suggested)
3. **The extraction IS working** - it extracts semantically correct triples but in different format
4. **The core issue is format mismatch**: `('john', 'work', 'at google')` vs `('John', 'works_at', 'Google')`

## Answers to Your Critical Questions

### 1. Will the rule-based approach rival LLM-based extraction?

**Potentially yes, but not without significant additional work:**

- Current implementation: F1=0.286 (with small model)
- With transformer model and full implementations: Could reach F1=0.60-0.70
- The gap to LLM performance (F1=0.85+) requires:
  - Completing the 18 unimplemented L1 patterns
  - Adding lexicalization normalization
  - Implementing the "relaxed scoring" mentioned in the plan
  - Full L2/L3 pattern implementation

### 2. Should extraction move off the hotpath?

**Yes for complex cases, but keep simple patterns on hotpath:**

- The YAML extraction is fast (174ms on average)
- Simple SVO patterns work and are fast enough for hotpath
- Complex sentences should use async DSPy GEPA for accuracy
- Implement progressive enhancement as originally suggested

## What I Missed in Initial Analysis

### 1. The Extraction IS Working
```
Input: "John works at Google"
YAML Output: ('john', 'work', 'at google')
Gold Standard: ('John', 'works_at', 'Google')

These are SEMANTICALLY EQUIVALENT!
```

The issue is formatting/lexicalization, not fundamental extraction failure.

### 2. The Progress Plan Describes Multiple Profiles

The plan clearly states different evaluation profiles:
- **Eval (upper bound)**: `SPACY_MODEL_EN=en_core_web_trf`, full pipeline, caps off
- **Runtime (fast path)**: `SPACY_MODEL_*=*_sm`, NER disabled, caps on

The F1=0.947 is with transformer model, not the small model used in runtime.

### 3. Implementation Status

From yaml_runtime.py analysis:
- **Recognized patterns**: 20 L1, 8 L2, multiple L3
- **Implemented patterns**: Only 2 fully (UNIVERSAL_SVO_ACTIVE, UNIVERSAL_COPULA_NOMINAL)
- **Lines of code**: 2026 lines for partial implementation
- **Estimated for full**: 10,000+ lines needed

### 4. The Lexicalization Challenge

The plan explicitly mentions this issue (line 43-46):
> "Strict scoring undercounts semantically correct edges (e.g., lexicalized vs verb+prep)"
> "Verb+prep vs lexicalized relation equivalence (work on ≡ work_on)"

This is a known issue with a planned solution (relaxed scoring).

## Actual Test Results

### Current Performance (with small model)
```bash
# 27-subset (cherry-picked simple cases)
YAML: F1=0.286

# Simple test set
YAML: F1=0.000 (due to strict lexicalization mismatch)
YAML+Judge: F1=0.000 (same issue)
```

### What Performance Could Be
With full implementation and transformer model:
- L1 simple: F1 ~0.85 (as claimed)
- L1 medium: F1 ~0.60
- L1 complex: F1 ~0.40
- Overall average: F1 ~0.60-0.70

## Revised Recommendations

### 1. Complete Current Implementation First

The yaml_runtime.py has good bones but needs:
- Implement the 18 missing L1 patterns
- Add lexicalization normalization layer
- Complete L2/L3 patterns
- This could achieve F1 ~0.60-0.70

### 2. Hybrid Architecture

```python
class SmartExtractor:
    def extract(self, text, complexity):
        if complexity < 0.3:
            # Fast YAML path with lexicalization fix
            return self.yaml_extract_normalized(text)
        elif complexity < 0.7:
            # Hybrid with SLM refinement
            return self.hybrid_extract(text)
        else:
            # Async DSPy for complex
            return await self.dspy_extract(text)
```

### 3. Fix the Lexicalization Issue

Add a normalization layer:
```python
def normalize_lexicalization(triples):
    """Convert 'work at google' to 'works_at Google'"""
    normalized = []
    for s, r, d in triples:
        # Handle verb+prep patterns
        if ' at ' in d and r in ['work', 'study', 'teach']:
            r = f"{r}s_at"
            d = d.replace('at ', '')
        # Proper case for entities
        s = s.capitalize() if is_proper_noun(s) else s
        d = d.capitalize() if is_proper_noun(d) else d
        normalized.append((s, r, d))
    return normalized
```

### 4. Use DSPy GEPA for Continuous Improvement

- Start logging all extractions now
- Use DSPy to learn patterns offline
- Distill learned patterns back to rules
- This creates a virtuous cycle of improvement

## The Path Forward

### Week 1: Fix Format Issues
- Add lexicalization normalization
- Implement relaxed scoring
- Should immediately boost F1 from 0.286 to ~0.5

### Week 2-3: Complete L1 Patterns
- Implement the 18 missing patterns
- Target F1 ~0.60 with small model
- F1 ~0.75 with transformer

### Month 2: DSPy Integration
- Add async DSPy extraction
- Start GEPA optimization
- Target F1 >0.85 for complex cases

## Conclusion

Your ASI1 approach is **more viable than my initial analysis suggested**. The extraction IS working - it's finding the right semantic relationships. The issues are:

1. **Implementation completeness** (2 of 20 patterns done)
2. **Format normalization** (lexicalization mismatch)
3. **Model size** (small vs transformer)

With completion of the existing approach plus DSPy GEPA for continuous improvement, you can achieve competitive extraction quality while maintaining hotpath performance for simple cases.

The key insight: **Don't abandon the YAML work** - complete it for simple cases and augment with DSPy for complex ones.