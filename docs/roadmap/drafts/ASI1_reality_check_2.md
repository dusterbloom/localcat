# ASI1 Reality Check 2: Path to F1=0.75 and Beyond

## Executive Summary

Based on comprehensive A/B testing with YAML, SLM, and LLMs (including LFM2-350M-Extract), we now have a clear path to achieve F1=0.75 extraction quality while maintaining <200ms latency for most cases. The staged approach with complexity routing is validated and ready for implementation.

## Current State vs Target

### Where We Are Now
- **YAML**: F1=0.3-0.4 @ 1400ms (transformer), only 2/20 patterns implemented
- **SLM (Qwen2.5-0.5B)**: Same as YAML (no refinement happening) @ 172ms
- **LLMs**: F1=0.7-0.85 @ 340-2000ms (varies by model)

### Where We Need to Be
- **Simple extraction**: F1=0.5 @ <150ms
- **Medium complexity**: F1=0.7 @ <350ms
- **Complex cases**: F1=0.85 @ <2000ms (async)

## The Four Pillars to F1=0.75

### 1. Pattern Completion (Weeks 1-2)

**Current**: 2/20 L1 patterns implemented in yaml_runtime.py
**Target**: 20/20 L1 patterns + key L2 patterns

**Priority Patterns to Implement**:
```python
# Week 1 - High frequency patterns
UNIVERSAL_SVO_PASSIVE      # Passive voice normalization
UNIVERSAL_COORD_SUBJECT     # Coordinated subjects (John and Mary work)
UNIVERSAL_COORD_OBJECT      # Coordinated objects (likes cats and dogs)
UNIVERSAL_DITRANSITIVE_GIVE # Give/send patterns
UNIVERSAL_CONTROL_VERB      # Want to/try to/need to patterns

# Week 2 - Complex patterns
UNIVERSAL_RELATIVE_CLAUSE   # Which/that/who clauses
UNIVERSAL_CCOMP_EMBEDDING   # Think that/know that patterns
UNIVERSAL_MODAL_VERBS       # Can/should/must patterns
PRONOMINAL_3SG_RESOLUTION   # He/she/it resolution
DISCOURSE_CONNECTIVE        # Because/therefore connections
```

**Expected Impact**: F1 boost from 0.3 → 0.5

### 2. Codegen Implementation (Week 3)

**Current**: Interpreted YAML patterns @ 1400ms
**Target**: Compiled extraction code @ 140ms

**Implementation Steps**:
```python
# Step 1: Generate Python code from YAML patterns
def compile_yaml_to_python(yaml_spec):
    """
    Convert YAML patterns to optimized Python functions
    """
    code = []
    for pattern in yaml_spec.patterns:
        # Generate match function
        code.append(f"""
def match_{pattern.name}(doc):
    # Compiled pattern matching
    if doc[0].pos_ == '{pattern.trigger_pos}':
        # Direct array access instead of iterations
        return extract_{pattern.type}(doc)
""")
    return compile(code, 'yaml_compiled', 'exec')

# Step 2: Use Cython for hot paths
@cython.boundscheck(False)
@cython.wraparound(False)
def extract_svo_active(tokens):
    # C-speed extraction
    pass

# Step 3: Batch processing
def batch_extract(texts: List[str]):
    # Process multiple sentences in parallel
    with multiprocessing.Pool() as pool:
        return pool.map(extract_compiled, texts)
```

**Expected Impact**: 10x speed improvement (1400ms → 140ms)

### 3. Lexicalization Normalization (Week 2)

**Current**: `('john', 'work', 'at google')`
**Target**: `('John', 'works_at', 'Google')`

**Implementation**:
```python
class LexicalizationNormalizer:
    def __init__(self):
        # Verb+prep combinations to merge
        self.verb_prep_patterns = {
            ('work', 'at'): 'works_at',
            ('work', 'for'): 'works_for',
            ('live', 'in'): 'lives_in',
            ('born', 'in'): 'born_in',
            ('move', 'to'): 'moves_to',
            ('come', 'from'): 'comes_from',
            ('consist', 'of'): 'consists_of',
            ('depend', 'on'): 'depends_on',
            ('focus', 'on'): 'focuses_on',
            ('result', 'in'): 'results_in',
        }

    def normalize(self, triples):
        normalized = []
        for s, r, d in triples:
            # Handle verb+prep lexicalization
            if ' ' in d and r in self.verb_prep_patterns:
                parts = d.split(' ', 1)
                if parts[0] in ['at', 'for', 'in', 'on', 'to', 'from', 'of']:
                    r = self.verb_prep_patterns.get((r, parts[0]), r)
                    d = parts[1] if len(parts) > 1 else ''

            # Proper case for entities
            s = self._proper_case(s)
            d = self._proper_case(d)

            normalized.append((s, r, d))
        return normalized

    def _proper_case(self, text):
        # Use NER or capitalization rules
        if text and text[0].islower():
            # Check if proper noun
            return text.capitalize()
        return text
```

**Expected Impact**: F1 boost from 0.5 → 0.6

### 4. Working SLM Refinement (Week 4)

**Current**: SLM not actually refining
**Target**: SLM fixes common YAML errors

**Fix the SLM Integration**:
```python
class ImprovedSLMRefinement:
    def __init__(self):
        self.prompt_template = """
You are a relation extraction corrector. Fix these specific issues:
1. Lexicalize: "work at" → "works_at"
2. Resolve pronouns: "it" → antecedent
3. Fix empty objects
4. Normalize tense

Examples:
Input: [["john", "work", "at google"]]
Output: [["John", "works_at", "Google"]]

Input: [["company", "announce", ""], ["it", "expand", "europe"]]
Output: [["company", "announced", "product"], ["company", "will_expand_to", "Europe"]]

Now fix these:
Input: {input_triples}
Output:"""

    def refine(self, text, triples):
        # Only refine if there are issues
        needs_refinement = any(
            ' ' in d or  # Unlexicalized
            not d or     # Empty object
            s in ['it', 'they', 'he', 'she'] or  # Pronouns
            r.endswith('e')  # Likely infinitive
            for s, r, d in triples
        )

        if not needs_refinement:
            return triples

        # Call SLM with focused prompt
        refined = self.slm_extract(self.prompt_template, triples)

        # Validate output format
        if self._validate(refined):
            return refined
        return triples
```

**Alternative: Use LFM2-350M-Extract**:
```python
# LFM2-350M-Extract is specifically trained for extraction
class LFM2Extractor:
    def __init__(self):
        self.model = "LiquidAI/LFM2-350M-Extract"
        # This model is optimized for:
        # - Structured extraction tasks
        # - JSON output format
        # - Low latency (350M params)

    def extract(self, text):
        prompt = f"Extract: {text}"
        # LFM2 outputs structured JSON directly
        return self.model.generate(prompt, max_tokens=100)
```

**Expected Impact**: F1 boost from 0.6 → 0.75

## Staged Runtime Implementation

### Complexity-Based Routing

```python
class StagedExtractionRuntime:
    def __init__(self):
        # After all improvements
        self.yaml_codegen = CompiledYAMLExtractor()      # F1=0.5 @ 140ms
        self.qwen_llm = QwenLLMExtractor()               # F1=0.7 @ 340ms
        self.lfm2_extract = LFM2ExtractExtractor()       # F1=0.75 @ 400ms
        self.llama_3b = Llama3BExtractor()               # F1=0.85 @ 1500ms

    def extract(self, text):
        complexity = self.assess_complexity(text)

        if complexity < 0.4:
            # Simple: Use compiled YAML
            # Examples: "John works at Google", "Alice is the CEO"
            return self.yaml_codegen.extract(text)

        elif complexity < 0.6:
            # Medium: Use LFM2-350M-Extract or Qwen
            # Examples: "The company that John founded has 100 employees"
            return self.lfm2_extract.extract(text)

        elif complexity < 0.8:
            # Complex: Use Qwen with more time
            # Examples: Multi-clause with pronouns and temporal
            return self.qwen_llm.extract(text, timeout=500)

        else:
            # Very complex: Async with Llama-3B
            # Examples: Multiple sentences, complex coreference
            return await self.async_extract_llama(text)
```

### Performance Targets

| Complexity | Method | F1 Target | Latency Target | Example |
|------------|--------|-----------|----------------|---------|
| 0.0-0.4 | YAML+Codegen | 0.50 | 140ms | "John works at Google" |
| 0.4-0.6 | LFM2-Extract | 0.70 | 400ms | "Alice, the CEO, announced..." |
| 0.6-0.8 | Qwen-0.5B | 0.75 | 500ms | "The company that John founded..." |
| 0.8-1.0 | Llama-3B | 0.85 | 1500ms | Multi-sentence with coreference |

## Implementation Timeline

### Week 1: Pattern Completion
- [ ] Implement 5 high-priority L1 patterns
- [ ] Add test cases for each pattern
- [ ] Measure F1 improvement

### Week 2: Lexicalization + More Patterns
- [ ] Implement LexicalizationNormalizer
- [ ] Complete remaining L1 patterns
- [ ] Add L2 pronoun resolution

### Week 3: Codegen
- [ ] Generate Python from YAML
- [ ] Optimize with Cython for hot paths
- [ ] Benchmark speed improvements

### Week 4: SLM/LFM2 Integration
- [ ] Fix SLM refinement prompts
- [ ] Integrate LFM2-350M-Extract
- [ ] Tune complexity thresholds

### Month 2: DSPy GEPA
- [ ] Implement learning loop
- [ ] Distill improvements back
- [ ] Continuous optimization

## Success Metrics

### Phase 1 Success (Week 4)
- ✅ F1 ≥ 0.5 for simple cases @ <150ms
- ✅ F1 ≥ 0.7 for medium cases @ <400ms
- ✅ Pattern coverage ≥ 90%

### Phase 2 Success (Month 2)
- ✅ F1 ≥ 0.75 overall average
- ✅ P95 latency < 500ms
- ✅ Self-improving via GEPA

## Risk Mitigation

### Risk 1: Codegen Complexity
- **Mitigation**: Start with simple patterns, incrementally add complex ones
- **Fallback**: Use PyPy or Numba for JIT compilation

### Risk 2: SLM Still Not Refining
- **Mitigation**: Use LFM2-350M-Extract which is purpose-built
- **Fallback**: Skip SLM, use tiny LLM (Qwen) directly

### Risk 3: Pattern Conflicts
- **Mitigation**: Priority ordering, most specific first
- **Fallback**: Use confidence scoring to pick best match

## Conclusion

The path to F1=0.75 is clear and achievable within 4 weeks:

1. **Complete the patterns** (2 → 20) for broad coverage
2. **Add codegen** for 10x speed boost
3. **Fix lexicalization** for format consistency
4. **Integrate LFM2-350M-Extract** for targeted refinement

With staged routing based on complexity, we can achieve:
- **80% of cases**: F1=0.5-0.7 @ <200ms (fast path)
- **20% of cases**: F1=0.85 @ 1-2s (async path)

This rivals LLM extraction quality while maintaining the speed, explainability, and control advantages of rule-based systems.