# Negation Scope Fix - Implementation Plan

**Date**: 2025-09-30
**Priority**: 🔴 CRITICAL
**Estimated Effort**: 6-8 hours
**Risk**: Low (localized changes)

---

## Problem Statement

The memory system applies negation globally to ALL extracted facts instead of tracking which negation applies to which fact. This breaks the fundamental ability for users to correct facts through natural conversation.

### Real-World Failure Case

**User says:**
> "So you don't remember that I told you already in like three hours ago maybe that I am currently unemployed and that I work from home but I don't have a great job anymore."

**Expected behavior:**
1. ✅ Extract: `(you, is, unemployed)` - ASSERT (new fact)
2. ✅ Extract: `(you, work_from_home, true)` - ASSERT (new fact)
3. ✅ Negate: `(you, has, great job)` - NEGATE (based on "don't have")
4. ✅ Ignore "don't remember" - meta-commentary, not fact negation

**Actual behavior:**
1. ❌ Extract: `(you, is, unemployed)` then NEGATE it (wrong!)
2. ❌ Extract: `(you, has, great job)` then NEGATE it (right by accident!)
3. ❌ Miss: `(you, work_from_home, true)` completely

---

## Root Cause Analysis

### 1. Flat Negation Counting

**Location**: `core/memory/memory_hotpath.py:394-395`

```python
# Count negations
elif dep == "neg":
    neg_count += 1  # ← Counts ALL negations without scope
```

**Problem**: Returns single global `neg_count` that gets applied to ALL extracted facts.

### 2. Global Negation Application

**Location**: `core/memory/memory_hotpath.py:265-278`

```python
# Apply negation globally if neg_count > 0
if neg_count > 0:
    for (s, r, d) in triples:
        self.store.negate_edge(s, r, d)  # ← Negates EVERYTHING
```

**Problem**: No per-triple negation tracking.

### 3. No Clause Boundary Detection

**Missing**: Logic to distinguish:
- Main clause negation: "I don't live here"
- Meta-verb negation: "You don't remember that I live here" (ignore)
- Embedded negation: "I told you I don't like it" (apply to embedded fact only)

---

## Solution Design

### Architecture: Three-Phase Approach

#### Phase 1: Negation Scope Tracking
Build a map from each token to its negation status.

#### Phase 2: Meta-Verb Context Filtering
Identify when facts are in reported speech (meta-verb context) and ignore meta-verb negations.

#### Phase 3: Per-Triple Negation Application
Apply negation only to the specific triples governed by each negation.

---

## Detailed Implementation

### Phase 1: Negation Scope Tracking

**Goal**: Map each verb to whether it's directly negated.

**New Data Structure**:
```python
# In _extract() method
negation_map: Dict[int, bool] = {}  # token.i → is_negated

# Build map during initial pass
for token in doc:
    if token.dep_ == "neg":
        # Mark the negation's head (verb) as negated
        negation_map[token.head.i] = True
```

**Example**:
```
"I don't have a great job"
       ↓
have.i → True (negated by "don't")
```

**Integration Point**: `memory_hotpath.py:313-397` in `_extract()` method

### Phase 2: Meta-Verb Context Detection

**Goal**: Identify when extraction happens in reported speech context.

**New Constants**:
```python
# Add to memory_hotpath.py module level (after imports)
META_VERBS = frozenset({
    # Speech verbs
    'tell', 'say', 'mention', 'claim', 'state', 'announce', 'declare',
    # Cognitive verbs
    'remember', 'recall', 'forget', 'know',
    # Belief verbs
    'think', 'believe', 'suppose', 'assume', 'feel', 'imagine',
    # Question verbs
    'ask', 'wonder', 'question'
})
```

**New Helper Method**:
```python
def _is_in_meta_context(self, token, negation_map) -> tuple[bool, bool]:
    """
    Check if token is in reported speech/meta-commentary context.

    Returns:
        (in_meta_context, meta_verb_negated)

    Examples:
        "I don't remember that I live here"
        → live: (True, True) - in meta-context, meta-verb negated
        → Result: Extract "live here" WITHOUT negation

        "I remember that I don't live here"
        → live: (True, False) - in meta-context, meta-verb not negated
        → Result: Extract "live here" WITH negation (from "don't")

        "I don't live here"
        → live: (False, False) - not in meta-context
        → Result: Apply negation normally
    """
    current = token.head

    # Walk up dependency tree
    while current.head != current:  # Until ROOT
        # Check if current is a complement clause
        if current.dep_ in {'ccomp', 'xcomp'} and current.pos_ in {'VERB', 'AUX'}:
            # Check if governed by meta-verb
            if current.head.lemma_ in META_VERBS:
                # Check if meta-verb is negated
                meta_verb_negated = current.head.i in negation_map
                return (True, meta_verb_negated)

        current = current.head

    return (False, False)
```

**Integration Points**:
- Call from: `_extract_acomp()`, `_extract_object()`, `_extract_attribute()`
- All handlers that create subject-predicate-object triples

### Phase 3: Per-Triple Negation Tracking

**Goal**: Attach negation metadata to each triple during extraction.

**New Data Structure**:
```python
# In _extract() method
triple_metadata: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

# During extraction, store metadata
triple = (subj, pred, obj)
triple_metadata[triple] = {
    "negated": is_negated,
    "in_meta_context": in_meta_context,
    "meta_verb_negated": meta_verb_negated
}
```

**Modified Return Signature**:
```python
# Current
def _extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any, Dict[str, str]]:

# New
def _extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], Dict[Tuple[str, str, str], Dict], Any, Dict[str, str]]:
    #                                                                            ↑
    #                                            Changed from `int` (neg_count) to Dict (triple_metadata)
```

**Negation Decision Logic**:
```python
def _should_negate_triple(triple_meta: Dict) -> bool:
    """
    Determine if triple should be negated based on context.

    Decision matrix:

    | in_meta_context | meta_verb_negated | triple_negated | Result    | Example                                    |
    |-----------------|-------------------|----------------|-----------|--------------------------------------------|
    | False           | -                 | False          | ASSERT    | "I live here"                             |
    | False           | -                 | True           | NEGATE    | "I don't live here"                       |
    | True            | True              | False          | ASSERT    | "don't remember that I live here"         |
    | True            | True              | True           | NEGATE    | "don't remember that I don't live here"   |
    | True            | False             | False          | ASSERT    | "remember that I live here"               |
    | True            | False             | True           | NEGATE    | "remember that I don't live here"         |
    """
    in_meta = triple_meta.get("in_meta_context", False)
    meta_neg = triple_meta.get("meta_verb_negated", False)
    triple_neg = triple_meta.get("negated", False)

    if not in_meta:
        # Simple case: apply negation directly
        return triple_neg
    else:
        # Meta-context: ignore meta-verb negation, use only triple's own negation
        return triple_neg
```

**Storage Application**:
```python
# In process_turn() method, replace lines 265-278:

for triple in triples:
    triple_meta = triple_metadata.get(triple, {})
    should_negate = self._should_negate_triple(triple_meta)

    subj, rel, obj = triple

    if should_negate:
        self.store.negate_edge(subj, rel, obj)
        logger.debug(f"[HotMem] Negated: {triple}")
    else:
        self.store.observe_edge(subj, rel, obj, weight=confidence)
        logger.debug(f"[HotMem] Observed: {triple}")
```

---

## Implementation Checklist

### Step 1: Add Module-Level Constants
- [ ] Add `META_VERBS` frozenset after imports in `memory_hotpath.py`

### Step 2: Modify `_extract()` Method
- [ ] Add `negation_map: Dict[int, bool] = {}` after doc parsing
- [ ] Build negation map in initial token loop (before dep handlers)
- [ ] Add `triple_metadata: Dict[Tuple[str, str, str], Dict] = {}`
- [ ] Pass `negation_map` to extraction handlers
- [ ] Change return type from `neg_count` to `triple_metadata`

### Step 3: Add Helper Methods
- [ ] Implement `_is_in_meta_context(token, negation_map)` method
- [ ] Implement `_should_negate_triple(triple_meta)` static method

### Step 4: Modify Extraction Handlers
- [ ] Update `_extract_acomp()` to:
  - Accept `negation_map` parameter
  - Call `_is_in_meta_context()`
  - Check if head verb is in `negation_map`
  - Store metadata in `triple_metadata`

- [ ] Update `_extract_object()` similarly
- [ ] Update `_extract_attribute()` similarly
- [ ] Update `_extract_subject()` for copula patterns

### Step 5: Modify Storage Logic
- [ ] Update `process_turn()` lines 265-278
- [ ] Replace global negation loop with per-triple logic
- [ ] Use `_should_negate_triple()` for each triple
- [ ] Add debug logging for negation decisions

### Step 6: Testing
- [ ] Create `test_negation_scope.py` with test cases:
  - Simple negation: "I don't live here"
  - Meta-verb with negation: "don't remember that I live here"
  - Embedded negation: "remember that I don't live here"
  - Multiple negations: "don't remember that I don't have a job"
  - Complex case: User's original sentence

- [ ] Run existing tests to ensure no regression
- [ ] Add integration test with real conversation flow

---

## Test Cases

### Test 1: Simple Negation (Baseline)
```python
Input: "I don't live in Paris"
Expected:
  - Extract: (you, lives_in, paris)
  - Negation: True (direct negation of main verb)
  - Result: NEGATE edge
```

### Test 2: Meta-Verb Negation (Core Fix)
```python
Input: "You don't remember that I live in Paris"
Expected:
  - Extract: (you, lives_in, paris)
  - in_meta_context: True
  - meta_verb_negated: True ("don't remember")
  - triple_negated: False
  - Result: ASSERT edge (ignore meta-verb negation)
```

### Test 3: Embedded Clause Negation
```python
Input: "You remember that I don't live in Paris"
Expected:
  - Extract: (you, lives_in, paris)
  - in_meta_context: True
  - meta_verb_negated: False
  - triple_negated: True ("don't live")
  - Result: NEGATE edge (apply embedded negation)
```

### Test 4: Multiple Negations (Complex)
```python
Input: "You don't remember that I told you I don't have a great job"
Expected:
  - Extract: (you, has, great job)
  - in_meta_context: True (under "told" under "remember")
  - meta_verb_negated: True ("don't remember")
  - triple_negated: True ("don't have")
  - Result: NEGATE edge (apply embedded negation, ignore meta)
```

### Test 5: Real-World Case
```python
Input: "So you don't remember that I told you already that I am currently unemployed and I don't have a great job anymore"
Expected:
  - Extract: (you, is, unemployed)
    - in_meta_context: True
    - triple_negated: False
    - Result: ASSERT

  - Extract: (you, has, great job)
    - in_meta_context: True
    - triple_negated: True ("don't have")
    - Result: NEGATE
```

---

## Performance Impact

**Expected overhead**: +2-5ms per extraction

**Breakdown**:
- Negation map building: +0.5ms (one pass over tokens)
- Meta-context checking: +1-2ms (tree traversal per triple)
- Metadata storage: +0.5ms (dict operations)

**Mitigation**:
- Use frozenset for O(1) meta-verb lookups
- Cache meta-context results per token
- Lazy evaluation - only check when extracting facts

**Acceptable**: Within 5ms extraction budget (current avg: 3.8ms → target: <8ms)

---

## Backward Compatibility

**Completely backward compatible**:
- ✅ No breaking changes to public API
- ✅ Graceful degradation (if negation tracking fails, falls back to current behavior)
- ✅ All existing tests should continue to pass
- ✅ Only affects negation handling logic (isolated change)

---

## Future Enhancements

### Phase 4: Tense Transformation (Optional)
```python
# "I don't have a job anymore" → "I had a job"
if is_negated and "anymore" in context:
    # Create past-tense edge
    self.store.observe_edge(subj, f"had_{rel}", obj, status=EdgeStatus.ARCHIVED)
```

### Phase 5: Temporal Negation (Optional)
```python
# "I didn't live there last year" → temporal scope
# Attach time metadata to negation
```

---

## Success Criteria

1. ✅ User's original sentence extracts correctly:
   - "I am unemployed" → ASSERT
   - "I don't have a great job" → NEGATE

2. ✅ All test cases pass (simple, meta-verb, embedded, complex)

3. ✅ No regression in existing tests

4. ✅ Performance stays under 10ms per extraction

5. ✅ User can correct facts through natural conversation

---

## Implementation Priority

**Recommended order**:

1. **Day 1 (2-3 hours)**: Core infrastructure
   - Add constants and helper methods
   - Modify `_extract()` to build negation map
   - Update return signature

2. **Day 2 (2-3 hours)**: Extraction handlers
   - Update `_extract_acomp()`, `_extract_object()`, `_extract_attribute()`
   - Add metadata tracking
   - Implement negation decision logic

3. **Day 2 (2-3 hours)**: Testing & validation
   - Create comprehensive test suite
   - Run regression tests
   - Test with real conversations
   - Performance validation

---

## Risk Mitigation

**Risk 1**: Breaking existing negation handling
- **Mitigation**: Add feature flag `MEMORY_PRECISE_NEGATION=true` (default: false)
- **Fallback**: Keep old `neg_count` logic as backup

**Risk 2**: Performance regression
- **Mitigation**: Add performance assertions to tests
- **Rollback**: Easy to disable via flag

**Risk 3**: Edge cases not covered
- **Mitigation**: Comprehensive test suite with real-world examples
- **Monitoring**: Log negation decisions for first 2 weeks

---

## Definition of Done

- [ ] All code changes implemented and tested
- [ ] Test suite passing (5/5 core cases + regression tests)
- [ ] Performance under 10ms per extraction
- [ ] Code reviewed and documented
- [ ] User's original case working correctly
- [ ] Merged to main branch
- [ ] Deployed and monitored for 48 hours

---

**Status**: Ready for implementation
**Next Step**: Begin Phase 1 - Core infrastructure