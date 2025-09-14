# ASI_ALT_REFINED.yaml FEEDBACK & IMPROVEMENT REQUEST

## 🎯 **CURRENT RESULTS (36 patterns loaded successfully!)**

### ✅ **What's Working Great:**
- **Perfect Coverage**: Extracts triples from ALL test sentences (100% coverage!)
- **Advanced Semantics**:
  - `('author', 'write', 'The book')` - EXCELLENT passive transformation
  - `('John', 'work_at', 'Google')` - Perfect oblique handling
  - `('John', 'has_property', 'tall')` - Great semantic attributes
  - `('The CEO of Microsoft', 'part_of', 'Microsoft')` - Complex nominals working!
- **Rich Extraction**: 2-7 triples per sentence (vs our champion's 1-2)

### ❌ **Issues to Fix:**

#### 1. **"My name is Alex Thompson" Pattern Broken**
```
Current Output: ('My', 'possess_of', 'My name'), ('My name', 'exist', '')
Expected Output: ('you', 'has_name', 'Alex Thompson')  ← This is CRITICAL
```

**Problem**: The winning pattern from our champion is missing. This specific pattern scored highest quality (1.0) in our tests.

**Fix Needed**: Add high-priority pattern (priority: 250) that specifically handles:
```yaml
- name: PERSONAL_NAME_DECLARATION
  priority: 250  # HIGHEST PRIORITY
  pattern:
    anchor: {pos: "AUX", lemma: "be|is|are|was|were"}
    edges:
      - {from: anchor, rel: "^nsubj", as: name_noun}
      - {from: name_noun, rel: "^poss", as: poss_pron}
      - {from: anchor, rel: "^attr", as: actual_name}
  emit:
    - subj: "you"
      pred: "has_name"
      obj: "{actual_name.text}"  # Use .text not .subtree
      canon: "PERSONAL_NAME"
  guards:
    name_noun_lemma_in: ["name"]
    poss_pron_lemma_in: ["my", "i"]
```

#### 2. **Over-extraction of "exist" Relations**
**Current**: Nearly every sentence gets `('X', 'exist', '')` triples
**Problem**: These low-quality triples dilute the semantic value
**Fix**: Restrict existential patterns to only sentences with explicit existential markers:
- "There is/are..." constructions
- Existential verbs: "exist", "occur", "happen"

#### 3. **Quality vs Quantity Balance**
**Target**: Beat our current champion's quality score of **0.489/1.000**
**Strategy**: Fewer, higher-quality triples > Many low-quality ones

## 📊 **Our Test Framework (For Your Reference)**

### **Quality Scoring Logic:**
```python
# High quality patterns get +0.8 points:
if "name" in sentence and "has_name" in predicate: +0.8
if "work" in sentence and "work" in predicate: +0.8
if "live" in sentence and "live" in predicate: +0.8

# Medium quality patterns get +0.5-0.7 points:
if "give/like/has" matches: +0.7
if basic relations like "be", "possess": +0.5

# Low quality patterns get +0.3 points:
if generic extraction: +0.3
if "quantified_as", "exist": +0.0-0.3 (penalty)
```

### **Test Sentences (Priority Order):**
1. "My name is Alex Thompson." ← **MUST** get 0.8+ quality
2. "John works at Google." ← **MUST** get 0.8+ quality
3. "Mary lives in Paris." ← **MUST** get 0.8+ quality
4. "She likes chocolate."
5. "John gave Mary a book."
6. "The book was written by the author."
7. "John has a car."
8. And 7 more complex sentences...

## 🚀 **IMPROVEMENT REQUESTS:**

1. **Fix "My name is" pattern** - This is our #1 quality winner
2. **Reduce existential over-generation** - Quality over quantity
3. **Ensure key semantic patterns score high**:
   - work_at, live_in, has_name should be top priority (200+ priority)
   - Complex patterns like passives can be lower priority (100-150)
   - Generic existence patterns should be lowest (50-100)

4. **Test against our current champion**:
   - **Current SOTA**: 0.489 quality, 92.9% correctness, 0.1ms speed
   - **Your target**: >0.500 quality, >95% correctness, <0.2ms speed

## 🎯 **Success Criteria:**
- **"My name is Alex Thompson"** → `('you', 'has_name', 'Alex Thompson')`
- **Quality score >0.500** (beat current 0.489)
- **All 14 test sentences working**
- **Rich but not overwhelming** (2-4 quality triples per sentence)

Your 36-pattern approach is VERY promising - just needs these tweaks to become the new champion! 🏆