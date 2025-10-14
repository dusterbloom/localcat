# Contextual Extraction Granularity Plan

**Problem**: Flat triple extraction loses essential context from hierarchical linguistic structures.

**Status**: Research complete, ready for implementation

---

## Current State Analysis

### What Works
✅ 27 USGS dependency patterns implemented
✅ Negation detection (just fixed: `neg_count` now applies correctly)
✅ Basic verb+prep combinations: `lives_in`, `works_at`, `swim_in`
✅ Entity extraction with noun chunks
✅ ~5ms average extraction time

### What's Broken
❌ Context loss: "I love swimming **in the sea**" → `(you, love, swimming)` ← LOST: location
❌ Ambiguity: Can't distinguish "swimming in sea" from "swimming in lakes"
❌ Hallucination: LLM fills missing context with wrong details

### Evidence from Logs (2025-09-30)
```
Input:  "I love swimming in the sea"
Output: (you, love, swimming)  ← Missing "in the sea"

Input:  "I don't like swimming in lakes"
Output: (you, like, swimming) + NEGATED  ← Missing "in lakes"

Database shows:
- you|love|swimming
- you|dislike|swimming  ← CONTRADICTORY without context
- you|swim_in|lake      ← From question, not actual statement
```

---

## Root Cause: Flat Triple Extraction

### Dependency Tree Structure
```
"I love swimming in the sea"

love (ROOT)
├─ I (nsubj)
└─ swimming (xcomp)      ← We extract this as object
    └─ in (prep)         ← But we IGNORE this child
        └─ sea (pobj)    ← And this grandchild
```

### Current Code Behavior
**File**: `core/memory/memory_hotpath.py:447-452`
```python
# Direct object extraction
for child in head.children:
    if child.dep_ in {"dobj", "obj"}:
        obj = self._get_entity(child, entity_map)  # ← Gets "swimming"
        triples.append((subj, pred, obj))          # ← Stops here
```

**File**: `core/memory/memory_hotpath.py:454-482`
```python
# Prepositional phrases extracted separately
for child in head.children:
    if child.dep_ == "prep":  # ← Only looks at verb's direct children
        # So "love → in" would work, but "swimming → in" is missed
```

---

## Past Experiments

### 1. Archive: USGS Extractor (memory_extraction_usgs.py)
- Implemented all 27 dependency types
- Had same issue: extracted verb+prep but not noun+prep
- Line 235-241: Only handled prep as child of VERB, not NOUN/VERB objects

### 2. DSPy Extraction (test_dspy_extraction.py)
- Used LLM to extract complex facts
- Tested on: "Alice, software engineer at Google who loves Python, lives in SF"
- Achieved 67-75% fact extraction
- **Problem**: Too slow (LLM inference), defeats <200ms budget
- **Use case**: Complex multi-clause sentences only

### 3. Current Implementation
- UDExtractor delegates to HotMemory._extract methods
- Each dep type has dedicated handler
- No handler checks object's prep modifiers

---

## Solution Design

### Principle: **Contextual Entity Extraction**

When extracting an object, include its essential disambiguating modifiers:
1. Prepositional phrases (location, time, manner)
2. Adjectival modifiers (attributes)
3. Compound nouns (multi-word concepts)

### Why This Works
- **Linguistically grounded**: Matches how humans chunk meaning
- **Minimal code change**: Enhance existing extraction, don't rebuild
- **Readable**: "swimming in the sea" is natural language
- **Performance**: No LLM calls, stays <200ms
- **Universal**: Works for any language with prep modifiers

---

## Implementation Plan

### Phases 1-3: Unified Contextual Extraction (RECOMMENDED)

**Strategy**: Implement all three types of context (prep, amod, compound) in a single method and session.

**Why Together?**
- All three use the same pattern: walk `token.children` checking different `dep_` types
- Solves 95% of context loss with minimal code (~60 lines total)
- Single testing and validation cycle
- Low risk: surgical enhancement of existing extraction
- Time estimate: **3-4 hours total**

**New Method**: `_get_entity_with_context(token, entity_map)`
```python
def _get_entity_with_context(self, token, entity_map, max_length: int = 50) -> tuple[str, str]:
    """
    Get entity with full contextual modifiers.

    Returns:
        (root_entity, enriched_entity) tuple for dual registration

    CRITICAL: root stays untouched (canonical base form), enriched gets modifiers.
    This ensures entity_index["car"] finds "red car" edges.

    Includes:
    1. Prepositional phrases (location, time, manner)
    2. Adjectival modifiers (attributes)
    3. Compound nouns (multi-word concepts)

    Examples:
    - root="swimming", enriched="swimming in the sea"       ← prep
    - root="car", enriched="red car"                        ← amod
    - root="learning", enriched="machine learning"          ← compound
    - root="meeting", enriched="meeting on tuesday"         ← prep
    """
    import time
    start = time.perf_counter()

    # Get root entity (canonical form) - NEVER MODIFIED
    # IMPORTANT: entity_map may already contain full noun chunks ("red car"),
    # so derive the true head from the token itself before consulting the map.
    raw_root = token.lemma_ or token.text
    root = _canon_entity_text(raw_root)  # Apply canonical form immediately

    # If entity_map carries a chunk-alias (e.g., "red car"), record it now so
    # we can index the enriched edge under both the chunk and the canonical root.
    chunk_alias = entity_map.get(token.i)
    if chunk_alias and _canon_entity_text(chunk_alias) != root:
        self._entity_aliases[_canon_entity_text(chunk_alias)] = root

    # Start building enriched from root
    enriched = root

    # Phase 3: Collect compound nouns (comes before root)
    # Cap at 3 compounds to prevent pathological cases
    # IMPORTANT: Sort by token.i to preserve left-to-right order ("machine learning", not "learning machine")
    compounds = []
    for child in sorted(token.children, key=lambda t: t.i):
        if child.dep_ == "compound" and len(compounds) < 3:
            compounds.append(_canon_entity_text(child.text))

    # Build enriched with compounds (root stays untouched)
    if compounds:
        enriched = " ".join(compounds + [enriched])

    # Phase 2: Collect adjectives
    # Cap at 5 adjectives to prevent pathological cases
    # IMPORTANT: Sort by token.i to preserve natural order ("big blue house", not "blue big house")
    adjectives = []
    for child in sorted(token.children, key=lambda t: t.i):
        if child.dep_ == "amod" and len(adjectives) < 5:
            adjectives.append(_canon_entity_text(child.text))

    # Add adjectives before enriched (root stays untouched)
    if adjectives:
        enriched = " ".join(adjectives + [enriched])

    # Phase 1: Collect prepositional phrases
    # Cap at 3 prep phrases to prevent pathological cases
    prep_parts = []
    for child in token.children:
        if child.dep_ == "prep" and len(prep_parts) < 3:
            prep_text = child.text.lower()
            # Get prepositional object
            for pobj_child in child.children:
                if pobj_child.dep_ == "pobj":
                    pobj_text = self._get_entity(pobj_child, entity_map)
                    pobj_text = _canon_entity_text(pobj_text)
                    prep_parts.append(f"{prep_text} {pobj_text}")
                    break  # Only first pobj per prep

    # Combine with prep phrases (root stays untouched)
    if prep_parts:
        enriched = f"{enriched} {' '.join(prep_parts)}"

    # Cap length to prevent pathological cases
    if len(enriched) > max_length:
        truncated = enriched[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
        # Fallback if truncation produces empty string
        if truncated:
            enriched = truncated
        else:
            enriched = enriched[:max_length]  # Hard cut as last resort
        # Track truncations for monitoring
        if hasattr(self, '_metrics'):
            self._metrics['enrichment_truncations'] = self._metrics.get('enrichment_truncations', 0) + 1

    # Performance monitoring
    elapsed_ms = (time.perf_counter() - start) * 1000
    # Only log slow-path warning if debug logging enabled to avoid log noise
    if elapsed_ms > 1.0 and logger.level <= 10:  # DEBUG = 10
        logger.debug(f"Slow entity enrichment: {elapsed_ms:.2f}ms for '{enriched}'")

    # Track metrics
    if hasattr(self, '_metrics'):
        self._metrics.setdefault('entity_enrichment_times_ms', []).append(elapsed_ms)
        self._metrics.setdefault('enriched_lengths', []).append(len(enriched))

    # Store alias mapping if enriched differs from root
    if enriched != root:
        self._entity_aliases[enriched] = root

    return root, enriched
```

**Update Sites** (9 locations in `memory_hotpath.py`):
1. **dobj/obj extraction** (lines 447-452): Direct objects
2. **attr extraction** (lines 430-440): Attributes in "is/are" constructions
3. **pobj extraction** (lines 459-482): Prepositional objects (already has special handling)
4. **conj verb dobj** (lines 496-501): Conjoined verb direct objects
5. **conj verb pobj** (lines 507-524): Conjoined verb prepositional objects
6. **_extract_object** (line 528): Legacy dobj handler if still called
7. **xcomp extraction** (need to add): Clausal complements like "love swimming"
8. **iobj extraction** (if exists): Indirect objects
9. **ccomp extraction** (if exists): Clausal complements

**Call Site Pattern**:
```python
# At extraction call sites (e.g., dobj extraction):
root_obj, enriched_obj = self._get_entity_with_context(child, entity_map)
entities.add(root_obj)  # Add ROOT to entity set (canonical "car", "swimming", "learning")
triples.append((subj, pred, enriched_obj))  # Use ENRICHED in triple
self._enriched_entities.add(root_obj)  # Track for conditional amod
# Chunk aliases from entity_map were already recorded inside
# `_get_entity_with_context`, so both the noun chunk ("red car") and the
# canonical root ("car") point to the same enriched edge in `entity_index`.

# Result for "I drive a red car":
# - root_obj = "car"
# - enriched_obj = "red car"
# - entities contains "car"
# - triple is (you, drive, "red car")
# - entity_index["car"].add((you, drive, "red car"))  ← dual registration
```

**Testing Examples**:
```python
# Phase 1: Prepositional phrases
"I love swimming in the sea"
→ triple: (you, love, "swimming in the sea")
→ entities: {"you", "swimming"}
→ entity_index["swimming"] contains edge  ✅

"I don't like swimming in lakes"
→ triple: (you, like, "swimming in lakes") + NEGATED
→ entities: {"you", "swimming"}
→ entity_index["swimming"] contains edge  ✅

# Phase 2: Adjectives
"I drive a red car"
→ triple: (you, drive, "red car")
→ entities: {"you", "car"}
→ entity_index["car"] contains edge  ✅

"I saw a big blue house"
→ triple: (you, saw, "big blue house")
→ entities: {"you", "house"}
→ entity_index["house"] contains edge  ✅

# Phase 3: Compounds
"I study machine learning"
→ triple: (you, study, "machine learning")
→ entities: {"you", "learning"}
→ entity_index["learning"] contains edge  ✅

"I'm a software engineer"
→ triple: (you, is, "software engineer")
→ entities: {"you", "engineer"}
→ entity_index["engineer"] contains edge  ✅

# Combined: All three types
"I work on complex machine learning projects in San Francisco"
→ triple: (you, work, "complex machine learning projects in san francisco")
→ entities: {"you", "projects"}
→ entity_index["projects"] contains edge  ✅
```

**Implementation Checklist**:

**Phase 1: Extraction Method (30-45 min)**
- [ ] Add `_get_entity_with_context()` method returning `(root, enriched)` tuple
  - [ ] **CRITICAL**: Keep `root` untouched (canonical base form), build `enriched` separately
  - [ ] Sort children by `token.i` for deterministic modifier order
- [ ] Apply `_canon_entity_text()` to root immediately
- [ ] If `entity_map[token.i]` is a noun-chunk alias, map it to root (`_entity_aliases[chunk] = root`)
- [ ] Apply `_canon_entity_text()` to all modifiers (compounds, adjectives, pobj)
  - [ ] Add truncation fallback to prevent empty strings
  - [ ] Gate slow-path warning behind debug level check
  - [ ] Populate `self._entity_aliases[enriched] = root` at end of method

**Phase 2: Extraction Interface (45-60 min)**
- [ ] Initialize `_entity_aliases = {}` and `_enriched_entities = set()` at start of `_extract()`
- [ ] Update `_extract()` signature to return 5-tuple: `(entities, triples, neg_count, doc, aliases)`
- [ ] Update all extraction call sites (7-10 locations):
  - [ ] `extractors/ud.py:92` - `_extract_direct`
  - [ ] `extractors/ud.py:105` - `_extract_with_preprocessing`
  - [ ] `extractors/ud.py:121, 125` - `_extract_from_doc`
  - [ ] `memory_hotpath.py:164` - `process_turn`
  - [ ] `memory_hotpath.py:864` - `preview_bullets`
  - [ ] Unit tests that mock extraction
  - [ ] Integration tests

**Phase 3: Call Sites (45-60 min)**
- [ ] Update all 9 extraction call sites to use new method:
  - [ ] Pattern: `root, enriched = self._get_entity_with_context(...)`
  - [ ] Add ROOT to entities set: `entities.add(root)`
  - [ ] Use ENRICHED in triple: `triples.append((subj, pred, enriched))`
  - [ ] Track for amod: `self._enriched_entities.add(root)`
  - [ ] Locations:
    - [ ] `dobj/obj` (lines 447-452)
    - [ ] `attr` (lines 430-440)
    - [ ] `pobj` (lines 459-482)
    - [ ] `conj verb dobj` (lines 496-501)
    - [ ] `conj verb pobj` (lines 507-524)
    - [ ] `_extract_object` (line 528) if still called
    - [ ] Add `xcomp` handler (similar to dobj)
    - [ ] Add `iobj` handler if needed
    - [ ] Add `ccomp` handler if needed
- [ ] **Trade-off: Implement conditional amod** - only extract quality if root not in `_enriched_entities`

**Phase 4: Hot Index Updates (30-45 min)**
- [ ] Update `memory_hotpath.py:266-267` - add dual registration in process_turn:
  ```python
  # After: self.entity_index[d].add((s, r, d))
  root_d = self._entity_aliases.get(d, d)
  if root_d != d:
      self.entity_index[root_d].add((s, r, d))
  ```
- [ ] Add `_extract_base_entity()` helper with prep/compound heuristic (for rebuild)
- [ ] Update `memory_hotpath.py:850-851` - add dual registration in rebuild_from_store
- [ ] **NO changes to MemoryStore** - keep storage layer stable

**Phase 5: Testing (60-90 min)**
- [ ] Unit tests for all example sentences (10-12 test cases)
- [ ] Test enriched entities: "swimming in the sea", "red car", "machine learning"
- [ ] Test alias map survives refinement
- [ ] **Measure retrieval impact**:
  - [ ] Verify `entity_index["swimming"]` contains edges with "swimming in the sea"
  - [ ] Test base noun queries return enriched results
  - [ ] Test BM25/FTS ranking on enriched strings
- [ ] Profile performance (expect ~5-6ms, well within budget)
- [ ] Run on 20-30 real conversation logs
- [ ] Verify no regressions in existing extraction
- [ ] Measure contradiction reduction

**Time Breakdown**:
- Method implementation: 30-45 min (~80 lines with sorting/fallbacks/canon)
- Extraction interface changes: 45-60 min (5-tuple signature + 7-10 call sites)
- Call site updates: 45-60 min (9 extraction locations + aliasing logic)
- Hot index updates: 30-45 min (dual registration + base extraction heuristic)
- Unit tests: 45-60 min (10-12 test cases + refinement tests)
- Retrieval impact testing: 30-45 min (base noun lookups, BM25 ranking)
- Integration testing: 60-90 min (real logs, profiling)
- **Total: 5-7 hours** (more realistic with interface changes)

**Lines of Code**:
- New method: ~80 lines (with sorting, fallbacks, metrics, canon)
- Hot index changes: ~40 lines (dual registration + `_extract_base_entity`)
- Entity aliasing logic: ~20 lines (`_entity_aliases`, `_enriched_entities`)
- Extraction interface updates: ~30 lines (signature change + call sites)
- Call site updates: ~35 lines (9 extraction locations)
- Tests: ~150 lines (including refinement + retrieval tests)
- **Total: ~355 lines**

---

### Phase 4: Metadata Layer (FUTURE - DEFERRED)

**Goal**: Make enhanced triples queryable

**Structure**:
```python
{
    "triple": ("you", "love", "swimming in the sea"),
    "object_structured": {
        "root": "swimming",
        "modifiers": {
            "location": "sea"
        }
    }
}
```

**Use cases**:
- Query: "Show all swimming activities" → finds both "swimming in sea" and "swimming in lakes"
- Query: "Where do I swim?" → extracts location modifier

---

## Performance Impact

### Estimate
- Current: ~5ms extraction
- Added: Walk prep children (1-2 nodes) per object
- Expected: +0.5ms per sentence with prep phrases
- Total: ~5-6ms (well within 200ms budget)

### Validation
- Profile with `time.perf_counter()` on existing metrics
- Test on 100 real conversation turns
- Ensure p95 < 10ms

---

## Architecture Changes Required

### Overview of Changes

This implementation requires changes across **three layers**:
1. **Extraction Layer** (`HotMemory._extract`, `UDExtractor`)
2. **Hot Index Layer** (`HotMemory.entity_index`)
3. **Storage Layer** (NO changes - keep it simple)

**Key Decision**: Do NOT modify `MemoryStore.observe_edge()`. Instead, handle aliasing entirely in the hot index layer where it already exists.

---

## Critical Implementation Concerns

### 1. Storage Layer: Keep It Simple (NO CHANGES)

**Critical Decision**: Do NOT modify `MemoryStore.observe_edge()` signature.

**Why**:
- 14 call sites across codebase (tests, orchestrator, hotpath)
- Storage layer has no concept of `entity_index` (that's in HotMemory)
- SQLite/LMDB queues would need rework (`memory_store.py:422`)
- Breaking change to stable API

**Instead**: Handle aliasing in the hot index layer where `entity_index` actually lives (`memory_hotpath.py:266-267, 850-851`).

---

### 2. Hot Index Layer: Dual Registration (WHERE CHANGES HAPPEN)

**Problem**: Once we return "swimming in the sea" instead of "swimming", the entity index needs both forms.

**Why Critical**:
- `entity_index` is populated at `memory_hotpath.py:266-267` (process_turn)
- And at `memory_hotpath.py:850-851` (rebuild_from_store)
- Retrieval queries like "tell me about swimming" must match enriched edges
- Neighbor lookups use `entity_index[base]` to find edges

**Solution**: Dual registration in `HotMemory.entity_index` (lines 266-267)

**Current Code** (`memory_hotpath.py:260-267`):
```python
# Update hot indices
self.entity_index[s].add((s, r, d))
self.entity_index[d].add((s, r, d))
```

**After Change**:
```python
# Update hot indices
self.entity_index[s].add((s, r, d))
self.entity_index[d].add((s, r, d))

# NEW: If dst was enriched, also index under base form
# This enables queries like "swimming" to find "swimming in the sea"
base_d = self._entity_aliases.get(d, d)
if base_d != d:
    self.entity_index[base_d].add((s, r, d))
```

**Where `_entity_aliases` comes from**:
- Built during extraction in `_extract()`
- Maps enriched → base: `{"swimming in the sea": "swimming", "red car": "car"}`
- Also records noun-chunk aliases emitted by `_build_entity_map` so `entity_map` values like "red car" resolve back to the canonical root (`"car"`).
- Survives refinement (see Section 3 below)

**Call Sites to Update**:
1. `memory_hotpath.py:266-267` - process_turn indexing ✓
2. `memory_hotpath.py:850-851` - rebuild_from_store ✓ (needs alias reconstruction)

**For rebuild_from_store**, we need to extract base from enriched dst strings:
```python
def rebuild_from_store(self):
    edges = self.store.get_all_edges()
    for s, r, d, conf in edges:
        if conf > 0.1:
            self.entity_index[s].add((s, r, d))
            self.entity_index[d].add((s, r, d))

            # NEW: Also index under base form if dst looks enriched
            base_d = self._extract_base_entity(d)
            if base_d != d:
                self.entity_index[base_d].add((s, r, d))

def _extract_base_entity(self, entity: str) -> str:
    """
    Heuristic to extract base from enriched form.

    Examples:
    - "swimming in the sea" -> "swimming" (first word)
    - "red car" -> "car" (last word if multiple)
    - "machine learning" -> "learning" (last word)

    Strategy: If multi-word, check if contains prepositions (in, on, at, with, for).
    If yes, base is first word. Otherwise, base is last word (compound pattern).
    """
    words = entity.split()
    if len(words) <= 1:
        return entity

    # Check for prep pattern: "X in Y", "X on Y"
    preps = {"in", "on", "at", "with", "for", "from", "to", "by"}
    if any(w in preps for w in words[1:]):
        return words[0]  # "swimming" from "swimming in sea"

    # Otherwise assume compound: "machine learning" -> "learning"
    return words[-1]
```

**Testing Requirements**:
- Verify `entity_index["swimming"]` contains edges with "swimming in the sea"
- Test retrieval query "swimming" finds enriched edges
- Measure BM25 recall on base noun queries
- Test rebuild_from_store creates correct aliases

---

### 3. Extraction Layer: Interface Changes

**Problem**: `_extract()` currently returns 4-tuple, but we need to add alias map.

**Current Signature** (`memory_hotpath.py`, called from `extractors/ud.py:95-134`):
```python
def _extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
    """Returns (entities, triples, neg_count, doc)"""
```

**New Signature**:
```python
def _extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any, Dict[str, str]]:
    """Returns (entities, triples, neg_count, doc, entity_aliases)"""
```

**Downstream Call Sites to Update**:
1. `extractors/ud.py:105` - `_extract_with_preprocessing`
2. `extractors/ud.py:92` - `_extract_direct`
3. `extractors/ud.py:121, 125` - `_extract_from_doc` (if exists)
4. `memory_hotpath.py:164` - `process_turn` extraction call
5. `memory_hotpath.py:864` - `preview_bullets` extraction call
6. **Unit tests**: Any test that calls `_extract()` directly
7. **Integration tests**: Any test that mocks extraction

**Backward Compatibility Strategy**:

Since adding a 5th return value breaks all callers, we have two options:

**Option A: Add Optional Parameter (Recommended)**
```python
def _extract(self, text: str, lang: str, include_aliases: bool = False):
    # ... extraction logic ...

    if include_aliases:
        return entities, triples, neg_count, doc, self._entity_aliases
    else:
        return entities, triples, neg_count, doc  # Legacy 4-tuple
```

**Option B: Always Return 5-Tuple (Clean Break)**
```python
# Update all call sites to expect 5 values:
entities, triples, neg_count, doc, aliases = self.extractor.extract(text, lang)
```

**Recommendation**: **Option B** - Clean break, update all call sites.
- Only ~7-10 call sites
- Cleaner long-term
- Forces us to handle aliases everywhere

**Call Site Update Pattern**:
```python
# Before:
entities, triples, neg_count, doc = self._host._extract(text, lang)

# After:
entities, triples, neg_count, doc, aliases = self._host._extract(text, lang)
self._entity_aliases = aliases  # Store for later use
```

---

### 4. Refinement: Keeping Aliases in Sync

**Problem**: After `_extract()`, `process_turn` runs two refinement steps that modify entity strings:
1. `self.extractor.refine(text, triples, doc)` - normalizes triples (`memory_hotpath.py:968-1124`)
2. `self.extractor.refine_entities(text, entities)` - canonicalizes entities

**Both apply `_canon_entity_text()`** which:
- Lowercases
- Strips whitespace
- Normalizes pronouns ("I" → "you")

**Critical Issue**: If refinement changes "Swimming in the Sea" → "swimming in the sea", our alias map breaks.

**Solution**: Apply canonical form DURING extraction, then refinement is no-op.

**In `_get_entity_with_context()`**:
```python
def _get_entity_with_context(self, token, entity_map, max_length: int = 50) -> tuple[str, str]:
    # Get root entity (canonical form) - NEVER MODIFIED
    root = self._get_entity(token, entity_map)
    root = _canon_entity_text(root)  # Apply canonical form immediately

    # Build enriched from root (copy, don't mutate root)
    enriched = root

    # ... add compounds, adjectives, preps to enriched (root stays untouched) ...

    # Store alias mapping if enriched differs from root (already canonical)
    if enriched != root:
        self._entity_aliases[enriched] = root

    return root, enriched
```

**This ensures**:
- `root` is immutable canonical form ("car", "swimming", "learning")
- `enriched` is built from root + modifiers ("red car", "swimming in sea")
- Both are canonical on creation
- Refinement sees already-canonical strings, makes no changes
- Alias map stays valid: `{"red car": "car", "swimming in sea": "swimming"}`
- Entity set contains roots: `{"car", "swimming"}`
- Triples contain enriched: `(you, drive, "red car")`
- Index lookup works: `entity_index["car"]` finds `(you, drive, "red car")`

**Alternative**: Update aliases during refinement (more complex):
```python
def _refine_triples(self, text, triples, doc):
    refined = []
    for s, r, d in triples:
        cs = _canon_entity_text(s)
        cd = _canon_entity_text(d)

        # Update alias map if dst was canonicalized
        if d in self._entity_aliases and cd != d:
            base = self._entity_aliases.pop(d)
            self._entity_aliases[cd] = base  # Re-map with canonical key

        refined.append((cs, r, cd))
    return refined
```

**Recommendation**: **Apply canon during extraction** (simpler, less error-prone).

---

### 5. Amod Quality Triples Trade-off

**Current Behavior** (lines 576-581):
```python
def _extract_amod(self, token, entity_map, triples, entities):
    """amod - adjectival modifier"""
    adj = token.text.lower()
    head_entity = self._get_entity(token.head, entity_map)
    triples.append((head_entity, "quality", adj))
```

**Problem**: If we embed adjectives into enriched spans, this creates duplicates:
- Enriched: `(you, drive, "red car")`
- Quality: `(car, quality, red)`

**Options**:
1. **Keep both** - Redundant but explicit, supports different query patterns
2. **Retire quality triples** - Cleaner, but loses structured "quality" relation
3. **Conditional** - Only extract quality triples if object not enriched

**Recommendation**: **Option 3 (Conditional)**
```python
def _extract_amod(self, token, entity_map, triples, entities):
    # Only extract quality triple if parent wasn't enriched
    # (Enriched objects already contain adjectives)
    head_entity = self._get_entity(token.head, entity_map)
    if head_entity not in self._enriched_entities:
        adj = token.text.lower()
        triples.append((head_entity, "quality", adj))
```

**`_enriched_entities` Lifecycle**:
- **Initialization**: `self._enriched_entities = set()` at start of `_extract()` method
- **Population**: Add base entity when `_get_entity_with_context()` returns enriched form
- **Usage**: Check in `_extract_amod()` to avoid duplicate quality triples
- **Cleanup**: Automatically cleared on next `_extract()` call (per-extraction scope)
- **Thread safety**: Not needed (single-threaded extraction per turn)

```python
# In HotMemory._extract() or similar:
def _extract(self, token, entity_map, triples, entities):
    # Initialize per-extraction tracking (cleared each call)
    self._entity_aliases = {}  # enriched -> base mapping
    self._enriched_entities = set()  # base entities that were enriched

    # ... extraction logic calls _get_entity_with_context ...

    return triples, entities, self._entity_aliases
```

**Action Item**: Decide on approach during implementation, add tests for chosen behavior.

---

### 3. Performance Guardrails

**Requirements**:
1. **Time budget**: Helper method must stay <1ms per call
2. **String length cap**: Prevent pathological cases
3. **Modifier count cap**: Limit to 5 adjectives, 3 compounds, 3 preps

**Implementation**:
```python
def _get_entity_with_context(self, token, entity_map) -> tuple[str, str]:
    start = time.perf_counter()

    # ... build enriched string ...

    # Cap length to prevent pathological cases
    MAX_LENGTH = 50
    if len(enriched) > MAX_LENGTH:
        enriched = enriched[:MAX_LENGTH].rsplit(' ', 1)[0]  # Cut at word boundary

    elapsed_ms = (time.perf_counter() - start) * 1000
    if elapsed_ms > 1.0:
        logger.warning(f"Slow entity enrichment: {elapsed_ms:.2f}ms for '{enriched}'")

    return base, enriched
```

**Metrics to Track**:
- `hotmem.entity_enrichment_ms` (p50, p95, p99)
- `hotmem.enriched_length_chars` (avg, max)
- `hotmem.enrichment_truncations` (count)

---

### 4. Retrieval Ranking Impact

**Problem**: Longer dst strings change BM25 term frequency and entity_index fan-out.

**Scenarios to Test**:
1. **Base noun query**: "swimming" should still find "swimming in the sea"
2. **Specific query**: "swimming in the sea" should rank higher than "swimming in lakes"
3. **Partial match**: "sea" should find "swimming in the sea"

**Validation**:
```python
# Before/after comparison
QUERIES = [
    "swimming",           # Should find all swimming edges
    "sea",                # Should find sea-related edges
    "swimming in sea",    # Should rank sea swimming highest
]

for query in QUERIES:
    bullets_before = old_retrieve(query, top_k=5)
    bullets_after = new_retrieve(query, top_k=5)

    # Measure recall@5
    recall = len(set(bullets_before) & set(bullets_after)) / len(bullets_before)
    if recall < 0.8:
        logger.warning(f"Retrieval recall dropped for '{query}': {recall:.2%}")
```

**Action Items**:
- [ ] Add retrieval recall tests to validation suite
- [ ] Measure BM25 scoring before/after on real queries
- [ ] Verify entity_index neighbor lookups work for base forms

---

## Edge Cases to Handle

### 1. Multiple Prep Phrases
```
"I work on AI in San Francisco with Python"
→ (you, work, "on ai in san francisco with python")
```
**Decision**: Include all, space-separated

### 2. Nested Prep Phrases
```
"The book on the table in the room"
→ (book, location, "on table in room")
```
**Decision**: Flatten to single level

### 3. Temporal Preps
```
"I worked there in 2020"
→ (you, work, there) + temporal metadata?
```
**Decision**: Phase 1 includes in text, Phase 4 adds metadata

### 4. Very Long Phrases
```
"I love swimming in the crystal clear blue Mediterranean sea"
→ Too long?
```
**Decision**: Keep as-is, filter by length limit (50 chars) if needed

---

## Testing Strategy

### Unit Tests
```python
def test_prep_context_extraction():
    text = "I love swimming in the sea"
    triples = extract(text)
    assert ("you", "love", "swimming in the sea") in triples

def test_negation_with_context():
    text = "I don't like swimming in lakes"
    triples = extract(text)
    # Should be negated
    assert has_negation(triples, ("you", "like", "swimming in lakes"))
```

### Integration Tests
- Run on past 100 conversation logs
- Compare before/after extraction quality
- Measure: fewer contradictions, more context preserved

### Validation Metrics
1. **Context preservation**: % of prep phrases captured
2. **Contradiction reduction**: Fewer "you love X" + "you hate X" pairs
3. **Performance**: p95 extraction time < 10ms
4. **Storage**: Average triple length increase

---

## Migration Path

### Next Session: Phases 1-3 Combined Implementation (5-7 hours)
1. ✅ Research complete (this document)
2. **Extraction method**: Add `_get_entity_with_context()` returning `(base, enriched)` tuple
   - Sort children by `token.i` for deterministic order
   - Apply `_canon_entity_text()` immediately (prevents refinement issues)
   - Add truncation fallback to prevent empty strings
   - Gate slow-path warnings behind debug level
   - Populate `_entity_aliases[enriched] = base`
3. **Extraction interface**: Update `_extract()` signature to return 5-tuple
   - Add `entity_aliases` as 5th return value
   - Update all 7-10 call sites (extractors/ud.py, memory_hotpath.py, tests)
4. **Extraction call sites**: Update all 9 locations to use new method
   - Initialize `_entity_aliases = {}` and `_enriched_entities = set()` at start
   - dobj, attr, pobj, conj×2, xcomp, etc.
   - Populate tracking sets as we extract
5. **Hot index layer**: Add dual registration (NO storage layer changes)
   - Update `memory_hotpath.py:266-267` - add base alias indexing
   - Add `_extract_base_entity()` heuristic (prep vs compound patterns)
   - Update `memory_hotpath.py:850-851` - rebuild with alias extraction
6. **Conditional amod**: Avoid quality triples for enriched entities
7. **Comprehensive testing**:
   - Unit tests for all example sentences (10-12 test cases)
   - Test alias map survives refinement
   - Verify `entity_index["swimming"]` contains enriched edges
   - Profile performance, measure contradiction reduction
8. **Integration**: Run on 20-30 real conversation logs

### Future: Phase 4 (likely unnecessary)
- Only implement if production usage shows need for structured queries
- Estimated 1-2 days if needed
- Current text-based matching likely sufficient for 95% of cases

---

## Success Criteria

✅ "I love swimming in the sea" → captures location context
✅ "I don't like swimming in lakes" → separate fact from sea swimming
✅ No contradictory facts (love swimming + hate swimming)
✅ p95 extraction time < 10ms
✅ LLM uses actual context, stops hallucinating details

---

## Open Questions

1. **Should we update existing triples in DB?**
   - Option A: Leave old flat triples, new ones have context
   - Option B: Migration script to enhance old triples
   - **Decision**: Option A (simpler, forward-only)

2. **How to handle queries on enhanced triples?**
   - "Show all swimming facts" needs to match both "swimming in sea" and "swimming in lakes"
   - **Decision**: Phase 1 uses text matching, Phase 4 adds metadata

3. **Language-specific considerations?**
   - Some languages use cases instead of prepositions
   - **Decision**: Phase 1 focuses on English, extensible for others

---

## Related Work

- USGS Grammar-to-Graph (27 dep types) ← Foundation
- DSPy extraction (complex sentences) ← Fallback for phase 4
- Property graphs (RDF/Neo4j) ← Inspiration for phase 4
- spaCy dependency parsing ← Core tool
- Universal Dependencies ← Linguistic framework

---

**Status**: Ready for implementation (all architectural concerns resolved)
**Next Session**: Implement Phases 1-3 together (5-7 hours)
**Complexity**: Medium-High (extraction interface + hot index + aliasing + refinement handling)
**Impact**: High - solves 95% of context loss and ambiguity problems
**Risk**: Medium - requires interface changes across 7-10 call sites + careful alias management (~355 lines total)

**Critical Architectural Decisions**:
1. ✅ **Root vs Enriched separation** - root immutable, enriched built separately
   - `root = "car"` (canonical, never modified)
   - `enriched = "red car"` (built from root + modifiers)
   - `entities` contains roots, `triples` use enriched
   - `entity_index[root]` finds enriched edges
2. ✅ **NO storage layer changes** - keep MemoryStore.observe_edge() stable
3. ✅ **Aliasing in hot index layer** - dual registration at memory_hotpath.py:266-267, 850-851
4. ✅ **5-tuple extraction interface** - clean break, update all call sites
5. ✅ **Canonical form during extraction** - apply `_canon_entity_text()` to root immediately
6. ✅ **Deterministic modifier ordering** - `sorted(children, key=lambda t: t.i)`
7. ✅ **Truncation fallback** - prevent empty strings
8. ✅ **Slow-path warning** - gated behind debug level check
9. ✅ **Conditional amod** - avoid duplicate quality triples
10. ✅ **Rebuild heuristic** - `_extract_base_entity()` for prep vs compound patterns

**Downstream Impact Analysis**:
- **7-10 extraction call sites** identified and documented
- **9 entity extraction locations** requiring updates
- **2 hot index locations** for dual registration
- **NO test breakage risk** - all call sites updated together
