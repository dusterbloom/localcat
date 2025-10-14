# Spec: Negation Propagation to Governing Predicate (AUX → VERB)

## Summary
Fix regression where negation attached to auxiliaries (e.g., "do" in "don't like") fails to negate the actual governing predicate ("like"). Implement a robust, minimal-cost negation propagation and add unit tests that fail under current behavior and pass with the fix. TDD first.

## Context & Motivation
- In spaCy parses, the `neg` dependency frequently attaches to an auxiliary (`AUX`) rather than the lexical verb.
- Our extractors check the main predicate (`head`) when deciding to create triples. When we only mark the immediate `neg` head as negated, we miss negation on the governing verb.
- Example: "I don't like horror movies" → `neg` on `do`, predicate is `like`. We must mark both the AUX and the governing predicate as negated.

## Goals
- Correctly identify negated predicates when `neg` attaches to `AUX` by propagating the negation upward to the governing lexical head.
- Preserve copula cases (e.g., "isn't happy") and support AUX chains (e.g., "hasn't been eating").
- Maintain O(n) pass to collect negation with no measurable perf regression.

## Non‑Goals
- Expanding beyond `neg` to cover adverbials like `never` (advmod) — out of scope here.
- Language‑specific handling beyond current English usage.

## Implementation Requirements
1. Build negation map in a single doc pass:
   - For every token `t` with `t.dep_ == 'neg'`:
     - Mark `head = t.head` as negated (covers copulas like "isn't happy").
     - If `head.pos_ == 'AUX'`, walk up the head chain until the first non‑`AUX` governor and mark that token as negated as well.
       - Pseudocode:
         - `walk = head`
         - `while walk.pos_ == 'AUX' and walk.head != walk: walk = walk.head`
         - Mark `walk` (if different from `head`).
   - Keep `neg_count = len(negated_tokens)` for backward compatibility.
2. Ensure all extraction handlers skip emitting triples when `_is_negated(head)` is true (already the intended pattern).
3. Keep current debug logging behavior for skipped negated predicates.

## Files to Modify
- `server/core/memory/memory_hotpath.py`
  - Stage 0 negation map construction.
  - No API changes; keep `_is_negated` signature and usage.

## Tests (TDD)
Add or update unit tests under `server/tests/unit/` to cover the following. Write tests first; confirm they fail before implementing.

1) Core cases
- "I'm not interested in classic cars" → must NOT store any edge containing "interested in classic".
- "I don't like horror movies" → must NOT store any edge containing "like horror".

2) AUX chain
- "He hasn't been eating meat" → must NOT store any edge with predicate equivalent to `eat` and object `meat`.

3) Mixed polarity
- "I like pizza but not pineapple" → should store the positive edge about `pizza` but NOT any edge about `pineapple`.

4) Positive control
- "I like science fiction" → should store the corresponding positive edge.

Implementation notes for tests:
- Use the existing `HotPathMemoryProcessor` or `HotMemory` extraction through the processing pathway (as done in `test_memory_system.py::test_negation_handling`).
- Mirror the assertion style already used (string containment on `src rel dst`).
- Keep tests fast; avoid external I/O beyond what fixtures already do.

## Acceptance Criteria
- All new tests pass and previously failing negation tests pass:
  - `server/tests/unit/test_memory_system.py::test_negation_handling`
  - New dedicated tests for AUX propagation, AUX chains, and mixed polarity.
- No regressions on existing unit and integration tests.
- Negation mapping performed in a single pass with negligible overhead.

## Validation Steps
- Run targeted unit tests:
  - `pytest -q server/tests/unit/test_memory_system.py::test_negation_handling`
  - `pytest -q server/tests/unit` (for the new tests)
- Optionally run a small integration subset touching memory extraction.

## Rollback Plan
- If propagation causes unintended false negatives, revert to marking only the immediate `neg` head and open a follow‑up spec to handle specific constructions.

## Notes
- Keep logging at debug level for negation skips to aid diagnosis without polluting normal logs.
- Do not alter `neg_count` semantics beyond counting the unique negated heads.
