# Spec: Coreference Upgrade (Latency-Aware, Gated)

Context
- `HotMemory` has scaffolding and heuristics for pronouns; a `CoreferenceProcessor` exists but is not fully integrated.
- Proper coreference resolution improves extraction fidelity (fewer ambiguous edges), especially for first/second/third person references.

Goals
- Integrate `CoreferenceProcessor` into the extraction seam with a tight timeout and fallbacks.
- Apply only when text complexity warrants it to protect latency budgets.
- Cache resolved docs to avoid duplicate work on final/interim phases.

Non‑Goals
- Do not introduce heavy models or block the hot path beyond configured timeout.
- Do not change graph/FTS algorithms; only improve upstream text quality.

Deliverables
1) Wire‑up in `server/core/memory/memory_hotpath.py`:
   - Before extraction, if enabled and `complexity_detector` indicates, run `CoreferenceProcessor.process(doc)` with timeout.
   - If it times out or fails, proceed with original doc.
   - Cache resolved doc keyed by (text hash, turn id) to reuse across phases.

2) Configuration:
   - `COREFERENCE_ENABLED=true|false` (default false if not already controlled by `MemoryConfig`).
   - `COREFERENCE_TIMEOUT_MS` (default 50ms) and `COREFERENCE_MIN_TEXT_LENGTH`.

3) Observability:
   - Record metrics: attempted, succeeded, timed out, average added latency.

Acceptance Criteria
- On pronoun‑heavy inputs, extraction produces more correct edges (fewer ambiguous entities) vs baseline.
- Added latency remains within configured timeout; hard stop on exceeding timeout.
- When disabled, behavior unchanged.

TDD Plan
- Unit tests:
  - `test_coref_improves_edges_on_pronoun_text()` (use synthetic doc where resolution is clear).
  - `test_coref_timeout_fallback_preserves_latency()` (mock long processing to hit timeout path).
- Integration test:
  - `test_end_to_end_extraction_with_coref_enabled()` verifies improved bullets without exceeding budget.

Commands (to be run by Droid Exec)
```bash
pytest server/tests/unit -k coreference -q
pytest server/tests/integration -k coreference -q
```

Owner
- Memory Systems Specialist (via Droid Exec)

