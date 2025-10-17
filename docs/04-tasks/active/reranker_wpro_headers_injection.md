# Spec: Prosody‑Weighted Re‑ranking (wpro) + Headers‑First Injection (REFRAG‑style)

Status: Draft (TDD‑first)
Owner: Memory/Retrieval
Goal: Improve retrieval precision and reduce context tokens by (a) adding a prosody term to re‑ranking for conversation items, and (b) switching to compact, typed headers for injection with automatic expansion when needed.

## Context
- Retrieval already computes composite scores (confidence, recency, usage, semantic). Prosody certainty exists in the audio pipeline but is not used in ranking.
- Injection currently emits full bullets; REFRAG suggests “compress → sense → expand”. We can mirror this with headers (compressed signals) and expand only for weak items.
- All changes must be env‑gated and safe; defaults keep current behavior.

## Objectives
1) Add `wpro` term (prosody certainty) to re‑ranking for `convo` items.
2) Add `headers` injection mode that emits compact, typed headers with scalar signals and auto‑expands weak items.
3) Provide component logging for top‑k selections (debug only) to validate ranking behavior.

## Non‑Goals
- No change to graph scoring formula beyond new signals surfaced in headers.
- No new DB schema; leverage existing store APIs. (If `turn_id` is unavailable for a given candidate, skip `wpro` gracefully.)
- No LLM tool integration in this phase (auto‑expand policy only).

## Env Flags (all optional)
- `MEMORY_WEIGHT_PROSODY` (default: `0.0`) — weight for `wpro` contribution (0 disables effect)
- `MEMORY_INJECTION_MODE` (default: `bullets`) — set `headers` to enable header injection
- `MEMORY_HEADER_EXPAND_THRESHOLD` (default: `0.65`) — expand header to full text when combined score below this threshold
- `MEMORY_LOG_COMPONENTS` (default: `false`) — log per‑candidate component scores for top‑k

## Deliverables (Code + Tests)
A) Re‑ranker: `wpro` for conversation candidates
B) Injection: headers‑first formatting with auto‑expand
C) Debug: optional component logging for top‑k
D) Tests: unit and integration to lock behavior

## Implementation Plan (file‑mapped)

### A) Re‑ranker: add `wpro`
- File: `server/core/memory/retrieval.py`
- In `_composite_score(query, candidate, ...)`:
  - If `candidate.source == "convo"`:
    - Ensure `candidate.meta` carries `turn_id` (if possible). If not available from FTS, leave `wpro=0.0`.
    - Obtain `session_id = getattr(self.host, 'current_session_id', None)`.
    - If both `session_id` and `turn_id` present:
      - `certainty, _ = self.host.store.get_turn_prosody(session_id, turn_id)`
      - `weight = float(os.getenv("MEMORY_WEIGHT_PROSODY", "0.0"))`
      - `components["wpro"] = max(0.0, min(1.0, certainty)) * weight`
    - Else: `components["wpro"] = 0.0`
  - For non‑`convo` sources: do not add `wpro` (keeps risk minimal).
- Performance: maintain a small per‑call cache dict keyed by `(session_id, turn_id)` to avoid repeated store reads.

Tests (new): `server/tests/unit/retrieval/test_reranker_wpro.py`
- Setup mock host + store. Seed two `convo` candidates with similar base scores and different stored certainties (e.g., 0.2 vs 0.9). With `MEMORY_WEIGHT_PROSODY=0.2`, the higher certainty ranks first.
- With `MEMORY_WEIGHT_PROSODY=0.0`, ordering unaffected.
- Missing `turn_id` → `wpro=0`, ranks fall back to baseline.

### B) Injection: headers‑first with auto‑expand
- File: `server/core/memory/retrieval.py`
- In `_apply_token_budget_and_deduplication(...)` when formatting final selections:
  - Check mode: `mode = os.getenv("MEMORY_INJECTION_MODE", "bullets").lower()`.
  - If `mode == "headers"`:
    - For `graph` candidates:
      - Emit typed header: `"{rel_display}: {dst_display} [conf=.xx rec=.yy use=n]"`.
      - Use existing per‑candidate scalars (confidence component for graph, recency factor, usage count). Omit long text.
    - For `convo` candidates:
      - Emit: `"convo: {short_gist} [bm25=.xx pro=.yy]"` where `short_gist` is a reduced truncation (e.g., 60 chars) to keep headers small. Include `pro` when available.
    - Auto‑expand rule:
      - Compute combined score as the sum of `components` for the candidate (already computed in the scoring loop).
      - If score < `float(os.getenv("MEMORY_HEADER_EXPAND_THRESHOLD", "0.65"))`, append the full original text after the header (for that candidate only).
    - Respect the same token budget constraints; headers reduce tokens in the common case.
  - Else: preserve legacy bullets unchanged.

Tests (new): `server/tests/integration/test_headers_injection_mode.py`
- With `MEMORY_INJECTION_MODE=headers`:
  - High‑score candidate: header only (no expansion)
  - Low‑score candidate: header + expanded text
- With `MEMORY_INJECTION_MODE=bullets`: outputs match legacy formatting

### C) Component logging (debug)
- File: `server/core/memory/retrieval.py`
- After computing `components` for each candidate in scoring:
  - If `MEMORY_LOG_COMPONENTS` is true and candidate is among the top‑k (e.g., first 3 after sorting), log a compact line: `components={wsrc:.2f,wconf:.2f,wrec:.2f,wuse:.2f,wsim:.2f,wdiv:-.2f,wpro:.2f}`.
- Tests: none required; guarded by env and log level.

## Safety & Backward Compatibility
- `MEMORY_WEIGHT_PROSODY` defaults to 0 → no behavior change.
- `MEMORY_INJECTION_MODE` defaults to `bullets` → identical output to current behavior.
- Missing `turn_id` in `convo` meta → `wpro=0` without error.

## Acceptance Criteria
- Unit tests demonstrate `wpro` affects ordering only when `MEMORY_WEIGHT_PROSODY>0` and `(session_id, turn_id)` is available.
- Integration tests show header outputs and auto‑expand only in `headers` mode; legacy mode unchanged.
- No measurable increase in per‑query latency beyond a single store lookup per distinct `(session_id, turn_id)` (cached within call).

## Implementation Order
1) Add `wpro` in `_composite_score` + unit tests
2) Add `headers` mode formatting + integration tests
3) Add optional component logging (debug only)
4) README: document new envs and the headers mode behavior

## Notes for Droid Exec (TDD)
- Use mocks for store prosody calls in re‑ranker unit tests.
- For headers tests, construct a minimal `Retrieval` host stub and verify formatted outputs under both modes.
- Keep changes narrowly scoped to `retrieval.py`; avoid altering unrelated code paths.
