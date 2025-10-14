# Spec: Prosody‑Aware Confidence + Turn Prosody Meta + Headers‑First Injection (Phase A)

Status: Draft (TDD-first)
Owner: Memory/Audio
Goal: Integrate prosody into extraction confidence and retrieval/injection via a safe, env‑gated, REFRAG‑style “compress → sense → expand” flow.

## Context
LocalCat already has:
- Prosody feature extraction (Parselmouth) and multi‑signal confidence fusion utilities.
- Pluggable confidence strategies and a modern retrieval pipeline.
- A desire to reduce token load and TTFT by injecting compact, high‑utility memory.

Today, prosody signals are not persisted per turn or consistently consumed by memory confidence/ranking. Injection always emits full bullets.

## Objectives
- Persist per‑turn prosody certainty/metadata (lightweight) for downstream use.
- Enable prosody‑aware confidence by default via factory/ENV, with safe fallback.
- Add a prosody term (wpro) to retrieval re‑ranking for `convo` items.
- Add headers‑first injection (compressed) with automatic expansion only when needed.
- Optionally bias background summarization to prefer high‑certainty turns.

## Non‑Goals
- No heavy model changes, no blocking calls in hot path.
- No semantic sidecar changes beyond optional confidence gating (deferred).
- No LLM tool integration in this phase (auto‑expand policy only).

## Deliverables (Code + Tests)
1) Turn prosody meta storage in `MemoryStore` (SQLite)
2) Prosody‑aware confidence strategy uses stored prosody if not provided inline
3) Retrieval re‑ranker adds `wpro` component for `convo` candidates (env‑gated weight)
4) Headers‑first injection + auto‑expand threshold (env‑gated)
5) Summarizer prosody bias (env‑gated, optional)
6) TDD: unit + integration tests for each deliverable
7) Docs: new env flags and behavior

## Env Flags (defaults conservative)
- `CONFIDENCE_STRATEGY` (default: `relation_type`) → set to `prosody_aware` to enable voice‑aware confidence
- `MEMORY_WEIGHT_PROSODY` (default: `0.0`) → weight for `wpro` component in retrieval
- `MEMORY_INJECTION_MODE` (default: `bullets`) → `headers` to enable compressed injection
- `MEMORY_HEADER_EXPAND_THRESHOLD` (default: `0.65`) → expand a header to full text if combined score below
- `SUMMARY_PROSODY_ENABLED` (default: `false`) → bias background summarizer by prosody

## Implementation Plan (Surgical, File‑mapped)

### A. Turn Prosody Meta (SQLite)
- File: `server/core/memory/memory_store.py`
- Schema: On init, create if not exists:
  - `turn_meta(session_id TEXT, turn_id INT, key TEXT, value TEXT, PRIMARY KEY(session_id, turn_id, key))`
- Methods:
  - `def set_turn_prosody(self, session_id: str, turn_id: int, certainty: float, meta: Optional[dict] = None) -> None`
    - Store `("prosody_certainty", f"{certainty:.3f}")`
    - If `meta` provided, store as JSON under key `prosody_meta` (optional)
  - `def get_turn_prosody(self, session_id: str, turn_id: int) -> Tuple[float, dict]`
    - Returns `(certainty, meta_dict)` with defaults `(0.5, {})` if missing or parse failure
- Notes:
  - Keep writes lightweight; do not block hot path. This API is used opportunistically.

Tests (new): `server/tests/unit/test_turn_meta.py`
- Writes and reads prosody certainty and meta
- Missing values return defaults
- JSON meta robust to invalid payloads (does not raise)

### B. Prosody‑Aware Confidence (fallback to store)
- Files:
  - `server/core/memory/confidence_strategy.py`
- Change: In `ProsodyAwareConfidence.score(...)`
  - If `context.prosody_features` is None and both `context.session_id` and `context.turn_id` exist, then:
    - call `context.store.get_turn_prosody(session_id, turn_id)`
    - Map `certainty` to a synthetic `ProsodyFeatures` with `certainty_modifier = clamp(certainty - 0.5, -0.3, +0.3)` (other fields can be neutral defaults)
  - Continue existing fusion + usage multipliers; preserve fallbacks
- Factory already supports `CONFIDENCE_STRATEGY=prosody_aware`

Tests (new): `server/tests/unit/test_prosody_confidence.py`
- Given low stored certainty, confidence < baseline
- Given high stored certainty, confidence > baseline
- Missing store/meta → falls back to baseline without error

### C. Retrieval: prosody term (wpro) for convo
- File: `server/core/memory/retrieval.py`
- In `_composite_score(...)`:
  - For `candidate.source == "convo"` only (safe initial scope):
    - Ensure `candidate.meta` contains `turn_id` (if missing from FTS path, carry it through where hits are assembled)
    - Get `session_id = getattr(self.host, 'current_session_id', None)`
    - If both present, call `self.host.store.get_turn_prosody(session_id, turn_id)`
    - Compute `wpro = certainty * weight` where `weight = float(os.getenv("MEMORY_WEIGHT_PROSODY", "0.0"))`
    - `components["wpro"] = wpro`
  - If missing certainty or ids, set `components["wpro"] = 0.0`
- Weights: make prosody weight independently env‑controlled; do not fold into existing `MEMORY_RERANK_WEIGHTS` bundle to minimize risk.
- Cache: maintain a small dict cache per `retrieve()` call keyed by `(session_id, turn_id)` to avoid repeated store hits.

Tests (new): `server/tests/unit/retrieval/test_prosody_rerank.py`
- Two convo candidates with similar BM25; set stored certainties 0.2 vs 0.9 → higher certainty ranks first when `MEMORY_WEIGHT_PROSODY>0`
- With `MEMORY_WEIGHT_PROSODY=0.0`, ordering unaffected

### D. Headers‑First Injection (+ auto‑expand threshold)
- File: `server/core/memory/retrieval.py`
- In `_apply_token_budget_and_deduplication(...)`, when formatting final bullets:
  - If `os.getenv("MEMORY_INJECTION_MODE", "bullets").lower() == "headers"`:
    - For `graph` candidate: emit header `"name: Alice [conf=.92 rec=.85 use=2]"` (relation‑typed header). For verbs `v:like`, emit `"like: hiking [..]"`.
    - For `convo` candidate: emit `"convo: <short gist> [bm25=.X pro=.YY]"` (no heavy text; the gist is the existing smart‑truncated snippet capped smaller than today, e.g. 60 chars).
    - Pull small scalars already computed: confidence (graph), recency factor, usage count, and prosody certainty (if available) into the bracket block.
    - Auto‑expand rule: compute the candidate’s combined score (sum of components); if below `float(os.getenv("MEMORY_HEADER_EXPAND_THRESHOLD", "0.65"))`, append the full text after the header (for that candidate only). Others remain headers.
  - Else (legacy): keep existing bullet formatting unchanged.
- Keep token budget checks identical; headers usually reduce tokens.

Tests (new): `server/tests/integration/test_headers_injection.py`
- With headers mode ON: outputs are headers, not full bullets
- A low‑score candidate expands; a high‑score one does not
- With headers mode OFF: identical to legacy behavior

### E. Summarizer Prosody Bias (optional)
- File: `server/core/memory/background_summarizer.py`
- When `SUMMARY_PROSODY_ENABLED=true`, prefer high‑certainty turns when selecting/ordering recent messages to summarize:
  - If the helper that fetches recent turns exposes turn ids, fetch `certainty = store.get_turn_prosody(session_id, turn_id)` and bias selection or drop very low certainty chatter.
  - If turn ids are unavailable in a path, skip biasing for those entries (graceful degrade).

Tests (new): `server/tests/unit/test_summarizer_prosody_bias.py`
- Given two candidate turns with different certainties, the higher certainty is preferred for summarization input

### F. (Optional, later) Frame Processor Prosody Capture
- File: `server/core/memory/frame_processor.py`
- Maintain `self._last_prosody_certainty: Optional[float]`
- On receiving a lightweight meta frame (or observing `AudioIntelligenceFrame` if accessible), store certainty temporarily
- On handling a `TranscriptionFrame` (final), call `store.set_turn_prosody(session_id, turn_id, certainty)` and clear
- TDD: `server/tests/integration/test_prosody_capture.py` simulating frame order to ensure persistence

## Rollout & Backward Compatibility
- All features are env‑gated; defaults keep current behavior:
  - Confidence stays baseline unless `CONFIDENCE_STRATEGY=prosody_aware`
  - Reranker prosody weight is 0.0 by default (no effect)
  - Injection defaults to legacy bullets
  - Summarizer bias defaults off
- Missing prosody/meta always falls back safely (defaults and zeros)

## Risks & Mitigations
- Extra store calls in retrieval: mitigate with per‑retrieve cache by `(session_id, turn_id)` and only for `convo` source.
- Formatting regressions: keep legacy path intact unless headers mode enabled; tests cover both modes.
- Prosody dependency: tests skip or mock when `parselmouth` unavailable; strategy falls back gracefully.

## Acceptance Criteria
- All new tests pass locally; existing tests unaffected with defaults.
- With `CONFIDENCE_STRATEGY=prosody_aware`, confidence increases for emphatic turns and decreases for hesitant/question‑like turns.
- With `MEMORY_WEIGHT_PROSODY>0`, prosody influences ordering for `convo` candidates.
- With `MEMORY_INJECTION_MODE=headers`, returned items are compact headers; auto‑expand triggers per threshold.
- With `SUMMARY_PROSODY_ENABLED=true`, summarizer favors high‑certainty turns when possible.

## Implementation Order (recommended)
1) Turn meta table + set/get + unit tests
2) Prosody‑aware confidence fallback + unit tests
3) Reranker `wpro` for `convo` + unit tests
4) Headers‑first injection + integration tests
5) Summarizer bias + unit tests
6) (Optional) Frame processor prosody capture + integration test
7) README doc updates (envs, behavior)

## Notes for Droid Exec
- TDD first. Add tests per sections above; skip or mock heavy deps.
- Keep changes minimal and surgically localized to the files listed.
- Do not alter unrelated behavior or defaults.
- Use small internal caches where store calls would be repeated within a single retrieval.
