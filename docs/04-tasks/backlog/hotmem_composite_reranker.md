# Spec: HotMem Composite Reranker (Confidence + Recency + Usage + Optional Embeddings)

Context
- Recent changes landed in HotMem and retrieval:
  - Enhanced FTS + BM25 and query expansion (5b3f12f).
  - Pronoun role mapping and identity/provenance scoping (77beb4a, 31909d9).
  - Confidence/recency are applied for graph in `retrieval.py` (6523f9a), but cross‑source re‑ranking still relies on simple index order and source bias.
- Logs show low-quality conversation bullets surfacing (e.g., interjections like “Oh my god.”), and final selection often forces one bullet per source regardless of relevance.
- Goal: Improve retrieval precision by re‑ranking all candidates with a unified score that blends confidence, recency, usage, and (optionally) semantic similarity, while keeping HotMem’s safe injection/summarization and latency budget.

Goals
- Add a composite re‑ranker that scores candidates from graph, convo, and summary with consistent factors.
- Track per‑edge usage (access_count, last_accessed) and feed it into ranking.
- Use Enhanced FTS scores instead of list order for conversation candidates.
- Optional embedding reranker (MiniLM) to improve fuzzy matching; strictly optional via env and optional requirements.
- Tighten convo bullet quality filter to drop non‑informative interjections/short utterances.

Non‑Goals
- No changes to extraction rules or schemas that break LMDB sync.
- No replacement of HotMem with a new memory engine.
- No base requirements changes; ML deps remain optional.

Acceptance Criteria
- With embeddings OFF (default):
  - Final ranking uses BM25/recency for convo, confidence/recency for graph, and removes obvious interjections.
  - Graph facts with higher weight/support and recent timestamps outrank older/weak facts, all else equal.
  - Selected graph bullets increment usage counts and influence subsequent ordering.
- With embeddings ON:
  - For fuzzy queries, semantically closer candidates are promoted vs. unrelated chatter.
  - Latency overhead for reranking stays under ~10–15ms for ≤24 candidates after warmup.
- No import errors when optional ML deps are absent; behavior matches embeddings OFF.

Deliverables
1) Retrieval: Composite Re‑ranker
   - File: `server/core/memory/retrieval.py`
   - Changes:
     - Preserve Enhanced FTS score in `_convo_retrieve` and include it in candidate metadata instead of discarding it.
     - Build a `Candidate` record for each source with fields: `{text, source, score_hint, ts, meta}` where:
       - graph.meta: `{edge_id, weight, pos, neg}`; score_hint unused (we compute from meta).
       - convo.meta: `{bm25_score}`; score_hint=bm25_score.
       - summary.meta: `{}`; score_hint optional.
     - Add `_composite_score(query, candidate)` that combines:
       - `wsrc` (source bias from `_get_source_priority`),
       - `wconf` (graph: normalized from weight + support (pos>neg); convo/summary: from score_hint or a constant prior),
       - `wrec` (recency decay using `RECENCY_HALF_LIFE_MS`),
       - `wuse` (usage boost from new usage table; small capped boost),
       - `wsim` (semantic sim; only if enabled and deps available).
     - Replace the current index-based cross‑source scoring with the composite.
     - Keep existing source budgeting and dedup logic; only change how candidates are scored and ordered.

2) Usage Tracking (Graph)
   - File: `server/core/memory/memory_store.py`
   - Changes:
     - Schema: Create table if not exists
       ```sql
       CREATE TABLE IF NOT EXISTS edge_usage(
         edge_id TEXT PRIMARY KEY,
         access_count INT DEFAULT 0,
         last_accessed INT DEFAULT 0
       );
       ```
     - Methods:
       - `increment_edge_usage(edge_id: str, ts_ms: int)` → UPSERT (count += 1, last_accessed = ts_ms).
       - `get_edge_usage(edge_id: str) -> (count: int, last_accessed: int)` with sensible defaults.
   - Hook:
     - After final selection in `Retrieval.retrieve()`, for `[graph]` bullets, resolve `edge_id` from candidate.meta and call `increment_edge_usage`.

3) Optional Embedding Reranker
   - New file: `server/core/memory/rerank_embeddings.py`
     - `EmbeddingReranker` with lazy init; backend SentenceTransformers (`all-MiniLM-L6-v2` by default).
     - `similarity(query: str, texts: List[str]) -> List[float]` returning cosine/IP scores; returns zeros if deps missing or disabled.
   - Integration:
     - In `Retrieval.retrieve()`, if `MEMORY_RERANK_EMBEDDINGS_ENABLED=true`:
       - Compute query embedding once, score the top‑N candidates (configurable) via reranker, and feed as `wsim`.
   - Config (env):
     - `MEMORY_RERANK_EMBEDDINGS_ENABLED=false`
     - `MEMORY_RERANK_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2`
     - `MEMORY_RERANK_MAX_CANDIDATES=24`
   - Requirements: Add encoder to `server/requirements-ml.txt` only; do not change base `requirements.txt`.

4) Convo Bullet Quality Filter
   - File: `server/core/memory/retrieval.py`
   - Tighten `_is_quality_bullet()`:
     - Drop common interjections/fillers unless followed by substantive content: `oh`, `wow`, `lol`, `yeah`, `hmm`, `uh`, `ok`, `okay`, `right`, `sure`, `thanks`.
     - Require at least one content token (heuristic: length ≥ 15 or contains a content verb/noun pattern); keep conservative to avoid regressions.

5) Configuration & Defaults
   - No change to `.env` needed unless enabling embeddings.
   - Maintain current `MEMORY_SOURCES` behavior; this spec only affects scoring/ranking.

6) Tests (TDD)
   - Unit tests under `server/tests/unit/memory_reranker/`:
     - `test_convo_preserves_bm25_ordering_when_relevant()`:
       - Simulate two convo hits with different BM25; confirm higher BM25 ranks above when other terms equal.
     - `test_graph_usage_boosts_frequently_used_facts()`:
       - Start with two graph facts close in weight/recency; after selecting A multiple times, A ranks above B.
     - `test_embeddings_disabled_behaves_like_baseline()`:
       - With embeddings off, ranking decisions ignore `wsim` and match composite score without sim.
     - `test_embeddings_enabled_promotes_semantically_close()` (importorskip `sentence_transformers`):
       - For a fuzzy query, a semantically close convo candidate outranks a filler candidate with similar recency.
     - `test_quality_filter_rejects_interjections()`:
       - Ensure `_is_quality_bullet("Oh my god.")` is False; a longer assertive sentence passes.
   - Integration tests under `server/tests/integration/`:
     - `test_final_selection_updates_edge_usage()`:
       - Run a retrieval cycle that selects a known graph bullet; verify `edge_usage` updated.
     - `test_latency_budget_with_embeddings()` (skipped on CI if slow):
       - Ensure rerank stays under the configured budget for ≤24 candidates after warmup.

7) Telemetry & Logging
   - Log per-candidate composite components (`wsrc/wsim/wconf/wrec/wuse`) for the final top‑k (debug level), with elapsed rerank time.
   - Emit source distribution counts (existing).

Migration/Compatibility
- Usage table is additive; no impact on existing schema or LMDB sync.
- With embeddings disabled (default), behavior is a strict improvement over baseline with no new deps.
- If `sentence_transformers` is missing, reranker becomes a no‑op and logs a single warning once.

Implementation Notes
- Use `RECENCY_HALF_LIFE_MS` from `memory_constants` for consistent decay.
- Normalize BM25 to [0,1] with a simple min/max clamp over current candidate pool; don’t persist normalization state.
- Keep weights readable and configurable via env (optional): `MEMORY_RERANK_WEIGHTS="{\"wsrc\":0.1,\"wconf\":0.35,\"wrec\":0.25,\"wuse\":0.1,\"wsim\":0.2}"` with safe defaults hard‑coded.
- Cap `wuse` boost (e.g., `log1p(count)` then scale) to avoid runaway popularity bias.
- Compute embeddings only for the short‑listed candidate pool to keep latency bounded; cache model instance.

TDD Plan
1) Write unit tests for scoring behavior and filters.
2) Implement usage table + store methods, wire usage updates on selection.
3) Preserve enhanced FTS score and feed it into composite scoring.
4) Add reranker module with lazy import and env gating; integrate.
5) Tighten convo bullet filter.
6) Run tests, profile rerank latency locally.

Commands (for Droid Exec)
```bash
# Unit
pytest server/tests/unit/memory_reranker -q

# Integration
pytest server/tests/integration -k rerank -q

# Optional performance (local only)
pytest server/tests/performance -k rerank -q
```

Owner
- Memory Systems Specialist (via Droid Exec)

Risk & Rollback
- If ranking regressions appear, disable embeddings via env and revert to prior re‑rank ordering behind a feature flag.
- The usage table is additive; can be left unused without side effects.

