# Spec: Semantic Memory Sidecar (Optional, Production-Safe)

Context
- HotMem today relies on graph + enhanced FTS + summaries. This is strong for factual recall and recent context but can miss fuzzy/semantic matches.
- We want an optional semantic sidecar that augments (not replaces) existing sources while preserving privacy, stability, and latency guarantees.

Goals
- Add a semantic retrieval source backed by a tiny sentence-embedding model and FAISS with stable 64-bit IDs.
- Keep it strictly optional and gated by env; boot cleanly without ML deps.
- Persist the semantic index on disk; support user/session/type namespacing.
- Integrate into existing re-ranker and token budget rules; return ≤2 compact bullets by default.

Non‑Goals
- Do not change the existing graph/FTS behavior or thresholds unless part of fusion logic.
- Do not ship a heavy model or force GPU; favor small, CPU/MPS-friendly encoders.

Deliverables
1) Module: `server/core/memory/semantic_sidecar.py`
   - `SemanticMemorySidecar` with:
     - FAISS `IndexIDMap2(IndexFlatIP)`; L2-normalized vectors; `add_with_ids`.
     - Stable 64‑bit IDs (xxhash64 or sequential) and metadata map `{id: {text, ts, user_id, session_id, kind}}`.
     - Duplicate suppression on ingest via similarity threshold.
     - `save(path)`/`load(path)` using `faiss.write_index/read_index` + JSON metadata.
     - `recall(query, k, scopes, token_budget)` returning texts + scores.
   - Optional embedding backends (env‑selectable):
     - SentenceTransformers (default): `all-MiniLM-L6-v2` or similar 256–384d model.
     - Fallback no‑op that disables semantic if deps missing.
   - Namespacing filters in recall: `user_id`, `session_id`, `kind` in metadata.

2) Integration:
   - Retrieval fusion: add a new source `semantic` to `server/core/memory/retrieval.py` and feed results into the existing cross‑source re‑ranker.
   - Ingestion hooks:
     - Finalized conversation turns (short strings)
     - Summaries (dense, short)
     - High‑confidence graph facts (humanized bullets)
   - Persistence location: `data/semantic_index/` with `index.faiss` and `metadata.json`.

3) Configuration (optionalized):
   - `MEMORY_SEMANTIC_ENABLED=true|false` (default false)
   - `MEMORY_SEMANTIC_EMBED_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`)
   - `MEMORY_SEMANTIC_DIR` (default `data/semantic_index`)
   - `MEMORY_SOURCES` gains `semantic` (off by default; opt‑in via env)
   - Respect existing optional‑ML pattern (see `tasks/optionalize_ml_deps.md`).

4) Performance/Latency targets:
   - Recall p95 ≤ 20ms at 10k vectors on CPU; degrade gracefully if disabled.
   - Ingestion amortized (batchable) and off the hot path for final turns.

Acceptance Criteria
- With semantic disabled: no import errors; behavior identical to current.
- With semantic enabled: top‑k recall returns sensible semantic matches for fuzzier queries missed by FTS/graph.
- Save → load produces identical top‑k for a fixed query and corpus (within tie‑break).
- Duplicate suppression prevents storing near‑identical entries.
- Retrieval fusion obeys token budget and returns ≤2 concise semantic bullets when selected.

TDD Plan
- Unit tests (`server/tests/semantic/`):
  - `test_ingest_persist_reload_keeps_ids_and_order()`
  - `test_duplicate_suppression_by_threshold()`
  - `test_recall_scoped_by_user_session_kind()`
  - `test_disabled_sidecar_noop_without_ml_deps()` (importorskip sentence_transformers)
- Integration tests (`server/tests/integration/`):
  - `test_fusion_includes_semantic_when_enabled_and_fts_misses()`
  - `test_token_budget_enforced_with_semantic_results()`
- Performance tests (`server/tests/performance/`):
  - `test_semantic_recall_p95_under_budget()` for a synthetic set (skip on CI if slow).

Implementation Notes
- Metric: cosine similarity implemented as IP over L2‑normalized vectors.
- Use `IndexIDMap2` to avoid manual positional mapping and enable stable persistence.
- If FAISS GPU is detected and enabled, convert GPU index to CPU before saving; restore on load.
- Keep sentence-transformers optional; if missing, log and disable semantic gracefully.
- In fusion, score semantic bullets with a reasonable source weight (slightly below convo FTS, above summaries) and let composite re‑ranker decide.

Commands (to be run by Droid Exec)
```bash
pytest server/tests/semantic -q
pytest server/tests/integration -k semantic -q
pytest server/tests/performance -k semantic -q
```

Owner
- Memory Systems Specialist (via Droid Exec)

