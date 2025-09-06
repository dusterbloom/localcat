# TOEXPLORE: LEANN Research Notes (Phase 1.5)

## Goal
Determine whether a GitHub-accessible “LEANN” vector/index library fits our HotMem constraints for semantic retrieval:
- Sub-50ms similarity queries on CPU
- Tiny on-disk footprint (~1–2MB at our scale)
- Pure-Python or simple native wheels (no complex toolchain)
- Offline-friendly; deterministic behavior; permissive license

## What We Need To Verify (on GitHub)
- Repository link, license, and release cadence
- Install method: pip/PyPI availability, binary wheels for macOS (ARM64/x86)
- Index type and complexity control (HNSW/graph-based? search params akin to `efSearch`/`M`)
- Persistence: on-disk save/load API; index size vs number of facts
- Python API surface:
  - add_document(text, metadata) or add_vector(vec, id)
  - search(text_or_vec, k) → [(id/metadata, score)]
  - optional batch APIs, deletes/updates
- Benchmarks (own or third-party): latency, recall@k for small indexes

## Integration Plan (if LEANN is viable)
- Optional dependency behind `HOTMEM_USE_LEANN=true`
- Initialize index at `data/memory_vectors.leann` with complexity=`HOTMEM_LEANN_COMPLEXITY` (default 16)
- On storage: encode fact text `"{src} {rel} {dst}"`, add with metadata `{src,rel,dst,weight,ts}`
- On retrieval: compute semantic score `sim_leann` and replace `lexical_overlap` in scoring
- Metrics: log enablement, similarity latency p95, and fallback counts
- A/B switch via env to compare semantic vs lexical

## Fallback Options (if LEANN is unavailable)
- hnswlib (HNSW index)
  - Pros: very fast CPU ANN, small footprint, simple save/load
  - Cons: vectorization needed (we’d reuse tiny sentence embeddings or char n-gram TF-IDF → centroids)
- sqlite-vec / sqlite-vss
  - Pros: SQLite extension for vectors; easy persistence; good for small corpora
  - Cons: need to bundle extension; perf varies
- Annoy / NGT
  - Pros: simple + stable; read-optimized index
  - Cons: tradeoffs in recall/perf vs HNSW

## Acceptance Criteria
- Queries ≤ 50ms (p95) for ≤ 10k facts on our hardware
- Index file ≤ 2MB at our expected scale
- API integrates with our scoring pipeline with minimal glue
- No network calls at runtime; deterministic save/load

## GitHub Research Plan (when network is enabled)
- Search queries:
  - "LEANN vector index", "LEANN ANN", "LEANN graph nearest neighbor", "Lightweight ANN LEANN"
  - Filter language: Python/C++; filter by recent commits
  - Look for documentation, examples, benchmarks; issues indicating production usage
- Cross-check PyPI:
  - `pip search leann` or PyPI web search (confirm availability)
- Shortlist + evaluate:
  - Clone repos; run microbenchmarks (build small index; query K=3; measure p95)
  - Check save/load APIs and file size growth vs facts

## Notes & Assumptions
- If “LEANN” is not a public library but an internal concept, we’ll implement an internal semantic index adapter with hnswlib as the first backend and keep the LEANN interface stable for later swaps.
- Scoring weights will shift to emphasize semantic similarity (e.g., α,β,γ,δ = 0.3, 0.2, 0.4, 0.1) once semantic is enabled.

---

This file outlines what we’ll verify once GitHub access is granted and provides concrete fallbacks to avoid blocking progress.

