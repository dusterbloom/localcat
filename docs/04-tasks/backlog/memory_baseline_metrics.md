# Spec: Baseline Metrics for HotMem (Extraction/Retrieval/Injection)

Context
- We need stable, automated measurements for extraction, retrieval, and injection latencies to protect the <200ms p95 hot path target.
- Tests should run without heavy ML deps and skip gracefully when optional features are disabled.

Goals
- Add performance tests capturing p50/p90/p95 for extraction and retrieval.
- Add correctness sanity tests for recall ordering and header/injection integrity.
- Provide a minimal benchmarking helper callable from tests.

Non‑Goals
- No algorithmic changes; purely measurement and small helpers.

Deliverables
1) Tests under `server/tests/performance/`:
   - `test_extraction_latency_baseline()`
     - Run extraction on a fixed set of sentences; record latencies; assert they are recorded and printed.
     - Set soft expectations (no strict fail thresholds initially; use warnings if over targets).
   - `test_retrieval_latency_baseline()`
     - Seed a small store and entity index; measure recall latency (graph + FTS path).
   - `test_injection_budget_baseline()`
     - Ensure injection runs under a small budget and enforces bullet/token caps.

2) Helper: `server/core/memory/metrics_helper.py`
   - Simple utilities to time extraction/retrieval/injection phases and aggregate percentiles.
   - Reused by tests only (no runtime dependency).

3) Output artifacts (optional):
   - Log concise metrics summary to test output; include counts and p95.

Acceptance Criteria
- Tests pass on clean boot without ML deps.
- Metrics collected and reported deterministically on a fixed corpus.
- No side effects on runtime behavior.

TDD Plan
- Implement helper first, then write tests to use it.
- Keep corpora small to avoid flakiness; mark as performance tests and allow `-k performance` selection.

Commands (to be run by Droid Exec)
```bash
pytest server/tests/performance -q
```

Owner
- Performance Analyzer (via Droid Exec)

