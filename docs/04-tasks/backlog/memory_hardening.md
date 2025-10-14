# Spec: Retrieval Hardening + Fusion + Token Budgeting

Context
- Current retrieval combines sources (graph, convo FTS, summary) with budget allocation and re‑ranking.
- We want stronger, explicit composite scoring, deterministic dedupe, and strict token budgets. Also, integrate the optional `semantic` source when available.

Goals
- Centralize a composite scorer across all sources (including optional semantic).
- Enforce a strict token budget and ≤2 bullets by default for injection.
- Improve dedupe (near‑duplicate collapse across sources) and greeting/intent gating.

Non‑Goals
- Do not change extraction logic in this spec (handled elsewhere).
- Do not introduce heavy ML dependencies here.

Deliverables
1) Composite scoring in `server/core/memory/retrieval.py`:
   - Score factors per candidate: base source priority, similarity/relevance (from each source), recency (existing), support/strength (graph), and diversity penalty.
   - Explicit, testable weights in one place with docstring.

2) Token budget enforcement in context injection:
   - Enforce max bullets (default 2) and max tokens for the memory message.
   - Truncate bullets deterministically (ellipsize long items), and keep tags `[graph]/[convo]/[summary]/[semantic]` for LLM guidance.

3) Dedupe and gating:
   - Cross‑source dedupe by normalized humanized text (case/alias insensitive).
   - Strengthen greeting/intent gating already present to avoid spurious injection.

4) Optional source integration:
   - If `semantic` is present in `MEMORY_SOURCES` and sidecar is enabled, include semantic candidates with a reasonable base weight (below convo, above summary by default).

Acceptance Criteria
- Deterministic ordering for the same inputs and weights.
- Token budget is never exceeded; bullets ≤ configured max.
- Greetings/short small talk do not inject memory unless `name` is relevant.
- With semantic enabled and FTS/graph weak, semantic can surface top bullets; with semantic disabled, behavior unchanged.

TDD Plan
- Unit tests:
  - `test_composite_scoring_deterministic_ordering()`
  - `test_token_budget_and_bullet_cap_enforced()`
  - `test_cross_source_deduplication()`
  - `test_greeting_intent_gating_suppresses_injection()`
- Integration tests:
  - `test_retrieval_fusion_with_semantic_enabled()`
  - `test_retrieval_fusion_without_semantic_identical_to_baseline()`

Implementation Notes
- Keep weights configurable via env (e.g., `MEMORY_WEIGHT_GRAPH`, `MEMORY_WEIGHT_CONVO`, `MEMORY_WEIGHT_SUMMARY`, `MEMORY_WEIGHT_SEMANTIC`) with sane defaults.
- Compute an estimated token count per bullet via chars/4 heuristic; clamp total.
- Reuse `EnhancedFTS` scores (BM25 + factors) as the per‑candidate relevance term.

Commands (to be run by Droid Exec)
```bash
pytest server/tests/unit -k memory_retrieval -q
pytest server/tests/integration -k memory_retrieval -q
```

Owner
- Memory Systems Specialist (via Droid Exec)

