# Local Agent Memory + Context Plan (Common‑Sense Edition)

Status: Draft
Owner: Memory/Context
Last updated: 2025‑10‑11

## Goals

- Make a small local agent feel sharp, grounded, and fast.
- Prefer conversation context; add only facts that help the next turn.
- Keep the prompt tiny and durable; support “forever” conversations.
- Avoid duplicate work; do one thing one way.

## Principles

- Conversation first. Retrieve from the current session before graph.
- Show less, mean more. ≤2 short bullets; omit anything not needed now.
- Low ceremony. Defaults should “just work” on low‑end models.
- Single source of truth. One set of envs; one indexing path; one injector.

## What Changes (High Level)

- Retrieval: `convo → graph fallback` (no summary injection by default).
- Bullets: at most 2, concise, deduped on the humanized fact.
- Context: persona → session header → memory block (single message).
- Prompt guide: 2 short lines embedded in the memory block header.
- Enhanced FTS: enabled and fed; basic FTS is a safe fallback.
- Sliding window: keep a rolling window of last N turn pairs; keep all system messages. Conversation can continue indefinitely.

## Implementation Plan (Step‑by‑Step)

1) Enhanced FTS ON (BM25 + expansion)
   - Initialize schema at store init.
   - On flush, index every conversation turn into `chunks_content` (triggers maintain `chunks_fts_enhanced`).
   - Retrieval uses Enhanced FTS first; if no hits, fallback to basic FTS (sanitized MATCH).

2) Retrieval Policy (simple, deterministic)
   - Env: `MEMORY_SOURCES=convo,graph`
   - Env: `MEMORY_SINGLE_SOURCE=true` (top‑priority source only)
   - Policy:
     - Try conversation FTS first (scoped to user/session when possible).
     - If empty, try graph edges (with allowlist; see below).
     - Never inject summaries by default; keep summarization offline.

3) Graph Allowlist (zero‑maintenance by default)
   - Default allowlist for injection: `name,lives_in,works_at,has`.
   - These relations are broadly useful and stable over time.
   - All other relations remain storable/searchable; they’re just not auto‑injected.
   - Retrieval still uses non‑allowlisted relations when the user explicitly asks about them (query‑driven), so no loss of capability.

4) Bullets (crisp by design)
   - Cap to `MEMORY_BULLETS_MAX=2` (1 is often enough).
   - 100–120 chars max; humanized, declarative.
   - Deduplicate by the humanized string. Drop identity tautologies.

5) Context Assembly (one way)
   - Messages: persona → session header → memory block.
   - Memory block header contains the micro‑guide (2 lines):
     - “Use if relevant; prefer [convo].”
     - “Don’t quote tags; keep it short.”
   - Inject as `role=user` by default (friendlier to small LMs). Flip to `system` if stronger bias is needed.

6) Sliding Window (forever conversations)
   - Keep all `system` messages.
   - Keep the last `CONTEXT_MAX_TURN_PAIRS` user/assistant pairs (default 4).
   - Prune on each injection to bound token growth.
   - Older turns remain available via FTS; durable facts live in the graph.

7) Extraction & Storage
   - UD extractor is the default; DSPy off by default.
   - SQLite is the default storage; LMDB optional.
   - One set of envs: `MEMORY_*` is the source of truth. Keep `HOTMEM_*` as legacy fallbacks (deprecate later).

## Rationale: Allowlist With Zero Maintenance

The allowlist only applies to unsolicited injection, not to storage or query‑driven retrieval. This yields “zero maintenance” in practice:

- The default set (`name,lives_in,works_at,has`) covers 90%+ of helpful ambient facts.
- If the user asks about other relations, FTS/graph still retrieve them (query‑driven), bypassing the allowlist.
- No tuning required: the allowlist is purposefully small, stable, and domain‑agnostic.

Optionally, we can make it self‑healing without additional ops work:

- Track a light “post‑injection correction ratio” per relation (how often the user corrects after this relation was injected). If a relation’s corrections exceed a threshold, temporarily exclude it from injection.
- Likewise, relations with high acceptance (no correction + follow‑up usage) can be auto‑promoted into the allowlist over time.
- These metrics are local and privacy‑preserving; no labeling required. Updates are rate‑limited and revertible (e.g., on restart).

This way, the default allowlist just works, and the agent can adapt in the background with zero manual maintenance.

## Env Defaults (Small‑Model Friendly)

```
MEMORY_SOURCES=convo,graph
MEMORY_SINGLE_SOURCE=true
MEMORY_BULLETS_MAX=2
MEMORY_INJECT_ROLE=user
CONTEXT_SLIDING_WINDOW=true
CONTEXT_MAX_TURN_PAIRS=4
ENABLE_DSPY_EXTRACTION=false

# Enhanced FTS stays enabled; basic FTS is fallback
ENHANCED_FTS_ENABLED=true  # implied by schema + writes

# Optional allowlist override
MEMORY_GRAPH_ALLOWLIST=name,lives_in,works_at,has
```

## Validation & Metrics

- Logs:
  - Verify source mix: single‑source bullets per turn.
  - Check Enhanced FTS hits; fallback only when necessary.
  - Confirm pruning keeps user/assistant to the last N pairs.
- Behavior:
  - Short, grounded answers; fewer off‑topic references.
  - Corrections drop; first‑try accuracy improves.
- Perf:
  - Injection and retrieval < 50ms; total memory path < 200ms p95.

## Risk & Mitigation

- Risk: Removing summary injection reduces “coherent recap”.
  - Mitigation: summaries remain available offline/on demand.
- Risk: Allowlist hides some relations.
  - Mitigation: query‑driven retrieval still returns them; allowlist is injection‑only.

## Rollout

1) Switch env defaults to convo‑first, single‑source, 2 bullets.
2) Enable Enhanced FTS indexing; observe hits vs. fallback.
3) Embed the micro‑guide into the memory header; set injection role to user.
4) Keep DSPy off; consider enabling for complex sentences only after validation.
5) (Optional) Add auto‑tuning for the allowlist using correction ratios.

—

Elegance is subtraction: one retrieval path, one memory block, two short bullets, and a tiny guide. The agent stays light, local, and helpful.

