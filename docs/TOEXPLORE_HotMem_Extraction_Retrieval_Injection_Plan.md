# TOEXPLORE: HotMem Extraction, Retrieval, and Context Injection Overhaul

## Summary
- Goal: Radically improve extraction precision, retrieval relevance, and context injection hygiene while keeping hot-path latency under 200ms p95.
- Scope: `server/memory_hotpath.py` (extraction, retrieval, scoring), `server/hotpath_processor.py` (injection policy), and minimal data-structure tweaks in `server/memory_store.py` (if needed for ts/weight hot reads).
- Defaults per requirements:
  - Languages: EN + ES/FR/DE/IT
  - Injection role: `system`
  - Ephemeral memory hints: sliding window of last 10 turns
  - Dynamic bullets per turn: 1–5 based on score/gating
  - No extra guardrails (fully offline)

## Non-Goals
- No remote calls or external model dependencies beyond spaCy small models.
- No change to pipeline topology (HotMem still runs before `context_aggregator.user()`; injection occurs before the user message is appended).

## Current Pain Points (Observed)
- Over-broad relations (e.g., `quality`, `modified_by`, generic preposition tails) dilute retrieval and bullet quality.
- Minimal coreference: only first-person → `you`; no resolution for he/she/they/it/this.
- Retrieval relies on predicate priority + timestamp lookup via `store.neighbors()` inside ranking; potential micro-hotspot and weak topical relevance.
- Injection persists a new memory message on every turn, can bloat context, and may repeat identical hints; role is `user` by default.

References:
- Extraction/Refine: `server/memory_hotpath.py:222`, `server/memory_hotpath.py:900`
- Retrieval/Ranking: `server/memory_hotpath.py:733`
- Bullet NLG: `server/memory_hotpath.py:665`
- Injection: `server/hotpath_processor.py:215`, `server/hotpath_processor.py:223`

## Design Overview
We will:
1) Constrain relations to a compact, conversational taxonomy; add light coref and hedging handling.
2) Rework retrieval scoring to include lexical relevance, diversity, novelty gating, and remove per-candidate LMDB scans in ranking.
3) Switch to system-role, ephemeral injection with a sliding window (last 10 turns) and dynamic 1–5 bullets based on score/intent/entity cues.
4) Add metrics for extraction/refinement/retrieval/injection and quality probes.

## Detailed Plan

### 1) Extraction Improvements (memory_hotpath.py)

1.1 Canonical Relation Taxonomy
- Replace or demote generic relations with a bounded set:
  - Identity/Attributes: `name`, `age`, `favorite_color`, `is`
  - Residence/Work: `lives_in`, `works_at`, `born_in`, `moved_from`
  - Participation/Movement: `participated_in`, `went_to`
  - Social/Ownership: `friend_of`, `owns`, `has`
  - Temporal/Quant: `time`, `duration`, `quantity`
- Map UD patterns to this taxonomy; drop raw `modified_by`, `quality` unless coupled to a concrete noun (and low weight). Keep generic `is/has` only when nothing more specific is available.

1.2 Coreference Lite
- Maintain a small “mention stack” per turn from recent proper nouns and noun chunks.
- Resolution heuristics:
  - Pronouns (`he/she/they/it/this/that`) → most recent compatible mention in current turn; fallback to last named entity in `recency_buffer`.
  - Preserve first-person mapping to `you` as today; add “we/our/us” → `you` unless there is a better multi-entity context.
- Keep O(1) per token via simple stacks; no global coref models.

1.3 Hedging/Negation Awareness
- Detect hedges: {"maybe", "I think", "probably", "kinda", "sort of", "not sure"}.
- Reduce confidence in `QualityFilter.should_store_fact()` when hedging present (e.g., −0.2), and tie negation counts to confidence (more neg tokens → more cautious unless it’s a correction intent).

1.4 Multi-Fact Segmentation
- For coordinated objects/subjects (UD `conj`), split into multiple clean triples rather than fusing phrases where appropriate.
- Special-case copulas with possessive subjects (already added) and extend to common family/pet patterns.

1.5 Multilingual (EN+ES/FR/DE/IT)
- Load spaCy small models per language with lemmatizer enabled:
  - en_core_web_sm, es_core_news_sm, fr_core_news_sm, de_core_news_sm, it_core_news_sm
- Disable only NER and textcat, keep lemmatizer and parser.
- Add minimal surface aliasing for language-specific prepositions/verbs mapping to canonical relations:
  - ES: "vive en"→`lives_in`, "trabaja en"→`works_at`
  - FR: "habite à/en"→`lives_in`, "travaille chez/à"→`works_at`
  - DE: "wohnt in"→`lives_in`, "arbeitet bei"→`works_at`
  - IT: "vive a/in"→`lives_in`, "lavora a/presso"→`works_at`
- Fallback: if model unavailable, use `spacy.blank(lang)` with sentencizer; extraction gracefully degrades to regex/refinement.

Implementation Notes (later):
- Augment `_load_nlp` to keep lemmatizer for all supported langs.
- Add a small per-lang map used in `_refine_triples()` when normalizing preposition tails.

### 2) Retrieval and Ranking (memory_hotpath.py)

2.1 Hot Index Metadata
- Extend hot RAM indices to avoid per-candidate `neighbors()` scans:
  - Keep `edge_meta[(s,r,d)] = {ts, weight}` updated in `observe_edge/negate_edge` flows (ts = last update)
  - On rebuild, read `edge.updated_at` from SQLite to populate `ts`.
- Keep `entity_index[entity] -> set[(s,r,d)]` as today.

2.2 Scoring Function
- Score triple t given query q:
  - score(t) = α·pred_priority(r) + β·recency(ts) + γ·lexical_overlap(q, s,r,d) + δ·edge_weight
  - Default weights: α=0.4, β=0.2, γ=0.3, δ=0.1 (tunable via env)
- `pred_priority`: keep current biases, extend for other canonical relations.
- `recency(ts)`: normalize by exponential decay over last N hours/days.
- `lexical_overlap`: token overlap between query and s/r/d; optionally mix in FTS rank from `search_fts(q)` for top-K entity terms.
- `edge_weight`: from hot index.

2.3 Diversity and Novelty
- Diversity: limit to 1 per relation per turn when possible; prefer distinct subjects if not `you`.
- Novelty: suppress reinjecting an identical bullet within the last `HOTMEM_NOVELTY_TURNS` (default 3) unless the query asks for it (e.g., contains the subject or relation keyword).

2.4 Dynamic K (1–5)
- Choose K based on score distribution and intent:
  - Reaction/Pure Question: K=0–2 (only if high lexical overlap or “Where/Who” targets known entities)
  - Question with Fact / Fact Statement / Multiple Facts: K=2–5
- Threshold-based: include bullets with score ≥ τ; adapt τ to keep total ≤5.

### 3) Injection Strategy (hotpath_processor.py)

3.1 System-Role, Ephemeral Window
- Inject as `role=system` with a compact, stable header:
  - "Use the following factual context if helpful:"
- Keep only the last `HOTMEM_HINT_WINDOW` (default 10) memory-hint turns.

Two approaches (choose A for simplicity, B later if needed):
- A) Rotating Single System Message
  - Maintain a single "HotMem Context" system message that aggregates bullets from the last N (10) turns (deduped, top-scored per turn).
  - On each turn, update/replace that single system message rather than appending a new one.
  - Pros: bounded context growth; straightforward pruning.
- B) Per-Turn Memory Messages with Pruning
  - Append one per turn; scan and prune messages tagged with a HotMem marker to keep only the last N.
  - Requires scanning and replacing context (via `LLMMessagesUpdateFrame`) when over window size.

3.2 Gating Policies
- Inject only when:
  - The turn mentions an entity present in hot memory; or
  - Intent ∈ {FACT_STATEMENT, QUESTION_WITH_FACT, MULTIPLE_FACTS}; or
  - A direct “where/when/who/what” question targeting known entities.
- Skip for REACTION/PURE_QUESTION unless the query targets a known entity.

3.3 Ordering
- Keep current placement: inject before `context_aggregator.user()` finalizes the user message so that the system hint precedes the user message.

3.4 Dedup and Formatting
- Deduplicate bullets against those already in the rotating system message.
- Keep bullet phrasing compact; preserve "you"-centric phrasing first, then other entities.

### 4) Metrics and Observability
- Track per-stage timings: `intent_ms`, `extraction_ms`, `update_ms`, `retrieval_ms`, `total_ms` (already present); log p95 every 30s.
- Track injection stats: bullets computed, injected, skipped by gating, pruned.
- Optional quality probe: heuristic “was hint used?” e.g., overlap between assistant response and injected subjects/relations.

### 5) Configuration (env)
- `HOTMEM_LANGS`: `en,es,fr,de,it`
- `HOTMEM_INJECT_ROLE`: default `system`
- `HOTMEM_INJECT_HEADER`: default "Use the following factual context if helpful:"
- `HOTMEM_HINT_WINDOW`: default `10`
- `HOTMEM_BULLETS_MAX`: default `5`
- `HOTMEM_NOVELTY_TURNS`: default `3`
- `HOTMEM_SCORE_WEIGHTS`: e.g., `0.4,0.2,0.3,0.1` for α,β,γ,δ

### 6) Data Model Touches (minimal)
- In-memory only (preferred): add `edge_meta[(s,r,d)] = {ts, weight}` in `HotMemory`; populate during observe/negate and rebuild.
- Optional (if needed): expose `edge.updated_at` in `MemoryStore.get_all_edges()` for rebuild to seed `ts` without extra queries.

## Testing Plan
- Update/extend 27-pattern tests to validate taxonomy mapping, multi-fact segmentation, and multilingual surface forms.
  - File: `server/test_27_patterns_updated.py` (new/updated).
- Add retrieval tests for dynamic K selection, diversity, novelty suppression, and lexical relevance.
- Add injection tests for gating, system-role message placement, and sliding window pruning (both approach A and B paths)
- Latency microbenchmarks per stage; ensure p95 < 200ms end-to-end with models prewarmed.

## Rollout Plan
- Phase 1 (Retrieval/Injection):
  - Implement scoring, novelty/diversity, system-role injection, rotating message window.
  - Add metrics and env toggles. Ship behind feature flags.
- Phase 2 (Extraction):
  - Relation taxonomy and coref-lite; hedging/negation adjustments.
  - Multilingual normalization maps and loaders.
- Phase 3 (Tuning):
  - Adjust weights/thresholds based on telemetry; refine bullet NLG.

## Risks & Mitigations
- spaCy model availability for non-EN languages → fallback to blank model + regex refinements; log once, continue.
- Context aggregator API surface for in-place update → prefer rotating single system message; if replace-in-place is not supported, rebuild context with `LLMMessagesUpdateFrame` occasionally.
- Over-injection in short/reactive turns → gating by intent + entity presence and dynamic K.

## Change Map (Future PRs)
- `server/memory_hotpath.py`
  - Add coref-lite mention stack; refine taxonomy mappings; extend `_refine_triples` with multilingual aliases.
  - Maintain `edge_meta` and use it in `_retrieve_context` instead of per-candidate `neighbors()` calls.
  - Implement scoring function with lexical overlap; add diversity/novelty; dynamic K.
  - Tighten bullet formatter for new relations.
- `server/hotpath_processor.py`
  - Switch default inject role to `system`; add gating and rotating system message maintenance.
  - Add metrics for injected/skipped/pruned counts.
- `server/memory_store.py` (optional)
  - Ensure `get_all_edges()` includes `updated_at` or add method to fetch (src,rel,dst,weight,updated_at) for rebuild.
- Tests under `server/` for extraction patterns, retrieval ranking, injection behavior.

## Open Questions (Resolved)
- Languages: EN+ES/FR/DE/IT
- Inject with system role: Yes
- Ephemeral sliding window: Keep last 10 turns
- Dynamic bullet count: 1–5
- Guardrails: None (fully offline)

---

This document specifies the planned changes without modifying code. Once approved, we can implement in small PRs following the rollout plan.

## Addendum: Concrete Implementation Details

### Data Structures (in-memory, hot path)
- entity_index: dict[str, set[(s,r,d)]]
  - Key: canonical entity; Value: distinct (s,r,d) triples touching that entity.
- edge_meta: dict[(s,r,d), {ts:int, weight:float, last_injected_turn:int|None}]
  - Metadata to avoid LMDB scans during ranking and enforce novelty.
- recency_buffer: deque[RecencyItem], maxlen=50 (existing)
  - RecencyItem: { s,r,d,text,timestamp,turn_id,score }
- inject_window: deque[InjectItem], maxlen=HOTMEM_HINT_WINDOW (default 10)
  - InjectItem: { turn_id:int, triple:(s,r,d), bullet:str, score:float }
- last_turn_entities: list[str]
  - Entities recognized post-refinement for gating.
- novelty_cache: dict[(s,r,d), int]
  - Last turn_id a triple was injected.

### Canonical Relation Mapping (taxonomy)
- Identity/Attributes: name, age, favorite_color, is
- Residence/Work: lives_in, works_at, born_in, moved_from
- Participation/Movement: participated_in, went_to
- Social/Ownership: friend_of, owns, has
- Temporal/Quant: time, duration, quantity

UD patterns → canonical relations (English):
- Copula: nsubj + cop + attr/acomp → (subj, is, attr)
- Passive name: nsubjpass(head∈{name,call}) + oprd → (subj, name, oprd)
- Prepositions with governing verb:
  - live + in/at → lives_in/lives_at
  - work + at/for/in → works_at/works_in
  - born + in → born_in
  - move + from → moved_from
  - participate + in → participated_in
  - go/went + to → went_to
- Possessive copula: "my X is Y" → (you, X_as_relation, Y)
- Have/has/had dobj → (subj, has, obj)
- Conjunction propagation; friend_of derivation from "X and Y are friends".

Multilingual surface aliasing (examples):
- ES: vive en→lives_in, trabaja en→works_at, nació en→born_in, se mudó de→moved_from
- FR: habite à/en→lives_in, travaille chez/à→works_at, né(e) à/en→born_in
- DE: wohnt in→lives_in, arbeitet bei→works_at, geboren in→born_in
- IT: vive a/in→lives_in, lavora a/presso→works_at, nato/a a/in→born_in

Implementation: per-language maps in refinement to rewrite underspecified rels ("_in", "_at", …) to canonical forms using verb lemma + preposition.

### Coreference-lite Algorithm
- Maintain mention_stack per turn with recent proper-noun/noun-chunk mentions.
- Pronouns:
  - he/she → last PERSON mention ≠ you
  - they → last plural/group mention else last PERSON
  - it/this/that → last concrete non-PERSON noun
  - we/our/us → you (unless clear group entity present)
- Backoff: current turn stack → last 3 named entities from recency_buffer.
- O(1) lookups; no neural coref to preserve latency.

### Hedging/Negation Handling
- Hedge lexicon: maybe, I think, probably, kinda, sort of, not sure, perhaps, possibly.
- If hedged: QualityFilter confidence −0.2.
- If neg_count>0 and intent≠CORRECTION: confidence −0.2; if CORRECTION: +0.3 and trigger conflict demotion.

### Retrieval Scoring Algorithm
- score(t) = α·pred_priority(r) + β·recency(ts) + γ·lexical_overlap(q,s,r,d) + δ·edge_weight
- pred_priority: { lives_in:100, works_at:95, born_in:90, moved_from:85, participated_in:80, friend_of:78, name:75, favorite_color:70, has:60, is:55 }
- recency(ts): exp(-Δ/T) where Δ = now - ts, T default 3 days (configurable); apply a same-session boost (e.g., ×1.2) for mentions within the current session/turn cluster and optional time-of-day weighting (today > yesterday at same local hour)
- lexical_overlap: token-overlap/Jaccard of content words between query and {s, r tokens, d}; optional small boost if entity appears in SQLite FTS search
- edge_weight: from edge_meta
- Default weights: α=0.4, β=0.2, γ=0.3, δ=0.1 (env-tunable)

Ranking flow:
- Gather candidates from entity_index for entities in query (non-"you" first, then "you")
- Lookup ts/weight from edge_meta (fallback ts=0, weight=0.3)
- Score, de-duplicate by (s,r,d)
- Diversity: ≤1 per relation by default; allow ≤2 for high-scoring pairs (score ≥ τ + ε), still preferring distinct subjects when not "you"
- Novelty: skip if novelty_cache[(s,r,d)] ≥ turn_id - HOTMEM_NOVELTY_TURNS
- Select top K via dynamic policy below

### Dynamic Bullet Count (K)
- Compute scores; define τ as 75th percentile or baseline 0.45 if few candidates
- REACTION/PURE_QUESTION: up to 2 if score ≥ τ and entity-targeted
- QUESTION_WITH_FACT/FACT_STATEMENT/MULTIPLE_FACTS: 2–5, capped by HOTMEM_BULLETS_MAX

### Injection Gating and Rotating System Message
- Inject only if candidates above τ and entity overlap, or intent ∈ {FACT_STATEMENT, QUESTION_WITH_FACT, MULTIPLE_FACTS}
- Rotating single system message (preferred):
  - Build from inject_window (last N turns), dedup by (s,r,d), sort by recency then score
  - Replace/update a single tagged system message before user message is added
  - Include freshness header, e.g., "Context from the last 10 conversational turns (updated: ISO 8601)"
  - Update novelty_cache[(s,r,d)] = turn_id for injected bullets

### Concurrency & Thread Safety
- Reader–writer locks around hot indices:
  - Read lock for retrieval/scoring on `entity_index` and `edge_meta`
  - Write lock for updates to `entity_index`, `edge_meta`, `recency_buffer`, `inject_window`
- Atomic updates for weights/metadata:
  - Update `(weight, ts, last_injected_turn)` under a single write lock to keep metadata coherent
  - Favor single-writer principle on the hot path; readers proceed concurrently
- Queue-based batching to reduce contention:
  - Keep persistence batching (SQLite/LMDB) off the hot lock
  - Optionally stage hot-index updates and commit once per turn under the write lock

### Error Handling and Fallbacks
- spaCy load failure → spacy.blank(lang) + sentencizer; rely on regex/refinement; retrieval continues
- LMDB/SQLite transient errors → catch/log, enqueue for retry in batch flush; do not block hot loop
- Context aggregator missing → log and skip injection; continue extraction/storage
- Token budget exceeded when building rotating message → trim by lowest score/oldest first; respect HOTMEM_BULLETS_MAX
- Kill switches: HOTMEM_DISABLE_INJECTION=1, HOTMEM_DISABLE_EXTRACTION=1

### Memory Lifecycle and Conflict Resolution
- Reinforcement: EWMA of weight; status from weight (existing)
- Corrections: demote conflicting (s,r,old_d) then observe (s,r,new_d); update entity_index accordingly
- Staleness decay: optional passive decay (e.g., 0.99/day) during rebuild/maintenance; status flips to stale/archive thresholds
- Forgetting: hard_forget(s[,r[,d]]) purges LMDB + tombstones in SQLite; expose via user command
- TTLs: optional env TTL per relation type; transition to stale if not reinforced

### Privacy and Security (offline)
- All processing local; no network
- Optional log minimization/redaction for PII (emails/phones) if enabled
- User controls: export memories; hard forget; per-user isolation by user_id
- At-rest: rely on OS/disk encryption if needed; no change to hot path

### Examples (EN + Multilingual)
- EN: "My name is Alex Thompson" → (you, name, alex thompson) → • Your name is Alex Thompson
- EN: "I live in Seattle and work at Microsoft" → (you, lives_in, seattle), (you, works_at, microsoft)
- ES: "Vivo en Madrid" → (you, lives_in, madrid)
- FR: "Je travaille chez Airbus" → (you, works_at, airbus)
- DE: "Ich bin 30 Jahre alt" → (you, age, 30) → • You are 30 years old
- IT: "Mi sono trasferito da Torino" → (you, moved_from, torino)

Sample rotating system message (system role):
Use the following factual context if helpful.
Context from the last 10 conversational turns (updated: 2025-01-01T12:34:56Z):
• Your name is Alex Thompson
• You live in Seattle
• You work at Microsoft

### Benchmarks (targets & method)
- Prewarm: spaCy/Stanza load < 1.5s (off hot path)
- Per-turn targets (CPU): intent_ms ≤ 20ms, extraction_ms ≤ 60ms, update_ms ≤ 10ms, retrieval_ms ≤ 20ms, total_ms p95 ≤ 200ms
- Method: 200-turn mixed multilingual set; measure mean/p95 per stage; validate against ≥100-fact gold set

### Success Metrics (beyond latency)
- Extraction precision ≥ 0.80 and recall ≥ 0.60 on curated multilingual set
- Retrieval relevance@3 ≥ 0.80 (human or heuristic overlap)
- Injection usefulness ≥ 0.60 (assistant references injected entities/relations when applicable)
- Duplicate reinjection within 3 turns < 10%
- Rotating system message ≤ HOTMEM_BULLETS_MAX bullets in 99% of turns
- Multilingual mapping success ≥ 90% on language-specific patterns

## Implementation TODOs

### Phase 1 — Retrieval & Injection (high priority)
- Add `edge_meta` cache for ts/weight (hot index)
- Implement scoring function and weights (env-tunable)
- Add lexical overlap and optional FTS boost
- Enforce diversity (≤1 per rel, ≤2 high-score)
- Add novelty suppression window (configurable turns)
- Compute dynamic bullet count (1–5) per intent
- Implement rotating system message with timestamp header
- Switch default injection role to `system`
- Add gating by intent and entity overlap
- Deduplicate against existing rotating message bullets
- Add reader–writer locks for hot indices
- Atomic updates for weights and metadata
- Stage hot-index commits once per turn
- Add telemetry for retrieval/injection metrics
- Add env flags: disable extraction/injection, weights, K, windows
- Write unit tests for scoring/diversity/novelty/dynamic K
- Write integration test for rotating message update/pruning
- Add benchmarks to record per-stage timings (p95)

Acceptance criteria (Phase 1)
- Retrieval_ms mean ≤ 20ms; total_ms p95 ≤ 200ms
- No duplicate reinjection within last N=3 turns (≤10%)
- Rotating message capped ≤ HOTMEM_BULLETS_MAX and includes timestamp
- Relevance@3 ≥ 0.80 on test prompts

### Phase 2 — Extraction (taxonomy, coref-lite, multilingual)
- Implement canonical relation mapping table
- Add per-language alias maps (EN/ES/FR/DE/IT)
- Keep lemmatizer+parser; disable NER/textcat
- Add coref-lite mention stack resolver
- Extend regex refinements (pets/family/preferences)
- Apply hedging/negation to QualityFilter confidence
- Expand multi-fact segmentation via conj propagation
- Add tests for 27 patterns + multilingual cases
- Update bullet formatter for new relations

Acceptance criteria (Phase 2)
- Extraction precision ≥ 0.80, recall ≥ 0.60 (gold set)
- Multilingual canonicalization success ≥ 90%
- No increase in mean extraction_ms > 60ms

### Phase 3 — Tuning & Lifecycle
- Tune weights/thresholds from telemetry distributions
- Add passive decay/TTL policy (optional)
- Improve correction demotion + conflict logging
- Add forgetting commands and verification tests
- Enhance privacy: optional log redaction toggles
- Expand benchmarks with multilingual/longer dialogs

Acceptance criteria (Phase 3)
- Reduced over-injection in reactive turns (>20% drop)
- Stable context hygiene across long sessions
- Documented ops toggles and runbooks
