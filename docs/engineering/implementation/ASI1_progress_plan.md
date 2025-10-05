ASI1 Progress and Plan (v0.9 → compiled)

Summary (current state)
- Strict YAML authoring + split spec (L1/L2/L3). Dev-only runtime for evaluation only.
- Edge meta stored per fact: rel_surface, tense/aspect/mood/voice/person/number, polarity, lang, prosody_certainty.
- Prosody pass-through: per-turn certainty plumbed into meta and confidence.
- Retrieval boosts (env-gated): tense/polarity-aware re-ranking using edge.meta.
- English upper bound (transformer, caps off): L1 can reach ≥0.85 on focused sets; broader L1/L2/L3 need coverage work.

Key artifacts
- YAML tools: server/scripts/normalize_yaml.py, server/scripts/validate_yaml.py
- Strict spec index: server/archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml (+ L1/L2/L3)
- Dev runtime: server/core/memory/extractors/yaml_runtime.py (L1 + micro L2/L3)
- Judge module: server/core/memory/judge.py (Lite + Distilled backends)
- Edge meta + prosody: server/core/memory/memory_hotpath.py, server/core/memory/memory_store.py
- Retrieval boosts: server/core/memory/retrieval.py
- Eval scripts: server/scripts/eval_extraction.py, server/scripts/eval_openie_carb.py (soft CaRB-style)
- Judge tooling: server/scripts/train_graph_judge.py, server/scripts/judge_collect_and_distill.py, server/scripts/eval_extraction_diff.py
- Evals: server/tests/data/yaml_eval_*.json (see manifest.json for list)

Recent improvements
- Prosody pass-through; edge meta persisted.
- EN L1: xcomp handling; object enrichment; verb-attached preps; initial phrasal (prt) + coord expansion + passive agent support.
- ES/IT: clitic object fallback.
- FR: clitics + ‘y’; DE: sb/oa; zh topic-comment (guarded without model).

 English accuracy (upper-bound profile: trf, NER on, caps off)
 - L1 small dev (yaml_eval_examples.json): F1 ≈ 0.857
 - L1 EN 27-subset (yaml_eval_l1_en_27.json): F1 ≈ 0.947
- L1 broad (current): easy/medium/long → 0.60 / 0.39* / 0.26  (*YAML_COREF=off for pure L1 scoring)
- L1 + Judge (distilled): medium/long → 0.45 / 0.35 (caps off/on as noted below)
- L2 EN (current): coref ≈ 0.23; discourse ≈ 0.20; temporal ≈ 0.19
 - L3 EN (current): relcl ≈ 0.09; clausal (YAML_LIGHT_VERBS=on) ≈ 0.19; phrasal TBD

Immediate plan (EN first, evidence-driven)
Stage A – EN L1 (next 1–2 passes)
- Finalize phrasal relations (verb_particle) including particle movement.
- Expand coordination (subjects/objects/verbs) with safe duplication control.
- Passive normalization (nsubjpass/agent → SVO) + keep object enrichment.
- Re-run yaml_eval_l1_en_easy/medium/long and report PRF + examples.

Diagnostic metrics (strict vs relaxed)
- Strict scoring (exact s/r/d match) undercounts semantically correct edges (e.g., minor NP differences, lexicalized vs verb+prep).
- Add a relaxed internal diagnostic (env-gated) that normalizes:
  - Determiners/quantifiers and common adjective noise (e.g., record/new/various).
  - Verb+prep vs lexicalized relation equivalence (work on ≡ work_on).
  - Nominalization mappings (effects of X on Y ≡ X affect Y).
- Keep strict PRF for acceptance; use relaxed PRF to guide normalization work.

Recent Stage A changes (implemented)
- Phrasals: verb+particle relations; coordination with inherited args.
- Objects: noun-chunk based NP building + argumental PPs by verb.
- Ditransitives: emit both objects (obj/iobj/dative).
- Passive: be/get+VBN fallback, agent ‘by’ handling, no-agent PP fallback.
- Control/xcomp: subject- vs object-control for complements (EN).
- Copula/raising: head lemma as relation (be, seem).
- Relative clauses: main-edge extraction from relcl.
- L2 assist: appositive aliasing prior to pronoun rewrite.
- Appositive-as-edge (env-gated): emit conservative PROPN→NOUN/PROPN 'is' edges.
- Pronoun resolution: stricter gating for it/this/that (same/prev sentence, non-person).
- Coordination: recursive conj collection across longer chains (subjects/objects).
- Env flags: YAML_COREF (disable coref for L1 eval), YAML_LIGHT_VERBS (L3 light-verb rewriting), YAML_SINGULARIZE (entity head singularization, default off).
 - Env flags (new): YAML_NOMINALS to enable conservative nominalization rewriting (effects of X on Y → X affect Y; increase in X → increase X; increase of X [by Y]? → increase X (by Y); agreement on X → agree_on X).
 
Judge (optional, precision booster)
- YAML_GRAPH_JUDGE=on enables a triple quality filter in YAMLExtractor.refine.
- Modes:
  - Lite (default): hand‑tuned linear scorer over lexicalized relation, object length, generic/pronoun objects, relation priors, and minimal NER type hints.
  - Distilled (env‑gated): set `YAML_GRAPH_JUDGE_MODEL=path/to/graph_judge.json` to load a logistic model trained via `server/scripts/train_graph_judge.py`. Runtime applies a dot product + sigmoid per triple; threshold via `YAML_GRAPH_JUDGE_THRESH` (optional; JSON may embed its own threshold).
- Training: `python -m scripts.train_graph_judge --dataset tests/data/yaml_eval_l1_en_medium.json --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml --out models/graph_judge.json --auto_calibrate`.
- Latency: Lite/Distilled add microseconds per triple. LLM‑judge reserved for offline re‑indexing.
 - Gray‑zone logging (self‑improvement):
   - `YAML_GRAPH_JUDGE_GRAY_BAND=0.10` (|score−threshold| within band) and `YAML_GRAPH_JUDGE_GRAYZONE_LOG=data/judge_grayzone.jsonl` to collect uncertain cases for idle‑time labeling/distillation.
 - Idle‑time distillation (supervised or semi‑supervised):
   - Supervised: `python -m scripts.judge_collect_and_distill --log data/judge_grayzone.jsonl --dataset tests/data/yaml_eval_l1_en_medium.json --yaml archive/.../ASI1_index_v0_9.yaml --out models/graph_judge.json --auto_calibrate`.
   - Semi‑supervised: omit `--dataset` to export features JSON; label with an LLM‑Judge (DSPy program), then re‑run with labels to distill.
 - Side‑by‑side diffs: `python -m scripts.eval_extraction_diff --dataset ... --yaml ... --judge_model models/graph_judge.json --out ../docs/reports/diff_l1_medium.md`.
 
SRL (future optional)
- Plan: add an optional SRL refine pass (env‑gated: `YAML_SRL=on`) to improve argument identification without changing the fast path by default.
- Scope when enabled:
  - Keep/drop PP arguments via PropBank roles (keep ARGs, drop adjunct ARGM where appropriate).
  - Repair subjects/objects for passives and clausal complements.
  - Run only for long/ambiguous sentences or in evaluation/batch re-indexing; cache model; prefer ONNXRuntime EPs.
- Default remains OFF to preserve low latency; rule/dictionary passes remain primary.
 - Discourse cues: because of, due to, owing to, as a result (of), as a consequence (of), in order to, so that, so as to.
 - Relative clauses: pied‑piping (to/for/in/at/with which/whom) and possessive 'whose'.
 - Verb→prep policies: adhere to, comply with, engage in/with, enter into, lead to, apply to, consist of/in; benefit from, result in/from, stem from, plus lexicalized agree_on/result_in/stem_from.
 - Optional ML (env‑gated): set `YAML_PP_SCORER=on` to enable a tiny linear PP keep/drop filter during verb→prep enrichment (default OFF; conservative drop-only after dictionary allowlists).

How to run (full EN sweep)
- L1 easy/medium/long (pure L1 scoring): set `YAML_COREF=off`, keep `YAML_SINGULARIZE=off`.
- L2 coref/discourse/temporal: default envs (no caps) per instructions below.
- L3 relcl/clausal: set `YAML_LIGHT_VERBS=on` for clausal light-verb rewriting.
 - L1 with Judge (distilled):
   - Medium (caps off): `SPACY_MODEL_EN=en_core_web_trf YAML_DENSITY_CAPS=off YAML_NOMINALS=on YAML_COREF=off YAML_SINGULARIZE=off YAML_GRAPH_JUDGE=on YAML_GRAPH_JUDGE_MODEL=models/graph_judge.json .venv/bin/python -m scripts.eval_extraction --dataset tests/data/yaml_eval_l1_en_medium.json --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml --lang en`
   - Long (caps on): `SPACY_MODEL_EN=en_core_web_trf YAML_DENSITY_CAPS=on YAML_NOMINALS=on YAML_COREF=off YAML_SINGULARIZE=off YAML_GRAPH_JUDGE=on YAML_GRAPH_JUDGE_MODEL=models/graph_judge.json .venv/bin/python -m scripts.eval_extraction --dataset tests/data/yaml_eval_l1_en_long.json --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml --lang en`

Stage B – EN L2
- Appositive coref + longer chains; better pronoun/definite resolution.
- Richer discourse/temporal triggers across sentence boundaries (because/result/contrast/if/when/after/before/then) with robust punctuation patterns.
  - Multiword discourse cues: because of, due to, owing to, as a result/as a consequence.
- Re-run yaml_eval_l2_en_coref/discourse/temporal.

Stage C – EN L3
- Light verbs (make a decision→decide, take a walk→walk), clausal complements (that/infinitivals), relative clauses (main-edge extraction), finalize phrasal suite.
- Re-run yaml_eval_l3_en_*.json sets and report PRF + examples.

Then: port targeted improvements to other languages once models are available.

Eval vs runtime profiles (env)
- Eval (upper bound):
  - SPACY_MODEL_EN=en_core_web_trf
  - SPACY_DISABLE=   (empty → full pipeline)
  - YAML_DENSITY_CAPS=off
- Runtime (fast path):
  - SPACY_MODEL_*=*_sm (or md)
  - SPACY_DISABLE=ner,textcat
  - YAML_DENSITY_CAPS=on (default)

Telemetry / Debugging
- MEMORY_STORE_META_TELEMETRY=true → log edge.meta per store
- MEMORY_TENSE_AWARE / MEMORY_POLARITY_AWARE → enable/disable boosts
- MEMORY_RETRIEVAL_TELEMETRY=true → log applied boosts
- MEMORY_DEBUG_BULLETS_META=true → append compact meta to [graph] bullets

Acceptance gates (EN first)
- L1 EN (broad evals): ≥0.70 in medium/long; target ≥0.90 on focused sets
- L2 EN (coref/discourse/temporal): ≥0.60
- L3 EN (phrasal/clausal/relcl): ≥0.75
- Then port to other languages with model availability

How to run (examples)
- L1 dev small:
  cd server && SPACY_MODEL_EN=en_core_web_trf SPACY_DISABLE= YAML_DENSITY_CAPS=off \
  .venv/bin/python -m scripts.eval_extraction \
    --dataset tests/data/yaml_eval_examples.json \
    --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml --lang en

- L1 broad:
  cd server && SPACY_MODEL_EN=en_core_web_trf SPACY_DISABLE= YAML_DENSITY_CAPS=off \
  .venv/bin/python -m scripts.eval_extraction \
    --dataset tests/data/yaml_eval_l1_en_medium.json \
    --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml --lang en

- CaRB-style (soft internal):
  SPACY_MODEL_EN=en_core_web_trf YAML_DENSITY_CAPS=off \
  uv run --project server --directory server -m scripts.eval_openie_carb \
    --sentences data/carb/carb_test_sentences.txt \
    --gold data/carb/carb_test.tsv \
    --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml

Open items
- Generate yaml_eval_l3_en_light_verbs.json and add to manifest once ready.
- Once non-EN models finish installing, repeat eval profile per language.
 - Expand verb→prep argument map (conservative; frequency-driven).
 - Add conservative light-verb rewrites beyond current small set.
 - Deepen coordination across longer conj chains and punctuation.
