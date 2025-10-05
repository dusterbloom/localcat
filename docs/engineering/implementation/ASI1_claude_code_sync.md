ASI1 Judge + Self‑Improving Extraction (Claude Code Sync)

Purpose
- Document what’s been implemented to address the concerns in backlog/drafts/ASI1_reality_check_corrected.md and guide further contributions.
- Provide exact file paths, env toggles, run commands, and next steps. Written for code contributors (Claude Code) to move fast and safely.

Executive Summary
- Addressed reality‑check concerns (format mismatch, strict scoring undercount, precision issues) by adding a pluggable GraphJudge filter with optional distillation, plus conservative normalization improvements.
- Clear, per‑example diffs and aggregate summaries show progress; gray‑zone logging + LM Studio labeler enable a self‑improving loop without touching the hot path latency significantly.

What We Shipped (code + files)
- Pluggable Judge module
  - File: server/core/memory/judge.py
  - Provides `GraphJudge` with two backends:
    - Lite scorer: hand‑tuned linear features (lexicalized relation, object length/contentfulness, generic/pronoun objects, minimal NER/type hints, soft schemas)
    - Distilled scorer: logistic weights loaded from JSON (env‑gated)
  - Language label map for EN/DE/FR/ES/IT to recognize PERSON/ORG/LOC across spaCy variants.
  - Relation schema registry: loads optional JSON of relation→expected object types; soft prior only.

- YAML extractor integration
  - File: server/core/memory/extractors/yaml_extractor.py
  - Calls `GraphJudge` in refine; includes gray‑zone logging (env‑gated) without touching extraction.

- Conservative normalization improvements
  - Nominalizations: “increase of X [by Y]” variants; broadened risk/responsibility heads.
  - Verb→prep lexicalization: apply_to, consist_of/in + keep/drop policy alignment.
  - Punctuation‑aware lists and relative clause pied‑piping case support.
  - Files: server/core/memory/extractors/yaml_runtime.py

- Tooling
  - Distilled judge trainer (dataset gold): server/scripts/train_graph_judge.py
  - Distilled judge trainer (LM‑labels): server/scripts/train_graph_judge_from_labels.py
  - Gray‑zone collector + distill: server/scripts/judge_collect_and_distill.py
  - LM Studio labeler (OpenAI‑compatible): server/scripts/llm_judge_labeler.py
  - Nightly orchestrator: server/scripts/judge_update_nightly.py
  - Side‑by‑side diff reporter: server/scripts/eval_extraction_diff.py
  - Diff summary builder: server/scripts/build_diff_summary.py

- Config
  - server/.env enables distilled judge + gray‑zone logging by default and points LLM judge to LM Studio:
    - `YAML_GRAPH_JUDGE=on`
    - `YAML_GRAPH_JUDGE_MODEL=models/graph_judge.json`
    - `YAML_GRAPH_JUDGE_GRAY_BAND=0.10`
    - `YAML_GRAPH_JUDGE_GRAYZONE_LOG=data/judge_grayzone.jsonl`
    - `YAML_GRAPH_JUDGE_SCHEMA=models/graph_judge_schema.json`
    - `LLM_JUDGE_BASE_URL=http://127.0.0.1:1234/v1`
    - `LLM_JUDGE_MODEL=llama-3.2-3b-instruct`
  - Relation schema sample: server/models/graph_judge_schema.json

How This Addresses the Reality‑Check Concerns
- Format mismatch (lexicalized vs verb+prep)
  - Judge favors lexicalized relations and penalizes low‑content objects; normalization added for apply/consist and nominalizations; diff reporter exposes mismatches explicitly.
- Strict scoring undercounts semantically correct edges
  - We added a precision gate (judge) rather than relaxed scoring, to boost F1 via better precision; relaxed scoring remains a future diagnostic if needed.
- Missing patterns/coverage
  - Incremental additions in yaml_runtime (nominals, verb→prep, pied‑piping). The plan keeps YAML for simple/high‑precision cases and uses judge + optional LLM‑judge only where it matters.
- Hybrid/async path
  - Gray‑zone logging + LM Studio labeler + distillation provide an idle‑time improvement loop; keeps hot path fast and deterministic.

Results (strict PRF; transformer profile)
- L1 medium (caps off): F1 ≈ 0.452 with distilled judge (vs ≈0.368–0.389 raw)
- L1 long (caps on): F1 ≈ 0.345 with distilled judge (vs ≈0.303–0.312 raw)
- Diffs: docs/reports/diff_l1_medium.md, docs/reports/diff_l1_long.md
- Summary: docs/reports/diff_l1_summary.md

Runbook (contributor quick start)
- Evals (distilled judge default ON)
  - Medium caps off:
    - `cd server && SPACY_MODEL_EN=en_core_web_trf YAML_DENSITY_CAPS=off YAML_NOMINALS=on .venv/bin/python -m scripts.eval_extraction --dataset tests/data/yaml_eval_l1_en_medium.json --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml --lang en`
  - Long caps on:
    - `cd server && SPACY_MODEL_EN=en_core_web_trf YAML_DENSITY_CAPS=on YAML_NOMINALS=on .venv/bin/python -m scripts.eval_extraction --dataset tests/data/yaml_eval_l1_en_long.json --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml --lang en`
- Diffs + summary:
  - `cd server && SPACY_MODEL_EN=en_core_web_trf YAML_NOMINALS=on .venv/bin/python -m scripts.build_diff_summary`

Self‑Improving Loop (idle time)
1) Collect gray‑zone cases (default ON)
   - `data/judge_grayzone.jsonl` accumulates uncertain triples while you use the system.
2) Label gray‑zone with LM Studio
   - `cd server && LLM_JUDGE_BASE_URL=http://127.0.0.1:1234/v1 LLM_JUDGE_MODEL=llama-3.2-3b-instruct .venv/bin/python -m scripts.llm_judge_labeler --log data/judge_grayzone.jsonl --out data/judge_labels.jsonl`
3) Distill a new judge (two options)
   - From labels: `.venv/bin/python -m scripts.train_graph_judge_from_labels --labels data/judge_labels.jsonl --yaml archive/.../ASI1_index_v0_9.yaml --out models/graph_judge.json --keep_rate 0.35`
   - Supervised from dataset: `.venv/bin/python -m scripts.judge_collect_and_distill --log data/judge_grayzone.jsonl --dataset tests/data/yaml_eval_l1_en_medium.json --yaml archive/.../ASI1_index_v0_9.yaml --out models/graph_judge.json --auto_calibrate`
4) Nightly orchestration (example)
   - `python -m scripts.judge_update_nightly --log data/judge_grayzone.jsonl --labels data/judge_labels.jsonl --yaml archive/.../ASI1_index_v0_9.yaml --out models/graph_judge.json --keep_rate 0.35`

Language Support Hooks
- LanguageLabelMap in judge.py provides PERSON/ORG/LOC recognition across EN/DE/FR/ES/IT; uses doc.lang_ (override via `YAML_GRAPH_JUDGE_LANG_OVERRIDE`).
- Extend server/models/graph_judge_schema.json per relation with language-agnostic NER labels when needed.
- When we enable non‑EN pipelines, the judge features remain effective with minimal changes.

Next Steps (for contributors)
- Schema expansion: add domain extensions (finance/medical) in `server/models/graph_judge_schema.json`.
- Pattern coverage: implement remaining L1 patterns and selective L2/L3 stubs in `yaml_runtime.py` guided by diff misses.
- Relaxed diagnostic scoring (optional): add an env‑gated relaxed scorer to compare semantic equivalences alongside strict PRF.
- Ops visibility: small CLI/endpoint to print current judge status (model path, threshold, schema path, gray‑zone stats).

Notes on Tests
- Several integration tests unrelated to extraction/judge currently fail (audio flows/emotion/prosody/provenance ordering, MLX TTS). Memory/extraction unit tests pass and judge/eval flows are green. Keep scope tight when iterating on judge/extraction.

