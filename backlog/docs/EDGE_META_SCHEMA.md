# Edge Metadata Schema (HotMem)

Edge metadata is stored in SQLite table `edge_meta(id, src, rel, dst, prov, lang, span, props, updated_at)` and complements the primary edge record (`edge`). This document defines conventions for fields and recommended keys in `props`.

## Columns
- src, rel, dst: identify the edge (same as in `edge`)
- prov: provenance source (e.g., `registry:enhanced_level3`, `ud_only`, `srl_ud`, `relik`)
- lang: language tag for extraction (e.g., `en`)
- span: optional free‑form span or indices if needed
- props: JSON object with additional metadata
- updated_at: Unix epoch seconds (int)

## props JSON: Reserved keys
- verb: base verb lemma for the relation (e.g., `watch`)
- prep: preposition if predicate is composed (e.g., `from`, `under`)
- original_predicate: original predicate before normalization
- normalized_relation: post‑normalization predicate (e.g., `watch_from`)
- rel_embedding: vector or compressed representation for relation semantics
- sentence_id: zero‑based sentence index in the processed doc
- roles: SRL roles mapping, e.g., `{ "ARG0": "parents", "ARG1": "benches" }`
- confidence: extraction confidence (0..1) if available
- source: extraction strategy or component (redundant with `prov` but acceptable for cross‑tooling)

## Conventions
- Predicates: store on edge label as lowercase `verb` or `verb_prep` (e.g., `live_in`, `watch_from`). Do not collapse `verb_prep` to plain verbs — keep spatial/temporal nuance.
- Keep `verb` and `prep` in `props` alongside `normalized_relation` for flexible querying.
- Use `prov` to capture the high‑level provenance (e.g., `registry:enhanced_level3`).
- Update `updated_at` on every write to reflect freshness; TTL/archival tasks act on this timestamp.

## Lifecycle & Retention (policy sketch)
- Promotion: positive evidence increases edge `weight` (EWMA), keeps `status=1` (active)
- Demotion: negative evidence decreases `weight` until `status=0` (stale) or `<0` (archivable)
- TTL: demote long‑inactive edges (e.g., >30 days) and purge archived ones (e.g., >90 days)
- Session links: `SessionStore` links edges to sessions for audit; keep until session retention policy elapses

