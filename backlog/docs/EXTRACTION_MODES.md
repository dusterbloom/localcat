# Extraction Modes

This project routes extraction through the strategy registry by default and sets `enhanced_level3` as the default extractor. Use the environment to switch modes or tune quality/speed trade‑offs.

## Defaults
- DEFAULT_EXTRACTION_STRATEGY: `enhanced_level3`
- FALLBACK_EXTRACTION_STRATEGY: empty (no fallback)
- Registry routing: `HOTMEM_ROUTE_TO_REGISTRY=true`

## Enhanced Level3 (spaCy small)
- Model: `en_core_web_sm`
- Traits: very fast (≈5–20ms short texts), conservative triples, clean predicates (`verb` or `verb_prep`).
- Suggested env (already the defaults):
```
ENHANCED_LEVEL3_SPACY_MODEL=en_core_web_sm
ENHANCED_LEVEL3_ENTITY_CONF=0.70
ENHANCED_LEVEL3_RELATION_CONF=0.65
ENHANCED_LEVEL3_TARGET_ENT=50
ENHANCED_LEVEL3_TARGET_REL=30
```

## Enhanced Level3 (Transformer Quality Preset)
- Model: `en_core_web_trf` (set `ENHANCED_LEVEL3_SPACY_MODEL=en_core_web_rtf`, alias maps to trf)
- Traits: higher recall on long/complex sentences; moderate latency (≈35–130ms).
- Suggested env:
```
ENHANCED_LEVEL3_SPACY_MODEL=en_core_web_rtf
ENHANCED_LEVEL3_ENTITY_CONF=0.65
ENHANCED_LEVEL3_RELATION_CONF=0.55
ENHANCED_LEVEL3_TARGET_REL=40
ENHANCED_LEVEL3_EXTRA_VERBS=argue,weigh,consider,foster,prove,seek,reveal,mitigate,preserve
```

## ASI1 / ASI2 (YAML)
- Strategies: `asi1`, `asi2`.
- Use when you want canonical pattern behavior or to validate coverage; slower than Enhanced Level3 quality path.

## Other strategies
- `enhanced_hotmem`, `ud`, `lightweight`, `hybrid`, `multilingual`, `pattern`: available for experiments; not default.

## Predicate and Object Conventions
- Predicates are lowercase, short, and may be composed: `verb` or `verb_prep` (e.g., `watch_from`, `watch_under`, `chase_across`).
- Store `verb` and `prep` in edge properties alongside `confidence`, `source`, `sentence_id`, and SRL roles.
- Object NPs are cleaned to exclude nested prepositional content (e.g., `wooden benches` vs `wooden benches under tall oak trees`). When nested context matters, the extractor emits an additional relation (e.g., `watch_under -> tall oak trees`).

## Provenance
- Edges extracted via registry include `prov=registry:<strategy>`; additional properties may carry `original_predicate` and normalization metadata.

## Switching modes
1) Edit `server/.env` (copy from `server/config/env.example`).
2) Set `DEFAULT_EXTRACTION_STRATEGY` and Enhanced Level3 envs as needed.
3) Restart the server.

