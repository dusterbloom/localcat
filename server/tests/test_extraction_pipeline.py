#!/usr/bin/env python3
"""
🎯 PURE SEMANTIC EXTRACTION PIPELINE DEMO

Shows clean semantic extraction pipeline:
- GLiNER entities (entity recognition)
- 🔥 SRL semantic relations (TRUE meaning with tense preservation + embeddings)
- Semantic filter (quality assurance)
- Temporal context (time awareness)
- Coreference resolution (entity linking)

DISABLED noisy components:
❌ UD relations (syntactic noise)
❌ GLiREL relations (semantic noise)

Run:
  python server/tests/test_extraction_pipeline.py
"""

import os
import sys
import time
import os.path as op

# Ensure 'server' is on sys.path when running from repo root
_here = op.dirname(op.abspath(__file__))
_server_root = op.dirname(_here)
if _server_root not in sys.path:
    sys.path.insert(0, _server_root)

try:
    import spacy
except Exception:
    spacy = None

from components.extraction.memory_extractor import MemoryExtractor
from components.memory.config import create_config
from components.processing.semantic_roles import SRLExtractor

# Optional components
try:
    from components.extraction.gliner_extractor import GLiNERExtractor
except Exception:
    GLiNERExtractor = None

try:
    from components.extraction.glirel_extractor import GLiRELExtractor
    GLIREL_AVAILABLE = True
except Exception:
    GLiRELExtractor = None
    GLIREL_AVAILABLE = False

try:
    from services.ud_utils import extract_all_ud_patterns
except Exception:
    extract_all_ud_patterns = None

try:
    from components.semantic.semantic_filter import SemanticRelationshipFilter
except Exception:
    SemanticRelationshipFilter = None

try:
    from components.temporal.temporal_extractor import TemporalContextExtractor
except Exception:
    TemporalContextExtractor = None

try:
    from components.coreference.coreference_resolver import CoreferenceResolver
except Exception:
    CoreferenceResolver = None


def _timeit(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    return out, ms


def _canon(s: str) -> str:
    return (s or "").strip()


def build_spacy(text: str):
    if not spacy:
        return None
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
    return nlp(text)


def gliner_entities(text: str):
    if GLiNERExtractor is None:
        return []
    try:
        extractor = GLiNERExtractor()
        res, ms = _timeit(extractor.extract, text)
        return res.entities, ms
    except Exception:
        return [], 0.0


def prune_entities(extractor: MemoryExtractor, text: str, ents):
    try:
        pruned, ms = _timeit(extractor._prune_entity_strings, text, list(ents))
        return pruned, ms
    except Exception:
        return list(ents), 0.0


def ud_relations(text: str):
    if not extract_all_ud_patterns:
        return [], 0.0
    try:
        res, ms = _timeit(extract_all_ud_patterns, text)
        triples = [(r.subject, r.relation, r.object) for r in res]
        return triples, ms
    except Exception:
        return [], 0.0


def glirel_relations(text: str, entities):
    if not GLIREL_AVAILABLE:
        return [], 0.0
    try:
        # Build char-offset spans from entity strings
        spans = []
        low = text.lower()
        for e in entities:
            p = low.find(e.lower())
            if p >= 0:
                spans.append({"text": text[p:p+len(e)], "start": p, "end": p+len(e), "label": "ENTITY"})
        if not spans:
            return [], 0.0

        gle = GLiRELExtractor()
        rels, ms = _timeit(gle.extract_with_gliner_integration, text=text, gliner_result=spans, threshold=0.4)
        return rels, ms
    except Exception:
        return [], 0.0


def semantic_filter(triples, text: str):
    if not SemanticRelationshipFilter:
        return triples, triples, 0.0
    cfg = {
        'semantic_filtering_enabled': True,
        'semantic_similarity_threshold': 0.8,
        'use_spacy_fallback': True,
    }
    try:
        filt = SemanticRelationshipFilter(cfg)
        res, ms = _timeit(filt.filter_relationships, triples, text)
        return res.filtered_triples, res.removed_triples, ms
    except Exception:
        return triples, [], 0.0


def temporal_context(triples, text: str):
    if not TemporalContextExtractor:
        return 0, 0.0
    cfg = {'temporal_extraction_enabled': True, 'use_spacy_fallback': True}
    try:
        t = TemporalContextExtractor(cfg)
        res, ms = _timeit(t.extract_temporal_context, triples, text)
        return int(res.extraction_stats.get('triples_with_context', 0)), ms
    except Exception:
        return 0, 0.0


def coref_resolve(triples, doc, text: str):
    if not CoreferenceResolver:
        return triples, 0.0
    try:
        c = CoreferenceResolver({'use_coref': True, 'coref_max_entities': 24, 'coref_device': 'cpu'})
        res, ms = _timeit(c.resolve_coreferences, triples, doc, text)
        return res.resolved_triples, ms
    except Exception:
        return triples, 0.0


def demo_sentence(text: str, extractor: MemoryExtractor):
    print("\n====================")
    print(f"Text: {text}")
    doc = build_spacy(text)

    # Entities via GLiNER (fallback is spaCy if GLiNER not available)
    ents = []
    gl_ms = 0.0
    if GLiNERExtractor is not None:
        try:
            ents, gl_ms = gliner_entities(text)
        except Exception:
            ents = []
    if not ents and doc is not None:
        ents = [ent.text for ent in getattr(doc, 'ents', [])]
    print(f"Entities (GLiNER/spaCy): {len(ents)} in {gl_ms:.1f}ms -> {ents[:8]}")

    # Prune entities
    pruned, pr_ms = prune_entities(extractor, text, ents)
    print(f"Pruned entities: {len(pruned)} in {pr_ms:.1f}ms -> {pruned[:8]}")

    # SRL semantic relations (PURE SEMANTIC EXTRACTION!)
    srl, srl_ms = srl_semantic_relations(text)
    print(f"🎯 SRL semantic: {len(srl)} in {srl_ms:.1f}ms -> {srl}")

    # DISABLED: UD relations (syntactic noise)
    # ud, ud_ms = ud_relations(text)
    # print(f"UD relations: {len(ud)} in {ud_ms:.1f}ms -> {ud[:5]}")

    # DISABLED: GLiREL relations (semantic noise)
    # glr, glr_ms = glirel_relations(text, pruned)
    # print(f"GLiREL relations: {len(glr)} in {glr_ms:.1f}ms -> {glr[:5]}")

    # Use pure semantic SRL relations only
    combined = list({(s, r, d) for (s, r, d) in srl})

    # Semantic filter
    kept, removed, sf_ms = semantic_filter(combined, text)
    print(f"Semantic filter: kept={len(kept)} removed={len(removed)} in {sf_ms:.1f}ms")

    # Temporal context
    ctx_cnt, tmp_ms = temporal_context(kept, text)
    print(f"Temporal context on {ctx_cnt} triples in {tmp_ms:.1f}ms")

    # Coreference (rule/neural guarded)
    coref_triples, cr_ms = coref_resolve(kept, doc, text)
    print(f"Coref resolved {len(coref_triples)} triples in {cr_ms:.1f}ms")


def srl_semantic_relations(text: str):
    """Extract semantic relations using SRL - multi-predicate extraction"""
    try:
        srl = SRLExtractor(use_normalizer=False)  # Fast version without embeddings
        doc = build_spacy(text)
        if not doc:
            return [], 0.0

        predications = srl.doc_to_predications(doc)
        semantic_triples, ms = _timeit(srl.predications_to_triples, predications)
        return semantic_triples, ms
    except Exception:
        return [], 0.0


def main():
    # Keep GLiREL device stable for tests if desired
    os.environ.setdefault('HOTMEM_GLIREL_DEVICE', 'cpu')

    cfg = create_config()
    # Enable SRL for semantic extraction
    cfg.features.use_srl = True
    extractor = MemoryExtractor(cfg.get_extractor_config())

    sentences = [
        # Simple
        "Alice feeds the cat in the morning.",
        # Medium
        "The tall boy who lives in Rome often plays piano in the evenings.",
        # Semantic showcase (shows SRL vs UD difference)
        "The CEO announced that the company would restructure after declining profits.",
        # Advanced
        "After the festival ended in July 2022, Maria, a renowned chef from Barcelona, moved to Paris where she began teaching at a culinary school while writing her memoirs.",
    ]

    print("\n=== Extraction Pipeline Demo (non-tech sentences) ===")
    for s in sentences:
        demo_sentence(s, extractor)


if __name__ == '__main__':
    main()

