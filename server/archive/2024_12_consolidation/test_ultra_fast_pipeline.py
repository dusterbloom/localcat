#!/usr/bin/env python3
"""
🚀 ULTRA FAST SEMANTIC EXTRACTION PIPELINE

Eliminates all bottlenecks:
- No GLiNER (8900ms -> 0ms)
- No embedding computation (5800ms -> 0ms)
- Pure SRL semantic extraction (~50ms total)

Target: <100ms per sentence
"""

import time
import spacy
from spacy.matcher import Matcher
from components.processing.semantic_roles import SRLExtractor

# CACHE MODELS GLOBALLY - load once, reuse forever
_nlp_cache = None
_srl_cache = None

def get_cached_models():
    """Get cached models, load only once"""
    global _nlp_cache, _srl_cache

    if _nlp_cache is None:
        print("🔄 Loading spaCy transformer model (one time only)...")
        _nlp_cache = spacy.load("en_core_web_trf")

        # Add custom temporal/location matcher following your SOTA specification
        if "temporal_extractor" not in _nlp_cache.pipe_names:
            matcher = Matcher(_nlp_cache.vocab)
            # Temporal patterns for "when" relations
            temporal_patterns = [
                [{"LOWER": "when"}, {"POS": "VERB"}],
                [{"LOWER": "in"}, {"ENT_TYPE": "TIME"}],
                [{"LOWER": "at"}, {"ENT_TYPE": "TIME"}],
                [{"LOWER": "during"}, {"ENT_TYPE": "EVENT"}],
            ]
            matcher.add("TEMPORAL_REL", temporal_patterns)

            @_nlp_cache.component("temporal_extractor")
            def extract_temporal_relations(doc):
                """Extract temporal relations for TRE (Temporal Relation Extraction)"""
                matches = matcher(doc)
                # Store temporal markers in doc extensions for SRL to use
                temporal_spans = []
                for match_id, start, end in matches:
                    span = doc[start:end]
                    temporal_spans.append(span)
                doc._.temporal_relations = temporal_spans
                return doc

            # Add custom doc extension
            if not spacy.tokens.Doc.has_extension("temporal_relations"):
                spacy.tokens.Doc.set_extension("temporal_relations", default=[])

            _nlp_cache.add_pipe("temporal_extractor", after="ner")

    if _srl_cache is None:
        print("🔄 Loading SRL extractor (one time only)...")
        _srl_cache = SRLExtractor(use_normalizer=False)  # Disable slow embeddings

    return _nlp_cache, _srl_cache

def fast_extraction(text: str):
    """Ultra fast semantic extraction without bottlenecks"""
    start_total = time.perf_counter()

    # Use cached models
    nlp, srl = get_cached_models()

    # Parse with spaCy (includes NER)
    start = time.perf_counter()
    doc = nlp(text)
    spacy_ms = (time.perf_counter() - start) * 1000

    # Extract entities (free with spaCy)
    entities = [ent.text.lower() for ent in doc.ents]

    # Extract semantic relations (fast)
    start = time.perf_counter()
    predications = srl.doc_to_predications(doc)
    pred_ms = (time.perf_counter() - start) * 1000

    # Convert to triples (no embeddings = fast)
    start = time.perf_counter()
    triples = srl.predications_to_triples(predications)
    triple_ms = (time.perf_counter() - start) * 1000

    total_ms = (time.perf_counter() - start_total) * 1000

    return {
        'entities': entities,
        'triples': triples,
        'timing': {
            'spacy': spacy_ms,
            'predications': pred_ms,
            'triples': triple_ms,
            'total': total_ms
        }
    }

def main():
    sentences = [
        "Alice feeds the cat in the morning.",
        "The tall boy who lives in Rome often plays piano in the evenings.",
        "The CEO announced that the company would restructure after declining profits.",
        "After the festival ended in July 2022, Maria, a renowned chef from Barcelona, moved to Paris where she began teaching at a culinary school while writing her memoirs."
    ]

    print("🚀 ULTRA FAST SEMANTIC EXTRACTION PIPELINE")
    print("=" * 60)

    total_time = 0

    for i, text in enumerate(sentences, 1):
        print(f"\n{i}. {text}")
        print("-" * 40)

        result = fast_extraction(text)
        timing = result['timing']
        total_time += timing['total']

        print(f"⚡ Entities: {len(result['entities'])} -> {result['entities']}")
        print(f"🎯 Semantic: {len(result['triples'])} -> {result['triples']}")
        print(f"⏱️  Timing: spaCy={timing['spacy']:.1f}ms | SRL={timing['predications']:.1f}ms | Triples={timing['triples']:.1f}ms")
        print(f"🏆 TOTAL: {timing['total']:.1f}ms")

        if timing['total'] > 100:
            print("❌ Still too slow!")
        else:
            print("✅ FAST ENOUGH!")

    avg_time = total_time / len(sentences)
    print(f"\n🎯 AVERAGE TIME: {avg_time:.1f}ms per sentence")
    print(f"🚀 SPEEDUP: {3000/avg_time:.1f}x faster than before!")

    if avg_time < 100:
        print("🏆 SUCCESS: Ready for real-time processing!")
    else:
        print("⚠️  Still needs optimization")

if __name__ == '__main__':
    main()