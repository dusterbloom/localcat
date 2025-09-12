#!/usr/bin/env python3
"""
Interactive HotMem pipeline inspector.

Run a single sentence through each extraction stage and report:
- Timings per stage (ReLiK, SRL, UD, entity map/ONNX NER, refinement, coref)
- Triples produced at each stage
- Store decisions (should_store + confidence) per refined triple

Usage:
  source server/.venv/bin/activate
  python scripts/inspect_pipeline.py "My brother Tom lives in Portland and teaches at Reed College."

Options:
  --lang en                  Language hint (default: en)
  --no-relik                 Disable ReLiK even if enabled in .env
  --no-srl                   Disable SRL even if enabled in .env
  --no-onnx-ner              Disable ONNX NER enrichment
  --no-coref                 Disable neural coref
  --dry                      Do not write to store (skip observe_edge)
  --json                     Print machine-readable JSON summary at the end

Notes:
- Loads flags from server/.env and lets CLI switches override selectively.
- Uses the same internal helpers as HotMem, but keeps steps separate for clarity.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.memory.memory_store import MemoryStore, Paths  # type: ignore
from components.memory.memory_hotpath import HotMemory, _load_nlp  # type: ignore
from components.memory.memory_intent import get_intent_classifier, get_quality_filter  # type: ignore


def fmt_ms(ms: float) -> str:
    return f"{ms:.1f}ms"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('text', help='Input sentence/text to inspect')
    ap.add_argument('--lang', default='en')
    ap.add_argument('--no-relik', action='store_true')
    ap.add_argument('--no-srl', action='store_true')
    ap.add_argument('--no-onnx-ner', action='store_true')
    ap.add_argument('--no-coref', action='store_true')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    # Load .env
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)

    text = args.text
    lang = args.lang

    # Use temp store if dry
    if args.dry:
        tmp_sql = os.path.abspath(os.path.join('/tmp', 'hotmem_inspect_memory.db'))
        tmp_lmdb = os.path.abspath(os.path.join('/tmp', 'hotmem_inspect_graph.lmdb'))
        paths = Paths(sqlite_path=tmp_sql, lmdb_dir=tmp_lmdb)
    else:
        paths = Paths()

    store = MemoryStore(paths=paths)
    hm = HotMemory(store)

    # Override feature flags per CLI
    if args.no_relik:
        hm.use_relik = False
    if args.no_srl:
        hm.use_srl = False
    if args.no_onnx_ner:
        hm.use_onnx_ner = False
    if args.no_coref:
        hm.use_coref = False

    # Prewarm
    hm.prewarm(lang)

    # Stage: intent
    t0 = time.perf_counter()
    intent_classifier = get_intent_classifier()
    intent = intent_classifier.analyze(text, lang)
    t_intent = (time.perf_counter() - t0) * 1000

    # Stage: ReLiK (direct)
    relik_triples: List[Tuple[str, str, str, float]] = []
    t_relik = 0.0
    if getattr(hm, 'use_relik', False) and getattr(hm, '_relik', None) is not None:
        # Keep a short gate to avoid huge texts in demo
        if len(text) <= int(os.getenv('HOTMEM_RELIK_MAX_CHARS', '480')):
            t0 = time.perf_counter()
            try:
                relik_triples = hm._relik.extract(text) or []  # type: ignore
            except Exception as e:
                relik_triples = []
            t_relik = (time.perf_counter() - t0) * 1000

    # Stage: spaCy doc
    t0 = time.perf_counter()
    nlp = _load_nlp(lang)
    doc = nlp(text) if nlp else None
    t_doc = (time.perf_counter() - t0) * 1000

    # Stage: UD entity map + ONNX NER enrichment timing
    t_entmap = 0.0
    ent_map = {}
    entities_pre: List[str] = []
    if doc is not None:
        t0 = time.perf_counter()
        # Use internal builder (will include ONNX NER enrichment if enabled)
        entities_set = set()
        ent_map = hm._build_entity_map(doc, entities_set)
        entities_pre = sorted(list(entities_set))
        t_entmap = (time.perf_counter() - t0) * 1000

    # Stage: UD 27 patterns
    t0 = time.perf_counter()
    ud_entities, ud_triples, ud_neg = hm._extract_from_doc(doc) if doc is not None else ([], [], 0)
    t_ud = (time.perf_counter() - t0) * 1000

    # Stage: SRL
    srl_triples: List[Tuple[str, str, str]] = []
    t_srl = 0.0
    if getattr(hm, 'use_srl', False):
        try:
            if hm._srl is None:
                from components.processing.semantic_roles import SRLExtractor  # type: ignore
                hm._srl = SRLExtractor(use_normalizer=True)
            t0 = time.perf_counter()
            preds = hm._srl.doc_to_predications(doc, lang) if doc is not None else []
            srl_triples = hm._srl.predications_to_triples(preds)
            t_srl = (time.perf_counter() - t0) * 1000
        except Exception:
            srl_triples = []

    # Combine pre-refine
    combined_pre = []
    for s, r, d, *_ in relik_triples:
        combined_pre.append((s, r, d))
    combined_pre.extend(srl_triples)
    combined_pre.extend(ud_triples)
    # Dedup preserving order
    seen = set()
    pre_refine = []
    for t in combined_pre:
        if t not in seen:
            pre_refine.append(t)
            seen.add(t)

    # Stage: refinement
    t0 = time.perf_counter()
    refined = hm._refine_triples(text, pre_refine, doc, intent, lang)
    t_refine = (time.perf_counter() - t0) * 1000

    # Stage: coref
    t0 = time.perf_counter()
    if getattr(hm, 'use_coref', False) and getattr(hm, '_coref_model', None) is not None:
        try:
            refined_coref = hm._apply_coref_neural(refined, doc)
        except Exception:
            refined_coref = hm._apply_coref_lite(refined, doc)
    else:
        refined_coref = hm._apply_coref_lite(refined, doc)
    t_coref = (time.perf_counter() - t0) * 1000

    # Stage: store decisions
    qf = get_quality_filter()
    decisions: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    t_lower = (text or '').lower()
    hedge_terms = ("maybe", "i think", "probably", "kinda", "sort of", "not sure", "perhaps", "possibly")
    hedged = any(ht in t_lower for ht in hedge_terms)
    for (s, r, d) in refined_coref:
        should, conf = qf.should_store_fact(s, r, d, intent)
        if hedged:
            conf = max(0.0, conf - 0.2)
        decisions.append({
            'triple': (s, r, d),
            'should_store': bool(should and conf >= float(os.getenv('HOTMEM_CONFIDENCE_THRESHOLD', '0.3'))),
            'confidence': round(conf, 3)
        })
    t_store = (time.perf_counter() - t0) * 1000

    # TEST RETRIEVAL FIX: Test the _retrieve_context method
    print("\n=== TESTING RETRIEVAL FIX ===")
    print("Testing _retrieve_context method with the extracted entities...")
    
    # Add some test data to the memory store first
    test_triples = [
        ("Steve Jobs", "founded", "Apple"),
        ("Steve Jobs", "co-founded", "Apple"),
        ("you", "has", "brother"),
        ("brother", "lives_in", "Portland"),
        ("brother", "teaches_at", "Reed College"),
        ("Marie Curie", "discovered", "radium"),
        ("Elon Musk", "works_at", "Tesla"),
    ]
    
    print(f"Adding {len(test_triples)} test triples to memory...")
    for s, r, d in test_triples:
        if not args.dry:
            store.observe_edge(s, r, d, conf=0.8, now_ts=int(time.time() * 1000))
            # Add to hot memory entity index
            hm.entity_index[s].add((s, r, d))
            hm.entity_index[d].add((s, r, d))
            # Add metadata
            hm.edge_meta[(s, r, d)] = {'ts': int(time.time() * 1000), 'weight': 0.8}
    
    # Extract entities from the input text for retrieval test
    extracted_entities = list(entities_pre) if entities_pre else []
    if not extracted_entities:
        # Fallback: use entities from refined triples
        for s, r, d in refined_coref:
            if s not in extracted_entities:
                extracted_entities.append(s)
            if d not in extracted_entities:
                extracted_entities.append(d)
    
    print(f"Extracted entities for retrieval: {extracted_entities}")
    
    # Test retrieval with the extracted entities
    try:
        t_retrieve_start = time.perf_counter()
        retrieved_bullets = hm._retrieve_context(text, extracted_entities, turn_id=1, intent=intent)
        t_retrieve = (time.perf_counter() - t_retrieve_start) * 1000
        
        print(f"\n✅ RETRIEVAL TEST RESULTS:")
        print(f"  - Retrieval time: {fmt_ms(t_retrieve)}")
        print(f"  - Retrieved bullets: {len(retrieved_bullets)}")
        
        if retrieved_bullets:
            print("  - Bullet content:")
            for i, bullet in enumerate(retrieved_bullets, 1):
                print(f"    {i}. {bullet}")
        else:
            print("  - No bullets retrieved (this was the bug!)")
            
        # Test that our all_edges fix works
        print(f"\n🔍 DEBUG INFO:")
        total_edges = sum(len(triples) for triples in hm.entity_index.values())
        print(f"  - Total edges in memory: {total_edges}")
        print(f"  - Entity index size: {len(hm.entity_index)}")
        
        # Detailed entity analysis
        print(f"\n📊 ENTITY ANALYSIS:")
        for entity in extracted_entities:
            if entity in hm.entity_index:
                entity_edges = hm.entity_index[entity]
                print(f"  - {entity}: {len(entity_edges)} edges")
                for s, r, d in entity_edges:
                    print(f"    - ({s}, {r}, {d})")
            else:
                print(f"  - {entity}: 0 edges")
        
        # Test LEANN processing specifically
        if hm.use_leann and len(text.strip()) >= 2:
            print("  - LEANN is enabled - testing all_edges fix...")
            try:
                # This should not crash anymore
                all_edges = set()
                for entity_triples_list in hm.entity_index.values():
                    all_edges.update(entity_triples_list)
                print(f"  - all_edges successfully created: {len(all_edges)} edges")
                print("  - ✅ all_edges fix is working!")
            except Exception as e:
                print(f"  - ❌ all_edges fix failed: {e}")
        else:
            print("  - LEANN is disabled, but fix is still important for future use")
            
    except Exception as e:
        print(f"❌ RETRIEVAL TEST FAILED: {e}")
        print("This indicates the retrieval bug is NOT fixed!")
        return 1

    # Print human summary
    summary = {
        'text': text,
        'lang': lang,
        'intent': getattr(getattr(intent, 'intent', None), 'name', str(intent.intent if intent else '')),
        'timings_ms': {
            'intent': t_intent,
            'relik': t_relik,
            'spacy_doc': t_doc,
            'entity_map': t_entmap,
            'ud_27': t_ud,
            'srl': t_srl,
            'refine': t_refine,
            'coref': t_coref,
            'store_decisions': t_store,
        },
        'entities_pre_enrichment': entities_pre,
        'relik_triples': relik_triples,
        'srl_triples': srl_triples,
        'ud_triples': ud_triples,
        'pre_refine_triples': pre_refine,
        'refined_triples': refined,
        'coref_triples': refined_coref,
        'decisions': decisions,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\n=== Pipeline Inspector ===")
        print(f"Text: {text}\nLang: {lang}\nIntent: {summary['intent']}")
        tm = summary['timings_ms']
        print("\nTimings:")
        for k in ['intent','relik','spacy_doc','entity_map','ud_27','srl','refine','coref','store_decisions']:
            print(f"  - {k}: {fmt_ms(tm[k])}")
        print("\nEntities (pre-enrichment):", summary['entities_pre_enrichment'])
        print("\nReLiK triples:")
        for t in summary['relik_triples']:
            print("  ", t)
        print("\nSRL triples:")
        for t in summary['srl_triples']:
            print("  ", t)
        print("\nUD triples:")
        for t in summary['ud_triples']:
            print("  ", t)
        print("\nPre-refine (dedup merged):")
        for t in summary['pre_refine_triples']:
            print("  ", t)
        print("\nRefined:")
        for t in summary['refined_triples']:
            print("  ", t)
        print("\nCoref:")
        for t in summary['coref_triples']:
            print("  ", t)
        print("\nStore decisions:")
        for d in summary['decisions']:
            print(f"  {d['triple']} -> store={d['should_store']} conf={d['confidence']}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

