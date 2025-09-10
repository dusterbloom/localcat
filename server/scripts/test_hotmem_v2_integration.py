#!/usr/bin/env python3
"""
Comprehensive HotMem v2 integration test.

Tests LLM-assisted relation extraction with various sentence patterns
and compares performance against other extraction methods.

Usage:
  source server/.venv/bin/activate
  python scripts/test_hotmem_v2_integration.py

Options:
  --json                     Print machine-readable JSON summary
  --no-llm-assisted          Disable LLM-assisted extraction
  --no-relik                 Disable ReLiK
  --no-srl                   Disable SRL
  --dry                      Do not write to store
  --compare                  Compare extraction methods side-by-side
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

from components.memory.memory_store import MemoryStore, Paths
from components.memory.memory_hotpath import HotMemory, _load_nlp
from components.memory.memory_intent import get_intent_classifier, get_quality_filter


def fmt_ms(ms: float) -> str:
    return f"{ms:.1f}ms"


def get_test_sentences() -> List[Dict[str, Any]]:
    """Return diverse test sentences covering various relation types."""
    return [
        {
            "text": "Tim Cook is the CEO of Apple and lives in California.",
            "entities": ["tim cook", "apple", "california"],
            "expected_triples": [
                ("tim cook", "ceo_of", "apple"),
                ("tim cook", "lives_in", "california")
            ],
            "category": "employment_location"
        },
        {
            "text": "Sarah works at Microsoft as a senior engineer and develops Windows.",
            "entities": ["sarah", "microsoft", "windows"],
            "expected_triples": [
                ("sarah", "works_for", "microsoft"),
                ("sarah", "develops", "windows")
            ],
            "category": "employment_development"
        },
        {
            "text": "Elon Musk founded Tesla and SpaceX in the early 2000s.",
            "entities": ["elon musk", "tesla", "spacex"],
            "expected_triples": [
                ("elon musk", "founder_of", "tesla"),
                ("elon musk", "founder_of", "spacex")
            ],
            "category": "founding_temporal"
        },
        {
            "text": "My brother Tom lives in Portland and teaches at Reed College.",
            "entities": ["tom", "portland", "reed college"],
            "expected_triples": [
                ("tom", "lives_in", "portland"),
                ("tom", "teaches_at", "reed college")
            ],
            "category": "family_education"
        },
        {
            "text": "Jennifer is married to Michael and they have two children.",
            "entities": ["jennifer", "michael"],
            "expected_triples": [
                ("jennifer", "married_to", "michael")
            ],
            "category": "personal_relationships"
        },
        {
            "text": "Amazon acquired Whole Foods in 2017 for $13.7 billion.",
            "entities": ["amazon", "whole foods"],
            "expected_triples": [
                ("amazon", "acquired", "whole foods")
            ],
            "category": "business_acquisition"
        },
        {
            "text": "The company is headquartered in New York and competes with Google.",
            "entities": ["new york", "google"],
            "expected_triples": [
                ("company", "headquartered_in", "new york"),
                ("company", "competes_with", "google")
            ],
            "category": "business_competition"
        },
        {
            "text": "Dr. Smith retired from Harvard University in 2020 after 30 years.",
            "entities": ["dr. smith", "harvard university"],
            "expected_triples": [
                ("dr. smith", "retired_from", "harvard university")
            ],
            "category": "retirement_temporal"
        },
        {
            "text": "Microsoft develops Windows and Office, while Apple creates macOS and iOS.",
            "entities": ["microsoft", "windows", "office", "apple", "macos", "ios"],
            "expected_triples": [
                ("microsoft", "develops", "windows"),
                ("microsoft", "develops", "office"),
                ("apple", "creates", "macos"),
                ("apple", "creates", "ios")
            ],
            "category": "product_development"
        },
        {
            "text": "The startup moved from San Francisco to Austin last year.",
            "entities": ["san francisco", "austin"],
            "expected_triples": [
                ("startup", "moved_from", "san francisco"),
                ("startup", "moved_to", "austin")
            ],
            "category": "location_change"
        }
    ]


def test_extraction_methods(hm: HotMemory, text: str, entities: List[str], 
                           args: argparse.Namespace) -> Dict[str, Any]:
    """Test different extraction methods on a single sentence."""
    lang = 'en'
    
    # Stage: intent
    t0 = time.perf_counter()
    intent_classifier = get_intent_classifier()
    intent = intent_classifier.analyze(text, lang)
    t_intent = (time.perf_counter() - t0) * 1000

    # Stage: spaCy doc
    t0 = time.perf_counter()
    nlp = _load_nlp(lang)
    doc = nlp(text) if nlp else None
    t_doc = (time.perf_counter() - t0) * 1000

    # Stage: ReLiK
    relik_triples: List[Tuple[str, str, str, float]] = []
    t_relik = 0.0
    if getattr(hm, 'use_relik', False) and getattr(hm, '_relik', None) is not None:
        if len(text) <= int(os.getenv('HOTMEM_RELIK_MAX_CHARS', '480')):
            t0 = time.perf_counter()
            try:
                relik_triples = hm._relik.extract(text) or []
            except Exception:
                relik_triples = []
            t_relik = (time.perf_counter() - t0) * 1000

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
                from components.processing.semantic_roles import SRLExtractor
                hm._srl = SRLExtractor(use_normalizer=True)
            t0 = time.perf_counter()
            preds = hm._srl.doc_to_predications(doc, lang) if doc is not None else []
            srl_triples = hm._srl.predications_to_triples(preds)
            t_srl = (time.perf_counter() - t0) * 1000
        except Exception:
            srl_triples = []

    # Stage: LLM-assisted extraction
    llm_triples: List[Tuple[str, str, str]] = []
    t_llm = 0.0
    if getattr(hm, 'assisted_enabled', False) and not args.no_llm_assisted:
        t0 = time.perf_counter()
        try:
            # Use the actual HotMem LLM-assisted extraction
            result = hm._assist_extract(text, entities, ud_triples)
            if result and 'triples' in result:
                for triple in result['triples']:
                    llm_triples.append((triple['s'], triple['r'], triple['d']))
        except Exception as e:
            print(f"  LLM-assisted extraction failed: {e}")
            import traceback
            traceback.print_exc()
            llm_triples = []
        t_llm = (time.perf_counter() - t0) * 1000

    return {
        'intent_time': t_intent,
        'doc_time': t_doc,
        'relik_time': t_relik,
        'ud_time': t_ud,
        'srl_time': t_srl,
        'llm_time': t_llm,
        'relik_triples': relik_triples,
        'ud_triples': ud_triples,
        'srl_triples': srl_triples,
        'llm_triples': llm_triples,
        'intent': getattr(getattr(intent, 'intent', None), 'name', str(intent.intent if intent else ''))
    }


def calculate_metrics(expected: List[Tuple[str, str, str]], 
                     actual: List[Tuple[str, str, str]]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    expected_set = set(expected)
    actual_set = set(actual)
    
    true_positives = len(expected_set.intersection(actual_set))
    false_positives = len(actual_set - expected_set)
    false_negatives = len(expected_set - actual_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--no-llm-assisted', action='store_true')
    ap.add_argument('--no-relik', action='store_true')
    ap.add_argument('--no-srl', action='store_true')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--compare', action='store_true', help='Compare methods side-by-side')
    args = ap.parse_args()

    # Load .env
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)

    # Use temp store if dry
    if args.dry:
        tmp_sql = os.path.abspath(os.path.join('/tmp', 'hotmem_v2_test.db'))
        tmp_lmdb = os.path.abspath(os.path.join('/tmp', 'hotmem_v2_test_graph.lmdb'))
        paths = Paths(sqlite_path=tmp_sql, lmdb_dir=tmp_lmdb)
    else:
        paths = Paths()

    store = MemoryStore(paths=paths)
    hm = HotMemory(store)

    # Override feature flags
    if args.no_relik:
        hm.use_relik = False
    if args.no_srl:
        hm.use_srl = False

    # Check LLM-assisted extraction status
    print(f"LLM-assisted extraction enabled: {getattr(hm, 'assisted_enabled', False)}")
    print(f"LLM-assisted model: {os.getenv('HOTMEM_LLM_ASSISTED_MODEL', 'None')}")
    print(f"LLM-assisted base URL: {os.getenv('HOTMEM_LLM_ASSISTED_BASE_URL', 'None')}")
    
    # Prewarm
    hm.prewarm('en')

    # Get test sentences
    test_cases = get_test_sentences()
    results = []

    print("🧪 HotMem v2 Integration Test")
    print("=" * 70)
    print(f"Testing {len(test_cases)} sentences with multiple extraction methods...\n")

    for i, case in enumerate(test_cases, 1):
        print(f"{i}. '{case['text']}'")
        print(f"   Category: {case['category']}")
        print(f"   Expected: {case['expected_triples']}")
        
        # Test extraction methods
        extraction_results = test_extraction_methods(hm, case['text'], case['entities'], args)
        
        # Calculate metrics for each method
        metrics = {}
        methods = ['relik', 'ud', 'srl', 'llm']
        for method in methods:
            actual_triples = []
            for triple in extraction_results[f'{method}_triples']:
                if len(triple) >= 3:
                    actual_triples.append((triple[0], triple[1], triple[2]))
            metrics[method] = calculate_metrics(case['expected_triples'], actual_triples)
        
        if args.compare:
            print(f"   Performance Comparison:")
            for method in methods:
                m = metrics[method]
                print(f"     {method.upper()}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")
        else:
            # Just show LLM performance
            llm_metrics = metrics['llm']
            print(f"   LLM-assisted: P={llm_metrics['precision']:.2f} R={llm_metrics['recall']:.2f} F1={llm_metrics['f1']:.2f}")
            
            if extraction_results['llm_triples']:
                print(f"   LLM Triples: {extraction_results['llm_triples']}")
        
        # Store results
        case_result = {
            'text': case['text'],
            'category': case['category'],
            'expected': case['expected_triples'],
            'extraction_results': extraction_results,
            'metrics': metrics,
            'timings': {
                'intent': extraction_results['intent_time'],
                'doc': extraction_results['doc_time'],
                'relik': extraction_results['relik_time'],
                'ud': extraction_results['ud_time'],
                'srl': extraction_results['srl_time'],
                'llm': extraction_results['llm_time']
            }
        }
        results.append(case_result)
        
        print()

    # Summary statistics
    if not args.json:
        print("=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        
        for method in methods:
            avg_precision = sum(r['metrics'][method]['precision'] for r in results) / len(results)
            avg_recall = sum(r['metrics'][method]['recall'] for r in results) / len(results)
            avg_f1 = sum(r['metrics'][method]['f1'] for r in results) / len(results)
            avg_time = sum(r['timings'][method] for r in results) / len(results)
            
            print(f"{method.upper()}:")
            print(f"  Avg Precision: {avg_precision:.3f}")
            print(f"  Avg Recall: {avg_recall:.3f}")
            print(f"  Avg F1: {avg_f1:.3f}")
            print(f"  Avg Time: {fmt_ms(avg_time)}")
            print()

    if args.json:
        print(json.dumps(results, indent=2))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())