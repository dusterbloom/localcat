#!/usr/bin/env python3
"""
A/B Test: UD Patterns vs spaCy Extractor

Compare extraction quality between:
A) Current HotMem UD patterns (27 dependency patterns)
B) Pure spaCy extractor from macos-local-voice-agents

Metrics:
- Extraction accuracy (correct facts)
- False positives (nonsense facts)
- Coverage (missed important facts)
- Processing time
"""

import os
import sys
import time
import json
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import tempfile

# Add both project paths
sys.path.insert(0, '/Users/peppi/Dev/localcat/server')
sys.path.insert(0, '/Users/peppi/Dev/macos-local-voice-agents/server/memory')

from components.memory.memory_hotpath import HotMemory, _load_nlp
from components.memory.memory_store import MemoryStore, Paths
from loguru import logger

# Import the spaCy extractor from other project
try:
    from spacy_fact_extractor import HighAccuracyFactExtractor
    SPACY_AVAILABLE = True
except ImportError:
    logger.warning("Could not import spacy_fact_extractor")
    SPACY_AVAILABLE = False


@dataclass
class TestCase:
    """Test case with sentence and expected facts"""
    text: str
    expected_facts: List[Tuple[str, str, str]]  # (subject, relation, object)
    category: str


@dataclass
class ExtractionResult:
    """Results from one extraction method"""
    facts: List[Tuple[str, str, str]]
    time_ms: float
    raw_output: any = None


def normalize_fact(s: str, r: str, o: str) -> Tuple[str, str, str]:
    """Normalize a fact for comparison"""
    # Normalize subjects
    s_lower = s.lower().strip()
    if s_lower in ["i", "me", "my", "mine", "you", "your"]:
        s = "user"
    
    # Normalize relations
    r = r.lower().strip().replace("_", " ")
    
    # Normalize objects
    o = o.lower().strip()
    
    return (s, r, o)


def score_extraction(extracted: List[Tuple], expected: List[Tuple]) -> Dict:
    """Score extraction quality"""
    # Normalize all facts
    extracted_norm = {normalize_fact(*f[:3]) for f in extracted if len(f) >= 3}
    expected_norm = {normalize_fact(*f) for f in expected}
    
    # Calculate metrics
    true_positives = extracted_norm & expected_norm
    false_positives = extracted_norm - expected_norm
    false_negatives = expected_norm - extracted_norm
    
    precision = len(true_positives) / len(extracted_norm) if extracted_norm else 0
    recall = len(true_positives) / len(expected_norm) if expected_norm else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': len(true_positives),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'total_extracted': len(extracted_norm),
        'total_expected': len(expected_norm)
    }


def extract_with_ud_patterns(text: str) -> ExtractionResult:
    """Extract using HotMem's UD patterns"""
    start = time.perf_counter()
    
    # Create temporary store
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = Paths(
            sqlite_path=os.path.join(tmpdir, 'test.db'),
            lmdb_dir=os.path.join(tmpdir, 'test.lmdb')
        )
        store = MemoryStore(paths)
        hm = HotMemory(store)
        
        # Use internal extraction
        nlp = _load_nlp('en')
        doc = nlp(text)
        
        # Extract using UD patterns
        entities, triples, _ = hm._extract_from_doc(doc)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return ExtractionResult(
            facts=triples,
            time_ms=elapsed_ms,
            raw_output={'entities': list(entities), 'triples': triples}
        )


def extract_with_spacy_extractor(text: str) -> ExtractionResult:
    """Extract using spaCy extractor from other project"""
    if not SPACY_AVAILABLE:
        return ExtractionResult(facts=[], time_ms=0)
    
    start = time.perf_counter()
    
    extractor = HighAccuracyFactExtractor()
    facts_dicts = extractor.extract_facts(text)
    
    # Convert to triples format
    facts = []
    for fd in facts_dicts:
        facts.append((
            fd.get('subject', ''),
            fd.get('predicate', ''),
            fd.get('value', '')
        ))
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return ExtractionResult(
        facts=facts,
        time_ms=elapsed_ms,
        raw_output=facts_dicts
    )


def run_ab_test():
    """Run comprehensive A/B test"""
    
    # Define test cases with expected facts
    test_cases = [
        TestCase(
            text="My brother Tom lives in Portland and teaches at Reed College.",
            expected_facts=[
                ("user", "has", "brother"),
                ("brother", "lives in", "Portland"),
                ("brother", "teaches at", "Reed College"),
                ("brother", "name", "Tom")
            ],
            category="personal"
        ),
        TestCase(
            text="Apple was founded by Steve Jobs and Steve Wozniak in 1976.",
            expected_facts=[
                ("Steve Jobs", "founded", "Apple"),
                ("Steve Wozniak", "founded", "Apple"),
                ("Apple", "founded in", "1976")
            ],
            category="historical"
        ),
        TestCase(
            text="Marie Curie discovered radium and polonium with her husband Pierre.",
            expected_facts=[
                ("Marie Curie", "discovered", "radium"),
                ("Marie Curie", "discovered", "polonium"),
                ("Marie Curie", "husband", "Pierre"),
            ],
            category="scientific"
        ),
        TestCase(
            text="I work as a software engineer at Google.",
            expected_facts=[
                ("user", "job", "software engineer"),
                ("user", "works at", "Google")
            ],
            category="professional"
        ),
        TestCase(
            text="My dog's name is Luna and she is 3 years old.",
            expected_facts=[
                ("user", "has", "dog"),
                ("user", "dog name", "Luna"),
                ("dog", "age", "3")
            ],
            category="pet"
        ),
        TestCase(
            text="Barcelona is the capital of Catalonia.",
            expected_facts=[
                ("Barcelona", "capital of", "Catalonia")
            ],
            category="geographical"
        ),
        TestCase(
            text="I graduated from MIT with a degree in computer science.",
            expected_facts=[
                ("user", "graduated from", "MIT"),
                ("user", "degree", "computer science")
            ],
            category="education"
        ),
        TestCase(
            text="Elon Musk is the CEO of Tesla and SpaceX.",
            expected_facts=[
                ("Elon Musk", "CEO of", "Tesla"),
                ("Elon Musk", "CEO of", "SpaceX")
            ],
            category="business"
        ),
        TestCase(
            text="The Eiffel Tower was built in 1889 for the World's Fair.",
            expected_facts=[
                ("Eiffel Tower", "built in", "1889"),
                ("Eiffel Tower", "built for", "World's Fair")
            ],
            category="historical"
        ),
        TestCase(
            text="I have two cats named Whiskers and Shadow.",
            expected_facts=[
                ("user", "has", "cats"),
                ("user", "cat name", "Whiskers"),
                ("user", "cat name", "Shadow")
            ],
            category="pet"
        )
    ]
    
    print("=" * 80)
    print("A/B TEST: UD Patterns vs spaCy Extractor")
    print("=" * 80)
    print()
    
    # Aggregate results
    ud_scores = []
    spacy_scores = []
    ud_times = []
    spacy_times = []
    
    # Category breakdown
    category_scores = defaultdict(lambda: {'ud': [], 'spacy': []})
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case.text[:60]}...")
        print(f"   Category: {test_case.category}")
        print(f"   Expected facts: {len(test_case.expected_facts)}")
        
        # Method A: UD Patterns
        print("\n   Method A: UD Patterns")
        ud_result = extract_with_ud_patterns(test_case.text)
        ud_score = score_extraction(ud_result.facts, test_case.expected_facts)
        ud_scores.append(ud_score)
        ud_times.append(ud_result.time_ms)
        category_scores[test_case.category]['ud'].append(ud_score['f1'])
        
        print(f"   ⏱️  Time: {ud_result.time_ms:.1f}ms")
        print(f"   📊 Extracted: {len(ud_result.facts)} facts")
        for fact in ud_result.facts[:5]:  # Show first 5
            print(f"      • {fact}")
        print(f"   📈 F1 Score: {ud_score['f1']:.2%}")
        print(f"      Precision: {ud_score['precision']:.2%}, Recall: {ud_score['recall']:.2%}")
        
        # Method B: spaCy Extractor
        if SPACY_AVAILABLE:
            print("\n   Method B: spaCy Extractor")
            spacy_result = extract_with_spacy_extractor(test_case.text)
            spacy_score = score_extraction(spacy_result.facts, test_case.expected_facts)
            spacy_scores.append(spacy_score)
            spacy_times.append(spacy_result.time_ms)
            category_scores[test_case.category]['spacy'].append(spacy_score['f1'])
            
            print(f"   ⏱️  Time: {spacy_result.time_ms:.1f}ms")
            print(f"   📊 Extracted: {len(spacy_result.facts)} facts")
            for fact in spacy_result.facts[:5]:  # Show first 5
                print(f"      • {fact}")
            print(f"   📈 F1 Score: {spacy_score['f1']:.2%}")
            print(f"      Precision: {spacy_score['precision']:.2%}, Recall: {spacy_score['recall']:.2%}")
        
        # Compare
        if SPACY_AVAILABLE:
            winner = "UD" if ud_score['f1'] > spacy_score['f1'] else "spaCy"
            print(f"\n   🏆 Winner: {winner}")
    
    # Overall Summary
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    
    # Average metrics
    avg_ud_f1 = sum(s['f1'] for s in ud_scores) / len(ud_scores)
    avg_ud_precision = sum(s['precision'] for s in ud_scores) / len(ud_scores)
    avg_ud_recall = sum(s['recall'] for s in ud_scores) / len(ud_scores)
    avg_ud_time = sum(ud_times) / len(ud_times)
    
    print(f"\n📊 UD Patterns:")
    print(f"   Average F1 Score: {avg_ud_f1:.2%}")
    print(f"   Average Precision: {avg_ud_precision:.2%}")
    print(f"   Average Recall: {avg_ud_recall:.2%}")
    print(f"   Average Time: {avg_ud_time:.1f}ms")
    
    if SPACY_AVAILABLE and spacy_scores:
        avg_spacy_f1 = sum(s['f1'] for s in spacy_scores) / len(spacy_scores)
        avg_spacy_precision = sum(s['precision'] for s in spacy_scores) / len(spacy_scores)
        avg_spacy_recall = sum(s['recall'] for s in spacy_scores) / len(spacy_scores)
        avg_spacy_time = sum(spacy_times) / len(spacy_times)
        
        print(f"\n📊 spaCy Extractor:")
        print(f"   Average F1 Score: {avg_spacy_f1:.2%}")
        print(f"   Average Precision: {avg_spacy_precision:.2%}")
        print(f"   Average Recall: {avg_spacy_recall:.2%}")
        print(f"   Average Time: {avg_spacy_time:.1f}ms")
        
        # Category breakdown
        print("\n📈 Performance by Category:")
        for category, scores in category_scores.items():
            if scores['ud'] and scores['spacy']:
                ud_cat_avg = sum(scores['ud']) / len(scores['ud'])
                spacy_cat_avg = sum(scores['spacy']) / len(scores['spacy'])
                winner = "UD" if ud_cat_avg > spacy_cat_avg else "spaCy"
                print(f"   {category:12s}: UD={ud_cat_avg:.2%}, spaCy={spacy_cat_avg:.2%} → {winner}")
        
        # Overall winner
        print("\n" + "=" * 80)
        if avg_ud_f1 > avg_spacy_f1:
            print(f"🏆 WINNER: UD Patterns (F1: {avg_ud_f1:.2%} vs {avg_spacy_f1:.2%})")
            print(f"   UD patterns are {((avg_ud_f1 / avg_spacy_f1) - 1) * 100:.1f}% better")
        else:
            print(f"🏆 WINNER: spaCy Extractor (F1: {avg_spacy_f1:.2%} vs {avg_ud_f1:.2%})")
            print(f"   spaCy is {((avg_spacy_f1 / avg_ud_f1) - 1) * 100:.1f}% better")
        
        # Speed comparison
        if avg_ud_time < avg_spacy_time:
            print(f"⚡ UD patterns are {avg_spacy_time / avg_ud_time:.1f}x faster")
        else:
            print(f"⚡ spaCy is {avg_ud_time / avg_spacy_time:.1f}x faster")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_ab_test()