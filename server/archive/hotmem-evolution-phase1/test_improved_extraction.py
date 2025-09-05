#!/usr/bin/env python3
"""
Test improved UD extraction that mimics Stanford OpenIE quality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory
import time

def test_extraction_comparison():
    """Compare current vs improved extraction"""
    
    print("=== Testing Current vs Target Extraction ===")
    
    # Initialize memory system
    paths = Paths(sqlite_path="test_improved.db", lmdb_dir="test_improved.lmdb")
    store = MemoryStore(paths)
    hot = HotMemory(store)
    
    # Test cases based on Stanford OpenIE results
    test_cases = [
        {
            'sentence': 'My favorite number is 77.',
            'current_expected': [('favorite number', 'is', '77'), ('you', 'favorite_number', '77')],
            'target': [('you', 'favorite_number', '77')],  # Single, clean triple
            'stanford_result': [('My number', 'is', '77'), ('My favorite number', 'is', '77')]
        },
        {
            'sentence': 'My name is Alex.',
            'current_expected': [('you', 'name', 'alex')],
            'target': [('you', 'name', 'Alex')],
            'stanford_result': [('My name', 'is', 'Alex')]
        },
        {
            'sentence': 'I live in San Francisco.',
            'current_expected': [('you', 'lives_in', 'san francisco')],
            'target': [('you', 'lives_in', 'San Francisco')],
            'stanford_result': [('I', 'live in', 'San Francisco')]
        },
        {
            'sentence': 'Caroline is a developer.',
            'current_expected': [('caroline', 'is', 'developer')],
            'target': [('caroline', 'is', 'developer')],
            'stanford_result': [('Caroline', 'is', 'developer')]
        }
    ]
    
    print(f"\n{'Sentence':<35} {'Current Count':<15} {'Target Count':<15} {'Time (ms)':<12}")
    print("-" * 80)
    
    for case in test_cases:
        sentence = case['sentence']
        
        # Test current extraction
        start_time = time.perf_counter()
        try:
            entities, triples, neg_count, doc = hot._extract(sentence, "en")
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Apply quality filtering
            from memory_intent import get_quality_filter, get_intent_classifier, IntentAnalysis, IntentType
            quality_filter = get_quality_filter()
            
            # Create a dummy intent for filtering
            dummy_intent = IntentAnalysis(
                IntentType.FACT_STATEMENT, 0.8, True, False, [], []
            )
            
            filtered_triples = []
            for s, r, d in triples:
                should_store, confidence = quality_filter.should_store_fact(s, r, d, dummy_intent)
                if should_store:
                    filtered_triples.append((s, r, d))
            
            sentence_short = sentence[:32] + "..." if len(sentence) > 35 else sentence
            current_count = len(filtered_triples)
            target_count = len(case['target'])
            
            print(f"{sentence_short:<35} {current_count:<15} {target_count:<15} {elapsed_ms:.1f}ms")
            
            # Detailed analysis
            print(f"   Current: {filtered_triples[:3]}")  # Show up to 3
            print(f"   Target:  {case['target']}")
            print(f"   Stanford: {case['stanford_result']}")
            
            # Quality assessment
            quality_score = assess_extraction_quality(filtered_triples, case['target'], case['stanford_result'])
            print(f"   Quality: {quality_score}/10")
            print()
            
        except Exception as e:
            print(f"{sentence[:35]:<35} ERROR: {e}")
    
    print("=== Analysis Complete ===")


def assess_extraction_quality(current_triples, target_triples, stanford_triples):
    """Assess quality of extraction compared to targets"""
    score = 0
    
    # Check if we got the right number of triples (avoid over-extraction)
    if len(current_triples) == len(target_triples):
        score += 2
    elif len(current_triples) <= len(target_triples) + 1:
        score += 1
    
    # Check if key entities are captured
    for target_s, target_r, target_d in target_triples:
        found_match = False
        for curr_s, curr_r, curr_d in current_triples:
            # Flexible matching - check if essence is captured
            if (target_s.lower() in curr_s.lower() or curr_s.lower() in target_s.lower()) and \
               (target_d.lower() in curr_d.lower() or curr_d.lower() in target_d.lower()):
                found_match = True
                break
        if found_match:
            score += 3
    
    # Check for unwanted noise (bad triples)
    noise_patterns = ['quality', 'favorite', 'is favorite']
    for curr_s, curr_r, curr_d in current_triples:
        if any(pattern in curr_r.lower() for pattern in noise_patterns):
            score = max(0, score - 1)
    
    return min(10, score)


if __name__ == "__main__":
    test_extraction_comparison()