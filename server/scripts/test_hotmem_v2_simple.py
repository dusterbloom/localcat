#!/usr/bin/env python3
"""
Simplified HotMem v2 integration test with longer timeout.

Tests LLM-assisted relation extraction with various sentence patterns.
"""
import argparse
import json
import os
import time
from dotenv import load_dotenv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.memory.memory_store import MemoryStore, Paths
from components.memory.memory_hotpath import HotMemory


def test_llm_integration():
    """Test LLM-assisted extraction with longer timeout."""
    
    # Load .env
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    
    # Temporarily increase timeout
    original_timeout = os.getenv('HOTMEM_LLM_ASSISTED_TIMEOUT_MS', '1200')
    os.environ['HOTMEM_LLM_ASSISTED_TIMEOUT_MS'] = '5000'  # 5 seconds
    
    # Initialize HotMem
    paths = Paths()
    store = MemoryStore(paths=paths)
    hm = HotMemory(store)
    hm.prewarm('en')
    
    print("🧪 HotMem v2 Integration Test")
    print("=" * 70)
    print(f"LLM-assisted extraction enabled: {getattr(hm, 'assisted_enabled', False)}")
    print(f"LLM-assisted model: {os.getenv('HOTMEM_LLM_ASSISTED_MODEL', 'None')}")
    print(f"LLM-assisted base URL: {os.getenv('HOTMEM_LLM_ASSISTED_BASE_URL', 'None')}")
    print(f"Timeout: 5000ms (increased from {original_timeout}ms)")
    print()
    
    # Test cases
    test_cases = [
        {
            "text": "Tim Cook is the CEO of Apple and lives in California.",
            "entities": ["tim cook", "apple", "california"],
            "expected": [("tim cook", "ceo_of", "apple"), ("tim cook", "lives_in", "california")]
        },
        {
            "text": "Sarah works at Microsoft as a senior engineer and develops Windows.",
            "entities": ["sarah", "microsoft", "windows"],
            "expected": [("sarah", "works_for", "microsoft"), ("sarah", "develops", "windows")]
        },
        {
            "text": "Elon Musk founded Tesla and SpaceX in the early 2000s.",
            "entities": ["elon musk", "tesla", "spacex"],
            "expected": [("elon musk", "founder_of", "tesla"), ("elon musk", "founder_of", "spacex")]
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. '{case['text']}'")
        print(f"   Entities: {case['entities']}")
        print(f"   Expected: {case['expected']}")
        
        # Test LLM-assisted extraction
        t0 = time.perf_counter()
        try:
            result = hm._assist_extract(case['text'], case['entities'], [])
            elapsed = (time.perf_counter() - t0) * 1000
            
            if result and 'triples' in result:
                extracted_triples = [(t['s'], t['r'], t['d']) for t in result['triples']]
                print(f"   ✅ LLM Result: {extracted_triples} ({elapsed:.0f}ms)")
                
                # Calculate metrics
                expected_set = set(case['expected'])
                actual_set = set(extracted_triples)
                true_positives = len(expected_set.intersection(actual_set))
                false_positives = len(actual_set - expected_set)
                false_negatives = len(expected_set - actual_set)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"   📊 Metrics: P={precision:.2f} R={recall:.2f} F1={f1:.2f}")
                
                results.append({
                    'text': case['text'],
                    'expected': case['expected'],
                    'actual': extracted_triples,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'time_ms': elapsed
                })
            else:
                print(f"   ❌ No triples extracted ({elapsed:.0f}ms)")
                results.append({
                    'text': case['text'],
                    'expected': case['expected'],
                    'actual': [],
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'time_ms': elapsed
                })
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'text': case['text'],
                'expected': case['expected'],
                'actual': [],
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'time_ms': 0,
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results:
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
        avg_time = sum(r['time_ms'] for r in results if 'time_ms' in r) / len(results)
        
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Average F1: {avg_f1:.3f}")
        print(f"Average Time: {avg_time:.0f}ms")
        
        success_rate = sum(1 for r in results if r['f1'] > 0) / len(results)
        print(f"Success Rate: {success_rate:.1%}")
    
    # Restore original timeout
    if original_timeout:
        os.environ['HOTMEM_LLM_ASSISTED_TIMEOUT_MS'] = original_timeout
    
    return results


if __name__ == '__main__':
    test_llm_integration()