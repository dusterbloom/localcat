#!/usr/bin/env python3
"""
HotMem model comparison test with proper timing.

Tests both relation-extractor-v2-mlx and hotmem-relation-classifier-mlx models
with accurate inference timing (excluding loading time).
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


def get_test_sentences() -> list:
    """Return larger test set for better comparison."""
    return [
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
        },
        {
            "text": "My brother Tom lives in Portland and teaches at Reed College.",
            "entities": ["tom", "portland", "reed college"],
            "expected": [("tom", "lives_in", "portland"), ("tom", "teaches_at", "reed college")]
        },
        {
            "text": "Jennifer is married to Michael and they have two children.",
            "entities": ["jennifer", "michael"],
            "expected": [("jennifer", "married_to", "michael")]
        },
        {
            "text": "Amazon acquired Whole Foods in 2017 for $13.7 billion.",
            "entities": ["amazon", "whole foods"],
            "expected": [("amazon", "acquired", "whole foods")]
        },
        {
            "text": "The company is headquartered in New York and competes with Google.",
            "entities": ["new york", "google"],
            "expected": [("company", "headquartered_in", "new york"), ("company", "competes_with", "google")]
        },
        {
            "text": "Dr. Smith retired from Harvard University in 2020 after 30 years.",
            "entities": ["dr. smith", "harvard university"],
            "expected": [("dr. smith", "retired_from", "harvard university")]
        },
        {
            "text": "Microsoft develops Windows and Office, while Apple creates macOS and iOS.",
            "entities": ["microsoft", "windows", "office", "apple", "macos", "ios"],
            "expected": [("microsoft", "develops", "windows"), ("microsoft", "develops", "office"), ("apple", "creates", "macos"), ("apple", "creates", "ios")]
        },
        {
            "text": "The startup moved from San Francisco to Austin last year.",
            "entities": ["san francisco", "austin"],
            "expected": [("startup", "moved_from", "san francisco"), ("startup", "moved_to", "austin")]
        },
        {
            "text": "John works at Google as a software engineer and manages the Search team.",
            "entities": ["john", "google", "search team"],
            "expected": [("john", "works_for", "google"), ("john", "manages", "search team")]
        },
        {
            "text": "Toyota produces cars and is based in Japan.",
            "entities": ["toyota", "japan"],
            "expected": [("toyota", "produces", "cars"), ("toyota", "based_in", "japan")]
        },
        {
            "text": "Emma graduated from Stanford University and joined Facebook.",
            "entities": ["emma", "stanford university", "facebook"],
            "expected": [("emma", "graduated_from", "stanford university"), ("emma", "joined", "facebook")]
        },
        {
            "text": "The partnership between Apple and Qualcomm ended in 2019.",
            "entities": ["apple", "qualcomm"],
            "expected": [("apple", "partnered_with", "qualcomm")]
        },
        {
            "text": "Mark Zuckerberg is the founder of Facebook and owns Instagram.",
            "entities": ["mark zuckerberg", "facebook", "instagram"],
            "expected": [("mark zuckerberg", "founder_of", "facebook"), ("mark zuckerberg", "owns", "instagram")]
        }
    ]


def calculate_metrics(expected: list, actual: list) -> dict:
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


def test_model(model_name: str, test_cases: list) -> dict:
    """Test a specific model with proper timing."""
    
    print(f"\n🧪 Testing model: {model_name}")
    print("=" * 70)
    
    # Load .env
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    
    # Set model
    os.environ['HOTMEM_LLM_ASSISTED_MODEL'] = model_name
    os.environ['HOTMEM_LLM_ASSISTED_TIMEOUT_MS'] = '3000'  # 3 seconds
    
    # Initialize HotMem (this includes loading time)
    paths = Paths()
    store = MemoryStore(paths=paths)
    hm = HotMemory(store)
    hm.prewarm('en')
    
    print(f"Model: {os.getenv('HOTMEM_LLM_ASSISTED_MODEL')}")
    print(f"Base URL: {os.getenv('HOTMEM_LLM_ASSISTED_BASE_URL')}")
    print()
    
    results = []
    inference_times = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"{i:2d}. '{case['text']}'")
        
        # Time only the inference (excluding loading)
        t0 = time.perf_counter()
        try:
            result = hm._assist_extract(case['text'], case['entities'], [])
            inference_time = (time.perf_counter() - t0) * 1000
            inference_times.append(inference_time)
            
            if result and 'triples' in result:
                extracted_triples = [(t['s'], t['r'], t['d']) for t in result['triples']]
                print(f"    ✅ {len(extracted_triples)} triples ({inference_time:.0f}ms)")
                print(f"       {extracted_triples}")
                
                metrics = calculate_metrics(case['expected'], extracted_triples)
                print(f"       P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")
                
                results.append({
                    'text': case['text'],
                    'expected': case['expected'],
                    'actual': extracted_triples,
                    'metrics': metrics,
                    'inference_time_ms': inference_time
                })
            else:
                print(f"    ❌ No triples extracted ({inference_time:.0f}ms)")
                results.append({
                    'text': case['text'],
                    'expected': case['expected'],
                    'actual': [],
                    'metrics': {'precision': 0, 'recall': 0, 'f1': 0},
                    'inference_time_ms': inference_time
                })
                
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            results.append({
                'text': case['text'],
                'expected': case['expected'],
                'actual': [],
                'metrics': {'precision': 0, 'recall': 0, 'f1': 0},
                'inference_time_ms': 0,
                'error': str(e)
            })
    
    # Summary statistics
    print("\n" + "=" * 70)
    print(f"SUMMARY for {model_name}")
    print("=" * 70)
    
    if results:
        avg_precision = sum(r['metrics']['precision'] for r in results) / len(results)
        avg_recall = sum(r['metrics']['recall'] for r in results) / len(results)
        avg_f1 = sum(r['metrics']['f1'] for r in results) / len(results)
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        successful_extractions = sum(1 for r in results if r['metrics']['f1'] > 0)
        success_rate = successful_extractions / len(results)
        
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Average F1: {avg_f1:.3f}")
        print(f"Average Inference Time: {avg_inference_time:.0f}ms")
        print(f"Success Rate: {success_rate:.1%} ({successful_extractions}/{len(results)})")
        print(f"Min/Max Inference Time: {min(inference_times):.0f}ms / {max(inference_times):.0f}ms")
        
        # Count extraction methods
        total_expected_triples = sum(len(r['expected']) for r in results)
        total_actual_triples = sum(len(r['actual']) for r in results)
        print(f"Total Triples: {total_actual_triples} extracted vs {total_expected_triples} expected")
    
    return {
        'model': model_name,
        'results': results,
        'avg_precision': avg_precision if results else 0,
        'avg_recall': avg_recall if results else 0,
        'avg_f1': avg_f1 if results else 0,
        'avg_inference_time_ms': avg_inference_time if results else 0,
        'success_rate': success_rate if results else 0,
        'inference_times': inference_times
    }


def main():
    """Compare both models."""
    test_cases = get_test_sentences()
    
    models_to_test = [
        "relation-extractor-v2-mlx",
        "hotmem-relation-classifier-mlx"
    ]
    
    all_results = []
    
    for model in models_to_test:
        result = test_model(model, test_cases)
        all_results.append(result)
    
    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    
    print(f"{'Model':<35} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time(ms)':<10} {'Success':<10}")
    print("-" * 85)
    
    for result in all_results:
        print(f"{result['model']:<35} {result['avg_precision']:<10.3f} {result['avg_recall']:<10.3f} "
              f"{result['avg_f1']:<10.3f} {result['avg_inference_time_ms']:<10.0f} {result['success_rate']:<10.1%}")
    
    # Determine winner
    if len(all_results) == 2:
        model1, model2 = all_results
        if model1['avg_f1'] > model2['avg_f1']:
            winner = model1['model']
            print(f"\n🏆 Winner: {winner} (higher F1 score)")
        elif model2['avg_f1'] > model1['avg_f1']:
            winner = model2['model']
            print(f"\n🏆 Winner: {winner} (higher F1 score)")
        else:
            if model1['avg_inference_time_ms'] < model2['avg_inference_time_ms']:
                winner = model1['model']
                print(f"\n🏆 Winner: {winner} (faster inference)")
            else:
                winner = model2['model']
                print(f"\n🏆 Winner: {winner} (faster inference)")


if __name__ == '__main__':
    main()