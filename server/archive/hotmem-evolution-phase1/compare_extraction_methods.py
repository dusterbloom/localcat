#!/usr/bin/env python3
"""
Compare extraction methods: Current UD-based vs Stanford OpenIE
"""
import sys
import os
import time

# Set Java path for Stanford OpenIE
os.environ['PATH'] = "/opt/homebrew/opt/openjdk@11/bin:" + os.environ.get('PATH', '')
sys.path.insert(0, os.path.dirname(__file__))

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory

def test_current_ud_extraction():
    """Test our current UD-based extraction"""
    print("=== Testing Current UD-based Extraction ===")
    
    # Initialize memory system
    paths = Paths(sqlite_path="test_compare.db", lmdb_dir="test_compare.lmdb")
    store = MemoryStore(paths)
    hot = HotMemory(store)
    
    # Test sentences
    test_sentences = [
        "My favorite number is 77.",
        "My dog's name is Potola.",
        "I have a favorite color.",
        "So my favorite number is 77. Favorite food is pizza.",
        "My name is Alex.",
        "I live in San Francisco.",
        "Caroline is a developer.",
        "The book is on the table."
    ]
    
    results = []
    total_time = 0
    
    for sentence in test_sentences:
        start_time = time.perf_counter()
        try:
            entities, triples, neg_count, doc = hot._extract(sentence, "en")
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            total_time += elapsed_ms
            
            results.append({
                'sentence': sentence,
                'triples': triples,
                'entities': entities,
                'time_ms': elapsed_ms,
                'success': True
            })
            
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            total_time += elapsed_ms
            results.append({
                'sentence': sentence,
                'triples': [],
                'entities': [],
                'time_ms': elapsed_ms,
                'success': False,
                'error': str(e)
            })
    
    avg_time = total_time / len(test_sentences)
    print(f"📊 Current UD Extraction Results:")
    print(f"   Average time: {avg_time:.1f}ms")
    print(f"   Total time: {total_time:.1f}ms")
    
    return results


def test_stanford_openie_extraction():
    """Test Stanford OpenIE extraction"""
    print("\n=== Testing Stanford OpenIE Extraction ===")
    
    try:
        from openie import StanfordOpenIE
    except ImportError as e:
        print(f"❌ Stanford OpenIE not available: {e}")
        return []
    
    # Test sentences (same as above)
    test_sentences = [
        "My favorite number is 77.",
        "My dog's name is Potola.", 
        "I have a favorite color.",
        "So my favorite number is 77. Favorite food is pizza.",
        "My name is Alex.",
        "I live in San Francisco.",
        "Caroline is a developer.",
        "The book is on the table."
    ]
    
    results = []
    total_time = 0
    
    try:
        with StanfordOpenIE() as client:
            for sentence in test_sentences:
                start_time = time.perf_counter()
                try:
                    triples = client.annotate(sentence)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    total_time += elapsed_ms
                    
                    # Convert Stanford OpenIE format to our format
                    converted_triples = []
                    for triple in triples:
                        if isinstance(triple, dict):
                            subj = triple.get('subject', '').strip()
                            rel = triple.get('relation', '').strip()
                            obj = triple.get('object', '').strip()
                            if subj and rel and obj:
                                converted_triples.append((subj, rel, obj))
                    
                    results.append({
                        'sentence': sentence,
                        'triples': converted_triples,
                        'raw_triples': triples,
                        'time_ms': elapsed_ms,
                        'success': True
                    })
                    
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    total_time += elapsed_ms
                    results.append({
                        'sentence': sentence,
                        'triples': [],
                        'time_ms': elapsed_ms,
                        'success': False,
                        'error': str(e)
                    })
        
        avg_time = total_time / len(test_sentences) if test_sentences else 0
        print(f"📊 Stanford OpenIE Results:")
        print(f"   Average time: {avg_time:.1f}ms")
        print(f"   Total time: {total_time:.1f}ms")
        
    except Exception as e:
        print(f"❌ Stanford OpenIE client error: {e}")
        return []
    
    return results


def compare_results(ud_results, openie_results):
    """Compare the two extraction methods"""
    print("\n=== Comparison Results ===")
    
    if not openie_results:
        print("❌ Cannot compare - Stanford OpenIE results unavailable")
        return
    
    print(f"{'Sentence':<50} {'UD Triples':<15} {'OpenIE Triples':<15} {'UD Time':<12} {'OpenIE Time':<12}")
    print("-" * 110)
    
    for ud, openie in zip(ud_results, openie_results):
        sentence = ud['sentence'][:47] + "..." if len(ud['sentence']) > 50 else ud['sentence']
        ud_count = len(ud['triples']) if ud['success'] else 0
        openie_count = len(openie['triples']) if openie['success'] else 0
        ud_time = f"{ud['time_ms']:.1f}ms" if ud['success'] else "ERROR"
        openie_time = f"{openie['time_ms']:.1f}ms" if openie['success'] else "ERROR"
        
        print(f"{sentence:<50} {ud_count:<15} {openie_count:<15} {ud_time:<12} {openie_time:<12}")
    
    # Quality analysis for specific problematic cases
    print(f"\n🔍 Detailed Analysis:")
    
    for i, (ud, openie) in enumerate(zip(ud_results, openie_results)):
        sentence = ud['sentence']
        print(f"\n{i+1}. \"{sentence}\"")
        
        if ud['success']:
            print(f"   UD: {ud['triples']}")
        else:
            print(f"   UD: ERROR - {ud.get('error', 'Unknown error')}")
        
        if openie['success']:
            print(f"   OpenIE: {openie['triples']}")
        else:
            print(f"   OpenIE: ERROR - {openie.get('error', 'Unknown error')}")
        
        # Analysis for specific cases we know are problematic
        if "favorite number is 77" in sentence.lower():
            print("   🎯 Analysis: This should extract ('you'/'I', 'favorite_number', '77')")
            ud_good = any('77' in str(t) and ('you' in str(t) or 'favorite' in str(t)) for t in ud.get('triples', []))
            openie_good = any('77' in str(t) and ('I' in str(t) or 'favorite' in str(t)) for t in openie.get('triples', []))
            print(f"   UD captures intent: {'✅' if ud_good else '❌'}")
            print(f"   OpenIE captures intent: {'✅' if openie_good else '❌'}")


def main():
    print("🚀 Starting Extraction Method Comparison")
    
    # Test current UD-based method
    ud_results = test_current_ud_extraction()
    
    # Test Stanford OpenIE method  
    openie_results = test_stanford_openie_extraction()
    
    # Compare results
    compare_results(ud_results, openie_results)
    
    print(f"\n✅ Comparison Complete!")


if __name__ == "__main__":
    main()