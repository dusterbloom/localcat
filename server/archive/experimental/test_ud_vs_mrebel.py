#!/usr/bin/env python3
"""
Head-to-head comparison: UD-based extraction vs mREBEL
Test on the 27 pattern examples to see which performs better
"""

import os
import time
import statistics
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from loguru import logger

# Set up test environment
os.environ["HOTMEM_SQLITE"] = "test_comparison.db" 
os.environ["HOTMEM_LMDB_DIR"] = "test_comparison.lmdb"

from memory_hotpath import HotMemory
from memory_store import MemoryStore
from mrebel_extractor import get_mrebel_extractor

@dataclass
class ComparisonResult:
    text: str
    ud_triples: List[Tuple[str, str, str]]
    mrebel_triples: List[Tuple[str, str, str]]
    ud_time_ms: float
    mrebel_time_ms: float
    expected_relations: List[str]  # What we expect to extract

# Test cases from the 27 pattern examples
TEST_CASES = [
    ComparisonResult(
        text="My name is Alex Thompson",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["name"]
    ),
    ComparisonResult(
        text="I live in Seattle", 
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["lives_in", "located_in", "residence"]
    ),
    ComparisonResult(
        text="I work at Microsoft",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["works_at", "employed_by", "occupation"]
    ),
    ComparisonResult(
        text="My dog's name is Potola",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["name", "has"]
    ),
    ComparisonResult(
        text="Caroline went to the LGBTQ support group",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["went_to", "attended", "participated"]
    ),
    ComparisonResult(
        text="Melanie painted a sunrise in 2022",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["painted", "created", "time"]
    ),
    ComparisonResult(
        text="Caroline moved from Sweden 4 years ago",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["moved_from", "origin", "time"]
    ),
    ComparisonResult(
        text="Caroline has had her current group of friends for 4 years",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["has", "duration"]
    ),
    ComparisonResult(
        text="Melanie has read Nothing is Impossible and Charlotte's Web",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["read", "has_read"]
    ),
    ComparisonResult(
        text="Caroline participated in a pride parade",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["participated_in", "attended"]
    ),
    ComparisonResult(
        text="The old red car belongs to me",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["belongs_to", "owns", "has"]
    ),
    ComparisonResult(
        text="Sarah and John are friends",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["friend_of", "friends", "relationship"]
    ),
    ComparisonResult(
        text="I have three pets",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["has", "quantity"]
    ),
    ComparisonResult(
        text="My favorite color is blue",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["favorite_color", "preference", "is"]
    ),
    ComparisonResult(
        text="I was born in 1995",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["born_in", "birth_year"]
    ),
    ComparisonResult(
        text="My son is named Jake",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["name", "has"]
    ),
    # Additional challenging cases
    ComparisonResult(
        text="Caroline likes reading",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["likes", "enjoys"]
    ),
    ComparisonResult(
        text="Caroline is single",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["is", "status"]
    ),
    ComparisonResult(
        text="Caroline researched adoption agencies",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["researched", "investigated"]
    ),
    ComparisonResult(
        text="My cat is called Whiskers",
        ud_triples=[], mrebel_triples=[], ud_time_ms=0, mrebel_time_ms=0,
        expected_relations=["name", "called", "has"]
    ),
]

def run_comparison():
    """Run the head-to-head comparison"""
    print("=" * 80)
    print("UD vs mREBEL EXTRACTION COMPARISON")
    print("=" * 80)
    print()
    
    # Initialize extractors
    print("🔧 Initializing extractors...")
    
    # UD extractor (HotMemory)
    store = MemoryStore()
    hot_memory = HotMemory(store)
    
    # mREBEL extractor  
    mrebel = get_mrebel_extractor()
    
    print("✅ Extractors ready")
    print()
    
    # Run comparisons
    ud_times = []
    mrebel_times = []
    ud_success_count = 0
    mrebel_success_count = 0
    
    for i, case in enumerate(TEST_CASES, 1):
        print(f"{i:2d}. Testing: '{case.text}'")
        
        # Test UD extraction
        start = time.perf_counter()
        try:
            entities, triples, neg_count, doc = hot_memory._extract(case.text, "en")
            case.ud_triples = triples
            case.ud_time_ms = (time.perf_counter() - start) * 1000
            ud_times.append(case.ud_time_ms)
            
            # Check if any expected relations found
            extracted_rels = [r for _, r, _ in triples]
            if any(expected in str(extracted_rels) for expected in case.expected_relations):
                ud_success_count += 1
                ud_status = "✅"
            else:
                ud_status = "❌"
                
        except Exception as e:
            case.ud_triples = []
            case.ud_time_ms = 0
            ud_status = f"💥 {e}"
        
        # Test mREBEL extraction
        try:
            case.mrebel_triples, case.mrebel_time_ms = mrebel.extract_with_timing(case.text)
            mrebel_times.append(case.mrebel_time_ms)
            
            # Check if any expected relations found
            extracted_rels = [r for _, r, _ in case.mrebel_triples]
            if any(expected in str(extracted_rels) for expected in case.expected_relations):
                mrebel_success_count += 1
                mrebel_status = "✅"
            else:
                mrebel_status = "❌"
                
        except Exception as e:
            case.mrebel_triples = []
            case.mrebel_time_ms = 0
            mrebel_status = f"💥 {e}"
        
        # Show results
        print(f"    UD      ({case.ud_time_ms:6.1f}ms): {case.ud_triples} {ud_status}")
        print(f"    mREBEL  ({case.mrebel_time_ms:6.1f}ms): {case.mrebel_triples} {mrebel_status}")
        print()
    
    # Summary statistics
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"Test Cases: {len(TEST_CASES)}")
    print()
    
    print("SUCCESS RATE:")
    print(f"  UD-based:  {ud_success_count:2d}/{len(TEST_CASES)} ({ud_success_count/len(TEST_CASES)*100:.1f}%)")
    print(f"  mREBEL:    {mrebel_success_count:2d}/{len(TEST_CASES)} ({mrebel_success_count/len(TEST_CASES)*100:.1f}%)")
    print()
    
    if ud_times:
        print("UD TIMING:")
        print(f"  Mean:  {statistics.mean(ud_times):7.1f}ms")
        print(f"  P95:   {statistics.quantiles(ud_times, n=20)[18] if len(ud_times) > 10 else max(ud_times):7.1f}ms")
        print(f"  Range: {min(ud_times):7.1f}ms - {max(ud_times):7.1f}ms")
        print()
    
    if mrebel_times:
        print("mREBEL TIMING:")
        print(f"  Mean:  {statistics.mean(mrebel_times):7.1f}ms")
        print(f"  P95:   {statistics.quantiles(mrebel_times, n=20)[18] if len(mrebel_times) > 10 else max(mrebel_times):7.1f}ms")
        print(f"  Range: {min(mrebel_times):7.1f}ms - {max(mrebel_times):7.1f}ms")
        print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("===============")
    
    if ud_success_count > mrebel_success_count:
        print("🏆 UD-based extraction has better accuracy")
        if ud_times and mrebel_times and statistics.mean(ud_times) < statistics.mean(mrebel_times):
            print("🚀 UD is also faster → Use UD as primary")
        else:
            print("🐌 But mREBEL might be faster → Consider hybrid approach")
    elif mrebel_success_count > ud_success_count:
        print("🏆 mREBEL has better accuracy")
        if mrebel_times and ud_times and statistics.mean(mrebel_times) < statistics.mean(ud_times):
            print("🚀 mREBEL is also faster → Consider switching to mREBEL")
        else:
            print("🐌 But UD is faster → Use hybrid: UD first, mREBEL fallback")
    else:
        print("🤝 Both have similar accuracy → Choose based on speed and complexity")
    
    return TEST_CASES

if __name__ == "__main__":
    # Suppress some logging for cleaner output
    logger.remove()
    logger.add(lambda msg: None if "pipecat" in msg or "memory_hotpath" in msg else print(msg, end=""))
    
    try:
        results = run_comparison()
    except KeyboardInterrupt:
        print("\n⏹️  Comparison interrupted by user")
    except Exception as e:
        print(f"\n💥 Comparison failed: {e}")
        import traceback
        traceback.print_exc()