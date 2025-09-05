#!/usr/bin/env python3
"""
Comprehensive HotMem test suite with diverse conversation patterns
Tests for edge cases, multilingual support, and complex relationships
"""

import json
import time
import os
import sys
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from loguru import logger

# Set up test environment
os.environ["HOTMEM_SQLITE"] = "test_comprehensive.db"
os.environ["HOTMEM_LMDB_DIR"] = "test_comprehensive.lmdb"

from hotpath_processor import HotPathMemoryProcessor


@dataclass
class TestCase:
    """A single test case for memory extraction and retrieval"""
    name: str
    input_turns: List[str]
    query_turn: str
    expected_facts: List[Tuple[str, str, str]]  # Expected triples
    expected_recall: List[str]  # Keywords that should appear in retrieval
    category: str = "general"


# Comprehensive test cases covering various patterns
TEST_CASES = [
    # === Basic Personal Information ===
    TestCase(
        name="Personal details",
        input_turns=[
            "My name is Alex Thompson",
            "I'm 28 years old", 
            "I live in Seattle",
            "I work at Microsoft as a software engineer"
        ],
        query_turn="Tell me what you know about me",
        expected_facts=[
            ("you", "name", "alex thompson"),
            ("you", "lives_in", "seattle"),
            ("you", "works_at", "microsoft")
        ],
        expected_recall=["alex thompson", "seattle", "microsoft"],
        category="personal"
    ),
    
    # === Family & Relationships ===
    TestCase(
        name="Family relationships",
        input_turns=[
            "My wife's name is Sarah",
            "We have two kids",
            "My son is named Jake and my daughter is Emma",
            "Sarah works as a doctor at the hospital"
        ],
        query_turn="What do you remember about my family?",
        expected_facts=[
            ("wife", "name", "sarah"),
            ("son", "name", "jake"),
            ("daughter", "name", "emma")
        ],
        expected_recall=["sarah", "jake", "emma"],
        category="family"
    ),
    
    # === Pets (Extended Potola test) ===
    TestCase(
        name="Multiple pets",
        input_turns=[
            "I have three pets",
            "My dog's name is Potola",
            "My cat is called Whiskers", 
            "My parrot's name is Polly",
            "Potola is a golden retriever",
            "Whiskers is 5 years old"
        ],
        query_turn="Tell me about my pets",
        expected_facts=[
            ("you", "has", "dog"),
            ("dog", "name", "potola"),
            ("cat", "name", "whiskers"),
            ("parrot", "name", "polly")
        ],
        expected_recall=["potola", "whiskers", "polly"],
        category="pets"
    ),
    
    # === Preferences & Interests ===
    TestCase(
        name="Preferences",
        input_turns=[
            "I love Italian food",
            "My favorite color is blue",
            "I enjoy hiking on weekends",
            "I don't like spicy food",
            "My favorite movie is Inception"
        ],
        query_turn="What are my preferences?",
        expected_facts=[
            ("you", "likes", "italian food"),
            ("you", "likes", "hiking"),
            ("favorite color", "attr_of", "blue"),
            ("favorite movie", "attr_of", "inception")
        ],
        expected_recall=["italian", "blue", "hiking", "inception"],
        category="preferences"
    ),
    
    # === Complex Possessive Chains ===
    TestCase(
        name="Complex possessives",
        input_turns=[
            "My brother's wife's name is Jennifer",
            "My company's CEO is Tim Cook",
            "My car's color is red",
            "My doctor's office is on Main Street"
        ],
        query_turn="What do you know about things related to me?",
        expected_facts=[
            ("brother", "has", "wife"),
            ("wife", "name", "jennifer"),
            ("car", "attr_of", "red")
        ],
        expected_recall=["jennifer", "red"],
        category="complex"
    ),
    
    # === Temporal Information ===
    TestCase(
        name="Temporal facts",
        input_turns=[
            "I was born in 1995",
            "I graduated from MIT in 2017",
            "I started my current job last year",
            "My birthday is March 15th",
            "I've been married for 5 years"
        ],
        query_turn="When did important events happen?",
        expected_facts=[
            ("you", "born", "1995"),
            ("you", "graduated", "mit"),
            ("birthday", "attr_of", "march 15th")
        ],
        expected_recall=["1995", "mit", "march"],
        category="temporal"
    ),
    
    # === Updates & Corrections ===
    TestCase(
        name="Fact updates",
        input_turns=[
            "My dog's name is Max",
            "Actually, I meant my dog's name is Rex",
            "I live in Boston",
            "Wait, I moved to New York last month"
        ],
        query_turn="Where do I live and what's my dog's name?",
        expected_facts=[
            ("dog", "name", "rex"),  # Should update to Rex
            ("you", "lives_in", "new york")  # Should update to New York
        ],
        expected_recall=["rex", "new york"],
        category="updates"
    ),
    
    # === Negations ===
    TestCase(
        name="Negations",
        input_turns=[
            "I don't have any cats",
            "I've never been to Japan",
            "I don't drink coffee",
            "I'm not married"
        ],
        query_turn="What don't I have or do?",
        expected_facts=[
            # Negations should be tracked differently
        ],
        expected_recall=["don't", "never", "not"],
        category="negations"
    ),
    
    # === Questions Within Context ===
    TestCase(
        name="Questions as context",
        input_turns=[
            "Do you know what my favorite sport is? It's tennis",
            "Can you remember that I'm allergic to peanuts?",
            "I wonder if you'll recall that I speak three languages"
        ],
        query_turn="What special things should you remember?",
        expected_facts=[
            ("favorite sport", "attr_of", "tennis"),
            ("you", "allergic_to", "peanuts"),
            ("you", "speaks", "three languages")
        ],
        expected_recall=["tennis", "peanuts", "languages"],
        category="questions"
    ),
    
    # === Mixed Languages (if supported) ===
    TestCase(
        name="Mixed languages",
        input_turns=[
            "My name is Giovanni",
            "Mi piace la pizza",  # I like pizza (Italian)
            "Je vis à Paris",  # I live in Paris (French)
            "Ich arbeite bei BMW"  # I work at BMW (German)
        ],
        query_turn="What do you know about me?",
        expected_facts=[
            ("you", "name", "giovanni"),
            # Language-specific extraction might vary
        ],
        expected_recall=["giovanni"],
        category="multilingual"
    )
]


def run_test_case(processor: HotPathMemoryProcessor, test_case: TestCase) -> Dict[str, Any]:
    """Run a single test case and return results"""
    results = {
        "name": test_case.name,
        "category": test_case.category,
        "success": True,
        "extracted_facts": [],
        "retrieval_performance": [],
        "recall_success": [],
        "errors": []
    }
    
    print(f"\n{'='*60}")
    print(f"Test: {test_case.name} ({test_case.category})")
    print(f"{'='*60}")
    
    # Process input turns
    all_triples = []
    for i, turn in enumerate(test_case.input_turns, 1):
        print(f"\nTurn {i}: '{turn}'")
        start = time.perf_counter()
        
        try:
            bullets, triples = processor.hot.process_turn(
                turn,
                session_id=f"test-{test_case.name}",
                turn_id=i
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            print(f"  Extracted: {triples}")
            print(f"  Time: {elapsed_ms:.1f}ms")
            
            all_triples.extend(triples)
            results["retrieval_performance"].append(elapsed_ms)
            
        except Exception as e:
            results["errors"].append(f"Turn {i}: {str(e)}")
            results["success"] = False
    
    results["extracted_facts"] = all_triples
    
    # Test query and retrieval
    print(f"\nQuery: '{test_case.query_turn}'")
    start = time.perf_counter()
    
    try:
        bullets, triples = processor.hot.process_turn(
            test_case.query_turn,
            session_id=f"test-{test_case.name}",
            turn_id=len(test_case.input_turns) + 1
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        print(f"  Retrieved bullets:")
        for bullet in bullets:
            print(f"    {bullet}")
        print(f"  Time: {elapsed_ms:.1f}ms")
        
        # Check recall
        bullets_text = " ".join(bullets).lower()
        for expected_word in test_case.expected_recall:
            if expected_word.lower() in bullets_text:
                results["recall_success"].append(expected_word)
                print(f"  ✅ Found '{expected_word}'")
            else:
                print(f"  ❌ Missing '{expected_word}'")
                results["success"] = False
                
    except Exception as e:
        results["errors"].append(f"Query: {str(e)}")
        results["success"] = False
    
    # Check if expected facts were extracted
    all_triples_str = str(all_triples).lower()
    for expected_fact in test_case.expected_facts:
        fact_found = any(
            all(part.lower() in str(triple).lower() for part in expected_fact)
            for triple in all_triples
        )
        if not fact_found:
            print(f"  ⚠️  Expected fact not extracted: {expected_fact}")
            # Don't fail on this, just warn
    
    return results


def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and summarize test results"""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    # Overall stats
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # By category
    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0, "times": []}
        categories[cat]["total"] += 1
        if result["success"]:
            categories[cat]["passed"] += 1
        categories[cat]["times"].extend(result["retrieval_performance"])
    
    print("\nBy Category:")
    for cat, stats in categories.items():
        avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
        print(f"  {cat:15} {stats['passed']}/{stats['total']} passed, avg {avg_time:.1f}ms")
    
    # Performance analysis
    all_times = []
    for result in results:
        all_times.extend(result["retrieval_performance"])
    
    if all_times:
        import statistics
        print(f"\nPerformance:")
        print(f"  Mean: {statistics.mean(all_times):.1f}ms")
        print(f"  Median: {statistics.median(all_times):.1f}ms")
        print(f"  P95: {statistics.quantiles(all_times, n=20)[18]:.1f}ms" if len(all_times) > 20 else f"  Max: {max(all_times):.1f}ms")
    
    # Failed tests
    failed = [r for r in results if not r["success"]]
    if failed:
        print("\nFailed Tests:")
        for result in failed:
            print(f"  - {result['name']}")
            for error in result.get("errors", []):
                print(f"    Error: {error}")
    
    # Blind spots analysis
    print("\nBlind Spots Detected:")
    blind_spots = []
    
    # Check various aspects
    for result in results:
        if result["category"] == "negations" and not result["success"]:
            blind_spots.append("Negation handling needs improvement")
        if result["category"] == "updates" and not result["success"]:
            blind_spots.append("Fact updating/correction needs work")
        if result["category"] == "complex" and not result["success"]:
            blind_spots.append("Complex possessive chains need refinement")
        if result["category"] == "multilingual" and not result["success"]:
            blind_spots.append("Multilingual support is limited")
    
    if blind_spots:
        for spot in set(blind_spots):
            print(f"  - {spot}")
    else:
        print("  None detected - system handles test cases well!")
    
    return passed == total


def main():
    """Run comprehensive test suite"""
    print("\n" + "="*60)
    print("COMPREHENSIVE HOTMEM TEST SUITE")
    print("="*60)
    
    # Initialize processor
    processor = HotPathMemoryProcessor(
        sqlite_path="test_comprehensive.db",
        lmdb_dir="test_comprehensive.lmdb",
        user_id="test-user"
    )
    
    print("\n✅ HotMem processor initialized")
    print(f"Running {len(TEST_CASES)} test cases across {len(set(tc.category for tc in TEST_CASES))} categories\n")
    
    # Run all tests
    results = []
    for test_case in TEST_CASES:
        result = run_test_case(processor, test_case)
        results.append(result)
        time.sleep(0.1)  # Brief pause between tests
    
    # Analyze results
    all_passed = analyze_results(results)
    
    # Get final metrics
    print("\n" + "="*60)
    print("SYSTEM METRICS")
    print("="*60)
    
    metrics = processor.hot.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict) and 'p95' in value:
            print(f"  {key}: p95={value['p95']:.1f}ms, mean={value['mean']:.1f}ms")
        else:
            print(f"  {key}: {value}")
    
    # Cleanup
    import shutil
    if os.path.exists("test_comprehensive.db"):
        os.remove("test_comprehensive.db")
    if os.path.exists("test_comprehensive.lmdb"):
        shutil.rmtree("test_comprehensive.lmdb")
    print("\nCleaned up test files.")
    
    # Final verdict
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed - see details above")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())