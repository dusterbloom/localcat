#!/usr/bin/env python3
"""
HotMem Evolution Testing: Analyze current performance and identify improvements.
Based on real conversation patterns and edge cases from live usage.
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths

def test_conversation_patterns():
    """Test patterns from actual conversations to identify improvement areas."""
    
    # Initialize HotMem
    paths = Paths(sqlite_path="test_evolution.db", lmdb_dir="test_evolution.lmdb")
    store = MemoryStore(paths)
    hot = HotMemory(store)
    
    # Test cases from real usage
    test_cases = [
        # Complex questions with embedded facts
        {
            "text": "Did I tell you that Potola is five years old?",
            "expected_intent": "question_with_fact",
            "expected_facts": [("Potola", "age", "five years old")],
            "description": "Question containing factual information"
        },
        
        # Repeated information (should reinforce)
        {
            "text": "My dog's name is Potola",
            "expected_intent": "fact_statement", 
            "expected_facts": [("dog", "name", "Potola"), ("you", "has", "dog")],
            "description": "Direct fact statement (repeated)"
        },
        
        # Complex multi-clause sentences
        {
            "text": "I live in San Francisco, work at Google, and my favorite color is blue",
            "expected_intent": "multiple_facts",
            "expected_facts": [
                ("you", "lives_in", "San Francisco"),
                ("you", "works_at", "Google"), 
                ("you", "favorite_color", "blue")
            ],
            "description": "Multiple facts in compound sentence"
        },
        
        # Conversational context
        {
            "text": "Amazing!",
            "expected_intent": "reaction",
            "expected_facts": [],
            "description": "Emotional reaction (should not extract facts)"
        },
        
        # Temporal context
        {
            "text": "I used to live in New York but now I'm in California",
            "expected_intent": "temporal_fact",
            "expected_facts": [
                ("you", "previously_lived_in", "New York"),
                ("you", "currently_lives_in", "California")
            ],
            "description": "Temporal context with past vs present"
        },
        
        # Conditional/hypothetical
        {
            "text": "If I had a cat, I would name it Whiskers",
            "expected_intent": "hypothetical",
            "expected_facts": [],  # Should not extract hypothetical facts
            "description": "Hypothetical statement (should not extract)"
        },
        
        # Corrections/updates
        {
            "text": "Actually, Potola is six years old, not five",
            "expected_intent": "correction",
            "expected_facts": [("Potola", "age", "six years old")],
            "description": "Fact correction (should update/demote old fact)"
        }
    ]
    
    print("=== HotMem Evolution Analysis ===\n")
    
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}: {case['description']}")
        print(f"Input: '{case['text']}'")
        
        start = time.perf_counter()
        bullets, facts = hot.process_turn(case['text'], "test-user", i)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        print(f"Extracted facts: {facts}")
        print(f"Generated bullets: {bullets[:3]}")
        print(f"Processing time: {elapsed_ms:.1f}ms")
        
        # Analyze quality
        quality_score = analyze_extraction_quality(case, facts, bullets)
        results.append({
            'case': case,
            'facts': facts,
            'bullets': bullets,
            'time_ms': elapsed_ms,
            'quality': quality_score
        })
        
        print(f"Quality score: {quality_score}/10")
        print("-" * 60)
    
    # Summary analysis
    print("\n=== ANALYSIS SUMMARY ===")
    avg_time = sum(r['time_ms'] for r in results) / len(results)
    avg_quality = sum(r['quality'] for r in results) / len(results)
    
    print(f"Average processing time: {avg_time:.1f}ms")
    print(f"Average quality score: {avg_quality:.1f}/10")
    print(f"Time budget remaining: {200 - avg_time:.1f}ms")
    
    # Identify improvement areas
    print("\n=== IMPROVEMENT OPPORTUNITIES ===")
    
    low_quality = [r for r in results if r['quality'] < 7]
    if low_quality:
        print("🔧 Low quality extractions:")
        for r in low_quality:
            print(f"  - {r['case']['description']}: {r['quality']}/10")
    
    slow_cases = [r for r in results if r['time_ms'] > 50]
    if slow_cases:
        print("⏱️  Slow processing cases:")
        for r in slow_cases:
            print(f"  - {r['case']['description']}: {r['time_ms']:.1f}ms")
    
    # Specific recommendations
    print("\n=== ENHANCEMENT RECOMMENDATIONS ===")
    print("1. 🎯 Intent Classification: Distinguish questions vs statements vs reactions")
    print("2. 🔄 Fact Reinforcement: Track repeated information and boost confidence") 
    print("3. 📝 Complex Parsing: Better handling of compound and conditional sentences")
    print("4. 🎚️  Quality Filtering: Score and filter low-quality facts before storage")
    print("5. ⏰ Temporal Awareness: Handle past vs present tense properly")
    print("6. 🔧 Fact Updates: Detect and handle corrections/updates to existing facts")
    
    return results

def analyze_extraction_quality(case, facts, bullets):
    """Analyze the quality of extraction results."""
    score = 10.0
    
    # Check if expected facts were found
    expected = case.get('expected_facts', [])
    if expected:
        found_facts = len([f for f in expected if any(
            f[0].lower() in fact_str.lower() and f[2].lower() in fact_str.lower() 
            for fact_str in [str(f) for f in facts]
        )])
        if found_facts == 0 and len(expected) > 0:
            score -= 4  # Major penalty for missing expected facts
        elif found_facts < len(expected):
            score -= 2  # Penalty for partial extraction
    
    # Check for useless facts
    useless_patterns = [
        ('you', 'tell', 'you'),  # Self-referential
        ('you', 'know', 'you'),  # Obvious
        ('i', 'tell', 'i'),      # Self-referential
    ]
    
    useless_count = sum(1 for fact in facts if fact in useless_patterns)
    score -= useless_count * 1.5
    
    # Check intent appropriateness
    intent = case.get('expected_intent', '')
    if intent == 'reaction' and len(facts) > 0:
        score -= 3  # Shouldn't extract facts from reactions
    elif intent == 'hypothetical' and len(facts) > 0:
        score -= 3  # Shouldn't extract hypothetical facts
    
    # Quality of bullets
    if not bullets and len(facts) > 0:
        score -= 2  # Should generate bullets for facts
    
    # Avoid generic bullets
    generic_bullets = [b for b in bullets if any(generic in b.lower() for generic in ['you has', 'you is', 'you are'])]
    score -= len(generic_bullets) * 0.5
    
    return max(0, min(10, score))

if __name__ == "__main__":
    test_conversation_patterns()