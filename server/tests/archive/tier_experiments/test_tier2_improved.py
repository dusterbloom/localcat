#!/usr/bin/env python3
"""
Test improved Tier 2 extraction with fixes for qwen3-0.6b-mlx model
"""

import sys
import os
sys.path.append('/Users/peppi/Dev/localcat/server')

from components.extraction.tiered_extractor import TieredRelationExtractor
import time

def test_tier2_extraction():
    """Test Tier 2 extraction with various sentences"""
    print("=== Testing Tier 2 Extraction (qwen3-0.6b-mlx) ===\n")
    
    # Initialize extractor with Tier 2 enabled
    extractor = TieredRelationExtractor(
        enable_gliner=False,  # Disable GLiNER for this test to focus on LLM
        enable_srl=False,
        enable_coref=False,
        llm_timeout_ms=10000  # 10 second timeout for testing
    )
    
    test_cases = [
        {
            "text": "Alice works at Tesla and reports to Bob who is the CTO.",
            "expected_entities": ["alice", "tesla", "bob"],
            "expected_relationships": ["works_at", "reports_to"]
        },
        {
            "text": "John studied at Reed College and now teaches mathematics there.",
            "expected_entities": ["john", "reed college", "mathematics"],
            "expected_relationships": ["studied_at", "teaches"]
        },
        {
            "text": "Sarah drives a Model S and lives in San Francisco.",
            "expected_entities": ["sarah", "model s", "san francisco"],
            "expected_relationships": ["drives", "lives_in"]
        },
        {
            "text": "The dog name is Potola",
            "expected_entities": ["dog", "potola"],
            "expected_relationships": ["name"]
        },
        # Add more complex sentences that should trigger Tier 2
        {
            "text": "The sophisticated artificial intelligence system developed by the research team at MIT has been successfully implemented across multiple departments within the organization, significantly improving operational efficiency and decision-making processes while simultaneously reducing costs and enhancing user satisfaction metrics throughout the entire enterprise ecosystem.",
            "expected_entities": ["artificial intelligence system", "research team", "mit"],
            "expected_relationships": ["developed_by", "implemented_at"]
        },
        {
            "text": "Notwithstanding the considerable challenges encountered during the implementation phase, the comprehensive digital transformation initiative spearheaded by senior management ultimately achieved its objectives despite initial resistance from various stakeholders throughout the organizational hierarchy.",
            "expected_entities": ["digital transformation initiative", "senior management"],
            "expected_relationships": ["spearheaded_by"]
        }
    ]
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['text']}")
        
        try:
            start_time = time.time()
            result = extractor.extract(test_case['text'])
            elapsed_ms = (time.time() - start_time) * 1000
            
            print(f"  Tier used: {result.tier_used}")
            print(f"  Time: {elapsed_ms:.1f}ms")
            print(f"  Entities: {result.entities}")
            print(f"  Relationships: {result.relationships}")
            print(f"  Confidence: {result.confidence:.2f}")
            
            # Check if this used Tier 2
            if result.tier_used == 2:
                print("  ✅ Used Tier 2 (LLM)")
                success_count += 1
            else:
                print("  ⚠️  Used fallback tier")
            
            # Check expectations
            entities_found = len(result.entities) >= 2
            relationships_found = len(result.relationships) >= 1
            
            if entities_found and relationships_found:
                print("  ✅ Good extraction quality")
            else:
                print("  ⚠️  Could improve extraction quality")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print("-" * 60)
    
    print(f"\n=== Direct Tier 2 Test ===")
    print("Testing Tier 2 method directly...")
    
    direct_test_text = "Alice works at Tesla and reports to Bob who is the CTO."
    try:
        start_time = time.time()
        direct_result = extractor._extract_tier2(direct_test_text)
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"Direct Tier 2 test:")
        print(f"  Text: {direct_test_text}")
        print(f"  Time: {elapsed_ms:.1f}ms")
        print(f"  Entities: {direct_result.entities}")
        print(f"  Relationships: {direct_result.relationships}")
        print(f"  Confidence: {direct_result.confidence:.2f}")
        
        if direct_result.entities and direct_result.relationships:
            print("  ✅ Direct Tier 2 test successful")
            success_count += 1
        else:
            print("  ❌ Direct Tier 2 test failed")
            
    except Exception as e:
        print(f"  ❌ Direct Tier 2 test error: {e}")
    
    print(f"\n=== Results ===")
    print(f"Tests passed: {success_count}/{total_tests + 1}")  # +1 for direct test
    print(f"Success rate: {success_count/(total_tests + 1)*100:.1f}%")
    
    # Show metrics
    metrics = extractor.get_metrics()
    print(f"\nMetrics: {metrics}")
    
    return success_count == total_tests + 1

if __name__ == "__main__":
    success = test_tier2_extraction()
    sys.exit(0 if success else 1)