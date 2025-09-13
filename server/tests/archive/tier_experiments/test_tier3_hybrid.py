#!/usr/bin/env python3

"""
Test Tier 3 Hybrid approach vs Full approach
Compares performance and quality of hybrid (Tier1 entities + LLM relationships) vs full LLM extraction
"""

import time
from components.extraction.tiered_extractor import TieredRelationExtractor

def test_tier3_hybrid_vs_full():
    """Compare hybrid vs full Tier 3 extraction"""
    
    extractor = TieredRelationExtractor()
    
    # Test sentences
    test_sentences = [
        "Dr. Sarah Williams works at MIT and develops artificial intelligence systems.",
        "The comprehensive research paper published by the Stanford team demonstrated how quantum computing algorithms could solve complex optimization problems.",
        "When the international climate summit concluded, representatives from 195 countries reached a historic agreement that addressed carbon emissions reduction targets.",
        "Despite challenging economic conditions, the innovative startup founded by Harvard graduates secured funding from multiple venture capital firms.",
        "The groundbreaking medical treatment developed through collaboration between Oxford and Cambridge universities showed remarkable efficacy in clinical trials."
    ]
    
    print("=== Tier 3 Hybrid vs Full Comparison ===\n")
    
    results = []
    
    for i, sentence in enumerate(test_sentences):
        print(f"Test {i+1}: {sentence[:80]}...")
        
        # Test hybrid approach first
        print("  Hybrid Approach:")
        start_time = time.perf_counter()
        
        try:
            # Get Tier 1 entities first
            tier1_result = extractor._extract_tier1(sentence)
            tier1_time = (time.perf_counter() - start_time) * 1000
            
            # Now test hybrid approach
            hybrid_result = extractor._extract_tier3(sentence)
            hybrid_time = (time.perf_counter() - start_time) * 1000
            
            print(f"    Tier 1 entities: {len(tier1_result.entities)} ({tier1_time:.1f}ms)")
            print(f"    Total time: {hybrid_time:.1f}ms")
            print(f"    Entities: {len(hybrid_result.entities)}")
            print(f"    Relationships: {len(hybrid_result.relationships)}")
            print(f"    Confidence: {hybrid_result.confidence:.2f}")
            
            if hybrid_result.relationships:
                print("    Relationships:")
                for rel in hybrid_result.relationships[:3]:
                    print(f"      - {rel}")
                    
        except Exception as e:
            hybrid_time = (time.perf_counter() - start_time) * 1000
            print(f"    ERROR: {e}")
            print(f"    Failed after: {hybrid_time:.1f}ms")
        
        # Test full approach
        print("  Full Approach:")
        start_time = time.perf_counter()
        
        try:
            full_result = extractor._extract_tier3_full(sentence)
            full_time = (time.perf_counter() - start_time) * 1000
            
            print(f"    Total time: {full_time:.1f}ms")
            print(f"    Entities: {len(full_result.entities)}")
            print(f"    Relationships: {len(full_result.relationships)}")
            print(f"    Confidence: {full_result.confidence:.2f}")
            
            if full_result.relationships:
                print("    Relationships:")
                for rel in full_result.relationships[:3]:
                    print(f"      - {rel}")
                    
        except Exception as e:
            full_time = (time.perf_counter() - start_time) * 1000
            print(f"    ERROR: {e}")
            print(f"    Failed after: {full_time:.1f}ms")
        
        print()
    
    print("=== Summary ===")
    print("Note: The actual implementation now uses the hybrid approach by default")
    print("This shows the performance improvement from using Tier 1 entities + LLM relationships only")

if __name__ == "__main__":
    test_tier3_hybrid_vs_full()