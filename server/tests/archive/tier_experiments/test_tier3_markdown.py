#!/usr/bin/env python3

"""
Test Tier 3 Markdown approach vs JSON approach
Compares performance and quality of markdown vs JSON output parsing
"""

import time
import json
from components.extraction.tiered_extractor import TieredRelationExtractor

def test_tier3_markdown_vs_json():
    """Compare markdown vs JSON Tier 3 extraction"""
    
    extractor = TieredRelationExtractor()
    
    # Test sentences
    test_sentences = [
        "Dr. Sarah Williams works at MIT and develops artificial intelligence systems.",
        "The comprehensive research paper published by the Stanford team demonstrated how quantum computing algorithms could solve complex optimization problems.",
        "When the international climate summit concluded, representatives from 195 countries reached a historic agreement that addressed carbon emissions reduction targets.",
        "Despite challenging economic conditions, the innovative startup founded by Harvard graduates secured funding from multiple venture capital firms.",
        "The groundbreaking medical treatment developed through collaboration between Oxford and Cambridge universities showed remarkable efficacy in clinical trials."
    ]
    
    print("=== Tier 3 Markdown vs JSON Comparison ===\n")
    
    results = []
    
    for i, sentence in enumerate(test_sentences):
        print(f"Test {i+1}: {sentence[:80]}...")
        
        # Test markdown approach
        print("  Markdown Approach:")
        start_time = time.perf_counter()
        
        try:
            # Get Tier 1 entities first
            tier1_result = extractor._extract_tier1(sentence)
            tier1_time = (time.perf_counter() - start_time) * 1000
            
            # Now test markdown approach
            markdown_result = extractor._extract_tier3(sentence)
            markdown_time = (time.perf_counter() - start_time) * 1000
            
            print(f"    Tier 1 entities: {len(tier1_result.entities)} ({tier1_time:.1f}ms)")
            print(f"    Total time: {markdown_time:.1f}ms")
            print(f"    Entities: {len(markdown_result.entities)}")
            print(f"    Relationships: {len(markdown_result.relationships)}")
            print(f"    Confidence: {markdown_result.confidence:.2f}")
            
            if markdown_result.relationships:
                print("    Relationships:")
                for rel in markdown_result.relationships[:3]:
                    print(f"      - {rel}")
                    
        except Exception as e:
            markdown_time = (time.perf_counter() - start_time) * 1000
            print(f"    ERROR: {e}")
            print(f"    Failed after: {markdown_time:.1f}ms")
        
        # Test old JSON approach for comparison
        print("  JSON Approach (Direct):")
        start_time = time.perf_counter()
        
        try:
            # Manually test the old JSON approach
            tier1_result = extractor._extract_tier1(sentence)
            
            if tier1_result.entities:
                entities = tier1_result.entities
                entity_list = "\n".join([f"- {entity}" for entity in entities])
                
                system_prompt = """You extract relationships between pre-identified entities. Output ONLY JSON:

{
  "relationships": [
    {"source": "entity_from_list", "target": "entity_from_list", "relation": "relation_type"}
  ]
}

Rules: Use only exact entity names from list. Return empty array if no relationships."""

                user_prompt = f"""Text: {sentence}

Entities found in text:
{entity_list}

Extract relationships between these entities:"""
                
                json_result = extractor._call_llm_tier3_json_old(system_prompt, user_prompt, extractor.tier3_model)
                json_time = (time.perf_counter() - start_time) * 1000
                
                if json_result:
                    relationships = extractor._parse_relationships_only(json_result)
                    print(f"    Total time: {json_time:.1f}ms")
                    print(f"    Relationships: {len(relationships)}")
                    
                    if relationships:
                        print("    Relationships:")
                        for rel in relationships[:3]:
                            print(f"      - {rel}")
                else:
                    print(f"    Total time: {json_time:.1f}ms")
                    print("    No JSON result")
            else:
                print("    No entities found")
                
        except Exception as e:
            json_time = (time.perf_counter() - start_time) * 1000
            print(f"    ERROR: {e}")
            print(f"    Failed after: {json_time:.1f}ms")
        
        print()
    
    # Summary
    print("=== Summary ===")
    print("The new markdown approach should:")
    print("- Have higher success rate (fewer parsing errors)")
    print("- Be faster (lower token count)")
    print("- Produce similar or better relationship quality")
    print("- Work with llama 3.2's natural output format")

if __name__ == "__main__":
    test_tier3_markdown_vs_json()