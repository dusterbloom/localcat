#!/usr/bin/env python3
"""
Focused Tier 3 test - check the actual markdown output and parsing
"""

import json
import time
from components.extraction.tiered_extractor import TieredRelationExtractor

class SimpleTier3Test:
    """Simple Tier 3 test to see what's happening"""
    
    def __init__(self):
        self.extractor = TieredRelationExtractor()
        
    def test_tier3_simple(self):
        """Test Tier 3 with a simple case"""
        
        print("🔍 SIMPLE TIER 3 TEST")
        print("=" * 50)
        
        # Simple test case
        text = "Alice works at Tesla. Bob manages Alice."
        
        print(f"Text: {text}")
        
        # Run full extraction
        start_time = time.perf_counter()
        result = self.extractor._extract_tier3(text)
        extraction_time = (time.perf_counter() - start_time) * 1000
        
        print(f"\n📊 RESULT:")
        print(f"   Time: {extraction_time:.1f}ms")
        print(f"   Entities: {len(result.entities)}")
        print(f"   Relationships: {len(result.relationships)}")
        print(f"   Tier used: {result.tier_used}")
        print(f"   Success: {result}")
        
        print(f"\n📝 ENTITIES:")
        for ent in result.entities:
            print(f"   - {ent}")
            
        print(f"\n🔗 RELATIONSHIPS:")
        for rel in result.relationships:
            print(f"   - {rel}")
            
        # Now test the raw LLM call
        print(f"\n🤖 RAW LLM TEST:")
        print("-" * 30)
        
        # Get entities first
        tier1_result = self.extractor._extract_tier1(text)
        entities = [str(ent) for ent in tier1_result.entities]
        print(f"Entities from Tier 1: {entities}")
        
        # Test the exact prompt being used
        system_prompt = """You are a relationship extraction expert. Given text and pre-identified entities, find relationships between them.
Output in markdown format:
## Relationships
- Entity1 -> relationship_type -> Entity2
- Entity3 -> relationship_type -> Entity4
Use only exact entity names from the provided list. If no relationships exist, output "## Relationships" followed by "No relationships found."""
        
        entity_list = "\n".join([f"- {entity}" for entity in entities])
        user_prompt = f"""Text: {text}
Entities found in text:
{entity_list}
Extract relationships between these entities:"""
        
        print(f"System prompt: {system_prompt}")
        print(f"User prompt: {user_prompt}")
        
        # Make the call
        raw_response = self.extractor._call_llm_tier3_markdown(system_prompt, user_prompt, self.extractor.tier3_model)
        print(f"\n📄 RAW RESPONSE:")
        print(f"{repr(raw_response)}")
        
        # Test parsing
        if raw_response:
            try:
                parsed = self.extractor._parse_markdown_relationships(raw_response, entities)
                print(f"\n✅ PARSED: {parsed}")
            except Exception as e:
                print(f"\n❌ PARSING ERROR: {e}")
                
                # Try manual parsing
                print(f"\n🔧 MANUAL PARSING ATTEMPT:")
                lines = raw_response.split('\n')
                for i, line in enumerate(lines):
                    print(f"   Line {i}: {repr(line)}")

if __name__ == "__main__":
    tester = SimpleTier3Test()
    tester.test_tier3_simple()