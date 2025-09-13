#!/usr/bin/env python3

"""
Debug actual markdown output from Tier 3 LLM
"""

import time
from components.extraction.tiered_extractor import TieredRelationExtractor

def debug_markdown_output():
    """Debug what the markdown output actually looks like"""
    
    extractor = TieredRelationExtractor()
    
    # Simple test sentence
    test_text = "Dr. Sarah Williams works at MIT and develops artificial intelligence systems."
    
    print(f"Testing: {test_text}")
    print("=" * 60)
    
    # Get Tier 1 entities
    tier1_result = extractor._extract_tier1(test_text)
    entities = tier1_result.entities
    
    print(f"Tier 1 entities: {entities}")
    print()
    
    # Create the markdown prompt
    system_prompt = """You are a relationship extraction expert. Given text and pre-identified entities, find relationships between them.

Output in markdown format:

## Relationships
- Entity1 -> relationship_type -> Entity2
- Entity3 -> relationship_type -> Entity4

Use only exact entity names from the provided list. If no relationships exist, output "## Relationships" followed by "No relationships found."""

    entity_list = "\n".join([f"- {entity}" for entity in entities])
    user_prompt = f"Text: {test_text}\n\nEntities found in text:\n{entity_list}\n\nExtract relationships between these entities:"
    
    print("SYSTEM PROMPT:")
    print(system_prompt)
    print("\nUSER PROMPT:")
    print(user_prompt)
    print("\n" + "=" * 60)
    
    # Get the actual markdown response
    try:
        markdown_result = extractor._call_llm_tier3_markdown(system_prompt, user_prompt, extractor.tier3_model)
        
        print("ACTUAL MARKDOWN RESPONSE:")
        print("-" * 30)
        print(repr(markdown_result))
        print("-" * 30)
        print("\nFORMATTED RESPONSE:")
        print(markdown_result)
        
        if markdown_result:
            # Parse it
            relationships = extractor._parse_markdown_relationships(markdown_result, entities)
            print(f"\nPARSED RELATIONSHIPS: {len(relationships)}")
            for i, rel in enumerate(relationships):
                print(f"  {i+1}. {rel}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_markdown_output()