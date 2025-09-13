#!/usr/bin/env python3

"""
Debug the new Tier 3 hybrid approach
See what entities Tier 1 provides and what relationships Tier 3 generates
"""

import json
import time
from components.extraction.tiered_extractor import TieredRelationExtractor

def debug_tier3_hybrid():
    """Debug the hybrid Tier 3 approach"""
    
    extractor = TieredRelationExtractor()
    
    test_text = "Dr. Sarah Williams works at MIT and develops artificial intelligence systems."
    
    print(f"Testing: {test_text}")
    print("=" * 60)
    
    # Step 1: Get Tier 1 entities
    print("STEP 1: Tier 1 Entity Extraction")
    print("-" * 30)
    tier1_start = time.perf_counter()
    tier1_result = extractor._extract_tier1(test_text)
    tier1_time = (time.perf_counter() - tier1_start) * 1000
    
    print(f"Tier 1 time: {tier1_time:.1f}ms")
    print(f"Entities found: {len(tier1_result.entities)}")
    for i, entity in enumerate(tier1_result.entities):
        print(f"  {i+1}. {entity}")
    print(f"Relationships from UD patterns: {len(tier1_result.relationships)}")
    for rel in tier1_result.relationships[:3]:
        print(f"  - {rel}")
    
    print()
    
    # Step 2: Generate hybrid prompt
    print("STEP 2: Hybrid Prompt Generation")
    print("-" * 30)
    
    entities = tier1_result.entities
    entity_list = "\\n".join([f"- {entity}" for entity in entities])
    
    system_prompt = """You are a relationship extraction expert. Given a text and pre-identified entities, extract only the relationships between them.

Output ONLY valid JSON with relationships array:
{
  "relationships": [
    {
      "source": "exact_entity_name_from_list", 
      "target": "exact_entity_name_from_list",
      "relation": "relationship_type"
    }
  ]
}

Rules:
- Use ONLY entity names from the provided list
- If no relationships exist, return empty relationships array
- Common relations: works_at, lives_in, developed, founded, CEO_of, located_in, part_of, studied_at"""

    user_prompt = f"""Text: {test_text}

Entities found in text:
{entity_list}

Extract relationships between these entities:"""
    
    print("System Prompt:")
    print(system_prompt[:200] + "...")
    print()
    print("User Prompt:")
    print(user_prompt)
    print()
    
    # Step 3: LLM call
    print("STEP 3: LLM Relationship Extraction")
    print("-" * 30)
    
    llm_start = time.perf_counter()
    
    try:
        import httpx
        
        response = httpx.post(
            f"{extractor.llm_base_url}/chat/completions",
            json={
                "model": extractor.tier3_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.1
            },
            timeout=extractor.llm_timeout_ms / 1000
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            llm_time = (time.perf_counter() - llm_start) * 1000
            
            print(f"LLM time: {llm_time:.1f}ms")
            print("RAW LLM RESPONSE:")
            print("-" * 20)
            print(repr(content))
            print("-" * 20)
            print()
            print("FORMATTED RESPONSE:")
            print(content)
            print()
            
            # Parse the response
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = content[json_start:json_end]
                    parsed = json.loads(json_text)
                    
                    print("PARSED RELATIONSHIPS:")
                    relationships_data = parsed.get('relationships', [])
                    if relationships_data:
                        for i, rel in enumerate(relationships_data):
                            print(f"  {i+1}. {rel['source']} --{rel['relation']}--> {rel['target']}")
                    else:
                        print("  No relationships found")
                else:
                    print("  No relationships array found")
                        
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                
            total_time = tier1_time + llm_time
            print(f"\\nTOTAL HYBRID TIME: {total_time:.1f}ms (Tier 1: {tier1_time:.1f}ms + LLM: {llm_time:.1f}ms)")
            
        else:
            print(f"LLM API Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"LLM Call Error: {e}")

if __name__ == "__main__":
    debug_tier3_hybrid()