#!/usr/bin/env python3

"""
Debug Tier 3 JSON response to see what the LLM is actually returning
"""

import json
import time
from components.extraction.tiered_extractor import TieredRelationExtractor

def debug_tier3_response():
    """Debug what Tier 3 LLM is actually returning"""
    
    extractor = TieredRelationExtractor()
    
    # Simple test sentence
    test_text = "When the international climate summit concluded, representatives from 195 countries reached a historic agreement"
    
    print(f"Testing: {test_text}")
    print("=" * 50)
    
    # Let's manually call the Tier 3 method to see the raw response
    try:
        # Get the raw system and user prompts
        system_prompt = """
                System: You are a world-class AI model specialized in extracting knowledge graphs from text. You analyze the input text to identify key entities and their relationships.
                Output ONLY valid JSON matching the provided schema. Do not add explanations, markdown, or extra text. If no entities/relationships are found, return an empty graph.

JSON Schema: {
  "entities": [
    {
      "id": "unique_id",
      "label": "entity_type", 
      "name": "entity_name"
    }
  ],
  "relationships": [
    {
      "source": "source_node_id",
      "target": "target_node_id", 
      "label": "relationship_type",
      "confidence": "Float"
    }
  ]
}

User: Extract a knowledge graph from this text. First, identify main entities (e.g., people, places, concepts). Then, find relationships between them (e.g., "works at", "located in"). Assign unique IDs to nodes starting from 1.

Text: Alice works at Google in California. She loves hiking in the mountains.

Example Input Text: Bob is CEO of Apple. He lives in New York.
Example Output: {
  "entities": [
    {"id": "1", "label": "Person", "name": "Bob"},
    {"id": "2", "label": "Company", "name": "Apple"},
    {"id": "3", "label": "Location", "name": "New York"}
  ],
  "relationships": [
    {"source": "1", "target": "2", "label": "CEO of", "confidence": 0.9 },
    {"source": "1", "target": "3", "label": "lives in", "confidence": 0.9 }
  ]
}

Now, extract from the provided text.
             """

        user_prompt = f"Text: {test_text}. "
        
        # Make the API call directly to see raw response
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
            
            print("RAW LLM RESPONSE:")
            print("-" * 30)
            print(repr(content))
            print("-" * 30)
            print("\nFORMATTED RESPONSE:")
            print(content)
            
            # Try to parse JSON
            print("\nJSON PARSING ATTEMPTS:")
            print("-" * 30)
            
            # Try 1: Direct parse
            try:
                parsed = json.loads(content)
                print("✅ Direct JSON parse successful")
                print(f"Parsed: {json.dumps(parsed, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"❌ Direct parse failed: {e}")
                
                # Try 2: Extract JSON block
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = content[json_start:json_end]
                    print(f"Extracted JSON block: {repr(json_text)}")
                    
                    try:
                        parsed = json.loads(json_text)
                        print("✅ Extracted block parse successful")
                        print(f"Parsed: {json.dumps(parsed, indent=2)}")
                    except json.JSONDecodeError as e2:
                        print(f"❌ Extracted block parse failed: {e2}")
                        
                        # Try 3: Fix common issues
                        try:
                            json_text = json_text.replace('name', '"name"').replace('type', '"type"')
                            json_text = json_text.replace('source', '"source"').replace('relation', '"relation"').replace('target', '"target"')
                            parsed = json.loads(json_text)
                            print("✅ Fixed JSON parse successful")
                            print(f"Parsed: {json.dumps(parsed, indent=2)}")
                        except json.JSONDecodeError as e3:
                            print(f"❌ Fixed JSON parse failed: {e3}")
                else:
                    print("❌ No JSON block found")
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_tier3_response()