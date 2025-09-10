#!/usr/bin/env python3
"""Direct test of LM Studio relation extractor model"""

import json
import urllib.request

def test_lm_studio_direct():
    """Test direct call to LM Studio relation extractor"""
    
    url = "http://127.0.0.1:1234/v1/chat/completions"
    
    # Simple test payload similar to what HotMem sends
    payload = {
        "model": "relation-extractor-v2-mlx",
        "messages": [
            {
                "role": "system",
                "content": "You are a relation linker. Link relations only between the provided ENTITIES based on the USER message.\nReturn exactly ONE JSON object with a top-level array field 'triples'.\nEach triple item must be an object with keys: s (subject), r (relation), d (destination).\nRules:\n- Use ONLY ENTITIES for s and d (exact match). Do not invent or alter entities. If you cannot match, skip.\n- r must be a short phrase present in the USER text (1–3 tokens), lowercase, spaces → underscores (e.g., 'teaches at' → 'teaches_at').\n- If the text clearly states an age like '3 years old' for an entity, you may output r='age' and d='3 years old'.\n- If no valid relations are found, return {\"triples\": []}.\n- Output only the JSON object. No extra text. No code fences. No trailing commas.\nChecklist: s ∈ ENTITIES; d ∈ ENTITIES; r from text (lowercase, underscores); ≤3 triples; JSON parses."
            },
            {
                "role": "user", 
                "content": "ENTITIES: [\"tim cook\", \"apple\", \"california\"]\nUSER: Tim Cook is the CEO of Apple and lives in California.\nASSISTANT:"
            }
        ],
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    print("🧪 Testing direct LM Studio call...")
    print(f"URL: {url}")
    print(f"Model: relation-extractor-v2-mlx")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        data = json.dumps(payload).encode('utf-8')
        headers = {
            "Content-Type": "application/json", 
            "Authorization": "Bearer not-needed"
        }
        
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode('utf-8')
            response = json.loads(body)
            
            print(f"\n✅ Response received:")
            print(json.dumps(response, indent=2))
            
            # Extract content
            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0].get('message', {}).get('content', '')
                print(f"\n📝 Content: {content}")
                
                # Try to parse as JSON
                try:
                    parsed = json.loads(content)
                    print(f"🎯 Parsed JSON: {parsed}")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parse error: {e}")
                    
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        try:
            error_body = e.read().decode('utf-8')
            print(f"Error details: {error_body}")
        except:
            pass
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_lm_studio_direct()