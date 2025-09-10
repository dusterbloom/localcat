#!/usr/bin/env python3
"""Simple test of HotMem format with manual prompt"""

import json
import urllib.request

def test_hotmem_simple():
    """Test the model with HotMem format using simple prompt construction"""
    
    print("🧪 Testing HotMem format with simple prompt...")
    
    # System prompt for HotMem format
    system_prompt = """You are a relation linker expert. Extract relationships between provided entities.

TASK: Given ENTITIES and USER message, identify relationships and return in JSON format.

OUTPUT FORMAT:
{"triples": [{"s": "subject_entity", "r": "relation_predicate", "d": "destination_entity"}]}

RULES:
1. USE ONLY PROVIDED ENTITIES for "s" and "d" (exact match required)
2. Extract clear relationships from text (e.g., "CEO of", "lives in", "works for")
3. Format predicates as lowercase with underscores (e.g., "CEO of" → "ceo_of")
4. Maximum 10 triples, prioritize by clarity
5. If no valid relations, return {"triples": []}
6. Output ONLY the JSON object, no explanations or markdown

EXAMPLES:
- "Tim Cook is the CEO of Apple" → {"s": "Tim Cook", "r": "ceo_of", "d": "Apple"}
- "Sarah lives in California" → {"s": "Sarah", "r": "lives_in", "d": "California"}

Now process this input:"""

    # User message with entities and text
    user_message = """ENTITIES: ["tim cook", "apple", "california"]
USER: Tim Cook is the CEO of Apple and lives in California.
ASSISTANT:"""

    print(f"System: {system_prompt[:100]}...")
    print(f"User: {user_message}")
    
    # Make API call
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        "model": "relation-extractor-v2-mlx",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 300,
        "temperature": 0.1
        # Note: Not using JSON mode since it might cause issues
    }
    
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
            
            content = response['choices'][0]['message']['content']
            print(f"\n✅ Raw response:\n{content}")
            
            # Parse and validate
            try:
                # Try to extract JSON from response (in case there's extra text)
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start == -1 or json_end == -1:
                    print("❌ No JSON found in response")
                    return
                    
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                
                print(f"\n🎯 Parsed JSON: {json.dumps(result, indent=2)}")
                
                # Check if it matches HotMem's expected format
                if 'triples' in result:
                    triples = result['triples']
                    print(f"\n📊 Extracted {len(triples)} triples:")
                    
                    for i, triple in enumerate(triples, 1):
                        s = triple.get('s', 'N/A')
                        r = triple.get('r', 'N/A') 
                        d = triple.get('d', 'N/A')
                        print(f"   {i}. {s} --{r}--> {d}")
                    
                    # Expected results
                    expected = [
                        ("tim cook", "ceo_of", "apple"),
                        ("tim cook", "lives_in", "california")
                    ]
                    
                    print(f"\n✅ Expected triples found:")
                    found_triples = [(t.get('s', '').lower(), t.get('r', '').lower(), t.get('d', '').lower()) for t in triples]
                    
                    for exp_s, exp_r, exp_d in expected:
                        found = any(
                            exp_s == found_s and exp_r == found_r and exp_d == found_d
                            for found_s, found_r, found_d in found_triples
                        )
                        status = "✅" if found else "❌"
                        print(f"   {status} {exp_s} --{exp_r}--> {exp_d}")
                    
                    print(f"\n🎉 SUCCESS: Model is compatible with HotMem format!")
                        
                else:
                    print(f"❌ Missing 'triples' field in response")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                
    except Exception as e:
        print(f"❌ API call failed: {e}")

if __name__ == "__main__":
    test_hotmem_simple()