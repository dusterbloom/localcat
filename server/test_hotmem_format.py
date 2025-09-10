#!/usr/bin/env python3
"""Test HotMem assisted extraction with correct format"""

import json
import urllib.request
from jinja2 import Template

def load_template():
    """Load the Jinja template"""
    with open('hotmem_assisted_prompt_template.jinja', 'r') as f:
        return Template(f.read())

def test_hotmem_format():
    """Test the model with HotMem's expected format"""
    
    # Load template
    template = load_template()
    
    # Test data
    entities = ["tim cook", "apple", "california"]
    user_text = "Tim Cook is the CEO of Apple and lives in California."
    
    # Render prompt
    prompt = template.render(
        entities=entities,
        user_text=user_text
    )
    
    print("🧪 Testing HotMem format with LM Studio...")
    print(f"\n📝 Entities: {entities}")
    print(f"📝 User text: {user_text}")
    print(f"\n📋 Generated prompt:\n{prompt}")
    
    # Make API call
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        "model": "relation-extractor-v2-mlx",
        "messages": [
            {"role": "system", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
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
            print(f"\n✅ Raw response: {content}")
            
            # Parse and validate
            try:
                result = json.loads(content)
                
                # Check if it matches HotMem's expected format
                if 'triples' in result:
                    triples = result['triples']
                    print(f"\n🎯 Extracted {len(triples)} triples:")
                    
                    for i, triple in enumerate(triples, 1):
                        s = triple.get('s', 'N/A')
                        r = triple.get('r', 'N/A') 
                        d = triple.get('d', 'N/A')
                        print(f"   {i}. {s} --{r}--> {d}")
                        
                        # Validate entities match
                        if s.lower() not in [e.lower() for e in entities]:
                            print(f"      ⚠️  Subject '{s}' not in entities list")
                        if d.lower() not in [e.lower() for e in entities]:
                            print(f"      ⚠️  Destination '{d}' not in entities list")
                    
                    # Expected triples for this test
                    expected = [
                        ("tim cook", "ceo_of", "apple"),
                        ("tim cook", "lives_in", "california")
                    ]
                    
                    print(f"\n📊 Validation:")
                    found_triples = [(t.get('s', '').lower(), t.get('r', '').lower(), t.get('d', '').lower()) for t in triples]
                    
                    for exp_s, exp_r, exp_d in expected:
                        found = any(
                            exp_s == found_s and exp_r == found_r and exp_d == found_d
                            for found_s, found_r, found_d in found_triples
                        )
                        status = "✅" if found else "❌"
                        print(f"   {status} {exp_s} --{exp_r}--> {exp_d}")
                        
                else:
                    print(f"❌ Missing 'triples' field in response")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                
    except Exception as e:
        print(f"❌ API call failed: {e}")

if __name__ == "__main__":
    test_hotmem_format()