#!/usr/bin/env python3
"""
Quick iteration test for LLM prompts and parameters
Focus on fixing:
1. Tier 2 (qwen3) repetition issue - stops after extraction
2. Tier 3 output format optimization
"""

import httpx
import json
import time

LLM_BASE_URL = "http://127.0.0.1:1234/v1"
TIER2_MODEL = "qwen3-0.6b-mlx"
TIER3_MODEL = "mlx-community/llama-3.2-1b-instruct"

def test_tier2_prompts():
    """Test different Tier 2 prompts to stop repetition"""
    text = "Alice works at Tesla and drives a Model 3. She reports to Bob who is the CTO."
    
    prompts = [
        # Original problematic prompt
        {
            "name": "Original",
            "prompt": f"""Extract the key facts from this text.

Instructions:
1. Find all people, companies, and products
2. Find the main relationships between them  
3. Ignore repetitions
4. Keep it simple

Text: {text}

Output clean JSON with:
- entities: array of {{"name", "type"}}
- relationships: array of {{"source", "relation", "target"}}

Limit to the 10 most important relationships.

JSON:"""
        },
        
        # Attempt 1: Clear stop instruction
        {
            "name": "With Stop",
            "prompt": f"""Extract facts from: {text}

Output ONLY this JSON format, then STOP:
{{"entities": [{{"name": "PersonName", "type": "Person"}}], "relationships": [{{"source": "Person", "relation": "works_at", "target": "Company"}}]}}

JSON:"""
        },
        
        # Attempt 2: One-shot example
        {
            "name": "One-Shot",
            "prompt": f"""Extract facts as JSON.

Example:
Text: "John works at Apple"
{{"entities": [{{"name": "John", "type": "Person"}}, {{"name": "Apple", "type": "Company"}}], "relationships": [{{"source": "John", "relation": "works_at", "target": "Apple"}}]}}

Text: {text}
JSON:"""
        },
        
        # Attempt 3: Direct instruction
        {
            "name": "Direct",
            "prompt": f"""Text: {text}

Extract as JSON (entities and relationships only):"""
        }
    ]
    
    for i, prompt_test in enumerate(prompts):
        print(f"\n=== TIER 2 TEST {i+1}: {prompt_test['name']} ===")
        
        # Test with different stop sequences
        for stop_seq in [None, ["\n\n"], ["JSON:", "Example:"], ["}"]]:
            print(f"Stop sequence: {stop_seq}")
            
            try:
                payload = {
                    "model": TIER2_MODEL,
                    "prompt": prompt_test["prompt"],
                    "max_tokens": 200,  # Reduced to prevent repetition
                    "temperature": 0.1,
                    "top_p": 0.9
                }
                
                if stop_seq:
                    payload["stop"] = stop_seq
                
                response = httpx.post(f"{LLM_BASE_URL}/completions", json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    text_output = result.get('choices', [{}])[0].get('text', '')
                    
                    print(f"Output: {text_output[:200]}...")
                    
                    # Try to extract JSON
                    json_start = text_output.find('{')
                    json_end = text_output.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = text_output[json_start:json_end]
                        try:
                            parsed = json.loads(json_text)
                            print(f"✅ SUCCESS: {parsed}")
                        except json.JSONDecodeError:
                            print(f"❌ JSON Parse Error")
                    else:
                        print("❌ No JSON found")
                else:
                    print(f"❌ API Error: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Exception: {e}")
            
            print("-" * 50)

def test_tier3_formats():
    """Test Tier 3 with different output formats"""
    text = "Alice works at Tesla and drives a Model 3. She reports to Bob who is the CTO."
    
    formats = [
        {
            "name": "Current Markdown",
            "prompt": f"""Extract the key facts from this complex text.

Instructions:
1. Find all people, companies, and products
2. Find the main relationships between them
3. Ignore repetitions
4. Keep it simple

Text: {text}

Output as markdown with:
## Entities
- Name (Type)

## Relationships  
- Source -> Relation -> Target

Limit to the 10 most important relationships."""
        },
        
        {
            "name": "Clean JSON",
            "prompt": f"""Extract facts from this text as clean JSON.

Text: {text}

Return only valid JSON with entities and relationships:"""
        },
        
        {
            "name": "Structured List",
            "prompt": f"""Extract key facts from: {text}

ENTITIES:
- Alice (Person)
- Tesla (Company)

RELATIONSHIPS:
- Alice -> works_at -> Tesla

Follow this exact format:"""
        }
    ]
    
    for i, format_test in enumerate(formats):
        print(f"\n=== TIER 3 TEST {i+1}: {format_test['name']} ===")
        
        try:
            response = httpx.post(
                f"{LLM_BASE_URL}/chat/completions",
                json={
                    "model": TIER3_MODEL,
                    "messages": [{"role": "user", "content": format_test["prompt"]}],
                    "max_tokens": 300,
                    "temperature": 0.1,
                    "top_p": 0.9
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"Output:\n{content}")
                print("✅ SUCCESS")
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("-" * 70)

def quick_test(tier, prompt, **params):
    """Quick single test for rapid iteration"""
    text = "Alice works at Tesla and drives a Model 3. She reports to Bob who is the CTO."
    
    print(f"\n=== QUICK TEST: Tier {tier} ===")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Params: {params}")
    
    try:
        if tier == 2:
            payload = {
                "model": TIER2_MODEL,
                "prompt": prompt.format(text=text),
                "max_tokens": params.get('max_tokens', 200),
                "temperature": params.get('temperature', 0.1),
                "top_p": params.get('top_p', 0.9)
            }
            if 'stop' in params:
                payload['stop'] = params['stop']
                
            response = httpx.post(f"{LLM_BASE_URL}/completions", json=payload, timeout=10)
            
        else:  # tier == 3
            response = httpx.post(
                f"{LLM_BASE_URL}/chat/completions",
                json={
                    "model": TIER3_MODEL,
                    "messages": [{"role": "user", "content": prompt.format(text=text)}],
                    "max_tokens": params.get('max_tokens', 300),
                    "temperature": params.get('temperature', 0.1),
                    "top_p": params.get('top_p', 0.9)
                },
                timeout=10
            )
        
        if response.status_code == 200:
            result = response.json()
            if tier == 2:
                output = result.get('choices', [{}])[0].get('text', '')
            else:
                output = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            print(f"Output:\n{output}")
            return output
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    return None

def main():
    print("=== LLM Prompt Iteration Tool ===")
    print("1. Run full Tier 2 tests")
    print("2. Run full Tier 3 tests") 
    print("3. Quick test mode")
    print("4. Exit")
    
    while True:
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == "1":
            test_tier2_prompts()
        elif choice == "2":
            test_tier3_formats()
        elif choice == "3":
            tier = int(input("Tier (2 or 3): "))
            prompt = input("Prompt template (use {text} placeholder): ")
            
            params = {}
            max_tokens = input("Max tokens (default 200/300): ")
            if max_tokens:
                params['max_tokens'] = int(max_tokens)
            
            temp = input("Temperature (default 0.1): ")
            if temp:
                params['temperature'] = float(temp)
            
            if tier == 2:
                stop = input("Stop sequences (comma separated, optional): ")
                if stop:
                    params['stop'] = [s.strip() for s in stop.split(',')]
            
            quick_test(tier, prompt, **params)
        elif choice == "4":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    # Automated testing
    test_tier2_prompts()
    # test_tier3_formats()
    
    # Interactive mode
    # main()