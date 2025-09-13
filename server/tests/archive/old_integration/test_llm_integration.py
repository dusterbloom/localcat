#!/usr/bin/env python3
"""
Test LLM integration for Tiered Extractor
Debug 400 errors from qwen3-0.6b-mlx and llama-3.2-1b models
"""

import asyncio
import httpx
import json
from loguru import logger

# LM Studio configuration
LLM_BASE_URL = "http://127.0.0.1:1234/v1"
TIER2_MODEL = "qwen3-0.6b-mlx"
TIER3_MODEL = "qwen/qwen3-1.7b"

def test_tier2_json_extraction(text: str):
    """Test Tier 2 model (qwen3-0.6b-mlx) with JSON output"""
    prompt = f"""Extract the key facts from this text.

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
    
    try:
        response = httpx.post(
            f"{LLM_BASE_URL}/completions",
            json={
                "model": TIER2_MODEL,
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.1
                # Removed response_format - LM Studio doesn't support it
            },
            timeout=10
        )
        
        print(f"[Tier 2] Status: {response.status_code}")
        print(f"[Tier 2] Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            result = response.json()
            text_output = result.get('choices', [{}])[0].get('text', '')
            print(f"[Tier 2] Extracted text: {text_output}")
            
            # Try to extract and parse JSON
            try:
                json_start = text_output.find('{')
                json_end = text_output.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = text_output[json_start:json_end]
                    parsed = json.loads(json_text)
                    print(f"[Tier 2] SUCCESS: {parsed}")
                    return True
                else:
                    print(f"[Tier 2] No JSON found in response")
                    return False
            except json.JSONDecodeError as e:
                print(f"[Tier 2] JSON Parse Error: {e}")
                return False
        else:
            print(f"[Tier 2] ERROR: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"[Tier 2] Exception: {e}")
        return False

def test_tier3_markdown_extraction(text: str):
    """Test Tier 3 model (llama-3.2-1b-instruct) with Markdown output"""
    prompt = f"""Extract the key facts from this complex text.

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
    
    try:
        response = httpx.post(
            f"{LLM_BASE_URL}/chat/completions",  # Use chat completions for instruct model
            json={
                "model": TIER3_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.1
            },
            timeout=10
        )
        
        print(f"[Tier 3] Status: {response.status_code}")
        print(f"[Tier 3] Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            result = response.json()
            text_output = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"[Tier 3] Extracted text: {text_output}")
            print(f"[Tier 3] SUCCESS")
            return True
        else:
            print(f"[Tier 3] ERROR: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"[Tier 3] Exception: {e}")
        return False

def test_model_availability():
    """Test if models are loaded in LM Studio"""
    try:
        response = httpx.get(f"{LLM_BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = [model['id'] for model in models.get('data', [])]
            print(f"Available models: {available_models}")
            
            tier2_available = TIER2_MODEL in available_models
            tier3_available = TIER3_MODEL in available_models
            
            print(f"Tier 2 model ({TIER2_MODEL}) available: {tier2_available}")
            print(f"Tier 3 model ({TIER3_MODEL}) available: {tier3_available}")
            
            return tier2_available, tier3_available
        else:
            print(f"Failed to get models: {response.status_code}")
            return False, False
    except Exception as e:
        print(f"Error checking models: {e}")
        return False, False

def main():
    """Run all tests"""
    print("=== Testing LLM Integration for Tiered Extractor ===\n")
    
    # Test model availability
    print("1. Checking model availability...")
    tier2_ok, tier3_ok = test_model_availability()
    print()
    
    # Test sample text
    sample_text = "Alice works at Tesla and drives a Model 3. She reports to Bob who is the CTO."
    
    if tier2_ok:
        print("2. Testing Tier 2 (JSON extraction)...")
        test_tier2_json_extraction(sample_text)
        print()
    else:
        print("2. Skipping Tier 2 test - model not available")
    
    if tier3_ok:
        print("3. Testing Tier 3 (Markdown extraction)...")
        test_tier3_markdown_extraction(sample_text)
        print()
    else:
        print("3. Skipping Tier 3 test - model not available")
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    main()