#!/usr/bin/env python3
"""
Focused Tier 3 debugging - test markdown output directly
The beast model should be producing great relations, not 1 per 22 entities
"""

import json
import time
from typing import List, Dict, Any
from components.extraction.tiered_extractor import TieredRelationExtractor

class Tier3Debugger:
    """Debug Tier 3 markdown output issues"""
    
    def __init__(self):
        self.extractor = TieredRelationExtractor()
        
    def test_direct_tier3_call(self):
        """Test Tier 3 LLM call directly"""
        
        # Test cases that should produce rich relations
        test_cases = [
            {
                "name": "Simple Test",
                "text": "Alice works at Tesla as an engineer. Bob is her manager.",
                "expected_relations": 3
            },
            {
                "name": "Medium Test", 
                "text": "Dr. Sarah Chen joined OpenAI in 2021 after completing her PhD at Stanford under Dr. Michael Jordan.",
                "expected_relations": 4
            },
            {
                "name": "Complex Test",
                "text": "Microsoft, founded by Bill Gates and Paul Allen in 1975, acquired LinkedIn for $26.2 billion in 2016. Satya Nadella became CEO in 2014.",
                "expected_relations": 6
            }
        ]
        
        print("🔍 TIER 3 DEBUGGING - Direct LLM Calls")
        print("=" * 60)
        
        for test_case in test_cases:
            print(f"\n📝 {test_case['name']}")
            print(f"Text: {test_case['text']}")
            print(f"Expected relations: {test_case['expected_relations']}")
            print("-" * 50)
            
            # Get entities first (simulate real flow)
            entities_result = self.extractor._extract_tier1(test_case['text'])
            entities = [str(ent) for ent in entities_result.entities]
            print(f"📝 Entities found: {entities}")
            
            # Test Tier 3 prompt directly
            try:
                system_prompt = self.extractor._build_tier3_system_prompt()
                user_prompt = self.extractor._build_tier3_user_prompt(test_case['text'], entities)
                
                print(f"\n🤖 SYSTEM PROMPT:")
                print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
                
                print(f"\n👤 USER PROMPT:")
                print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)
                
                # Make the LLM call
                start_time = time.perf_counter()
                raw_response = self.extractor._call_llm_tier3(system_prompt, user_prompt)
                call_time = (time.perf_counter() - start_time) * 1000
                
                print(f"\n⏱️  LLM call time: {call_time:.1f}ms")
                print(f"📄 RAW RESPONSE:")
                print("-" * 30)
                print(raw_response)
                print("-" * 30)
                
                # Try to parse as markdown
                if raw_response:
                    try:
                        relationships = self.extractor._parse_tier3_markdown(raw_response)
                        print(f"\n✅ PARSED RELATIONSHIPS: {len(relationships)}")
                        for i, rel in enumerate(relationships):
                            print(f"   {i+1}. {rel}")
                    except Exception as e:
                        print(f"\n❌ MARKDOWN PARSING ERROR: {e}")
                        
                        # Try to see if it's valid JSON
                        try:
                            json_data = json.loads(raw_response)
                            print(f"📋 Found JSON instead: {json_data}")
                        except:
                            print("📋 Not JSON either")
                else:
                    print("\n❌ NO RESPONSE FROM LLM")
                
                print(f"\n🎯 EXPECTED: {test_case['expected_relations']} relations")
                print(f"📊 GAP: {test_case['expected_relations'] - (len(relationships) if 'relationships' in locals() else 0)}")
                    
            except Exception as e:
                print(f"\n💥 ERROR: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n" + "=" * 80)
    
    def test_entity_formatting(self):
        """Test how entities are being formatted for Tier 3"""
        
        print("\n🔍 ENTITY FORMATTING TEST")
        print("=" * 50)
        
        test_text = "Alice works at Tesla as an engineer. Bob is her manager."
        entities_result = self.extractor._extract_tier1(test_text)
        entities = [str(ent) for ent in entities_result.entities]
        
        print(f"Raw entities: {entities_result.entities}")
        print(f"String entities: {entities}")
        
        # Test the user prompt building
        user_prompt = self.extractor._build_tier3_user_prompt(test_text, entities)
        print(f"\nUSER PROMPT:\n{user_prompt}")
    
    def test_markdown_parsing(self):
        """Test markdown parsing with known good examples"""
        
        print("\n🔍 MARKDOWN PARSING TEST")
        print("=" * 50)
        
        # Test cases that should parse correctly
        test_markdowns = [
            # Simple markdown
            """## Relationships
1. **Alice** works at **Tesla**
2. **Bob** is manager of **Alice**""",
            
            # JSON format
            """{"relationships": [
    {"subject": "Alice", "relation": "works at", "object": "Tesla"},
    {"subject": "Bob", "relation": "manages", "object": "Alice"}
]}""",
            
            # Plain text
            """Alice works at Tesla.
Bob manages Alice.""",
            
            # Malformed
            """Some random text that isn't structured"""
        ]
        
        for i, markdown in enumerate(test_markdowns):
            print(f"\nTest {i+1}:")
            print(f"Input: {markdown[:100]}...")
            try:
                result = self.extractor._parse_tier3_markdown(markdown)
                print(f"✅ Result: {result}")
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    debugger = Tier3Debugger()
    
    # Test entity formatting first
    debugger.test_entity_formatting()
    
    # Test markdown parsing
    debugger.test_markdown_parsing()
    
    # Test direct Tier 3 calls
    debugger.test_direct_tier3_call()