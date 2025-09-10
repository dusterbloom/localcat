"""
HotMem V3 + LM Studio Integration Test
Real-time testing of relation-extractor-v2-mlx model
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any, List, Optional
import logging

# Import HotMem V3 components
from components.hotmem_v3.core.hotmem_v3 import HotMemV3, HotMemV3Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LMStudioIntegration:
    """Integration with LM Studio for relation extraction"""
    
    def __init__(self, model_name: str = "relation-extractor-v2-mlx", 
                 api_base: str = "http://localhost:1234"):
        self.model_name = model_name
        self.api_base = api_base
        self.api_url = f"{api_base}/v1/chat/completions"
        
        # Test connection
        self.test_connection()
    
    def test_connection(self):
        """Test connection to LM Studio"""
        try:
            response = requests.get(f"{self.api_base}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"✅ Connected to LM Studio. Available models: {len(models.get('data', []))}")
                
                # Check if our model is available
                available_models = [m['id'] for m in models.get('data', [])]
                if self.model_name in available_models:
                    logger.info(f"✅ Found target model: {self.model_name}")
                else:
                    logger.warning(f"⚠️ Model {self.model_name} not found. Available: {available_models[:5]}")
            else:
                logger.error(f"❌ LM Studio connection failed: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to LM Studio: {e}")
            logger.info("💡 Make sure LM Studio is running and the model is loaded")
    
    def extract_relations(self, text: str, temperature: float = 0.1) -> Dict[str, Any]:
        """Extract relations using LM Studio model"""
        
        system_prompt = """You are an expert relation extraction system. Extract entities and relations from the given text and output them in JSON format.

Entities should include people, organizations, locations, and other important nouns with their types.
Relations should connect entities with meaningful predicates like 'works_for', 'located_in', 'develops', 'friends_with', etc.

Output format:
{
  "entities": [
    {"text": "Entity Name", "type": "PERSON/ORG/LOC/PRODUCT", "confidence": 0.9}
  ],
  "relations": [
    {"subject": "Entity1", "predicate": "relation_type", "object": "Entity2", "confidence": 0.8}
  ],
  "confidence": 0.85
}

Only output the JSON, no other text."""

        user_prompt = f"Extract entities and relations from this text:\n\n{text}"
        
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": 512,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Extract JSON from response
                try:
                    # Find JSON in the response
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start != -1 and json_end != -1:
                        json_str = content[json_start:json_end]
                        extraction_result = json.loads(json_str)
                        
                        # Ensure required fields
                        extraction_result.setdefault('entities', [])
                        extraction_result.setdefault('relations', [])
                        extraction_result.setdefault('confidence', 0.5)
                        
                        return extraction_result
                    else:
                        logger.warning(f"No JSON found in response: {content[:200]}...")
                        return {"entities": [], "relations": [], "confidence": 0.0}
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    logger.error(f"Raw content: {content}")
                    return {"entities": [], "relations": [], "confidence": 0.0}
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"entities": [], "relations": [], "confidence": 0.0}
                
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {"entities": [], "relations": [], "confidence": 0.0}

class HotMemLMStudioTest:
    """Test HotMem V3 with LM Studio integration"""
    
    def __init__(self):
        self.lm_studio = LMStudioIntegration()
        self.hotmem = None
        
        # Initialize HotMem V3
        self.initialize_hotmem()
    
    def initialize_hotmem(self):
        """Initialize HotMem V3 with LM Studio integration"""
        try:
            config = HotMemV3Config(
                model_path=None,  # Using LM Studio instead
                enable_real_time=True,
                confidence_threshold=0.7,
                enable_streaming=True
            )
            
            self.hotmem = HotMemV3(config)
            
            # Override the inference to use LM Studio
            self.hotmem.lm_studio_integration = self.lm_studio
            
            logger.info("✅ HotMem V3 initialized with LM Studio integration")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize HotMem V3: {e}")
    
    def test_basic_extraction(self):
        """Test basic relation extraction"""
        print("\n" + "="*60)
        print("🧪 BASIC RELATION EXTRACTION TEST")
        print("="*60)
        
        test_cases = [
            "I work at Google as a software engineer in Mountain View.",
            "Sarah is friends with John and they both work at Microsoft.",
            "Apple develops the iPhone and is headquartered in Cupertino.",
            "My name is Emma and I'm a data scientist at Amazon in Seattle.",
            "Tesla was founded by Elon Musk and manufactures electric vehicles.",
            "I met my colleague Michael at the Starbucks in downtown San Francisco.",
            "Netflix produces original content and is based in Los Gatos, California."
        ]
        
        results = []
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {text}")
            
            start_time = time.time()
            result = self.lm_studio.extract_relations(text)
            processing_time = time.time() - start_time
            
            print(f"⏱️  Processing time: {processing_time:.3f}s")
            print(f"🎯 Confidence: {result.get('confidence', 0):.2f}")
            print(f"👥 Entities: {len(result.get('entities', []))}")
            print(f"🔗 Relations: {len(result.get('relations', []))}")
            
            if result.get('entities'):
                print("   Entities:")
                for entity in result['entities']:
                    print(f"     - {entity.get('text', 'N/A')} ({entity.get('type', 'N/A')})")
            
            if result.get('relations'):
                print("   Relations:")
                for rel in result['relations']:
                    print(f"     - {rel.get('subject', 'N/A')} --{rel.get('predicate', 'N/A')}--> {rel.get('object', 'N/A')}")
            
            results.append({
                'text': text,
                'result': result,
                'processing_time': processing_time
            })
        
        return results
    
    def test_conversation_simulation(self):
        """Test simulated conversation flow"""
        print("\n" + "="*60)
        print("💬 CONVERSATION SIMULATION TEST")
        print("="*60)
        
        conversation = [
            "Hi, I'm Alex and I work at Spotify as a product manager.",
            "That's cool! I'm Sarah and I'm a software engineer at Google.",
            "Nice to meet you! My friend Michael also works at Google in the Mountain View office.",
            "Oh really? I know Michael! We met at the tech conference in San Francisco last year.",
            "Small world! Google is such a great company to work for.",
            "Yeah, I love it here. Spotify is also amazing, especially the work culture in Stockholm."
        ]
        
        print("Simulating conversation flow...")
        
        for i, turn in enumerate(conversation, 1):
            print(f"\n🗣️  Turn {i}: {turn}")
            
            result = self.lm_studio.extract_relations(turn)
            
            print(f"   Extracted: {len(result.get('entities', []))} entities, {len(result.get('relations', []))} relations")
            
            # Show significant extractions
            if result.get('relations'):
                for rel in result['relations']:
                    confidence = rel.get('confidence', 0)
                    if confidence > 0.7:
                        print(f"   ✅ High-confidence relation: {rel.get('subject')} --{rel.get('predicate')}--> {rel.get('object')} ({confidence:.2f})")
            
            # Small delay to simulate real-time processing
            time.sleep(0.5)
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n" + "="*60)
        print("🛡️  ERROR HANDLING TEST")
        print("="*60)
        
        edge_cases = [
            "",  # Empty text
            "Hello",  # Too simple
            "The quick brown fox jumps over the lazy dog.",  # No entities
            "1234567890!@#$%^&*()",  # No meaningful content
            "a" * 1000,  # Very long repetitive text
            "I work at",  # Incomplete sentence
        ]
        
        for i, text in enumerate(edge_cases, 1):
            print(f"\n🧪 Edge case {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            try:
                result = self.lm_studio.extract_relations(text)
                print(f"   Result: {len(result.get('entities', []))} entities, {len(result.get('relations', []))} relations")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    def test_performance_benchmark(self):
        """Test performance benchmarks"""
        print("\n" + "="*60)
        print("⚡ PERFORMANCE BENCHMARK")
        print("="*60)
        
        # Test with varying text lengths
        test_texts = [
            "Short text.",
            "This is a medium length text with some entities like Google and Apple.",
            "This is a longer text that contains multiple entities and relations. For example, I work at Microsoft as a software engineer in the Seattle office. My colleague Sarah works in the Redmond campus. We both report to our manager Michael who is based in the main headquarters. Microsoft develops Windows and Office products, and is headquartered in Redmond, Washington.",
        ]
        
        print("Testing different text lengths...")
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📏 Length test {i} ({len(text)} chars)")
            
            # Run multiple times for average
            times = []
            for _ in range(5):
                start_time = time.time()
                result = self.lm_studio.extract_relations(text)
                processing_time = time.time() - start_time
                times.append(processing_time)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"   Average time: {avg_time:.3f}s")
            print(f"   Min time: {min_time:.3f}s")
            print(f"   Max time: {max_time:.3f}s")
            print(f"   Entities: {len(result.get('entities', []))}")
            print(f"   Relations: {len(result.get('relations', []))}")
    
    def test_hotmem_integration(self):
        """Test integration with HotMem V3 pipeline"""
        print("\n" + "="*60)
        print("🔗 HOTMEM V3 INTEGRATION TEST")
        print("="*60)
        
        if not self.hotmem:
            print("❌ HotMem V3 not initialized")
            return
        
        # Test text processing through HotMem V3
        test_text = "I work at Apple as a designer in the Cupertino office."
        
        print(f"📝 Processing: {test_text}")
        
        try:
            # Use LM Studio integration
            result = self.hotmem.lm_studio_integration.extract_relations(test_text)
            
            print(f"✅ Extraction successful")
            print(f"   Entities: {len(result.get('entities', []))}")
            print(f"   Relations: {len(result.get('relations', []))}")
            
            # Get knowledge graph
            graph = self.hotmem.get_knowledge_graph()
            print(f"📊 Knowledge graph: {len(graph.get('entities', []))} entities, {len(graph.get('relations', []))} relations")
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 STARTING HOTMEM V3 + LM STUDIO TESTS")
        print("="*60)
        
        # Test connection first
        print("🔌 Testing LM Studio connection...")
        try:
            response = requests.get(f"{self.lm_studio.api_base}/v1/models", timeout=5)
            if response.status_code == 200:
                print("✅ LM Studio connection successful")
            else:
                print("❌ LM Studio connection failed")
                return
        except:
            print("❌ Cannot connect to LM Studio. Please check if it's running.")
            return
        
        # Run all test suites
        try:
            self.test_basic_extraction()
            self.test_conversation_simulation()
            self.test_error_handling()
            self.test_performance_benchmark()
            self.test_hotmem_integration()
            
            print("\n" + "="*60)
            print("🎉 ALL TESTS COMPLETED!")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            logger.exception("Test suite error")

def main():
    """Main test function"""
    tester = HotMemLMStudioTest()
    tester.run_all_tests()

if __name__ == "__main__":
    main()