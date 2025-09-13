#!/usr/bin/env python3
"""
Test ReLiK Small/Tiny models for fast relation extraction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time

def test_relik_models():
    """Test ReLiK Small and Tiny models"""
    print("🚀 TESTING RELIK SMALL/TINY MODELS")
    print("=" * 50)
    
    # Test text
    test_text = """
    Dr. Sarah Chen is the AI research director at OpenAI. She joined the company in 2021 
    after completing her PhD at Stanford under the supervision of Dr. Michael Jordan. 
    The researcher previously worked at Google Brain where she collaborated with Dr. Fei-Fei Li. 
    They developed several groundbreaking papers on transformer architectures. 
    The company was founded in 2015 and currently has 500 employees.
    """
    
    print(f"📝 Test text: {test_text.strip()}")
    
    # Test models
    models_to_test = [
        ("relik-ie/relik-relation-extraction-small", "Small (~110M params)"),
        ("relik-ie/relik-cie-tiny", "Tiny (~50M params)")
    ]
    
    for model_name, model_desc in models_to_test:
        print(f"\n🔬 Testing {model_desc}")
        print("-" * 40)
        
        try:
            # Import and load model
            from relik import Relik
            
            print(f"⏳ Loading {model_name}...")
            start_load = time.perf_counter()
            
            relik = Relik.from_pretrained(model_name)
            load_time = (time.perf_counter() - start_load) * 1000
            
            print(f"✅ Model loaded in {load_time:.1f}ms")
            
            # Test inference
            print(f"🔄 Testing inference...")
            start_infer = time.perf_counter()
            
            result = relik(test_text)
            infer_time = (time.perf_counter() - start_infer) * 1000
            
            print(f"✅ Inference completed in {infer_time:.1f}ms")
            
            # Extract and display results
            print(f"📊 Results:")
            if hasattr(result, 'triplets'):
                print(f"   Relations found: {len(result.triplets)}")
                for i, triplet in enumerate(result.triplets[:5], 1):  # Show first 5
                    print(f"   {i}. {triplet}")
            elif hasattr(result, 'relations'):
                print(f"   Relations found: {len(result.relations)}")
                for i, rel in enumerate(result.relations[:5], 1):  # Show first 5
                    print(f"   {i}. {rel}")
            else:
                print(f"   Raw result type: {type(result)}")
                print(f"   Result: {result}")
            
            # Test with entities (faster mode)
            print(f"\n⚡ Testing with pre-extracted entities...")
            entities = ["Dr. Sarah Chen", "OpenAI", "Stanford", "Dr. Michael Jordan", "Google Brain", "Dr. Fei-Fei Li"]
            
            start_entities = time.perf_counter()
            result_entities = relik(test_text, entities=entities)
            entities_time = (time.perf_counter() - start_entities) * 1000
            
            print(f"✅ Entity-based inference in {entities_time:.1f}ms")
            
            if hasattr(result_entities, 'triplets'):
                print(f"   Relations with entities: {len(result_entities.triplets)}")
                for i, triplet in enumerate(result_entities.triplets[:3], 1):
                    print(f"   {i}. {triplet}")
            
            print(f"📈 Performance Summary:")
            print(f"   Load time: {load_time:.1f}ms")
            print(f"   Inference time: {infer_time:.1f}ms")
            print(f"   Entity-based time: {entities_time:.1f}ms")
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 ReLiK testing completed!")

if __name__ == "__main__":
    test_relik_models()