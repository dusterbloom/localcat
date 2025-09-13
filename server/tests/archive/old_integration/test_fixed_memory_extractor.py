#!/usr/bin/env python3
"""
Test the fixed MemoryExtractor with fast ReLiK (no retriever)
"""

import time
from components.extraction.memory_extractor import MemoryExtractor

def test_fixed_memory_extractor():
    """Test MemoryExtractor with the ReLiK fix"""
    print("🧪 TESTING FIXED MEMORY EXTRACTOR")
    print("=" * 60)
    
    # Test configuration with ReLiK enabled
    config = {
        'use_srl': False,
        'use_onnx_ner': False,
        'use_onnx_srl': False,
        'use_relik': True,  # Enable ReLiK with the fix
        'use_gliner': True,
        'use_coref': False,
        'llm_base_url': 'http://127.0.0.1:1234/v1'
    }
    
    # Test text
    test_text = """
    Dr. Sarah Chen is the AI research director at OpenAI. She joined the company in 2021 
    after completing her PhD at Stanford under the supervision of Dr. Michael Jordan. 
    The researcher previously worked at Google Brain where she collaborated with Dr. Fei-Fei Li. 
    They developed several groundbreaking papers on transformer architectures. 
    The company was founded in 2015 and currently has 500 employees.
    """
    
    print(f"📝 Test text length: {len(test_text)} characters")
    print()
    
    # Test extraction (this should now be fast)
    print("🔄 Testing extraction with fast ReLiK...")
    start_time = time.perf_counter()
    
    try:
        extractor = MemoryExtractor(config)
        result = extractor.extract(test_text, use_cache=False)
        elapsed_time = (time.perf_counter() - start_time) * 1000
        
        print(f"✅ Extraction completed in {elapsed_time:.1f}ms")
        print(f"   Entities found: {len(result.entities)}")
        print(f"   Relations found: {len(result.triples)}")
        
        # Show some results
        print("\n📋 Sample entities:")
        for entity in list(result.entities)[:5]:
            print(f"   - {entity}")
        
        print("\n📋 Sample relations:")
        for i, triple in enumerate(result.triples[:5]):
            print(f"   {i+1}. {triple}")
        
        # Performance assessment
        if elapsed_time < 1000:
            print(f"\n🎉 EXCELLENT: Under 1 second! ({elapsed_time:.1f}ms)")
        elif elapsed_time < 3000:
            print(f"\n👍 GOOD: Acceptable performance ({elapsed_time:.1f}ms)")
        else:
            print(f"\n⚠️  SLOW: Still taking too long ({elapsed_time:.1f}ms)")
        
        return result
        
    except Exception as e:
        elapsed_time = (time.perf_counter() - start_time) * 1000
        print(f"❌ Extraction failed: {e}")
        print(f"   Failed after: {elapsed_time:.1f}ms")
        import traceback
        traceback.print_exc()
        return None

def test_caching():
    """Test that caching works properly with the fast ReLiK"""
    print("\n" + "=" * 60)
    print("🧪 TESTING CACHING WITH FAST RELIK")
    print("=" * 60)
    
    config = {
        'use_srl': False,
        'use_onnx_ner': False,
        'use_onnx_srl': False,
        'use_relik': True,
        'use_gliner': True,
        'use_coref': False,
        'llm_base_url': 'http://127.0.0.1:1234/v1'
    }
    
    test_text = "Steve Jobs founded Apple Inc. in Cupertino, California."
    
    print("🔄 Testing cold start...")
    extractor = MemoryExtractor(config)
    
    start_time = time.perf_counter()
    result1 = extractor.extract(test_text, use_cache=False)
    cold_time = (time.perf_counter() - start_time) * 1000
    
    print("🔄 Testing cached extraction...")
    start_time = time.perf_counter()
    result2 = extractor.extract(test_text, use_cache=True)
    warm_time = (time.perf_counter() - start_time) * 1000
    
    print(f"\n📊 Performance Summary:")
    print(f"   Cold start: {cold_time:.1f}ms")
    print(f"   Cached: {warm_time:.1f}ms")
    print(f"   Speedup: {cold_time/warm_time:.1f}x")
    
    if warm_time < 50:
        print("🎉 Caching is working perfectly!")
    
    return result1, result2

if __name__ == "__main__":
    result = test_fixed_memory_extractor()
    test_caching()