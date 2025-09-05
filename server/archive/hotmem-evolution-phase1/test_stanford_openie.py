#!/usr/bin/env python3
"""
Test Stanford OpenIE for better triplet extraction quality
"""
import sys
import os
import time

# Set Java path
os.environ['PATH'] = "/opt/homebrew/opt/openjdk@11/bin:" + os.environ.get('PATH', '')

def test_stanford_openie():
    """Test Stanford OpenIE on our problematic sentences"""
    
    print("=== Testing Stanford OpenIE ===")
    
    try:
        from openie import StanfordOpenIE
        print("✅ Successfully imported Stanford OpenIE")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test sentences that our current system struggles with
    test_sentences = [
        "My favorite number is 77.",
        "My dog's name is Potola.",
        "I have a favorite color.",
        "So my favorite number is 77. Favorite food is pizza.",
        "Do you know my name?",
        "My name is Alex.",
        "I live in San Francisco.",
        "Caroline is a developer.",
        "The book is on the table."
    ]
    
    try:
        print("\n🚀 Initializing Stanford OpenIE client...")
        with StanfordOpenIE() as client:
            print("✅ Client initialized successfully!")
            
            total_time = 0
            for i, sentence in enumerate(test_sentences):
                print(f"\n--- Test {i+1}: '{sentence}' ---")
                
                start_time = time.perf_counter()
                try:
                    # Extract triples
                    triples = client.annotate(sentence)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    total_time += elapsed_ms
                    
                    print(f"⏱️  Processing time: {elapsed_ms:.1f}ms")
                    print(f"📊 Extracted {len(triples)} triples:")
                    
                    for triple in triples:
                        if isinstance(triple, dict):
                            subject = triple.get('subject', 'N/A')
                            relation = triple.get('relation', 'N/A') 
                            obj = triple.get('object', 'N/A')
                            confidence = triple.get('confidence', 'N/A')
                            print(f"   • ({subject}, {relation}, {obj}) [confidence: {confidence}]")
                        else:
                            print(f"   • {triple}")
                    
                    if not triples:
                        print("   • No triples extracted")
                        
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    total_time += elapsed_ms
                    print(f"❌ Error processing sentence: {e}")
                    print(f"⏱️  Time before error: {elapsed_ms:.1f}ms")
            
            avg_time = total_time / len(test_sentences)
            print(f"\n📈 Performance Summary:")
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Average per sentence: {avg_time:.1f}ms")
            print(f"   Target: <30ms (✅ {'PASS' if avg_time < 30 else 'FAIL'})")
            
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        print("   This might be due to Java path issues or missing dependencies")
    
    print("\n=== Stanford OpenIE Test Complete ===")


if __name__ == "__main__":
    test_stanford_openie()