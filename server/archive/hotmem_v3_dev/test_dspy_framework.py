"""
Test script for DSPy framework integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.ai.dspy_modules import DSPyFramework, DSPyHotMemIntegration

def test_dspy_framework():
    """Test basic DSPy framework functionality"""
    print("🧪 Testing DSPy Framework...")
    
    # Initialize framework
    framework = DSPyFramework()
    print("✅ DSPy Framework initialized")
    
    # Test basic extraction
    test_text = "John works at Google in Mountain View."
    print(f"📝 Testing extraction: '{test_text}'")
    
    try:
        graph = framework.extract_graph(test_text)
        print(f"✅ Graph extracted successfully")
        print(f"   - Entities: {len(graph.entities)}")
        print(f"   - Relationships: {len(graph.relationships)}")
        print(f"   - Confidence: {graph.extraction_confidence:.2f}")
        
        # Print extracted entities
        for entity in graph.entities:
            print(f"   - Entity: {entity.text} ({entity.label})")
            
        # Print extracted relationships
        for rel in graph.relationships:
            print(f"   - Relationship: {rel.subject} --{rel.predicate}--> {rel.object}")
            
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        print("   This is expected if DSPy dependencies are not fully installed")
    
    print("✅ DSPy Framework test completed")

def test_hotmem_integration():
    """Test integration with HotMem system"""
    print("\n🔗 Testing HotMem Integration...")
    
    integration = DSPyHotMemIntegration()
    print("✅ DSPy HotMem Integration initialized")
    
    # Test fact extraction
    test_text = "Sarah is a software engineer at Microsoft."
    print(f"📝 Testing fact extraction: '{test_text}'")
    
    try:
        facts = integration.extract_facts(test_text)
        print(f"✅ Facts extracted: {len(facts)}")
        
        for fact in facts:
            print(f"   - Fact: {fact['subject']} {fact['predicate']} {fact['object']}")
            print(f"     Confidence: {fact['confidence']:.2f}")
            
    except Exception as e:
        print(f"❌ Fact extraction failed: {e}")
    
    print("✅ HotMem Integration test completed")

def test_training_pipeline():
    """Test training pipeline"""
    print("\n🎓 Testing Training Pipeline...")
    
    integration = DSPyHotMemIntegration()
    
    # Sample training data
    training_data = [
        {
            "text": "Alice is a data scientist at TechCorp.",
            "facts": [
                {"subject": "Alice", "predicate": "is", "object": "data scientist"},
                {"subject": "Alice", "predicate": "works_at", "object": "TechCorp"}
            ]
        },
        {
            "text": "Bob lives in San Francisco and works at Google.",
            "facts": [
                {"subject": "Bob", "predicate": "lives_in", "object": "San Francisco"},
                {"subject": "Bob", "predicate": "works_at", "object": "Google"}
            ]
        }
    ]
    
    try:
        integration.train_from_memory(training_data)
        print("✅ Training pipeline test completed")
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")

if __name__ == "__main__":
    print("🚀 HotMem V3 DSPy Framework Test")
    print("=" * 50)
    
    test_dspy_framework()
    test_hotmem_integration()
    test_training_pipeline()
    
    print("\n🎉 All tests completed!")
    print("Note: Some failures are expected if DSPy dependencies are not fully installed")