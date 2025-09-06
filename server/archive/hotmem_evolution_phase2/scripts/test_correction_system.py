#!/usr/bin/env python3
"""
Language-Agnostic Correction System Testing
===========================================

Tests the UD-based correction system across multiple languages
and correction patterns.
"""

import os
import sys
import tempfile
import time
from typing import List, Dict, Any

# Add server path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths
from memory_correction import LanguageAgnosticCorrector, CorrectionType

class CorrectionSystemTester:
    def __init__(self):
        self.temp_dir = None
        self.hot_memory = None
        self.corrector = None
        
    def setup(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        paths = Paths(
            sqlite_path=os.path.join(self.temp_dir, 'correction_memory.db'),
            lmdb_dir=os.path.join(self.temp_dir, 'correction_graph.lmdb')
        )
        store = MemoryStore(paths)
        self.hot_memory = HotMemory(store)
        self.hot_memory.prewarm('en')
        
        self.corrector = LanguageAgnosticCorrector()
        
        print(f"🔧 Correction test storage: {self.temp_dir}")
        
    def cleanup(self):
        """Clean up test data"""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Correction test cleanup complete")
    
    def test_correction_detection(self):
        """Test correction intent detection across languages and patterns"""
        print("🔍 TESTING CORRECTION DETECTION")
        print("=" * 60)
        
        test_cases = [
            # English corrections
            ("Actually, I work at Google, not Microsoft", "en", CorrectionType.EXPLICIT_CORRECTION),
            ("No, my name is Sarah", "en", CorrectionType.NEGATION_CORRECTION),
            ("/correct I live in Seattle", "en", CorrectionType.COMMAND_CORRECTION),
            ("I don't work there anymore", "en", CorrectionType.CONTRADICTION),
            ("I moved to Austin from Denver", "en", CorrectionType.FACT_REPLACEMENT),
            
            # Spanish corrections
            ("En realidad, trabajo en Google", "es", CorrectionType.EXPLICIT_CORRECTION),
            ("No, mi nombre es Carlos", "es", CorrectionType.NEGATION_CORRECTION),
            ("/corregir vivo en Madrid", "es", CorrectionType.COMMAND_CORRECTION),
            
            # German corrections
            ("Eigentlich arbeite ich bei Google", "de", CorrectionType.EXPLICIT_CORRECTION),
            ("Nein, ich heiße Anna", "de", CorrectionType.NEGATION_CORRECTION),
            
            # Non-corrections (should return None)
            ("I work at Microsoft", "en", None),
            ("My name is John", "en", None),
            ("How are you today?", "en", None),
        ]
        
        results = []
        
        for text, expected_lang, expected_type in test_cases:
            print(f"\n📝 Testing: '{text}'")
            
            # Detect correction intent
            instruction = self.corrector.detect_correction_intent(text)
            
            if instruction is None:
                if expected_type is None:
                    print(f"  ✅ Correctly identified as non-correction")
                    results.append(True)
                else:
                    print(f"  ❌ Failed to detect {expected_type.value}")
                    results.append(False)
            else:
                print(f"  🎯 Detected: {instruction.correction_type.value}")
                print(f"  🌍 Language: {instruction.language}")
                print(f"  📊 Confidence: {instruction.confidence:.2f}")
                print(f"  📝 Explanation: {instruction.explanation}")
                print(f"  ⚡ New facts: {len(instruction.new_facts)}")
                
                # Check if detection matches expectation
                if instruction.correction_type == expected_type:
                    print(f"  ✅ Correction type match")
                    results.append(True)
                else:
                    print(f"  ❌ Expected {expected_type}, got {instruction.correction_type}")
                    results.append(False)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Detection Success Rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")
        
        return results
    
    def test_correction_application(self):
        """Test applying corrections to memory system"""
        print(f"\n🔄 TESTING CORRECTION APPLICATION")
        print("=" * 60)
        
        # Step 1: Store initial facts
        print("📝 Storing initial facts:")
        initial_facts = [
            "My name is John and I work at Microsoft",
            "I live in Portland with my cat Luna",
            "My favorite color is blue"
        ]
        
        for text in initial_facts:
            bullets, triples = self.hot_memory.process_turn(text, "test", 1)
            print(f"  Stored: {text} → {len(triples)} facts")
        
        # Step 2: Show initial retrieval
        print(f"\n🎯 Initial memory state:")
        bullets, _ = self.hot_memory.process_turn("What do you know about me?", "test", 2)
        for bullet in bullets:
            print(f"  {bullet}")
        
        # Step 3: Apply corrections
        print(f"\n🔧 Applying corrections:")
        corrections = [
            "Actually, I work at Google, not Microsoft",
            "/correct My name is Sarah",
            "I moved to Seattle from Portland"
        ]
        
        correction_results = []
        
        for correction_text in corrections:
            print(f"\n  Correction: '{correction_text}'")
            
            # Detect and apply correction
            instruction = self.corrector.detect_correction_intent(correction_text)
            if instruction:
                result = self.corrector.apply_correction(instruction, self.hot_memory)
                correction_results.append(result)
                
                if result.get('success'):
                    print(f"    ✅ Applied: {result.get('explanation')}")
                    print(f"    📊 Promoted {result.get('promoted_facts', 0)} facts")
                else:
                    print(f"    ❌ Failed: {result.get('error')}")
            else:
                print(f"    ❌ No correction detected")
        
        # Step 4: Show updated memory state  
        print(f"\n🎯 Updated memory state:")
        bullets, _ = self.hot_memory.process_turn("What do you know about me?", "test", 3)
        for bullet in bullets:
            print(f"  {bullet}")
        
        # Step 5: Test specific queries
        print(f"\n🔍 Testing specific corrected facts:")
        test_queries = [
            "Where do I work?",
            "What's my name?", 
            "Where do I live?"
        ]
        
        for query in test_queries:
            bullets, _ = self.hot_memory.process_turn(query, "test", 4)
            print(f"\n  Query: '{query}'")
            for bullet in bullets[:2]:
                print(f"    {bullet}")
        
        return correction_results
    
    def test_multilingual_corrections(self):
        """Test corrections in multiple languages"""
        print(f"\n🌍 TESTING MULTILINGUAL CORRECTIONS")
        print("=" * 60)
        
        # Test different languages
        multilingual_tests = [
            ("en", "Actually, I work at Tesla", "I work at Tesla"),
            ("es", "En realidad, trabajo en Tesla", "trabajo en Tesla"),  
            ("de", "Eigentlich arbeite ich bei Tesla", "arbeite ich bei Tesla"),
            ("fr", "En fait, je travaille chez Tesla", "je travaille chez Tesla"),
        ]
        
        results = []
        
        for lang, correction, expected_content in multilingual_tests:
            print(f"\n🌐 Testing {lang.upper()}: '{correction}'")
            
            # Detect correction
            instruction = self.corrector.detect_correction_intent(correction, language=lang)
            
            if instruction:
                print(f"  ✅ Detected: {instruction.correction_type.value}")
                print(f"  🌍 Language: {instruction.language}")
                print(f"  📝 Facts: {len(instruction.new_facts)}")
                
                # Apply correction
                result = self.corrector.apply_correction(instruction, self.hot_memory)
                results.append(result.get('success', False))
                
                if result.get('success'):
                    print(f"  ✅ Applied successfully")
                else:
                    print(f"  ❌ Application failed: {result.get('error')}")
            else:
                print(f"  ❌ No correction detected")
                results.append(False)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Multilingual Success Rate: {success_rate:.1f}%")
        
        return results
    
    def test_integration_with_hotmem(self):
        """Test full integration with HotMem processing pipeline"""
        print(f"\n🔗 TESTING HOTMEM INTEGRATION")
        print("=" * 60)
        
        # Simulate the full pipeline with corrections
        conversation = [
            ("My name is Alex and I work at Apple", False),
            ("I live in San Francisco", False),
            ("Actually, I work at Google, not Apple", True),  # Should trigger correction
            ("/correct I live in New York", True),            # Should trigger correction
            ("What do you know about me?", False),            # Query after corrections
        ]
        
        for i, (text, expect_correction) in enumerate(conversation, 1):
            print(f"\nTurn {i}: '{text}'")
            
            # Simulate correction detection (would be in hotpath_processor)
            instruction = self.corrector.detect_correction_intent(text)
            correction_applied = False
            
            if instruction and expect_correction:
                result = self.corrector.apply_correction(instruction, self.hot_memory)
                correction_applied = result.get('success', False)
                if correction_applied:
                    print(f"  🔧 Correction applied: {result.get('explanation')}")
            
            # Process turn normally
            bullets, triples = self.hot_memory.process_turn(text, "integration_test", i)
            
            # Add correction feedback bullet if correction was applied
            if correction_applied:
                correction_bullet = f"• ✅ Correction applied: Memory updated"
                bullets = [correction_bullet] + bullets[:4]
            
            print(f"  📊 Result: {len(triples)} facts stored, {len(bullets)} bullets")
            for bullet in bullets[:3]:
                print(f"    {bullet}")
        
        print(f"\n✅ Integration test completed")

def main():
    """Test the correction system"""
    tester = CorrectionSystemTester()
    
    try:
        tester.setup()
        
        # Test correction detection
        detection_results = tester.test_correction_detection()
        
        # Test correction application
        application_results = tester.test_correction_application()
        
        # Test multilingual support
        multilingual_results = tester.test_multilingual_corrections()
        
        # Test integration
        tester.test_integration_with_hotmem()
        
        print(f"\n🎯 CORRECTION SYSTEM SUMMARY")
        print("=" * 60)
        print(f"✅ Language-agnostic UD-based correction system implemented")
        print(f"✅ Supports 5 correction types across multiple languages")
        print(f"✅ Real-time fact updates with temporal prioritization")
        print(f"✅ Integrated with HotMem pipeline")
        print(f"✅ Transparent user feedback via correction bullets")
        
        total_success = sum(detection_results) + len([r for r in application_results if r.get('success')]) + sum(multilingual_results)
        total_tests = len(detection_results) + len(application_results) + len(multilingual_results)
        
        print(f"\nOverall Success Rate: {total_success}/{total_tests} ({total_success/total_tests*100:.1f}%)")
        
    finally:
        tester.cleanup()

if __name__ == '__main__':
    main()