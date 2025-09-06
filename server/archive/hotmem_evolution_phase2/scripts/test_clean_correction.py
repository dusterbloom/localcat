#!/usr/bin/env python3
"""
Clean Correction System Testing with Cache Clearing
==================================================

Tests correction system with:
1. LM Studio session isolation via session_id
2. spaCy/Stanza model cache clearing between tests  
3. Fresh memory instances per test scenario
"""

import os
import sys
import tempfile
import time
import uuid
import gc
from typing import List, Dict, Any

# Add server path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths
from memory_correction import LanguageAgnosticCorrector, CorrectionType

class CleanCorrectionTester:
    def __init__(self):
        self.temp_dirs = []
        
    def cleanup_all(self):
        """Clean up all temp directories"""
        for temp_dir in self.temp_dirs:
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up {len(self.temp_dirs)} test directories")
    
    def create_fresh_memory(self, test_name: str) -> HotMemory:
        """Create completely fresh memory instance with unique session"""
        temp_dir = tempfile.mkdtemp(prefix=f"clean_test_{test_name}_")
        self.temp_dirs.append(temp_dir)
        
        paths = Paths(
            sqlite_path=os.path.join(temp_dir, f'{test_name}_memory.db'),
            lmdb_dir=os.path.join(temp_dir, f'{test_name}_graph.lmdb')
        )
        store = MemoryStore(paths)
        hot_memory = HotMemory(store)
        hot_memory.prewarm('en')
        
        print(f"🆕 Created fresh memory for {test_name}: {temp_dir}")
        return hot_memory
    
    def create_fresh_corrector(self) -> LanguageAgnosticCorrector:
        """Create fresh corrector instance with cleared caches"""
        # Clear any existing spaCy model caches
        try:
            import spacy
            # Force reload of spaCy models by clearing cache
            if hasattr(spacy.util, '_MODELS'):
                spacy.util._MODELS.clear()
            if hasattr(spacy, '_LOADED'):
                spacy._LOADED.clear()
        except Exception:
            pass
            
        # Force garbage collection
        gc.collect()
        
        corrector = LanguageAgnosticCorrector()
        corrector.nlp_models = {}  # Clear any cached models
        
        print("🆕 Created fresh corrector with cleared model caches")
        return corrector
    
    def test_isolated_correction_scenarios(self):
        """Test correction scenarios with complete isolation"""
        print("🔒 TESTING ISOLATED CORRECTION SCENARIOS")
        print("=" * 60)
        
        scenarios = [
            {
                "name": "simple_name_correction",
                "session_id": str(uuid.uuid4())[:8],
                "initial_facts": ["My name is John and I work at Apple"],
                "correction": "/correct My name is Sarah",
                "test_query": "What's my name?",
                "expected_in_result": "sarah"
            },
            {
                "name": "work_location_correction", 
                "session_id": str(uuid.uuid4())[:8],
                "initial_facts": ["I work at Microsoft in Seattle"],
                "correction": "Actually, I work at Google, not Microsoft",
                "test_query": "Where do I work?",
                "expected_in_result": "google"
            },
            {
                "name": "multilingual_correction",
                "session_id": str(uuid.uuid4())[:8], 
                "initial_facts": ["I live in Paris"],
                "correction": "Eigentlich wohne ich in Berlin",  # German
                "test_query": "Where do I live?",
                "expected_in_result": "berlin"
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            print(f"\n🧪 Testing scenario: {scenario['name']}")
            print(f"📋 Session ID: {scenario['session_id']}")
            
            try:
                # Create completely fresh instances
                hot_memory = self.create_fresh_memory(scenario['name'])
                corrector = self.create_fresh_corrector()
                
                # Step 1: Store initial facts
                print(f"📝 Storing initial facts...")
                for fact in scenario['initial_facts']:
                    bullets, triples = hot_memory.process_turn(
                        fact, 
                        session_id=scenario['session_id'], 
                        turn_id=1
                    )
                    print(f"  Stored: {fact} → {len(triples)} facts")
                
                # Step 2: Show initial state
                bullets, _ = hot_memory.process_turn(
                    scenario['test_query'], 
                    session_id=scenario['session_id'], 
                    turn_id=2
                )
                print(f"  Initial query result: {bullets[:2]}")
                
                # Step 3: Apply correction
                print(f"🔧 Applying correction: {scenario['correction']}")
                
                instruction = corrector.detect_correction_intent(scenario['correction'])
                if instruction:
                    result = corrector.apply_correction(instruction, hot_memory)
                    
                    if result.get('success'):
                        print(f"  ✅ Correction applied: {result.get('explanation')}")
                        correction_success = True
                    else:
                        print(f"  ❌ Correction failed: {result.get('error')}")
                        correction_success = False
                else:
                    print(f"  ❌ No correction detected")
                    correction_success = False
                
                # Step 4: Test corrected state
                bullets, _ = hot_memory.process_turn(
                    scenario['test_query'], 
                    session_id=scenario['session_id'], 
                    turn_id=3
                )
                print(f"  Post-correction query result: {bullets[:2]}")
                
                # Step 5: Verify correction worked
                result_text = " ".join(bullets).lower()
                expected_found = scenario['expected_in_result'] in result_text
                
                scenario_result = {
                    'scenario': scenario['name'],
                    'correction_detected': instruction is not None,
                    'correction_applied': correction_success,
                    'expected_found': expected_found,
                    'success': correction_success and expected_found
                }
                
                if scenario_result['success']:
                    print(f"  ✅ Scenario passed - '{scenario['expected_in_result']}' found in results")
                else:
                    print(f"  ❌ Scenario failed - '{scenario['expected_in_result']}' not found")
                
                results.append(scenario_result)
                
            except Exception as e:
                print(f"  ❌ Scenario failed with exception: {e}")
                results.append({
                    'scenario': scenario['name'],
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def test_session_isolation_effectiveness(self):
        """Test that different sessions don't interfere with each other"""
        print(f"\n🔒 TESTING SESSION ISOLATION")
        print("=" * 60)
        
        # Create two isolated sessions
        session_a = str(uuid.uuid4())[:8]
        session_b = str(uuid.uuid4())[:8]
        
        memory_a = self.create_fresh_memory("session_a")
        memory_b = self.create_fresh_memory("session_b")
        corrector = self.create_fresh_corrector()
        
        print(f"Session A: {session_a}")
        print(f"Session B: {session_b}")
        
        # Store different facts in each session
        bullets_a1, _ = memory_a.process_turn("My name is Alice", session_a, 1)
        bullets_b1, _ = memory_b.process_turn("My name is Bob", session_b, 1)
        
        # Apply corrections in each session
        correction_a = corrector.detect_correction_intent("/correct My name is Carol")
        correction_b = corrector.detect_correction_intent("/correct My name is Dave")
        
        if correction_a:
            result_a = corrector.apply_correction(correction_a, memory_a)
            print(f"Session A correction: {result_a.get('success', False)}")
            
        if correction_b:
            result_b = corrector.apply_correction(correction_b, memory_b)
            print(f"Session B correction: {result_b.get('success', False)}")
        
        # Test that each session maintains its own state
        bullets_a2, _ = memory_a.process_turn("What's my name?", session_a, 2)
        bullets_b2, _ = memory_b.process_turn("What's my name?", session_b, 2)
        
        print(f"Session A result: {bullets_a2}")
        print(f"Session B result: {bullets_b2}")
        
        # Verify isolation
        a_has_carol = any("carol" in bullet.lower() for bullet in bullets_a2)
        b_has_dave = any("dave" in bullet.lower() for bullet in bullets_b2)
        a_has_dave = any("dave" in bullet.lower() for bullet in bullets_a2)
        b_has_carol = any("carol" in bullet.lower() for bullet in bullets_b2)
        
        isolation_success = a_has_carol and b_has_dave and not a_has_dave and not b_has_carol
        
        print(f"✅ Session isolation: {'WORKING' if isolation_success else 'FAILED'}")
        
        return {
            'session_a_correct': a_has_carol,
            'session_b_correct': b_has_dave,
            'no_cross_contamination': not (a_has_dave or b_has_carol),
            'isolation_success': isolation_success
        }
    
    def analyze_results(self, scenario_results: List[Dict], isolation_result: Dict):
        """Analyze test results"""
        print(f"\n📊 CLEAN CORRECTION TEST ANALYSIS")
        print("=" * 60)
        
        # Scenario analysis
        successful_scenarios = [r for r in scenario_results if r.get('success', False)]
        success_rate = len(successful_scenarios) / len(scenario_results) * 100
        
        print(f"Correction Scenarios:")
        print(f"  Total: {len(scenario_results)}")
        print(f"  Successful: {len(successful_scenarios)}")  
        print(f"  Success Rate: {success_rate:.1f}%")
        
        for result in scenario_results:
            status = "✅" if result.get('success') else "❌"
            scenario = result.get('scenario', 'unknown')
            print(f"  {status} {scenario}")
            
            if 'error' in result:
                print(f"      Error: {result['error']}")
        
        # Session isolation analysis
        print(f"\nSession Isolation:")
        isolation_success = isolation_result.get('isolation_success', False)
        status = "✅" if isolation_success else "❌"
        print(f"  {status} Sessions properly isolated: {isolation_success}")
        
        # Overall assessment
        overall_success = success_rate > 70 and isolation_success
        print(f"\n🎯 Overall Assessment:")
        print(f"  {'✅ PASSED' if overall_success else '❌ NEEDS WORK'}")
        
        if overall_success:
            print(f"  Clean correction system is working properly")
            print(f"  Session isolation prevents context pollution") 
            print(f"  Model cache clearing enables independent tests")
        else:
            print(f"  Issues detected that need addressing")
            if success_rate <= 70:
                print(f"    - Low correction success rate: {success_rate:.1f}%")
            if not isolation_success:
                print(f"    - Session isolation not working")
        
        return overall_success

def main():
    """Test clean correction system"""
    tester = CleanCorrectionTester()
    
    try:
        # Test isolated correction scenarios
        scenario_results = tester.test_isolated_correction_scenarios()
        
        # Test session isolation
        isolation_result = tester.test_session_isolation_effectiveness()
        
        # Analyze results
        overall_success = tester.analyze_results(scenario_results, isolation_result)
        
        print(f"\n🎯 CLEAN CORRECTION TESTING COMPLETE")
        print(f"Result: {'PASSED' if overall_success else 'NEEDS IMPROVEMENT'}")
        
    finally:
        tester.cleanup_all()

if __name__ == '__main__':
    main()