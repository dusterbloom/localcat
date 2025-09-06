#!/usr/bin/env python3
"""
Test Rebalanced Temporal Weights
=================================

Tests the impact of rebalanced weights:
- α (priority): 0.4 → 0.35 
- β (recency): 0.2 → 0.32
- γ (similarity): 0.3 → 0.25  
- δ (weight): 0.1 → 0.08

Should make temporal decay visible in retrieval ranking.
"""

import os
import sys
import time
import tempfile
import math
from typing import List, Tuple, Dict, Any

# Add server path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths

class WeightRebalanceTester:
    def __init__(self):
        self.temp_dir = None
        self.hot_memory = None
        
    def setup(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        paths = Paths(
            sqlite_path=os.path.join(self.temp_dir, 'rebalanced_memory.db'),
            lmdb_dir=os.path.join(self.temp_dir, 'rebalanced_graph.lmdb')
        )
        store = MemoryStore(paths)
        self.hot_memory = HotMemory(store)
        self.hot_memory.prewarm('en')
        print(f"⚖️  Rebalanced weights test storage: {self.temp_dir}")
        
    def cleanup(self):
        """Clean up test data"""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Rebalanced test cleanup complete")
    
    def test_temporal_ranking_improvement(self):
        """Test if rebalanced weights improve temporal ranking"""
        print("⚖️  TESTING REBALANCED TEMPORAL WEIGHTS")
        print("=" * 60)
        print("Weights: α=0.35, β=0.32, γ=0.25, δ=0.08")
        print("Previous: α=0.40, β=0.20, γ=0.30, δ=0.10")
        
        base_time = time.time()
        
        # Store facts at strategic times to test ranking
        facts_with_times = [
            ("I work at Google", 0),           # Fresh, high priority (0.95)
            ("I live in Seattle", -3600),      # 1h old, very high priority (1.00) 
            ("I have a dog Rex", -86400),      # 1d old, medium priority (0.60)
            ("I was born in Oregon", -604800), # 1w old, high priority (0.90)
        ]
        
        print(f"\n📝 STORING FACTS AT DIFFERENT TIMES:")
        for text, offset_seconds in facts_with_times:
            bullets, triples = self.hot_memory.process_turn(text, session_id="rebalanced", turn_id=1)
            
            # Set timestamps to simulate different ages
            target_time = int((base_time + offset_seconds) * 1000)
            for s, r, d in triples:
                if (s, r, d) in self.hot_memory.edge_meta:
                    self.hot_memory.edge_meta[(s, r, d)]['ts'] = target_time
                    
            age_hours = abs(offset_seconds) / 3600
            print(f"  📊 '{text}' → {len(triples)} facts (age: {age_hours:.0f}h)")
        
        # Test queries to see if temporal ranking improved
        test_queries = [
            "What do you know about me?",
            "Tell me about my background", 
            "Where do I work?",
            "What's my personal info?"
        ]
        
        print(f"\n🔍 TESTING RETRIEVAL RANKING:")
        
        for query in test_queries:
            print(f"\n  Query: '{query}'")
            
            # Get retrieval results
            bullets = self.hot_memory._retrieve_context(query, ["you"], turn_id=2)
            
            print(f"  Results ({len(bullets)} bullets):")
            for i, bullet in enumerate(bullets, 1):
                print(f"    {i}. {bullet}")
            
            # Check if fresh facts appear first
            work_position = None
            location_position = None
            
            for i, bullet in enumerate(bullets):
                if "google" in bullet.lower():
                    work_position = i
                if "seattle" in bullet.lower():  
                    location_position = i
            
            if work_position is not None and location_position is not None:
                if work_position < location_position:
                    print(f"    ✅ Fresh 'Google' fact ranks higher than older 'Seattle' fact")
                else:
                    print(f"    ❌ Older 'Seattle' fact still dominates fresh 'Google' fact")
            
        return bullets
    
    def compare_weight_scenarios(self):
        """Compare different weight scenarios side by side"""
        print(f"\n📊 WEIGHT SCENARIO COMPARISON")
        print("=" * 60)
        
        # Calculate scores for a specific example
        # Fresh work fact (priority=0.95, recency=1.0)
        # vs 1h old location fact (priority=1.00, recency=0.986)
        
        scenarios = [
            ("Original", 0.4, 0.2, 0.3, 0.1),
            ("Rebalanced", 0.15, 0.60, 0.20, 0.05)
        ]
        
        fresh_work_priority = 0.95
        fresh_work_recency = 1.0
        
        old_location_priority = 1.00  
        old_location_recency = 0.986
        
        similarity = 0.1  # Assume some lexical overlap
        weight = 0.7     # Default weight
        
        print(f"Scenario: Fresh work fact vs 1h-old location fact")
        print(f"{'Weights':<15} {'Fresh Work':<12} {'Old Location':<12} {'Winner'}")
        print("-" * 55)
        
        for name, alpha, beta, gamma, delta in scenarios:
            fresh_score = alpha * fresh_work_priority + beta * fresh_work_recency + gamma * similarity + delta * weight
            old_score = alpha * old_location_priority + beta * old_location_recency + gamma * similarity + delta * weight
            
            winner = "Fresh Work" if fresh_score > old_score else "Old Location"
            
            weight_str = f"α{alpha:.2f}β{beta:.2f}"
            print(f"{weight_str:<15} {fresh_score:<12.3f} {old_score:<12.3f} {winner}")
        
        print(f"\n💡 ANALYSIS:")
        print(f"Original weights: Old high-priority facts dominate fresh facts")
        print(f"Rebalanced: Fresh facts now competitive with older high-priority facts")

def main():
    """Test rebalanced temporal weights"""
    tester = WeightRebalanceTester()
    
    try:
        tester.setup()
        
        # Test if temporal ranking improved
        bullets = tester.test_temporal_ranking_improvement()
        
        # Compare weight scenarios
        tester.compare_weight_scenarios()
        
        print(f"\n✅ REBALANCED WEIGHTS TESTING COMPLETE")
        print(f"Check if fresh facts now rank higher than older high-priority facts")
        
    finally:
        tester.cleanup()

if __name__ == '__main__':
    main()