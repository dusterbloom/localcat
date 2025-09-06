#!/usr/bin/env python3
"""
Debug Recency Weight Scoring Issue
==================================

Deep investigation into why temporal decay isn't affecting retrieval ranking.
Instruments the scoring formula and traces actual values through retrieval.
"""

import os
import sys
import time
import tempfile
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Add server path  
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths

@dataclass
class ScoredFact:
    """Debug container for scored fact with all components"""
    triple: Tuple[str, str, str]
    score: float
    priority: float
    recency: float 
    similarity: float
    weight: float
    timestamp_ms: int
    age_hours: float
    query: str

class RecencyDebugger:
    def __init__(self):
        self.temp_dir = None
        self.hot_memory = None
        self.fact_storage_log = []
        self.retrieval_log = []
        
    def setup(self):
        """Set up debug environment"""
        self.temp_dir = tempfile.mkdtemp()
        paths = Paths(
            sqlite_path=os.path.join(self.temp_dir, 'debug_memory.db'),
            lmdb_dir=os.path.join(self.temp_dir, 'debug_graph.lmdb')
        )
        store = MemoryStore(paths)
        self.hot_memory = HotMemory(store)
        self.hot_memory.prewarm('en')
        print(f"🔧 Debug storage created at: {self.temp_dir}")
        
    def cleanup(self):
        """Clean up debug data"""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Debug cleanup complete")
    
    def monkey_patch_retrieve_context(self):
        """Instrument the _retrieve_context method to capture scoring details"""
        original_method = self.hot_memory._retrieve_context
        debug_self = self
        
        def debug_retrieve_context(self, query: str, entities: List[str], turn_id: int, intent=None) -> List[str]:
            """Debug version that logs all scoring components"""
            print(f"\n🔍 DEBUG RETRIEVAL: '{query}' (entities: {entities})")
            
            bullets: List[str] = []
            seen_triples = set()
            scored_facts = []
            
            # Copy the core retrieval logic with debug logging
            def _tokens(t: str):
                out = set()
                for w in (t or "").lower().split():
                    w = ''.join(ch for ch in w if ch.isalnum())
                    if len(w) >= 3:
                        out.add(w)
                return out
            qtok = _tokens(query)
            
            # Query entities setup (same as original)
            ent_set = [e for e in entities if e]
            non_you = [e for e in ent_set if e != "you"]
            include_you = any(e == "you" for e in ent_set)
            query_entities = non_you[:4]
            if include_you:
                query_entities.append("you")
                
            # Gating check (same as original)
            t = (query or "").lower()
            attr_keywords = (
                "name", "age", "live", "lives", "hometown", "from",
                "work", "works", "job", "company", "employer",
                "favorite", "colour", "color", "pet", "dog", "cat",
                "spouse", "partner", "wife", "husband", "child", "kid", "kids"
            )
            remember_request = ("remember" in t) or ("save this" in t) or ("note this" in t)
            has_attr = any(kw in t for kw in attr_keywords)
            if include_you and not non_you:
                if not has_attr and not remember_request:
                    print("  ❌ Gated out - generic 'you' query without attributes")
                    return []
            
            # Scoring parameters
            pred_pri = {
                "lives_in": 1.00, "works_at": 0.95, "born_in": 0.90, "moved_from": 0.85,
                "participated_in": 0.80, "friend_of": 0.78, "name": 0.75,
                "favorite_color": 0.70, "favorite_number": 0.70, "has": 0.60, "is": 0.55,
            }
            alpha, beta, gamma, delta = 0.4, 0.2, 0.3, 0.1
            recency_T_ms = 3 * 24 * 3600 * 1000  # 3 days
            now_ms = int(time.time() * 1000)
            
            print(f"  🎯 Scoring weights: α={alpha} β={beta} γ={gamma} δ={delta}")
            print(f"  ⏰ Recency decay: T={recency_T_ms//3600000}h, now={now_ms}")
            
            # Collect and score all candidates
            allowed_rels = {
                "name", "age", "favorite_color", "lives_in", "works_at", "born_in", 
                "moved_from", "participated_in", "went_to", "friend_of", "owns", "has", 
                "favorite_number",
            }
            pronoun_skip = {"he","she","they","him","her","them","who","whom","whose","which"}
            
            for entity in query_entities:
                print(f"  📋 Checking entity: '{entity}'")
                if entity in debug_self.hot_memory.entity_index:
                    candidates = list(debug_self.hot_memory.entity_index[entity])
                    print(f"    Found {len(candidates)} candidate facts")
                    
                    for s, r, d in candidates:
                        if s in pronoun_skip or r not in allowed_rels:
                            continue
                            
                        # Get metadata
                        meta = debug_self.hot_memory.edge_meta.get((s, r, d), {})
                        ts = int(meta.get('ts', 0))
                        weight_val = float(meta.get('weight', 0.3))
                        
                        # Calculate components
                        age_ms = max(0, now_ms - ts) if ts > 0 else 999999999
                        age_hours = age_ms / (3600 * 1000)
                        recency = math.exp(-age_ms / max(1, recency_T_ms)) if ts > 0 else 0.0
                        priority = pred_pri.get(r, 0.5)
                        
                        # Similarity (simplified lexical)
                        stok = _tokens(s) | _tokens(r) | _tokens(d)
                        if qtok and stok:
                            inter = len(qtok & stok)
                            union = len(qtok | stok)
                            similarity = inter / union if union else 0.0
                        else:
                            similarity = 0.0
                        
                        # Final score
                        score = alpha * priority + beta * recency + gamma * similarity + delta * weight_val
                        
                        scored_fact = ScoredFact(
                            triple=(s, r, d),
                            score=score,
                            priority=priority,
                            recency=recency,
                            similarity=similarity,
                            weight=weight_val,
                            timestamp_ms=ts,
                            age_hours=age_hours,
                            query=query
                        )
                        scored_facts.append(scored_fact)
                        
                        print(f"    📊 ({s}, {r}, {d})")
                        print(f"        Score: {score:.3f} = α·{priority:.2f} + β·{recency:.3f} + γ·{similarity:.2f} + δ·{weight_val:.2f}")
                        print(f"        Age: {age_hours:.1f}h (ts: {ts})")
                else:
                    print(f"    ❌ Entity not found in index")
            
            # Sort by score and show ranking
            scored_facts.sort(key=lambda x: x.score, reverse=True)
            
            print(f"\n  📈 RANKING ({len(scored_facts)} total facts):")
            print(f"  {'Rank':<4} {'Score':<8} {'Pri':<6} {'Rec':<8} {'Sim':<6} {'Age':<8} {'Fact'}")
            print("  " + "-" * 80)
            
            for i, fact in enumerate(scored_facts[:10], 1):
                s, r, d = fact.triple
                fact_str = f"({s}, {r}, {d})"
                print(f"  {i:<4} {fact.score:<8.3f} {fact.priority:<6.2f} {fact.recency:<8.3f} {fact.similarity:<6.2f} {fact.age_hours:<8.1f}h {fact_str[:40]}...")
            
            if len(scored_facts) > 10:
                print(f"  ... and {len(scored_facts)-10} more facts")
            
            # Store debug log
            debug_self.retrieval_log.append({
                'query': query,
                'entities': entities,
                'scored_facts': scored_facts,
                'total_candidates': len(scored_facts)
            })
            
            # Return top facts as bullets (simplified)
            for fact in scored_facts[:5]:
                s, r, d = fact.triple
                bullets.append(debug_self.hot_memory._format_memory_bullet(s, r, d))
                
            print(f"  ✅ Returning {len(bullets)} bullets")
            return bullets
        
        # Replace the method
        import types
        self.hot_memory._retrieve_context = types.MethodType(debug_retrieve_context, self.hot_memory)
    
    def test_temporal_scoring(self):
        """Test temporal scoring with facts at different ages"""
        print("🔬 TESTING TEMPORAL SCORING MECHANICS")
        print("=" * 60)
        
        # Monkey patch retrieval for debugging
        self.monkey_patch_retrieve_context()
        
        base_time = time.time()
        
        # Store facts at different times by manipulating timestamps
        test_facts = [
            ("I work at Microsoft", 0),      # Fresh (now)
            ("I live in Seattle", -3600),    # 1 hour ago  
            ("I have a cat Luna", -86400),   # 1 day ago
            ("I graduated from MIT", -259200000), # 3 days ago (should be heavily decayed)
            ("I was born in Oregon", -604800000)  # 1 week ago (should be very low)
        ]
        
        # Store each fact with simulated timestamps
        for text, offset_seconds in test_facts:
            print(f"\n📝 Storing: '{text}' (offset: {offset_seconds//3600}h)")
            
            # Store the fact
            bullets, triples = self.hot_memory.process_turn(text, session_id="debug", turn_id=1)
            
            # Manually adjust timestamps in edge_meta to simulate age
            target_time = int((base_time + offset_seconds) * 1000)
            for s, r, d in triples:
                if (s, r, d) in self.hot_memory.edge_meta:
                    self.hot_memory.edge_meta[(s, r, d)]['ts'] = target_time
                    print(f"    ✅ Set timestamp for ({s}, {r}, {d}) to {target_time}")
            
            self.fact_storage_log.append({
                'text': text,
                'offset_hours': offset_seconds // 3600,
                'timestamp_ms': target_time,
                'triples': triples
            })
        
        # Now test queries at different time points to see if recency affects ranking
        test_queries = [
            "What do you know about my work?",
            "Where do I live?", 
            "Tell me about my education",
            "What do you remember about me?"
        ]
        
        print(f"\n🔍 TESTING RETRIEVAL WITH TEMPORAL RANKING")
        for query in test_queries:
            print(f"\n" + "="*60)
            bullets = self.hot_memory._retrieve_context(query, ["you"], turn_id=2)
            print(f"Final bullets returned: {bullets}")
    
    def analyze_temporal_issues(self):
        """Analyze why temporal decay might not be working"""
        print(f"\n📊 TEMPORAL ANALYSIS")
        print("=" * 60)
        
        if not self.retrieval_log:
            print("❌ No retrieval data to analyze")
            return
        
        # Check if recency scores are varying
        all_recency_scores = []
        for retrieval in self.retrieval_log:
            for fact in retrieval['scored_facts']:
                all_recency_scores.append(fact.recency)
        
        if all_recency_scores:
            min_rec = min(all_recency_scores)
            max_rec = max(all_recency_scores)
            avg_rec = sum(all_recency_scores) / len(all_recency_scores)
            
            print(f"🔢 Recency Score Distribution:")
            print(f"  Min: {min_rec:.6f}")  
            print(f"  Max: {max_rec:.6f}")
            print(f"  Avg: {avg_rec:.6f}")
            print(f"  Range: {max_rec - min_rec:.6f}")
            
            if (max_rec - min_rec) < 0.01:
                print("❌ PROBLEM: Recency scores have very low variance!")
                print("   This suggests timestamps are too similar or decay constant is wrong")
            else:
                print("✅ Recency scores show good temporal spread")
        
        # Check if high-recency facts are winning
        print(f"\n🏆 TOP SCORING FACTS ANALYSIS:")
        for i, retrieval in enumerate(self.retrieval_log):
            query = retrieval['query']
            facts = retrieval['scored_facts'][:5]  # Top 5
            
            print(f"\n  Query {i+1}: '{query}'")
            if facts:
                # Check if most recent facts score highest
                facts_by_recency = sorted(facts, key=lambda x: x.recency, reverse=True)
                facts_by_score = sorted(facts, key=lambda x: x.score, reverse=True)
                
                print(f"    Top by recency: {facts_by_recency[0].triple} (rec: {facts_by_recency[0].recency:.3f})")
                print(f"    Top by score:   {facts_by_score[0].triple} (score: {facts_by_score[0].score:.3f})")
                
                # Check correlation
                if facts_by_recency[0].triple == facts_by_score[0].triple:
                    print("    ✅ Most recent fact also scores highest")
                else:
                    print("    ❌ Most recent fact doesn't score highest - other components dominating")
                    top_score_fact = facts_by_score[0]
                    print(f"    🔍 Top scorer components: pri={top_score_fact.priority:.2f}, rec={top_score_fact.recency:.3f}, sim={top_score_fact.similarity:.2f}")
        
        # Check recency weight impact
        alpha, beta = 0.4, 0.2  # Current weights
        print(f"\n⚖️  WEIGHT IMPACT ANALYSIS:")
        print(f"  Current weights: α(priority)={alpha}, β(recency)={beta}")
        print(f"  Recency contribution to score: β × recency_score")
        
        if all_recency_scores:
            max_recency_contribution = beta * max(all_recency_scores)
            max_priority_contribution = alpha * 1.0  # Max priority is 1.0
            
            print(f"  Max recency contribution: {max_recency_contribution:.3f}")
            print(f"  Max priority contribution: {max_priority_contribution:.3f}")
            
            if max_recency_contribution < max_priority_contribution * 0.5:
                print("  ⚠️  Recency weight may be too low compared to priority")
                suggested_beta = max_priority_contribution / max(all_recency_scores) * 0.8
                print(f"  💡 Suggested β increase: {suggested_beta:.2f}")
    
    def calculate_expected_recency(self, hours_ago: int) -> float:
        """Calculate expected recency score"""
        if hours_ago <= 0:
            return 1.0
        recency_T_ms = 3 * 24 * 3600 * 1000  # 3 days in ms
        age_ms = hours_ago * 3600 * 1000     # Convert hours to ms
        return math.exp(-age_ms / recency_T_ms)

def main():
    """Run recency scoring debug analysis"""
    debugger = RecencyDebugger()
    
    try:
        debugger.setup()
        
        # Test the temporal scoring mechanism
        debugger.test_temporal_scoring()
        
        # Analyze the results
        debugger.analyze_temporal_issues()
        
        # Show expected vs actual recency decay
        print(f"\n📉 EXPECTED RECENCY DECAY CURVE:")
        time_points = [0, 1, 6, 24, 72, 168]  # hours
        for hours in time_points:
            expected = debugger.calculate_expected_recency(hours)
            print(f"  {hours:3d}h: {expected:.6f}")
        
    finally:
        debugger.cleanup()

if __name__ == '__main__':
    main()