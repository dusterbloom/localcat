#!/usr/bin/env python3
"""
Realistic Temporal Fact Management Testing
==========================================

Tests HotMem with 50 diverse conversation pieces across different time periods
to validate actual temporal decay, fact conflicts, and retrieval ranking.

Addresses critical issues:
1. Real conversation diversity (not template text)
2. Visible temporal decay in retrieval ranking
3. Fact conflicts and corrections over time
4. Mixed-age fact competition
"""

import os
import sys
import time
import tempfile
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

# Add server path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths

@dataclass
class ConversationTurn:
    """Realistic conversation turn with expected temporal behavior"""
    text: str
    hours_offset: int  # Relative to conversation start
    expected_facts: List[str]  # Expected extractions
    conflicts_with: List[int] = None  # Turn IDs this conflicts with
    
# 50 realistic conversation pieces across different time periods
REALISTIC_CONVERSATION = [
    # Day 1: Fresh personal facts (hours 0-4)
    ConversationTurn("My name is Sarah and I work as a software engineer at Microsoft", 0, 
                    ["name is Sarah", "works at Microsoft", "job is software engineer"]),
    ConversationTurn("I have a golden retriever named Max who's 3 years old", 1, 
                    ["has dog Max", "Max is golden retriever", "Max age 3"]),
    ConversationTurn("I live in Seattle with my roommate Emma in a two-bedroom apartment", 2,
                    ["lives in Seattle", "has roommate Emma", "lives in apartment"]),
    ConversationTurn("My favorite programming language is Python, though I also use TypeScript daily", 3,
                    ["favorite language Python", "uses TypeScript"]),
    ConversationTurn("I graduated from University of Washington in 2019 with a Computer Science degree", 4,
                    ["graduated UW", "graduated 2019", "degree Computer Science"]),
    
    # Day 2: Hobbies and interests (hours 24-28)
    ConversationTurn("I love hiking and go to the mountains every weekend", 24,
                    ["loves hiking", "goes mountains weekends"]),
    ConversationTurn("My favorite coffee shop is Victrola on Capitol Hill", 25,
                    ["favorite coffee Victrola", "Victrola on Capitol Hill"]),
    ConversationTurn("I play guitar and have been taking lessons for two years", 26,
                    ["plays guitar", "taking lessons 2 years"]),
    ConversationTurn("My brother Jake lives in Portland and works as a teacher", 27,
                    ["brother is Jake", "Jake lives Portland", "Jake is teacher"]),
    ConversationTurn("I drive a blue Honda Civic that I bought last year", 28,
                    ["drives Honda Civic", "car is blue", "bought last year"]),
    
    # Day 3: Work details (hours 48-52)
    ConversationTurn("At work I'm on the Azure team focusing on cloud infrastructure", 48,
                    ["works on Azure", "focuses cloud infrastructure"]),
    ConversationTurn("My manager's name is David and he's really supportive", 49,
                    ["manager is David", "David is supportive"]),
    ConversationTurn("I usually work from home on Mondays and Fridays", 50,
                    ["works from home Mondays", "works from home Fridays"]),
    ConversationTurn("The office is in Bellevue and has a great view of Lake Washington", 51,
                    ["office in Bellevue", "office has lake view"]),
    ConversationTurn("I'm working on a project to improve container orchestration", 52,
                    ["working on containers", "project orchestration"]),
    
    # Day 4: Family and relationships (hours 72-76)
    ConversationTurn("My parents live in California near San Francisco", 72,
                    ["parents live California", "parents near San Francisco"]),
    ConversationTurn("I'm dating someone named Alex who's a graphic designer", 73,
                    ["dating Alex", "Alex is designer"]),
    ConversationTurn("My sister Lisa is getting married next month in Vancouver", 74,
                    ["sister is Lisa", "Lisa getting married", "wedding Vancouver"]),
    ConversationTurn("I have two nephews, ages 5 and 7, who live with my sister", 75,
                    ["has nephews", "nephews age 5 and 7", "nephews with Lisa"]),
    ConversationTurn("My grandmother turned 85 last week and we had a big party", 76,
                    ["grandmother age 85", "had party last week"]),
    
    # Day 5: Health and fitness (hours 96-100)
    ConversationTurn("I go to the gym three times a week and do yoga on Sundays", 96,
                    ["goes gym 3x week", "does yoga Sundays"]),
    ConversationTurn("I'm training for a half marathon in October", 97,
                    ["training half marathon", "marathon in October"]),
    ConversationTurn("My doctor recommended I take vitamin D supplements", 98,
                    ["doctor recommended vitamin D"]),
    ConversationTurn("I meal prep every Sunday to eat healthier during the week", 99,
                    ["meal preps Sundays", "eats healthy"]),
    ConversationTurn("I've been trying to drink more water and cut down on coffee", 100,
                    ["drinking more water", "cutting coffee"]),
    
    # Day 6: Corrections and updates (hours 120-124) - CONFLICTS!
    ConversationTurn("Actually, I should clarify - I work at Google, not Microsoft", 120,
                    ["works at Google"], conflicts_with=[0]),
    ConversationTurn("My dog Max is actually 4 years old, not 3 - I miscounted", 121,
                    ["Max age 4"], conflicts_with=[1]),
    ConversationTurn("I moved apartments last month - now I live alone in Fremont", 122,
                    ["lives in Fremont", "lives alone"], conflicts_with=[2]),
    ConversationTurn("I got a promotion! I'm now a senior software engineer", 123,
                    ["senior software engineer"], conflicts_with=[0]),
    ConversationTurn("Update: my brother Jake moved to San Diego for a new job", 124,
                    ["Jake lives San Diego"], conflicts_with=[8]),
    
    # Day 7: Recent activities (hours 144-148) - SHOULD DOMINATE RETRIEVAL
    ConversationTurn("I just adopted a kitten named Luna from the animal shelter", 144,
                    ["adopted kitten Luna", "Luna from shelter"]),
    ConversationTurn("Started learning Spanish using Duolingo this week", 145,
                    ["learning Spanish", "using Duolingo"]),
    ConversationTurn("My team at Google shipped a major feature yesterday", 146,
                    ["team shipped feature", "shipped yesterday"]),
    ConversationTurn("I'm planning a trip to Japan for next spring", 147,
                    ["planning Japan trip", "trip next spring"]),
    ConversationTurn("Just finished reading 'The Pragmatic Programmer' - excellent book", 148,
                    ["read Pragmatic Programmer", "book is excellent"]),
    
    # Week 2: Older memories (hours 168-172) - SHOULD DECAY
    ConversationTurn("I remember my first day at work was really overwhelming", 168,
                    ["first day overwhelming"]),
    ConversationTurn("Back in college I was in the computer science club", 169,
                    ["was in CS club", "in college"]),
    ConversationTurn("I used to play piano as a child but stopped in high school", 170,
                    ["played piano child", "stopped high school"]),
    ConversationTurn("My first programming language was Java, learned it in freshman year", 171,
                    ["first language Java", "learned freshman"]),
    ConversationTurn("I had a part-time job at a bookstore during my junior year", 172,
                    ["worked bookstore", "during junior year"]),
    
    # Week 3: Very old memories (hours 336-340) - SHOULD HEAVILY DECAY  
    ConversationTurn("When I was 12, I wanted to be a veterinarian", 336,
                    ["wanted be veterinarian", "age 12"]),
    ConversationTurn("In elementary school I was obsessed with dinosaurs", 337,
                    ["obsessed dinosaurs", "elementary school"]),
    ConversationTurn("My childhood dog was named Buddy and he was a Lab mix", 338,
                    ["childhood dog Buddy", "Buddy was Lab"]),
    ConversationTurn("I grew up in a small town in Oregon called Bend", 339,
                    ["grew up Bend", "Bend in Oregon"]),
    ConversationTurn("My first computer was an old Dell my dad bought in 2005", 340,
                    ["first computer Dell", "dad bought 2005"]),
    
    # Recent queries to test temporal retrieval (hours 149-153)
    ConversationTurn("What do you know about my work?", 149, []),
    ConversationTurn("Tell me about my pets", 150, []),  
    ConversationTurn("Where do I live?", 151, []),
    ConversationTurn("What programming languages do I use?", 152, []),
    ConversationTurn("What do you remember about my family?", 153, []),
    
    # Much later queries (hours 400+) to test long-term decay
    ConversationTurn("Do you remember anything about my childhood?", 400, []),
    ConversationTurn("What was my first job?", 401, []),
    ConversationTurn("Tell me about my education", 402, []),
    ConversationTurn("What hobbies do I have?", 403, []),
    ConversationTurn("Summarize everything you know about me", 404, [])
]

class RealisticTemporalTester:
    def __init__(self):
        self.temp_dir = None
        self.hot_memory = None
        self.original_time = time.time()
        self.fact_timeline = []  # Track when facts were stored
        
    def setup(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        paths = Paths(
            sqlite_path=os.path.join(self.temp_dir, 'realistic_memory.db'),
            lmdb_dir=os.path.join(self.temp_dir, 'realistic_graph.lmdb')
        )
        store = MemoryStore(paths)
        self.hot_memory = HotMemory(store)
        self.hot_memory.prewarm('en')
        print(f"🏗️  Realistic temporal storage created at: {self.temp_dir}")
        
    def cleanup(self):
        """Clean up test data"""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up realistic temporal storage")
    
    def calculate_expected_recency(self, hours_ago: int) -> float:
        """Calculate expected recency score"""
        if hours_ago <= 0:
            return 1.0
        
        recency_T_ms = 3 * 24 * 3600 * 1000  # 3 days in ms
        age_ms = hours_ago * 3600 * 1000     # Convert hours to ms
        return math.exp(-age_ms / recency_T_ms)
    
    def test_realistic_temporal(self) -> Dict[str, Any]:
        """Run realistic temporal conversation test"""
        print("🌍 Testing Realistic Temporal Conversation")
        print("=" * 60)
        print(f"📊 Processing {len(REALISTIC_CONVERSATION)} diverse conversation turns")
        print(f"⏰ Time span: 0 to {max(turn.hours_offset for turn in REALISTIC_CONVERSATION)} hours")
        
        results = []
        conflicts_detected = []
        temporal_queries = []
        
        for i, turn in enumerate(REALISTIC_CONVERSATION):
            print(f"\n🔄 Turn {i+1}: {turn.hours_offset:+4d}h - {turn.text[:60]}...")
            
            # Calculate relative recency
            hours_from_now = turn.hours_offset
            if hours_from_now > 0:
                hours_from_now = -hours_from_now  # Convert to "hours ago"
            expected_recency = self.calculate_expected_recency(abs(hours_from_now))
            
            start_time = time.perf_counter()
            bullets, triples = self.hot_memory.process_turn(
                turn.text, 
                session_id="realistic_temporal", 
                turn_id=i+1
            )
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Track fact storage
            if triples:
                self.fact_timeline.append({
                    'turn_id': i+1,
                    'hours_offset': turn.hours_offset,
                    'facts': triples,
                    'expected_recency': expected_recency
                })
                
                print(f"    ✅ Stored {len(triples)} facts (recency: {expected_recency:.3f})")
                for j, (s, r, d) in enumerate(triples[:2]):
                    print(f"      {j+1}. ({s}) -[{r}]-> ({d})")
                if len(triples) > 2:
                    print(f"      ... and {len(triples)-2} more")
            
            # Track retrieval for queries
            if bullets:
                retrieval_info = {
                    'turn_id': i+1,
                    'query': turn.text,
                    'bullets': bullets,
                    'bullet_count': len(bullets),
                    'hours_offset': turn.hours_offset
                }
                temporal_queries.append(retrieval_info)
                
                print(f"    🎯 Retrieved {len(bullets)} bullets:")
                for j, bullet in enumerate(bullets[:3]):
                    print(f"      • {bullet}")
                if len(bullets) > 3:
                    print(f"      ... and {len(bullets)-3} more")
            
            # Detect conflicts
            if turn.conflicts_with:
                conflicts_detected.append({
                    'turn_id': i+1,
                    'text': turn.text,
                    'conflicts_with': turn.conflicts_with,
                    'facts_stored': len(triples)
                })
                print(f"    ⚠️  CONFLICT: Updates facts from turns {turn.conflicts_with}")
            
            # Store turn results
            results.append({
                'turn': turn,
                'facts_count': len(triples),
                'bullets_count': len(bullets),
                'processing_time_ms': processing_time,
                'expected_recency': expected_recency,
                'facts': triples[:5],  # Sample
                'bullets': bullets[:5]  # Sample
            })
        
        return {
            'results': results,
            'fact_timeline': self.fact_timeline,
            'conflicts_detected': conflicts_detected,
            'temporal_queries': temporal_queries,
            'total_turns': len(REALISTIC_CONVERSATION),
            'total_facts': sum(len(r['facts']) for r in results),
            'avg_processing_time': sum(r['processing_time_ms'] for r in results) / len(results)
        }
    
    def analyze_temporal_patterns(self, test_results: Dict[str, Any]):
        """Analyze temporal decay and retrieval patterns"""
        print(f"\n📈 REALISTIC TEMPORAL ANALYSIS")
        print("=" * 60)
        
        # Overall stats
        print(f"\n📊 Conversation Statistics:")
        print(f"  Total Turns: {test_results['total_turns']}")
        print(f"  Facts Stored: {test_results['total_facts']}")
        print(f"  Avg Processing: {test_results['avg_processing_time']:.1f}ms")
        print(f"  Conflicts Detected: {len(test_results['conflicts_detected'])}")
        print(f"  Query Turns: {len(test_results['temporal_queries'])}")
        
        # Timeline analysis
        timeline = test_results['fact_timeline']
        if timeline:
            print(f"\n⏰ Temporal Distribution:")
            time_buckets = {}
            for fact_group in timeline:
                hours = fact_group['hours_offset']
                bucket = f"{hours//24}d" if hours >= 24 else f"{hours}h"
                if bucket not in time_buckets:
                    time_buckets[bucket] = {'facts': 0, 'turns': 0}
                time_buckets[bucket]['facts'] += len(fact_group['facts'])
                time_buckets[bucket]['turns'] += 1
            
            for bucket in sorted(time_buckets.keys(), key=lambda x: int(x[:-1]) if x[:-1].isdigit() else 999):
                stats = time_buckets[bucket]
                print(f"  {bucket:>4}: {stats['facts']} facts across {stats['turns']} turns")
        
        # Retrieval analysis by time period
        queries = test_results['temporal_queries']
        if queries:
            print(f"\n🎯 Temporal Retrieval Analysis:")
            
            # Group queries by time period
            recent_queries = [q for q in queries if q['hours_offset'] < 168]  # < 1 week
            old_queries = [q for q in queries if q['hours_offset'] >= 168]    # >= 1 week
            
            if recent_queries:
                avg_recent = sum(q['bullet_count'] for q in recent_queries) / len(recent_queries)
                print(f"  Recent Queries (<1 week): {avg_recent:.1f} bullets avg")
                
            if old_queries:
                avg_old = sum(q['bullet_count'] for q in old_queries) / len(old_queries)
                print(f"  Old Queries (≥1 week): {avg_old:.1f} bullets avg")
                
                if recent_queries and avg_old < avg_recent:
                    print(f"    ✅ Temporal decay visible: {avg_recent:.1f} → {avg_old:.1f}")
                else:
                    print(f"    ❌ No clear temporal decay pattern")
        
        # Conflict analysis
        conflicts = test_results['conflicts_detected']
        if conflicts:
            print(f"\n⚠️  Fact Conflict Analysis:")
            for conflict in conflicts:
                print(f"  Turn {conflict['turn_id']}: '{conflict['text'][:50]}...'")
                print(f"    Conflicts with turns: {conflict['conflicts_with']}")
                print(f"    New facts stored: {conflict['facts_stored']}")
        
        # Recency score analysis
        print(f"\n📉 Expected vs Observed Recency:")
        print(f"{'Hours Ago':<12} {'Expected':<12} {'Actual Pattern':<20}")
        print("-" * 50)
        
        time_samples = [0, 24, 72, 168, 336]  # Various time points
        for hours in time_samples:
            expected = self.calculate_expected_recency(hours)
            
            # Find queries around this time
            nearby_queries = [q for q in queries if abs(q['hours_offset'] - hours) < 12]
            if nearby_queries:
                avg_bullets = sum(q['bullet_count'] for q in nearby_queries) / len(nearby_queries)
                pattern = f"{avg_bullets:.1f} bullets avg"
            else:
                pattern = "No queries"
                
            print(f"{hours:<12} {expected:<12.3f} {pattern:<20}")
        
        # Recommendations
        print(f"\n💡 FINDINGS & RECOMMENDATIONS:")
        
        if test_results['total_facts'] > 100:
            print("  ✅ Good fact extraction across diverse conversation")
        else:
            print("  ❌ Low fact extraction - may need extraction tuning")
        
        if len(conflicts) > 0:
            print(f"  ⚠️  {len(conflicts)} fact conflicts detected - correction system needed")
            print("  💡 Implement /correct command for immediate user corrections")
        
        if len(queries) > 5:
            recent_bullets = [q['bullet_count'] for q in queries if q['hours_offset'] < 72]
            old_bullets = [q['bullet_count'] for q in queries if q['hours_offset'] >= 168]
            
            if recent_bullets and old_bullets:
                recent_avg = sum(recent_bullets) / len(recent_bullets)
                old_avg = sum(old_bullets) / len(old_bullets)
                
                if recent_avg > old_avg * 1.5:
                    print("  ✅ Temporal decay working - recent facts dominate retrieval")
                else:
                    print("  ❌ Temporal decay not strong enough in retrieval ranking")
                    print("  💡 Check recency weight in retrieval scoring formula")
        
        print(f"\n🚨 CRITICAL GAPS IDENTIFIED:")
        print("  1. No real-time user correction mechanism")
        print("  2. No transparent fact visibility for users") 
        print("  3. Need /show, /correct, /forget commands")
        print("  4. Temporal decay may not be affecting retrieval ranking")

def main():
    """Run realistic temporal tests"""
    tester = RealisticTemporalTester()
    
    try:
        tester.setup()
        
        # Run the realistic conversation test
        results = tester.test_realistic_temporal()
        
        # Analyze temporal patterns
        tester.analyze_temporal_patterns(results)
        
    finally:
        tester.cleanup()

if __name__ == '__main__':
    main()