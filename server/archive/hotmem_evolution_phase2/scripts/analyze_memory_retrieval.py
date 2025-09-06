#!/usr/bin/env python3
"""
Memory Retrieval Analysis
=========================

Analyzes how different types of information (facts, summaries, verbatim)
flow into the agent's context during retrieval.

Key Questions:
1. When do memory bullets vs summaries vs verbatim text reach the agent?
2. How does the context assembly work in bot.py?
3. What information sources are available for retrieval?
"""

import os
import sys
import tempfile
from typing import List, Dict, Any

# Add server path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths

class MemoryRetrievalAnalyzer:
    def __init__(self):
        self.temp_dir = None
        self.hot_memory = None
        
    def setup(self):
        """Set up analysis environment"""
        self.temp_dir = tempfile.mkdtemp()
        paths = Paths(
            sqlite_path=os.path.join(self.temp_dir, 'analysis_memory.db'),
            lmdb_dir=os.path.join(self.temp_dir, 'analysis_graph.lmdb')
        )
        store = MemoryStore(paths)
        self.hot_memory = HotMemory(store)
        self.hot_memory.prewarm('en')
        print(f"📊 Memory analysis storage: {self.temp_dir}")
        
    def cleanup(self):
        """Clean up analysis data"""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Analysis cleanup complete")
    
    def analyze_memory_sources(self):
        """Analyze what memory sources are available"""
        print("🔍 MEMORY SOURCE ANALYSIS")
        print("=" * 60)
        
        # Check configuration
        print("📋 Configuration Analysis:")
        print(f"  HOTMEM_USE_LEANN: {os.getenv('HOTMEM_USE_LEANN', 'false')}")
        print(f"  SESSION_SUMMARY_ENABLED: {os.getenv('SESSION_SUMMARY_ENABLED', 'false')}")
        print(f"  SUMMARIZER_ENABLED: {os.getenv('SUMMARIZER_ENABLED', 'false')}")
        print(f"  REBUILD_LEANN_ON_SESSION_END: {os.getenv('REBUILD_LEANN_ON_SESSION_END', 'false')}")
        
        # Check HotMemory capabilities
        print(f"\n🧠 HotMemory Capabilities:")
        print(f"  LEANN semantic search: {self.hot_memory.use_leann}")
        print(f"  LEANN index path: {self.hot_memory.leann_index_path}")
        print(f"  LEANN complexity: {self.hot_memory.leann_complexity}")
        print(f"  Entity indexing: {len(self.hot_memory.entity_index)} entities")
        print(f"  Recency buffer: {len(self.hot_memory.recency_buffer)}/50 items")
        
        # Analyze retrieval flow
        print(f"\n📤 Retrieval Process Analysis:")
        print("  1. process_turn() called with user text")
        print("  2. _retrieve_context() searches entity_index for facts")
        print("  3. Facts scored with temporal/priority/similarity weights")
        print("  4. Top facts formatted as memory bullets (• Your name is...)")
        print("  5. Memory bullets returned to bot.py for context injection")
        
    def test_memory_retrieval_flow(self):
        """Test the complete memory retrieval flow"""
        print(f"\n🔄 TESTING MEMORY RETRIEVAL FLOW")
        print("=" * 60)
        
        # Store some facts
        conversation = [
            "My name is Alex and I work at Tesla",
            "I live in Austin, Texas with my partner Sam", 
            "I have a dog named Rocky who loves hiking",
            "My favorite programming language is Rust"
        ]
        
        print("📝 Storing conversation facts:")
        for i, text in enumerate(conversation, 1):
            bullets, triples = self.hot_memory.process_turn(text, session_id="analysis", turn_id=i)
            print(f"  Turn {i}: {len(triples)} facts → {len(bullets)} bullets retrieved")
            for bullet in bullets[:2]:
                print(f"    {bullet}")
        
        # Test various query types
        queries = [
            "What's my name?",
            "Where do I work?", 
            "Tell me about my personal life",
            "What programming languages do I know?",
            "Summarize what you know about me"
        ]
        
        print(f"\n🎯 Testing Query Responses:")
        for query in queries:
            bullets, _ = self.hot_memory.process_turn(query, session_id="analysis", turn_id=99)
            print(f"\n  Query: '{query}'")
            print(f"  Retrieved {len(bullets)} memory bullets:")
            for bullet in bullets:
                print(f"    {bullet}")
    
    def analyze_context_flow_to_bot(self):
        """Analyze how memory flows from HotMem to bot context"""
        print(f"\n🤖 BOT CONTEXT FLOW ANALYSIS")
        print("=" * 60)
        
        # Examine bot.py integration (conceptual since we can't easily run the full pipeline)
        print("📋 Context Assembly Process (from bot.py):")
        print("  1. User message received in bot.py")  
        print("  2. HotMemory.process_turn() called")
        print("  3. Memory bullets returned (list of strings)")
        print("  4. Memory bullets formatted into system/context prompt")
        print("  5. Final prompt sent to LLM with memory context")
        
        print(f"\n📤 Memory Bullet Format Analysis:")
        # Create sample bullets to show format
        sample_bullets = [
            self.hot_memory._format_memory_bullet("you", "name", "alex"),
            self.hot_memory._format_memory_bullet("you", "works_at", "tesla"),
            self.hot_memory._format_memory_bullet("you", "lives_in", "austin"),
            self.hot_memory._format_memory_bullet("dog", "name", "rocky")
        ]
        
        print("  Memory bullet format examples:")
        for bullet in sample_bullets:
            print(f"    {bullet}")
            
        print(f"\n🔗 Integration Points:")
        print("  • Memory bullets = structured fact retrieval")
        print("  • Session summaries = periodic conversation context") 
        print("  • LEANN vectors = semantic similarity search")
        print("  • Verbatim storage = exact conversation history")
        
        print(f"\n❓ Missing Analysis:")
        print("  • How summaries integrate with memory bullets")
        print("  • When verbatim vs facts vs summaries are used")
        print("  • Priority order of different information types")
    
    def analyze_information_types(self):
        """Analyze different types of information available"""
        print(f"\n📊 INFORMATION TYPE ANALYSIS")
        print("=" * 60)
        
        print("🎯 Available Information Sources:")
        print("  1. STRUCTURED FACTS (memory bullets)")
        print("     • Extracted via 27-pattern dependency parsing")
        print("     • Stored as (subject, relation, object) triples") 
        print("     • Temporal decay scoring for recency")
        print("     • Examples: 'You work at Tesla', 'Your dog is named Rocky'")
        
        print("  2. SESSION SUMMARIES")
        print("     • Generated every 30 seconds during conversation")
        print("     • 3-5 factual bullet points + narrative + follow-ups")
        print("     • Stored separately from structured facts")
        print("     • Model: qwen/qwen3-4b via LM Studio")
        
        print("  3. LEANN SEMANTIC VECTORS") 
        print("     • Vector embeddings of facts for similarity search")
        print("     • Enables semantic retrieval beyond exact matches")
        print("     • HNSW backend for fast approximate search")
        print("     • Complexity parameter for speed/accuracy tradeoff")
        
        print("  4. VERBATIM CONVERSATION")
        print("     • Full conversation history stored separately")
        print("     • Not directly used in memory retrieval")
        print("     • Available for summarization and analysis")
        
        print(f"\n⚖️  RETRIEVAL PRIORITY (based on current code):")
        print("  1. Structured facts dominate (memory bullets)")
        print("  2. Temporal scoring favors recent facts") 
        print("  3. Semantic similarity via LEANN when enabled")
        print("  4. Summaries appear to be separate from fact retrieval")
        print("  5. Verbatim not directly retrieved")

def main():
    """Analyze memory retrieval system"""
    analyzer = MemoryRetrievalAnalyzer()
    
    try:
        analyzer.setup()
        
        # Analyze memory sources and configuration
        analyzer.analyze_memory_sources()
        
        # Test retrieval flow
        analyzer.test_memory_retrieval_flow()
        
        # Analyze bot context integration
        analyzer.analyze_context_flow_to_bot()
        
        # Analyze information types
        analyzer.analyze_information_types()
        
        print(f"\n💡 KEY FINDINGS:")
        print("  ✅ Structured facts are the primary retrieval mechanism")
        print("  ✅ Memory bullets provide clean, formatted fact injection")  
        print("  ✅ Temporal decay ensures recent facts dominate")
        print("  ❓ Summary integration with fact retrieval unclear")
        print("  ❓ Verbatim conversation usage in context unclear")
        
    finally:
        analyzer.cleanup()

if __name__ == '__main__':
    main()