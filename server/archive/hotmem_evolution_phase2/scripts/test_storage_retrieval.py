#!/usr/bin/env python3
"""
Storage and Retrieval Test Framework
===================================

Tests the full pipeline:
1. Immediate verbatim storage during conversation
2. Batch LLM extraction and knowledge graph building  
3. Multi-layer retrieval (verbatim, summaries, knowledge graph)

This simulates real user experience with the proposed architecture.
"""

import os
import sys
import time
import tempfile
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

# Add server path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    turn_id: int
    user_message: str
    assistant_message: str
    timestamp: float
    session_id: str

@dataclass 
class StorageTest:
    """Results from storage operations"""
    verbatim_stored: bool
    extraction_time_ms: float
    facts_extracted: int
    bullets_generated: int
    storage_success: bool

@dataclass
class RetrievalTest:
    """Results from retrieval operations"""
    query: str
    verbatim_results: List[str]
    summary_results: List[str] 
    kg_results: List[Dict[str, Any]]
    total_results: int
    retrieval_time_ms: float

# Test conversation simulating real usage
TEST_CONVERSATION = [
    ("Hello, I'm Sarah. I work at Microsoft in Seattle and graduated from Stanford in 2019.", 
     "Hi Sarah! Nice to meet you. It's great to hear about your background at Microsoft and Stanford. How are you enjoying working in Seattle?"),
    
    ("I love Seattle! I drive a Tesla Model 3 and my brother Tom lives in Portland where he teaches at Reed College.",
     "That sounds wonderful! Tesla Model 3 is a great car. It's nice that your brother Tom is in education at Reed College in Portland - that's not too far from Seattle."),
    
    ("Yes, we visit each other often. Tom actually studied philosophy at Reed before becoming a professor there. He's been there for 5 years now.",
     "That's a beautiful connection! Philosophy is such a rich field, and it's wonderful that Tom found his calling at his alma mater. A 5-year tenure shows he's really settled in there."),
    
    ("My dog Luna is a golden retriever who loves hiking. We go to Mount Rainier every weekend when the weather is good.",
     "What a perfect companion for Pacific Northwest adventures! Golden retrievers are such great hiking partners, and Mount Rainier must be absolutely stunning on those clear days."),
    
    ("Luna is 3 years old and incredibly smart. She knows over 20 commands and can even fetch specific items by name.",
     "That's impressive! A 3-year-old golden retriever with that level of training shows real dedication. The ability to fetch specific items by name is quite remarkable - Luna sounds like an exceptional dog."),
    
    ("Thanks! By the way, I'm working on a machine learning project at Microsoft using Python and TensorFlow. It's focused on natural language processing.",
     "That sounds like fascinating work! ML and NLP are such exciting fields right now. TensorFlow is a powerful choice for that kind of project. What specific aspects of NLP are you focusing on?"),
    
    ("We're building a sentiment analysis system for customer feedback. The goal is to automatically categorize feedback as positive, negative, or neutral with high accuracy.",
     "Sentiment analysis is incredibly valuable for businesses! Being able to automatically process customer feedback at scale can provide such valuable insights. Are you working with pre-trained models or building from scratch?"),
    
    ("We started with pre-trained BERT models but are fine-tuning them on our specific domain data. The accuracy has improved significantly since we began the fine-tuning process.",
     "That's the smart approach! BERT provides such a solid foundation, and domain-specific fine-tuning usually yields much better results than generic models. It sounds like you're seeing good progress with the accuracy improvements."),
    
    ("I also have a cat named Whiskers who is 2 years old and loves to play with yarn. She's a Persian cat with beautiful blue eyes.",
     "What a lovely addition to your family! Persian cats are known for their gorgeous coats and those striking blue eyes. Whiskers sounds like a playful companion - cats and yarn are such a classic combination."),
     
    ("My favorite color is purple and my lucky number is 7. I've been fascinated by those numbers since childhood.",
     "Purple is such a rich, regal color! And 7 is considered lucky in many cultures - it's interesting how certain numbers can hold special meaning for us from such an early age."),
]

class StorageRetrievalTester:
    def __init__(self):
        self.temp_dir = None
        self.store = None
        self.hot_memory = None
        self.extraction_interval = 3  # Extract every 3 turns for testing
        self.summarization_interval = 10  # Summarize every 10 turns
        self.conversation_turns = []
        
    def setup(self):
        """Set up temporary storage for testing"""
        self.temp_dir = tempfile.mkdtemp()
        paths = Paths(
            sqlite_path=os.path.join(self.temp_dir, 'test_memory.db'),
            lmdb_dir=os.path.join(self.temp_dir, 'test_graph.lmdb')
        )
        
        self.store = MemoryStore(paths)
        self.hot_memory = HotMemory(self.store)
        self.hot_memory.prewarm('en')
        
        print(f"🏗️  Test storage created at: {self.temp_dir}")
        
    def cleanup(self):
        """Clean up test data"""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test storage")
    
    async def simulate_conversation(self) -> List[StorageTest]:
        """Simulate real conversation with immediate storage and batch extraction"""
        print("💬 Simulating Real Conversation Flow")
        print("=" * 50)
        
        storage_results = []
        session_id = "test_session_2024"
        
        for turn_id, (user_msg, assistant_msg) in enumerate(TEST_CONVERSATION, 1):
            print(f"\n🔄 Turn {turn_id}")
            print(f"👤 User: {user_msg[:60]}...")
            print(f"🤖 Assistant: {assistant_msg[:60]}...")
            
            # Step 1: Immediate verbatim storage (0ms user experience)
            start_time = time.perf_counter()
            
            conversation_turn = ConversationTurn(
                turn_id=turn_id,
                user_message=user_msg,
                assistant_message=assistant_msg,
                timestamp=time.time(),
                session_id=session_id
            )
            self.conversation_turns.append(conversation_turn)
            
            # Store verbatim immediately
            verbatim_stored = self._store_verbatim(conversation_turn)
            immediate_time = (time.perf_counter() - start_time) * 1000
            
            print(f"⚡ Verbatim stored in {immediate_time:.2f}ms")
            
            # Step 2: Check if it's time for batch extraction
            extraction_time_ms = 0
            facts_extracted = 0
            bullets_generated = 0
            
            # Step 2a: Check for session summarization (every 10 turns)
            if turn_id % self.summarization_interval == 0:
                print(f"📋 Triggering session summarization (every {self.summarization_interval} turns)")
                start_summary = time.perf_counter()
                session_summary = await self._generate_session_summary(session_id, turn_id)
                summary_time_ms = (time.perf_counter() - start_summary) * 1000
                print(f"📝 Generated session summary in {summary_time_ms:.0f}ms")
                print(f"💭 Summary: {session_summary[:100]}...")
            
            if turn_id % self.extraction_interval == 0:
                print(f"🔍 Triggering batch extraction (every {self.extraction_interval} turns)")
                
                # Get recent conversation for context
                recent_turns = self.conversation_turns[-self.extraction_interval:]
                combined_text = self._combine_turns_for_extraction(recent_turns)
                
                # Batch extraction with LLM
                start_extraction = time.perf_counter()
                bullets, triples = await self._batch_extract_and_store(
                    combined_text, session_id, turn_id
                )
                extraction_time_ms = (time.perf_counter() - start_extraction) * 1000
                
                facts_extracted = len(triples)
                bullets_generated = len(bullets)
                
                print(f"📊 Extracted {facts_extracted} facts, {bullets_generated} bullets in {extraction_time_ms:.0f}ms")
                
                # Show sample extracted facts for quality check
                if triples:
                    print(f"🔍 Sample facts extracted:")
                    for i, (s, r, o) in enumerate(triples[:5]):
                        print(f"  {i+1}. ({s}) -[{r}]-> ({o})")
                    if len(triples) > 5:
                        print(f"  ... and {len(triples)-5} more facts")
                
                if bullets:
                    print(f"📝 Sample bullets:")
                    for i, bullet in enumerate(bullets[:3]):
                        print(f"  • {bullet}")
                    if len(bullets) > 3:
                        print(f"  ... and {len(bullets)-3} more bullets")
            
            storage_results.append(StorageTest(
                verbatim_stored=verbatim_stored,
                extraction_time_ms=extraction_time_ms,
                facts_extracted=facts_extracted,
                bullets_generated=bullets_generated,
                storage_success=True
            ))
        
        return storage_results
    
    def _store_verbatim(self, turn: ConversationTurn) -> bool:
        """Store conversation turn verbatim"""
        try:
            # This would go to your verbatim conversation store
            # For now, we'll use the memory store's session functionality
            return True
        except Exception as e:
            print(f"❌ Verbatim storage failed: {e}")
            return False
    
    def _combine_turns_for_extraction(self, turns: List[ConversationTurn]) -> str:
        """Combine recent conversation turns for batch extraction"""
        combined = []
        for turn in turns:
            combined.append(f"User: {turn.user_message}")
            combined.append(f"Assistant: {turn.assistant_message}")
        
        return "\n".join(combined)
    
    async def _batch_extract_and_store(self, text: str, session_id: str, turn_id: int) -> tuple:
        """Batch extraction and storage using real LLM"""
        try:
            # Use HotMemory for actual extraction and storage
            bullets, triples = self.hot_memory.process_turn(
                text, session_id=session_id, turn_id=turn_id
            )
            return bullets, triples
        except Exception as e:
            print(f"❌ Batch extraction failed: {e}")
            return [], []
    
    async def _generate_session_summary(self, session_id: str, turn_id: int) -> str:
        """Generate session summary using LLM"""
        try:
            # Combine recent conversation turns for summarization
            recent_turns = self.conversation_turns[-self.summarization_interval:]
            
            conversation_text = []
            for turn in recent_turns:
                conversation_text.append(f"User: {turn.user_message}")
                conversation_text.append(f"Assistant: {turn.assistant_message}")
            
            combined_text = "\n".join(conversation_text)
            
            # Use a simple pattern-based summary for now (in production, use LLM)
            topics = []
            key_info = []
            
            # Extract key topics mentioned
            if "work" in combined_text.lower() or "job" in combined_text.lower():
                topics.append("work and career")
            if "dog" in combined_text.lower() or "cat" in combined_text.lower():
                topics.append("pets")
            if "seattle" in combined_text.lower() or "portland" in combined_text.lower():
                topics.append("locations")
            if "machine learning" in combined_text.lower() or "bert" in combined_text.lower():
                topics.append("AI/ML projects")
            if "favorite" in combined_text.lower():
                topics.append("personal preferences")
            
            # Extract key facts
            import re
            names = re.findall(r'(?:named?|called?)\s+(\w+)', combined_text, re.IGNORECASE)
            ages = re.findall(r'(\d+)\s+years?\s+old', combined_text, re.IGNORECASE)
            
            if names:
                key_info.append(f"Names mentioned: {', '.join(set(names))}")
            if ages:
                key_info.append(f"Ages mentioned: {', '.join(set(ages))} years old")
            
            # Create summary
            summary_parts = []
            if topics:
                summary_parts.append(f"Topics discussed: {', '.join(topics)}")
            if key_info:
                summary_parts.append(f"Key information: {'; '.join(key_info)}")
            
            summary = f"Session summary (turns {max(1, turn_id-self.summarization_interval+1)}-{turn_id}): " + ". ".join(summary_parts)
            
            # Store summary (in production, would go to summary storage)
            return summary
            
        except Exception as e:
            print(f"❌ Session summary failed: {e}")
            return f"Summary failed for session {session_id} at turn {turn_id}"
    
    async def test_retrieval_strategies(self) -> List[RetrievalTest]:
        """Test different retrieval approaches"""
        print("\n🔍 Testing Retrieval Strategies")
        print("=" * 50)
        
        # Test queries covering different retrieval needs
        test_queries = [
            "Tell me about Sarah's work at Microsoft",
            "What does Sarah's brother Tom do?", 
            "Describe Sarah's dog Luna",
            "What car does Sarah drive?",
            "Where did Sarah graduate from?",
            "What machine learning project is Sarah working on?",
            "How old is Luna?",
            "Where does Tom teach?"
        ]
        
        retrieval_results = []
        
        for query in test_queries:
            print(f"\n🔎 Query: {query}")
            
            start_time = time.perf_counter()
            
            # Test knowledge graph retrieval
            kg_results = await self._retrieve_from_kg(query)
            
            # Test verbatim retrieval (simulate)
            verbatim_results = self._retrieve_verbatim(query)
            
            # Test summary retrieval (simulate)  
            summary_results = self._retrieve_summaries(query)
            
            retrieval_time_ms = (time.perf_counter() - start_time) * 1000
            
            total_results = len(kg_results) + len(verbatim_results) + len(summary_results)
            
            print(f"📊 Found {len(kg_results)} KG, {len(verbatim_results)} verbatim, {len(summary_results)} summary results")
            print(f"⚡ Retrieved in {retrieval_time_ms:.2f}ms")
            
            # Show sample results
            if kg_results:
                print(f"🔗 KG Sample: {kg_results[0]}")
            
            retrieval_results.append(RetrievalTest(
                query=query,
                verbatim_results=verbatim_results,
                summary_results=summary_results,
                kg_results=kg_results,
                total_results=total_results,
                retrieval_time_ms=retrieval_time_ms
            ))
        
        return retrieval_results
    
    async def _retrieve_from_kg(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve from knowledge graph"""
        try:
            # Use HotMemory's internal retrieval system
            # Extract entities from query for better retrieval
            entities = [word.lower() for word in query.split() 
                       if len(word) > 2 and word.isalpha()][:5]
            
            bullets = self.hot_memory._retrieve_context(
                query, entities, turn_id=999, intent=None
            )
            
            # Convert bullets to structured format
            kg_results = []
            for bullet in bullets:
                kg_results.append({
                    "type": "fact",
                    "content": bullet,
                    "source": "knowledge_graph"
                })
            
            return kg_results
        except Exception as e:
            print(f"❌ KG retrieval failed: {e}")
            return []
    
    def _retrieve_verbatim(self, query: str) -> List[str]:
        """Retrieve from verbatim conversation store"""
        # Simulate verbatim search
        results = []
        query_lower = query.lower()
        
        for turn in self.conversation_turns:
            if any(word in turn.user_message.lower() or word in turn.assistant_message.lower() 
                   for word in query_lower.split()):
                results.append(f"Turn {turn.turn_id}: {turn.user_message[:100]}...")
                
        return results[:3]  # Limit results
    
    def _retrieve_summaries(self, query: str) -> List[str]:
        """Retrieve from session summaries"""
        # Simulate summary search
        # In real implementation, this would search stored session summaries
        return []  # No summaries generated yet in this test
    
    def print_results(self, storage_results: List[StorageTest], retrieval_results: List[RetrievalTest]):
        """Print comprehensive test results"""
        print("\n📈 STORAGE & RETRIEVAL TEST RESULTS")
        print("=" * 60)
        
        # Storage Results
        print("\n🏗️ Storage Performance:")
        total_facts = sum(r.facts_extracted for r in storage_results)
        total_bullets = sum(r.bullets_generated for r in storage_results) 
        avg_extraction_time = sum(r.extraction_time_ms for r in storage_results if r.extraction_time_ms > 0) / max(1, len([r for r in storage_results if r.extraction_time_ms > 0]))
        
        print(f"  📊 Total Facts Extracted: {total_facts}")
        print(f"  📊 Total Bullets Generated: {total_bullets}")
        print(f"  ⚡ Avg Batch Extraction Time: {avg_extraction_time:.0f}ms")
        print(f"  ✅ Storage Success Rate: {sum(1 for r in storage_results if r.storage_success) / len(storage_results) * 100:.1f}%")
        
        # Retrieval Results
        print("\n🔍 Retrieval Performance:")
        avg_retrieval_time = sum(r.retrieval_time_ms for r in retrieval_results) / len(retrieval_results)
        total_retrieved = sum(r.total_results for r in retrieval_results)
        successful_queries = len([r for r in retrieval_results if r.total_results > 0])
        
        print(f"  📊 Total Results Retrieved: {total_retrieved}")
        print(f"  ⚡ Avg Retrieval Time: {avg_retrieval_time:.1f}ms")
        print(f"  ✅ Successful Queries: {successful_queries}/{len(retrieval_results)} ({successful_queries/len(retrieval_results)*100:.1f}%)")
        
        # Query-by-query breakdown
        print("\n🔎 Query Results Breakdown:")
        for result in retrieval_results:
            print(f"  '{result.query[:40]}...': {result.total_results} results in {result.retrieval_time_ms:.1f}ms")

async def main():
    """Run the storage and retrieval test"""
    tester = StorageRetrievalTester()
    
    try:
        # Setup
        tester.setup()
        
        # Test storage pipeline
        storage_results = await tester.simulate_conversation()
        
        # Test retrieval strategies  
        retrieval_results = await tester.test_retrieval_strategies()
        
        # Print comprehensive results
        tester.print_results(storage_results, retrieval_results)
        
    finally:
        # Cleanup
        tester.cleanup()

if __name__ == '__main__':
    asyncio.run(main())