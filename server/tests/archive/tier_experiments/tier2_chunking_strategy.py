#!/usr/bin/env python3
"""
Chunking Strategy for Tier2 - Optimized for production voice assistant
Handles sentences with many entities by splitting into manageable chunks
"""

import re
import time
from typing import List, Tuple
from components.extraction.tiered_extractor import TieredRelationExtractor

class Tier2Chunker:
    """Intelligent chunking for Tier2 to handle entity overload"""
    
    def __init__(self, max_entities_per_chunk: int = 6):
        self.max_entities = max_entities_per_chunk
        self.extractor = TieredRelationExtractor()
        
    def estimate_entities(self, text: str) -> int:
        """Rough estimate of entity count in text"""
        # Simple heuristics for entity estimation
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        org_indicators = len(re.findall(r'\b(company|corporation|inc|llc|university|institute)\b', text, re.I))
        person_indicators = len(re.findall(r'\b(dr|mr|mrs|ms|prof)\.?\s', text, re.I))
        
        return proper_nouns + org_indicators + person_indicators
    
    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks based on entity count and structure"""
        estimated_entities = self.estimate_entities(text)
        
        if estimated_entities <= self.max_entities:
            return [text]
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_entities = 0
        
        for sentence in sentences:
            sentence_entities = self.estimate_entities(sentence)
            
            if current_entities + sentence_entities <= self.max_entities:
                current_chunk += sentence + ". "
                current_entities += sentence_entities
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
                current_entities = sentence_entities
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract_with_chunking(self, text: str) -> Tuple[List[Tuple[str, str, str]], dict]:
        """Extract relationships with intelligent chunking"""
        start_time = time.perf_counter()
        
        chunks = self.split_into_chunks(text)
        all_relationships = []
        chunk_results = []
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.perf_counter()
            
            try:
                result = self.extractor._extract_tier2(chunk)
                chunk_time = (time.perf_counter() - chunk_start) * 1000
                
                chunk_results.append({
                    'chunk_num': i + 1,
                    'chunk_text': chunk[:100] + '...' if len(chunk) > 100 else chunk,
                    'relationships': len(result.relationships),
                    'entities': len(result.entities),
                    'time_ms': chunk_time
                })
                
                all_relationships.extend(result.relationships)
                
            except Exception as e:
                chunk_results.append({
                    'chunk_num': i + 1,
                    'chunk_text': chunk[:100] + '...' if len(chunk) > 100 else chunk,
                    'error': str(e),
                    'time_ms': (time.perf_counter() - chunk_start) * 1000
                })
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Remove duplicates (simplified approach)
        unique_relationships = list(dict.fromkeys(all_relationships))
        
        stats = {
            'total_chunks': len(chunks),
            'total_relationships': len(all_relationships),
            'unique_relationships': len(unique_relationships),
            'total_time_ms': total_time,
            'avg_chunk_time_ms': total_time / len(chunks) if chunks else 0,
            'chunks': chunk_results
        }
        
        return unique_relationships, stats
    
    def test_chunking_strategy(self):
        """Test the chunking strategy on various complex texts"""
        
        test_cases = [
            {
                "name": "Simple (no chunking needed)",
                "text": "Alice works at Tesla as an engineer.",
                "expected_chunks": 1
            },
            {
                "name": "Medium (no chunking needed)",
                "text": "Dr. Sarah Chen joined OpenAI as research director in 2021.",
                "expected_chunks": 1
            },
            {
                "name": "Complex (chunking beneficial)",
                "text": "Dr. Sarah Chen, who joined OpenAI as research director in 2021, previously worked at Google Brain where she collaborated with Dr. Fei-Fei Li on computer vision projects before moving to Stanford University to teach machine learning courses.",
                "expected_chunks": 2
            },
            {
                "name": "Very Complex (chunking essential)",
                "text": "Microsoft Corporation, founded by Bill Gates and Paul Allen in 1975, acquired LinkedIn Corporation for $26.2 billion in 2016 under the leadership of CEO Satya Nadella who had previously worked at Sun Microsystems before joining Microsoft in 1992 where he replaced Steve Ballmer and led the company's transformation to cloud computing with products like Azure and Office 365 while competing with Amazon Web Services and Google Cloud Platform.",
                "expected_chunks": 3
            }
        ]
        
        print("🧩 TIER2 CHUNKING STRATEGY TEST")
        print("=" * 80)
        
        for test_case in test_cases:
            print(f"\n📝 {test_case['name']}")
            print(f"Text: {test_case['text'][:120]}...")
            print(f"Expected chunks: {test_case['expected_chunks']}")
            print("-" * 60)
            
            # Test chunking
            chunks = self.split_into_chunks(test_case['text'])
            print(f"Actual chunks: {len(chunks)}")
            
            for i, chunk in enumerate(chunks):
                entities = self.estimate_entities(chunk)
                print(f"  Chunk {i+1}: {entities} entities - {chunk[:80]}...")
            
            # Test extraction with chunking
            print(f"\n🔍 Extraction Results:")
            relationships, stats = self.extract_with_chunking(test_case['text'])
            
            print(f"  ⏱️  Total time: {stats['total_time_ms']:.0f}ms")
            print(f"  📊 Total relationships: {stats['total_relationships']}")
            print(f"  🎯 Unique relationships: {stats['unique_relationships']}")
            print(f"  📈 Avg chunk time: {stats['avg_chunk_time_ms']:.0f}ms")
            
            for chunk_result in stats['chunks']:
                if 'error' in chunk_result:
                    print(f"  ❌ Chunk {chunk_result['chunk_num']}: {chunk_result['error']}")
                else:
                    print(f"  ✅ Chunk {chunk_result['chunk_num']}: {chunk_result['relationships']} relations, {chunk_result['entities']} entities ({chunk_result['time_ms']:.0f}ms)")
        
        # Performance comparison
        print(f"\n🏆 CHUNKING vs NO CHUNKING COMPARISON")
        print("=" * 80)
        
        complex_case = test_cases[3]  # Very complex case
        
        # Test without chunking
        print(f"\n📊 Without Chunking:")
        try:
            start = time.perf_counter()
            result = self.extractor._extract_tier2(complex_case['text'])
            no_chunk_time = (time.perf_counter() - start) * 1000
            print(f"   Time: {no_chunk_time:.0f}ms")
            print(f"   Relationships: {len(result.relationships)}")
            print(f"   Entities: {len(result.entities)}")
        except Exception as e:
            no_chunk_time = float('inf')
            print(f"   Failed: {e}")
        
        # Test with chunking
        print(f"\n📊 With Chunking:")
        relationships, stats = self.extract_with_chunking(complex_case['text'])
        print(f"   Time: {stats['total_time_ms']:.0f}ms")
        print(f"   Relationships: {stats['unique_relationships']}")
        print(f"   Chunks: {stats['total_chunks']}")
        
        if no_chunk_time != float('inf'):
            speedup = no_chunk_time / stats['total_time_ms']
            print(f"   🚀 Speedup: {speedup:.1f}x")
        
        return relationships, stats

if __name__ == "__main__":
    chunker = Tier2Chunker()
    results = chunker.test_chunking_strategy()