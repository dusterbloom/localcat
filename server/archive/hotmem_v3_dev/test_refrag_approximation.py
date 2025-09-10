#!/usr/bin/env python3
"""
REFRAG Approximation Test with LM Studio
=========================================

Tests compression and attention masking techniques inspired by REFRAG paper
using small compressor models and larger decoder models via LM Studio.

Compressor models tested:
- google/gemma-3-270m
- qwen2.5-0.5b-instruct-mlx
- lfm2-350m

Main model:
- qwen/qwen3-4b
"""

import os
import sys
import time
import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import requests
from dotenv import load_dotenv

# Add server path for LEANN access
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

# Try to import LEANN if available
try:
    from leann_adapter import LEANNAdapter
    LEANN_AVAILABLE = True
except ImportError:
    LEANN_AVAILABLE = False
    print("⚠️  LEANN not available, will use simple similarity instead")

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), 'server', '.env'))

@dataclass
class CompressionResult:
    """Result from chunk compression"""
    chunk_id: int
    original_text: str
    compressed: str
    compression_ratio: float
    relevance_score: float
    should_expand: bool
    time_ms: float

@dataclass
class TestSentence:
    """Test case from complex patterns"""
    text: str
    expected_facts: List[str]
    name: str

# Selected test sentences from complex_patterns.py
TEST_SENTENCES = [
    TestSentence(
        name="Nested Relative Clauses",
        text="The professor who taught the class that I took last semester, whose research focuses on AI ethics, recently published a book about algorithmic bias that became a bestseller.",
        expected_facts=["professor teaches class", "you took class", "research focuses on AI ethics", "professor published book"]
    ),
    TestSentence(
        name="Complex Coordination",
        text="My brother Tom, who lives in Portland and teaches philosophy at Reed College, is writing a book about ethics while simultaneously preparing for his sabbatical in Europe.",
        expected_facts=["Tom lives in Portland", "Tom teaches philosophy", "Tom teaches at Reed College", "Tom is writing book"]
    ),
]

class ApproximateREFRAG:
    """
    Approximate REFRAG implementation using LM Studio models.
    """
    
    def __init__(self, compressor_model: str, main_model: str = "qwen/qwen3-4b"):
        self.compressor_model = compressor_model
        self.main_model = main_model
        self.lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        self.chunk_size = 32  # Tokens per chunk (approximate)
        
        # Initialize LEANN if available
        self.leann = None
        if LEANN_AVAILABLE:
            try:
                self.leann = LEANNAdapter()
                print(f"✅ LEANN initialized for chunk filtering")
            except Exception as e:
                print(f"⚠️  LEANN initialization failed: {e}")
        
        # Cache for compressed chunks
        self.chunk_cache: Dict[str, str] = {}
        self.attention_masks: Dict[str, np.ndarray] = {}
    
    def chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """Split text into chunks"""
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size // 4):  # Approximate word-to-token ratio
            chunk = " ".join(words[i:i + chunk_size // 4])
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def compress_chunk(self, chunk: str, query: str) -> Tuple[str, float]:
        """Compress a chunk using the small model"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{self.compressor_model}:{chunk[:50]}"
        if cache_key in self.chunk_cache:
            return self.chunk_cache[cache_key], 0.0
        
        prompt = f"""Compress this text to its key information relevant to the query.
Query: {query}
Text: {chunk}
Compressed (max 10 words):"""
        
        try:
            response = requests.post(
                f"{self.lm_studio_url}/completions",
                json={
                    "model": self.compressor_model,
                    "prompt": prompt,
                    "max_tokens": 20,
                    "temperature": 0.1,
                    "stop": ["\n"]
                }
            )
            
            if response.status_code == 200:
                compressed = response.json()["choices"][0]["text"].strip()
                self.chunk_cache[cache_key] = compressed
                elapsed = (time.time() - start_time) * 1000
                return compressed, elapsed
            else:
                print(f"⚠️  Compression failed: {response.status_code}")
                return chunk[:50], 0.0
                
        except Exception as e:
            print(f"⚠️  Compression error: {e}")
            return chunk[:50], 0.0
    
    def calculate_relevance(self, chunk: str, query: str) -> float:
        """Calculate chunk relevance using LEANN or simple similarity"""
        if self.leann:
            try:
                # Use LEANN for semantic similarity
                results = self.leann.search(query, k=1, text_corpus=[chunk])
                if results:
                    return results[0].get('score', 0.5)
            except:
                pass
        
        # Fallback: simple keyword overlap
        query_words = set(query.lower().split())
        chunk_words = set(chunk.lower().split())
        if not query_words:
            return 0.0
        overlap = len(query_words & chunk_words) / len(query_words)
        return min(overlap, 1.0)
    
    def create_attention_mask(self, chunks: List[str], relevance_scores: List[float]) -> np.ndarray:
        """Create attention mask based on relevance scores"""
        n = len(chunks)
        mask = np.zeros((n, n))
        
        # Allow attention to highly relevant chunks
        threshold = np.percentile(relevance_scores, 60)  # Top 40% chunks
        
        for i in range(n):
            for j in range(n):
                # Self-attention always allowed
                if i == j:
                    mask[i, j] = 1.0
                # Cross-attention based on relevance
                elif relevance_scores[i] > threshold or relevance_scores[j] > threshold:
                    mask[i, j] = 0.8
                # Sparse attention for less relevant chunks
                else:
                    mask[i, j] = 0.2
        
        return mask
    
    async def process_with_refrag(self, text: str, query: str) -> Dict[str, Any]:
        """Process text using approximate REFRAG technique"""
        print(f"\n🔄 Processing with {self.compressor_model}")
        print(f"📝 Query: {query}")
        
        start_time = time.time()
        
        # Step 1: Chunk the text
        chunks = self.chunk_text(text)
        print(f"📦 Created {len(chunks)} chunks")
        
        # Step 2: Calculate relevance and compress
        compression_results = []
        total_compression_time = 0.0
        
        for i, chunk in enumerate(chunks):
            relevance = self.calculate_relevance(chunk, query)
            
            # Decide whether to compress based on relevance
            should_compress = relevance < 0.5  # Compress less relevant chunks
            
            if should_compress:
                compressed, comp_time = self.compress_chunk(chunk, query)
                total_compression_time += comp_time
            else:
                compressed = chunk  # Keep full resolution
                comp_time = 0
            
            compression_results.append(CompressionResult(
                chunk_id=i,
                original_text=chunk,
                compressed=compressed,
                compression_ratio=len(compressed) / len(chunk) if chunk else 1.0,
                relevance_score=relevance,
                should_expand=not should_compress,
                time_ms=comp_time
            ))
        
        # Step 3: Create attention mask
        relevance_scores = [r.relevance_score for r in compression_results]
        attention_mask = self.create_attention_mask(chunks, relevance_scores)
        
        # Step 4: Build context for main model
        context_parts = []
        for result in compression_results:
            if result.should_expand:
                context_parts.append(f"[FULL] {result.original_text}")
            else:
                context_parts.append(f"[COMPRESSED] {result.compressed}")
        
        compressed_context = "\n".join(context_parts)
        
        # Step 5: Generate with main model
        print(f"\n🤖 Generating with {self.main_model}")
        
        final_prompt = f"""Context (with compression markers):
{compressed_context}

Query: {query}

Extract the key facts from the context:"""
        
        try:
            response = requests.post(
                f"{self.lm_studio_url}/completions",
                json={
                    "model": self.main_model,
                    "prompt": final_prompt,
                    "max_tokens": 200,
                    "temperature": 0.3
                }
            )
            
            if response.status_code == 200:
                extraction = response.json()["choices"][0]["text"].strip()
            else:
                extraction = f"Error: {response.status_code}"
                
        except Exception as e:
            extraction = f"Error: {e}"
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_compression = np.mean([r.compression_ratio for r in compression_results])
        compressed_chunks = sum(1 for r in compression_results if not r.should_expand)
        
        return {
            "compressor_model": self.compressor_model,
            "main_model": self.main_model,
            "extraction": extraction,
            "stats": {
                "total_chunks": len(chunks),
                "compressed_chunks": compressed_chunks,
                "avg_compression_ratio": avg_compression,
                "total_time_s": total_time,
                "compression_time_ms": total_compression_time,
                "context_reduction": len(compressed_context) / len(text),
                "attention_sparsity": 1.0 - np.mean(attention_mask)
            },
            "compression_results": compression_results,
            "attention_mask_sample": attention_mask[:5, :5].tolist() if len(attention_mask) > 0 else []
        }

async def run_comparison_tests():
    """Run comparison tests with different compressor models"""
    
    compressor_models = [
        "google/gemma-3-270m",
        "qwen2.5-0.5b-instruct-mlx", 
        "lfm2-350m"
    ]
    
    print("=" * 80)
    print("REFRAG Approximation Tests")
    print("=" * 80)
    
    # Test with different compressors
    for sentence in TEST_SENTENCES:
        print(f"\n\n🧪 Testing: {sentence.name}")
        print(f"📄 Text: {sentence.text[:100]}...")
        print(f"✅ Expected facts: {', '.join(sentence.expected_facts)}")
        
        results = []
        
        for compressor in compressor_models:
            refrag = ApproximateREFRAG(compressor_model=compressor)
            
            # Create a query based on expected facts
            query = "Extract information about people, their actions, and relationships"
            
            result = await refrag.process_with_refrag(sentence.text, query)
            results.append(result)
            
            # Print results
            print(f"\n📊 Results for {compressor}:")
            print(f"  - Chunks: {result['stats']['total_chunks']} total, {result['stats']['compressed_chunks']} compressed")
            print(f"  - Compression ratio: {result['stats']['avg_compression_ratio']:.2%}")
            print(f"  - Context reduction: {result['stats']['context_reduction']:.2%}")
            print(f"  - Attention sparsity: {result['stats']['attention_sparsity']:.2%}")
            print(f"  - Total time: {result['stats']['total_time_s']:.2f}s")
            print(f"  - Extraction preview: {result['extraction'][:150]}...")
        
        # Compare results
        print("\n📈 Comparison Summary:")
        print("-" * 60)
        print(f"{'Model':<30} {'Time(s)':<10} {'Compression':<12} {'Sparsity':<10}")
        print("-" * 60)
        for r in results:
            model = r['compressor_model'].split('/')[-1][:28]
            print(f"{model:<30} {r['stats']['total_time_s']:<10.2f} {r['stats']['context_reduction']:<12.2%} {r['stats']['attention_sparsity']:<10.2%}")

async def test_with_leann_filtering():
    """Test LEANN-based chunk filtering"""
    print("\n" + "=" * 80)
    print("LEANN-Based Chunk Filtering Test")
    print("=" * 80)
    
    if not LEANN_AVAILABLE:
        print("⚠️  LEANN not available, skipping test")
        return
    
    sentence = TEST_SENTENCES[0]
    refrag = ApproximateREFRAG(compressor_model="qwen2.5-0.5b-instruct-mlx")
    
    # Test with a specific query
    query = "What did the professor do?"
    
    print(f"\n📝 Query: {query}")
    print(f"📄 Text: {sentence.text}")
    
    result = await refrag.process_with_refrag(sentence.text, query)
    
    print("\n🔍 Chunk Relevance Analysis:")
    for cr in result['compression_results'][:5]:  # Show first 5 chunks
        status = "EXPANDED" if cr.should_expand else "COMPRESSED"
        print(f"  Chunk {cr.chunk_id}: relevance={cr.relevance_score:.2f} [{status}]")
        print(f"    Original: {cr.original_text[:50]}...")
        if not cr.should_expand:
            print(f"    Compressed: {cr.compressed}")
    
    print(f"\n📊 Attention Mask Sample (5x5):")
    if result['attention_mask_sample']:
        for row in result['attention_mask_sample']:
            print("  " + " ".join(f"{val:.1f}" for val in row))

async def main():
    """Main test runner"""
    
    # Check LM Studio connection
    try:
        lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        response = requests.get(f"{lm_studio_url}/models")
        if response.status_code == 200:
            models = response.json().get("data", [])
            print(f"✅ Connected to LM Studio with {len(models)} models")
            print(f"   Available models: {', '.join(m['id'] for m in models[:3])}...")
        else:
            print(f"⚠️  LM Studio connection issue: {response.status_code}")
            print("   Make sure LM Studio is running and models are loaded")
            return
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio: {e}")
        print("   Please start LM Studio and load the required models")
        return
    
    # Run tests
    await run_comparison_tests()
    await test_with_leann_filtering()
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())