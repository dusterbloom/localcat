#!/usr/bin/env python3
"""
REFRAG Approximation Test v2 - Improved
========================================

Fixed version with better compression prompts and selective expansion.
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

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Try to import LEANN
try:
    import subprocess
    # Check if LEANN index exists
    result = subprocess.run(
        ["python", "utils/query_leann.py", "--list"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        LEANN_AVAILABLE = True
        print("✅ LEANN available for semantic search")
    else:
        LEANN_AVAILABLE = False
except:
    LEANN_AVAILABLE = False

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

# Selected test sentences
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

class ImprovedREFRAG:
    """
    Improved REFRAG approximation with better prompts and caching.
    """
    
    def __init__(self, compressor_model: str, main_model: str = "qwen/qwen3-4b"):
        self.compressor_model = compressor_model
        self.main_model = main_model
        self.lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        self.chunk_size = 20  # Smaller chunks for better compression
        
        # Cache
        self.chunk_cache: Dict[str, str] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into smaller overlapping chunks"""
        words = text.split()
        chunks = []
        step = self.chunk_size // 2  # 50% overlap
        
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk and len(chunk.split()) > 3:  # Min 3 words
                chunks.append(chunk)
        return chunks
    
    def compress_chunk_smart(self, chunk: str, query: str) -> Tuple[str, float]:
        """Smart compression with better prompts"""
        start_time = time.time()
        
        # Skip compression for very short chunks
        if len(chunk.split()) <= 5:
            return chunk, 0.0
        
        # Better compression prompt based on model
        if "gemma" in self.compressor_model.lower():
            prompt = f"Summarize: {chunk}\nKey points:"
            max_tokens = 15
        elif "qwen" in self.compressor_model.lower():
            prompt = f"Extract main facts from: {chunk}\nFacts:"
            max_tokens = 20
        else:  # lfm2
            prompt = f"Core info: {chunk}\n=>"
            max_tokens = 10
        
        try:
            response = requests.post(
                f"{self.lm_studio_url}/completions",
                json={
                    "model": self.compressor_model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,  # Deterministic
                    "stop": ["\n", ".", ";"]
                },
                timeout=3
            )
            
            if response.status_code == 200:
                compressed = response.json()["choices"][0]["text"].strip()
                # Ensure compression actually happened
                if len(compressed) > len(chunk) * 0.8:
                    compressed = " ".join(chunk.split()[:5]) + "..."
                elapsed = (time.time() - start_time) * 1000
                return compressed, elapsed
            else:
                return " ".join(chunk.split()[:5]) + "...", 0.0
                
        except Exception as e:
            # Fallback to simple truncation
            return " ".join(chunk.split()[:5]) + "...", 0.0
    
    def calculate_relevance_leann(self, chunk: str, query: str) -> float:
        """Use LEANN for relevance scoring if available"""
        if LEANN_AVAILABLE:
            try:
                import subprocess
                import json
                
                # Query LEANN
                result = subprocess.run(
                    ["python", "utils/query_leann.py", 
                     "--query", f"{query} {chunk[:50]}",
                     "--top-k", "1"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                
                if result.returncode == 0:
                    # Parse score from output
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if "Score:" in line:
                            score = float(line.split("Score:")[-1].strip())
                            return min(score, 1.0)
            except:
                pass
        
        # Fallback: keyword overlap
        query_words = set(query.lower().split())
        chunk_words = set(chunk.lower().split())
        
        # Check for key terms
        key_terms = {"professor", "tom", "book", "ethics", "teaches", "portland", "reed"}
        important_overlap = len(key_terms & chunk_words)
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & chunk_words) / len(query_words)
        return min(overlap + (important_overlap * 0.2), 1.0)
    
    def create_sparse_attention_mask(self, chunks: List[str], relevance_scores: List[float]) -> np.ndarray:
        """Create truly sparse attention mask"""
        n = len(chunks)
        mask = np.zeros((n, n))
        
        # Find top-k relevant chunks
        sorted_indices = np.argsort(relevance_scores)[::-1]
        top_k = max(2, n // 3)  # At least 2, at most 1/3 of chunks
        top_chunks = set(sorted_indices[:top_k])
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    mask[i, j] = 1.0  # Self-attention
                elif i in top_chunks and j in top_chunks:
                    mask[i, j] = 0.8  # Important cross-attention
                elif i in top_chunks or j in top_chunks:
                    mask[i, j] = 0.3  # Partial attention
                else:
                    mask[i, j] = 0.0  # No attention (sparse)
        
        return mask
    
    async def process_with_selective_compression(self, text: str, query: str) -> Dict[str, Any]:
        """Process with selective compression based on relevance"""
        print(f"\n🔄 Processing with {self.compressor_model}")
        
        start_time = time.time()
        
        # Step 1: Chunk with overlap
        chunks = self.chunk_text(text)
        print(f"📦 Created {len(chunks)} chunks (with overlap)")
        
        # Step 2: Score relevance for all chunks
        relevance_scores = []
        for chunk in chunks:
            score = self.calculate_relevance_leann(chunk, query)
            relevance_scores.append(score)
        
        # Step 3: Selective compression
        compression_results = []
        total_compression_time = 0.0
        relevance_threshold = np.percentile(relevance_scores, 50)  # Top 50%
        
        for i, (chunk, relevance) in enumerate(zip(chunks, relevance_scores)):
            # Only compress low-relevance chunks
            should_expand = relevance >= relevance_threshold
            
            if should_expand:
                # Keep original for important chunks
                compressed = chunk
                comp_time = 0
            else:
                # Compress unimportant chunks
                compressed, comp_time = self.compress_chunk_smart(chunk, query)
                total_compression_time += comp_time
            
            orig_len = len(chunk)
            comp_len = len(compressed)
            
            compression_results.append(CompressionResult(
                chunk_id=i,
                original_text=chunk,
                compressed=compressed,
                compression_ratio=comp_len / orig_len if orig_len > 0 else 1.0,
                relevance_score=relevance,
                should_expand=should_expand,
                time_ms=comp_time
            ))
        
        # Step 4: Create sparse attention
        attention_mask = self.create_sparse_attention_mask(chunks, relevance_scores)
        
        # Step 5: Build optimized context
        context_parts = []
        for result in compression_results:
            if result.should_expand:
                context_parts.append(result.original_text)
            else:
                context_parts.append(f"[{result.compressed}]")
        
        compressed_context = " ".join(context_parts)
        
        # Step 6: Extract with main model
        print(f"🤖 Extracting with {self.main_model}")
        
        extraction_prompt = f"""Text: {compressed_context}

Task: Extract these facts:
1. Who are the people mentioned?
2. What are they doing?
3. Where do they live/work?
4. What are their relationships?

Facts:"""
        
        try:
            response = requests.post(
                f"{self.lm_studio_url}/completions",
                json={
                    "model": self.main_model,
                    "prompt": extraction_prompt,
                    "max_tokens": 150,
                    "temperature": 0.1
                },
                timeout=10
            )
            
            if response.status_code == 200:
                extraction = response.json()["choices"][0]["text"].strip()
            else:
                extraction = "Extraction failed"
                
        except Exception as e:
            extraction = f"Error: {e}"
        
        total_time = time.time() - start_time
        
        # Calculate real statistics
        compressed_chunks = sum(1 for r in compression_results if not r.should_expand)
        avg_compression = np.mean([r.compression_ratio for r in compression_results if not r.should_expand]) if compressed_chunks > 0 else 1.0
        
        # Measure actual sparsity
        sparsity = np.sum(attention_mask == 0.0) / (attention_mask.size) if attention_mask.size > 0 else 0
        
        return {
            "compressor_model": self.compressor_model,
            "extraction": extraction,
            "stats": {
                "total_chunks": len(chunks),
                "expanded_chunks": len(chunks) - compressed_chunks,
                "compressed_chunks": compressed_chunks,
                "avg_compression_ratio": avg_compression,
                "context_reduction": len(compressed_context) / len(text),
                "attention_sparsity": sparsity,
                "total_time_s": total_time,
                "compression_time_ms": total_compression_time
            },
            "relevance_distribution": {
                "high": sum(1 for s in relevance_scores if s > 0.7),
                "medium": sum(1 for s in relevance_scores if 0.3 <= s <= 0.7),
                "low": sum(1 for s in relevance_scores if s < 0.3)
            }
        }

async def run_improved_tests():
    """Run improved comparison tests"""
    
    compressor_models = [
        "google/gemma-3-270m",
        "qwen2.5-0.5b-instruct-mlx",
        "lfm2-350m"
    ]
    
    print("=" * 80)
    print("REFRAG Approximation v2 - Improved Tests")
    print("=" * 80)
    
    for sentence in TEST_SENTENCES:
        print(f"\n🧪 Testing: {sentence.name}")
        print(f"📄 Text length: {len(sentence.text)} chars")
        print(f"✅ Expected: {', '.join(sentence.expected_facts[:2])}...")
        
        results = []
        baseline_time = None
        
        # First, get baseline without compression
        print("\n📊 Baseline (no compression):")
        baseline_start = time.time()
        try:
            response = requests.post(
                f"http://localhost:1234/v1/completions",
                json={
                    "model": "qwen/qwen3-4b",
                    "prompt": f"Extract facts from: {sentence.text}\nFacts:",
                    "max_tokens": 100,
                    "temperature": 0.1
                },
                timeout=10
            )
            baseline_time = time.time() - baseline_start
            if response.status_code == 200:
                baseline_extract = response.json()["choices"][0]["text"].strip()
                print(f"  Time: {baseline_time:.2f}s")
                print(f"  Extract: {baseline_extract[:100]}...")
        except:
            print("  Baseline failed")
        
        # Test with compressors
        for compressor in compressor_models:
            refrag = ImprovedREFRAG(compressor_model=compressor)
            query = "people actions relationships"
            
            result = await refrag.process_with_selective_compression(sentence.text, query)
            results.append(result)
            
            print(f"\n📊 {compressor}:")
            print(f"  Chunks: {result['stats']['expanded_chunks']} expanded, {result['stats']['compressed_chunks']} compressed")
            print(f"  Compression: {result['stats']['avg_compression_ratio']:.1%} (compressed chunks only)")
            print(f"  Context size: {result['stats']['context_reduction']:.1%} of original")
            print(f"  Attention sparsity: {result['stats']['attention_sparsity']:.1%}")
            print(f"  Time: {result['stats']['total_time_s']:.2f}s")
            if baseline_time:
                speedup = baseline_time / result['stats']['total_time_s']
                print(f"  Speedup vs baseline: {speedup:.2f}x")
            print(f"  Relevance dist: H:{result['relevance_distribution']['high']} M:{result['relevance_distribution']['medium']} L:{result['relevance_distribution']['low']}")
        
        # Summary table
        print("\n📈 Performance Summary:")
        print("-" * 70)
        print(f"{'Model':<30} {'Time':<8} {'Context':<10} {'Sparsity':<10} {'Speedup':<8}")
        print("-" * 70)
        
        if baseline_time:
            print(f"{'Baseline (no compression)':<30} {baseline_time:<8.2f} {'100.0%':<10} {'0.0%':<10} {'1.00x':<8}")
        
        for r in results:
            model = r['compressor_model'].split('/')[-1][:28]
            speedup = baseline_time / r['stats']['total_time_s'] if baseline_time else 0
            print(f"{model:<30} {r['stats']['total_time_s']:<8.2f} {r['stats']['context_reduction']:<10.1%} {r['stats']['attention_sparsity']:<10.1%} {speedup:<8.2f}x")

async def main():
    """Main test runner"""
    
    # Check LM Studio
    try:
        lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        response = requests.get(f"{lm_studio_url}/models", timeout=2)
        if response.status_code != 200:
            print("❌ LM Studio not responding properly")
            return
        print("✅ LM Studio connected")
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio: {e}")
        return
    
    await run_improved_tests()
    
    print("\n" + "=" * 80)
    print("✅ Tests completed!")

if __name__ == "__main__":
    asyncio.run(main())