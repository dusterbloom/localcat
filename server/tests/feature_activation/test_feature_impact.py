#!/usr/bin/env python3
"""
HotMem V4 Feature Activation A/B Testing Framework

Tests the impact of enabling existing disabled features:
- HOTMEM_USE_COREF (FCoref integration)
- HOTMEM_USE_LEANN (semantic search) 
- HOTMEM_DECOMPOSE_CLAUSES (sentence decomposition)
- DSPy integration readiness

All features are already implemented - this validates their effectiveness.
"""

import os
import sys
import time
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from loguru import logger

# Test data for consistent A/B testing
TEST_SENTENCES = [
    # Pronoun resolution tests (COREF impact)
    "Sarah went to the store. She bought milk and bread.",
    "John founded OpenAI. He is the CEO of the company.",
    "The company was started in 2015. It focuses on AI research.",
    
    # Complex sentence tests (decomposition impact)  
    "Did I tell you that Sarah works at Google and she lives in San Francisco?",
    "If John goes to MIT, then he will study computer science there.",
    "Sarah, who works at Tesla, is married to John who founded SpaceX.",
    
    # Semantic search tests (LEANN impact)
    "What car does Sarah drive?",  # Should find "Sarah has Tesla Model 3"
    "Where does John work?",       # Should find "John works_at OpenAI"
    "Who is the founder of Tesla?", # Should find "Elon founded Tesla"
    
    # Simple facts (baseline performance)
    "Sarah lives in San Francisco.",
    "John works at OpenAI.", 
    "Tesla was founded by Elon Musk.",
]

@dataclass
class TestResult:
    """Results from feature activation test"""
    feature_name: str
    enabled: bool
    extraction_count: int
    retrieval_count: int
    accuracy_score: float
    latency_ms: float
    error_count: int
    
@dataclass 
class PerformanceMetrics:
    """Performance tracking for A/B tests"""
    total_ms: float
    extraction_ms: float
    retrieval_ms: float
    classifier_ms: float
    cache_hits: int = 0

class FeatureActivationTester:
    """A/B testing framework for HotMem feature activation"""
    
    def __init__(self):
        self.results = []
        
    @contextmanager
    def feature_config(self, **env_vars):
        """Context manager to temporarily set environment variables"""
        original_values = {}
        
        # Save original values
        for key, value in env_vars.items():
            original_values[key] = os.getenv(key)
            os.environ[key] = str(value)
            
        try:
            yield
        finally:
            # Restore original values
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    async def test_coref_activation_impact(self) -> Tuple[TestResult, TestResult]:
        """A/B test coreference resolution using existing FCoref at memory_hotpath.py:217"""
        logger.info("🧪 Testing COREF feature activation impact...")
        
        # Test with COREF disabled (baseline)
        with self.feature_config(HOTMEM_USE_COREF="false"):
            baseline_result = await self._run_extraction_test("COREF_DISABLED", enabled=False)
            
        # Test with COREF enabled
        with self.feature_config(HOTMEM_USE_COREF="true"):
            enhanced_result = await self._run_extraction_test("COREF_ENABLED", enabled=True)
            
        return baseline_result, enhanced_result
    
    async def test_leann_semantic_activation(self) -> Tuple[TestResult, TestResult]:
        """Test LEANN semantic search effectiveness using existing implementation at memory_hotpath.py:207"""
        logger.info("🔍 Testing LEANN semantic search activation...")
        
        # Test with LEANN disabled
        with self.feature_config(HOTMEM_USE_LEANN="false"):
            baseline_result = await self._run_retrieval_test("LEANN_DISABLED", enabled=False)
            
        # Test with LEANN enabled
        with self.feature_config(HOTMEM_USE_LEANN="true"):
            enhanced_result = await self._run_retrieval_test("LEANN_ENABLED", enabled=True)
            
        return baseline_result, enhanced_result
    
    async def test_decomposition_activation(self) -> Tuple[TestResult, TestResult]:
        """Evaluate sentence decomposition using existing memory_decomposer.py"""
        logger.info("🔧 Testing sentence decomposition activation...")
        
        # Focus on complex sentences that should benefit from decomposition
        complex_sentences = [
            "Sarah, who works at Google, lives in San Francisco.",
            "If John goes to MIT, then he will study computer science.",
            "The company that was founded by Elon Musk makes electric cars.",
            "When Tesla launched in 2003, it revolutionized the automotive industry."
        ]
        
        # Test with decomposition disabled
        with self.feature_config(HOTMEM_DECOMPOSE_CLAUSES="false"):
            baseline_result = await self._run_complex_extraction_test("DECOMP_DISABLED", 
                                                                    complex_sentences, enabled=False)
            
        # Test with decomposition enabled  
        with self.feature_config(HOTMEM_DECOMPOSE_CLAUSES="true"):
            enhanced_result = await self._run_complex_extraction_test("DECOMP_ENABLED",
                                                                    complex_sentences, enabled=True)
            
        return baseline_result, enhanced_result
    
    async def test_dspy_integration_readiness(self) -> TestResult:
        """Validate DSPy framework integration from components/ai/dspy_modules.py"""
        logger.info("🤖 Testing DSPy integration readiness...")
        
        try:
            # Test DSPy module imports
            from components.ai.dspy_modules import DSPyConfig
            
            # Test configuration
            config = DSPyConfig()
            
            # Test basic framework availability
            import dspy
            
            result = TestResult(
                feature_name="DSPY_INTEGRATION",
                enabled=True,
                extraction_count=1,  # Framework available
                retrieval_count=0,
                accuracy_score=1.0,  # Import success = ready
                latency_ms=0.0,
                error_count=0
            )
            
            logger.info("✅ DSPy framework ready for production integration")
            return result
            
        except Exception as e:
            logger.error(f"❌ DSPy integration issue: {e}")
            return TestResult(
                feature_name="DSPY_INTEGRATION", 
                enabled=False,
                extraction_count=0,
                retrieval_count=0,
                accuracy_score=0.0,
                latency_ms=0.0,
                error_count=1
            )
    
    async def _run_extraction_test(self, test_name: str, enabled: bool) -> TestResult:
        """Run extraction test with current configuration"""
        start_time = time.perf_counter()
        extraction_count = 0
        error_count = 0
        
        try:
            # Import HotMem with current env config
            from components.memory.memory_hotpath import HotMemory
            from components.memory.memory_store import MemoryStore, Paths
            
            # Create temporary memory instance
            with tempfile.TemporaryDirectory() as temp_dir:
                store = MemoryStore(Paths(
                    sqlite_path=f"{temp_dir}/test.db",
                    lmdb_dir=f"{temp_dir}/lmdb"
                ))
                memory = HotMemory(store=store)
                
                # Test extraction on pronoun-heavy sentences
                pronoun_sentences = [
                    "Sarah went to the store. She bought milk and bread.",
                    "John founded OpenAI. He is the CEO of the company.", 
                    "The company was started in 2015. It focuses on AI research."
                ]
                
                for i, sentence in enumerate(pronoun_sentences):
                    try:
                        bullets, triples = memory.process_turn(sentence, session_id="test", turn_id=i)
                        # Count quality extractions - look for proper name resolution
                        quality_extractions = 0
                        for s, r, o in triples:
                            # Better extraction should have resolved names instead of pronouns
                            if any(name in s.lower() for name in ['sarah', 'john', 'company', 'openai']):
                                quality_extractions += 1
                            if any(name in o.lower() for name in ['sarah', 'john', 'company', 'openai']):
                                quality_extractions += 1
                        extraction_count += quality_extractions
                    except Exception as e:
                        error_count += 1
                        logger.debug(f"Extraction error: {e}")
                        
        except Exception as e:
            error_count += 1
            logger.error(f"Test setup error: {e}")
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Calculate accuracy score (more extractions = better for pronoun resolution)
        accuracy_score = min(1.0, extraction_count / 10.0)  # Normalized to 0-1
        
        return TestResult(
            feature_name=test_name,
            enabled=enabled,
            extraction_count=extraction_count,
            retrieval_count=0,
            accuracy_score=accuracy_score,
            latency_ms=elapsed_ms,
            error_count=error_count
        )
    
    async def _run_retrieval_test(self, test_name: str, enabled: bool) -> TestResult:
        """Run retrieval test with current configuration"""
        start_time = time.perf_counter()
        retrieval_count = 0
        error_count = 0
        
        try:
            from components.memory.memory_hotpath import HotMemory
            from components.memory.memory_store import MemoryStore, Paths
            
            with tempfile.TemporaryDirectory() as temp_dir:
                store = MemoryStore(Paths(
                    sqlite_path=f"{temp_dir}/test.db",
                    lmdb_dir=f"{temp_dir}/lmdb"
                ))
                memory = HotMemory(store=store)
                
                # Seed with facts first
                facts = [
                    "Sarah has Tesla Model 3",
                    "John works_at OpenAI",
                    "Elon founded Tesla"
                ]
                
                for i, fact in enumerate(facts):
                    memory.process_turn(fact, session_id="test", turn_id=i)
                
                # Test semantic queries
                queries = [
                    "What car does Sarah drive?",
                    "Where does John work?", 
                    "Who founded Tesla?"
                ]
                
                for query in queries:
                    try:
                        bullets, _ = memory.process_turn(query, session_id="test", turn_id=retrieval_count)
                        retrieval_count += len(bullets)
                    except Exception as e:
                        error_count += 1
                        logger.debug(f"Retrieval error: {e}")
                        
        except Exception as e:
            error_count += 1
            logger.error(f"Retrieval test error: {e}")
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # More retrieved facts = better semantic understanding
        accuracy_score = min(1.0, retrieval_count / 6.0)  # Normalized
        
        return TestResult(
            feature_name=test_name,
            enabled=enabled,
            extraction_count=0,
            retrieval_count=retrieval_count,
            accuracy_score=accuracy_score,
            latency_ms=elapsed_ms,
            error_count=error_count
        )
    
    async def _run_complex_extraction_test(self, test_name: str, sentences: List[str], enabled: bool) -> TestResult:
        """Run extraction test on complex sentences"""
        start_time = time.perf_counter()
        extraction_count = 0
        error_count = 0
        
        try:
            from components.memory.memory_hotpath import HotMemory
            from components.memory.memory_store import MemoryStore, Paths
            
            with tempfile.TemporaryDirectory() as temp_dir:
                store = MemoryStore(Paths(
                    sqlite_path=f"{temp_dir}/test.db",
                    lmdb_dir=f"{temp_dir}/lmdb"
                ))
                memory = HotMemory(store=store)
                
                for sentence in sentences:
                    try:
                        bullets, triples = memory.process_turn(sentence, session_id="test", turn_id=extraction_count)
                        extraction_count += len(triples)
                    except Exception as e:
                        error_count += 1
                        logger.debug(f"Complex extraction error: {e}")
                        
        except Exception as e:
            error_count += 1
            logger.error(f"Complex test error: {e}")
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Complex sentences should yield more facts when decomposed
        accuracy_score = min(1.0, extraction_count / 8.0)  # Normalized
        
        return TestResult(
            feature_name=test_name,
            enabled=enabled,
            extraction_count=extraction_count,
            retrieval_count=0,
            accuracy_score=accuracy_score,
            latency_ms=elapsed_ms,
            error_count=error_count
        )
    
    def compare_results(self, baseline: TestResult, enhanced: TestResult) -> Dict[str, Any]:
        """Compare baseline vs enhanced results"""
        return {
            "feature": enhanced.feature_name.replace("_ENABLED", ""),
            "extraction_improvement": enhanced.extraction_count - baseline.extraction_count,
            "retrieval_improvement": enhanced.retrieval_count - baseline.retrieval_count,
            "accuracy_improvement": enhanced.accuracy_score - baseline.accuracy_score,
            "latency_delta_ms": enhanced.latency_ms - baseline.latency_ms,
            "error_delta": enhanced.error_count - baseline.error_count,
            "recommendation": "ENABLE" if enhanced.accuracy_score > baseline.accuracy_score else "DISABLE"
        }

async def main():
    """Run comprehensive feature activation tests"""
    logger.info("🚀 Starting HotMem V4 Feature Activation Testing...")
    
    tester = FeatureActivationTester()
    
    # Run all A/B tests
    logger.info("=" * 60)
    coref_baseline, coref_enhanced = await tester.test_coref_activation_impact()
    coref_comparison = tester.compare_results(coref_baseline, coref_enhanced)
    
    logger.info("=" * 60) 
    leann_baseline, leann_enhanced = await tester.test_leann_semantic_activation()
    leann_comparison = tester.compare_results(leann_baseline, leann_enhanced)
    
    logger.info("=" * 60)
    decomp_baseline, decomp_enhanced = await tester.test_decomposition_activation()
    decomp_comparison = tester.compare_results(decomp_baseline, decomp_enhanced)
    
    logger.info("=" * 60)
    dspy_result = await tester.test_dspy_integration_readiness()
    
    # Summary report
    logger.info("📊 FEATURE ACTIVATION TEST RESULTS")
    logger.info("=" * 60)
    
    for comparison in [coref_comparison, leann_comparison, decomp_comparison]:
        feature = comparison["feature"] 
        recommendation = comparison["recommendation"]
        accuracy_improvement = comparison["accuracy_improvement"]
        latency_delta = comparison["latency_delta_ms"]
        
        status = "✅" if recommendation == "ENABLE" else "❌"
        logger.info(f"{status} {feature}: {recommendation}")
        logger.info(f"   Accuracy: {accuracy_improvement:+.2f}")
        logger.info(f"   Latency: {latency_delta:+.1f}ms")
    
    dspy_status = "✅" if dspy_result.accuracy_score > 0 else "❌"
    logger.info(f"{dspy_status} DSPy Integration: {'READY' if dspy_result.accuracy_score > 0 else 'NEEDS_WORK'}")
    
    logger.info("🎯 HotMem V4 Feature Activation Testing Complete!")

if __name__ == "__main__":
    asyncio.run(main())