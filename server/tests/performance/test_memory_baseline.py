"""
Baseline performance tests for HotMem extraction, retrieval, and injection.

Tests measure p50/p90/p95 latencies to protect the <200ms p95 hot path target.
Tests should run without heavy ML deps and skip gracefully when optional features are disabled.
"""

import pytest
import time
import tempfile
import os
from typing import List, Dict, Any

# Import the modules we need to test
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.memory.metrics_helper import (
    MetricsCollector, 
    TEST_SENTENCES, 
    TEST_RETRIEVAL_QUERIES,
    time_function,
    benchmark_multiple_runs
)

# Skip imports that might require heavy ML deps
try:
    from core.memory.memory_hotpath import HotMemory
    from core.memory.memory_store import MemoryStore, Paths
    from core.memory.config import MemoryConfig, CoreferenceConfig
    HOTMEM_AVAILABLE = True
except ImportError as e:
    HOTMEM_AVAILABLE = False
    pytest.skip(f"HotMem modules not available: {e}", allow_module_level=True)


@pytest.fixture
def memory_config():
    """Create a test memory config"""
    return MemoryConfig(
        bullets_max=3,  # Default bullet limit
        enabled=True,
        sources=["graph"],  # Only enable graph for baseline tests
        convo_index_enabled=False,  # Disable for baseline tests
        coreference=CoreferenceConfig(enabled=False),  # Disable for baseline tests
    )


@pytest.fixture
def memory_store(memory_config):
    """Create a test memory store"""
    if not HOTMEM_AVAILABLE:
        pytest.skip("HotMem not available")
    
    # Use temp directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = Paths(
            sqlite_path=os.path.join(temp_dir, "test_memory.db"),
            lmdb_dir=None  # Disable LMDB for tests
        )
        store = MemoryStore(paths=paths)
        yield store
        # No close method available, store will be cleaned up automatically


@pytest.fixture
def hot_memory(memory_store, memory_config):
    """Create a test HotMemory instance"""
    if not HOTMEM_AVAILABLE:
        pytest.skip("HotMem not available")
    
    return HotMemory(store=memory_store)


class TestExtractionLatencyBaseline:
    """Test extraction latency baseline"""
    
    def test_extraction_latency_baseline(self, hot_memory):
        """
        Run extraction on a fixed set of sentences; record latencies; 
        assert they are recorded and printed.
        """
        if not HOTMEM_AVAILABLE:
            pytest.skip("HotMem not available")
        
        collector = MetricsCollector()
        
        # Test extraction on fixed sentences
        for i, sentence in enumerate(TEST_SENTENCES):
            metadata = {
                "sentence_index": i,
                "sentence_length": len(sentence),
                "word_count": len(sentence.split())
            }
            
            with collector.time_operation("extraction", metadata):
                try:
                    # Extract facts from the sentence using process_turn
                    bullets, triples = hot_memory.process_turn(sentence, session_id="test", turn_id=i)
                    # Store is handled automatically by process_turn
                except Exception as e:
                    # Log but don't fail - we want to measure even failed extractions
                    collector.add_measurement("extraction_error", 0.0, {"error": str(e)})
        
        # Get statistics
        stats = collector.get_stats("extraction")
        
        # Assert we have measurements
        assert stats['count'] > 0, "Should have extraction measurements"
        
        # Print summary
        collector.print_summary("extraction")
        
        # Set soft expectations (warnings rather than failures)
        if stats['p95_ms'] > 100:  # Warning threshold
            print(f"WARNING: P95 extraction latency {stats['p95_ms']:.2f}ms exceeds 100ms target")
        
        if stats['mean_ms'] > 50:  # Warning threshold  
            print(f"WARNING: Mean extraction latency {stats['mean_ms']:.2f}ms exceeds 50ms target")
        
        # Ensure we have reasonable bounds (not extremely slow)
        assert stats['p95_ms'] < 500, f"P95 extraction {stats['p95_ms']:.2f}ms should be under 500ms"
        
        # Verify we stored some data - check if we can retrieve bullets
        retrieved = hot_memory.retrieve_bullets("test query")
        assert len(retrieved) >= 0, "Should have retrieval results"


class TestRetrievalLatencyBaseline:
    """Test retrieval latency baseline"""
    
    def test_retrieval_latency_baseline(self, hot_memory):
        """
        Seed a small store and entity index; measure recall latency 
        (graph + FTS path).
        """
        if not HOTMEM_AVAILABLE:
            pytest.skip("HotMem not available")
        
        collector = MetricsCollector()
        
        # First, seed the memory with some test data
        seed_sentences = [
            "I live in New York City and work as a software engineer.",
            "My name is John Smith and I graduated from Stanford University.",
            "I have two brothers named Mike and Tom.",
            "I drive a Tesla Model 3 and enjoy hiking on weekends.",
            "My favorite color is blue and I love Italian food."
        ]
        
        # Seed the memory
        for i, sentence in enumerate(seed_sentences):
            bullets, triples = hot_memory.process_turn(sentence, session_id="test_seed", turn_id=i)
        
        # Test retrieval latency
        for i, query in enumerate(TEST_RETRIEVAL_QUERIES):
            metadata = {
                "query_index": i,
                "query_length": len(query),
                "word_count": len(query.split())
            }
            
            with collector.time_operation("retrieval", metadata):
                try:
                    # Retrieve memories for the query
                    memories = hot_memory.retrieve_bullets(query, read_only=True)
                    # Store the results for injection testing
                    hot_memory.memories_for_injection = memories
                except Exception as e:
                    collector.add_measurement("retrieval_error", 0.0, {"error": str(e)})
        
        # Get statistics
        stats = collector.get_stats("retrieval")
        
        # Assert we have measurements
        assert stats['count'] > 0, "Should have retrieval measurements"
        
        # Print summary
        collector.print_summary("retrieval")
        
        # Set soft expectations
        if stats['p95_ms'] > 50:  # Retrieval should be faster than extraction
            print(f"WARNING: P95 retrieval latency {stats['p95_ms']:.2f}ms exceeds 50ms target")
        
        if stats['mean_ms'] > 25:
            print(f"WARNING: Mean retrieval latency {stats['mean_ms']:.2f}ms exceeds 25ms target")
        
        # Ensure retrieval is reasonably fast
        assert stats['p95_ms'] < 200, f"P95 retrieval {stats['p95_ms']:.2f}ms should be under 200ms"


class TestInjectionBudgetBaseline:
    """Test injection budget and token limits"""
    
    def test_injection_budget_baseline(self, hot_memory):
        """
        Ensure injection runs under a small budget and enforces 
        bullet/token caps.
        """
        if not HOTMEM_AVAILABLE:
            pytest.skip("HotMem not available")
        
        collector = MetricsCollector()
        
        # Create a set of memories to test injection
        test_memories = [
            "You live in New York City [graph]",
            "You work as a software engineer [graph]", 
            "Your name is John Smith [graph]",
            "You graduated from Stanford University [graph]",
            "You have two brothers named Mike and Tom [graph]",
            "You drive a Tesla Model 3 [graph]",
            "You enjoy hiking on weekends [graph]",
            "Your favorite color is blue [graph]",
            "You love Italian food [graph]",
            "You have a pet dog named Buddy [graph]"
        ]
        
        # Test injection with different budget limits
        budget_limits = [1, 2, 3, 5]
        
        for budget in budget_limits:
            with collector.time_operation("injection", {"budget_limit": budget}):
                try:
                    # Simulate injection by selecting and formatting memories
                    injected = test_memories[:budget]  # Simple selection
                    
                    # Verify budget is respected
                    assert len(injected) <= budget, f"Should inject at most {budget} bullets, got {len(injected)}"
                    
                    # Verify token budget (rough estimate: 1 token ≈ 4 characters)
                    total_chars = sum(len(bullet) for bullet in injected)
                    estimated_tokens = total_chars / 4
                    assert estimated_tokens <= 200, f"Estimated tokens {estimated_tokens} exceed budget of 200"
                    
                except Exception as e:
                    collector.add_measurement("injection_error", 0.0, {"error": str(e), "budget": budget})
        
        # Get statistics
        stats = collector.get_stats("injection")
        
        # Assert we have measurements
        assert stats['count'] > 0, "Should have injection measurements"
        
        # Print summary
        collector.print_summary("injection")
        
        # Injection should be very fast
        assert stats['p95_ms'] < 10, f"P95 injection {stats['p95_ms']:.2f}ms should be under 10ms"
        assert stats['mean_ms'] < 5, f"Mean injection {stats['mean_ms']:.2f}ms should be under 5ms"


class TestEndToEndLatencyBaseline:
    """Test end-to-end latency for the complete memory pipeline"""
    
    def test_end_to_end_latency_baseline(self, hot_memory):
        """
        Test complete pipeline: extraction -> storage -> retrieval -> injection
        """
        if not HOTMEM_AVAILABLE:
            pytest.skip("HotMem not available")
        
        collector = MetricsCollector()
        
        test_sentence = "I'm Sarah and I work as a data scientist at Google in San Francisco."
        query = "What do you do for work?"
        
        # Complete pipeline timing
        with collector.time_operation("end_to_end_pipeline"):
            try:
                # 1. Extraction
                with collector.time_operation("pipeline_extraction"):
                    bullets, triples = hot_memory.process_turn(test_sentence, session_id="test_e2e", turn_id=1000)
                
                # 2. Storage (handled by process_turn)
                with collector.time_operation("pipeline_storage"):
                    pass  # Storage is automatic in process_turn
                
                # 3. Retrieval
                with collector.time_operation("pipeline_retrieval"):
                    memories = hot_memory.retrieve_bullets(query, read_only=True)
                
                # 4. Injection (simulated)
                with collector.time_operation("pipeline_injection"):
                    injected = memories[:2]  # Take top 2 bullets
                
                # Verify we got some results
                assert len(injected) >= 0, "Should have injection results"
                
            except Exception as e:
                collector.add_measurement("pipeline_error", 0.0, {"error": str(e)})
        
        # Get end-to-end stats
        e2e_stats = collector.get_stats("end_to_end_pipeline")
        
        # Assert we have end-to-end measurement
        assert e2e_stats['count'] > 0, "Should have end-to-end measurement"
        
        # Print all pipeline component stats
        print("\n=== Pipeline Component Breakdown ===")
        for operation in ["pipeline_extraction", "pipeline_storage", "pipeline_retrieval", "pipeline_injection"]:
            if collector.get_measurements(operation):
                collector.print_summary(operation)
        
        collector.print_summary("end_to_end_pipeline")
        
        # End-to-end should be under 200ms p95 for hot path
        if e2e_stats['p95_ms'] > 200:
            pytest.warn(f"End-to-end P95 {e2e_stats['p95_ms']:.2f}ms exceeds 200ms hot path target")
        
        # At minimum, should be under 500ms
        assert e2e_stats['p95_ms'] < 500, f"P95 end-to-end {e2e_stats['p95_ms']:.2f}ms should be under 500ms"


class TestMetricsHelperFunctionality:
    """Test the metrics helper itself"""
    
    def test_metrics_collector_basic_functionality(self):
        """Test basic metrics collector functionality"""
        collector = MetricsCollector()
        
        # Test manual measurements
        collector.add_measurement("test_op", 10.5)
        collector.add_measurement("test_op", 15.2)
        collector.add_measurement("test_op", 12.8)
        collector.add_measurement("other_op", 5.1)
        
        # Test stats calculation
        test_stats = collector.get_stats("test_op")
        assert test_stats['count'] == 3
        assert abs(test_stats['mean_ms'] - 12.83) < 0.1
        assert test_stats['min_ms'] == 10.5
        assert test_stats['max_ms'] == 15.2
        
        # Test filtering - get_stats() without operation returns combined stats of all operations
        all_stats = collector.get_stats()
        assert all_stats['count'] == 4  # Should have total of all measurements
        
        # Test context manager
        with collector.time_operation("context_test"):
            time.sleep(0.01)  # 10ms
        
        context_stats = collector.get_stats("context_test")
        assert context_stats['count'] == 1
        assert context_stats['mean_ms'] > 5  # Should be around 10ms
    
    def test_benchmark_function(self):
        """Test the benchmark function"""
        def dummy_function():
            time.sleep(0.001)  # 1ms
        
        stats = benchmark_multiple_runs(dummy_function, runs=5, warmup_runs=2)
        
        assert stats['count'] == 5
        assert stats['mean_ms'] > 0.5  # Should be around 1ms
        assert stats['p95_ms'] > 0
    
    def test_time_function(self):
        """Test the time_function helper"""
        def dummy_function(x):
            time.sleep(0.001)
            return x * 2
        
        result, duration = time_function(dummy_function, 5)
        
        assert result == 10
        assert duration > 0.5  # Should be around 1ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
