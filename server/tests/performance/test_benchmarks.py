"""
Performance benchmarks for LocalCat components
"""

import pytest
import asyncio
import time
import psutil
import os
from typing import List, Dict, Any
from tests.conftest import benchmark_function, measure_time, measure_memory_usage


class TestMemoryProcessorPerformance:
    """Performance benchmarks for MemoryProcessor"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_processing_latency(self, memory_processor):
        """Test memory processing latency"""
        from tests.conftest import MockTranscriptionFrame
        
        frame = MockTranscriptionFrame("This is a test message for memory processing benchmark.")
        
        # Mock the heavy operations
        memory_processor._extract_memory_facts = asyncio.coroutine(lambda x: [{"text": "test fact", "confidence": 0.8}])
        memory_processor._store_memory_facts = asyncio.coroutine(lambda x: None)
        
        # Benchmark processing time
        def process_memory():
            pushed_frames = []
            
            async def push_frame(f, direction=None):
                pushed_frames.append(f)
            
            memory_processor.push_frame = push_frame
            return asyncio.run(memory_processor.process_frame(frame, None))
        
        results = benchmark_function(process_memory, iterations=100)
        
        # Assert performance requirements
        assert results['avg_time'] < 0.1  # < 100ms average
        assert results['max_time'] < 0.5  # < 500ms maximum
        
        print(f"Memory processing performance: {results}")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_throughput(self, memory_processor):
        """Test memory processing throughput"""
        from tests.conftest import MockTranscriptionFrame
        
        # Create multiple frames
        frames = [
            MockTranscriptionFrame(f"Test message {i} for throughput testing.") 
            for i in range(100)
        ]
        
        # Mock operations
        memory_processor._extract_memory_facts = asyncio.coroutine(lambda x: [{"text": "test fact", "confidence": 0.8}])
        memory_processor._store_memory_facts = asyncio.coroutine(lambda x: None)
        
        # Measure throughput
        start_time = time.time()
        
        tasks = []
        for frame in frames:
            task = asyncio.create_task(self._process_single_frame(memory_processor, frame))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate throughput
        throughput = len(frames) / total_time  # frames per second
        
        assert throughput > 10  # At least 10 frames per second
        print(f"Memory throughput: {throughput:.2f} frames/second")
    
    @pytest.mark.performance
    def test_memory_usage(self, memory_processor):
        """Test memory usage of memory processor"""
        initial_memory = measure_memory_usage()
        
        # Create processor and process some data
        from tests.conftest import MockTranscriptionFrame
        
        frame = MockTranscriptionFrame("Test message for memory usage benchmark.")
        
        async def test_memory_usage():
            pushed_frames = []
            
            async def push_frame(f, direction=None):
                pushed_frames.append(f)
            
            memory_processor.push_frame = push_frame
            await memory_processor.process_frame(frame, None)
        
        asyncio.run(test_memory_usage())
        
        final_memory = measure_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Assert memory usage is reasonable
        assert memory_increase < 50  # Less than 50MB increase
        print(f"Memory usage increase: {memory_increase:.2f} MB")


class TestExtractionProcessorPerformance:
    """Performance benchmarks for ExtractionProcessor"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_extraction_latency(self, extraction_processor):
        """Test extraction processing latency"""
        from tests.conftest import MockTextFrame
        
        frame = MockTextFrame("John Doe works at Acme Corporation in New York City.")
        
        # Mock extraction strategy
        mock_strategy = Mock()
        mock_strategy.extract = asyncio.coroutine(lambda x: {
            "text": x,
            "entities": [
                {"text": "John Doe", "label": "PERSON", "confidence": 0.9},
                {"text": "Acme Corporation", "label": "ORG", "confidence": 0.8},
                {"text": "New York City", "label": "GPE", "confidence": 0.7}
            ],
            "facts": [
                {"text": "John Doe works at Acme Corporation", "confidence": 0.85}
            ],
            "confidence": 0.8,
            "strategy_used": "test"
        })
        
        extraction_processor.extraction_registry.get_strategy = Mock(return_value=mock_strategy)
        
        # Benchmark extraction time
        def process_extraction():
            pushed_frames = []
            
            async def push_frame(f, direction=None):
                pushed_frames.append(f)
            
            extraction_processor.push_frame = push_frame
            return asyncio.run(extraction_processor.process_frame(frame, None))
        
        results = benchmark_function(process_extraction, iterations=50)
        
        # Assert performance requirements
        assert results['avg_time'] < 0.5  # < 500ms average
        assert results['max_time'] < 2.0  # < 2 seconds maximum
        
        print(f"Extraction processing performance: {results}")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_multi_strategy_extraction(self, extraction_processor):
        """Test multi-strategy extraction performance"""
        from tests.conftest import MockTextFrame
        
        frame = MockTextFrame("Complex text with multiple entities and relationships.")
        
        # Mock multiple strategies
        strategies = []
        for i in range(3):
            strategy = Mock()
            strategy.extract = asyncio.coroutine(lambda x: {
                "text": x,
                "entities": [{"text": f"Entity{i}", "label": "TEST", "confidence": 0.8}],
                "facts": [{"text": f"Fact{i}", "confidence": 0.7}],
                "confidence": 0.75,
                "strategy_used": f"strategy{i}"
            })
            strategies.append(strategy)
        
        extraction_processor.extraction_registry.get_available_strategies = Mock(return_value=[f"strategy{i}" for i in range(3)])
        
        async def mock_get_strategy(name):
            return strategies[int(name[-1])]
        
        extraction_processor.extraction_registry.get_strategy = mock_get_strategy
        
        # Benchmark multi-strategy extraction
        start_time = time.time()
        
        pushed_frames = []
        
        async def push_frame(f, direction=None):
            pushed_frames.append(f)
        
        extraction_processor.push_frame = push_frame
        await extraction_processor.process_frame(frame, None)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assert concurrent processing is efficient
        assert processing_time < 2.0  # < 2 seconds for 3 strategies
        print(f"Multi-strategy extraction time: {processing_time:.3f} seconds")


class TestQualityProcessorPerformance:
    """Performance benchmarks for QualityProcessor"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_quality_validation_latency(self, quality_processor):
        """Test quality validation latency"""
        from tests.conftest import MockTextFrame
        
        frame = MockTextFrame("Test text for quality validation.")
        frame.metadata = {
            'extraction_result': {
                "text": "John Doe works at Acme Corporation",
                "entities": [
                    {"text": "John Doe", "label": "PERSON", "confidence": 0.9},
                    {"text": "Acme Corporation", "label": "ORG", "confidence": 0.8}
                ],
                "facts": [
                    {"text": "John Doe works at Acme Corporation", "confidence": 0.85}
                ],
                "confidence": 0.8
            }
        }
        
        # Benchmark quality validation
        def process_quality():
            pushed_frames = []
            
            async def push_frame(f, direction=None):
                pushed_frames.append(f)
            
            quality_processor.push_frame = push_frame
            return asyncio.run(quality_processor.process_frame(frame, None))
        
        results = benchmark_function(process_quality, iterations=100)
        
        # Assert performance requirements
        assert results['avg_time'] < 0.05  # < 50ms average
        assert results['max_time'] < 0.2  # < 200ms maximum
        
        print(f"Quality validation performance: {results}")


class TestContextProcessorPerformance:
    """Performance benchmarks for ContextProcessor"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_context_processing_latency(self, context_processor):
        """Test context processing latency"""
        from tests.conftest import MockLLMMessagesFrame
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, I'm John from Acme Corp. How are you doing today?"},
            {"role": "assistant", "content": "Hello John! I'm doing well, thank you for asking."}
        ]
        
        frame = MockLLMMessagesFrame(messages)
        
        # Mock context gathering
        async def mock_gather_relevant_context(msgs):
            return [
                {"content": "User works at Acme Corporation", "context_type": "memory", "priority": 0.8},
                {"content": "User's name is John", "context_type": "memory", "priority": 0.7}
            ]
        
        context_processor._gather_relevant_context = mock_gather_relevant_context
        
        # Benchmark context processing
        def process_context():
            pushed_frames = []
            
            async def push_frame(f, direction=None):
                pushed_frames.append(f)
            
            context_processor.push_frame = push_frame
            return asyncio.run(context_processor.process_frame(frame, None))
        
        results = benchmark_function(process_context, iterations=50)
        
        # Assert performance requirements
        assert results['avg_time'] < 0.1  # < 100ms average
        assert results['max_time'] < 0.5  # < 500ms maximum
        
        print(f"Context processing performance: {results}")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_context_handling(self, context_processor):
        """Test handling of large context windows"""
        from tests.conftest import MockLLLMessagesFrame
        
        # Create large conversation history
        messages = [{"role": "system", "content": "You are a helpful assistant"}]
        
        for i in range(100):
            messages.append({"role": "user", "content": f"User message {i}: This is a test message with some content."})
            messages.append({"role": "assistant", "content": f"Assistant response {i}: This is a response to the user's message."})
        
        frame = MockLLLMessagesFrame(messages)
        
        # Mock context gathering
        async def mock_gather_relevant_context(msgs):
            # Return large context
            return [
                {"content": f"Context item {i}", "context_type": "memory", "priority": 0.5}
                for i in range(50)
            ]
        
        context_processor._gather_relevant_context = mock_gather_relevant_context
        
        # Benchmark large context processing
        start_time = time.time()
        
        pushed_frames = []
        
        async def push_frame(f, direction=None):
            pushed_frames.append(f)
        
        context_processor.push_frame = push_frame
        await context_processor.process_frame(frame, None)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assert large context handling is efficient
        assert processing_time < 1.0  # < 1 second for large context
        print(f"Large context processing time: {processing_time:.3f} seconds")


class TestPipelinePerformance:
    """Performance benchmarks for complete pipeline"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_latency(self, pipeline_builder):
        """Test end-to-end pipeline processing latency"""
        from tests.conftest import MockTranscriptionFrame
        
        frame = MockTranscriptionFrame("John Doe works at Acme Corporation in New York.")
        
        # Mock all processor operations
        for processor in pipeline_builder.built_pipeline or []:
            if hasattr(processor, '_extract_memory_facts'):
                processor._extract_memory_facts = asyncio.coroutine(lambda x: [{"text": "test fact", "confidence": 0.8}])
                processor._store_memory_facts = asyncio.coroutine(lambda x: None)
                processor._get_memory_context = asyncio.coroutine(lambda x: ["Test context"])
            
            if hasattr(processor, 'extraction_registry'):
                mock_strategy = Mock()
                mock_strategy.extract = asyncio.coroutine(lambda x: {
                    "text": x,
                    "entities": [{"text": "John Doe", "label": "PERSON", "confidence": 0.9}],
                    "facts": [{"text": "John Doe works at Acme Corporation", "confidence": 0.85}],
                    "confidence": 0.8,
                    "strategy_used": "test"
                })
                processor.extraction_registry.get_strategy = Mock(return_value=mock_strategy)
            
            if hasattr(processor, '_gather_relevant_context'):
                processor._gather_relevant_context = asyncio.coroutine(lambda x: [])
        
        # Build pipeline
        pipeline = pipeline_builder.build_pipeline()
        
        # Benchmark end-to-end processing
        async def process_through_pipeline():
            current_frame = frame
            
            for processor in pipeline:
                pushed_frames = []
                
                async def push_frame(f, direction=None):
                    pushed_frames.append(f)
                
                processor.push_frame = push_frame
                await processor.process_frame(current_frame, None)
                
                if pushed_frames:
                    current_frame = pushed_frames[-1]
            
            return current_frame
        
        results = benchmark_function(process_through_pipeline, iterations=20)
        
        # Assert end-to-end performance requirements
        assert results['avg_time'] < 1.0  # < 1 second average
        assert results['max_time'] < 3.0  # < 3 seconds maximum
        
        print(f"End-to-end pipeline performance: {results}")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_pipeline_memory_usage(self, pipeline_builder):
        """Test pipeline memory usage"""
        initial_memory = measure_memory_usage()
        
        # Process some data through pipeline
        from tests.conftest import MockTranscriptionFrame
        
        frame = MockTranscriptionFrame("Test message for pipeline memory benchmark.")
        
        async def test_pipeline_memory():
            # Mock operations to avoid actual processing
            for processor in pipeline_builder.built_pipeline or []:
                if hasattr(processor, '_extract_memory_facts'):
                    processor._extract_memory_facts = asyncio.coroutine(lambda x: [])
                    processor._store_memory_facts = asyncio.coroutine(lambda x: None)
                    processor._get_memory_context = asyncio.coroutine(lambda x: [])
                
                if hasattr(processor, 'extraction_registry'):
                    mock_strategy = Mock()
                    mock_strategy.extract = asyncio.coroutine(lambda x: {
                        "text": x,
                        "entities": [],
                        "facts": [],
                        "confidence": 0.8,
                        "strategy_used": "test"
                    })
                    processor.extraction_registry.get_strategy = Mock(return_value=mock_strategy)
                
                if hasattr(processor, '_gather_relevant_context'):
                    processor._gather_relevant_context = asyncio.coroutine(lambda x: [])
            
            # Process frame
            current_frame = frame
            pipeline = pipeline_builder.build_pipeline()
            
            for processor in pipeline:
                pushed_frames = []
                
                async def push_frame(f, direction=None):
                    pushed_frames.append(f)
                
                processor.push_frame = push_frame
                await processor.process_frame(current_frame, None)
                
                if pushed_frames:
                    current_frame = pushed_frames[-1]
        
        await test_pipeline_memory()
        
        final_memory = measure_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Assert memory usage is reasonable
        assert memory_increase < 100  # Less than 100MB increase
        print(f"Pipeline memory usage increase: {memory_increase:.2f} MB")


@pytest.mark.performance
class TestSystemPerformance:
    """System-level performance benchmarks"""
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, temp_dir):
        """Test performance with concurrent sessions"""
        from processors.memory_processor import MemoryProcessor, MemoryProcessorConfig
        from tests.conftest import MockTranscriptionFrame
        
        # Create multiple memory processors (simulating multiple sessions)
        processors = []
        for i in range(10):
            config = MemoryProcessorConfig(
                sqlite_path=str(temp_dir / f"test_memory_{i}.db"),
                lmdb_dir=str(temp_dir / f"test_lmdb_{i}"),
                user_id=f"user_{i}",
                enable_metrics=False
            )
            processor = MemoryProcessor(config)
            processors.append(processor)
        
        # Create test frames
        frames = [
            MockTranscriptionFrame(f"Test message for session {i}")
            for i in range(10)
        ]
        
        # Mock operations
        for processor in processors:
            processor._extract_memory_facts = asyncio.coroutine(lambda x: [{"text": "test fact", "confidence": 0.8}])
            processor._store_memory_facts = asyncio.coroutine(lambda x: None)
        
        # Process concurrently
        start_time = time.time()
        
        tasks = []
        for processor, frame in zip(processors, frames):
            task = asyncio.create_task(self._process_single_frame(processor, frame))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate concurrent throughput
        throughput = len(frames) / total_time
        
        assert throughput > 5  # At least 5 concurrent sessions per second
        print(f"Concurrent session throughput: {throughput:.2f} sessions/second")
    
    async def _process_single_frame(self, processor, frame):
        """Helper method to process frame through processor"""
        pushed_frames = []
        
        async def push_frame(f, direction=None):
            pushed_frames.append(f)
        
        processor.push_frame = push_frame
        await processor.process_frame(frame, None)
        
        return pushed_frames[0] if pushed_frames else None