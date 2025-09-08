"""
Integration tests for LocalCat pipeline components
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from processors.memory_processor import MemoryProcessor, MemoryProcessorConfig
from processors.extraction_processor import ExtractionProcessor, ExtractionProcessorConfig
from processors.quality_processor import QualityProcessor, QualityProcessorConfig
from processors.context_processor import ContextProcessor, ContextProcessorConfig
from pipeline_builder import PipelineBuilder, PipelineConfig
from config import Config, EnvironmentType


class TestPipelineIntegration:
    """Test cases for pipeline integration"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_processing(self, test_config, temp_dir, sample_text):
        """Test full pipeline processing with all components"""
        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            enable_memory=True,
            enable_extraction=True,
            enable_quality=True,
            enable_context=True,
            enable_metrics=False
        )
        
        # Create pipeline builder
        builder = PipelineBuilder(pipeline_config)
        
        # Build pipeline
        pipeline = builder.build_pipeline()
        
        assert len(pipeline) == 4  # memory, extraction, quality, context
        
        # Test pipeline processing
        from tests.conftest import MockTranscriptionFrame
        
        frame = MockTranscriptionFrame(sample_text)
        
        # Process through pipeline (simplified for testing)
        current_frame = frame
        
        for processor in pipeline:
            # Mock push_frame for each processor
            pushed_frames = []
            
            async def push_frame(f, direction=None):
                pushed_frames.append(f)
            
            processor.push_frame = push_frame
            
            await processor.process_frame(current_frame, None)
            
            if pushed_frames:
                current_frame = pushed_frames[-1]
        
        # Verify frame was processed
        assert current_frame is not None
    
    @pytest.mark.asyncio
    async def test_memory_extraction_integration(self, test_config, temp_dir, sample_text):
        """Test integration between memory and extraction processors"""
        # Create processors
        memory_config = MemoryProcessorConfig(
            sqlite_path=str(temp_dir / "test.db"),
            lmdb_dir=str(temp_dir / "test_lmdb"),
            user_id="test-user",
            enable_metrics=False
        )
        
        extraction_config = ExtractionProcessorConfig(
            default_strategy="lightweight",
            enable_metrics=False
        )
        
        memory_processor = MemoryProcessor(memory_config)
        extraction_processor = ExtractionProcessor(extraction_config)
        
        # Mock extraction registry
        mock_strategy = Mock()
        mock_strategy.extract = AsyncMock(return_value={
            "text": sample_text,
            "entities": [{"text": "John", "label": "PERSON", "confidence": 0.8}],
            "facts": [{"text": "John is testing", "confidence": 0.7}],
            "confidence": 0.75,
            "strategy_used": "test"
        })
        
        extraction_processor.extraction_registry.get_strategy = Mock(return_value=mock_strategy)
        
        from tests.conftest import MockTranscriptionFrame, MockTextFrame
        
        # Test frame flow
        frame = MockTranscriptionFrame(sample_text)
        
        # Process through extraction first
        pushed_frames = []
        
        async def push_frame(f, direction=None):
            pushed_frames.append(f)
        
        extraction_processor.push_frame = push_frame
        await extraction_processor.process_frame(frame, None)
        
        # Verify extraction added metadata
        assert len(pushed_frames) == 1
        extracted_frame = pushed_frames[0]
        assert hasattr(extracted_frame, 'metadata')
        assert 'extraction_result' in extracted_frame.metadata
        
        # Process through memory
        memory_frames = []
        
        async def memory_push_frame(f, direction=None):
            memory_frames.append(f)
        
        memory_processor.push_frame = memory_push_frame
        
        # Mock memory operations
        memory_processor._extract_memory_facts = AsyncMock(return_value=[
            {"text": "John is testing", "confidence": 0.8}
        ])
        memory_processor._store_memory_facts = AsyncMock()
        
        await memory_processor.process_frame(extracted_frame, None)
        
        # Verify memory processor handled the frame
        assert len(memory_frames) == 1
    
    @pytest.mark.asyncio
    async def test_quality_validation_integration(self, test_config, sample_extraction_result):
        """Test integration between extraction and quality processors"""
        # Create processors
        extraction_config = ExtractionProcessorConfig(
            default_strategy="lightweight",
            enable_metrics=False
        )
        
        quality_config = QualityProcessorConfig(
            min_confidence_threshold=0.5,
            min_overall_quality_threshold=0.4,
            enable_correction=False,
            enable_metrics=False
        )
        
        extraction_processor = ExtractionProcessor(extraction_config)
        quality_processor = QualityProcessor(quality_config)
        
        from tests.conftest import MockTextFrame
        
        # Create frame with extraction result
        frame = MockTextFrame("Test text")
        frame.metadata = {
            'extraction_result': sample_extraction_result,
            'extraction_quality': 0.8,
            'extraction_strategy': 'test'
        }
        
        # Process through quality processor
        pushed_frames = []
        
        async def push_frame(f, direction=None):
            pushed_frames.append(f)
        
        quality_processor.push_frame = push_frame
        await quality_processor.process_frame(frame, None)
        
        # Verify quality validation
        assert len(pushed_frames) == 1
        quality_frame = pushed_frames[0]
        
        # Check that quality metadata was added
        if hasattr(quality_frame, 'metadata') and 'extraction_result' in quality_frame.metadata:
            result = quality_frame.metadata['extraction_result']
            assert 'quality_metrics' in result
            assert 'quality_level' in result
            assert 'quality_score' in result
    
    @pytest.mark.asyncio
    async def test_context_memory_integration(self, test_config, temp_dir):
        """Test integration between context and memory processors"""
        # Create processors
        memory_config = MemoryProcessorConfig(
            sqlite_path=str(temp_dir / "test.db"),
            lmdb_dir=str(temp_dir / "test_lmdb"),
            user_id="test-user",
            enable_metrics=False
        )
        
        context_config = ContextProcessorConfig(
            max_context_items=10,
            max_context_tokens=200,
            enable_metrics=False
        )
        
        memory_processor = MemoryProcessor(memory_config)
        context_processor = ContextProcessor(context_config)
        
        from tests.conftest import MockLLMMessagesFrame
        
        # Create LLM messages frame
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, I'm John from Acme Corp"}
        ]
        
        frame = MockLLMMessagesFrame(messages)
        
        # Mock memory context retrieval
        memory_context = ["User works at Acme Corporation", "User's name is John"]
        
        async def mock_get_memory_context(msgs):
            return memory_context
        
        memory_processor._get_memory_context = mock_get_memory_context
        
        # Process through memory processor
        memory_frames = []
        
        async def memory_push_frame(f, direction=None):
            memory_frames.append(f)
        
        memory_processor.push_frame = memory_push_frame
        await memory_processor.process_frame(frame, None)
        
        # Verify memory context injection
        assert len(memory_frames) == 1
        memory_frame = memory_frames[0]
        
        # Process through context processor
        context_frames = []
        
        async def context_push_frame(f, direction=None):
            context_frames.append(f)
        
        context_processor.push_frame = context_push_frame
        
        # Mock context gathering
        async def mock_gather_relevant_context(msgs):
            return []
        
        context_processor._gather_relevant_context = mock_gather_relevant_context
        
        await context_processor.process_frame(memory_frame, None)
        
        # Verify context processing
        assert len(context_frames) == 1
    
    @pytest.mark.asyncio
    async def test_pipeline_builder_dependency_resolution(self, test_config):
        """Test pipeline builder dependency resolution"""
        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            enable_memory=True,
            enable_extraction=True,
            enable_quality=True,
            enable_context=True,
            enable_metrics=False
        )
        
        # Create pipeline builder with custom dependencies
        builder = PipelineBuilder(pipeline_config)
        
        # Add custom dependencies
        builder.add_dependency("quality", "extraction")
        builder.add_dependency("memory", "quality")
        builder.add_dependency("context", "memory")
        
        # Build pipeline
        pipeline = builder.build_pipeline()
        
        # Verify pipeline order respects dependencies
        processor_names = []
        for processor in pipeline:
            # Get processor name from class
            name = processor.__class__.__name__.lower().replace('processor', '')
            processor_names.append(name)
        
        # Verify order: extraction -> quality -> memory -> context
        expected_order = ['extraction', 'quality', 'memory', 'context']
        assert processor_names == expected_order
    
    @pytest.mark.asyncio
    async def test_pipeline_configuration_validation(self, test_config):
        """Test pipeline configuration validation"""
        # Test with invalid configuration
        pipeline_config = PipelineConfig(
            enable_memory=True,
            enable_extraction=True,
            enable_quality=True,
            enable_context=True,
            max_pipeline_latency=-1.0  # Invalid value
        )
        
        builder = PipelineBuilder(pipeline_config)
        
        # Should still build but with validation warnings
        pipeline = builder.build_pipeline()
        
        assert len(pipeline) == 4
    
    @pytest.mark.asyncio
    async def test_error_propagation_through_pipeline(self, test_config, temp_dir):
        """Test error propagation through pipeline stages"""
        # Create processors
        memory_config = MemoryProcessorConfig(
            sqlite_path=str(temp_dir / "test.db"),
            lmdb_dir=str(temp_dir / "test_lmdb"),
            user_id="test-user",
            enable_metrics=False
        )
        
        extraction_config = ExtractionProcessorConfig(
            default_strategy="lightweight",
            enable_metrics=False
        )
        
        memory_processor = MemoryProcessor(memory_config)
        extraction_processor = ExtractionProcessor(extraction_config)
        
        from tests.conftest import MockTranscriptionFrame
        
        frame = MockTranscriptionFrame("Test text")
        
        # Mock extraction to fail
        extraction_processor.extraction_registry.get_strategy = Mock(return_value=None)
        
        # Process through extraction
        extraction_frames = []
        
        async def extraction_push_frame(f, direction=None):
            extraction_frames.append(f)
        
        extraction_processor.push_frame = extraction_push_frame
        await extraction_processor.process_frame(frame, None)
        
        # Verify frame was still passed through despite extraction failure
        assert len(extraction_frames) == 1
        
        # Process through memory
        memory_frames = []
        
        async def memory_push_frame(f, direction=None):
            memory_frames.append(f)
        
        memory_processor.push_frame = memory_push_frame
        await memory_processor.process_frame(extraction_frames[0], None)
        
        # Verify memory processor handled the frame
        assert len(memory_frames) == 1
    
    @pytest.mark.asyncio
    async def test_pipeline_metrics_collection(self, test_config):
        """Test metrics collection across pipeline"""
        # Create pipeline configuration with metrics enabled
        pipeline_config = PipelineConfig(
            enable_memory=True,
            enable_extraction=True,
            enable_quality=True,
            enable_context=True,
            enable_metrics=True
        )
        
        builder = PipelineBuilder(pipeline_config)
        pipeline = builder.build_pipeline()
        
        # Verify all processors have metrics enabled
        for processor in pipeline:
            if hasattr(processor, 'config'):
                assert processor.config.enable_metrics is True
        
        # Get pipeline info
        info = builder.get_pipeline_info()
        
        assert info['total_processors'] == 4
        assert info['enabled_processors'] == 4
        assert 'processors' in info
        assert len(info['processors']) == 4
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_processing(self, test_config, temp_dir):
        """Test concurrent processing of multiple frames"""
        # Create processors
        memory_config = MemoryProcessorConfig(
            sqlite_path=str(temp_dir / "test.db"),
            lmdb_dir=str(temp_dir / "test_lmdb"),
            user_id="test-user",
            enable_metrics=False
        )
        
        extraction_config = ExtractionProcessorConfig(
            default_strategy="lightweight",
            enable_metrics=False
        )
        
        memory_processor = MemoryProcessor(memory_config)
        extraction_processor = ExtractionProcessor(extraction_config)
        
        from tests.conftest import MockTranscriptionFrame
        
        # Create multiple frames
        frames = [
            MockTranscriptionFrame(f"Test text {i}") for i in range(5)
        ]
        
        # Mock extraction
        mock_strategy = Mock()
        mock_strategy.extract = AsyncMock(return_value={
            "text": "mocked extraction",
            "entities": [],
            "facts": [],
            "confidence": 0.8,
            "strategy_used": "test"
        })
        
        extraction_processor.extraction_registry.get_strategy = Mock(return_value=mock_strategy)
        
        # Process frames concurrently
        tasks = []
        for frame in frames:
            task = asyncio.create_task(self._process_frame_through_processor(extraction_processor, frame))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all frames were processed
        assert len(results) == 5
        for result in results:
            assert not isinstance(result, Exception)
    
    async def _process_frame_through_processor(self, processor, frame):
        """Helper method to process frame through processor"""
        pushed_frames = []
        
        async def push_frame(f, direction=None):
            pushed_frames.append(f)
        
        processor.push_frame = push_frame
        await processor.process_frame(frame, None)
        
        return pushed_frames[0] if pushed_frames else None