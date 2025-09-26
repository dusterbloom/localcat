"""
Comprehensive test suite for coreference resolution integration.

Tests the SOLID/DRY architecture implementation including:
- SharedNLPManager functionality
- TextProcessor strategy pattern
- CoreferenceProcessor behavior
- UDExtractor composition
- Configuration management
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
import spacy
from spacy.tokens import Doc

# Import the modules we're testing
from server.core.memory.nlp_manager import SharedNLPManager, get_nlp_manager, get_nlp_model, get_nlp_with_coref
from server.core.memory.processors.base import TextProcessor, ProcessorChain, NoOpProcessor
from server.core.memory.processors.coreference import CoreferenceProcessor
from server.core.memory.extractors.ud import UDExtractor
from server.core.memory.config import MemoryConfig, CoreferenceConfig, validate_memory_config


class TestSharedNLPManager:
    """Test the SharedNLPManager for DRY compliance."""

    def test_singleton_pattern(self):
        """Test that get_nlp_manager returns the same instance."""
        manager1 = get_nlp_manager()
        manager2 = get_nlp_manager()
        assert manager1 is manager2

    def test_model_caching(self):
        """Test that models are cached properly."""
        manager = SharedNLPManager()

        with patch('spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp

            # First call should load model
            result1 = manager.get_model("en")
            assert mock_load.call_count == 1
            assert result1 is mock_nlp

            # Second call should use cache
            result2 = manager.get_model("en")
            assert mock_load.call_count == 1  # No additional calls
            assert result2 is mock_nlp

    def test_model_with_components(self):
        """Test loading models with specific components."""
        manager = SharedNLPManager()

        with patch('spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_nlp.pipe_names = []
            mock_nlp.add_pipe = Mock()
            mock_load.return_value = mock_nlp

            result = manager.get_model("en", components=["coref"])

            assert mock_load.called
            mock_nlp.add_pipe.assert_called_once_with("coref")

    def test_component_already_exists(self):
        """Test that existing components aren't added twice."""
        manager = SharedNLPManager()

        with patch('spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_nlp.pipe_names = ["coref"]  # Component already exists
            mock_nlp.add_pipe = Mock()
            mock_load.return_value = mock_nlp

            manager.get_model("en", components=["coref"])

            mock_nlp.add_pipe.assert_not_called()

    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        manager = SharedNLPManager()

        key1 = manager._build_cache_key("en", ["coref", "ner"])
        key2 = manager._build_cache_key("en", ["ner", "coref"])  # Same components, different order

        assert key1 == key2  # Should be the same due to sorting

    def test_backward_compatibility_functions(self):
        """Test that backward compatibility functions work."""
        with patch.object(get_nlp_manager(), 'get_model') as mock_get:
            mock_nlp = Mock()
            mock_get.return_value = mock_nlp

            result = get_nlp_model("en")
            mock_get.assert_called_once_with("en")
            assert result is mock_nlp

    def test_coreference_model_function(self):
        """Test the coreference-specific model function."""
        with patch.object(get_nlp_manager(), 'get_model') as mock_get:
            mock_nlp = Mock()
            mock_get.return_value = mock_nlp

            result = get_nlp_with_coref("en")
            mock_get.assert_called_once_with("en", components=["coref"])
            assert result is mock_nlp


class TestTextProcessorStrategy:
    """Test the TextProcessor strategy pattern for OCP/DIP compliance."""

    def test_processor_chain_empty(self):
        """Test processor chain with no processors."""
        chain = ProcessorChain([])
        mock_doc = Mock(spec=Doc)

        result = chain.process(mock_doc)
        assert result is mock_doc

    def test_processor_chain_single(self):
        """Test processor chain with single processor."""
        mock_processor = Mock(spec=TextProcessor)
        mock_processor.name = "test"
        mock_processor._record_metric = Mock()
        processed_doc = Mock(spec=Doc)
        mock_processor.process.return_value = processed_doc

        chain = ProcessorChain([mock_processor])
        input_doc = Mock(spec=Doc)

        result = chain.process(input_doc)

        mock_processor.process.assert_called_once_with(input_doc)
        assert result is processed_doc

    def test_processor_chain_multiple(self):
        """Test processor chain with multiple processors."""
        # Create mock processors
        processor1 = Mock(spec=TextProcessor)
        processor1.name = "processor1"
        processor1._record_metric = Mock()
        doc1 = Mock(spec=Doc)
        processor1.process.return_value = doc1

        processor2 = Mock(spec=TextProcessor)
        processor2.name = "processor2"
        processor2._record_metric = Mock()
        doc2 = Mock(spec=Doc)
        processor2.process.return_value = doc2

        chain = ProcessorChain([processor1, processor2])
        input_doc = Mock(spec=Doc)

        result = chain.process(input_doc)

        # Verify chaining
        processor1.process.assert_called_once_with(input_doc)
        processor2.process.assert_called_once_with(doc1)
        assert result is doc2

    def test_processor_failure_continues_chain(self):
        """Test that processor failures don't break the chain."""
        # Failing processor
        failing_processor = Mock(spec=TextProcessor)
        failing_processor.name = "failing"
        failing_processor._record_metric = Mock()
        failing_processor.process.side_effect = Exception("Test failure")

        # Working processor
        working_processor = Mock(spec=TextProcessor)
        working_processor.name = "working"
        working_processor._record_metric = Mock()
        output_doc = Mock(spec=Doc)
        working_processor.process.return_value = output_doc

        chain = ProcessorChain([failing_processor, working_processor])
        input_doc = Mock(spec=Doc)

        result = chain.process(input_doc)

        # Should continue with original doc after failure
        working_processor.process.assert_called_once_with(input_doc)
        assert result is output_doc

    def test_noop_processor(self):
        """Test the NoOpProcessor implementation."""
        processor = NoOpProcessor()
        mock_doc = Mock(spec=Doc)

        result = processor.process(mock_doc)
        assert result is mock_doc
        assert processor.name == "noop"


class TestCoreferenceProcessor:
    """Test the CoreferenceProcessor for SRP compliance."""

    def test_processor_initialization(self):
        """Test processor initialization with default values."""
        processor = CoreferenceProcessor()

        assert processor.name == "coreference"
        assert processor.timeout_ms == 50
        assert processor.min_text_length == 10
        assert processor.lang == "en"
        assert processor._nlp is None

    def test_short_text_skipping(self):
        """Test that short texts are skipped."""
        processor = CoreferenceProcessor(min_text_length=10)

        mock_doc = Mock(spec=Doc)
        mock_doc.text = "short"  # Less than 10 characters

        with patch.object(processor, '_record_metric') as mock_record:
            result = processor.process(mock_doc)

            assert result is mock_doc
            mock_record.assert_called_once()
            call_args = mock_record.call_args
            assert call_args[0][1] is True  # success=True
            assert "skipped_short_text" in call_args[0][2]  # details

    def test_model_loading_failure(self):
        """Test graceful handling of model loading failure."""
        processor = CoreferenceProcessor()

        with patch('server.core.memory.processors.coreference.get_nlp_with_coref') as mock_get_nlp:
            mock_get_nlp.return_value = None  # Simulate failure

            mock_doc = Mock(spec=Doc)
            mock_doc.text = "This is a longer text that should trigger processing."

            result = processor.process(mock_doc)

            assert result is mock_doc  # Should return original on failure

    def test_timeout_protection(self):
        """Test that timeout protection works."""
        processor = CoreferenceProcessor(timeout_ms=1)  # Very short timeout

        # Mock the model to be available
        mock_nlp = Mock()
        mock_coref_doc = Mock()
        mock_coref_doc._ = Mock()
        mock_coref_doc._.coref_clusters = []

        # Make the processing take longer than timeout
        def slow_processing(text):
            time.sleep(0.002)  # 2ms, longer than 1ms timeout
            return mock_coref_doc

        mock_nlp.side_effect = slow_processing
        processor._nlp = mock_nlp

        mock_doc = Mock(spec=Doc)
        mock_doc.text = "This is a longer text that should trigger processing."

        result = processor.process(mock_doc)

        assert result is mock_doc  # Should return original due to timeout

    def test_coreference_resolution(self):
        """Test actual coreference resolution logic."""
        processor = CoreferenceProcessor()

        # Create mock objects for the full pipeline
        mock_nlp = Mock()
        mock_coref_doc = Mock(spec=Doc)
        mock_coref_doc.text = "John went to the store. John bought milk."  # Resolved text
        mock_coref_doc._ = Mock()

        # Mock coreference clusters
        mock_mention1 = Mock()
        mock_mention1.text = "John"
        mock_mention1.start = 0
        mock_mention1.start_char = 0
        mock_mention1.end_char = 4

        mock_mention2 = Mock()
        mock_mention2.text = "He"
        mock_mention2.start = 5
        mock_mention2.start_char = 25
        mock_mention2.end_char = 27

        mock_cluster = [mock_mention1, mock_mention2]
        mock_coref_doc._.coref_clusters = [mock_cluster]

        mock_nlp.return_value = mock_coref_doc
        processor._nlp = mock_nlp

        # Mock the clean NLP model
        with patch('server.core.memory.processors.coreference.get_nlp_model') as mock_get_clean:
            mock_clean_nlp = Mock()
            mock_resolved_doc = Mock(spec=Doc)
            mock_clean_nlp.return_value = mock_resolved_doc
            mock_get_clean.return_value = mock_clean_nlp

            mock_doc = Mock(spec=Doc)
            mock_doc.text = "John went to the store. He bought milk."

            result = processor.process(mock_doc)

            # Should have processed the text
            mock_nlp.assert_called_once_with(mock_doc.text)


class TestUDExtractorComposition:
    """Test the UDExtractor composition for ISP compliance."""

    def test_backward_compatibility(self):
        """Test that UDExtractor works without text processors."""
        mock_host = Mock()
        mock_host._extract.return_value = (["entity"], [("s", "p", "o")], 0, None)

        extractor = UDExtractor(mock_host)

        result = extractor.extract("test text", "en")

        mock_host._extract.assert_called_once_with("test text", "en")
        assert result == (["entity"], [("s", "p", "o")], 0, None)

    def test_with_text_processors(self):
        """Test UDExtractor with text processors enabled."""
        mock_host = Mock()
        mock_doc = Mock(spec=Doc)
        mock_host._extract.return_value = (["entity"], [("s", "p", "o")], 0, mock_doc)

        mock_processor = Mock(spec=TextProcessor)
        mock_processor.name = "test"
        processed_doc = Mock(spec=Doc)
        processed_doc.text = "processed text"
        mock_processor.process.return_value = processed_doc

        extractor = UDExtractor(mock_host, text_processors=[mock_processor])

        # Mock the re-extraction after processing
        mock_host._extract.side_effect = [
            (["entity"], [("s", "p", "o")], 0, mock_doc),  # Initial extraction
            (["processed_entity"], [("ps", "pp", "po")], 0, processed_doc)  # Re-extraction
        ]

        result = extractor.extract("test text", "en")

        # Should have called extract twice
        assert mock_host._extract.call_count == 2
        mock_processor.process.assert_called_once_with(mock_doc)

    def test_processor_failure_fallback(self):
        """Test that processor failures fall back gracefully."""
        mock_host = Mock()
        mock_doc = Mock(spec=Doc)
        mock_host._extract.return_value = (["entity"], [("s", "p", "o")], 0, mock_doc)

        failing_processor = Mock(spec=TextProcessor)
        failing_processor.name = "failing"
        failing_processor.process.side_effect = Exception("Test failure")

        extractor = UDExtractor(mock_host, text_processors=[failing_processor])

        result = extractor.extract("test text", "en")

        # Should return original extraction despite processor failure
        assert result == (["entity"], [("s", "p", "o")], 0, mock_doc)

    def test_metrics_collection(self):
        """Test processor metrics collection."""
        mock_host = Mock()
        mock_host._extract.return_value = ([], [], 0, None)

        mock_processor = Mock(spec=TextProcessor)
        mock_processor.get_metrics_summary.return_value = {"processor": "test", "calls": 1}

        extractor = UDExtractor(mock_host, text_processors=[mock_processor])

        metrics = extractor.get_processor_metrics()

        assert len(metrics) == 1
        assert metrics[0]["processor"] == "test"


class TestConfigurationManagement:
    """Test type-safe configuration management."""

    def test_coreference_config_defaults(self):
        """Test CoreferenceConfig default values."""
        config = CoreferenceConfig()

        assert config.enabled is False
        assert config.timeout_ms == 50
        assert config.min_text_length == 10
        assert config.model_name == "en_core_web_sm"
        assert config.fallback_enabled is True
        assert config.lang == "en"

    def test_coreference_config_validation(self):
        """Test CoreferenceConfig validation."""
        with pytest.raises(ValueError, match="timeout_ms must be positive"):
            CoreferenceConfig(timeout_ms=0)

        with pytest.raises(ValueError, match="min_text_length must be non-negative"):
            CoreferenceConfig(min_text_length=-1)

    def test_memory_config_from_env(self):
        """Test MemoryConfig creation from environment variables."""
        with patch.dict('os.environ', {
            'MEMORY_ENABLED': 'true',
            'MEMORY_BULLETS_MAX': '5',
            'MEMORY_COREFERENCE_ENABLED': 'true',
            'MEMORY_COREFERENCE_TIMEOUT_MS': '100'
        }):
            config = MemoryConfig.from_env()

            assert config.enabled is True
            assert config.bullets_max == 5
            assert config.coreference.enabled is True
            assert config.coreference.timeout_ms == 100

    def test_config_validation(self):
        """Test configuration validation function."""
        # Valid configuration
        valid_config = MemoryConfig()
        issues = validate_memory_config(valid_config)
        assert len(issues) == 0

        # Invalid configuration - coreference enabled but processors disabled
        invalid_config = MemoryConfig()
        invalid_config.coreference.enabled = True
        invalid_config.processors.enabled = False

        issues = validate_memory_config(invalid_config)
        assert len(issues) > 0
        assert any("processors to be enabled" in issue for issue in issues)

    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = MemoryConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "enabled" in config_dict
        assert "coreference" in config_dict
        assert isinstance(config_dict["coreference"], dict)


class TestIntegration:
    """Integration tests for the complete coreference system."""

    def test_full_pipeline_integration(self):
        """Test the complete pipeline with mocked components."""
        # This test verifies that all components work together
        # In a real scenario, this would use actual spaCy models

        # Create mock host
        mock_host = Mock()
        mock_doc = Mock(spec=Doc)
        mock_doc.text = "John went to the store. He bought milk."
        mock_host._extract.return_value = (["john", "store"], [("john", "went_to", "store")], 0, mock_doc)

        # Create processor with mocked NLP
        processor = CoreferenceProcessor(timeout_ms=100)

        # Create extractor with processor
        extractor = UDExtractor(mock_host, text_processors=[processor])

        # Mock the NLP loading to avoid actual model dependencies
        with patch('server.core.memory.processors.coreference.get_nlp_with_coref') as mock_get_coref:
            with patch('server.core.memory.processors.coreference.get_nlp_model') as mock_get_clean:
                # Mock coreference-enabled model
                mock_coref_nlp = Mock()
                mock_coref_doc = Mock(spec=Doc)
                mock_coref_doc.text = "John went to the store. He bought milk."
                mock_coref_doc._ = Mock()
                mock_coref_doc._.coref_clusters = []  # No clusters for this test
                mock_coref_nlp.return_value = mock_coref_doc
                mock_get_coref.return_value = mock_coref_nlp

                # Mock clean model
                mock_clean_nlp = Mock()
                mock_get_clean.return_value = mock_clean_nlp

                # Run extraction
                result = extractor.extract("John went to the store. He bought milk.", "en")

                # Verify the pipeline ran
                assert result is not None
                entities, triples, neg_count, doc = result
                assert isinstance(entities, list)
                assert isinstance(triples, list)

    @pytest.mark.integration
    def test_configuration_integration(self):
        """Test that configuration properly controls behavior."""
        config = MemoryConfig()
        config.coreference.enabled = True
        config.coreference.timeout_ms = 25

        # Verify configuration affects processor creation
        processor = CoreferenceProcessor(
            timeout_ms=config.coreference.timeout_ms,
            min_text_length=config.coreference.min_text_length,
            lang=config.coreference.lang
        )

        assert processor.timeout_ms == 25
        assert processor.min_text_length == config.coreference.min_text_length