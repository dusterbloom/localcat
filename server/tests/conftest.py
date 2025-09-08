"""
Test configuration and utilities for LocalCat tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from config.config import Config, EnvironmentType, get_config
from components.memory.memory_store import MemoryStore
from processors.memory_processor import MemoryProcessor, MemoryProcessorConfig
from processors.extraction_processor import ExtractionProcessor, ExtractionProcessorConfig
from processors.quality_processor import QualityProcessor, QualityProcessorConfig
from processors.context_processor import ContextProcessor, ContextProcessorConfig
from core.pipeline_builder import PipelineBuilder, PipelineConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration"""
    config = Config()
    config.environment = EnvironmentType.TESTING
    config.database.sqlite_path = str(temp_dir / "test_memory.db")
    config.database.lmdb_dir = str(temp_dir / "test_lmdb")
    config.development.enable_test_mode = True
    config.monitoring.log_level = "DEBUG"
    return config


@pytest.fixture
def memory_store(test_config):
    """Create memory store for tests"""
    store = MemoryStore(
        sqlite_path=test_config.database.sqlite_path,
        lmdb_dir=test_config.database.lmdb_dir
    )
    yield store
    # Cleanup
    import asyncio
    asyncio.run(store.close())


@pytest.fixture
def memory_processor(test_config, memory_store):
    """Create memory processor for tests"""
    config = MemoryProcessorConfig(
        sqlite_path=test_config.database.sqlite_path,
        lmdb_dir=test_config.database.lmdb_dir,
        user_id="test-user",
        enable_metrics=False
    )
    return MemoryProcessor(config)


@pytest.fixture
def extraction_processor(test_config):
    """Create extraction processor for tests"""
    config = ExtractionProcessorConfig(
        default_strategy="lightweight",
        fallback_strategy="enhanced_hotmem",
        enable_multi_strategy=False,
        enable_metrics=False
    )
    return ExtractionProcessor(config)


@pytest.fixture
def quality_processor(test_config):
    """Create quality processor for tests"""
    config = QualityProcessorConfig(
        min_confidence_threshold=0.5,
        min_overall_quality_threshold=0.4,
        enable_correction=False,
        enable_metrics=False
    )
    return QualityProcessor(config)


@pytest.fixture
def context_processor(test_config):
    """Create context processor for tests"""
    config = ContextProcessorConfig(
        max_context_items=20,
        max_context_tokens=500,
        enable_metrics=False
    )
    return ContextProcessor(config)


@pytest.fixture
def pipeline_builder(test_config):
    """Create pipeline builder for tests"""
    config = PipelineConfig(
        enable_memory=True,
        enable_extraction=True,
        enable_quality=True,
        enable_context=True,
        enable_metrics=False
    )
    return PipelineBuilder(config)


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return "Hello, my name is John Doe and I work at Acme Corporation in New York."


@pytest.fixture
def sample_entities():
    """Sample entities for testing"""
    return [
        {"text": "John Doe", "label": "PERSON", "confidence": 0.9},
        {"text": "Acme Corporation", "label": "ORG", "confidence": 0.8},
        {"text": "New York", "label": "GPE", "confidence": 0.7}
    ]


@pytest.fixture
def sample_facts():
    """Sample facts for testing"""
    return [
        {"text": "John Doe works at Acme Corporation", "confidence": 0.85},
        {"text": "Acme Corporation is located in New York", "confidence": 0.75}
    ]


@pytest.fixture
def sample_extraction_result(sample_entities, sample_facts):
    """Sample extraction result for testing"""
    return {
        "text": "Hello, my name is John Doe and I work at Acme Corporation in New York.",
        "entities": sample_entities,
        "facts": sample_facts,
        "relations": [],
        "confidence": 0.8,
        "strategy_used": "test_strategy"
    }


class MockFrame:
    """Mock frame for testing"""
    def __init__(self, text="", metadata=None):
        self.text = text
        self.metadata = metadata or {}


class MockTranscriptionFrame(MockFrame):
    """Mock transcription frame"""
    pass


class MockTextFrame(MockFrame):
    """Mock text frame"""
    pass


class MockLLMMessagesFrame:
    """Mock LLM messages frame"""
    def __init__(self, messages):
        self.messages = messages


@pytest.fixture
def mock_transcription_frame(sample_text):
    """Create mock transcription frame"""
    return MockTranscriptionFrame(sample_text)


@pytest.fixture
def mock_text_frame(sample_text):
    """Create mock text frame"""
    return MockTextFrame(sample_text)


@pytest.fixture
def mock_llm_messages_frame():
    """Create mock LLM messages frame"""
    return MockLLMMessagesFrame([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ])


@pytest.fixture
def mock_push_frame():
    """Mock push_frame function"""
    pushed_frames = []
    
    async def push_frame(frame, direction=None):
        pushed_frames.append(frame)
    
    return push_frame, pushed_frames


# Test utilities
def create_test_memory_store(temp_dir):
    """Create a test memory store"""
    return MemoryStore(
        sqlite_path=str(temp_dir / "test.db"),
        lmdb_dir=str(temp_dir / "test_lmdb")
    )


def create_test_pipeline_config():
    """Create test pipeline configuration"""
    return PipelineConfig(
        enable_memory=True,
        enable_extraction=True,
        enable_quality=True,
        enable_context=True,
        enable_metrics=False
    )


def assert_processor_metrics(processor, expected_metrics):
    """Assert processor has expected metrics"""
    metrics = processor.get_metrics()
    for key, expected_value in expected_metrics.items():
        assert metrics.get(key) == expected_value, f"Metric {key} mismatch"


def async_test(coro):
    """Decorator to make async functions work with pytest"""
    def wrapper(*args, **kwargs):
        import asyncio
        return asyncio.run(coro(*args, **kwargs))
    return wrapper


# Performance test utilities
import time
from contextlib import contextmanager


@contextmanager
def measure_time():
    """Measure execution time"""
    start = time.time()
    yield
    end = time.time()
    return end - start


def measure_memory_usage():
    """Measure memory usage"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def benchmark_function(func, *args, iterations=100, **kwargs):
    """Benchmark a function"""
    times = []
    
    for _ in range(iterations):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        times.append(end - start)
    
    return {
        'min_time': min(times),
        'max_time': max(times),
        'avg_time': sum(times) / len(times),
        'total_time': sum(times),
        'iterations': iterations,
        'result': result
    }


# Test data generators
def generate_test_text(length=100):
    """Generate test text of specified length"""
    import random
    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    return " ".join(random.choice(words) for _ in range(length))


def generate_test_entities(count=5):
    """Generate test entities"""
    import random
    names = ["John", "Jane", "Bob", "Alice", "Charlie"]
    orgs = ["Acme", "TechCorp", "DataInc", "SysCo", "NetOrg"]
    locations = ["New York", "London", "Tokyo", "Paris", "Berlin"]
    
    entities = []
    for i in range(count):
        entity_type = random.choice(["PERSON", "ORG", "GPE"])
        if entity_type == "PERSON":
            text = random.choice(names) + " " + random.choice(["Smith", "Doe", "Johnson"])
        elif entity_type == "ORG":
            text = random.choice(orgs) + " " + random.choice(["Inc", "Ltd", "Corp"])
        else:
            text = random.choice(locations)
        
        entities.append({
            "text": text,
            "label": entity_type,
            "confidence": random.uniform(0.5, 1.0)
        })
    
    return entities


def generate_test_facts(count=3):
    """Generate test facts"""
    import random
    subjects = ["John", "The company", "The system", "The project", "The team"]
    verbs = ["works", "develops", "manages", "creates", "maintains"]
    objects = ["software", "data", "systems", "processes", "applications"]
    
    facts = []
    for i in range(count):
        fact = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}"
        facts.append({
            "text": fact,
            "confidence": random.uniform(0.6, 1.0)
        })
    
    return facts