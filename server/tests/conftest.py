"""
Pytest configuration and fixtures for LocalCat streaming tests
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path

# Add server and pipecat to path
_SERVER_ROOT = Path(__file__).parent.parent
_PIPECAT_SRC = _SERVER_ROOT / "pipecat" / "src"
for p in [_SERVER_ROOT, _PIPECAT_SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Set environment variables for testing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

# Disable model downloads in CI
if os.getenv("CI"):
    os.environ["HF_HUB_OFFLINE"] = "1"


# ============= Fixtures =============

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for tests that don't need real LLM"""
    class MockLLMService:
        def __init__(self):
            self.model = "mock-model"

        async def generate(self, text):
            return f"Mock response for: {text}"

        def create_context_aggregator(self, context, **kwargs):
            class MockAggregator:
                def user(self):
                    return self
                def assistant(self):
                    return self
                async def process_frame(self, frame, direction):
                    pass
            return MockAggregator()

    return MockLLMService()


@pytest.fixture
def mock_tts_service():
    """Mock TTS service for tests that don't need real TTS"""
    class MockTTSService:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def run_tts(self, text):
            # Return mock audio frames
            for i in range(3):
                yield {"audio": b"mock_audio_data", "chunk": i}

    return MockTTSService()


@pytest.fixture
def mock_stt_service():
    """Mock STT service for tests that don't need real STT"""
    class MockSTTService:
        async def transcribe(self, audio):
            return "mock transcription"

    return MockSTTService()


@pytest.fixture(scope="session")
def model_cache_dir():
    """Ensure model cache directory exists"""
    cache_dir = Path.home() / ".cache" / "localcat" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# ============= Markers and Skips =============

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "fast: marks tests as fast unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_models: marks tests requiring ML models")
    config.addinivalue_line("markers", "requires_llm: marks tests requiring LLM server")
    config.addinivalue_line("markers", "ci: marks tests suitable for CI/CD")
    config.addinivalue_line("markers", "skip_ci: marks tests to skip in CI")


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on environment"""
    skip_slow = pytest.mark.skip(reason="Slow test skipped (use --run-slow to run)")
    skip_models = pytest.mark.skip(reason="Model download required (use --download-models)")
    skip_ci = pytest.mark.skip(reason="Test skipped in CI environment")
    skip_llm = pytest.mark.skip(reason="LLM server not available")

    for item in items:
        # Skip slow tests unless explicitly requested
        if "slow" in item.keywords and not config.getoption("--run-slow", default=False):
            item.add_marker(skip_slow)

        # Skip model tests in CI unless models are cached
        if "requires_models" in item.keywords and os.getenv("CI"):
            if not os.getenv("MODELS_CACHED"):
                item.add_marker(skip_models)

        # Skip CI-inappropriate tests
        if "skip_ci" in item.keywords and os.getenv("CI"):
            item.add_marker(skip_ci)

        # Skip LLM tests if server not running
        if "requires_llm" in item.keywords:
            import httpx
            try:
                # Quick check if LLM server is running
                response = httpx.get("http://localhost:11434/api/tags", timeout=1)
                if response.status_code != 200:
                    item.add_marker(skip_llm)
            except:
                item.add_marker(skip_llm)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--download-models",
        action="store_true",
        default=False,
        help="Allow model downloads during tests"
    )
    parser.addoption(
        "--ci-only",
        action="store_true",
        default=False,
        help="Run only CI-appropriate tests"
    )


# ============= Test Helpers =============

@pytest.fixture
def assert_timing():
    """Helper to assert timing constraints"""
    def _assert_timing(actual_ms, expected_ms, tolerance=0.2):
        """
        Assert that actual timing is within tolerance of expected

        Args:
            actual_ms: Actual time in milliseconds
            expected_ms: Expected time in milliseconds
            tolerance: Acceptable deviation as fraction (0.2 = 20%)
        """
        lower_bound = expected_ms * (1 - tolerance)
        upper_bound = expected_ms * (1 + tolerance)
        assert lower_bound <= actual_ms <= upper_bound, \
            f"Timing {actual_ms:.1f}ms outside range [{lower_bound:.1f}, {upper_bound:.1f}]ms"

    return _assert_timing


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary audio file for testing"""
    import numpy as np
    import soundfile as sf

    audio_path = tmp_path / "test_audio.wav"

    # Generate 1 second of silence
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    audio = np.zeros(samples, dtype=np.float32)

    sf.write(audio_path, audio, sample_rate)
    return audio_path