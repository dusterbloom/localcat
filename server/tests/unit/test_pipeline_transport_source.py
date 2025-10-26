import sys
import types
import pytest
from pipecat.processors.frame_processor import FrameProcessor


class DummyProcessor(FrameProcessor):
    async def process_frame(self, frame, direction=None):
        await super().process_frame(frame, direction)
        return frame


class DummyAggregator:
    def user(self):
        return DummyProcessor()

    def assistant(self):
        return DummyProcessor()


class DummyTransport:
    def __init__(self):
        self.input_stage = DummyProcessor()
        self.output_stage = DummyProcessor()

    def input(self):
        return self.input_stage

    def output(self):
        return self.output_stage


class DummyBackgroundWhisper(DummyProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _monkeypatch_background_whisper(monkeypatch):
    import core.stt.background_whisper as bw_module
    monkeypatch.setattr(bw_module, "BackgroundWhisperProcessor", DummyBackgroundWhisper)


@pytest.mark.asyncio
async def test_standard_pipeline_has_top_level_transport_input(monkeypatch):
    """Ensure create_pipeline starts with a top-level transport.input() source.

    Also verify it does not attempt to create another transport.input() inside
    ParallelPipeline branches by ensuring pipeline construction completes while
    using a dummy BackgroundWhisper (no MLX dependency).
    """
    # Monkeypatch BackgroundWhisperProcessor to avoid MLX import/initialization
    _monkeypatch_background_whisper(monkeypatch)

    # Build minimal services with dummy processors
    services = {
        "stt": DummyProcessor(),
        "rtvi": DummyProcessor(),
        "memory": DummyProcessor(),
        "context_aggregator": DummyAggregator(),
        "llm": DummyProcessor(),
        "text_aggregator": DummyProcessor(),
        "tts": DummyProcessor(),
        "audio_intelligence": None,
        "mic_probe": None,
        "context": None,
    }

    # Minimal config: disable intro/video paths so standard pipeline is used
    from config import VoiceAgentConfig
    cfg = VoiceAgentConfig()
    cfg.enable_intro_pipeline = False
    cfg.video_input_enabled = False

    from core.factory import VoiceAgentFactory
    factory = VoiceAgentFactory(cfg)

    transport = DummyTransport()
    pipeline = factory.create_pipeline(transport, services)

    # Access first linked stage after Pipeline source
    first_stage = getattr(pipeline, "_source")._next
    assert first_stage is transport.input_stage, "Pipeline must start with transport.input()"

    # Walk the chain and assert a ParallelPipeline exists
    from pipecat.pipeline.parallel_pipeline import ParallelPipeline
    cursor = first_stage
    found_parallel = False
    # Follow single-linked list via _next until sink
    while cursor is not None and hasattr(cursor, "_next"):
        if isinstance(cursor, ParallelPipeline):
            found_parallel = True
            break
        cursor = cursor._next
    assert found_parallel, "ParallelPipeline not found in stages"


@pytest.mark.asyncio
async def test_pipeline_builds_with_video_enabled(monkeypatch):
    """Ensure pipeline constructs when video is enabled by stubbing VisionContextInjector."""
    _monkeypatch_background_whisper(monkeypatch)

    # Stub core.video.VisionContextInjector to avoid PIL dependency during import
    dummy_video = types.ModuleType("core.video")
    dummy_video.VisionContextInjector = DummyProcessor
    monkeypatch.setitem(sys.modules, "core.video", dummy_video)

    services = {
        "stt": DummyProcessor(),
        "rtvi": DummyProcessor(),
        "memory": DummyProcessor(),
        "context_aggregator": DummyAggregator(),
        "llm": DummyProcessor(),
        "text_aggregator": DummyProcessor(),
        "tts": DummyProcessor(),
        "audio_intelligence": None,
        "mic_probe": None,
        "context": object(),  # enable video branch
    }

    from config import VoiceAgentConfig
    cfg = VoiceAgentConfig()
    cfg.enable_intro_pipeline = False
    cfg.video_input_enabled = True

    from core.factory import VoiceAgentFactory
    factory = VoiceAgentFactory(cfg)

    transport = DummyTransport()
    pipeline = factory.create_pipeline(transport, services)

    # Assertions via linked chain
    first_stage = getattr(pipeline, "_source")._next
    assert first_stage is transport.input_stage
    from pipecat.pipeline.parallel_pipeline import ParallelPipeline
    cursor = first_stage
    found_parallel = False
    while cursor is not None and hasattr(cursor, "_next"):
        if isinstance(cursor, ParallelPipeline):
            found_parallel = True
            break
        cursor = cursor._next
    assert found_parallel
