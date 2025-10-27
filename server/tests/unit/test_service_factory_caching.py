"""
Unit tests for ServiceFactory caching behavior when enabled.

Validates:
- STT is cached across calls when SERVICE_FACTORY_CACHE_STT=true
- TTS is cached per 'use_boundaries' value when SERVICE_FACTORY_CACHE_TTS=true
"""

import sys
import types
import pytest

from config import VoiceAgentConfig


class _Dummy:
    def __init__(self, tag: str):
        self._tag = tag


@pytest.mark.unit
def test_stt_is_cached_when_enabled(monkeypatch):
    from core.factories import service_factory as sf

    # Enable STT caching
    monkeypatch.setenv("SERVICE_FACTORY_CACHE_STT", "true")

    # Fake streaming STT module
    mod_stream = types.ModuleType("core.stt.parakeet_streaming")
    # Return new instance each time to prove caching at factory level
    def _make_stt(*a, **k):
        return _Dummy("stt")
    mod_stream.ParakeetStreamingSTT = _make_stt
    monkeypatch.setitem(sys.modules, "core.stt.parakeet_streaming", mod_stream)

    cfg = VoiceAgentConfig()
    cfg.stt_engine = "parakeet_streaming"
    factory = sf.ServiceFactory(cfg)

    stt1 = factory.create_stt_service()
    stt2 = factory.create_stt_service()
    assert stt1 is stt2


@pytest.mark.unit
def test_tts_is_cached_per_boundaries(monkeypatch):
    from core.factories import service_factory as sf

    # Enable TTS caching
    monkeypatch.setenv("SERVICE_FACTORY_CACHE_TTS", "true")

    # Patch MLX Kokoro to return fresh objects; factory should cache
    monkeypatch.setattr(sf, "MLXKokoroTTSService", lambda *a, **k: _Dummy("tts"), raising=True)

    cfg = VoiceAgentConfig()
    cfg.tts_engine = "kokoro_mlx"
    factory = sf.ServiceFactory(cfg)

    tts_true_1 = factory.create_tts_service(use_boundaries=True)
    tts_true_2 = factory.create_tts_service(use_boundaries=True)
    tts_false_1 = factory.create_tts_service(use_boundaries=False)
    tts_false_2 = factory.create_tts_service(use_boundaries=False)

    assert tts_true_1 is tts_true_2
    assert tts_false_1 is tts_false_2
    assert tts_true_1 is not tts_false_1

