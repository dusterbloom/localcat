"""
Phase 0 characterization tests for current ServiceFactory behavior.

These tests lock in observable behavior before refactoring. They focus on:
- STT fallback order and handling for specific engines
- TTS fallback for unknown engines
- Caching semantics (LLM cached, STT/TTS not cached by ServiceFactory)

All heavy dependencies are patched with lightweight dummies to keep tests fast
and independent of external packages.
"""

import types
import sys

import pytest

from config import VoiceAgentConfig


class _Dummy:
    """Simple dummy object to tag instances by type."""
    def __init__(self, tag: str = ""):
        self._tag = tag


@pytest.mark.unit
def test_stt_unknown_engine_unknown_prefers_macos_then_whisper(monkeypatch):
    from core.factories import service_factory as sf

    # Provide macOS native STT and Whisper fallbacks
    mod_macos = types.ModuleType("core.stt.macos_native")
    mod_macos.MacOSNativeSTT = lambda *a, **k: _Dummy("macos_native")
    monkeypatch.setitem(sys.modules, "core.stt.macos_native", mod_macos)

    monkeypatch.setattr(sf, "MLXModel", types.SimpleNamespace(MEDIUM="MEDIUM"), raising=True)
    monkeypatch.setattr(sf, "WhisperSTTServiceMLX", lambda *a, **k: _Dummy("whisper_mlx"), raising=True)

    cfg = VoiceAgentConfig()
    cfg.stt_engine = "nonexistent_engine"

    factory = sf.ServiceFactory(cfg)
    stt = factory.create_stt_service()
    # On macOS with macOS-native available, prefer macOS-native; otherwise Whisper MLX
    assert isinstance(stt, _Dummy)
    assert stt._tag in {"macos_native", "whisper_mlx"}


@pytest.mark.unit
def test_stt_parakeet_isolated_fallbacks_to_streaming(monkeypatch):
    from core.factories import service_factory as sf

    # Prepare fake modules to avoid importing heavy/real deps
    mod_iso = types.ModuleType("core.stt.parakeet_isolated")
    mod_iso.ParakeetIsolatedSTT = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail iso"))
    monkeypatch.setitem(sys.modules, "core.stt.parakeet_isolated", mod_iso)

    mod_stream = types.ModuleType("core.stt.parakeet_streaming")
    mod_stream.ParakeetStreamingSTT = lambda *a, **k: _Dummy("parakeet_streaming")
    monkeypatch.setitem(sys.modules, "core.stt.parakeet_streaming", mod_stream)

    # Whisper fallback should NOT be used in this test path
    monkeypatch.setattr(sf, "WhisperSTTServiceMLX", lambda *a, **k: _Dummy("whisper_mlx"), raising=True)
    monkeypatch.setattr(sf, "MLXModel", types.SimpleNamespace(MEDIUM="MEDIUM"), raising=True)

    cfg = VoiceAgentConfig()
    cfg.stt_engine = "parakeet_isolated"

    factory = sf.ServiceFactory(cfg)
    stt = factory.create_stt_service()
    assert isinstance(stt, _Dummy) and stt._tag == "parakeet_streaming"


@pytest.mark.unit
def test_stt_parakeet_isolated_all_failures_fall_back_to_macos_or_whisper(monkeypatch):
    from core.factories import service_factory as sf

    # Fake modules so imports don't load heavy backends
    mod_iso = types.ModuleType("core.stt.parakeet_isolated")
    mod_iso.ParakeetIsolatedSTT = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail iso"))
    monkeypatch.setitem(sys.modules, "core.stt.parakeet_isolated", mod_iso)

    mod_stream = types.ModuleType("core.stt.parakeet_streaming")
    mod_stream.ParakeetStreamingSTT = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail str"))
    monkeypatch.setitem(sys.modules, "core.stt.parakeet_streaming", mod_stream)

    # Provide Whisper MLX fallback target
    monkeypatch.setattr(sf, "MLXModel", types.SimpleNamespace(MEDIUM="MEDIUM"), raising=True)
    monkeypatch.setattr(sf, "WhisperSTTServiceMLX", lambda *a, **k: _Dummy("whisper_mlx"), raising=True)

    # Optionally provide macOS native (if available it should be preferred)
    mod_macos = types.ModuleType("core.stt.macos_native")
    mod_macos.MacOSNativeSTT = lambda *a, **k: _Dummy("macos_native")
    monkeypatch.setitem(sys.modules, "core.stt.macos_native", mod_macos)

    cfg = VoiceAgentConfig()
    cfg.stt_engine = "parakeet_isolated"

    factory = sf.ServiceFactory(cfg)
    stt = factory.create_stt_service()
    assert isinstance(stt, _Dummy) and stt._tag in {"macos_native", "whisper_mlx"}


@pytest.mark.unit
def test_stt_whisper_mlx_direct_success(monkeypatch):
    from core.factories import service_factory as sf

    # Inject fake whisper_mlx module
    mod_wmlx = types.ModuleType("core.stt.whisper_mlx")
    mod_wmlx.DirectMLXWhisperSTT = lambda *a, **k: _Dummy("direct_mlx_whisper")
    monkeypatch.setitem(sys.modules, "core.stt.whisper_mlx", mod_wmlx)

    cfg = VoiceAgentConfig()
    cfg.stt_engine = "whisper_mlx_direct"

    factory = sf.ServiceFactory(cfg)
    stt = factory.create_stt_service()
    assert isinstance(stt, _Dummy) and stt._tag == "direct_mlx_whisper"


@pytest.mark.unit
def test_stt_whisper_mlx_direct_failure_falls_back_to_macos_or_whisper(monkeypatch):
    from core.factories import service_factory as sf

    # Direct MLX Whisper raises, should fall back to WhisperSTTServiceMLX
    mod_wmlx = types.ModuleType("core.stt.whisper_mlx")
    mod_wmlx.DirectMLXWhisperSTT = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "core.stt.whisper_mlx", mod_wmlx)
    monkeypatch.setattr(sf, "MLXModel", types.SimpleNamespace(MEDIUM="MEDIUM"), raising=True)
    monkeypatch.setattr(sf, "WhisperSTTServiceMLX", lambda *a, **k: _Dummy("whisper_mlx"), raising=True)

    cfg = VoiceAgentConfig()
    cfg.stt_engine = "whisper_mlx_direct"

    factory = sf.ServiceFactory(cfg)
    # Optionally provide macOS native, which should be preferred if available
    mod_macos = types.ModuleType("core.stt.macos_native")
    mod_macos.MacOSNativeSTT = lambda *a, **k: _Dummy("macos_native")
    monkeypatch.setitem(sys.modules, "core.stt.macos_native", mod_macos)

    stt = factory.create_stt_service()
    assert isinstance(stt, _Dummy) and stt._tag in {"macos_native", "whisper_mlx"}


@pytest.mark.unit
def test_tts_unknown_engine_falls_back_to_mlx_kokoro(monkeypatch):
    from core.factories import service_factory as sf

    # Patch Siri and MLX Kokoro options
    monkeypatch.setattr(sf.ServiceFactory, "_try_create_siri_tts", lambda *a, **k: _Dummy("siri"), raising=True)
    monkeypatch.setattr(sf, "MLXKokoroTTSService", lambda *a, **k: _Dummy("kokoro_mlx"), raising=True)

    cfg = VoiceAgentConfig()
    cfg.tts_engine = "unknown"

    factory = sf.ServiceFactory(cfg)
    tts = factory.create_tts_service()
    assert isinstance(tts, _Dummy) and tts._tag in {"siri", "kokoro_mlx"}


@pytest.mark.unit
def test_llm_service_is_cached(monkeypatch):
    from core.factories import service_factory as sf

    # Patch OpenAILLMService constructor to produce unique objects
    class _LLMDummy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def create_context_aggregator(self, *a, **k):
            return _Dummy("ctx")

    monkeypatch.setattr(sf, "OpenAILLMService", _LLMDummy, raising=True)
    # Avoid actual prewarm side-effect
    monkeypatch.setattr(sf, "_prewarm_llm_service", lambda *a, **k: None, raising=True)

    cfg = VoiceAgentConfig()
    factory = sf.ServiceFactory(cfg)

    llm1 = factory.create_llm_service()
    llm2 = factory.create_llm_service()
    assert llm1 is llm2  # Cached instance reused


@pytest.mark.unit
def test_stt_and_tts_not_cached_in_servicefactory(monkeypatch):
    from core.factories import service_factory as sf

    # Fake streaming STT module and class
    mod_stream = types.ModuleType("core.stt.parakeet_streaming")
    mod_stream.ParakeetStreamingSTT = lambda *a, **k: _Dummy("stt_instance")
    monkeypatch.setitem(sys.modules, "core.stt.parakeet_streaming", mod_stream)

    monkeypatch.setattr(sf, "WhisperSTTServiceMLX", lambda *a, **k: _Dummy("whisper_mlx"), raising=True)
    monkeypatch.setattr(sf, "MLXModel", types.SimpleNamespace(MEDIUM="MEDIUM"), raising=True)

    monkeypatch.setattr(sf, "MLXKokoroTTSService", lambda *a, **k: _Dummy("tts_instance"), raising=True)

    cfg = VoiceAgentConfig()
    cfg.stt_engine = "parakeet_streaming"
    cfg.tts_engine = "kokoro_mlx"

    factory = sf.ServiceFactory(cfg)

    # Force caching off to characterize legacy behavior
    monkeypatch.setenv("SERVICE_FACTORY_CACHE_STT", "false")
    monkeypatch.setenv("SERVICE_FACTORY_CACHE_TTS", "false")

    stt1 = factory.create_stt_service()
    stt2 = factory.create_stt_service()
    tts1 = factory.create_tts_service()
    tts2 = factory.create_tts_service()

    assert stt1 is not stt2
    assert tts1 is not tts2


@pytest.mark.unit
def test_clear_cache_resets_llm_cache(monkeypatch):
    from core.factories import service_factory as sf

    # Patch OpenAILLMService so we can see identity changes
    class _LLMDummy:
        pass

    monkeypatch.setattr(sf, "OpenAILLMService", lambda **k: _LLMDummy(), raising=True)
    monkeypatch.setattr(sf, "_prewarm_llm_service", lambda *a, **k: None, raising=True)

    cfg = VoiceAgentConfig()
    factory = sf.ServiceFactory(cfg)

    llm1 = factory.create_llm_service()
    factory.clear_cache()
    llm2 = factory.create_llm_service()
    assert llm1 is not llm2
