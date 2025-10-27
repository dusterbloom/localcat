"""
Tests for macOS-native fallbacks (STT) and Siri fallback (TTS) enabled by default.
"""

import sys
import types
import pytest

from config import VoiceAgentConfig


class _Dummy:
    def __init__(self, tag: str):
        self._tag = tag


@pytest.mark.unit
def test_stt_parakeet_chain_falls_back_to_macos_native(monkeypatch):
    from core.factories import service_factory as sf

    # Simulate macOS
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)

    # Fail isolated and streaming
    mod_iso = types.ModuleType("core.stt.parakeet_isolated")
    mod_iso.ParakeetIsolatedSTT = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("iso fail"))
    monkeypatch.setitem(sys.modules, "core.stt.parakeet_isolated", mod_iso)

    mod_stream = types.ModuleType("core.stt.parakeet_streaming")
    mod_stream.ParakeetStreamingSTT = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stream fail"))
    monkeypatch.setitem(sys.modules, "core.stt.parakeet_streaming", mod_stream)

    # macOS native succeeds
    mod_macos = types.ModuleType("core.stt.macos_native")
    mod_macos.MacOSNativeSTT = lambda *a, **k: _Dummy("macos_native")
    monkeypatch.setitem(sys.modules, "core.stt.macos_native", mod_macos)

    # Ensure Whisper is not used
    monkeypatch.setattr(sf, "MLXModel", types.SimpleNamespace(MEDIUM="MEDIUM"), raising=True)
    monkeypatch.setattr(sf, "WhisperSTTServiceMLX", lambda *a, **k: _Dummy("whisper"), raising=True)

    cfg = VoiceAgentConfig()
    cfg.stt_engine = "parakeet_isolated"
    factory = sf.ServiceFactory(cfg)
    stt = factory.create_stt_service()
    assert isinstance(stt, _Dummy) and stt._tag == "macos_native"


@pytest.mark.unit
def test_stt_fallback_to_whisper_when_macos_native_fails(monkeypatch):
    from core.factories import service_factory as sf

    # Simulate macOS
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)

    # Fail isolated and streaming and macOS native
    mod_iso = types.ModuleType("core.stt.parakeet_isolated")
    mod_iso.ParakeetIsolatedSTT = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("iso fail"))
    monkeypatch.setitem(sys.modules, "core.stt.parakeet_isolated", mod_iso)

    mod_stream = types.ModuleType("core.stt.parakeet_streaming")
    mod_stream.ParakeetStreamingSTT = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stream fail"))
    monkeypatch.setitem(sys.modules, "core.stt.parakeet_streaming", mod_stream)

    mod_macos = types.ModuleType("core.stt.macos_native")
    mod_macos.MacOSNativeSTT = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mac fail"))
    monkeypatch.setitem(sys.modules, "core.stt.macos_native", mod_macos)

    # Whisper batch fallback
    monkeypatch.setattr(sf, "MLXModel", types.SimpleNamespace(MEDIUM="MEDIUM"), raising=True)
    monkeypatch.setattr(sf, "WhisperSTTServiceMLX", lambda *a, **k: _Dummy("whisper"), raising=True)

    cfg = VoiceAgentConfig()
    cfg.stt_engine = "parakeet_isolated"
    factory = sf.ServiceFactory(cfg)
    stt = factory.create_stt_service()
    assert isinstance(stt, _Dummy) and stt._tag == "whisper"


@pytest.mark.unit
def test_tts_falls_back_to_siri_then_last_resort(monkeypatch):
    from core.factories import service_factory as sf

    # Simulate macOS
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)

    # Primary MLX Kokoro fails
    monkeypatch.setattr(sf, "MLXKokoroTTSService", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mlx fail")), raising=True)

    # Siri fallback succeeds by monkeypatching helper
    def _fake_siri(self, tts_config, use_boundaries):
        return _Dummy("siri")
    monkeypatch.setattr(sf.ServiceFactory, "_try_create_siri_tts", _fake_siri, raising=True)

    cfg = VoiceAgentConfig()
    cfg.tts_engine = "kokoro_mlx"
    factory = sf.ServiceFactory(cfg)
    tts = factory.create_tts_service()
    assert isinstance(tts, _Dummy) and tts._tag == "siri"


@pytest.mark.unit
def test_tts_siri_failure_uses_last_resort(monkeypatch):
    from core.factories import service_factory as sf

    # Simulate macOS
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)

    # Primary MLX Kokoro fails
    monkeypatch.setattr(sf, "MLXKokoroTTSService", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mlx fail")), raising=True)

    # Siri fallback fails
    def _fail_siri(self, tts_config, use_boundaries):
        raise RuntimeError("siri fail")
    monkeypatch.setattr(sf.ServiceFactory, "_try_create_siri_tts", _fail_siri, raising=True)

    # Last resort for kokoro_mlx is Professional in our implementation
    monkeypatch.setattr(
        sf, "ProfessionalKokoroTTSService",
        lambda *a, **k: _Dummy("professional"), raising=True
    )

    cfg = VoiceAgentConfig()
    cfg.tts_engine = "kokoro_mlx"
    factory = sf.ServiceFactory(cfg)
    tts = factory.create_tts_service()
    assert isinstance(tts, _Dummy) and tts._tag == "professional"

