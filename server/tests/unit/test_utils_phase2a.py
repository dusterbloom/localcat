"""
Unit tests for Phase 2a utilities: model_resolver and service_validator.
"""

import types

from core.factories.utils.model_resolver import resolve_parakeet_model_path
from core.factories.utils.service_validator import TTSServiceValidator


def test_model_resolver_no_tauri_returns_input():
    assert resolve_parakeet_model_path("foo/bar") == "foo/bar"


def test_tts_service_validator_happy_path():
    svc = types.SimpleNamespace(_pipeline=types.SimpleNamespace(lang_code="en"), _voice="abc")
    assert TTSServiceValidator().is_functional(svc) is True


def test_tts_service_validator_missing_attrs():
    assert TTSServiceValidator().is_functional(object()) is False

