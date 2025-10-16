"""
Unified configuration package for voice agent system.

Provides modular, composable configuration with:
- Type-safe parsing utilities
- Base classes for all config sections
- Validation and error handling
- Backward compatibility with legacy env vars
"""

from .base_config import (
    BaseConfiguration,
    ConfigurationError,
    LLMConfiguration,
    STTConfiguration,
    TTSConfiguration,
    VisionConfiguration,
)
from .parsers import (
    _parse_bool,
    _parse_int,
    _parse_float,
    _parse_list,
    _parse_enum,
    parse_with_validator,
)
from .settings import VoiceAgentConfig

__all__ = [
    # Base classes
    "BaseConfiguration",
    "ConfigurationError",
    # Component configs
    "LLMConfiguration",
    "STTConfiguration",
    "TTSConfiguration",
    "VisionConfiguration",
    "VoiceAgentConfig",
    # Parsers
    "_parse_bool",
    "_parse_int",
    "_parse_float",
    "_parse_list",
    "_parse_enum",
    "parse_with_validator",
]
