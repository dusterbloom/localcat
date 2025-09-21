"""
Centralized configuration management for voice agent.

Provides structured configuration with environment variable support,
validation, and preset configurations for different deployment scenarios.
"""

from .settings import VoiceAgentConfig

__all__ = ["VoiceAgentConfig"]