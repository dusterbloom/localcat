"""
Core production components for the voice agent.

This package contains the production-ready implementations for:
- STT (Speech-to-Text) services
- TTS (Text-to-Speech) services
- LLM (Language Model) integration
- Memory management system
- Pipeline orchestration

Default Configuration:
- STT: Kyutai Streaming
- TTS: Professional Kokoro
- LLM: Gemma3n via local OpenAI server
- Memory: HotPath with session tracking
"""

__version__ = "2.0.0"