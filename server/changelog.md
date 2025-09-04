# Changelog

All notable changes to LocalCat Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-09-04

### Added
- Initial LocalCat voice agent implementation
- WebRTC transport for real-time audio communication
- WhisperSTTServiceMLX for speech-to-text
- TTSMLXIsolated for text-to-speech with Kokoro model
- Smart turn detection with CoreML analyzer
- Custom mem0 memory service integration
- Two-model architecture support (conversation + memory extraction)
- Dynamic JSON schema detection for memory operations
- Graceful fallback mechanisms for local LLM compatibility
- FAISS vector storage for persistent memories
- Environment-based configuration system

### Fixed
- mem0 async_mode parameter compatibility with Pipecat
- LM Studio JSON schema format requirements (json_object → json_schema)
- System instructions being stored as user memories
- Empty JSON response handling for local models
- Thinking token interference with conversation flow

### Technical
- OpenAI-compatible API support via Ollama/LM Studio
- Support for gemma3:4b (conversation) and qwen2.5-7b-instruct (memory)
- Embedding support with nomic-embed-text model
- Portable solution without site-packages modifications

[Unreleased]: https://github.com/peppi/localcat/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/peppi/localcat/releases/tag/v0.1.0
