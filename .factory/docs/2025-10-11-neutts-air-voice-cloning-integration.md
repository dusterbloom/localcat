# NeuTTS Air Voice Cloning Integration Plan

## Overview
Integrate NeuTTS Air as a voice cloning alternative to Kokoro TTS, providing instant voice cloning with 3-second reference audio while maintaining sub-800ms latency targets.

## Key Components
1. **Core Service**: `NeuTTSAirService` class with Pipecat TTSService compatibility
2. **Reference Management**: Speaker profile system for audio samples and encoded references  
3. **Process Isolation**: Worker process to avoid Metal conflicts (leveraging existing pattern)
4. **Factory Integration**: Extend `create_tts_service()` in `core/factory.py`
5. **Configuration**: Environment variables in `config/settings.py`

## Implementation Phases
**Week 1-2**: Core service and process isolation  
**Week 3**: Factory integration and configuration  
**Week 4**: Performance optimization and quality assurance  
**Week 5**: Testing and documentation  

## Technical Approach
- **Voice Cloning**: 3-15 second reference audio → encoded reference → synthesis
- **Streaming**: Real-time generation with overlap-add for low latency
- **Model**: 0.5B parameters, GGUF format, Apple Silicon optimized
- **Fallback**: Automatic Kokoro fallback for reliability

## Success Criteria
- Maintain <800ms voice-to-voice latency
- Voice cloning similarity >85% MOS  
- <50ms additional overhead vs current system
- 99.9% reliability with fallback support

The plan leverages existing architecture patterns to minimize risk while adding significant voice cloning capabilities to the voice agent system.