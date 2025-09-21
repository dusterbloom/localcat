# Voice Agent Project Restructuring Plan

## Executive Summary

Based on comprehensive analysis from agent-architect and tech-debt-guardian, this plan addresses the critical technical debt in our 105+ file voice agent codebase while preserving all experimental work and establishing clean production defaults.

## Current State Analysis

### Critical Issues Identified
- **Code Duplication**: 7 TTS implementations sharing 80%+ common code (1,757 lines total)
- **Configuration Chaos**: 90+ environment variables scattered across 16 files
- **Testing Debt**: 35 test files inconsistently organized across multiple directories
- **SOLID Violations**: God classes like `FastTextAggregator` handling multiple responsibilities
- **Experiment Sprawl**: Valuable research mixed with production code

### Performance Achievements to Preserve
- ✅ Sub-800ms voice-to-voice latency achieved
- ✅ ProfessionalKokoroTTSService eliminates audio artifacts
- ✅ Kyutai streaming STT with <150ms latency
- ✅ 60% code reduction from previous consolidation

## Target Architecture

### Production Stack (Default)
```
Primary: Kyutai STT → Gemma3n LLM → Professional Kokoro TTS
Backup:  Whisper-MLX STT → Gemma3n LLM → Kokoro-MLX TTS
```

### New Directory Structure
```
server/
├── core/                          # Production-ready components
│   ├── stt/
│   │   ├── __init__.py
│   │   ├── base_stt.py           # Abstract base class
│   │   ├── kyutai_streaming.py   # Default STT
│   │   └── whisper_mlx.py        # Backup STT
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── base_tts.py           # Abstract base class
│   │   ├── kokoro_professional.py # Default TTS
│   │   └── kokoro_mlx.py         # Backup TTS
│   ├── llm/
│   │   ├── __init__.py
│   │   └── gemma_service.py      # LLM integration
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── hotpath_processor.py
│   │   ├── memory_store.py
│   │   └── session_tracker.py
│   └── pipeline/
│       ├── __init__.py
│       ├── voice_agent.py        # Main pipeline
│       └── frame_processors.py   # Custom processors
├── config/
│   ├── __init__.py
│   ├── settings.py               # Centralized configuration
│   ├── presets/
│   │   ├── minimal.env           # Basic setup
│   │   ├── default.env           # Recommended setup
│   │   └── advanced.env          # Full features
│   └── validation.py             # Config validation
├── experiments/                   # Archived research
│   ├── tts_engines/
│   │   ├── 2024_marvis_exploration/
│   │   ├── 2024_moshi_experiments/
│   │   └── 2024_piper_testing/
│   ├── stt_engines/
│   │   └── 2024_whisper_variants/
│   ├── optimization/
│   │   ├── chunking_strategies/
│   │   ├── latency_improvements/
│   │   └── memory_optimization/
│   └── audio_processing/
│       ├── artifact_fixes/
│       └── quality_improvements/
├── tests/
│   ├── unit/
│   │   ├── test_stt/
│   │   ├── test_tts/
│   │   ├── test_memory/
│   │   └── test_pipeline/
│   ├── integration/
│   │   ├── test_e2e_streaming.py
│   │   └── test_pipeline_integration.py
│   ├── performance/
│   │   ├── benchmark_latency.py
│   │   └── stress_tests.py
│   └── fixtures/
├── tools/
│   ├── __init__.py
│   ├── text_formatter.py
│   ├── audio_processor.py
│   └── diagnostics.py
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py
│   ├── health_checks.py
│   └── alerts.py
├── docs/
│   ├── architecture.md
│   ├── optimization_guide.md
│   ├── troubleshooting.md
│   └── api_reference.md
├── bot.py                        # Main entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Implementation Phases

### Phase 1: Foundation Cleanup (Week 1-2)
**Priority: CRITICAL**

#### 1.1 Create Core Directory Structure
- [ ] Create new core/ directory with subdirectories
- [ ] Create config/ directory with centralized settings
- [ ] Create experiments/ directory with chronological organization

#### 1.2 Archive Experimental Files
**Files to Archive:**
```
experiments/tts_engines/2024_marvis_exploration/
├── marvis_worker.py
├── test_marvis_*.py
└── marvis_optimization_notes.md

experiments/tts_engines/2024_moshi_experiments/
├── tts_moshi_*.py
├── test_moshi.py
└── moshi_findings.md

experiments/tts_engines/2024_piper_testing/
├── tts_piper_*.py
├── test_piper.py
└── piper_evaluation.md

experiments/audio_processing/artifact_fixes/
├── test_sentence_ending_analysis.py
├── analyze_audio_artifacts.py
├── debug_text_processing.py
├── investigate_kokoro_params.py
└── professional_fix_validation/
```

#### 1.3 Centralize Configuration
Create `config/settings.py`:
```python
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class VoiceAgentConfig:
    # STT Configuration
    stt_engine: str = "kyutai"  # kyutai | whisper_mlx
    stt_model: str = "kyutai/moshi-mlx"
    stt_chunk_length_ms: int = 100

    # TTS Configuration
    tts_engine: str = "kokoro_professional"  # kokoro_professional | kokoro_mlx
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    tts_fade_duration_ms: float = 50.0

    # LLM Configuration
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "gemma3n-4b"

    # Performance Targets
    target_voice_to_voice_latency_ms: int = 800
    target_stt_latency_ms: int = 150
    target_tts_ttfb_ms: int = 487

    # Memory Configuration
    memory_enabled: bool = True
    hotpath_enabled: bool = True
    session_persistence: bool = True

    @classmethod
    def from_env(cls) -> 'VoiceAgentConfig':
        """Load configuration from environment variables."""
        return cls(
            stt_engine=os.getenv("STT_ENGINE", cls.stt_engine),
            tts_engine=os.getenv("TTS_ENGINE", cls.tts_engine),
            # ... map all environment variables
        )
```

### Phase 2: TTS Consolidation (Week 2-3)
**Priority: HIGH**

#### 2.1 Create TTS Base Class
Create `core/tts/base_tts.py`:
```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator
import numpy as np
from pipecat.frames.frames import Frame
from pipecat.services.tts_service import TTSService

class BaseTTSService(TTSService, ABC):
    """Abstract base class for all TTS implementations."""

    def __init__(self, voice: str, speed: float, sample_rate: int, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._voice = voice
        self._speed = speed
        self._sample_rate = sample_rate

    @abstractmethod
    async def _generate_audio(self, text: str) -> tuple[np.ndarray, int]:
        """Generate raw audio for text. Must be implemented by subclasses."""
        pass

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Standard TTS pipeline with professional audio processing."""
        # Common implementation using _generate_audio()
        pass
```

#### 2.2 Refactor TTS Implementations
- [ ] Move `tts_kokoro_professional.py` → `core/tts/kokoro_professional.py`
- [ ] Move `tts_mlx_kokoro.py` → `core/tts/kokoro_mlx.py`
- [ ] Refactor both to extend `BaseTTSService`
- [ ] Remove duplicated audio processing code

#### 2.3 Archive Legacy TTS Files
Move to `experiments/tts_engines/2024_legacy/`:
- `tts_native_kokoro.py`
- `tts_hybrid_service.py`
- `fastapi_*.py` files
- All `test_tts_*.py` files

### Phase 3: STT Consolidation (Week 3-4)
**Priority: HIGH**

#### 3.1 Create STT Base Class
Similar pattern to TTS consolidation.

#### 3.2 Establish Production STT Services
- [ ] Move `kyutai_streaming_stt.py` → `core/stt/kyutai_streaming.py`
- [ ] Create `core/stt/whisper_mlx.py` (backup implementation)
- [ ] Refactor to use common base class

### Phase 4: Pipeline Integration (Week 4-5)
**Priority: MEDIUM**

#### 4.1 Create Main Pipeline Service
Create `core/pipeline/voice_agent.py`:
```python
from ..stt.kyutai_streaming import KyutaiStreamingSTT
from ..tts.kokoro_professional import ProfessionalKokoroTTS
from ..llm.gemma_service import GemmaLLMService
from ..memory.hotpath_processor import HotPathProcessor

class VoiceAgentPipeline:
    """Main voice agent pipeline with configurable components."""

    def __init__(self, config: VoiceAgentConfig):
        self.config = config
        self._setup_components()

    def _setup_components(self):
        # Initialize STT, TTS, LLM based on config
        pass

    async def create_pipeline(self) -> Pipeline:
        # Create Pipecat pipeline with all components
        pass
```

#### 4.2 Update Main Entry Point
Update `bot.py` to use new architecture:
```python
from core.pipeline.voice_agent import VoiceAgentPipeline
from config.settings import VoiceAgentConfig

async def main():
    config = VoiceAgentConfig.from_env()
    agent = VoiceAgentPipeline(config)
    pipeline = await agent.create_pipeline()
    # ... rest of main logic
```

### Phase 5: Testing & Validation (Week 5-6)
**Priority: MEDIUM**

#### 5.1 Reorganize Test Structure
- [ ] Move all tests to `tests/` directory with proper categorization
- [ ] Create comprehensive integration tests
- [ ] Add performance regression tests

#### 5.2 Create Migration Validation
- [ ] Ensure latency targets still met (<800ms voice-to-voice)
- [ ] Validate audio quality with ProfessionalKokoroTTS
- [ ] Test fallback mechanisms (Whisper-MLX + Kokoro-MLX)

## Migration Strategy

### Backward Compatibility
During migration, maintain old imports through adapter pattern:
```python
# In root __init__.py
from core.tts.kokoro_professional import ProfessionalKokoroTTSService
# Allows existing bot.py imports to continue working
```

### Validation Checkpoints
1. **After Phase 1**: All files archived, new structure created
2. **After Phase 2**: TTS consolidation complete, audio quality maintained
3. **After Phase 3**: STT consolidation complete, latency targets met
4. **After Phase 4**: Full pipeline integration working
5. **After Phase 5**: All tests passing, performance validated

## Success Metrics

### Technical Debt Reduction
- [ ] Reduce TTS implementation code by 60% (from 1,757 lines)
- [ ] Centralize 90+ environment variables into single config system
- [ ] Organize 35 test files into structured hierarchy
- [ ] Achieve 90% test coverage for core components

### Performance Preservation
- [ ] Maintain <800ms voice-to-voice latency
- [ ] Preserve ProfessionalKokoroTTS audio quality improvements
- [ ] Keep Kyutai STT <150ms latency performance

### Developer Experience
- [ ] Reduce onboarding time by 50% through clear structure
- [ ] Enable easy A/B testing between STT/TTS implementations
- [ ] Provide comprehensive documentation and troubleshooting guides

## Risk Mitigation

### High Risk Areas
1. **Pipeline Integration**: Risk of breaking existing functionality
   - **Mitigation**: Maintain adapter layer during transition

2. **Performance Regression**: Risk of latency increases
   - **Mitigation**: Comprehensive benchmarking before/after each phase

3. **Configuration Conflicts**: Risk of breaking existing deployments
   - **Mitigation**: Graceful fallback to environment variables

### Rollback Plan
Each phase includes rollback procedures to previous working state if issues arise.

## Next Steps

1. **Immediate**: Begin Phase 1 foundation cleanup
2. **Week 1**: Complete directory structure and experimental archival
3. **Week 2**: Start TTS consolidation with base class pattern
4. **Week 3**: Continue with STT consolidation
5. **Week 4**: Integrate full pipeline
6. **Week 5**: Comprehensive testing and validation

This restructuring will transform your voice agent from an experimental codebase into a production-ready, maintainable system while preserving all the valuable optimization work you've completed.