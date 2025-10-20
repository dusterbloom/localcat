# TTS Bug Fix Plan: EspeakWrapper Issue

## Problem Analysis

The error `type object 'EspeakWrapper' has no attribute 'set_data_path'` indicates a version mismatch in the `phonemizer` library. The neutts-air code expects an older API that no longer exists in current phonemizer versions.

## Root Cause

1. **API Change**: The `phonemizer` library updated its API, removing `EspeakWrapper.set_data_path` method
2. **Missing Dependency**: `phonemizer` is not in the main `requirements.txt` but required by neutts-air
3. **Import Path Issue**: The code tries to import from old API structure

## Solution Plan

### Phase 1: Immediate Fix (Critical)

#### 1.1 Add Missing Dependencies
```bash
# Add to requirements.txt
phonemizer==3.3.0
espeak-ng>=1.51.0
```

#### 1.2 Fix Phonemizer Initialization
Update `/server/external/neutts-air/neuttsair/neutts.py`:

**Current problematic code:**
```python
from phonemizer.backend import EspeakBackend
# ... expecting EspeakWrapper.set_data_path somewhere
```

**Fixed code:**
```python
from phonemizer.backend import EspeakBackend
# Remove any EspeakWrapper.set_data_path calls
# Use proper EspeakBackend initialization
```

#### 1.3 Update TTS Worker Integration
The TTS worker should not depend on neutts-air if it's using Kokoro. The issue is likely from import conflicts.

### Phase 2: Code Investigation (High Priority)

#### 2.1 Identify TTS Path Dependencies
```python
# Check what's actually being used:
# - kokoro_worker_optimized.py (MLX-based)
# - neutts-air (separate TTS system)
# - tts_mlx_ultra_low_latency.py (orchestrator)
```

#### 2.2 Remove Unnecessary Neutts-Air Imports
If the ultra-low-latency TTS doesn't actually use neutts-air, remove the import to avoid conflicts.

#### 2.3 Conditional TTS Loading
```python
# Make neutts-air optional
try:
    from external.neuttsair.neutts import NeuTTSAir
    NEUTTS_AVAILABLE = True
except ImportError:
    NEUTTS_AVAILABLE = False
    logger.warning("Neutts-Air not available, using Kokoro-only")
```

### Phase 3: Robust Solution (Medium Priority)

#### 3.1 Environment Detection
```python
def detect_tts_environment():
    """Detect available TTS backends and choose appropriate one"""
    backends = []
    
    # Check MLX/Kokoro
    try:
        import mlx_audio
        backends.append("kokoro")
    except ImportError:
        pass
    
    # Check Neutts-Air
    try:
        from external.neuttsair.neutts import NeuTTSAir
        backends.append("neutts")
    except ImportError:
        pass
    
    return backends
```

#### 3.2 Fallback Strategy
```python
# In tts_mlx_ultra_low_latency.py
def get_available_tts_backend():
    backends = detect_tts_environment()
    
    if "kokoro" in backends:
        return "kokoro"
    elif "neutts" in backends:
        return "neutts"
    else:
        raise RuntimeError("No TTS backend available")
```

### Phase 4: Testing & Validation (High Priority)

#### 4.1 Unit Test for Phonemizer
```python
def test_phonemizer_import():
    """Test that phonemizer works correctly"""
    try:
        from phonemizer.backend import EspeakBackend
        backend = EspeakBackend('en')
        assert backend.phonemize(['test']) == ['tˈɛst']
        logger.info("✅ Phonemizer working correctly")
    except Exception as e:
        logger.error(f"❌ Phonemizer test failed: {e}")
```

#### 4.2 TTS Integration Test
```python
def test_tts_initialization():
    """Test TTS worker initialization"""
    # Test each backend independently
    # Verify no import conflicts
    # Confirm audio generation works
```

#### 4.3 Environment Validation
```python
def test_environment_setup():
    """Validate all TTS dependencies are available"""
    required_packages = ['phonemizer', 'mlx_audio', 'espeak-ng']
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} available")
        except ImportError:
            logger.error(f"❌ {package} missing")
```

## Implementation Steps

### Step 1: Fix Dependencies (Immediate)
1. Add `phonemizer==3.3.0` and `espeak-ng>=1.51.0` to requirements.txt
2. Update environment and restart services
3. Test if error persists

### Step 2: Fix Import Issues (If Step 1 fails)
1. Examine neuttsair/neutts.py for EspeakWrapper usage
2. Update to use current phonemizer API
3. Remove deprecated method calls

### Step 3: Isolate TTS Systems (If needed)
1. Make neutts-air import optional
2. Ensure kokoro worker can run independently
3. Add proper error handling for missing backends

### Step 4: Comprehensive Testing
1. Test all TTS backends independently
2. Test integration with voice pipeline
3. Verify latency targets (<800ms end-to-end)

## Risk Mitigation

### High Risk
- **Breaking existing TTS**: Test on development environment first
- **Dependency conflicts**: Use virtual environments

### Medium Risk  
- **Performance impact**: Benchmark after changes
- **Memory usage**: Monitor for increased memory consumption

### Low Risk
- **Import errors**: Proper try/catch blocks
- **Version mismatches**: Pin dependency versions

## Expected Outcome

1. **Immediate**: TTS initializes without EspeakWrapper errors
2. **Short-term**: Both Kokoro and Neutts backends work independently  
3. **Long-term**: Robust TTS system with automatic fallback and proper dependency management

## Success Metrics

- ✅ No more `EspeakWrapper` attribute errors
- ✅ TTS initialization succeeds in <5 seconds
- ✅ Audio generation works with target latency <800ms
- ✅ All TTS backends can be tested independently
- ✅ Graceful fallback when optional backends unavailable

This plan addresses the immediate bug while improving the overall TTS system architecture for long-term stability.