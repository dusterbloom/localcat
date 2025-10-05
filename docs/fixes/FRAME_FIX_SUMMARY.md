# Frame Initialization Bug Fix

## Problem
All custom audio intelligence frames were failing with:
```
'EnrollmentProgressFrame' object has no attribute 'id'
```

## Root Causes

### 1. Missing `super().__post_init__()` Calls
All custom frames (UnknownSpeakerDetectedFrame, StartEnrollmentFrame, SpeakerChangedFrame, EnrollmentProgressFrame, AudioIntelligenceFrame) overrode `__post_init__` but never called the parent's initialization.

**Result:** The Frame base class fields (`id`, `name`, `pts`, `metadata`, etc.) were never initialized.

### 2. `@property name()` Conflict
Custom frames defined `name` as a read-only property, but Pipecat's Frame.__post_init__() tries to SET `self.name`.

**Result:** `AttributeError: property 'name' of 'Frame' object has no setter`

## Solution

### Changed in `/server/core/audio/audio_intelligence.py`:

1. **Added `super().__post_init__()` to all frame classes:**
   ```python
   def __post_init__(self):
       super().__post_init__()  # CRITICAL: Initialize Frame.id, name, etc.
       if self.timestamp is None:
           self.timestamp = time.time()
   ```

2. **Replaced `@property name()` with `__str__()`:**
   ```python
   # Before:
   @property
   def name(self) -> str:
       return f"EnrollmentProgressFrame({self.current_sample}/{self.total_samples})"
   
   # After:
   def __str__(self) -> str:
       return f"EnrollmentProgressFrame({self.current_sample}/{self.total_samples})"
   ```

## Affected Frames
- ✅ UnknownSpeakerDetectedFrame
- ✅ StartEnrollmentFrame  
- ✅ SpeakerChangedFrame
- ✅ EnrollmentProgressFrame
- ✅ AudioIntelligenceFrame

## Testing
```python
from core.audio.audio_intelligence import EnrollmentProgressFrame

frame = EnrollmentProgressFrame(
    current_sample=1,
    total_samples=3,
    consistency=0.85
)

print(frame.id)    # ✅ Works: 0
print(frame.name)  # ✅ Works: EnrollmentProgressFrame#0
print(str(frame))  # ✅ Works: EnrollmentProgressFrame(1/3)
```

## Next Steps
1. Restart server with correct environment variables
2. Test intro pipeline enrollment flow
3. Verify EnrollmentProgressFrame emissions reach EnrollmentCoordinator
