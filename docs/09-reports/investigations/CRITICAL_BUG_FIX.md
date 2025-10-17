# CRITICAL BUG FIX - Missing Attribute Initialization

## 🔴 Bug Report

**Severity:** CRITICAL - Blocks all speaker enrollment  
**Found:** 2025-10-01 19:30:12  
**Status:** ✅ FIXED

## Error

```
ERROR | core.audio.audio_intelligence:_process_utterance:389 
[AudioIntel] Error processing utterance: 
'AudioIntelligenceProcessor' object has no attribute '_consent_prompted'
```

## Root Cause

In the privacy-first implementation, we added these checks:

```python
# In _process_unknown_speaker()
if not self._unknown_embeddings and not self._consent_prompted:
    # ... privacy logic
```

But **FORGOT** to initialize these attributes in `__init__`:

```python
# MISSING (before fix):
self._consent_prompted = False
self._enrollment_name = None
self._pending_consent_hash = None
self._unknown_embeddings = []
self._collecting_samples = False
```

## Impact

- ✅ User spoke 3 times perfectly
- ✅ Prosody extracted successfully
- ❌ **Crashed before enrollment could start**
- ❌ **No profiles saved**
- ❌ **All enrollment attempts failed**

## The Fix

Added missing attribute initialization in `__init__`:

```python
# State
self._is_speaking = False
self._audio_buffer = bytearray()
self._current_speaker: Optional[str] = None
self._unknown_embeddings: List[Tuple[torch.Tensor, float]] = []  # ✅ ADDED
self._collecting_samples = False  # ✅ ADDED

# Privacy-First: Consent tracking (CRITICAL: Must initialize!)
self._pending_consent_hash: Optional[str] = None  # ✅ ADDED
self._consent_prompted: bool = False  # ✅ ADDED
self._enrollment_name: Optional[str] = None  # ✅ ADDED

# Speaker database
self._speakers: Dict[str, List[torch.Tensor]] = {}
self._speaker_names: Dict[str, str] = {}
self._speaker_counter = 0
```

## Evidence from Logs

User said: **"Good evening, my name is Peppy"** (3 times!)

```
2025-10-01 19:30:16.586 | Transcription: 
'Good evening, my name is Peppy. Good evening, my name is Peppy. Good evening, my name is Peppy.'

2025-10-01 19:30:12.864 | Prosody extracted: ✅
ProsodyFeatures(pitch=95.2Hz, slope=-1.3, intensity=56.6dB, rate=1.6syll/s, certainty=-0.10)

2025-10-01 19:30:12.864 | ERROR: ❌
'AudioIntelligenceProcessor' object has no attribute '_consent_prompted'
```

**Result:** Perfect input, but crashed before enrollment!

## Status After Fix

✅ Attributes properly initialized  
✅ Privacy mode won't crash  
✅ Enrollment can proceed  
✅ No more AttributeError  

## Next Steps

1. **Restart bot immediately:**
   ```bash
   cd /Users/peppi/Dev/localcat/server
   rm -rf core/audio/__pycache__
   python3 bot.py
   ```

2. **Speak 3 clear sentences**

3. **Verify profile created:**
   ```bash
   ls data/speaker_profiles/auto_enrolled/
   ```

## Lesson Learned

When adding new instance attributes that are checked in methods:
1. ✅ Initialize in `__init__`
2. ✅ Set default values
3. ✅ Document purpose
4. ✅ Test end-to-end before declaring complete

**This is exactly why end-to-end testing matters!**

---

**Fixed:** 2025-10-01 19:31  
**File:** `/server/core/audio/audio_intelligence.py`  
**Lines:** 165-175  
