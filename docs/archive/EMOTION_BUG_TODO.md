# Emotion Detection Bug - TODO

## Status: TEMPORARILY DISABLED

**Issue:** SpeechBrain emotion model API is not working as expected

## Error
```
'ModuleDict' object has no attribute 'compute_features'
```

## What We Tried

### Attempt 1: Direct classify_batch
```python
out_prob, score, index, text_lab = self._emotion_model.classify_batch(audio_cpu)
```
**Result:** Still crashes with same error

### Attempt 2: Force CPU
```python
audio_cpu = audio_tensor.cpu()
```
**Result:** Still crashes

## Root Cause

The emotion model (`speechbrain/emotion-recognition-wav2vec2-IEMOCAP`) appears to have a different API than expected. The `EncoderClassifier.classify_batch()` method is failing internally.

## Current Workaround

**Disabled emotion detection:**
```bash
AUDIO_INTEL_ENABLE_EMOTION=false
```

This allows:
- ✅ Speaker recognition to work
- ✅ Prosody analysis to work
- ✅ System to be stable

Missing:
- ❌ Emotion labels (angry/happy/sad/neutral)
- ❌ Valence/arousal values

## Next Steps to Fix

1. **Test the model standalone:**
```python
from speechbrain.inference.classifiers import EncoderClassifier

model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="models/emotion"
)

# Try different APIs:
# Option A: classify_batch
result = model.classify_batch(audio_tensor)

# Option B: classify_file  
result = model.classify_file("test.wav")

# Option C: encode_batch + separate classifier
embedding = model.encode_batch(audio_tensor)
```

2. **Check SpeechBrain documentation:**
   - https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP
   - Look at example usage in model card

3. **Alternative: Use different emotion model:**
   - `speechbrain/emotion-recognition-wav2vec2` (simpler)
   - Or use librosa + sklearn for basic emotion

4. **Debug with minimal test:**
```python
import torch
from speechbrain.inference.classifiers import EncoderClassifier

# Load model
model = EncoderClassifier.from_hparams(...)

# Create test audio
audio = torch.randn(1, 16000)  # 1 second

# Try classification
print(dir(model))  # See available methods
print(model.classify_batch(audio))
```

## Impact

**Low priority fix because:**
- Speaker recognition works without it ✅
- Prosody analysis works without it ✅
- Can be added back later without breaking anything

**Medium priority because:**
- Emotion data is valuable for TRUE confidence
- Part of Session 2 goals
- Users expect it to work

## Temporary Solution

System works perfectly without emotion:
```
Speaker: Speaker_2 (83% confidence) ✅
Prosody: pitch=162Hz, certainty=-0.10 ✅
Emotion: (disabled) ⚠️
```

Once we fix the API, re-enable with:
```bash
AUDIO_INTEL_ENABLE_EMOTION=true
```

---

**Created:** 2025-10-01
**Status:** Known issue, temporarily disabled
**Priority:** Medium (non-blocking)
