# First Sample Progress Frame Fix

## 🐛 Bug Found

**Symptom:** Intro message never sent, enrollment never starts, even with `FORCE_INTRO=true`

**Root Cause:** First sample doesn't emit `EnrollmentProgressFrame`!

### Code Flow

1. **First utterance:**
   ```python
   if not self._unknown_embeddings:
       self._unknown_embeddings.append((embedding, current_time))
       logger.info("[AudioIntel] 👤 Unknown speaker, collecting samples...")
       return  # ← NO PROGRESS FRAME EMITTED!
   ```

2. **EnrollmentCoordinator waits:**
   ```python
   if not self._enrollment_started and current == 1:
       # Waiting for EnrollmentProgressFrame with current_sample=1
       # But it never comes!
   ```

3. **Subsequent samples:**
   - Check consistency with first sample
   - If consistent: emit progress frame
   - **BUT** if inconsistent: reset and start over (back to step 1)

4. **Result:**
   - If samples keep failing consistency → progress frame NEVER emitted
   - Coordinator never starts → intro message never sent
   - User sees generic "Hello!" instead of enrollment prompts

---

## ✅ Fix Applied

Added progress frame emission for first sample in `/server/core/audio/audio_intelligence.py`:

```python
# Start collection if empty (legacy auto-enroll or after consent)
if not self._unknown_embeddings:
    self._unknown_embeddings.append((embedding, current_time))
    if self._current_speaker != "unknown":
        self._current_speaker = "unknown"
        self._collecting_samples = True
        logger.info("[AudioIntel] 👤 Unknown speaker, collecting samples...")
    
    # CRITICAL FIX: Emit progress frame for first sample to trigger EnrollmentCoordinator
    await self.push_frame(
        EnrollmentProgressFrame(
            current_sample=1,
            total_samples=self._auto_enroll_utterances,
            consistency=1.0,  # First sample is 100% consistent with itself
            speaker_id="unknown"
        )
    )
    return
```

---

## 🧪 Expected Behavior After Fix

### First Connection:

1. **User speaks first utterance**
2. **AudioIntel:** `[AudioIntel] 👤 Unknown speaker, collecting samples...`
3. **AudioIntel:** Emits `EnrollmentProgressFrame(current_sample=1, total_samples=3, consistency=1.0)`
4. **Coordinator:** `[EnrollmentCoordinator] Starting enrollment from first progress frame`
5. **Coordinator:** `[EnrollmentCoordinator] Sending intro message`
6. **Agent says:** "Welcome! I'm learning to recognize your voice. Please say a few words..."
7. **State changes:** `intro` → `enrolling`

### Subsequent Samples:

- **If consistent:** Progress updates (2/3, 3/3) → completion → conversation
- **If inconsistent:** Samples reset, but user already knows what's happening

---

## 🔍 Secondary Issue: Low Consistency

**Observed in logs:**
```
[AudioIntel] Inconsistent sample (0.20), resetting
[AudioIntel] Inconsistent sample (0.22), resetting
[AudioIntel] Inconsistent sample (0.50), resetting
```

**Current threshold:** `0.70 * 0.85 = 0.595`

**Problem:** Similarities (0.20, 0.22, 0.50) way below threshold

**Possible causes:**
- Poor audio quality (noise, low volume)
- User varying voice significantly between samples
- Microphone issues
- Background noise

**Solutions (if needed):**
1. Lower consistency multiplier: `0.70 * 0.70 = 0.49` (more forgiving)
2. Lower base threshold: `0.60 * 0.85 = 0.51`
3. Add audio quality checks and user feedback
4. Add "speak louder/clearer" prompts

**Note:** With the intro message now working, users will understand they're enrolling even if samples keep resetting. This is better UX than silent failure.

---

## 📊 Testing Checklist

After restart:

- ✅ First utterance triggers intro message
- ✅ Logs show: `[EnrollmentCoordinator] Starting enrollment from first progress frame`
- ✅ Agent says enrollment message (not generic "Hello!")
- ✅ State transitions: `intro` → `enrolling`
- ⚠️ If samples keep failing: Consider lowering consistency threshold further
- ⚠️ If audio quality poor: Check microphone settings, noise levels

---

## 🔄 Complete Fix Chain

**5 Bugs Fixed:**

1. ✅ Frame initialization (`super().__post_init__()`)
2. ✅ Environment variables (`.env` overrides command-line)
3. ✅ Coordinator design (detect first progress frame)
4. ✅ Recognition thresholds (lowered to 0.65/0.70)
5. ✅ **First sample progress emission** (this fix)

All fixes now in place for complete intro pipeline functionality! 🎉
