# Privacy-First Speaker Recognition Design

## 🔴 Current Problem

**Privacy Issue:**
```
Unknown speaker → Audio buffered → Embeddings calculated → Auto-enroll silently
❌ NO CONSENT ASKED
❌ Data recorded without permission
❌ Speaker enrolled as "Speaker_1" without identification
```

## ✅ Desired Privacy-First Flow

```
1. Unknown speaker detected (FIRST utterance)
   ↓
2. Bot asks: "I don't recognize you, may I know your name?"
   ↓
3. User responds: "My name is Alice"
   ↓
4. WITH CONSENT: Enroll as "Alice"
   ↓
5. Future utterances: Recognized as "Alice"
```

## 🎯 Elegant Solution Architecture

### Phase 1: Detect Unknown Speaker (Immediate)
```python
# First utterance from unknown speaker
→ Calculate embedding (minimal processing)
→ Check if recognized
→ If UNKNOWN:
    - Emit UnknownSpeakerDetectedFrame
    - Enter CONSENT_PENDING mode
    - DON'T store embeddings yet
```

### Phase 2: Request Consent
```python
# Bot handler receives UnknownSpeakerDetectedFrame
→ Inject prompt: "I don't recognize you, may I know your name?"
→ Wait for user response
```

### Phase 3: Explicit Enrollment (With Consent)
```python
# User says: "My name is Alice"
→ Extract name from transcription
→ Emit StartEnrollmentFrame(name="Alice")
→ Collect 2-3 more utterances with name
→ Save profile as "Alice"
→ Emit SpeakerEnrolledFrame(name="Alice")
```

## 🔒 Privacy Modes

### Mode 1: EPHEMERAL (No storage until consent)
```python
privacy_mode = "ephemeral"
# Unknown speaker:
- Process audio in memory only
- DON'T save embeddings
- DON'T store in profiles
- Discard after recognition attempt
```

### Mode 2: CONSENT_PENDING (Temporary storage)
```python
privacy_mode = "consent_pending"
# Unknown speaker:
- Store embeddings temporarily
- Mark as "consent_pending"
- If consent given → Convert to profile
- If consent denied → Delete all data
- Auto-delete after 5 minutes
```

### Mode 3: AUTO_ENROLL (Current behavior - less private)
```python
privacy_mode = "auto_enroll"
# Unknown speaker:
- Auto-enroll as "Speaker_1"
- No consent required (for testing only)
```

## 📐 Implementation Plan

### 1. New Frames

```python
@dataclass
class UnknownSpeakerDetectedFrame(SystemFrame):
    """First utterance from unrecognized speaker"""
    embedding_preview: str  # Hash for tracking
    timestamp: float

@dataclass
class StartEnrollmentFrame(SystemFrame):
    """User gave consent and provided name"""
    speaker_name: str
    timestamp: float

@dataclass  
class SpeakerEnrolledFrame(SystemFrame):
    """Enrollment completed"""
    speaker_id: str
    speaker_name: str
    utterances_used: int
```

### 2. Privacy-Aware AudioIntelligenceProcessor

```python
class AudioIntelligenceProcessor:
    def __init__(
        self,
        privacy_mode: str = "consent_pending",  # NEW
        require_consent: bool = True,  # NEW
        consent_timeout_sec: int = 300,  # 5 min
        ...
    ):
        self._privacy_mode = privacy_mode
        self._require_consent = require_consent
        self._consent_timeout = consent_timeout_sec
        
        # Consent tracking
        self._pending_embeddings: Dict[str, PendingEnrollment] = {}
        self._enrollment_in_progress: Optional[str] = None
```

### 3. Modified Unknown Speaker Handling

```python
async def _process_unknown_speaker(self, embedding, ...):
    """Privacy-aware unknown speaker handling"""
    
    if self._require_consent:
        # Check if this is FIRST unknown detection
        if len(self._unknown_embeddings) == 0:
            # FIRST utterance → Ask for consent
            await self.push_frame(
                UnknownSpeakerDetectedFrame(
                    embedding_preview=self._hash_embedding(embedding),
                    timestamp=time.time()
                )
            )
            
            if self._privacy_mode == "ephemeral":
                # Don't store anything yet
                logger.info("[AudioIntel] 🔒 Unknown speaker, awaiting consent")
                return
            
            elif self._privacy_mode == "consent_pending":
                # Store temporarily
                self._pending_embeddings[preview_hash] = PendingEnrollment(
                    embeddings=[embedding],
                    expires_at=time.time() + self._consent_timeout
                )
                logger.info("[AudioIntel] 🕐 Unknown speaker, consent pending")
                return
        
        # Waiting for StartEnrollmentFrame with name...
    
    else:
        # Legacy auto-enroll mode (no consent)
        await self._legacy_auto_enroll(embedding)
```

### 4. Consent Handler (Process StartEnrollmentFrame)

```python
async def process_frame(self, frame, direction):
    """Handle enrollment consent"""
    
    if isinstance(frame, StartEnrollmentFrame):
        # User gave consent with name!
        logger.info(f"[AudioIntel] ✅ Consent received for: {frame.speaker_name}")
        
        # Move from pending → active enrollment
        if self._pending_embeddings:
            preview = list(self._pending_embeddings.keys())[0]
            pending = self._pending_embeddings.pop(preview)
            
            # Start collecting more samples for enrollment
            self._unknown_embeddings = [(emb, time.time()) for emb in pending.embeddings]
            self._enrollment_name = frame.speaker_name
            self._enrollment_in_progress = True
        
        return await self.push_frame(frame, direction)
```

### 5. Bot Handler for Unknown Speaker Prompt

```python
# In bot.py or memory handler
async def process_frame(self, frame, direction):
    if isinstance(frame, UnknownSpeakerDetectedFrame):
        # Inject consent prompt into LLM context
        prompt = (
            "SYSTEM: An unrecognized speaker has been detected. "
            "Please politely ask: 'I don't recognize you, may I know your name?'"
        )
        
        await self.push_frame(
            ContextUpdateFrame(
                system_message=prompt,
                priority="immediate"
            ),
            FrameDirection.DOWNSTREAM
        )
```

### 6. Name Extraction from Transcription

```python
# In transcription handler
def extract_name_from_response(text: str) -> Optional[str]:
    """Extract name from user response"""
    patterns = [
        r"my name is (\w+)",
        r"i'm (\w+)",
        r"i am (\w+)",
        r"call me (\w+)",
        r"(\w+) here",
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            name = match.group(1).capitalize()
            return name
    
    return None

# When transcription arrives:
if enrollment_in_progress:
    name = extract_name_from_response(transcription)
    if name:
        await push_frame(StartEnrollmentFrame(speaker_name=name))
```

## 🎬 Complete Flow Example

### Scenario: Unknown Speaker Alice

```
[T=0s] Alice speaks: "Hello"
↓
[AudioIntel] Unknown speaker detected
[AudioIntel] 🔒 Privacy mode: consent_pending
→ Emit UnknownSpeakerDetectedFrame

[T=1s] Bot receives frame
→ Inject prompt: "I don't recognize you, may I know your name?"
[Bot speaks]: "I don't recognize you, may I know your name?"

[T=5s] Alice responds: "My name is Alice"
↓
[Transcription] "My name is Alice"
→ Extract name: "Alice"
→ Emit StartEnrollmentFrame(name="Alice")

[T=6s] AudioIntel receives StartEnrollmentFrame
[AudioIntel] ✅ Consent received for: Alice
[AudioIntel] Starting enrollment for Alice
→ Collecting 2 more utterances...

[T=15s] Alice speaks 2 more times
↓
[AudioIntel] ✨ Enrolled: Alice (3 utterances, 0.89 confidence)
→ Emit SpeakerEnrolledFrame(name="Alice")
[Bot]: "Nice to meet you, Alice!"

[T=20s] Alice speaks again
↓
[AudioIntel] 🎯 Recognized: Alice (conf=0.92)
→ Normal processing
```

## 📊 Comparison

### Current (No Privacy)
```
Unknown → Auto-enroll → "Speaker_1" → Done
Time: 15s
Consent: ❌ None
Privacy: ❌ Low
```

### Privacy-First
```
Unknown → Ask name → Wait consent → Enroll with name → Done
Time: 30s (longer but respectful)
Consent: ✅ Explicit
Privacy: ✅ High
```

## 🔧 Configuration

```bash
# .env
AUDIO_INTEL_PRIVACY_MODE=consent_pending  # ephemeral | consent_pending | auto_enroll
AUDIO_INTEL_REQUIRE_CONSENT=true
AUDIO_INTEL_CONSENT_TIMEOUT_SEC=300
AUDIO_INTEL_AUTO_DELETE_PENDING=true
```

## ✅ Benefits

1. **Privacy Compliant**: No data stored without consent
2. **User Friendly**: Natural conversation flow
3. **Named Profiles**: "Alice" instead of "Speaker_1"
4. **GDPR Ready**: Explicit consent, data deletion
5. **Fallback Safe**: Can disable for testing

## 🚀 Migration Path

### Phase 1: Add privacy modes (backward compatible)
- Default: `privacy_mode="auto_enroll"` (current behavior)
- No breaking changes

### Phase 2: Enable consent by default
- Default: `privacy_mode="consent_pending"`
- Bot prompts for name

### Phase 3: Ephemeral mode for maximum privacy
- Default: `privacy_mode="ephemeral"`
- No storage until explicit consent

---

**This design respects user privacy while maintaining the audio intelligence features!**
