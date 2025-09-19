# LocalCat Voice Intelligence Implementation Plan

This plan synthesizes the discussion thread into a phased, testable roadmap for enhancing LocalCat with non-blocking audio intelligence. It prioritizes the AudioTee + parallel pipelines architecture to maintain <800ms hotpath latency while adding speaker personalization and prosody-driven reactivity. Total effort: 6-8 weeks for MVP, assuming 1-2 devs familiar with Pipecat/MLX/Python.

## Objectives
- Fork audio post-VAD: Hotpath (stateless/onboarding) vs. Intelligence path (recognized user with memory/prosody).
- Enable paths: Temp (no mem, generic) → Personalized (speaker ID → prosody analysis → context injection).
- Libraries: SOTA picks (SpeechBrain for recog/diarization, Librosa/Parselmouth for prosody) for local/MLX compatibility.
- Success Metrics: <100ms added latency; >85% speaker recog accuracy (VoxCeleb-like); prosody detection AR >70% on IEMOCAP; full e2e tests pass.

## Best References
- **Pipecat Docs**: https://docs.pipecat.ai/processors/pipelines (for branching/async processors); https://docs.pipecat.ai/transports/webrtc (your client integration).
- **SpeechBrain**: GitHub speechbrain/speechbrain (ECAPA-TDNN tutorial: https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/speaker-recognition.html); PyPI speechbrain==0.5.15; HF models: speechbrain/spkrec-ecapa-voxceleb.
- **pyannote.audio**: GitHub pyannote/pyannote-audio (streaming diarization: https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/streaming_diarization.ipynb); PyPI pyannote.audio==3.1.1; Requires HF token.
- **Librosa/Parselmouth**: Librosa docs https://librosa.org/doc/latest/feature.html#prosodic-features (pyin for pitch); Parselmouth tutorial https://parselmouth.readthedocs.io/en/stable/praat_tutorial.html; PyPI librosa==0.10.1, parselmouth==0.4.2.
- **MLX Porting**: MLX examples repo https://github.com/ml-explore/mlx-examples (audio ports); HF MLX converters: https://huggingface.co/docs/transformers/main/en/accelerate/usage_guides/mlx.
- **Datasets/Benchmarks**: VoxCeleb2 for speaker (http://www.robots.ox.ac.uk/~vgg/data/voxceleb/); IEMOCAP for prosody (https://github.com/HLT-MT/FGE); INTERSPEECH 2024 papers on prosody (e.g., \"Prosody-Aware Speaker Embeddings\").
- **Ethics/Privacy**: Differential privacy for embeddings (add via diffprivlib PyPI); GDPR voice biometrics guidelines (https://gdpr.eu/biometrics/).

## Prerequisites
- Update requirements.txt/pyproject.toml: Add `speechbrain pyannote.audio librosa parselmouth pyannote.audio transformers diffprivlib` (uv sync after).
- Cache models: `speechbrain/spkrec-ecapa-voxceleb` via HF CLI; Test MLX conversion on sample audio.
- Dev Setup: Extend CLAUDE.md with \"Voice Intelligence\" section; Run `pytest` baseline before changes.
- Testing: Use locomo10.json for multi-turn; Augment with synthetic prosody (librosa effects).

## Phase 1: AudioTee & Parallel Pipeline Scaffold (Week 1-2, Effort: Low)
### Goal
Implement non-blocking fork post-VAD; Basic path switching without ML.

### Tasks
1. Create `server/audio_intelligence.py`: Define AudioTee class (custom Pipecat Processor).
   - Duplicate np.float32 audio frames (zero-copy views).
   - Use asyncio.Queue for hotpath (immediate) vs. intel (async task).
   - Merge: Shared context queue for LLM injection (e.g., JSON {\"user\": \"unknown\", \"prosody\": {}}).
2. Update `bot.py`: Insert Tee after Silero VAD.
   - Hotpath: Pipeline([smart_turn, stt, llm_generic, tts])  # Stateless prompts.
   - Intel Path: Pipeline([speaker_processor, prosody_processor]) → context_injector.
   - Path Logic: If recognized (threshold 0.8), switch to personalized LLM (inject memory from memory_store.py).
3. Client UI: Add enrollment button in `client/src/app/page.tsx` (WebRTC record 5-10s clip → POST to /enroll).
4. Enrollment: New endpoint `/enroll` in FastAPI; Compute embedding → Store hashed (bcrypt) in memory_store.py (add speaker_profiles table, SQLite).

### Code Skeleton (Draft for audio_intelligence.py)
```python
import asyncio
import numpy as np
from pipecat.frames import AudioRawFrame
from pipecat.processors import FrameProcessor, Pipeline
from typing import Dict, Any

class AudioTee(FrameProcessor):
    def __init__(self, hotpath: Pipeline, intel: Pipeline):
        super().__init__()
        self.hotpath = hotpath
        self.intel = intel
        self.context_queue = asyncio.Queue()  # For merging intel results

    async def process_frame(self, frame: AudioRawFrame, direction):
        if direction != FrameDirection.DOWNSTREAM:
            return frame

        # Zero-copy duplicate (view)
        hot_audio = np.array(frame.audio)  # Or frame.audio.view() if immutable
        intel_audio = hot_audio.copy() if not hot_audio.flags.writeable else hot_audio

        # Hotpath immediate
        hot_frame = AudioRawFrame(hot_audio, frame.timestamp)
        await self.hotpath.process_frame(hot_frame, FrameDirection.DOWNSTREAM)

        # Intel async
        asyncio.create_task(self._process_intel(intel_audio, frame.timestamp))

        return frame  # Pass through for compatibility

    async def _process_intel(self, audio: np.ndarray, timestamp: float):
        intel_frame = AudioRawFrame(audio, timestamp)
        result = await self.intel.process_frame(intel_frame, FrameDirection.DOWNSTREAM)
        if isinstance(result, Dict):  # e.g., {\"speaker\": \"alice\", \"mood\": \"excited\"}
            await self.context_queue.put(result)
```

### Tests
- New `test_audio_tee.py` (unit: mock frames, assert no latency >10ms); Integrate into `test_e2e_streaming.py` (measure fork overhead).

### Milestone
Run bot.py; Verify hotpath latency unchanged; Log \"Path: temp\" for unrecognized audio.

### Risks/Mitigations
- Queue overflows—add backpressure (drop old intel frames). Privacy: Hash embeddings immediately.

## Phase 2: Speaker Recognition & Diarization (Week 3-4, Effort: Medium)
### Goal
ID users/multi-speakers; Enable recognized path with memory injection.

### Tasks
1. Implement `speaker_processor.py`: Use SpeechBrain ECAPA-TDNN.
   - Extract embedding: `verifier = SpeakerRecognition.from_hparams(source=\"speechbrain/spkrec-ecapa-voxceleb\")`.
   - Compare vs. stored profiles (cosine sim >0.8 → ID).
   - Diarization: If >1 speaker (pyannote pipeline), tag primary (e.g., \"multi: alice_leading\").
   - Output: Dict to context_queue.
2. Extend memory_store.py: Add `speaker_profiles` (SQLite: id, user_hash, embedding_vector, enroll_date).
   - On recog: Load user memory (from hotmem) → Inject to LLM prompt (e.g., \"Context: Alice's history...\").
3. Path Switching: In llm_processor, poll context_queue; If \"recognized\", use personalized prompt template.
4. Enrollment Flow: Client records → Server computes embedding (differential privacy noise via diffprivlib) → Store.
5. Handle Multi-User: Diarization clusters; Pause hotpath if overlap (extend smart_turn v2).

### References
SpeechBrain tutorial (above); pyannote streaming notebook; Test on VoxCeleb subsets (download 10min clips for eval).

### Tests
`test_speaker_recog.py` (mock audio, assert EER <5%); E2E: Simulate 2-users, verify path switch + memory injection.

### Milestone
90% recog accuracy on clean audio; Logs show \"Recognized: alice → Personalized path\".

### Risks/Mitigations
Noisy audio—add VAD filtering; Accents bias—fine-tune on diverse data (e.g., locomo10.json voices).

## Phase 3: Prosody Detection & Reactivity (Week 5-6, Effort: Medium)
### Goal
Analyze non-verbals; Adapt LLM/TTS based on mood/energy.

### Tasks
1. Implement `prosody_processor.py`: Librosa for features + Parselmouth for advanced.
   - Extract: Pitch (pyin), energy (RMS), tempo, pauses (silence ratios).
   - Classify: Simple ML (scikit-learn SVM on features) or MLX-ported wav2vec2 for 6 states (neutral, excited, tired, etc.).
   - Thresholds: E.g., pitch variance >20% = \"enthusiastic\"; Energy > mean+1σ = \"urgent\".
   - Output: Dict {\"mood\": \"excited\", \"features\": {...}} to context_queue.
2. Reactivity: In LLM prompt: \"[Prosody: excited] Adapt tone: Match energy.\" For TTS: Pass metadata to Kokoro (e.g., speed up for high tempo).
3. Learning Loop: Extend memory_store.py—Track prosody trends per user (e.g., Bayesian update baselines); Post-response, score via next prosody shift.
4. Non-Verbals: Detect laughter (spectral peaks), fillers (post-STT count \"um\").
5. UI Feedback: Client shows mood icons in transcript (WebSocket push from server).

### References
Librosa prosody tutorial; Parselmouth pitch tracking; IEMOCAP eval script (adapt for AR metric).

### Tests
`test_prosody.py` (augment audio with librosa shifts, assert detection >70% AR); E2E: Verify prompt injection changes LLM output (e.g., energetic response).

### Milestone
Detect prosody in real-time (<50ms/chunk); LLM adapts (manual review: 80% appropriate tone match).

### Risks/Mitigations
Compute spikes—offload to worker (like tts_mlx_isolated.py); Bias—audit on diverse accents; Consent UI for non-verbal tracking.

## Phase 4: Integration, Testing, & Polish (Week 7-8, Effort: Low-Medium)
### Goal
Full system; Production-ready.

### Tasks
1. Merge Paths: Handle lag—If intel >300ms, fallback to hotpath with partial context.
2. Error Handling: Retry on ML failures; Fallback to temp path.
3. Optimization: MLX ports for all models; Profile latency (add to hotmem.log).
4. Full Tests: Update run_all_tests.py; Coverage >85%; Load test (10 sessions, noisy audio).
5. Docs/UI: Update README with \"Voice Intelligence\" setup; Client: Enrollment modal, mood viz.
6. Ethics: Add config flags (e.g., --no-prosody); Delete data on /forget endpoint.

### Tests
Integration: Multi-turn with recog + prosody reactivity; Benchmarks: Latency <900ms E2T; Accuracy suite.

### Milestone
Demo: Onboarding → Enroll → Recognized session with adaptive responses. Commit to backlog.md as \"Voice Intelligence MVP\".

### Risks/Mitigations
Integration bugs—Modular tests; Scalability—Limit intel to 1-2 speakers initially.

## Post-MVP Extensions (Future Sprints)
- Multimodal: Fuse with vision (Phase 5?).
- Advanced Learning: RLHF on prosody feedback.
- Community: Pipecat contrib (AudioTee as plugin); Blog: \"Local Affective AI with LocalCat\".

## Resource Allocation
Start with Phase 1 prototype (1 week POA). Track in issues: #voice-intel-tee, #speaker-recog, etc. Budget: 200-300 dev hours.

Date: 2025-09-19
Status: Draft
Phases: 1-4 (MVP)
Refs: [See above]