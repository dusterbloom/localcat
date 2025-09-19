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

## Phase 5: Multimodal Vision Fusion with FastVLM (Weeks 9-12, Effort: Medium, Post-Voice MVP)
### Goal
Integrate Apple's FastVLM (CVPR 2025) for efficient, high-resolution visual non-verbals. Fuse with voice intelligence (prosody + speaker ID) for holistic, empathetic reactivity—e.g., detect "forced smile + flat tone = insincere" and adapt LLM/TTS responses. Leverage FastVLM's hybrid encoder (FastViT-HD) for <150ms vision latency on Apple Silicon, maintaining <900ms total E2T. Target: +10-15% accuracy on multimodal emotion tasks (e.g., MEAD dataset).

### Key Innovations from FastVLM (arXiv:2412.13303)
- **Architecture**: Hybrid conv-transformer vision encoder (conv stem + 3 conv stages + 2 transformer stages) pre-trained with MobileCLIP. Outputs 4× fewer tokens than FastViT (16× vs. ViT-L/14 at 336px), reducing TTFT by 3-85x vs. baselines (e.g., 85x faster than LLaVA-OneVision 0.5B).
- **Efficiency**: Native high-res (up to 1152x1152 without tiling; optional 2x2 AnyRes for ultra-HD). Models: 0.5B/1.5B/7B LLMs (quantized int4/int8/fp16 on HF). Benchmarks: Avg-5 (GQA/TextVQA/DocVQA/SeedBench/POPE) 68-75%; Emotion/scene AR ~80%.
- **On-Device**: MLX-based inference (150ms TTFT on iPhone 16 Pro); CoreML export for Neural Engine. Repo: apple/ml-fastvlm (code, checkpoints, iOS/macOS demo app).

### Tasks
1. **Setup & Integration (Week 9)**:
   - Clone apple/ml-fastvlm; uv add mlx coremltools av (for video frames).
   - Download HF models (e.g., apple/FastVLM-0.5B-int4 for speed); Test inference: `python infer.py --model 0.5B --image sample.jpg --prompt "Describe mood and scene"`.
   - Benchmark on M3: TTFT <150ms at 336-672px; Profile power (<5% overhead).

2. **VisionTee Scaffold (Week 10)**:
   - Create `server/vision_intelligence.py`: Custom Pipecat FrameProcessor (parallel to AudioTee).
     - Input: WebRTC video frames (av.VideoFrame from client stream).
     - Preprocess: Downsample/resize (PIL: 336-672px, every 200ms for 5fps); Optional tiling (if >672px, split 2x2 via repo utils).
     - Inference: MLX model (`model.apply(image_tensor)`); Extract: Embeddings → Dict {"emotion": "stressed", "confidence": 0.92, "scene": "office", "details": ["furrowed brows", "coffee mug"]}.
     - Output: Async queue merge with audio intel (JSON fusion).
   - Fallback: CoreML Vision (VNRecognizeEmotionsRequest for quick emotions, <50ms).

3. **Multimodal Fusion & Reactivity (Week 11)**:
   - Extend `audio_intelligence.py` context_injector: Multimodal queue polling → LLM prompt: "[Audio: hesitant][Visual: stressed, details: furrowed brows] → Empathetic, slow-paced response."
   - TTS Adaptation: Pass visual metadata to Kokoro (e.g., speed up for "energetic gaze"; mirror emotion via prosody params).
   - Learning Loop: Update memory_store.py—Store multimodal baselines (e.g., PCA on FastVLM embeddings + prosody vectors per user); Score via next-frame shifts (Bayesian update).
   - Enrollment: Client records photo+voice → Compute/store FastVLM embeddings (hashed, diffprivlib noise).

4. **Client UI, Tests, & Polish (Week 12)**:
   - Client: Add video track in `client/src/app/page.tsx` (navigator.mediaDevices.getUserMedia({video: true})); WebSocket push visual icons (e.g., mood emojis in transcript).
   - Optimization: MLX async batching; CoreML export (`ct.convert` to .mlmodel for ANE); Adaptive res (low for idle, high on demand).
   - Ethics: UI consent toggle for video; /forget endpoint deletes visual data; Bias audit (test on diverse MEAD/RAF-DB subsets).

### Code Skeleton (Draft for vision_intelligence.py)
```python
import asyncio
import numpy as np
from av import VideoFrame  # For WebRTC frames
from mlx.core import array  # MLX tensors
from pipecat.frames import VideoFrame as PipecatVideoFrame  # Assuming Pipecat video support
from pipecat.processors import FrameProcessor
from .fastvlm_model import FastVLM  # From repo import

class VisionTee(FrameProcessor):
    def __init__(self, model_path: str = "apple/FastVLM-0.5B-int4"):
        self.model = FastVLM.from_pretrained(model_path)  # MLX load
        self.context_queue = asyncio.Queue()  # Merge with audio

    async def process_frame(self, frame: PipecatVideoFrame, direction):
        if direction != FrameDirection.DOWNSTREAM:
            return frame

        # Extract/Preprocess: RGB np.array from VideoFrame
        img_array = np.array(frame.to_image())  # PIL or cv2 resize to 336x336
        img_tensor = array(img_array.transpose(2, 0, 1) / 255.0)  # Normalize [3,H,W]

        # Async Inference (fire-and-forget)
        asyncio.create_task(self._process_vision(img_tensor, frame.timestamp))

        return frame  # Hotpath pass-through

    async def _process_vision(self, img_tensor, timestamp):
        # FastVLM: Tokens + Prompt (e.g., "Analyze emotion and key details")
        outputs = self.model.generate(img_tensor, prompt="Describe facial emotion, scene objects, and fine details like expressions.")  # Returns text/embeddings
        result = {"visual": {"emotion": parse_emotion(outputs), "scene": parse_scene(outputs), "confidence": 0.9}}
        await self.context_queue.put(result)  # Fuse downstream
```

### Tests
- Unit: `test_vision_tee.py` (mock VideoFrame, assert <100ms inference + outputs match expected dict).
- Integration: `test_multimodal_fusion.py` (synthetic audio+video, verify prompt injection → LLM adapts tone).
- E2E: Extend `test_e2e_streaming.py`—Voice+camera session; Metrics: TTFT <900ms, AR >80% on MEAD (multimodal emotion accuracy).
- Benchmarks: TTFT/latency on M3 (vs. LLaVA baseline); Power profiling (MLX tools).

### Milestone
Live demo: WebRTC video+audio → Real-time fused response (e.g., "You look stressed and sound hesitant—take a deep breath?"). Logs: "Fused Context: Visual 92% stressed + Audio urgent → Calming adaptation." Commit: "feat: FastVLM multimodal integration."

### Risks/Mitigations
- Porting/Deps: MLX/CoreML conflicts—Isolate worker; Test on iOS sim via demo app.
- Compute: Video spikes—Adaptive sampling (5fps); Fallback to audio-only if >150ms.
- Bias/Privacy: Facial analysis risks—Audit diverse datasets; Explicit consent, auto-delete visuals.
- Scalability: Limit to 1-2 users; ONNX export for non-Apple (coremltools, +20ms).

## Post-MVP Extensions (Future Sprints)
- Advanced Multimodal: Video temporal (frame-seq + SAM segmentation); RLHF on fused feedback.
- Cross-Platform: ONNX Runtime fallback for Linux/Windows.
- Community: Pipecat contrib (VisionTee + FastVLM processor); Blog: "On-Device Multimodal Empathy with LocalCat & FastVLM."

## Resource Allocation
Start with Phase 1 prototype (1 week POA). Track in issues: #voice-intel-tee, #speaker-recog, #fastvlm-integration. Budget: 350-450 dev hours (incl. Phase 5).

Date: 2025-09-19 (Updated: 2025-10-01 for FastVLM)
Status: Draft
Phases: 1-5 (MVP + Vision)
Refs: [See above] + FastVLM: https://machinelearning.apple.com/research/fast-vision-language-models; arXiv:2412.13303; GitHub: apple/ml-fastvlm; HF: apple/FastVLM-*.