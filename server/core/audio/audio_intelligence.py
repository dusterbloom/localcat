"""
Audio Intelligence Processor - Session 1: Speaker Recognition MVP
Unified processor for speaker recognition, emotion, and prosody using SpeechBrain
"""

import asyncio
import os
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
from loguru import logger
import difflib

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    SystemFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.pipeline import FrameDirection

# Check SpeechBrain availability
try:
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.inference.classifiers import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    logger.warning("SpeechBrain not available. Install with: pip install speechbrain")


@dataclass
class UnknownSpeakerDetectedFrame(SystemFrame):
    """
    Emitted on FIRST utterance from unrecognized speaker (Privacy-First)
    Bot should ask: "I don't recognize you, may I know your name?"
    """
    embedding_hash: str  # For tracking pending enrollment
    timestamp: float = None
    
    def __post_init__(self):
        super().__post_init__()  # CRITICAL: Initialize Frame.id, name, etc.
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def __str__(self) -> str:
        """For Pipecat frame logging"""
        return f"UnknownSpeakerDetectedFrame({self.embedding_hash[:8]})"


@dataclass
class StartEnrollmentFrame(SystemFrame):
    """
    User provided consent and name (Privacy-First)
    Triggers enrollment with given name instead of "Speaker_1"
    """
    speaker_name: str
    timestamp: float = None
    
    def __post_init__(self):
        super().__post_init__()  # CRITICAL: Initialize Frame.id, name, etc.
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def __str__(self) -> str:
        """For Pipecat frame logging"""
        return f"StartEnrollmentFrame({self.speaker_name})"


@dataclass
class SpeakerChangedFrame(SystemFrame):
    """Emitted when speaker changes (SystemFrame = immediate processing)"""
    speaker_id: str
    speaker_name: Optional[str] = None
    confidence: float = 0.0
    auto_enrolled: bool = False
    timestamp: float = None
    
    def __post_init__(self):
        super().__post_init__()  # CRITICAL: Initialize Frame.id, name, etc.
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def __str__(self) -> str:
        """For Pipecat frame logging"""
        return f"SpeakerChangedFrame({self.speaker_id})"


@dataclass
class EnrollmentProgressFrame(SystemFrame):
    """
    Emitted during enrollment to track progress.
    Used by EnrollmentCoordinator to provide user feedback.
    """
    current_sample: int
    total_samples: int
    consistency: float
    speaker_id: str = "unknown"
    timestamp: float = None
    
    def __post_init__(self):
        super().__post_init__()  # CRITICAL: Initialize Frame.id, name, etc.
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def __str__(self) -> str:
        """For Pipecat frame logging"""
        return f"EnrollmentProgressFrame({self.current_sample}/{self.total_samples})"
    
    @property
    def progress_percentage(self) -> float:
        """Progress as percentage (0-100)"""
        if self.total_samples == 0:
            return 0.0
        return (self.current_sample / self.total_samples) * 100


@dataclass
class AudioIntelligenceFrame(SystemFrame):
    """
    Unified audio intelligence frame with all extracted features
    Session 3: Added prosody features for TRUE confidence
    """
    speaker_id: str
    speaker_confidence: float
    # Session 2: Emotion fields
    emotion: Optional[str] = None  # angry, happy, sad, neutral, fearful, disgust, surprised
    emotion_confidence: float = 0.0
    valence: float = 0.0  # -1 (negative) to +1 (positive)
    arousal: float = 0.0  # 0 (calm) to 1 (excited)
    # Session 3: Prosody features
    prosody_features: Optional[Any] = None  # ProsodyFeatures object
    prosody_certainty: float = 0.0  # Certainty modifier from prosody
    timestamp: float = None
    
    def __post_init__(self):
        super().__post_init__()  # CRITICAL: Initialize Frame.id, name, etc.
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def __str__(self) -> str:
        """For Pipecat frame logging"""
        return f"AudioIntelligenceFrame({self.speaker_id})"


class AudioIntelligenceProcessor(FrameProcessor):
    """
    Session 1: MVP with SpeechBrain speaker recognition
    
    Buffers audio during speech and processes complete utterances for:
    - Speaker recognition (this session)
    - Emotion detection (session 2)
    - Prosody analysis (session 3)
    
    Runs in parallel pipeline (non-blocking).
    """
    
    def __init__(
        self,
        profile_dir: str = "data/speaker_profiles",
        similarity_threshold: float = 0.75,
        min_utterance_duration_sec: float = 1.0,
        auto_enroll_utterances: int = 3,
        consistency_threshold: float = 0.65,
        sample_rate: int = 16000,
        device: str = "cpu",  # Use "mps" for Apple Silicon, "cuda" for NVIDIA
        enable_emotion: bool = True,  # Session 2: Enable emotion detection
        enable_prosody: bool = True,  # Session 3: Enable prosody analysis
        # Privacy-First settings
        privacy_mode: str = "auto_enroll",  # ephemeral | consent_pending | auto_enroll
        require_consent: bool = False,  # If True, ask for name before enrolling
        consent_timeout_sec: int = 300,  # 5 min timeout for pending consent
    ):
        """
        Args:
            profile_dir: Directory to store speaker profiles
            similarity_threshold: Cosine similarity threshold for recognition
            min_utterance_duration_sec: Minimum audio length to process
            auto_enroll_utterances: Utterances needed before auto-enrolling
            consistency_threshold: Min similarity between utterances for enrollment
            sample_rate: Audio sample rate (16kHz default)
            device: PyTorch device ("cpu", "mps", "cuda")
        """
        super().__init__()
        self._enabled: bool = True  # Allow runtime pause/resume
        self._speaker_recognition_enabled: bool = True  # Separate control for speaker recognition

        if not SPEECHBRAIN_AVAILABLE:
            raise ImportError("SpeechBrain required. Install: pip install speechbrain")
        
        # Configuration
        self._profile_dir = Path(profile_dir).expanduser()
        self._similarity_threshold = similarity_threshold
        self._min_utterance_duration = min_utterance_duration_sec
        self._auto_enroll_utterances = auto_enroll_utterances
        self._consistency_threshold = max(0.630, consistency_threshold)  # Min 0.70 for real-world robustness
        self._sample_rate = sample_rate
        self._device = device
        self._enable_emotion = enable_emotion
        self._enable_prosody = enable_prosody
        
        # Privacy-First settings
        self._privacy_mode = privacy_mode
        self._require_consent = require_consent
        self._consent_timeout = consent_timeout_sec
        
        # State
        self._is_speaking = False
        self._audio_buffer = bytearray()
        self._current_speaker: Optional[str] = None
        self._unknown_embeddings: List[Tuple[torch.Tensor, float]] = []
        self._collecting_samples = False
        self._early_recognition_done = False
        try:
            self._early_ms = int(os.getenv("AUDIO_INTEL_EARLY_MS", "700"))
        except Exception:
            self._early_ms = 700
        
        # Privacy-First: Consent tracking (CRITICAL: Must initialize!)
        self._pending_consent_hash: Optional[str] = None
        self._consent_prompted: bool = False
        self._enrollment_name: Optional[str] = None
        
        # Speaker database
        self._speakers: Dict[str, List[torch.Tensor]] = {}  # speaker_id -> embeddings
        self._speaker_names: Dict[str, str] = {}  # speaker_id -> real_name
        self._speaker_counter = 0
        
        # Auto-enrollment state
        self._unknown_embeddings: List[Tuple[torch.Tensor, float]] = []  # (embedding, timestamp)
        self._collecting_samples = False
        
        # Initialize SpeechBrain speaker recognition model
        # Enable MPS fallback for unsupported operations
        if device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            logger.info("[AudioIntel] MPS fallback enabled for unsupported ops")
        
        logger.info(f"[AudioIntel] Loading SpeechBrain ECAPA-TDNN model ({device})...")
        try:
            savedir = self._profile_dir / "models" / "speaker"

            # Pre-populate savedir from bundled model if available and savedir is empty
            hf_home = Path(os.environ.get("HF_HOME", ""))
            bundled_model = hf_home / "hub" / "models--speechbrain--spkrec-ecapa-voxceleb"

            logger.info(f"[AudioIntel] HF_HOME={hf_home}, bundled_model={bundled_model}")
            logger.info(f"[AudioIntel] bundled_model.exists()={bundled_model.exists()}")
            logger.info(f"[AudioIntel] savedir={savedir}, has hyperparams={(savedir / 'hyperparams.yaml').exists()}")

            if bundled_model.exists() and not (savedir / "hyperparams.yaml").exists():
                logger.info(f"[AudioIntel] Pre-copying bundled model from {bundled_model} to {savedir}")
                savedir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copytree(bundled_model, savedir, dirs_exist_ok=True)

                # Update hyperparams.yaml to use local paths instead of Hub
                hyperparams_file = savedir / "hyperparams.yaml"
                if hyperparams_file.exists():
                    content = hyperparams_file.read_text()
                    # Replace Hub identifier with local path (current directory)
                    content = content.replace(
                        "pretrained_path: speechbrain/spkrec-ecapa-voxceleb",
                        "pretrained_path: ."
                    )
                    hyperparams_file.write_text(content)
                    logger.info("[AudioIntel] Updated hyperparams.yaml to use local paths")

                logger.info("[AudioIntel] Bundled model copied successfully")
            else:
                logger.debug(f"[AudioIntel] Skipping pre-copy (bundled exists={bundled_model.exists()}, savedir empty={not (savedir / 'hyperparams.yaml').exists()})")

            # Use local path if model exists in savedir, otherwise download from Hub
            if (savedir / "hyperparams.yaml").exists():
                logger.info(f"[AudioIntel] Loading from local savedir: {savedir}")
                source = str(savedir)
            else:
                logger.info("[AudioIntel] Loading from Hub: speechbrain/spkrec-ecapa-voxceleb")
                source = "speechbrain/spkrec-ecapa-voxceleb"

            self._speaker_model = SpeakerRecognition.from_hparams(
                source=source,
                savedir=str(savedir),
                run_opts={"device": device}
            )
            logger.info(f"[AudioIntel] SpeechBrain speaker model loaded on {device}")
        except Exception as e:
            logger.error(f"[AudioIntel] Failed to load SpeechBrain speaker model: {e}")
            raise
        
        # Session 2: Initialize emotion recognition model
        if self._enable_emotion:
            logger.info(f"[AudioIntel] Loading emotion recognition model ({device})...")
            try:
                self._emotion_model = EncoderClassifier.from_hparams(
                    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                    savedir=str(self._profile_dir / "models" / "emotion"),
                    run_opts={"device": device}
                )
                logger.info(f"[AudioIntel] Emotion model loaded on {device}")
            except Exception as e:
                logger.warning(f"[AudioIntel] Emotion model failed to load: {e}")
                self._enable_emotion = False
                self._emotion_model = None
        else:
            self._emotion_model = None
        
        # Session 3: Initialize prosody analyzer
        if self._enable_prosody:
            try:
                from .prosody_analyzer import ProsodyAnalyzer
                self._prosody_analyzer = ProsodyAnalyzer(sample_rate=sample_rate)
                logger.info("[AudioIntel] Prosody analyzer initialized")
            except ImportError as e:
                logger.warning(f"[AudioIntel] Prosody analyzer not available: {e}")
                self._enable_prosody = False
                self._prosody_analyzer = None
        else:
            self._prosody_analyzer = None
        
        # Load existing profiles
        self._load_profiles()
        
        logger.info(
            f"[AudioIntel] Initialized with {len(self._speakers)} speaker profiles, "
            f"threshold={similarity_threshold:.2f}"
        )
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames - buffer audio and handle speech events"""
        await super().process_frame(frame, direction)
        # If disabled, pass frames through without processing
        if not self._enabled:
            await self.push_frame(frame, direction)
            return
        
        # Buffer audio during speech
        if isinstance(frame, InputAudioRawFrame):
            if self._is_speaking:
                self._audio_buffer.extend(frame.audio)
                # Attempt early recognition once per utterance
                try:
                    await self._attempt_early_recognition()
                except Exception as e:
                    logger.debug(f"[AudioIntel] Early recognition skipped: {e}")
        
        # Speech boundary events
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._is_speaking = True
            self._audio_buffer.clear()
            self._early_recognition_done = False
            logger.debug("[AudioIntel] User started speaking, buffer cleared")
        
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._is_speaking:
                self._is_speaking = False
                # Process utterance in background (non-blocking)
                asyncio.create_task(self._process_utterance())
        
        # Always push frame downstream (parallel pipeline pattern)
        await self.push_frame(frame, direction)

    # --- Runtime control ---------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable audio intelligence processing at runtime."""
        self._enabled = bool(enabled)
        if self._enabled:
            logger.info("[AudioIntel] Processing ENABLED")
        else:
            logger.info("[AudioIntel] Processing DISABLED (paused until re-enabled)")

    def set_speaker_recognition_enabled(self, enabled: bool) -> None:
        """
        Enable/disable speaker recognition while keeping prosody active.

        This allows privacy-preserving mode where speaker recognition is disabled
        but prosody analysis continues for TRUE confidence scoring.

        Args:
            enabled: True to enable speaker recognition, False to disable it
        """
        self._speaker_recognition_enabled = bool(enabled)
        if self._speaker_recognition_enabled:
            logger.info("[AudioIntel] Speaker recognition ENABLED")
        else:
            logger.info("[AudioIntel] Speaker recognition DISABLED (prosody still active)")
    
    async def _process_utterance(self):
        """Process buffered audio for speaker recognition and prosody"""
        try:
            # Check duration
            duration = len(self._audio_buffer) / (self._sample_rate * 2)  # 16-bit audio
            if duration < self._min_utterance_duration:
                logger.debug(f"[AudioIntel] Skipping short utterance ({duration:.2f}s)")
                return

            # Convert to float32 numpy array
            audio_array = np.frombuffer(self._audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

            # Convert to torch tensor for SpeechBrain
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)  # Add batch dimension

            # Move to model device (MPS fallback handles unsupported ops)
            if self._device != "cpu":
                audio_tensor = audio_tensor.to(self._device)

            # Session 1: Extract speaker embedding (only if speaker recognition enabled)
            speaker_embedding = None
            if self._speaker_recognition_enabled:
                with torch.no_grad():
                    speaker_embedding = self._speaker_model.encode_batch(audio_tensor)
                    speaker_embedding = speaker_embedding.squeeze().cpu()  # Move to CPU for comparison
            
            # Session 2: Extract emotion (if enabled)
            emotion = None
            emotion_confidence = 0.0
            valence = 0.0
            arousal = 0.0
            
            if self._enable_emotion and self._emotion_model:
                try:
                    with torch.no_grad():
                        # SpeechBrain EncoderClassifier expects (batch, time)
                        # Ensure audio is on CPU for emotion model (some ops not supported on MPS)
                        audio_cpu = audio_tensor.cpu() if audio_tensor.device.type != "cpu" else audio_tensor
                        
                        # Use encode_batch for embeddings, then classify
                        out_prob, score, index, text_lab = self._emotion_model.classify_batch(audio_cpu)
                        
                        # Extract results
                        emotion = text_lab[0] if len(text_lab) > 0 else "neutral"
                        emotion_confidence = float(score[0]) if len(score) > 0 else 0.5
                        
                        # Map IEMOCAP labels to valence/arousal
                        valence_map = {
                            "ang": -0.6, "angry": -0.6,
                            "hap": 0.8, "happy": 0.8,
                            "sad": -0.7,
                            "neu": 0.0, "neutral": 0.0
                        }
                        arousal_map = {
                            "ang": 0.9, "angry": 0.9,
                            "hap": 0.7, "happy": 0.7,
                            "sad": 0.3,
                            "neu": 0.2, "neutral": 0.2
                        }
                        
                        valence = valence_map.get(emotion, 0.0)
                        arousal = arousal_map.get(emotion, 0.5)
                        
                        logger.debug(
                            f"[AudioIntel] Emotion: {emotion} "
                            f"(conf={emotion_confidence:.2f}, v={valence:.1f}, a={arousal:.1f})"
                        )
                except Exception as e:
                    logger.debug(f"[AudioIntel] Emotion detection skipped: {e}")
                    # Graceful degradation - continue without emotion
                    pass
            
            # Session 3: Extract prosody features (ALWAYS extract, regardless of speaker recognition)
            prosody_features = None
            prosody_certainty = 0.0

            if self._enable_prosody and self._prosody_analyzer:
                try:
                    prosody_features = self._prosody_analyzer.extract(audio_array)
                    if prosody_features:
                        prosody_certainty = prosody_features.certainty_modifier
                        logger.debug(f"[AudioIntel] Prosody: {prosody_features}")
                except Exception as e:
                    logger.warning(f"[AudioIntel] Prosody extraction failed: {e}")

            # If speaker recognition is disabled, emit prosody-only frame and return
            if not self._speaker_recognition_enabled:
                # Emit AudioIntelligenceFrame with prosody only (no speaker info)
                await self.push_frame(
                    AudioIntelligenceFrame(
                        speaker_id=self._current_speaker or "unknown",
                        speaker_confidence=0.0,
                        emotion=emotion,
                        emotion_confidence=emotion_confidence,
                        valence=valence,
                        arousal=arousal,
                        prosody_features=prosody_features,
                        prosody_certainty=prosody_certainty,
                    )
                )
                logger.debug(f"[AudioIntel] Emitted prosody-only frame (speaker recognition disabled)")
                return

            # Find best matching speaker (only if speaker recognition enabled)
            best_match, best_similarity = self._find_best_match(speaker_embedding)

            # Check if recognized
            if best_match and best_similarity >= self._similarity_threshold:
                if best_match != self._current_speaker:
                    # Speaker changed
                    self._current_speaker = best_match
                    real_name = self._get_valid_name(self._speaker_names.get(best_match))
                    
                    logger.info(
                        f"[AudioIntel] 🎯 Speaker recognized: {best_match} "
                        f"({real_name or 'unnamed'}) confidence={best_similarity:.2f}"
                    )
                    
                    # Emit speaker changed event
                    await self.push_frame(
                        SpeakerChangedFrame(
                            speaker_id=best_match,
                            speaker_name=real_name,
                            confidence=best_similarity,
                            auto_enrolled=False
                        )
                    )
                    
                    # Emit unified intelligence frame (Session 3: with prosody)
                    await self.push_frame(
                        AudioIntelligenceFrame(
                            speaker_id=best_match,
                            speaker_confidence=best_similarity,
                            emotion=emotion,
                            emotion_confidence=emotion_confidence,
                            valence=valence,
                            arousal=arousal,
                            prosody_features=prosody_features,
                            prosody_certainty=prosody_certainty,
                        )
                    )
                
                # Adapt profile (incremental learning with small alpha)
                self._adapt_profile(best_match, speaker_embedding)
            
            else:
                # Unknown speaker - auto-enroll
                await self._process_unknown_speaker(
                    speaker_embedding, emotion, emotion_confidence, valence, arousal,
                    prosody_features, prosody_certainty
                )
        
        except Exception as e:
            logger.error(f"[AudioIntel] Error processing utterance: {e}", exc_info=True)

    async def _attempt_early_recognition(self):
        """Try to recognize returning user mid‑utterance for faster UX."""
        if self._early_recognition_done:
            return
        # Require at least _early_ms of audio
        min_bytes = int(self._sample_rate * (self._early_ms / 1000.0) * 2)  # 16‑bit mono
        if len(self._audio_buffer) < min_bytes:
            return
        # Build tensor from current buffer slice
        audio_array = np.frombuffer(bytes(self._audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
        if self._device != "cpu":
            audio_tensor = audio_tensor.to(self._device)
        with torch.no_grad():
            emb = self._speaker_model.encode_batch(audio_tensor).squeeze().cpu()
        best_match, best_similarity = self._find_best_match(emb)
        if best_match and best_similarity >= self._similarity_threshold:
            if best_match != self._current_speaker:
                self._current_speaker = best_match
                real_name = self._get_valid_name(self._speaker_names.get(best_match))
                logger.info(
                    f"[AudioIntel] ⚡ Early recognition: {best_match} ({real_name or 'unnamed'}) conf={best_similarity:.2f}"
                )
                await self.push_frame(
                    SpeakerChangedFrame(
                        speaker_id=best_match,
                        speaker_name=real_name,
                        confidence=best_similarity,
                        auto_enrolled=False
                    )
                )
                await self.push_frame(
                    AudioIntelligenceFrame(
                        speaker_id=best_match,
                        speaker_confidence=best_similarity,
                        emotion=None,
                        emotion_confidence=0.0,
                    )
                )
        self._early_recognition_done = True
    
    def _find_best_match(self, embedding: torch.Tensor) -> Tuple[Optional[str], float]:
        """Find best matching speaker by cosine similarity"""
        best_match = None
        best_similarity = 0.0
        
        embedding = embedding.cpu()
        
        for speaker_id, stored_embeddings in self._speakers.items():
            for stored_embedding in stored_embeddings:
                stored_embedding = stored_embedding.cpu()
                
                # Cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    embedding.unsqueeze(0),
                    stored_embedding.unsqueeze(0)
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker_id
        
        return best_match, best_similarity
    
    def _adapt_profile(self, speaker_id: str, new_embedding: torch.Tensor):
        """Incrementally adapt speaker profile (EMA with small alpha)"""
        try:
            if speaker_id not in self._speakers or not self._speakers[speaker_id]:
                return
            
            alpha = 0.05  # Small alpha for stability
            centroid = self._speakers[speaker_id][0]
            updated = (1 - alpha) * centroid + alpha * new_embedding
            
            # Re-normalize
            updated = updated / torch.norm(updated)
            
            self._speakers[speaker_id][0] = updated
            logger.debug(f"[AudioIntel] Adapted profile for {speaker_id}")
        
        except Exception as e:
            logger.error(f"[AudioIntel] Profile adaptation failed: {e}")
    
    async def _process_unknown_speaker(self, embedding: torch.Tensor, 
                                       emotion: Optional[str] = None,
                                       emotion_confidence: float = 0.0,
                                       valence: float = 0.0,
                                       arousal: float = 0.0,
                                       prosody_features: Optional[Any] = None,
                                       prosody_certainty: float = 0.0):
        """Privacy-aware unknown speaker handling"""
        current_time = time.time()
        
        # Clean old samples (>5 min window)
        self._unknown_embeddings = [
            (emb, ts) for emb, ts in self._unknown_embeddings
            if current_time - ts < self._consent_timeout
        ]
        
        # PRIVACY-FIRST: First unknown detection
        if not self._unknown_embeddings and not self._consent_prompted:
            if self._require_consent:
                # Ask for consent BEFORE enrolling
                emb_hash = self._hash_embedding(embedding)
                self._pending_consent_hash = emb_hash
                self._consent_prompted = True
                
                logger.info("[AudioIntel] 🔒 Unknown speaker detected - requesting consent")
                
                # Emit frame to trigger bot prompt
                await self.push_frame(
                    UnknownSpeakerDetectedFrame(
                        embedding_hash=emb_hash
                    )
                )
                
                if self._privacy_mode == "ephemeral":
                    # Don't store anything until consent
                    logger.debug("[AudioIntel] Ephemeral mode: not storing data")
                    return
                elif self._privacy_mode == "consent_pending":
                    # Store temporarily for potential enrollment
                    self._unknown_embeddings.append((embedding, current_time))
                    logger.debug("[AudioIntel] Consent pending: temporary storage")
                    return
            
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
                    consistency=0.75,  # First sample is 100% consistent with itself
                    speaker_id="unknown"
                )
            )
            return
        
        # Check consistency with existing samples
        similarities = []
        for stored_embedding, _ in self._unknown_embeddings:
            similarity = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0),
                stored_embedding.cpu().unsqueeze(0)
            ).item()
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        
        # If consistent, add to collection
        if avg_similarity >= self._consistency_threshold * 0.85:
            self._unknown_embeddings.append((embedding, current_time))
            current_count = len(self._unknown_embeddings)
            
            logger.debug(
                f"[AudioIntel] Sample {current_count}/{self._auto_enroll_utterances} "
                f"(consistency={avg_similarity:.2f})"
            )
            
            # Emit progress frame for EnrollmentCoordinator
            await self.push_frame(
                EnrollmentProgressFrame(
                    current_sample=current_count,
                    total_samples=self._auto_enroll_utterances,
                    consistency=avg_similarity,
                    speaker_id="unknown"
                )
            )
            
            # Check if ready to enroll
            if current_count >= self._auto_enroll_utterances:
                await self._auto_enroll_speaker(
                    emotion=emotion,
                    emotion_confidence=emotion_confidence,
                    valence=valence,
                    arousal=arousal,
                    prosody_features=prosody_features,
                    prosody_certainty=prosody_certainty,
                )
        else:
            # Inconsistent - reset (normal for real-world audio with noise)
            logger.debug(
                f"[AudioIntel] Inconsistent sample ({avg_similarity:.2f}), resetting"
            )
            self._unknown_embeddings = [(embedding, current_time)]
    
    async def _auto_enroll_speaker(
        self,
        *,
        emotion: Optional[str] = None,
        emotion_confidence: float = 0.0,
        valence: float = 0.0,
        arousal: float = 0.0,
        prosody_features: Optional[Any] = None,
        prosody_certainty: float = 0.0,
    ):
        """Enroll speaker (with name if provided via consent)"""
        try:
            embeddings = [emb for emb, _ in self._unknown_embeddings]
            
            # Calculate consistency
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = torch.nn.functional.cosine_similarity(
                        embeddings[i].unsqueeze(0),
                        embeddings[j].cpu().unsqueeze(0)
                    ).item()
                    similarities.append(sim)
            
            consistency = np.mean(similarities) if similarities else 0.0
            
            if consistency >= self._consistency_threshold:
                # Create speaker ID (use provided name if available)
                if self._enrollment_name:
                    speaker_id = self._enrollment_name
                    logger.info(f"[AudioIntel] Enrolling as: {speaker_id} (user-provided name)")
                else:
                    self._speaker_counter += 1
                    speaker_id = f"Speaker_{self._speaker_counter}"
                    logger.info(f"[AudioIntel] Auto-enrolling as: {speaker_id}")
                
                # Compute centroid
                centroid = torch.stack([e.cpu() for e in embeddings]).mean(dim=0)
                centroid = centroid / torch.norm(centroid)
                
                # Store profile
                self._speakers[speaker_id] = [centroid]
                self._save_profile(speaker_id, [centroid])
                
                # Update state
                self._current_speaker = speaker_id
                self._unknown_embeddings.clear()
                self._collecting_samples = False
                
                logger.info(
                    f"[AudioIntel] ✨ Enrolled: {speaker_id} "
                    f"(samples={len(embeddings)}, consistency={consistency:.2f})"
                )
                
                # Clear enrollment name for next speaker
                self._enrollment_name = None
                
                # Emit events
                await self.push_frame(
                    SpeakerChangedFrame(
                        speaker_id=speaker_id,
                        speaker_name=None,
                        confidence=consistency,
                        auto_enrolled=True
                    )
                )
                
                await self.push_frame(
                    AudioIntelligenceFrame(
                        speaker_id=speaker_id,
                        speaker_confidence=consistency,
                        emotion=emotion,
                        emotion_confidence=emotion_confidence,
                        valence=valence,
                        arousal=arousal,
                        prosody_features=prosody_features,
                        prosody_certainty=prosody_certainty,
                    )
                )
            else:
                logger.warning(
                    f"[AudioIntel] Enrollment rejected: consistency={consistency:.2f}"
                )
                self._unknown_embeddings.clear()
        
        except Exception as e:
            logger.error(f"[AudioIntel] Auto-enrollment failed: {e}", exc_info=True)
    
    def _hash_embedding(self, embedding: torch.Tensor) -> str:
        """Create hash for tracking pending enrollment (privacy)"""
        import hashlib
        emb_bytes = embedding.cpu().numpy().tobytes()
        return hashlib.sha256(emb_bytes).hexdigest()[:16]
    
    async def _handle_enrollment_consent(self, frame: StartEnrollmentFrame):
        """Handle user consent with name (Privacy-First)"""
        logger.info(f"[AudioIntel] ✅ Consent received for: {frame.speaker_name}")
        
        # Store name for enrollment
        self._enrollment_name = frame.speaker_name
        self._consent_prompted = False  # Reset for future unknowns
        
        if self._privacy_mode == "ephemeral":
            # Was not storing - need to start fresh collection
            logger.info(f"[AudioIntel] Starting enrollment for {frame.speaker_name}")
            self._unknown_embeddings.clear()
        elif self._privacy_mode == "consent_pending":
            # Already have 1 sample - continue collection
            logger.info(f"[AudioIntel] Continuing enrollment for {frame.speaker_name}")
    
    def _save_profile(self, speaker_id: str, embeddings: List[torch.Tensor]):
        """Save speaker profile to disk"""
        try:
            auto_dir = self._profile_dir / "auto_enrolled"
            auto_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = auto_dir / f"{speaker_id}.pt"
            
            # Save as PyTorch tensor
            torch.save({
                "speaker_id": speaker_id,
                "embeddings": [e.cpu() for e in embeddings],
                "auto_enrolled": True,
                "enrolled_at": time.time(),
            }, filepath)
            
            logger.debug(f"[AudioIntel] Saved profile: {filepath}")
        
        except Exception as e:
            logger.error(f"[AudioIntel] Failed to save profile: {e}")
    
    def _load_profiles(self):
        """Load speaker profiles from disk"""
        try:
            auto_dir = self._profile_dir / "auto_enrolled"
            if not auto_dir.exists():
                return
            
            for filepath in auto_dir.glob("*.pt"):
                try:
                    data = torch.load(filepath, map_location="cpu")
                    
                    speaker_id = data["speaker_id"]
                    embeddings = data["embeddings"]
                    
                    self._speakers[speaker_id] = embeddings
                    
                    # Track counter
                    if speaker_id.startswith("Speaker_"):
                        num = int(speaker_id.split("_")[1])
                        self._speaker_counter = max(self._speaker_counter, num)
                    
                    logger.info(f"[AudioIntel] Loaded profile: {speaker_id}")
                
                except Exception as e:
                    logger.error(f"[AudioIntel] Failed to load {filepath.name}: {e}")
            
            # Load speaker names
            names_file = self._profile_dir / "speaker_names.json"
            if names_file.exists():
                with open(names_file) as f:
                    data = json.load(f)
                    raw_map = data.get("mappings", {})
                    sanitized = {}
                    removed = 0
                    for sid, nm in raw_map.items():
                        valid = self._get_valid_name(nm)
                        if valid:
                            sanitized[sid] = valid
                        else:
                            removed += 1
                    self._speaker_names = sanitized
                    logger.info(f"[AudioIntel] Loaded {len(self._speaker_names)} valid name mappings (removed {removed} invalid)")
                    if removed > 0 and os.getenv("SANITIZE_SPEAKER_NAMES_ON_LOAD", "true").lower() in ("1", "true", "yes"):
                        try:
                            with open(names_file, "w") as wf:
                                json.dump({"mappings": self._speaker_names}, wf, indent=2)
                            logger.info("[AudioIntel] Rewrote speaker_names.json without invalid entries")
                        except Exception as e:
                            logger.warning(f"[AudioIntel] Failed to rewrite speaker_names.json: {e}")
        
        except Exception as e:
            logger.error(f"[AudioIntel] Failed to load profiles: {e}")
    
    @property
    def current_speaker(self) -> Optional[str]:
        """Get current speaker ID"""
        return self._current_speaker
    
    def get_speaker_name(self, speaker_id: str) -> Optional[str]:
        """Get real name for speaker ID"""
        return self._speaker_names.get(speaker_id)
    
    def set_speaker_name(self, speaker_id: str, name: str) -> bool:
        """Assign a validated real name to speaker ID. Returns True if saved."""
        if speaker_id not in self._speakers:
            logger.warning(f"[AudioIntel] Unknown speaker ID: {speaker_id}")
            return False

        norm = self._normalize_name_candidate(name)
        if not self._is_valid_name_candidate(norm):
            logger.warning(f"[AudioIntel] Rejected invalid speaker name: '{name}'")
            return False

        self._speaker_names[speaker_id] = norm

        try:
            names_file = self._profile_dir / "speaker_names.json"
            with open(names_file, "w") as f:
                json.dump({"mappings": self._speaker_names}, f, indent=2)
            logger.info(f"[AudioIntel] Named {speaker_id} as '{norm}'")
            return True
        except Exception as e:
            logger.error(f"[AudioIntel] Failed to save name mapping: {e}")
            return False

    # ----- Name validation helpers -----
    def _normalize_name_candidate(self, text: str) -> str:
        import re
        cleaned = re.sub(r"[^A-Za-z'\-\s]", "", (text or "")).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.title()

    def _fixed_phrase(self) -> str:
        return os.getenv("ENROLLMENT_FIXED_PHRASE", "LocalCat learns my voice.").strip()

    def _is_valid_name_candidate(self, candidate: str) -> bool:
        if not candidate:
            return False
        if len(candidate) > 20:
            return False
        tokens = candidate.split()
        if not (1 <= len(tokens) <= 3):
            return False
        try:
            ratio = difflib.SequenceMatcher(None, candidate.lower(), self._fixed_phrase().lower()).ratio()
            if ratio >= 0.6:
                return False
        except Exception:
            pass
        return any(c.isalpha() for c in tokens[0])

    def _get_valid_name(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        norm = self._normalize_name_candidate(name)
        return norm if self._is_valid_name_candidate(norm) else None
