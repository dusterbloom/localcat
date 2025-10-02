"""
Enrollment Coordinator - Single Responsibility Principle

Coordinates enrollment process and generates user feedback.
Orchestrates between AudioIntelligenceProcessor events and pipeline router.
Adds privacy-first onboarding with ephemeral choice and name capture.
"""

import asyncio
import os
import difflib
from pathlib import Path
from typing import Optional
from loguru import logger

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import FrameDirection

from .audio_intelligence import (
    UnknownSpeakerDetectedFrame,
    SpeakerChangedFrame,
    EnrollmentProgressFrame,
)
from .enrollment_state import EnrollmentState
from .enrollment_messages import EnrollmentMessages
from .pipeline_router import SpeakerEnrollmentRouter


class EnrollmentCoordinator(FrameProcessor):
    """
    Coordinates enrollment process and generates user feedback.
    
    Single Responsibility: Enrollment orchestration and user feedback only.
    Interface Segregation: Minimal dependencies on router and messages.
    Dependency Inversion: Depends on abstractions (router interface).
    
    Responsibilities:
    - Listen for enrollment events from AudioIntelligenceProcessor
    - Generate appropriate user feedback messages
    - Update router state based on enrollment progress
    - Handle first-time vs returning user flows
    """
    
    def __init__(
        self,
        router: SpeakerEnrollmentRouter,
        profile_dir: str = "data/speaker_profiles",
        skip_for_returning: bool = True,
        include_privacy_explanation: bool = False,
        messages: Optional[EnrollmentMessages] = None,
        *,
        audio_intel: Optional[object] = None,
        memory: Optional[object] = None,
        enable_ephemeral_choice: bool = True,
    ):
        """
        Initialize enrollment coordinator.
        
        Args:
            router: Pipeline router to control state transitions
            profile_dir: Directory containing speaker profiles
            skip_for_returning: Skip intro for users with existing profiles
            include_privacy_explanation: Include privacy info in intro
            messages: Custom messages (optional, uses defaults if None)
        """
        super().__init__()
        self._router = router
        self._profile_dir = Path(profile_dir)
        self._skip_for_returning = skip_for_returning
        self._include_privacy = include_privacy_explanation
        self._messages = messages or EnrollmentMessages.from_env()
        self._audio_intel = audio_intel
        self._memory = memory
        self._enable_ephemeral_choice = enable_ephemeral_choice
        
        # State tracking
        self._intro_sent = False
        self._last_progress = 0
        self._enrollment_started = False
        self._awaiting_choice = enable_ephemeral_choice
        self._name_capture_pending = False
        self._pending_speaker_id: Optional[str] = None
        self._ephemeral_selected = False
        # Fixed phrase configuration
        self._fixed_phrase = os.getenv("ENROLLMENT_FIXED_PHRASE", "LocalCat learns my voice.").strip()
        self._require_fixed_phrase = os.getenv("ENROLLMENT_REQUIRE_FIXED_PHRASE", "false").lower() in ("1", "true", "yes")
        self._returning_greeting = os.getenv("VOICE_AGENT_RETURNING_GREETING", "welcome_back").strip().lower()
        # Name capture confirmation
        self._pending_name_candidate: Optional[str] = None
        self._awaiting_name_confirmation: bool = False
        # Choice keyword sets (configurable)
        self._sign_me_up_terms = self._load_terms(
            os.getenv(
                "SIGN_ME_UP_TERMS",
                "sign me up|register me|enroll me|get started|create profile",
            )
        )
        self._sign_in_terms = self._load_terms(
            os.getenv(
                "SIGN_IN_TERMS",
                "sign in|log in|i'm back|its me|it's me|recognize me",
            )
        )
        self._anonymous_terms = self._load_terms(
            os.getenv(
                "ANONYMOUS_TERMS",
                "anonymous|private|don't store|do not store|skip|no",
            )
        )
        self._sign_in_requested: bool = False
        self._sign_in_timeout_task: Optional[asyncio.Task] = None
        
        logger.debug(
            f"[EnrollmentCoordinator] Initialized "
            f"(skip_returning={skip_for_returning}, "
            f"include_privacy={include_privacy_explanation})"
        )
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process enrollment-related frames and generate feedback."""
        await super().process_frame(frame, direction)
        
        # CRITICAL: Always forward StartFrame immediately for pipeline initialization
        from pipecat.frames.frames import StartFrame
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            # Kick off initial choice when enabled
            if self._enable_ephemeral_choice and self._awaiting_choice:
                await self._router.update_state(EnrollmentState.CHOICE)
                await self._send_choice_message(direction)
            return
        
        # Handle unknown speaker detection (first utterance)
        if isinstance(frame, UnknownSpeakerDetectedFrame):
            await self._handle_unknown_speaker_detected(frame, direction)
            # Forward system frame
            await self.push_frame(frame, direction)
            return

        # Handle enrollment progress updates
        elif isinstance(frame, EnrollmentProgressFrame):
            await self._handle_enrollment_progress(frame, direction)
            # Forward system frame
            await self.push_frame(frame, direction)
            return
        
        # Handle speaker changed (enrollment complete or recognized)
        elif isinstance(frame, SpeakerChangedFrame):
            await self._handle_speaker_changed(frame, direction)
            # Forward system frame
            await self.push_frame(frame, direction)
            return

        # Handle user transcriptions for choice and name capture flows
        elif isinstance(frame, TranscriptionFrame):
            # Swallow user transcriptions during onboarding states to prevent LLM/memory pollution
            state = self._router.current_state
            if state in (
                EnrollmentState.CHOICE,
                EnrollmentState.INTRO,
                EnrollmentState.ENROLLING,
                EnrollmentState.TRANSITION,
                EnrollmentState.NAME_CAPTURE,
            ):
                await self._handle_transcription(frame, direction)
                # Do NOT forward this transcription downstream during onboarding
                return
            else:
                # Normal conversation: allow downstream
                await self.push_frame(frame, direction)
                return
        
        # Forward all other frames
        await self.push_frame(frame, direction)
    
    async def _send_choice_message(self, direction: FrameDirection):
        """Ask the user to choose anonymous chat vs quick enrollment."""
        text = (
            "Would you like to sign me up (quick voice enrollment ~10s), sign in (if you've been here before), "
            "or chat anonymously without storing anything? Say 'sign me up', 'sign in', or 'anonymous'."
        )
        await self.push_frame(TextFrame(text), direction)
        # Optional: hint UI input is available
        await self.push_frame(TextFrame("You can also choose by clicking in the UI."), direction)

    async def _send_intro_message(self, direction: FrameDirection):
        """
        Send introduction message and prepare for enrollment.
        Extracted as helper to support both consent and auto-enroll flows.
        """
        if self._intro_sent:
            return  # Already handled
        
        # Send introduction message
        intro_text = self._messages.get_intro(
            include_privacy=self._include_privacy
        )
        logger.info(f"[EnrollmentCoordinator] Sending intro: '{intro_text[:80]}...'")
        
        # Update router to intro state
        await self._router.update_state(EnrollmentState.INTRO)
        
        await self.push_frame(TextFrame(intro_text), direction)
        # If using fixed phrase, instruct the user
        if self._fixed_phrase:
            instruct = f"Please repeat exactly: '{self._fixed_phrase}' three times."
            await self.push_frame(TextFrame(instruct), direction)
        
        self._intro_sent = True
        
        # Brief pause after intro before starting enrollment
        await asyncio.sleep(0.1)
    
    async def _handle_unknown_speaker_detected(
        self,
        frame: UnknownSpeakerDetectedFrame,
        direction: FrameDirection
    ):
        """
        Handle first detection of unknown speaker (consent-required mode).
        Send intro message and transition to enrolling state.
        """
        # If user hasn't chosen yet, ignore unknown-speaker prompts
        if self._awaiting_choice:
            logger.debug("[EnrollmentCoordinator] Ignoring UnknownSpeakerDetected while awaiting choice")
            return
        if self._intro_sent:
            return  # Already handled
        
        logger.info("[EnrollmentCoordinator] Unknown speaker detected - starting intro")
        
        await self._send_intro_message(direction)
        self._enrollment_started = True
        
        # Automatically transition to enrolling state
        # (AudioIntelligenceProcessor will start collecting samples)
        await self._router.update_state(EnrollmentState.ENROLLING, progress=0)
    
    async def _handle_enrollment_progress(
        self,
        frame: EnrollmentProgressFrame,
        direction: FrameDirection
    ):
        """
        Handle enrollment progress updates.
        Provide feedback at key milestones.
        """
        current = frame.current_sample
        total = frame.total_samples
        
        # If awaiting choice, do not start enrollment yet
        if self._awaiting_choice:
            logger.debug("[EnrollmentCoordinator] Ignoring EnrollmentProgress while awaiting choice")
            return

        # CRITICAL FIX: Start enrollment on first progress frame if not already started
        # This handles auto_enroll mode where UnknownSpeakerDetectedFrame is not emitted
        if not self._enrollment_started and current == 1:
            logger.info("[EnrollmentCoordinator] Starting enrollment from first progress frame")
            await self._send_intro_message(direction)
            self._enrollment_started = True
            await self._router.update_state(EnrollmentState.ENROLLING, progress=0)
            
            # CRITICAL: Give intro message time to be spoken before sending progress update
            # The intro message takes ~2-3 seconds to synthesize and speak
            await asyncio.sleep(3.0)  # Wait for intro to finish speaking
        
        if not self._enrollment_started:
            return  # Still not in enrollment flow
        
        # Only provide feedback on progress changes
        if current == self._last_progress:
            return
        
        self._last_progress = current
        
        logger.debug(
            f"[EnrollmentCoordinator] Progress: {current}/{total} "
            f"(consistency={frame.consistency:.2f})"
        )
        
        # Provide feedback for each sample collected
        if current < total:
            progress_text = self._messages.get_progress(current, total)
            logger.info(f"[EnrollmentCoordinator] Sending progress: '{progress_text}'")
            await self.push_frame(TextFrame(progress_text), direction)
            
            # Update router with current progress
            await self._router.update_state(
                EnrollmentState.ENROLLING,
                progress=current,
                consistency=frame.consistency
            )
    
    async def _handle_speaker_changed(
        self,
        frame: SpeakerChangedFrame,
        direction: FrameDirection
    ):
        """
        Handle speaker changed event.
        This fires when enrollment completes or a known speaker is recognized.
        """
        # In anonymous mode, suppress enrollment/returning greetings
        if self._ephemeral_selected:
            return

        # While awaiting choice, ignore recognition changes
        if self._awaiting_choice:
            logger.debug("[EnrollmentCoordinator] Ignoring SpeakerChanged while awaiting choice")
            return

        # Always handle auto-enrolled completion immediately
        if frame.auto_enrolled:
            await self._handle_enrollment_complete(frame, direction)
            return

        # Recognized returning user with a known name → fast path
        if frame.speaker_name:
            await self._handle_returning_user(frame, direction)
            return

        # Recognized existing unnamed profile → capture a friendly name
        await self._router.update_state(
            EnrollmentState.NAME_CAPTURE,
            speaker_id=frame.speaker_id,
            consistency=frame.confidence
        )
        self._name_capture_pending = True
        self._pending_speaker_id = frame.speaker_id
        await self.push_frame(TextFrame("Great, I recognized your voice. What name or ID should I use for you?"), direction)
    
    async def _handle_enrollment_complete(
        self,
        frame: SpeakerChangedFrame,
        direction: FrameDirection
    ):
        """
        Handle enrollment completion.
        Send completion message and transition to conversation.
        """
        logger.info(
            f"[EnrollmentCoordinator] Enrollment complete: {frame.speaker_id} "
            f"(confidence={frame.confidence:.2f})"
        )
        
        # Move to transition -> name capture
        await self._router.update_state(
            EnrollmentState.TRANSITION,
            progress=self._router.current_progress.total,
            speaker_id=frame.speaker_id,
            consistency=frame.confidence
        )

        # Optional short success cue followed by name capture prompt
        await self.push_frame(TextFrame("Perfect! I've enrolled your voice."), direction)
        await asyncio.sleep(0.2)

        # Ask for preferred ID/name
        await self._router.update_state(EnrollmentState.NAME_CAPTURE, speaker_id=frame.speaker_id)
        self._name_capture_pending = True
        self._pending_speaker_id = frame.speaker_id
        await self.push_frame(TextFrame("What name or ID should I use for you? You can also type it in the UI."), direction)
    
    async def _handle_returning_user(
        self,
        frame: SpeakerChangedFrame,
        direction: FrameDirection
    ):
        """
        Handle returning user recognition.
        Skip intro and go directly to conversation.
        """
        logger.info(
            f"[EnrollmentCoordinator] Returning user: {frame.speaker_id} "
            f"(confidence={frame.confidence:.2f})"
        )
        
        # Optional: set memory user identity to recognized name/id
        try:
            if self._memory and hasattr(self._memory, 'set_user_identity'):
                name_or_id = frame.speaker_name or frame.speaker_id
                self._memory.set_user_identity(name_or_id)
        except Exception as e:
            logger.warning(f"[EnrollmentCoordinator] Failed to set memory user identity: {e}")

        # Go directly to conversation mode
        await self._router.update_state(
            EnrollmentState.CONVERSATION,
            speaker_id=frame.speaker_id,
            consistency=frame.confidence
        )
        
        # Send greeting (configurable minimal/welcome_back)
        if self._returning_greeting == "minimal":
            await self.push_frame(TextFrame("Hello, how can I help you today?"), direction)
        else:
            welcome_text = self._messages.get_welcome_back(frame.speaker_name)
            await self.push_frame(TextFrame(welcome_text), direction)
        
        self._intro_sent = True  # Mark as handled
        
        logger.info("[EnrollmentCoordinator] Skipped intro for returning user")

    async def _handle_transcription(self, frame: TranscriptionFrame, direction: FrameDirection):
        """Handle user transcriptions during choice and name capture flows."""
        text = (frame.text or "").strip()
        if not text:
            return

        # Handle initial choice
        if self._enable_ephemeral_choice and self._awaiting_choice and self._router.current_state == EnrollmentState.CHOICE:
            normalized = text.lower()
            if self._contains_any(normalized, self._sign_me_up_terms):
                self._awaiting_choice = False
                await self._router.update_state(EnrollmentState.INTRO)
                await self._send_intro_message(direction)
                self._enrollment_started = True
                await self._router.update_state(EnrollmentState.ENROLLING, progress=0)
                return
            if self._contains_any(normalized, self._anonymous_terms):
                self._awaiting_choice = False
                self._ephemeral_selected = True
                # Enable ephemeral memory (no storage/extraction)
                try:
                    if self._memory and hasattr(self._memory, 'set_ephemeral_mode'):
                        self._memory.set_ephemeral_mode(True)
                except Exception as e:
                    logger.warning(f"[EnrollmentCoordinator] Failed to set ephemeral mode: {e}")
                await self.push_frame(TextFrame("Okay, let's chat anonymously. Nothing will be stored."), direction)
                await self._router.update_state(EnrollmentState.CONVERSATION)
                # Send neutral greeting to start the convo without personal data
                await self.push_frame(TextFrame("Hello, how can I help you today?"), direction)
                return
            if self._contains_any(normalized, self._sign_in_terms):
                # Enter returning-user fast path
                self._awaiting_choice = False
                self._sign_in_requested = True
                await self.push_frame(TextFrame("Okay — say a few words and I’ll sign you in."), direction)
                # Start a soft timeout in case recognition doesn’t trigger
                if not self._sign_in_timeout_task:
                    self._sign_in_timeout_task = asyncio.create_task(self._sign_in_timeout(direction))
                return

        # Handle fixed-phrase guidance (nudge only)
        if self._router.current_state == EnrollmentState.ENROLLING and self._fixed_phrase and self._require_fixed_phrase:
            ratio = difflib.SequenceMatcher(None, text.lower(), self._fixed_phrase.lower()).ratio()
            if ratio < 0.6:
                await self.push_frame(TextFrame(f"Please try to say exactly: '{self._fixed_phrase}'."), direction)

        # Handle name capture with validation + confirmation
        if self._name_capture_pending and self._router.current_state == EnrollmentState.NAME_CAPTURE:
            # Confirmation branch
            if self._awaiting_name_confirmation:
                norm = text.lower()
                if any(x in norm for x in ("yes", "correct", "right", "that's right", "that is right")):
                    name = self._pending_name_candidate or ""
                    sid = self._pending_speaker_id
                    # Persist name to audio intelligence
                    saved_ok = False
                    if sid and self._audio_intel and hasattr(self._audio_intel, 'set_speaker_name'):
                        try:
                            saved_ok = bool(self._audio_intel.set_speaker_name(sid, name))
                        except Exception as e:
                            logger.warning(f"[EnrollmentCoordinator] Failed to set speaker name: {e}")
                    if not saved_ok:
                        await self.push_frame(TextFrame("That didn’t look like a valid name. Please say a short name."), direction)
                        # Stay in NAME_CAPTURE
                        return
                    # Bind memory identity
                    try:
                        if self._memory and hasattr(self._memory, 'set_user_identity'):
                            self._memory.set_user_identity(name)
                    except Exception as e:
                        logger.warning(f"[EnrollmentCoordinator] Failed to set memory user identity: {e}")
                    await self.push_frame(TextFrame(f"Thanks, {name}."), direction)
                    self._name_capture_pending = False
                    self._pending_speaker_id = None
                    self._pending_name_candidate = None
                    self._awaiting_name_confirmation = False
                    await self._router.update_state(EnrollmentState.CONVERSATION, speaker_id=sid)
                    logger.info("[EnrollmentCoordinator] Transitioned to conversation mode")
                    return
                if any(x in norm for x in ("no", "nope", "not correct", "wrong")):
                    self._pending_name_candidate = None
                    self._awaiting_name_confirmation = False
                    await self.push_frame(TextFrame("No problem — what name should I use?"), direction)
                    return
                # If neither yes/no, gently reprompt
                await self.push_frame(TextFrame("Please say 'yes' or 'no'."), direction)
                return

            # Validation branch
            candidate = self._normalize_name_candidate(text)
            if not self._is_valid_name_candidate(candidate):
                await self.push_frame(TextFrame("I didn’t catch a valid name. Please say a short name, or type it in the UI."), direction)
                return
            # Ask for confirmation
            self._pending_name_candidate = candidate
            self._awaiting_name_confirmation = True
            await self.push_frame(TextFrame(f"Did I get that right: '{candidate}'? Say 'yes' or 'no'."), direction)
            return

    def _load_terms(self, raw: str) -> set:
        parts = [p.strip().lower() for p in raw.replace(",", "|").split("|") if p.strip()]
        return set(parts)

    def _contains_any(self, text: str, terms: set) -> bool:
        return any(t in text for t in terms)

    def _normalize_name_candidate(self, text: str) -> str:
        # Keep only letters, hyphens, apostrophes, and spaces; collapse spaces
        import re
        cleaned = re.sub(r"[^A-Za-z'\-\s]", "", text).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Title-case simple names
        return cleaned.title()

    def _is_valid_name_candidate(self, candidate: str) -> bool:
        if not candidate:
            return False
        if len(candidate) > 20:
            return False
        tokens = candidate.split()
        if not (1 <= len(tokens) <= 3):
            return False
        # Heuristic: candidate must not be too similar to fixed phrase
        try:
            ratio = difflib.SequenceMatcher(None, candidate.lower(), self._fixed_phrase.lower()).ratio()
            if ratio >= 0.6:
                return False
        except Exception:
            pass
        # Require letters in first token
        return any(c.isalpha() for c in tokens[0])

    async def _sign_in_timeout(self, direction: FrameDirection):
        try:
            await asyncio.sleep(2.0)
            # If still not in conversation, suggest fallback
            if self._router.current_state != EnrollmentState.CONVERSATION and self._sign_in_requested:
                await self.push_frame(TextFrame("I couldn’t confidently recognize you. Try again, or say 'sign me up' to create a new profile."), direction)
        except Exception:
            pass

    
    def has_existing_profiles(self) -> bool:
        """Check if any speaker profiles exist"""
        try:
            auto_dir = self._profile_dir / "auto_enrolled"
            if not auto_dir.exists():
                return False
            
            # Check for .pt files
            profiles = list(auto_dir.glob("*.pt"))
            return len(profiles) > 0
        except Exception as e:
            logger.warning(f"[EnrollmentCoordinator] Error checking profiles: {e}")
            return False
