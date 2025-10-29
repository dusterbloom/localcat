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
import re
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
        context_aggregator: Optional[object] = None,
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
        self._context_aggregator = context_aggregator
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
                # Removed generic 'no' to avoid false triggers (e.g., "I know")
                "anonymous|private|don't store|do not store|skip",
            )
        )
        self._sign_in_requested: bool = False
        self._sign_in_timeout_task: Optional[asyncio.Task] = 5.0
        # Suppress one immediate transcription after returning-user recognition
        self._suppress_next_transcription: bool = False
        self._suppress_deadline_ts: float = 0.0
        
        logger.debug(
            f"[EnrollmentCoordinator] Initialized "
            f"(skip_returning={skip_for_returning}, "
            f"include_privacy={include_privacy_explanation})"
        )

    def should_send_initial_prompt(self) -> bool:
        """
        Check if coordinator wants to send initial choice prompt.
        Called by bot.py on_pipeline_started hook.
        """
        return self._enable_ephemeral_choice and self._awaiting_choice

    def get_initial_prompts(self) -> list:
        """
        Return frames for initial choice prompt.
        Called by bot.py on_pipeline_started hook.
        """
        from pipecat.frames.frames import TextFrame

        if not self.should_send_initial_prompt():
            return []

        text = (
            "Would you like to sign up, sign in, "
            "or chat anonymously without storing anything? Say 'sign up', 'recognize me', or 'anonymous'."
        )

        return [
            TextFrame(text),
            TextFrame("So...")
        ]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process enrollment-related frames and generate feedback."""
        await super().process_frame(frame, direction)
        
        # CRITICAL: Always forward StartFrame immediately for pipeline initialization
        from pipecat.frames.frames import StartFrame, OutputTransportReadyFrame
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            # DO NOT trigger intro here - wait for OutputTransportReadyFrame to ensure connection is ready
            return

        # Simply forward OutputTransportReadyFrame
        # (Initial prompts now handled by bot.py's on_pipeline_started hook)
        if isinstance(frame, OutputTransportReadyFrame):
            await self.push_frame(frame, direction)
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
                # Drop the very next transcription immediately after returning-user recognition
                if self._suppress_next_transcription:
                    try:
                        now = asyncio.get_event_loop().time()
                        if now <= self._suppress_deadline_ts:
                            self._suppress_next_transcription = False
                            logger.debug("[EnrollmentCoordinator] Dropped post-recognition transcription to avoid LLM overlap")
                            return
                    except Exception:
                        pass
                    self._suppress_next_transcription = False
                # Conversation mode: handle logout to re-enable recognition
                try:
                    text = getattr(frame, 'text', '') or ''
                    norm = text.strip().lower()
                    if any(t in norm for t in ("logout", "log out", "sign out")):
                        if self._audio_intel and hasattr(self._audio_intel, 'set_enabled'):
                            self._audio_intel.set_enabled(True)
                        self._awaiting_choice = True
                        await self._router.update_state(EnrollmentState.CHOICE)
                        await self._send_choice_message(direction)
                        return
                except Exception:
                    pass
                # Normal conversation: allow downstream
                await self.push_frame(frame, direction)
                return
        
        # Forward all other frames
        await self.push_frame(frame, direction)
    
    async def _send_choice_message(self, direction: FrameDirection):
        """
        Ask the user to choose anonymous chat vs quick enrollment.
        Used when user logs out during conversation (not for initial prompt).
        """
        # NOTE: Mic is automatically muted during CHOICE state by mic_gate_filter in factory.py
        logger.info("[EnrollmentCoordinator] Sending choice prompt (logout scenario)")
        text = (
            "Would you like to sign up, sign in, "
            "or chat anonymously without storing anything? Say 'sign up', 'recognize me', or 'anonymous'."
        )
        await self.push_frame(TextFrame(text), direction)
        await self.push_frame(TextFrame("So..."), direction)

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
        
        # CRITICAL FIX: Remove sentence-ending punctuation to prevent Pipecat from splitting
        # Even with aggregate_sentences=False, Pipecat may split on ! ? .
        # Replace with commas to keep the flow natural without triggering splits
        import re
        intro_clean = re.sub(r'([!?])\s+', r', ', intro_text)  # ! ? → ,
        intro_clean = re.sub(r'(\.)(\s+[A-Z])', r',\2', intro_clean)  # . before capital → ,

        # Optionally combine intro + instruction into a single TTS pass to avoid engine boundary splitting
        combine = os.getenv("ENROLLMENT_COMBINE_INTRO", "true").lower() in ("1", "true", "yes")
        if combine and self._fixed_phrase:
            instruct = f"Please repeat exactly: '{self._fixed_phrase}' three times."
            instruct_clean = re.sub(r'([!?])\s+', r', ', instruct)
            instruct_clean = re.sub(r'(\.)(\s+[A-Z])', r',\2', instruct_clean)
            combined = f"{intro_clean} {instruct_clean}"
            await self.push_frame(TextFrame(combined), direction)
        else:
            await self.push_frame(TextFrame(intro_clean), direction)
            # If using fixed phrase, instruct the user
            if self._fixed_phrase:
                instruct = f"Please repeat exactly: '{self._fixed_phrase}' three times."
                instruct_clean = re.sub(r'([!?])\s+', r', ', instruct)
                instruct_clean = re.sub(r'(\.)(\s+[A-Z])', r',\2', instruct_clean)
                await self.push_frame(TextFrame(instruct_clean), direction)
        
        self._intro_sent = True

        # No need for artificial delay - intro message will play naturally through TTS pipeline
        # and enrollment will start when user speaks after intro finishes
    
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

        # Don't transition to ENROLLING immediately - let intro message finish speaking
        # The transition will happen naturally when the next EnrollmentProgressFrame arrives
        # and is processed by _handle_enrollment_progress
    
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
            # Don't transition to ENROLLING immediately or block with sleep - let intro finish
            # The state will transition naturally on the next progress frame
            return  # Skip processing this progress frame, let intro play first
        
        if not self._enrollment_started:
            return  # Still not in enrollment flow

        # If intro has been sent but we're still in INTRO state, transition to ENROLLING now
        if self._intro_sent and self._router.current_state == EnrollmentState.INTRO:
            logger.info("[EnrollmentCoordinator] Transitioning INTRO → ENROLLING (intro complete, user speaking)")
            await self._router.update_state(EnrollmentState.ENROLLING, progress=0)

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
            # Pause audio intelligence during recognized sessions (privacy/perf)
            try:
                if self._audio_intel and hasattr(self._audio_intel, 'set_enabled'):
                    self._audio_intel.set_enabled(False)
            except Exception:
                pass
            return

        # Recognized existing unnamed profile → capture a friendly name
        await self._router.update_state(
            EnrollmentState.NAME_CAPTURE,
            speaker_id=frame.speaker_id,
            consistency=frame.confidence
        )
        self._name_capture_pending = True
        self._pending_speaker_id = frame.speaker_id

        # CRITICAL FIX: Pause audio intelligence during name capture to prevent state confusion
        # Without this, audio intelligence continues collecting enrollment samples
        try:
            if self._audio_intel and hasattr(self._audio_intel, 'set_enabled'):
                self._audio_intel.set_enabled(False)
        except Exception:
            pass

        # CRITICAL FIX: Suppress the next transcription frame to prevent immediate validation
        # The utterance that triggered recognition is still in the pipeline and would
        # be incorrectly interpreted as a name attempt, causing "didn't catch valid name" error
        try:
            loop = asyncio.get_event_loop()
            self._suppress_next_transcription = True
            self._suppress_deadline_ts = loop.time() + 1.0  # Short window
            logger.debug("[EnrollmentCoordinator] Suppressing next transcription (post-recognition for name capture)")
        except Exception:
            pass

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
        # Pause audio intelligence after successful enrollment
        try:
            if self._audio_intel and hasattr(self._audio_intel, 'set_enabled'):
                self._audio_intel.set_enabled(False)
        except Exception:
            pass
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
        # Suppress the just-finished utterance: drop the next transcription frame
        try:
            loop = asyncio.get_event_loop()
            self._suppress_next_transcription = True
            self._suppress_deadline_ts = loop.time() + 1.0  # short window
            logger.debug("[EnrollmentCoordinator] Suppressing next transcription (post-recognition)")
        except Exception:
            pass

    # NOTE: A duplicate _handle_speaker_changed was previously defined later in this
    # file, which unconditionally transitioned to CONVERSATION and bypassed the
    # onboarding gating. That method has been removed to preserve the intended
    # flow defined above (auto-enroll -> NAME_CAPTURE; returning user -> CONVERSATION
    # with audio intelligence paused and next transcription suppressed).

    async def _handle_transcription(self, frame: TranscriptionFrame, direction: FrameDirection):
        """Handle user transcriptions during choice and name capture flows."""
        text = (frame.text or "").strip()
        if not text:
            return

        logger.info(f"[EnrollmentCoordinator] Processing transcription in state {self._router.current_state.value}: '{text[:50]}...'")

        # CRITICAL: Drop suppressed transcriptions (e.g., post-recognition stale frames)
        if self._suppress_next_transcription:
            try:
                now = asyncio.get_event_loop().time()
                if now <= self._suppress_deadline_ts:
                    self._suppress_next_transcription = False
                    logger.debug(f"[EnrollmentCoordinator] Dropped suppressed transcription: '{text}'")
                    return
            except Exception:
                pass
            self._suppress_next_transcription = False

        # Handle initial choice
        if self._enable_ephemeral_choice and self._awaiting_choice and self._router.current_state == EnrollmentState.CHOICE:
            normalized = text.lower()
            if self._contains_any(normalized, self._sign_me_up_terms):
                self._awaiting_choice = False
                await self._router.update_state(EnrollmentState.INTRO)
                await self._send_intro_message(direction)
                self._enrollment_started = True
                # CRITICAL FIX: Don't transition to ENROLLING immediately - let intro message finish speaking.
                # The transition to ENROLLING will happen naturally when:
                # 1. User speaks after intro → _handle_unknown_speaker_detected triggers
                # 2. Or when first EnrollmentProgressFrame arrives → _handle_enrollment_progress triggers
                # Immediate transition was causing intro TTS to be interrupted/stopped prematurely.
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

                # Clear context history and remove Context Guide in anonymous mode
                try:
                    if self._context_aggregator and hasattr(self._context_aggregator, 'set_anonymous_mode'):
                        self._context_aggregator.set_anonymous_mode(True)
                        logger.info("[EnrollmentCoordinator] Enabled anonymous mode in context aggregator")
                except Exception as e:
                    logger.warning(f"[EnrollmentCoordinator] Failed to set anonymous mode in context: {e}")

                # Disable audio intelligence in anonymous mode (privacy)
                try:
                    if self._audio_intel and hasattr(self._audio_intel, 'set_enabled'):
                        self._audio_intel.set_enabled(False)
                        logger.info("[EnrollmentCoordinator] Disabled audio intelligence for anonymous mode")
                except Exception as e:
                    logger.warning(f"[EnrollmentCoordinator] Failed to disable audio intelligence: {e}")

                await self.push_frame(TextFrame("Okay, let's chat anonymously. Nothing will be stored."), direction)
                await self._router.update_state(EnrollmentState.CONVERSATION)
                # Send neutral greeting to start the convo without personal data
                await self.push_frame(TextFrame("Hello, how can I help you today?"), direction)
                return
            if self._contains_any(normalized, self._sign_in_terms):
                # Enter returning-user fast path
                self._awaiting_choice = False
                self._sign_in_requested = True
                await self.push_frame(TextFrame("Okay — say a few words and I'll sign you in."), direction)
                # Start a soft timeout in case recognition doesn't trigger
                if not self._sign_in_timeout_task:
                    self._sign_in_timeout_task = asyncio.create_task(self._sign_in_timeout(direction))
                return

            # If no keywords matched during CHOICE, consume transcription without forwarding
            # The prompt is clear about expected responses, so just wait for valid keywords
            logger.debug(f"[EnrollmentCoordinator] No CHOICE keywords matched, consuming: '{text[:50]}'")

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
        """Robust term detection with word-boundary matching for single words.

        - Single-word terms must match as whole words (\bterm\b)
        - Multi-word phrases fall back to substring matching
        """
        for t in terms:
            if not t:
                continue
            if " " in t:
                # Phrase: simple substring
                if t in text:
                    return True
            else:
                # Single word: enforce word boundaries
                if re.search(rf"\b{re.escape(t)}\b", text):
                    return True
        return False

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
