import asyncio
import pytest


class DummyRouter:
    def __init__(self):
        from core.audio.enrollment_state import EnrollmentState
        self._state = EnrollmentState.CHOICE
        self._progress = 0

    @property
    def current_state(self):
        return self._state

    async def update_state(self, new_state, **kwargs):
        self._state = new_state
        self._progress = kwargs.get("progress", self._progress)


class DummyMemory:
    def __init__(self):
        self.ephemeral = False
        self.identity = None

    def set_ephemeral_mode(self, enabled: bool):
        self.ephemeral = bool(enabled)

    def set_user_identity(self, user_id: str):
        self.identity = user_id


class DummyAudioIntel:
    def __init__(self):
        self.enabled = True

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)


class FrameSink:
    def __init__(self):
        self.frames = []

    async def push(self, frame):
        self.frames.append(frame)

    def clear(self):
        self.frames.clear()


@pytest.mark.asyncio
async def test_auto_enroll_enters_name_capture_and_blocks_transcriptions(monkeypatch):
    from core.audio.enrollment_coordinator import EnrollmentCoordinator
    from core.audio.audio_intelligence import SpeakerChangedFrame
    from core.audio.enrollment_state import EnrollmentState
    from pipecat.frames.frames import TranscriptionFrame

    router = DummyRouter()
    mem = DummyMemory()
    ai = DummyAudioIntel()
    coord = EnrollmentCoordinator(router=router, memory=mem, audio_intel=ai)

    # Capture all frames pushed downstream
    sink = FrameSink()

    async def _push(frame, direction):
        await sink.push(frame)

    coord.push_frame = _push  # type: ignore

    # Simulate auto-enroll completion
    sc = SpeakerChangedFrame(
        speaker_id="Speaker_1", speaker_name=None, confidence=0.85, auto_enrolled=True
    )
    await coord.process_frame(sc, None)

    # Coordinator should transition to NAME_CAPTURE (onboarding continues)
    assert router.current_state.name == "NAME_CAPTURE"

    # Clear any frames produced during completion prompts
    sink.clear()

    # During onboarding, transcriptions must NOT be forwarded downstream
    await coord.process_frame(
        TranscriptionFrame(text="A quick brown fox jumped over a lazy dog."), None
    )

    # Ensure no TranscriptionFrame was pushed downstream; only internal TextFrames are allowed
    assert not any(
        f.__class__.__name__.endswith("TranscriptionFrame") for f in sink.frames
    ), "Enrollment transcriptions should be swallowed during onboarding"


@pytest.mark.asyncio
async def test_returning_user_disables_audio_intel_and_suppresses_next_transcription(monkeypatch):
    from core.audio.enrollment_coordinator import EnrollmentCoordinator
    from core.audio.audio_intelligence import SpeakerChangedFrame
    from pipecat.frames.frames import TranscriptionFrame

    router = DummyRouter()
    mem = DummyMemory()
    ai = DummyAudioIntel()
    coord = EnrollmentCoordinator(router=router, memory=mem, audio_intel=ai)

    # Capture frames
    sink = FrameSink()

    async def _push(frame, direction):
        await sink.push(frame)

    coord.push_frame = _push  # type: ignore

    # Simulate recognized returning user (has a known name)
    sc = SpeakerChangedFrame(
        speaker_id="Speaker_1", speaker_name="Peppy", confidence=0.92, auto_enrolled=False
    )
    await coord.process_frame(sc, None)

    # Audio intelligence should be disabled for recognized sessions
    assert ai.enabled is False

    # Clear any welcome messages before testing suppression
    sink.clear()

    # Immediately after recognition, the next transcription should be dropped
    await coord.process_frame(TranscriptionFrame(text="hello"), None)

    # No frames should be pushed for the suppressed transcription
    assert len(sink.frames) == 0, "First post-recognition transcription should be suppressed"

