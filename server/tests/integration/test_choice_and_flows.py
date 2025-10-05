import asyncio
import pytest

# Minimal stubs to exercise EnrollmentCoordinator choice logic without full Pipecat

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
        self.names = {}

    def set_speaker_name(self, sid: str, name: str):
        self.names[sid] = name

class Sink:
    def __init__(self):
        self.texts = []

    async def push(self, text):
        self.texts.append(text)


@pytest.mark.asyncio
async def test_anonymous_choice_swallow_and_ephemeral(monkeypatch):
    from core.audio.enrollment_coordinator import EnrollmentCoordinator
    from core.audio.enrollment_state import EnrollmentState
    from pipecat.frames.frames import TranscriptionFrame

    router = DummyRouter()
    mem = DummyMemory()
    ai = DummyAudioIntel()
    coord = EnrollmentCoordinator(router=router, memory=mem, audio_intel=ai)

    # Patch push_frame to capture outputs
    sink = Sink()
    async def _push(frame, direction):
        from pipecat.frames.frames import TextFrame
        if isinstance(frame, TextFrame):
            await sink.push(frame.text)
    coord.push_frame = _push  # type: ignore

    # Coordinator starts in CHOICE and awaits choice
    frame = TranscriptionFrame(text="anonymous")
    await coord.process_frame(frame, None)

    assert router.current_state.name == "CONVERSATION"
    assert mem.ephemeral is True
    # We should have pushed a neutral greeting
    assert any("Hello" in t for t in sink.texts)


@pytest.mark.asyncio
async def test_sign_me_up_flow_enters_enrolling(monkeypatch):
    from core.audio.enrollment_coordinator import EnrollmentCoordinator
    from core.audio.enrollment_state import EnrollmentState
    from pipecat.frames.frames import TranscriptionFrame

    router = DummyRouter()
    mem = DummyMemory()
    ai = DummyAudioIntel()
    coord = EnrollmentCoordinator(router=router, memory=mem, audio_intel=ai)

    async def _nop(*args, **kwargs):
        return None
    coord.push_frame = _nop  # type: ignore

    frame = TranscriptionFrame(text="sign me up")
    await coord.process_frame(frame, None)
    # After intro
    assert router.current_state.name in ("INTRO", "ENROLLING")


@pytest.mark.asyncio
async def test_name_capture_validation_and_confirm(monkeypatch):
    from core.audio.enrollment_coordinator import EnrollmentCoordinator
    from core.audio.enrollment_state import EnrollmentState
    from core.audio.audio_intelligence import SpeakerChangedFrame
    from pipecat.frames.frames import TranscriptionFrame

    router = DummyRouter()
    mem = DummyMemory()
    ai = DummyAudioIntel()
    coord = EnrollmentCoordinator(router=router, memory=mem, audio_intel=ai)

    async def _nop(*args, **kwargs):
        return None
    coord.push_frame = _nop  # type: ignore

    # Simulate auto-enroll completion
    sc = SpeakerChangedFrame(speaker_id="Speaker_1", speaker_name=None, confidence=0.8, auto_enrolled=True)
    await coord.process_frame(sc, None)
    assert router.current_state.name == "NAME_CAPTURE"

    # Provide invalid name (too long phrase)
    await coord.process_frame(TranscriptionFrame(text="A quick brown fox jumped over a lazy dog"), None)
    # Provide valid short name and confirm
    await coord.process_frame(TranscriptionFrame(text="Ana"), None)
    await coord.process_frame(TranscriptionFrame(text="yes"), None)

    assert router.current_state.name == "CONVERSATION"
    assert mem.identity == "Ana"
