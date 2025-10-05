"""
Example: Privacy-First Speaker Name Handler

This shows how to handle UnknownSpeakerDetectedFrame and StartEnrollmentFrame
in your bot pipeline to create a consent-based speaker enrollment flow.
"""

import re
from pipecat.frames.frames import Frame, SystemFrame, TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import FrameDirection
from loguru import logger

from .audio_intelligence import UnknownSpeakerDetectedFrame, StartEnrollmentFrame


class SpeakerNameManager(FrameProcessor):
    """
    Privacy-First speaker name manager
    
    Handles:
    1. UnknownSpeakerDetectedFrame → Ask for name
    2. User response → Extract name → Emit StartEnrollmentFrame
    3. Pass through to enrollment
    """
    
    def __init__(self):
        super().__init__()
        self._awaiting_name_response = False
        self._unknown_speaker_hash: str = None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for privacy-first speaker enrollment"""
        
        # 1. Unknown speaker detected → Ask for name
        if isinstance(frame, UnknownSpeakerDetectedFrame):
            logger.info("[SpeakerNameManager] Unknown speaker detected, requesting name")
            self._awaiting_name_response = True
            self._unknown_speaker_hash = frame.embedding_hash
            
            # Inject prompt into LLM context
            prompt_frame = TextFrame(
                text="SYSTEM: An unrecognized speaker has been detected. "
                     "Please politely ask: 'I don't recognize you, may I know your name?'"
            )
            await self.push_frame(prompt_frame, FrameDirection.DOWNSTREAM)
            
            # Also pass through original frame
            return await self.push_frame(frame, direction)
        
        # 2. User responds with name
        if self._awaiting_name_response and isinstance(frame, TranscriptionFrame):
            name = self._extract_name_from_text(frame.text)
            
            if name:
                logger.info(f"[SpeakerNameManager] Extracted name: {name}")
                self._awaiting_name_response = False
                
                # Emit enrollment consent frame
                consent_frame = StartEnrollmentFrame(speaker_name=name)
                await self.push_frame(consent_frame, FrameDirection.DOWNSTREAM)
                
                # Acknowledge to user
                ack_frame = TextFrame(
                    text=f"SYSTEM: User identified as {name}. Enrollment starting."
                )
                await self.push_frame(ack_frame, FrameDirection.DOWNSTREAM)
        
        # Pass through all frames
        return await self.push_frame(frame, direction)
    
    def _extract_name_from_text(self, text: str) -> str:
        """Extract name from user response"""
        patterns = [
            r"my name is (\w+)",
            r"i'?m (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"this is (\w+)",
            r"(\w+) here",
            r"it'?s (\w+)",
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).capitalize()
                # Ignore common false positives
                if name.lower() not in {"my", "the", "a", "an", "is", "am", "here"}:
                    return name
        
        return None


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def create_privacy_first_pipeline(factory):
    """
    Example of integrating privacy-first speaker enrollment into bot pipeline
    
    Pipeline order:
    WebRTC Input → AudioIntelligence → SpeakerNameManager → Memory → LLM → TTS
    """
    
    # Create processors
    audio_intel = factory.create_audio_intelligence_processor()  # Privacy-aware
    name_manager = SpeakerNameManager()  # Handles consent flow
    memory = factory.create_memory_service()
    llm = factory.create_llm_service()
    tts = factory.create_tts_service()
    
    # Link pipeline
    pipeline = Pipeline([
        input_transport,  # WebRTC input
        audio_intel,      # Detects unknown speakers → emits UnknownSpeakerDetectedFrame
        name_manager,     # Asks for name → emits StartEnrollmentFrame
        memory,           # Stores facts with speaker_id
        llm,              # Generates responses
        tts,              # Speaks responses
        output_transport  # WebRTC output
    ])
    
    return pipeline


# ============================================================================
# CONFIGURATION
# ============================================================================

"""
In .env:

# Enable privacy-first mode
AUDIO_INTEL_PRIVACY_MODE=consent_pending  # or 'ephemeral' for maximum privacy
AUDIO_INTEL_REQUIRE_CONSENT=true

# Now the flow is:
# 1. Unknown speaker talks
# 2. Bot asks: "I don't recognize you, may I know your name?"
# 3. User: "My name is Alice"
# 4. Bot: "Nice to meet you, Alice!" (enrollment starts)
# 5. After 2-3 more utterances → Alice is enrolled
# 6. Future: Bot recognizes Alice automatically
"""


# ============================================================================
# TESTING
# ============================================================================

async def test_privacy_flow():
    """Test the privacy-first enrollment flow"""
    
    # Create manager
    manager = SpeakerNameManager()
    
    # Simulate unknown speaker detected
    unknown_frame = UnknownSpeakerDetectedFrame(embedding_hash="abc123")
    await manager.process_frame(unknown_frame, FrameDirection.DOWNSTREAM)
    # Expected: Bot asks "May I know your name?"
    
    # Simulate user response
    response_frame = TranscriptionFrame(text="My name is Alice", user_id="test")
    await manager.process_frame(response_frame, FrameDirection.DOWNSTREAM)
    # Expected: StartEnrollmentFrame(speaker_name="Alice") emitted
    
    print("✅ Privacy-first flow tested successfully!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_privacy_flow())
