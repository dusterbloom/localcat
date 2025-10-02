"""Audio intelligence module for speaker recognition, emotion, and prosody"""
from .audio_intelligence import (
    AudioIntelligenceProcessor,
    AudioIntelligenceFrame,
    SpeakerChangedFrame,
    UnknownSpeakerDetectedFrame,
    StartEnrollmentFrame,
)
from .prosody_analyzer import ProsodyAnalyzer, ProsodyFeatures
from .confidence_fusion import ConfidenceFusion, LinguisticFeatures

__all__ = [
    "AudioIntelligenceProcessor",
    "AudioIntelligenceFrame", 
    "SpeakerChangedFrame",
    "UnknownSpeakerDetectedFrame",
    "StartEnrollmentFrame",
    "ProsodyAnalyzer",
    "ProsodyFeatures",
    "ConfidenceFusion",
    "LinguisticFeatures",
]
