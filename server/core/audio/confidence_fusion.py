"""
Multi-Signal Confidence Fusion
Combines prosody, emotion, and fact type into TRUE confidence
"""

from typing import Optional
from dataclasses import dataclass
from loguru import logger

from .prosody_analyzer import ProsodyFeatures


@dataclass
class LinguisticFeatures:
    """Linguistic certainty markers from text"""
    hedge_count: int  # "maybe", "I think", "probably"
    certainty_count: int  # "definitely", "always", "never"
    is_question: bool
    certainty_score: float  # 0.0 to 1.0


class LinguisticCertaintyAnalyzer:
    """Detect certainty/uncertainty from text patterns"""
    
    HEDGE_WORDS = {
        "maybe", "perhaps", "possibly", "probably", "might", "could",
        "i think", "i believe", "i guess", "i suppose", "sort of", "kind of"
    }
    
    CERTAINTY_MARKERS = {
        "definitely", "certainly", "absolutely", "always", "never",
        "clearly", "obviously", "sure", "positive", "certain"
    }
    
    def analyze(self, text: str) -> LinguisticFeatures:
        """Extract linguistic certainty features"""
        text_lower = text.lower()
        
        hedge_count = sum(1 for h in self.HEDGE_WORDS if h in text_lower)
        certainty_count = sum(1 for c in self.CERTAINTY_MARKERS if c in text_lower)
        is_question = text.strip().endswith('?')
        
        # Calculate certainty score
        certainty = 0.5  # Neutral baseline
        certainty += certainty_count * 0.15
        certainty -= hedge_count * 0.20
        certainty -= 0.25 if is_question else 0.0
        
        certainty = max(0.0, min(1.0, certainty))
        
        return LinguisticFeatures(
            hedge_count=hedge_count,
            certainty_count=certainty_count,
            is_question=is_question,
            certainty_score=certainty
        )


class ConfidenceFusion:
    """
    Fuse multiple signals into TRUE confidence score.
    
    Weights (optimized for voice agents):
    - Base confidence (fact type): 40%
    - Prosody features: 35%
    - Linguistic certainty: 25%
    
    This replaces arbitrary hardcoded confidence (0.85-0.95) with voice-aware confidence.
    """
    
    def __init__(self):
        self.linguistic_analyzer = LinguisticCertaintyAnalyzer()
    
    def calculate(
        self,
        relation: str,
        text: str,
        prosody: Optional[ProsodyFeatures] = None,
        emotion: Optional[str] = None,
        arousal: Optional[float] = None,
    ) -> float:
        """
        Calculate TRUE confidence from multiple signals.
        
        Args:
            relation: Relation type (name, lives_in, etc)
            text: Text span containing the fact
            prosody: Prosody features from audio
            emotion: Detected emotion
            arousal: Emotional arousal (0-1)
        
        Returns:
            TRUE confidence score (0.0 to 1.0)
        """
        # Base confidence from fact type (same as before, but lower weight)
        base_conf = self._get_base_confidence(relation)
        
        # Prosody confidence (if available)
        prosody_conf = 0.5  # Neutral default
        if prosody:
            prosody_conf = self._calculate_prosody_confidence(prosody)
        
        # Linguistic confidence
        linguistic = self.linguistic_analyzer.analyze(text)
        linguistic_conf = linguistic.certainty_score
        
        # Weighted fusion
        final_conf = (
            base_conf * 0.40 +
            prosody_conf * 0.35 +
            linguistic_conf * 0.25
        )
        
        # Emotion boost: High arousal + negative emotion on correction = higher confidence
        if emotion == "angry" and arousal and arousal > 0.7:
            final_conf += 0.10  # "NO! I said ALICE!" gets boost
        
        # Clamp to valid range
        return max(0.0, min(1.0, final_conf))
    
    def _get_base_confidence(self, relation: str) -> float:
        """
        Base confidence from relation type.
        Same as memory_hotpath.py but exposed here.
        """
        if relation == "name":
            return 0.90  # Names usually permanent
        elif relation.startswith("v:"):
            return 0.70  # Verbs/actions less certain
        elif relation in {"lives_in", "works_at", "born_in"}:
            return 0.80  # Biographical facts
        elif relation in {"likes", "prefers", "enjoys"}:
            return 0.65  # Preferences change
        else:
            return 0.75  # Default
    
    def _calculate_prosody_confidence(self, prosody: ProsodyFeatures) -> float:
        """
        Calculate confidence from prosody features.
        
        Prosody features already provide certainty_modifier (-0.3 to +0.3)
        Map to confidence range (0.0 to 1.0)
        """
        # Start from neutral (0.5)
        confidence = 0.5
        
        # Apply prosody certainty modifier
        confidence += prosody.certainty_modifier
        
        # Additional adjustments based on specific features
        
        # Very short utterances are less reliable
        if prosody.duration_sec < 0.5:
            confidence -= 0.10
        
        # Very high variance in pitch = emotional/uncertain
        if prosody.pitch_std > 50:
            confidence -= 0.05
        
        return max(0.0, min(1.0, confidence))
