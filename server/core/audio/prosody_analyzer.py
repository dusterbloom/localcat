"""
Prosody Analysis for TRUE Confidence Calculation
Extracts pitch, stress, speaking rate from audio using Parselmouth (Praat)
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    logger.warning("Parselmouth not available. Install: pip install praat-parselmouth")


@dataclass
class ProsodyFeatures:
    """
    Prosody features extracted from audio
    """
    # Pitch features
    pitch_mean: float  # Hz
    pitch_std: float  # Variation
    pitch_slope: float  # Falling (-) or rising (+) in Hz/s
    
    # Energy/stress
    intensity_mean: float  # dB
    intensity_peak: float  # Peak loudness
    
    # Fluency
    speaking_rate: float  # Syllables per second (estimate)
    pause_count: int  # Number of pauses
    
    # Duration
    duration_sec: float
    
    # Derived confidence modifiers
    certainty_modifier: float  # -1.0 to +1.0
    
    def __repr__(self):
        return (
            f"ProsodyFeatures(pitch={self.pitch_mean:.1f}Hz, slope={self.pitch_slope:.1f}, "
            f"intensity={self.intensity_mean:.1f}dB, rate={self.speaking_rate:.1f}syll/s, "
            f"certainty={self.certainty_modifier:+.2f})"
        )


class ProsodyAnalyzer:
    """
    Extract prosody features from audio for confidence calculation.
    
    Based on research:
    - Falling pitch = certainty (statements)
    - Rising pitch = uncertainty (questions)
    - High intensity = emphasis/importance
    - Fast fluent speech = confidence
    - Many pauses = uncertainty
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        pitch_floor: float = 75.0,  # Hz (typical male: 75, female: 100)
        pitch_ceiling: float = 500.0,  # Hz
    ):
        if not PARSELMOUTH_AVAILABLE:
            raise ImportError("Parselmouth required. Install: pip install praat-parselmouth")
        
        self.sample_rate = sample_rate
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
    
    def extract(self, audio: np.ndarray) -> Optional[ProsodyFeatures]:
        """
        Extract prosody features from audio array.
        
        Args:
            audio: Audio samples (float32, -1 to 1)
        
        Returns:
            ProsodyFeatures or None if extraction fails
        """
        try:
            # Create Parselmouth Sound object
            sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
            duration = sound.duration
            
            if duration < 0.3:  # Too short for reliable prosody
                return None
            
            # Extract pitch (F0)
            pitch = sound.to_pitch(
                time_step=0.01,
                pitch_floor=self.pitch_floor,
                pitch_ceiling=self.pitch_ceiling
            )
            
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames
            
            if len(pitch_values) < 5:  # Too few voiced frames
                pitch_mean = 0.0
                pitch_std = 0.0
                pitch_slope = 0.0
            else:
                pitch_mean = float(np.mean(pitch_values))
                pitch_std = float(np.std(pitch_values))
                
                # Calculate pitch slope (linear regression)
                time_points = np.arange(len(pitch_values))
                pitch_slope = float(np.polyfit(time_points, pitch_values, 1)[0])
                # Normalize by duration
                pitch_slope = pitch_slope * len(pitch_values) / duration
            
            # Extract intensity (loudness)
            intensity = sound.to_intensity()
            intensity_values = intensity.values[0]
            intensity_mean = float(np.mean(intensity_values))
            intensity_peak = float(np.max(intensity_values))
            
            # Estimate speaking rate (very rough - syllables per second)
            # Use intensity peaks as proxy for syllables
            intensity_threshold = intensity_mean + 3  # dB above mean
            peaks = intensity_values > intensity_threshold
            # Count transitions as syllable boundaries
            syllable_estimate = max(1, np.sum(np.diff(peaks.astype(int)) > 0))
            speaking_rate = syllable_estimate / duration
            
            # Count pauses (low intensity periods > 150ms)
            pause_threshold = intensity_mean - 10  # dB below mean
            low_intensity = intensity_values < pause_threshold
            # Find consecutive low intensity regions
            pause_regions = np.split(np.where(low_intensity)[0], np.where(np.diff(np.where(low_intensity)[0]) > 1)[0] + 1)
            # Count pauses longer than 150ms (15 frames at 10ms step)
            pause_count = sum(1 for region in pause_regions if len(region) > 15)
            
            # Calculate certainty modifier based on prosody
            certainty = self._calculate_certainty(
                pitch_slope=pitch_slope,
                speaking_rate=speaking_rate,
                pause_count=pause_count,
                intensity_peak=intensity_peak,
                duration=duration
            )
            
            return ProsodyFeatures(
                pitch_mean=pitch_mean,
                pitch_std=pitch_std,
                pitch_slope=pitch_slope,
                intensity_mean=intensity_mean,
                intensity_peak=intensity_peak,
                speaking_rate=speaking_rate,
                pause_count=pause_count,
                duration_sec=duration,
                certainty_modifier=certainty
            )
        
        except Exception as e:
            logger.warning(f"[Prosody] Extraction failed: {e}")
            return None
    
    def _calculate_certainty(
        self,
        pitch_slope: float,
        speaking_rate: float,
        pause_count: int,
        intensity_peak: float,
        duration: float
    ) -> float:
        """
        Calculate certainty modifier from prosody features.
        
        Returns: -0.3 to +0.3 modifier to add to base confidence
        """
        certainty = 0.0
        
        # Pitch slope: Falling = certain, Rising = uncertain
        if pitch_slope < -10:  # Falling (statement)
            certainty += 0.15
        elif pitch_slope > 10:  # Rising (question)
            certainty -= 0.20
        
        # Speaking rate: Fast fluent = confident, Slow = uncertain
        if speaking_rate > 4.5:  # Fast and fluent
            certainty += 0.10
        elif speaking_rate < 2.5:  # Slow/hesitant
            certainty -= 0.15
        
        # Pauses: Few = confident, Many = uncertain
        pauses_per_sec = pause_count / duration if duration > 0 else 0
        if pauses_per_sec < 0.3:  # < 1 pause per 3 seconds
            certainty += 0.05
        elif pauses_per_sec > 1.0:  # > 1 pause per second
            certainty -= 0.10
        
        # Intensity: High peak = emphatic/certain
        if intensity_peak > 70:  # Loud/emphatic
            certainty += 0.05
        
        # Clamp to reasonable range
        return max(-0.3, min(0.3, certainty))
