"""
Professional audio processing utilities for TTS services.
Provides artifact-free audio conversion with proper fade handling.
"""

import numpy as np
from typing import Tuple, Optional
from loguru import logger


class AudioProcessor:
    """
    High-quality audio processing for TTS output.

    Handles:
    - Safe float-to-int16 conversion with headroom
    - Automatic fade-out for clean sentence endings
    - DC offset removal
    - Clipping protection with smart limiting
    - Buffer boundary artifact prevention
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        fade_duration_ms: float = 50.0,
        target_peak_db: float = -3.0,
        dc_filter_cutoff: float = 20.0
    ):
        """
        Initialize audio processor with professional defaults.

        Args:
            sample_rate: Audio sample rate in Hz
            fade_duration_ms: Fade-out duration in milliseconds
            target_peak_db: Target peak level in dB (prevents clipping)
            dc_filter_cutoff: High-pass filter cutoff for DC removal (Hz)
        """
        self.sample_rate = sample_rate
        self.fade_samples = int(fade_duration_ms * sample_rate / 1000)
        self.target_peak_linear = 10 ** (target_peak_db / 20.0)  # -3dB = 0.707
        self.dc_filter_cutoff = dc_filter_cutoff

        logger.debug(f"AudioProcessor initialized: {fade_duration_ms}ms fade, {target_peak_db}dB peak")

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset using a simple high-pass filter.

        Args:
            audio: Input audio array (float)

        Returns:
            Audio with DC offset removed
        """
        if len(audio) < 2:
            return audio

        # Simple high-pass filter approximation
        # For real-time processing, this is more efficient than scipy filters
        dc_mean = np.mean(audio)
        return audio - dc_mean

    def apply_smart_limiter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply smart limiting to prevent clipping while preserving dynamics.

        Args:
            audio: Input audio array (float, normalized to ±1.0)

        Returns:
            Limited audio with preserved dynamics
        """
        if len(audio) == 0:
            return audio

        peak = np.max(np.abs(audio))

        if peak <= self.target_peak_linear:
            return audio  # No limiting needed

        # Calculate gain reduction needed
        gain_reduction = self.target_peak_linear / peak

        # Apply smooth limiting (soft knee)
        limited = audio * gain_reduction

        logger.debug(f"Applied limiting: {peak:.3f} → {np.max(np.abs(limited)):.3f} (gain: {gain_reduction:.3f})")

        return limited

    def apply_fade_out(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply smooth fade-out to prevent abrupt endings.

        Args:
            audio: Input audio array

        Returns:
            Audio with fade-out applied
        """
        if len(audio) <= self.fade_samples:
            # For very short audio, apply shorter fade
            fade_samples = max(1, len(audio) // 4)
        else:
            fade_samples = self.fade_samples

        if fade_samples <= 1:
            return audio

        # Create smooth fade curve (cosine fade for natural sound)
        fade_curve = np.cos(np.linspace(0, np.pi/2, fade_samples)) ** 2

        # Apply fade to the end of the audio
        audio_faded = audio.copy()
        audio_faded[-fade_samples:] *= fade_curve

        return audio_faded

    def process_tts_audio(
        self,
        raw_audio: np.ndarray,
        apply_fade: bool = True,
        remove_dc: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Complete professional audio processing chain for TTS output.

        Args:
            raw_audio: Raw audio from TTS model (float, any range)
            apply_fade: Whether to apply fade-out
            remove_dc: Whether to remove DC offset

        Returns:
            Tuple of (processed_int16_audio, processing_stats)
        """
        if len(raw_audio) == 0:
            return np.array([], dtype=np.int16), {"error": "empty_audio"}

        processing_stats = {
            "input_samples": len(raw_audio),
            "input_peak": float(np.max(np.abs(raw_audio))),
            "input_rms": float(np.sqrt(np.mean(raw_audio ** 2))),
        }

        # Step 1: Ensure audio is in float format and normalized range
        if raw_audio.dtype != np.float32 and raw_audio.dtype != np.float64:
            # Assume it's already int16 and convert to float
            audio_float = raw_audio.astype(np.float32) / 32768.0
        else:
            audio_float = raw_audio.astype(np.float32)

        # Step 2: Remove DC offset if requested
        if remove_dc:
            audio_float = self.remove_dc_offset(audio_float)
            processing_stats["dc_removed"] = True

        # Step 3: Apply smart limiting to prevent clipping
        audio_float = self.apply_smart_limiter(audio_float)
        processing_stats["peak_after_limiting"] = float(np.max(np.abs(audio_float)))

        # Step 4: Apply fade-out if requested
        if apply_fade:
            audio_float = self.apply_fade_out(audio_float)
            processing_stats["fade_applied"] = True
            processing_stats["fade_samples"] = self.fade_samples

        # Step 5: Convert to int16 with proper scaling
        # Use target peak instead of full scale to maintain headroom
        audio_int16 = np.clip(
            audio_float * 32767.0,
            -32768,
            32767
        ).astype(np.int16)

        processing_stats.update({
            "output_samples": len(audio_int16),
            "output_peak": int(np.max(np.abs(audio_int16))),
            "output_rms": float(np.sqrt(np.mean((audio_int16.astype(np.float32) / 32768.0) ** 2))),
            "final_sample": int(audio_int16[-1]) if len(audio_int16) > 0 else 0,
        })

        return audio_int16, processing_stats

    def validate_audio_quality(self, audio_int16: np.ndarray) -> dict:
        """
        Validate processed audio quality and detect potential issues.

        Args:
            audio_int16: Processed audio in int16 format

        Returns:
            Quality metrics and warnings
        """
        if len(audio_int16) == 0:
            return {"status": "error", "message": "empty_audio"}

        # Calculate quality metrics
        peak_value = np.max(np.abs(audio_int16))
        rms_value = np.sqrt(np.mean((audio_int16.astype(np.float32) / 32768.0) ** 2))
        final_sample = abs(audio_int16[-1])

        # Check for common issues
        warnings = []

        if peak_value >= 32767:
            warnings.append("clipping_detected")

        if final_sample > 100:
            warnings.append("abrupt_ending")

        if rms_value < 0.01:
            warnings.append("very_quiet")
        elif rms_value > 0.5:
            warnings.append("very_loud")

        # Check for ending artifacts (sudden changes in last 100 samples)
        if len(audio_int16) > 100:
            ending_samples = audio_int16[-100:]
            ending_diffs = np.abs(np.diff(ending_samples.astype(np.float32)))
            max_ending_jump = np.max(ending_diffs)

            if max_ending_jump > 1000:
                warnings.append("ending_artifacts")

        return {
            "status": "clean" if not warnings else "warnings",
            "peak_value": int(peak_value),
            "rms_value": float(rms_value),
            "final_sample": int(final_sample),
            "warnings": warnings,
            "duration_seconds": len(audio_int16) / self.sample_rate
        }


class TTSAudioFrame:
    """
    Enhanced audio frame with processing metadata.
    Compatible with Pipecat TTSAudioRawFrame but with additional quality info.
    """

    def __init__(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        num_channels: int = 1,
        processing_stats: Optional[dict] = None,
        quality_metrics: Optional[dict] = None
    ):
        self.audio = audio_bytes
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.processing_stats = processing_stats or {}
        self.quality_metrics = quality_metrics or {}

    def to_pipecat_frame(self):
        """Convert to standard Pipecat TTSAudioRawFrame."""
        from pipecat.frames.frames import TTSAudioRawFrame
        return TTSAudioRawFrame(
            audio=self.audio,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels
        )

    def get_duration(self) -> float:
        """Get audio duration in seconds."""
        sample_count = len(self.audio) // (2 * self.num_channels)  # 2 bytes per int16 sample
        return sample_count / self.sample_rate

    def log_quality_summary(self):
        """Log a concise quality summary."""
        if self.quality_metrics and self.processing_stats:
            status = self.quality_metrics.get("status", "unknown")
            peak = self.quality_metrics.get("peak_value", 0)
            final = self.quality_metrics.get("final_sample", 0)
            warnings = len(self.quality_metrics.get("warnings", []))

            logger.debug(
                f"🎵 Audio processed: {status} | peak:{peak} final:{final} "
                f"warnings:{warnings} | {self.get_duration():.2f}s"
            )


# Convenience function for easy integration
def create_clean_audio_frame(
    raw_audio: np.ndarray,
    sample_rate: int = 24000,
    apply_fade: bool = True,
    log_quality: bool = True
) -> TTSAudioFrame:
    """
    One-shot function to create a clean, artifact-free audio frame.

    Args:
        raw_audio: Raw audio from TTS model
        sample_rate: Audio sample rate
        apply_fade: Whether to apply fade-out
        log_quality: Whether to log quality metrics

    Returns:
        Clean TTSAudioFrame ready for streaming
    """
    processor = AudioProcessor(sample_rate=sample_rate)

    # Process audio
    clean_audio, stats = processor.process_tts_audio(
        raw_audio,
        apply_fade=apply_fade
    )

    # Validate quality
    quality = processor.validate_audio_quality(clean_audio)

    # Create frame
    frame = TTSAudioFrame(
        audio_bytes=clean_audio.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        processing_stats=stats,
        quality_metrics=quality
    )

    if log_quality:
        frame.log_quality_summary()

    return frame