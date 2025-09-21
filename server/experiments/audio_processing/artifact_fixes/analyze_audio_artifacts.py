#!/usr/bin/env python3
"""
Advanced audio analysis to detect specific artifacts at sentence endings.
This tool analyzes the generated WAV files to identify the weird sound issue.
"""

import wave
import numpy as np
from pathlib import Path
import sys

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_wav_file(filepath: str) -> tuple[np.ndarray, int]:
    """Load a WAV file and return audio data and sample rate."""
    with wave.open(filepath, 'rb') as wav_file:
        frames = wav_file.readframes(-1)
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()

        # Convert to numpy array
        if wav_file.getsampwidth() == 2:  # 16-bit
            audio_data = np.frombuffer(frames, dtype=np.int16)
        elif wav_file.getsampwidth() == 4:  # 32-bit
            audio_data = np.frombuffer(frames, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {wav_file.getsampwidth()}")

        # Handle stereo by taking left channel
        if channels == 2:
            audio_data = audio_data[::2]

        return audio_data, sample_rate


def detect_silence_and_artifacts(audio_data: np.ndarray, sample_rate: int,
                                 silence_threshold: float = 0.01,
                                 min_silence_duration: float = 0.05) -> dict:
    """Detect silence periods and potential artifacts."""

    # Normalize audio to 0-1 range
    audio_normalized = np.abs(audio_data.astype(float)) / 32768.0

    # Calculate RMS in sliding windows
    window_size = int(0.01 * sample_rate)  # 10ms windows
    hop_size = window_size // 2

    rms_values = []
    time_stamps = []

    for i in range(0, len(audio_normalized) - window_size, hop_size):
        window = audio_normalized[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        rms_values.append(rms)
        time_stamps.append(i / sample_rate)

    rms_values = np.array(rms_values)
    time_stamps = np.array(time_stamps)

    # Find silence periods
    silence_mask = rms_values < silence_threshold

    # Find the end of the audio (last non-silent region)
    last_non_silent_idx = np.where(~silence_mask)[0]
    if len(last_non_silent_idx) > 0:
        audio_end_time = time_stamps[last_non_silent_idx[-1]]
        audio_end_sample = int(audio_end_time * sample_rate)
    else:
        audio_end_time = len(audio_data) / sample_rate
        audio_end_sample = len(audio_data)

    # Analyze the ending region (last 500ms or 10% of audio, whichever is smaller)
    ending_duration = min(0.5, len(audio_data) / sample_rate * 0.1)
    ending_start_sample = max(0, audio_end_sample - int(ending_duration * sample_rate))
    ending_audio = audio_data[ending_start_sample:audio_end_sample + int(0.1 * sample_rate)]  # Add 100ms buffer

    # Detect sudden changes (potential clicks/pops/artifacts)
    if len(ending_audio) > 1:
        audio_diff = np.abs(np.diff(ending_audio.astype(float)))

        # Threshold for detecting sudden changes (3x standard deviation)
        change_threshold = np.mean(audio_diff) + 3 * np.std(audio_diff)
        artifact_indices = np.where(audio_diff > change_threshold)[0]

        # Convert to time stamps
        artifact_times = (ending_start_sample + artifact_indices) / sample_rate
    else:
        artifact_times = []
        change_threshold = 0

    # Check for DC offset or bias
    dc_offset = np.mean(audio_data.astype(float))

    # Check for clipping
    max_value = np.max(np.abs(audio_data))
    clipping_ratio = np.sum(np.abs(audio_data) >= 32767) / len(audio_data)

    return {
        'rms_values': rms_values,
        'time_stamps': time_stamps,
        'audio_end_time': audio_end_time,
        'ending_start_time': ending_start_sample / sample_rate,
        'ending_audio': ending_audio,
        'artifact_times': artifact_times,
        'change_threshold': change_threshold,
        'dc_offset': dc_offset,
        'max_amplitude': max_value,
        'clipping_ratio': clipping_ratio,
        'total_duration': len(audio_data) / sample_rate
    }


def analyze_sentence_endings(audio_files: list[str]):
    """Analyze multiple audio files for sentence ending artifacts."""

    print("🔍 ADVANCED AUDIO ARTIFACT ANALYSIS")
    print("=" * 60)

    results = {}

    for filepath in audio_files:
        if not Path(filepath).exists():
            print(f"❌ File not found: {filepath}")
            continue

        print(f"\n📁 Analyzing: {Path(filepath).name}")

        try:
            audio_data, sample_rate = load_wav_file(filepath)
            analysis = detect_silence_and_artifacts(audio_data, sample_rate)

            print(f"   Duration: {analysis['total_duration']:.3f}s")
            print(f"   Max amplitude: {analysis['max_amplitude']}/32767")
            print(f"   DC offset: {analysis['dc_offset']:.2f}")
            print(f"   Clipping ratio: {analysis['clipping_ratio']:.4f}")
            print(f"   Audio ends at: {analysis['audio_end_time']:.3f}s")

            # Check for artifacts at the end
            if len(analysis['artifact_times']) > 0:
                print(f"   ⚠️  ARTIFACTS DETECTED:")
                for i, artifact_time in enumerate(analysis['artifact_times']):
                    time_from_end = analysis['total_duration'] - artifact_time
                    print(f"      Artifact {i+1}: {artifact_time:.3f}s ({time_from_end:.3f}s from end)")
            else:
                print(f"   ✅ No obvious artifacts detected")

            # Analyze the ending specifically
            ending_duration = analysis['audio_end_time'] - analysis['ending_start_time']
            if ending_duration > 0:
                ending_rms = np.sqrt(np.mean((analysis['ending_audio'].astype(float) / 32768.0) ** 2))
                overall_rms = np.sqrt(np.mean((audio_data.astype(float) / 32768.0) ** 2))

                print(f"   Ending analysis:")
                print(f"     Ending duration: {ending_duration:.3f}s")
                print(f"     Ending RMS: {ending_rms:.4f}")
                print(f"     Overall RMS: {overall_rms:.4f}")
                print(f"     RMS ratio: {ending_rms/overall_rms:.2f}")

                if ending_rms > overall_rms * 1.5:
                    print(f"     ⚠️  ENDING TOO LOUD - possible artifact!")
                elif ending_rms < overall_rms * 0.1:
                    print(f"     ⚠️  ENDING TOO QUIET - possible abrupt cutoff!")

                # Check for fade-out vs abrupt ending
                last_100ms = analysis['ending_audio'][-int(0.1 * sample_rate):] if len(analysis['ending_audio']) > int(0.1 * sample_rate) else analysis['ending_audio']
                if len(last_100ms) > 10:
                    fade_gradient = np.abs(last_100ms[-1] - last_100ms[0]) / len(last_100ms)
                    print(f"     Fade gradient: {fade_gradient:.2f}")

                    if fade_gradient < 0.1:
                        print(f"     ⚠️  ABRUPT ENDING - no fade-out detected!")

            results[filepath] = analysis

        except Exception as e:
            print(f"❌ Failed to analyze {filepath}: {e}")

    return results


def generate_waveform_plots(results: dict, output_dir: str = "audio_analysis"):
    """Generate waveform plots highlighting potential issues."""

    if not HAS_MATPLOTLIB:
        print(f"\n⚠️  matplotlib not available - skipping waveform plots")
        return

    print(f"\n📊 Generating waveform plots...")

    for filepath, analysis in results.items():
        filename = Path(filepath).stem

        try:
            audio_data, sample_rate = load_wav_file(filepath)

            # Create time axis
            time_axis = np.arange(len(audio_data)) / sample_rate

            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Plot full waveform
            ax1.plot(time_axis, audio_data)
            ax1.set_title(f'Full Waveform: {filename}')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)

            # Mark the ending region
            ending_start = analysis['ending_start_time']
            ax1.axvline(x=ending_start, color='red', linestyle='--', alpha=0.7, label='Ending region start')
            ax1.axvline(x=analysis['audio_end_time'], color='orange', linestyle='--', alpha=0.7, label='Audio end')

            # Mark artifacts
            for artifact_time in analysis['artifact_times']:
                ax1.axvline(x=artifact_time, color='red', linestyle='-', alpha=0.5, label='Artifact')

            ax1.legend()

            # Plot RMS envelope
            ax2.plot(analysis['time_stamps'], analysis['rms_values'])
            ax2.set_title(f'RMS Envelope: {filename}')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('RMS')
            ax2.grid(True, alpha=0.3)

            # Mark the same regions on RMS plot
            ax2.axvline(x=ending_start, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(x=analysis['audio_end_time'], color='orange', linestyle='--', alpha=0.7)

            plt.tight_layout()

            # Save plot
            plot_path = Path(output_dir) / f"{filename}_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   ✅ Plot saved: {plot_path}")

        except Exception as e:
            print(f"   ❌ Failed to generate plot for {filepath}: {e}")


def main():
    """Main analysis function."""

    audio_dir = Path("audio_analysis")
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        print("Run test_sentence_ending_analysis.py first to generate audio files.")
        return

    # Find all WAV files
    wav_files = list(audio_dir.glob("*.wav"))

    if not wav_files:
        print(f"❌ No WAV files found in {audio_dir}")
        return

    # Analyze all files
    results = analyze_sentence_endings([str(f) for f in wav_files])

    # Generate plots
    generate_waveform_plots(results)

    # Summary
    print(f"\n📋 ANALYSIS SUMMARY:")
    print(f"   Files analyzed: {len(results)}")

    artifact_files = []
    for filepath, analysis in results.items():
        if len(analysis['artifact_times']) > 0:
            artifact_files.append(Path(filepath).name)

    if artifact_files:
        print(f"   Files with artifacts: {', '.join(artifact_files)}")
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Check the generated plots for visual confirmation")
        print(f"   2. Listen to the specific time stamps where artifacts were detected")
        print(f"   3. Compare individual chunks vs. full audio to isolate the issue")
        print(f"   4. Consider adjusting TTS parameters or text preprocessing")
    else:
        print(f"   ✅ No obvious artifacts detected in automated analysis")
        print(f"   💡 The 'weird sound' may be:")
        print(f"      - Subtle pronunciation artifacts not caught by amplitude analysis")
        print(f"      - Issues with specific phonemes or voice characteristics")
        print(f"      - Model-specific synthesis artifacts")
        print(f"      - Problems with text preprocessing or tokenization")


if __name__ == "__main__":
    main()