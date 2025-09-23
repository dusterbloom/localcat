#!/usr/bin/env python3
"""Comprehensive test for TTS stability and text processing."""

import asyncio
import time
import sys
import os

_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")
for p in (_SERVER_ROOT, _PIPECAT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.tts.kokoro_professional import ProfessionalKokoroTTSService as NativeKokoroTTSService
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame


async def test_tts_stability():
    """Test TTS with various text scenarios to check for skipping issues."""

    print("🧪 Testing TTS Stability and Text Processing")
    print("=" * 60)

    try:
        # Initialize TTS service
        tts = NativeKokoroTTSService(
            voice="af_heart",
            speed=1.0,
            sample_rate=24000
        )

        print("✅ TTS initialization successful")

        # Test cases for potential text skipping scenarios
        test_cases = [
            {
                "name": "Simple sentence",
                "text": "Hello world, this is a simple test."
            },
            {
                "name": "Multiple sentences",
                "text": "This is the first sentence. This is the second sentence. And this is the third one."
            },
            {
                "name": "Long sentence",
                "text": "This is a very long sentence that should be properly processed by the TTS system without any parts being skipped or missing during the generation process."
            },
            {
                "name": "Text with quotes",
                "text": 'He said "Hello there" and then continued with his explanation.'
            },
            {
                "name": "Technical text",
                "text": "The system uses API endpoints with authentication tokens for secure communication."
            },
            {
                "name": "Numbers and symbols",
                "text": "The temperature is 25.5°C and the pressure is 1,013 hPa."
            },
            {
                "name": "Empty text",
                "text": ""
            },
            {
                "name": "Whitespace only",
                "text": "   \n\t  "
            },
            {
                "name": "Very short text",
                "text": "Hi"
            }
        ]

        passed_tests = 0
        total_tests = len(test_cases)

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🎤 Test {i}/{total_tests}: {test_case['name']}")
            print(f"   Text: '{test_case['text']}'")

            try:
                start_time = time.time()
                frames_received = {
                    'started': False,
                    'audio': False,
                    'stopped': False
                }
                audio_bytes = 0

                frame_generator = tts.run_tts(test_case['text'])

                try:
                    async for frame in frame_generator:
                        if isinstance(frame, TTSStartedFrame):
                            frames_received['started'] = True
                        elif isinstance(frame, TTSAudioRawFrame):
                            frames_received['audio'] = True
                            audio_bytes += len(frame.audio)
                            if not frames_received.get('ttfb_logged'):
                                ttfb = (time.time() - start_time) * 1000
                                print(f"   🚀 TTFB: {ttfb:.1f}ms")
                                frames_received['ttfb_logged'] = True
                        elif isinstance(frame, TTSStoppedFrame):
                            frames_received['stopped'] = True
                except GeneratorExit:
                    pass

                # Analyze results
                if test_case['text'].strip():
                    # Non-empty text should produce audio
                    if frames_received['started'] and frames_received['audio'] and frames_received['stopped']:
                        print(f"   ✅ PASSED - Generated {audio_bytes} bytes of audio")
                        passed_tests += 1
                    else:
                        print(f"   ❌ FAILED - Missing frames: started={frames_received['started']}, audio={frames_received['audio']}, stopped={frames_received['stopped']}")
                else:
                    # Empty text should still produce start/stop frames
                    if frames_received['started'] and frames_received['stopped'] and not frames_received['audio']:
                        print(f"   ✅ PASSED - Properly handled empty text")
                        passed_tests += 1
                    else:
                        print(f"   ❌ FAILED - Incorrect empty text handling: started={frames_received['started']}, audio={frames_received['audio']}, stopped={frames_received['stopped']}")

            except Exception as e:
                print(f"   ❌ FAILED - Exception: {e}")

        print(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("✅ ALL TESTS PASSED - TTS stability looks good")
            return True
        else:
            print("❌ SOME TESTS FAILED - TTS stability issues detected")
            return False

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_tts_stability())
    sys.exit(0 if success else 1)