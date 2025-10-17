#!/usr/bin/env python3
"""
Debug what's actually available in the Parakeet streaming context
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def debug_streaming_context():
    """Debug the streaming context API"""
    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT

        print("🔍 Debugging Parakeet streaming context API...")

        stt = ParakeetStreamingSTT(
            enable_vad=False,
            context_size=(256, 256),
            depth=3
        )

        if stt._streaming_context:
            print(f"✅ Streaming context available: {type(stt._streaming_context)}")
            print(f"📋 Available attributes:")

            attrs = [attr for attr in dir(stt._streaming_context) if not attr.startswith('_')]
            for attr in attrs:
                try:
                    value = getattr(stt._streaming_context, attr)
                    print(f"   {attr}: {type(value)} = {value}")
                except Exception as e:
                    print(f"   {attr}: Error accessing - {e}")

            # Test with some audio
            import numpy as np
            import mlx.core as mx

            print(f"\n🧪 Testing with sample audio...")
            sample_audio = np.random.randn(16000).astype(np.float32) * 0.1  # 1 second of noise
            audio_mx = mx.array(sample_audio)

            stt._streaming_context.add_audio(audio_mx)

            print(f"📊 After adding audio:")

            # Check result property
            result = getattr(stt._streaming_context, 'result', None)
            if result:
                print(f"   result: {result}")
                if hasattr(result, 'text'):
                    print(f"   result.text: '{result.text}'")
                else:
                    print(f"   result (no .text attr): {type(result)}")
            else:
                print(f"   result: NOT FOUND")

            # Check tokens
            if hasattr(stt._streaming_context, 'finalized_tokens'):
                print(f"   finalized_tokens: {stt._streaming_context.finalized_tokens}")
            else:
                print(f"   finalized_tokens: NOT FOUND")

            if hasattr(stt._streaming_context, 'draft_tokens'):
                print(f"   draft_tokens: {stt._streaming_context.draft_tokens}")
            else:
                print(f"   draft_tokens: NOT FOUND")

            # Check if there's a decode method
            if hasattr(stt._streaming_context, 'decode'):
                print(f"   has decode method: YES")
            else:
                print(f"   has decode method: NO")

        else:
            print("❌ No streaming context available")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_streaming_context()