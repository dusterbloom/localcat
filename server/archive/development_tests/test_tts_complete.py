#!/usr/bin/env python3
"""
Comprehensive TTS stress test harness to debug sentence completion issues.
Tests various response patterns and tracks completion rates.
"""

import asyncio
import sys
import os
import time
import json
from typing import List, Dict, Any
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from tts_mlx_ultra_low_latency import TTSMLXUltraLowLatency
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from tools.text_formatter import sanitize_for_kokoro
from text_chunker import chunk_text_ultra_low_latency, estimate_tokens


class TTSTestResult:
    """Track TTS test results and metrics."""

    def __init__(self, test_name: str, input_text: str):
        self.test_name = test_name
        self.input_text = input_text
        self.sanitized_text = ""
        self.expected_sentences = 0
        self.processed_sentences = 0
        self.audio_chunks = 0
        self.total_audio_bytes = 0
        self.ttfb_ms = None
        self.total_duration_ms = None
        self.success = False
        self.error = None
        self.sentence_details = []
        # Token-based metrics
        self.estimated_tokens = 0
        self.token_chunks = 0
        self.avg_chunk_time_ms = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "input_length": len(self.input_text),
            "sanitized_length": len(self.sanitized_text),
            "estimated_tokens": self.estimated_tokens,
            "token_chunks": self.token_chunks,
            "expected_sentences": self.expected_sentences,
            "processed_sentences": self.processed_sentences,
            "completion_rate": self.processed_sentences / max(1, self.expected_sentences),
            "audio_chunks": self.audio_chunks,
            "total_audio_bytes": self.total_audio_bytes,
            "ttfb_ms": self.ttfb_ms,
            "total_duration_ms": self.total_duration_ms,
            "avg_chunk_time_ms": self.avg_chunk_time_ms,
            "success": self.success,
            "error": self.error,
            "sentence_details": self.sentence_details
        }


class TTSStressTester:
    """Comprehensive TTS testing framework."""

    def __init__(self):
        self.tts_service = None
        self.results = []

        # Test cases that simulate real LLM responses, including complex ones that cause PC heating
        self.test_cases = [
            {
                "name": "2_sentence_simple",
                "text": "Hello there! How are you doing today?"
            },
            {
                "name": "3_sentence_normal",
                "text": "That's a great question! I can help you with that. Let me know what you need."
            },
            {
                "name": "4_sentence_complex",
                "text": "That's an interesting question! As your assistant, I can access information to help you. I can still look things up for you. Would you like me to do that?"
            },
            {
                "name": "5_sentence_long",
                "text": "Good morning! I hope you're having a wonderful day so far. There are many things we could discuss today. What would you like to focus on first? I'm here to help with whatever you needed."
            },
            {
                "name": "token_chunk_test_medium",
                "text": "This response contains exactly the right amount of content to test our token-based chunking algorithm effectively. We need to verify that the system properly breaks down text into optimal 175-250 token chunks while maintaining natural speech flow and prosody. The chunking should happen seamlessly without causing any performance degradation or stuttering issues that previously affected the system."
            },
            {
                "name": "token_chunk_test_large",
                "text": "This is a comprehensive test of the ultra-low latency TTS system that implements token-based chunking according to Kokoro FastAPI best practices, specifically targeting 175 to 250 tokens per chunk with a maximum absolute limit of 450 tokens to prevent performance degradation and ensure consistent 40 to 80 millisecond time-to-first-byte latency. The system should intelligently break down this longer response into multiple token-optimized chunks, processing each chunk separately through the Kokoro MLX worker process while maintaining smooth audio playback and preventing the PC heating issues that occurred with large text blocks. Each chunk should stream audio immediately upon generation rather than waiting for the entire response to complete, thereby achieving the ultra-low latency streaming behavior that makes conversational AI feel natural and responsive."
            },
            {
                "name": "complex_technical_explanation",
                "text": "The implementation of token-based text chunking in Kokoro TTS represents a significant advancement in ultra-low latency speech synthesis. By pre-processing text into optimal token ranges of 175-250 tokens before sending to the MLX worker process, we can achieve consistent time-to-first-byte performance while preventing the system overload that occurs when processing large text blocks as single units. This approach leverages the natural language processing capabilities of the tokenization algorithm to identify semantic boundaries within the text, ensuring that chunk breaks occur at linguistically appropriate points rather than arbitrary character limits. The resulting audio maintains natural prosody and intonation while delivering the responsiveness required for real-time conversational applications. Furthermore, this chunking strategy aligns with Apple Silicon hardware optimization patterns, maximizing the efficiency of MLX framework operations on M-series processors."
            },
            {
                "name": "problematic_quotes",
                "text": 'I can help with "complex queries" and technical questions. This should work smoothly now. The "internet" word caused issues before.'
            },
            {
                "name": "mixed_punctuation",
                "text": "Question marks work? Exclamations too! Normal periods. What about (parenthetical expressions) and some — dashes?"
            },
            {
                "name": "conversational_flow",
                "text": "That's a clever thought! It seems we're both connected in a rather unique way. I exist within your computer's processing power. This creates an interesting digital relationship. What do you think about that?"
            }
        ]

    async def setup_tts(self):
        """Initialize TTS service for testing with token-based chunking."""
        self.tts_service = TTSMLXUltraLowLatency(
            model="mlx-community/Kokoro-82M-bf16",
            voice="af_heart",
            use_boundaries=False,  # Disable boundaries for token-based chunking
            buffer_ms=50  # Ultra-low latency target
        )
        return await self.tts_service._initialize_if_needed()

    def count_sentences(self, text: str) -> int:
        """Count expected sentences in text."""
        import re
        # Count sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return len([s for s in sentences if s.strip()])

    async def test_single_case(self, test_case: Dict[str, str]) -> TTSTestResult:
        """Test a single TTS case and collect detailed metrics."""
        result = TTSTestResult(test_case["name"], test_case["text"])

        try:
            print(f"\\n🧪 Testing: {test_case['name']}")
            print(f"Input: {test_case['text']}")

            # Track sanitization and token metrics
            result.sanitized_text = sanitize_for_kokoro(test_case["text"], max_sentence_length=200)
            result.expected_sentences = self.count_sentences(result.sanitized_text)
            result.estimated_tokens = estimate_tokens(result.sanitized_text)

            # Show token chunking preview
            token_chunks = chunk_text_ultra_low_latency(result.sanitized_text)
            result.token_chunks = len(token_chunks)

            print(f"Sanitized: {result.sanitized_text}")
            print(f"Expected sentences: {result.expected_sentences}")
            print(f"Estimated tokens: {result.estimated_tokens}")
            print(f"Token chunks: {result.token_chunks}")
            for i, chunk in enumerate(token_chunks):
                print(f"  Chunk {i+1}: '{chunk}' (~{estimate_tokens(chunk)} tokens)")

            start_time = time.time()
            frames_received = []

            # Collect all TTS frames
            async for frame in self.tts_service.run_tts(test_case["text"]):
                frames_received.append(frame)

                if isinstance(frame, TTSStartedFrame):
                    print("🎤 TTS Started")
                elif isinstance(frame, TTSAudioRawFrame):
                    result.audio_chunks += 1
                    result.total_audio_bytes += len(frame.audio)
                    if result.ttfb_ms is None:
                        result.ttfb_ms = (time.time() - start_time) * 1000
                    print(f"🔊 Audio chunk {result.audio_chunks}: {len(frame.audio)} bytes")
                elif isinstance(frame, TTSStoppedFrame):
                    print("⏹️  TTS Stopped")

            result.total_duration_ms = (time.time() - start_time) * 1000
            result.success = result.audio_chunks > 0
            result.avg_chunk_time_ms = result.total_duration_ms / max(1, result.token_chunks)

            # Estimate processed sentences based on audio chunks
            # This is a rough estimate since we don't have direct sentence tracking
            result.processed_sentences = min(result.audio_chunks, result.expected_sentences)

            print(f"✅ Completed: {result.audio_chunks} chunks, {result.total_audio_bytes} bytes")
            print(f"⏱️  TTFB: {result.ttfb_ms:.1f}ms, Total: {result.total_duration_ms:.1f}ms")
            print(f"📊 Avg time per token chunk: {result.avg_chunk_time_ms:.1f}ms")

        except Exception as e:
            result.error = str(e)
            result.success = False
            print(f"❌ Error: {e}")

        return result

    async def run_stress_test(self):
        """Run comprehensive stress test suite."""
        print("🚀 Starting TTS Stress Test Suite")
        print("=" * 50)

        # Setup TTS
        if not await self.setup_tts():
            print("❌ Failed to initialize TTS service")
            return

        print("✅ TTS service initialized")

        # Run all test cases
        for test_case in self.test_cases:
            result = await self.test_single_case(test_case)
            self.results.append(result)

            # Brief pause between tests
            await asyncio.sleep(0.5)

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive test report."""
        print("\\n" + "=" * 60)
        print("📊 TTS STRESS TEST REPORT")
        print("=" * 60)

        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)

        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")

        print("\\n📈 DETAILED RESULTS:")
        print("-" * 60)

        for result in self.results:
            data = result.to_dict()
            status = "✅" if result.success else "❌"
            completion = data["completion_rate"] * 100

            print(f"{status} {data['test_name']}:")
            print(f"    Tokens: {data['estimated_tokens']} (~{data['token_chunks']} chunks)")
            print(f"    Completion: {completion:.1f}% ({data['processed_sentences']}/{data['expected_sentences']} sentences)")
            print(f"    Audio: {data['audio_chunks']} chunks, {data['total_audio_bytes']} bytes")
            if data['ttfb_ms']:
                print(f"    Timing: TTFB {data['ttfb_ms']:.1f}ms, Total {data['total_duration_ms']:.1f}ms")
                if data['avg_chunk_time_ms']:
                    print(f"    Per chunk: {data['avg_chunk_time_ms']:.1f}ms avg")
            if data['error']:
                print(f"    Error: {data['error']}")
            print()

        # Identify issues
        incomplete_tests = [r for r in self.results if r.success and r.processed_sentences < r.expected_sentences]
        if incomplete_tests:
            print("⚠️  INCOMPLETE SENTENCE PROCESSING:")
            for result in incomplete_tests:
                print(f"    {result.test_name}: {result.processed_sentences}/{result.expected_sentences} sentences")

        # Save detailed results to JSON
        self.save_results_json()

    def save_results_json(self):
        """Save detailed results to JSON file."""
        output_file = "tts_stress_test_results.json"
        results_data = {
            "timestamp": time.time(),
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results if r.success),
                "avg_completion_rate": sum(r.processed_sentences/max(1, r.expected_sentences) for r in self.results) / len(self.results)
            },
            "results": [r.to_dict() for r in self.results]
        }

        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"📄 Detailed results saved to: {output_file}")


async def main():
    """Run the TTS stress test."""
    tester = TTSStressTester()
    await tester.run_stress_test()


if __name__ == "__main__":
    asyncio.run(main())