#!/usr/bin/env python3
"""Quick test of updated token chunking."""

from text_chunker import chunk_text_ultra_low_latency, estimate_tokens

# Test the medium-sized text that should now be chunked
test_text = """This response contains exactly the right amount of content to test our token-based chunking algorithm effectively. We need to verify that the system properly breaks down text into optimal 175-250 token chunks while maintaining natural speech flow and prosody. The chunking should happen seamlessly without causing any performance degradation or stuttering issues that previously affected the system."""

print(f"Original text: {len(test_text)} chars, ~{estimate_tokens(test_text)} tokens")
print(f"Text: {test_text}")
print()

chunks = chunk_text_ultra_low_latency(test_text)
print(f"Chunked into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks, 1):
    tokens = estimate_tokens(chunk)
    print(f"  Chunk {i}: {tokens} tokens")
    print(f"    '{chunk}'")
    print()

# Test a longer technical text
long_text = """The implementation of token-based text chunking in Kokoro TTS represents a significant advancement in ultra-low latency speech synthesis. By pre-processing text into optimal token ranges of 175-250 tokens before sending to the MLX worker process, we can achieve consistent time-to-first-byte performance while preventing the system overload that occurs when processing large text blocks as single units. This approach leverages the natural language processing capabilities of the tokenization algorithm to identify semantic boundaries within the text, ensuring that chunk breaks occur at linguistically appropriate points rather than arbitrary character limits. The resulting audio maintains natural prosody and intonation while delivering the responsiveness required for real-time conversational applications. Furthermore, this chunking strategy aligns with Apple Silicon hardware optimization patterns, maximizing the efficiency of MLX framework operations on M-series processors."""

print(f"\nLong text: {len(long_text)} chars, ~{estimate_tokens(long_text)} tokens")
long_chunks = chunk_text_ultra_low_latency(long_text)
print(f"Chunked into {len(long_chunks)} chunks:")
for i, chunk in enumerate(long_chunks, 1):
    tokens = estimate_tokens(chunk)
    print(f"  Chunk {i}: {tokens} tokens")
    print(f"    '{chunk}'")
    print()