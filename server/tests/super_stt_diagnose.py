#!/usr/bin/env python3
"""
Super STT diagnostic for Kyutai/Moshi streaming

Goals
- Generate a known speech waveform (via Kokoro TTS) and feed it into the KyutaiStreamingSTT model
- Exercise both MLX and Candle repos and verify audio tokenizer wiring
- Report sample-rate handling, audio amplitude, token-level stats, and final text

Usage
  server/.venv/bin/python server/tests/super_stt_diagnose.py \
    --hf-repo kyutai/stt-1b-en_fr-mlx \
    --text "hello how are you testing streaming" \
    --vad

  You can also test Candle:
    --hf-repo kyutai/stt-1b-en_fr-candle

Notes
- Requires moshi_mlx and Mimi weights to be available locally (Hugging Face cache)
- Uses Kokoro TTS via tts_mlx_isolated to generate clean 24kHz speech without mic/WebRTC
"""

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from loguru import logger

# Ensure local server/ is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(THIS_DIR)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from kyutai_streaming_stt import KyutaiStreamingSTT, rustymimi, mx, models  # type: ignore
from tts_mlx_isolated import TTSMLXIsolated


@dataclass
class DiagResult:
    tokens: List[int]
    text: str
    pad_count: int
    eos_count: int
    bos_count: int
    unk_count: int
    audio_blocks: int
    sample_rate_in: int
    sample_rate_target: int
    first_tokens: List[Tuple[int, str]]


async def generate_tts_wave(text: str) -> Tuple[np.ndarray, int]:
    """Generate TTS audio for a given text using Kokoro at 24 kHz.

    Returns (audio_f32, sample_rate)
    """
    tts = TTSMLXIsolated(model="mlx-community/Kokoro-82M-bf16", voice="af_heart", sample_rate=24000)
    audio_chunks: List[bytes] = []
    sr = 24000

    async for frame in tts.run_tts(text):
        # Only capture audio frames
        if hasattr(frame, "audio") and hasattr(frame, "sample_rate"):
            audio_chunks.append(frame.audio)
            sr = getattr(frame, "sample_rate", 24000)

    if not audio_chunks:
        raise RuntimeError("No audio chunks produced by TTS")

    audio_b = b"".join(audio_chunks)
    audio_np = np.frombuffer(audio_b, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np, sr


def _encode_audio_block(stt: KyutaiStreamingSTT, block_final: np.ndarray):
    """Encode 80ms block to Mimi tokens, handling rustymimi vs moshi_mlx Mimi."""
    if isinstance(stt._audio_tokenizer, rustymimi.Tokenizer):  # Candle
        return stt._audio_tokenizer.encode_step(block_final)
    else:
        return stt._audio_tokenizer.encode_step(mx.array(block_final))


def diagnose_with_known_audio(stt: KyutaiStreamingSTT, audio_f32: np.ndarray, source_rate: int) -> DiagResult:
    """Run a synchronous diagnostic pass using STT internals, block by block."""
    # Resample to target
    # Note: use stt._resample_audio to match implementation, but protect if signature differs
    audio_rs = stt._resample_audio(audio_f32, source_rate)  # type: ignore

    # Slice into 80ms blocks of 1920 at 24 kHz
    target_rate = getattr(stt, "_target_sample_rate", 24000)
    block_size = 1920
    total_blocks = len(audio_rs) // block_size

    tokens: List[int] = []
    text_parts: List[str] = []
    pad = stt._pad_token_id
    eos_ids = stt._eos_token_ids
    bos = stt._bos_token_id

    pad_count = eos_count = bos_count = unk_count = 0
    first_tokens: List[Tuple[int, str]] = []

    for i in range(total_blocks):
        block = audio_rs[i * block_size:(i + 1) * block_size]
        # Prepare shape: (1, 1, 1920)
        block_sd = block.reshape(-1, 1)  # (1920, 1)
        block_1_1920 = block_sd[None, :, 0]  # (1, 1920)
        block_final = block_1_1920[None, 0:1]  # (1, 1, 1920)

        # Encode tokens
        other_audio_tokens = _encode_audio_block(stt, block_final)
        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[:, :, :stt._lm_config.other_codebooks]

        # Generate one text token
        if stt._enable_vad:
            text_token, _vad = stt._gen.step_with_extra_heads(other_audio_tokens[0])
        else:
            text_token = stt._gen.step(other_audio_tokens[0])

        tt = int(text_token[0].item())
        tokens.append(tt)

        # Classify
        if tt == 0:
            unk_count += 1
        elif tt == pad:
            pad_count += 1
        elif tt in eos_ids:
            eos_count += 1
        elif tt == bos:
            bos_count += 1
        else:
            piece = stt._text_tokenizer.id_to_piece(tt).replace("▁", " ")
            text_parts.append(piece)
            if len(first_tokens) < 12:
                first_tokens.append((tt, piece))

    final_text = "".join(text_parts).strip()

    return DiagResult(
        tokens=tokens,
        text=final_text,
        pad_count=pad_count,
        eos_count=eos_count,
        bos_count=bos_count,
        unk_count=unk_count,
        audio_blocks=total_blocks,
        sample_rate_in=source_rate,
        sample_rate_target=target_rate,
        first_tokens=first_tokens,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=os.getenv("KYUTAI_STT_REPO", "kyutai/stt-1b-en_fr-mlx"))
    parser.add_argument("--text", default="hello how are you testing moshi streaming")
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS and use a test tone (not recommended)")
    args = parser.parse_args()

    logger.info(f"Repo: {args.hf_repo}, VAD: {args.vad}")

    # Prepare audio
    if not args.no_tts:
        logger.info("Generating diagnostic TTS audio with Kokoro at 24 kHz...")
        audio_f32, sr_in = await generate_tts_wave(args.text)
        logger.info(f"TTS produced {audio_f32.shape[0]} samples at {sr_in} Hz")
    else:
        logger.warning("Using synthetic tone (may not produce meaningful STT tokens)")
        sr_in = 24000
        t = np.linspace(0, 2.0, int(2.0 * sr_in), endpoint=False)
        audio_f32 = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

    logger.info(f"Input amplitude: max={np.max(np.abs(audio_f32)):.4f}, mean={np.mean(np.abs(audio_f32)):.4f}")

    # Instantiate STT
    stt = KyutaiStreamingSTT(hf_repo=args.hf_repo, enable_vad=args.vad, max_steps=4096)
    logger.info(f"Model ready. Using {'RustyMimi' if isinstance(stt._audio_tokenizer, rustymimi.Tokenizer) else 'moshi_mlx Mimi'} tokenizer")

    # Diagnose
    result = diagnose_with_known_audio(stt, audio_f32, sr_in)

    # Report
    total = len(result.tokens)
    logger.info("===== STT Diagnostic Summary =====")
    logger.info(f"Blocks processed: {result.audio_blocks}")
    logger.info(f"Sample rate in: {result.sample_rate_in} -> target: {result.sample_rate_target}")
    logger.info(f"Token counts: total={total}, PAD={result.pad_count}, UNK={result.unk_count}, BOS={result.bos_count}, EOS={result.eos_count}")
    if result.first_tokens:
        preview = ", ".join([f"{tid}:{txt!r}" for tid, txt in result.first_tokens])
        logger.info(f"First tokens: {preview}")
    else:
        logger.info("First tokens: (none)")
    logger.info(f"Final text: {result.text!r}")

    # Heuristic status
    if result.pad_count > 0.9 * total:
        logger.warning("High PAD ratio (>90%). Check tokenizer backend vs repo (-mlx vs -candle) and input audio content/level.")
    if len(result.text) == 0:
        logger.warning("No decoded text produced. Try switching hf-repo variant, or test with/without VAD.")


if __name__ == "__main__":
    asyncio.run(main())

