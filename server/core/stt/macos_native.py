"""
macOS Native STT using SFSpeechRecognizer via Swift sidecar.

Design:
- Spawn sidecar `macos-stt` which reads raw PCM (16k mono, int16) from stdin
  and emits JSON lines with partial/final transcriptions to stdout.
- This service writes audio bytes to the process as they arrive and
  forwards transcripts as Pipecat frames.

The sidecar is bundled under: app/src-tauri/sidecar/macos-stt/macos-stt
"""

import asyncio
import json
from typing import AsyncGenerator, Optional
from pathlib import Path
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    ErrorFrame,
)
from pipecat.services.ai_services import STTService


class MacOSNativeSTT(STTService):
    def __init__(self, language: str = "en-US", sample_rate: int = 16000, on_device: bool = True):
        super().__init__()
        self._language = language
        self._sample_rate = sample_rate
        self._on_device = on_device
        self._proc: Optional[asyncio.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue()

    async def _start_sidecar(self):
        if self._proc is not None:
            return

        # Resolve binary path (Tauri Resources or dev path)
        candidates = []
        # TAURI_RESOURCE_DIR if provided
        import os as _os
        trd = _os.getenv("TAURI_RESOURCE_DIR")
        if trd:
            candidates.append(Path(trd) / "sidecar" / "macos-stt" / "macos-stt")
        # Dev and alternate production paths
        candidates.extend([
            Path(__file__).resolve().parents[3] / "app" / "src-tauri" / "sidecar" / "macos-stt" / "macos-stt",
            Path(__file__).resolve().parents[2] / "_up_" / "macos-stt" / "macos-stt",
        ])

        binary_path = None
        for c in candidates:
            if c.exists():
                binary_path = str(c)
                break
        if not binary_path:
            raise FileNotFoundError("macos-stt sidecar not found")

        cmd = [
            binary_path,
            "--stdin-pcm",
            "--rate",
            str(self._sample_rate),
            "--lang",
            self._language,
        ]
        if self._on_device:
            cmd.append("--on-device")

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.debug(f"Spawned macOS STT sidecar: PID {self._proc.pid}")

        async def _reader():
            assert self._proc and self._proc.stdout
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line.decode("utf-8", errors="ignore"))
                    await self._queue.put(obj)
                except Exception:
                    logger.debug(f"macos-stt: non-JSON line: {line[:80]!r}")

        self._reader_task = asyncio.create_task(_reader())

    def _resample_int16(self, pcm: memoryview, from_rate: int, to_rate: int) -> bytes:
        if from_rate == to_rate:
            return pcm.tobytes()
        try:
            import numpy as np
        except ImportError:
            # Fallback: naive drop/dup (not ideal but avoids crash)
            if from_rate > to_rate:
                step = from_rate // to_rate
                return pcm.tobytes()[:: step * 2]
            else:
                # duplicate samples
                raw = pcm.tobytes()
                return b"".join(raw[i:i+2] * (to_rate // from_rate) for i in range(0, len(raw), 2))
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        ratio = float(to_rate) / float(from_rate)
        out_len = int(arr.shape[0] * ratio)
        if out_len <= 1:
            return pcm.tobytes()
        x_old = np.linspace(0, 1, arr.shape[0], endpoint=False)
        x_new = np.linspace(0, 1, out_len, endpoint=False)
        out = np.interp(x_new, x_old, arr)
        out = np.clip(out, -32768, 32767).astype(np.int16)
        return out.tobytes()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            await self._start_sidecar()
            assert self._proc and self._proc.stdin

            # Optional resample from input rate to sidecar rate
            in_rate = int((__import__('os').getenv('STT_INPUT_SAMPLE_RATE') or self._sample_rate))
            if in_rate != self._sample_rate:
                pcm = memoryview(audio)
                audio = self._resample_int16(pcm, in_rate, self._sample_rate)

            # Write raw PCM bytes to stdin of sidecar (int16 mono)
            self._proc.stdin.write(audio)
            await self._proc.stdin.drain()

            # Drain any available transcripts without blocking the hot loop
            for _ in range(4):
                try:
                    obj = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                text = (obj.get("text") or "").strip()
                if not text:
                    continue
                if obj.get("final") is True:
                    yield TranscriptionFrame(text=text)
                else:
                    yield InterimTranscriptionFrame(text=text)

        except Exception as e:
            logger.error(f"macOS STT error: {e}")
            yield ErrorFrame(error=str(e))

    async def cleanup(self):
        try:
            if self._proc:
                self._proc.terminate()
        except Exception:
            pass
        await super().cleanup()
