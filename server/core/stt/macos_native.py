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
import time
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
        self._process_healthy = False
        self._audio_buffer = bytearray()  # Buffer audio during utterance
        self._is_speaking = False  # Track if user is currently speaking
        logger.info(f"🎙️  MacOSNativeSTT initialized: lang={language}, rate={sample_rate}, on_device={on_device}")

    async def _start_sidecar(self) -> bool:
        if self._proc is not None and self._process_healthy:
            return True

        # Reset process state for fresh start
        self._process_healthy = False

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
            logger.warning("macos-stt sidecar binary not found")
            return False

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

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            logger.debug(f"Spawned macOS STT sidecar: PID {self._proc.pid}")
            self._process_healthy = True

            # Fixed: Authorization no longer hangs with proper implementation
            # Only monitor for actual process crashes
            async def _monitor_process_health():
                while self._process_healthy and self._proc:
                    await asyncio.sleep(1.0)
                    if self._proc.returncode is not None:
                        logger.debug(f"macOS STT process ended with code: {self._proc.returncode}")
                        self._process_healthy = False
                        break

            asyncio.create_task(_monitor_process_health())

            async def _reader():
                assert self._proc and self._proc.stdout
                try:
                    first_output = True
                    while True:
                        line = await self._proc.stdout.readline()
                        if not line:
                            if first_output:
                                logger.debug("macos-stt: Process ended without output (authorization issue)")
                            else:
                                logger.debug("macos-stt: stdout closed, process ended")
                            break

                        first_output = False
                        try:
                            obj = json.loads(line.decode("utf-8", errors="ignore"))
                            await self._queue.put(obj)
                        except Exception:
                            logger.debug(f"macos-stt: non-JSON line: {line[:80]!r}")
                except Exception as e:
                    logger.error(f"macos-stt reader error: {e}")
                finally:
                    self._process_healthy = False

            self._reader_task = asyncio.create_task(_reader())
            return True

        except Exception as e:
            logger.error(f"Failed to start macOS STT sidecar: {e}")
            self._process_healthy = False
            return False

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

    async def _process_complete_utterance(self, audio: bytes):
        """Process a complete utterance by spawning a fresh sidecar subprocess."""
        try:
            logger.info(f"🚀 Processing complete utterance: {len(audio)} bytes")

            # Spawn fresh sidecar for this utterance
            started = await self._start_sidecar()
            if not started:
                logger.warning("Failed to start sidecar for utterance")
                return

            # Resample audio if needed
            in_rate = int((__import__('os').getenv('STT_INPUT_SAMPLE_RATE') or self._sample_rate))
            if in_rate != self._sample_rate:
                pcm = memoryview(audio)
                audio = self._resample_int16(pcm, in_rate, self._sample_rate)

            # Write all audio and close stdin to trigger endAudio()
            logger.info(f"📤 Sending {len(audio)} bytes to sidecar and closing stdin")
            self._proc.stdin.write(audio)
            await self._proc.stdin.drain()
            self._proc.stdin.close()  # This triggers endAudio() in Swift → isFinal=true

            # Wait for final result with timeout
            timeout = 5.0
            start_time = time.time()
            last_interim_text = ""

            while time.time() - start_time < timeout:
                try:
                    obj = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                    text = (obj.get("text") or "").strip()
                    if not text:
                        continue

                    is_final = obj.get("final") is True or obj.get("type") == "finalized"

                    # Log interim results but don't yield them
                    if not is_final:
                        last_interim_text = text
                        logger.debug(f"⏳ Interim: '{text}'")
                        continue

                    # Only yield final results - one time!
                    logger.info(f"✅ Final transcription: '{text}' (interim was: '{last_interim_text}')")
                    timestamp = str(int(time.time() * 1000))
                    user_id = getattr(self, 'user_id', 'default-user')
                    final_frame = TranscriptionFrame(text=text, user_id=user_id, timestamp=timestamp)
                    await self.push_frame(final_frame)
                    break  # Stop after first final result

                except asyncio.TimeoutError:
                    continue

            # Cleanup subprocess
            await self.cleanup()

        except Exception as e:
            logger.error(f"Error processing utterance: {e}")

    async def process_frame(self, frame: Frame, direction):
        """Handle VAD events to trigger per-utterance finalization."""
        from pipecat.frames.frames import (
            AudioRawFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame
        )

        # Detect start of speech - reset buffer
        if isinstance(frame, UserStartedSpeakingFrame):
            logger.info("🎤 User started speaking - preparing new utterance")
            self._is_speaking = True
            self._audio_buffer.clear()
            self._queue = asyncio.Queue()  # Fresh queue for this utterance

        # Detect end of speech - finalize the utterance
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info(f"🛑 User stopped speaking - finalizing utterance ({len(self._audio_buffer)} bytes buffered)")
            self._is_speaking = False

            # Process the complete utterance
            if len(self._audio_buffer) > 0:
                await self._process_complete_utterance(bytes(self._audio_buffer))
                self._audio_buffer.clear()

        # Buffer audio while user is speaking
        elif isinstance(frame, AudioRawFrame) and self._is_speaking:
            self._audio_buffer.extend(frame.audio)
            logger.debug(f"📦 Buffered {len(frame.audio)} bytes (total: {len(self._audio_buffer)})")

        await super().process_frame(frame, direction)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        DISABLED: We handle everything in process_frame() using VAD events.

        This prevents the base STTService from processing audio chunks,
        which would cause duplicate transcriptions.
        """
        # Return immediately without yielding anything
        # All processing happens in process_frame() -> _process_complete_utterance()
        return
        yield  # Make this a generator (unreachable but required for type signature)

    async def cleanup(self):
        try:
            # Cancel reader task first
            if self._reader_task and not self._reader_task.done():
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except asyncio.CancelledError:
                    pass

            # Terminate sidecar process gracefully
            if self._proc:
                try:
                    self._proc.terminate()
                    # Give process a chance to clean up gracefully
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("Force killing macos-stt sidecar (timeout)")
                    self._proc.kill()
                    await self._proc.wait()
                except Exception as e:
                    logger.debug(f"Error terminating macos-stt sidecar: {e}")

        except Exception as e:
            logger.debug(f"Error during MacOSNativeSTT cleanup: {e}")

        await super().cleanup()
