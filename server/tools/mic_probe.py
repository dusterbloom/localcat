"""
MicProbe: Pipecat processor that logs microphone input characteristics.

- Logs input AudioRawFrame sample rate, channels, frame size
- Computes peak and RMS amplitude to detect silence or clipping
- Throttles logs to avoid flooding; configurable via env

Enable in bot.py by setting ENABLE_MIC_PROBE=true
"""

import os
import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, StartFrame, AudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor as BaseProcessor, FrameDirection


class MicProbe(BaseProcessor):
    def __init__(self, log_every: int | None = None, warmup_logs: int | None = None):
        super().__init__()
        self._count = 0
        self._log_every = log_every or int(os.getenv("MIC_PROBE_LOG_EVERY", "20"))
        self._warmup_logs = warmup_logs or int(os.getenv("MIC_PROBE_WARMUP_LOGS", "10"))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, AudioRawFrame):
            self._count += 1
            # Interpret PCM16 bytes to float32 in [-1, 1]
            try:
                pcm = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
                peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
                rms = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
                level_db = (20.0 * np.log10(rms + 1e-12))
            except Exception as e:
                logger.warning(f"[MicProbe] Failed to parse audio bytes: {e}")
                peak = 0.0
                rms = 0.0
                level_db = -120.0

            sr = getattr(frame, "sample_rate", None)
            ch = getattr(frame, "num_channels", None)
            nbytes = len(frame.audio) if hasattr(frame, "audio") and frame.audio else 0

            # Throttle logging
            should_log = (self._count <= self._warmup_logs) or (self._count % self._log_every == 0)
            if should_log:
                logger.info(
                    f"[MicProbe] frame#{self._count} sr={sr} ch={ch} bytes={nbytes} peak={peak:.4f} rms={rms:.4f} ({level_db:.1f} dBFS)"
                )

            await self.push_frame(frame, direction)
            return

        # Non-audio frames: pass through
        await self.push_frame(frame, direction)

