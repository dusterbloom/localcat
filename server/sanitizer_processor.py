"""
SanitizerProcessor: strips control tags like '/no_think' from assistant text frames
so they never surface to TTS or the user.
"""
import re
from typing import Any

from pipecat.processors.frame_processor import FrameProcessor as BaseProcessor, FrameDirection

try:
    from pipecat.frames.frames import Frame, TextFrame
except Exception:  # pragma: no cover
    # Fallback types if specific classes are not importable
    class Frame:  # type: ignore
        pass
    class TextFrame(Frame):  # type: ignore
        pass


_TAG_PATTERN = re.compile(r"\s*/?no_think\b", re.IGNORECASE)


def _sanitize_text(text: str) -> str:
    if not text:
        return text
    out = _TAG_PATTERN.sub("", text)
    # Collapse repeated whitespace produced by removals
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


class SanitizerProcessor(BaseProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # REQUIRED: call parent to set state
        await super().process_frame(frame, direction)

        # Best-effort sanitize any frame carrying visible text
        try:
            if hasattr(frame, "text"):
                txt = getattr(frame, "text", None)
                if isinstance(txt, str) and txt:
                    clean = _sanitize_text(txt)
                    if clean != txt:
                        setattr(frame, "text", clean)
        except Exception:
            pass

        # Forward frame unmodified otherwise
        await self.push_frame(frame, direction)

