"""
Context building utilities for HotMem (Phase 1B)

Provides:
- Bullet formatting, deduplication, and capping
- MemoryContextFrame for downstream processors
"""

from typing import List

from pipecat.frames.frames import Frame


def format_bullets(bullets: List[str], max_bullets: int = 3) -> List[str]:
    """Deduplicate, cap, and normalize bullets.

    Keeps order, drops empties, ensures each line starts with "• ".
    """
    seen = set()
    out: List[str] = []
    cap = max(0, int(max_bullets))
    if cap == 0:
        return []
    for b in bullets:
        if not b:
            continue
        s = b.strip()
        if not s:
            continue
        if not s.startswith("• "):
            s = "• " + s
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def build_message(role: str, header: str, bullets: List[str]) -> dict:
    """Build a context message dict for the aggregator."""
    body = header.rstrip() + "\n" + "\n".join(bullets) if bullets else header.rstrip()
    return {"role": role, "content": body}


class MemoryContextFrame(Frame):
    """Frame carrying memory context ready to be inserted by downstream."""

    def __init__(self, role: str, header: str, bullets: List[str]):
        super().__init__()
        self.role = role
        self.header = header
        self.bullets = bullets

