"""Frames for context compaction pipeline."""

from pipecat.frames.frames import DataFrame
from typing import List, Dict, Any


class ContextCompactionFrame(DataFrame):
    """Carries a compacted summary to replace old messages in context."""

    summary: str
    """The compacted summary text."""

    messages_replaced: int
    """Number of messages that were summarized."""

    cutoff_index: int
    """Index in original message list up to which messages were compacted."""
