"""
Lightweight question type classifier with optional model backend.

Falls back to heuristics when a model isn't configured.
Types: 'slot', 'episodic', 'yesno_pref', 'general', 'meta'
"""

from __future__ import annotations

import os
from typing import Optional
from loguru import logger


class QuestionTypeClassifier:
    def __init__(self):
        self.model_name = os.getenv("MEMORY_QTYPE_MODEL") or ""
        self._backend = None
        if self.model_name:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self._tok = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self._mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
                self._backend = "hf"
            except Exception as e:
                logger.warning(f"[QType] model load failed ({self.model_name}): {e}; falling back to heuristics")
                self._backend = None

    def classify(self, text: str) -> str:
        if not text or not text.strip():
            return "general"
        if self._backend == "hf":
            try:
                t = self._tok(text, return_tensors="pt", truncation=True, max_length=128)
                out = self._mdl(**t)
                idx = int(out.logits.argmax(dim=-1).item())
                # Expect labels to be provided via env, else heuristic mapping
                labels = (os.getenv("MEMORY_QTYPE_LABELS") or "slot,episodic,yesno_pref,general,meta").split(",")
                if 0 <= idx < len(labels):
                    return labels[idx].strip() or "general"
            except Exception as e:
                logger.debug(f"[QType] model inference failed: {e}")
        # Heuristic fallback
        q = text.strip().lower()
        if any(kw in q for kw in ("what did i say", "i said", "we talked", "i mentioned", "yesterday", "last week", "on ")):
            return "episodic"
        if q.startswith("do i ") or q.startswith("do you ") or " like " in q or " prefer " in q or " love " in q:
            return "yesno_pref"
        if q.startswith("can you") or q.startswith("are you able") or q.startswith("would you"):
            return "meta"
        if any(q.startswith(w + " ") for w in ("who", "what", "where", "when", "which")):
            return "slot"
        return "general"

