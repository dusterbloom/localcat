"""Jina reranker (v3) via HuggingFace Transformers.

Provides pairwise query→text scoring using the official pipeline
(pair tokenization + score head). Returns entailment probabilities
in [0, 1] suitable for use as a reranking signal.

Environment variables:
  - MEMORY_RERANK_JINA_ENABLED (bool)
  - MEMORY_RERANK_JINA_MODEL (default: jinaai/jina-reranker-v3-large)
  - MEMORY_RERANK_JINA_MAXLEN (default: 512)
  - MEMORY_RERANK_JINA_MAX_CANDIDATES (default: 24)
  - MEMORY_RERANK_JINA_TRUST_REMOTE_CODE (default: true)
  - TRANSFORMERS_OFFLINE (honored if set)
"""

from __future__ import annotations

import os
from typing import List, Optional
from loguru import logger


class JinaReranker:
    def __init__(self):
        self._enabled: Optional[bool] = None
        self._model_name: Optional[str] = None
        self._maxlen: int = 512
        self._max_candidates: int = 24
        self._trust_remote: bool = True
        self._tokenizer = None
        self._model = None
        self._warned = False

    def _check_enabled(self) -> bool:
        if self._enabled is None:
            self._enabled = os.getenv("MEMORY_RERANK_JINA_ENABLED", "false").lower() in ("1", "true", "yes")
            if self._enabled:
                self._model_name = os.getenv("MEMORY_RERANK_JINA_MODEL", "jinaai/jina-reranker-v3-large")
                try:
                    self._maxlen = int(os.getenv("MEMORY_RERANK_JINA_MAXLEN", "512"))
                except Exception:
                    self._maxlen = 512
                try:
                    self._max_candidates = int(os.getenv("MEMORY_RERANK_JINA_MAX_CANDIDATES", "24"))
                except Exception:
                    self._max_candidates = 24
                self._trust_remote = os.getenv("MEMORY_RERANK_JINA_TRUST_REMOTE_CODE", "true").lower() in ("1", "true", "yes")
        return bool(self._enabled)

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            local_only = os.getenv("TRANSFORMERS_OFFLINE", "").strip() != ""
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                local_files_only=local_only,
                trust_remote_code=self._trust_remote,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name,
                local_files_only=local_only,
                trust_remote_code=self._trust_remote,
            )
            logger.info(f"[JinaReranker] Loaded model: {self._model_name}")
        except Exception as e:
            if not self._warned:
                logger.warning(f"[JinaReranker] Failed to load {self._model_name}: {e}")
                self._warned = True
            self._enabled = False

    def score(self, query: str, texts: List[str]) -> List[float]:
        """Return entailment probabilities for (query, text) pairs.

        Returns zeros if disabled or unavailable.
        """
        if not self._check_enabled() or not texts:
            return [0.0] * len(texts)
        # Limit candidates
        if len(texts) > self._max_candidates:
            texts = texts[: self._max_candidates]
        try:
            import torch
            self._load_model()
            if self._model is None or self._tokenizer is None:
                return [0.0] * len(texts)
            enc = self._tokenizer(
                [query] * len(texts),
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self._maxlen,
                padding=True,
            )
            with torch.no_grad():
                out = self._model(**enc)
            scores_attr = getattr(out, "scores", None)
            if scores_attr is not None:
                s = torch.as_tensor(scores_attr).view(-1).sigmoid().cpu().tolist()
                # Pad if we clipped texts
                return s + [0.0] * (len(texts) - len(s))
            logits = getattr(out, "logits", None)
            if logits is None:
                return [0.0] * len(texts)
            if logits.ndim == 1:
                s = logits.sigmoid().cpu().tolist()
                return s + [0.0] * (len(texts) - len(s))
            if logits.shape[-1] == 1:
                s = logits.squeeze(-1).sigmoid().cpu().tolist()
                return s + [0.0] * (len(texts) - len(s))
            # 3-class: map to entail index 2 by convention
            probs = logits.softmax(dim=-1).cpu().tolist()
            res = []
            for p in probs:
                p_ent = float(p[2]) if len(p) > 2 else 0.0
                res.append(p_ent)
            return res + [0.0] * (len(texts) - len(res))
        except Exception as e:
            logger.debug(f"[JinaReranker] Inference failed: {e}")
            return [0.0] * len(texts)


_jina_reranker_instance: Optional[JinaReranker] = None


def get_jina_reranker() -> JinaReranker:
    global _jina_reranker_instance
    if _jina_reranker_instance is None:
        _jina_reranker_instance = JinaReranker()
    return _jina_reranker_instance

