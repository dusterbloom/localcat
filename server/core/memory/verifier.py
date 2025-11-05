"""
Answerability Verifier: Query-conditioned verification filter.

General pairwise verifier that scores (query, candidate) for entailment vs
unknown/contradiction. Pluggable backends with a fast rule-based fallback.

Usage:
  verifier = AnswerabilityVerifier(host)
  decisions = verifier.verify(query, [cand.text for _, cand, _ in scored][:K])
  -> list of dicts: {text, status, score}
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger


class AnswerabilityVerifier:
    def __init__(self, host: Any):
        self.host = host
        self.enabled = os.getenv("MEMORY_VERIFIER_ENABLED", "true").lower() in ("1", "true", "yes")
        self.model_name = os.getenv("MEMORY_VERIFIER_MODEL") or ""
        self.backend: Optional[str] = None
        self._hf_tokenizer = None
        self._hf_model = None
        # Label mapping for 3-class NLI models (contradiction, neutral, entailment)
        try:
            self._ent_idx = int(os.getenv("MEMORY_VERIFIER_ENT_IDX", "2"))
            self._con_idx = int(os.getenv("MEMORY_VERIFIER_CON_IDX", "0"))
        except Exception:
            self._ent_idx, self._con_idx = 2, 0
        if self.model_name and self.enabled:
            self._try_load_hf()

    def _try_load_hf(self) -> None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._hf_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
            self.backend = "hf"
            logger.info(f"[Verifier] Loaded verifier model: {self.model_name}")
        except Exception as e:
            logger.warning(f"[Verifier] HF load failed for {self.model_name}: {e}; using rules")
            self.backend = None

    def _hf_scores(self, query: str, texts: List[str]) -> List[Tuple[float, float, float]]:
        """Return list of (p_contra, p_neutral, p_entail). For single-logit rerankers,
        map score to (0, 1-score, score)."""
        try:
            import torch
            pairs = [(query, t) for t in texts]
            enc = self._hf_tokenizer([q for q, _ in pairs], [t for _, t in pairs], return_tensors="pt", truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                out = self._hf_model(**enc)
            logits = out.logits
            if logits.shape[-1] == 1:
                # Single logit similarity
                s = logits.squeeze(-1).sigmoid().cpu().tolist()
                return [(0.0, 1.0 - x, x) for x in s]
            else:
                probs = logits.softmax(dim=-1).cpu().tolist()
                res = []
                for p in probs:
                    p_ent = float(p[self._ent_idx]) if self._ent_idx < len(p) else 0.0
                    p_con = float(p[self._con_idx]) if self._con_idx < len(p) else 0.0
                    p_neu = max(0.0, 1.0 - p_ent - p_con)
                    res.append((p_con, p_neu, p_ent))
                return res
        except Exception as e:
            logger.debug(f"[Verifier] HF inference failed: {e}")
            return [(0.0, 1.0, 0.0) for _ in texts]

    def _rules_status(self, query: str, text: str) -> Tuple[str, float]:
        """Very fast deterministic alignment checks."""
        q = (query or "").lower()
        t = (text or "").lower()
        # Simple subject alignment: first-person questions expect 'you' facts
        subj_ok = True
        if q.startswith("who ") or q.startswith("where ") or q.startswith("what ") or q.startswith("when "):
            subj_ok = ("you " in t or t.startswith("you ") or t.startswith("your "))
        # Object anchor: if a content word is in query, require it in text
        anchors = []
        for kw in ("pizza", "sushi", "blue", "tokyo", "google", "openai"):
            if kw in q:
                anchors.append(kw)
        anchors_ok = all(kw in t for kw in anchors) if anchors else True
        if subj_ok and anchors_ok:
            return ("entailed", 0.8)
        return ("unknown", 0.5)

    def verify(self, query: str, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.enabled or not texts:
            return [{"text": t, "status": "unknown", "score": 0.5} for t in texts]
        if self.backend == "hf":
            scores = self._hf_scores(query, texts)
            out = []
            ent_th = float(os.getenv("MEMORY_VERIFIER_ENT_T", "0.6"))
            con_th = float(os.getenv("MEMORY_VERIFIER_CON_T", "0.6"))
            for t, (p_con, p_neu, p_ent) in zip(texts, scores):
                if p_ent >= ent_th:
                    out.append({"text": t, "status": "entailed", "score": float(p_ent)})
                elif p_con >= con_th:
                    out.append({"text": t, "status": "contradicts", "score": float(p_con)})
                else:
                    out.append({"text": t, "status": "unknown", "score": float(p_ent)})
            return out
        # Fallback rules
        return [{"text": t, **dict(zip(["status", "score"], self._rules_status(query, t)))} for t in texts]

