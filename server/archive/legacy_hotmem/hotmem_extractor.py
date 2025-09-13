"""
HotMem integration for relation extraction.

This adapter provides enhanced relation extraction capabilities for HotMem,
supporting multiple extraction strategies including Hugging Face models and
custom extraction patterns. It's designed to work with various relation
extraction models and provides fallback mechanisms when primary methods fail.

If model loading fails, the extractor becomes a no-op.
"""

from __future__ import annotations

import json
import re
import time
from typing import List, Tuple, Any, Dict, Optional
import re
from loguru import logger


_HOTMEM_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


class HotMemExtractor:
    def __init__(self, model_id: str = "relik-ie/relik-relation-extraction-small", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self.pipe = None
        self.extractor = None
        self.backend = None  # 'relik' | 'hf' | None
        self._load()

    def _load(self) -> None:
        # Return cached if available
        key = (self.model_id, (self.device or 'auto'))
        cached = _HOTMEM_CACHE.get(key)
        if cached:
            try:
                self.backend = cached.get('backend')
                self.extractor = cached.get('extractor')
                self.pipe = cached.get('pipe')
                logger.info(f"[HotMem] Using cached model: {self.model_id} ({self.backend})")
                return
            except Exception:
                pass

        # 1) Prefer native ReLiK loader if available
        try:
            from relik import Relik  # type: ignore
            try:
                # Some implementations may accept device or device_map
                if self.device and self.device.lower() != 'auto':
                    self.relik = Relik.from_pretrained(self.model_id, device=self.device)
                else:
                    self.relik = Relik.from_pretrained(self.model_id)
            except TypeError:
                # Fallback without kwargs
                self.relik = Relik.from_pretrained(self.model_id)
            self.backend = 'relik'
            logger.info(f"[ReLiK] Loaded native model: {self.model_id}")
            # Optional runtime config overrides
            import os
            try:
                top_k = int(os.getenv('HOTMEM_RELIK_TOP_K', '8'))
                if hasattr(self.relik, 'top_k'):
                    self.relik.top_k = top_k
            except Exception:
                pass
            try:
                ws = int(os.getenv('HOTMEM_RELIK_WINDOW_SIZE', '48'))
                if hasattr(self.relik, 'window_size'):
                    self.relik.window_size = ws
                wst = int(os.getenv('HOTMEM_RELIK_WINDOW_STRIDE', '32'))
                if hasattr(self.relik, 'window_stride'):
                    self.relik.window_stride = wst
            except Exception:
                pass
            _RELIK_CACHE[key] = {'backend': self.backend, 'relik': self.relik, 'pipe': None}
            return
        except Exception as e:
            logger.warning(f"[ReLiK] Native loader unavailable: {e}")
            self.relik = None

        # 2) Fallback to HF text2text pipeline with device selection
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline  # type: ignore
            try:
                import torch  # type: ignore
            except Exception:
                torch = None  # type: ignore
            tok = AutoTokenizer.from_pretrained(self.model_id)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            target = (self.device or "auto").lower()
            target_device = "cpu"
            if target == "auto":
                if torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    target_device = "mps"
                elif torch is not None and hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                    target_device = "cuda"
            elif target in {"mps", "cuda", "cpu"}:
                if target == "mps" and torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    target_device = "mps"
                elif target == "cuda" and torch is not None and torch.cuda.is_available():
                    target_device = "cuda"
                else:
                    target_device = "cpu"
            try:
                mdl.to(target_device)
                logger.info(f"[ReLiK] HF model moved to {target_device}")
            except Exception:
                pass
            self.pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok)
            self.backend = 'hf'
            logger.info(f"[ReLiK] Loaded HF text2text model: {self.model_id}")
            _RELIK_CACHE[key] = {'backend': self.backend, 'relik': None, 'pipe': self.pipe}
        except Exception as e:
            logger.warning(f"[ReLiK] Failed to load HF model ({self.model_id}): {e}")
            self.pipe = None
            self.backend = None

    @staticmethod
    def _canon_rel(rel: str) -> str:
        r = (rel or "").strip().lower().replace(" ", "_")
        # Common canonical mappings
        if r in {"born_in", "be_born_in", "was_born_in"}:
            return "born_in"
        if r in {"live_in", "lives_in", "resides_in"}:
            return "lives_in"
        if r in {"work_at", "works_at", "work_for", "works_for", "employer"}:
            return "works_at"
        if r in {"teach_at", "teaches_at"}:
            return "teach_at"
        if r in {"move_from", "moved_from"}:
            return "moved_from"
        if r in {"go_to", "went_to"}:
            return "went_to"
        if r in {"educated_at", "alumni_of"}:
            return "went_to"
        if r in {"married_to"}:
            return "married_to"
        return r

    @staticmethod
    def _parse_json(text: str) -> List[Tuple[str, str, str, float]]:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "triples" in data:
                triples = data.get("triples") or []
            elif isinstance(data, list):
                triples = data
            else:
                return []
            out = []
            for item in triples:
                if isinstance(item, dict):
                    s = str(item.get("s") or item.get("subject") or "").strip()
                    r = str(item.get("r") or item.get("relation") or "").strip()
                    d = str(item.get("d") or item.get("object") or "").strip()
                    c = float(item.get("score") or item.get("confidence") or 0.8)
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    s, r, d = str(item[0]), str(item[1]), str(item[2])
                    c = 0.8
                else:
                    continue
                if s and r and d:
                    out.append((s, r, d, c))
            return out
        except Exception:
            return []

    @staticmethod
    def _parse_inline(text: str) -> List[Tuple[str, str, str, float]]:
        # Fallback regex for patterns like (s, r, d)
        triples = []
        for m in re.finditer(r"\(([^,]+),\s*([^,]+),\s*([^\)]+)\)", text):
            s, r, d = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            if s and r and d:
                triples.append((s, r, d, 0.7))
        return triples

    def extract(self, text: str) -> List[Tuple[str, str, str, float]]:
        if not text:
            return []
        low_text = (text or '').lower()
        # Time budget (ms)
        try:
            max_ms = float(os.getenv('HOTMEM_RELIK_MAX_MS', '2000'))
        except Exception:
            max_ms = 2000.0
        t_start = time.perf_counter()
        # Native ReLiK backend
        if self.backend == 'relik' and self.relik is not None:
            try:
                # Supported call styles
                if hasattr(self.relik, 'extract'):
                    res = self.relik.extract(text)
                elif hasattr(self.relik, 'predict'):
                    res = self.relik.predict(text)
                elif callable(self.relik):
                    res = self.relik(text)
                else:
                    res = None

                if res is None:
                    return []

                # Try common container attributes on native output
                candidates = []
                for attr in ['triplets', 'triplet', 'relations', 'predictions', 'results', 'items', 'data']:
                    v = getattr(res, attr, None)
                    if isinstance(v, list) and v:
                        candidates = v
                        break

                # Fallback: dict conversion
                if not candidates:
                    dct = None
                    for m in ['as_dict', 'to_dict']:
                        fn = getattr(res, m, None)
                        if callable(fn):
                            try:
                                dct = fn()
                                break
                            except Exception:
                                pass
                    if dct is None and hasattr(res, '__dict__'):
                        dct = getattr(res, '__dict__')
                    if isinstance(dct, dict):
                        # Look for first list of dicts nested under any key
                        for k, v in dct.items():
                            if isinstance(v, list) and v and isinstance(v[0], (dict, list, tuple)):
                                candidates = v
                                break

                def _as_text(v: Any) -> str:
                    if isinstance(v, str):
                        return v
                    # ReLiK mention/span objects often expose `.text`
                    txt = getattr(v, 'text', None)
                    if isinstance(txt, str) and txt:
                        return txt
                    # If repr contains text='...'
                    try:
                        s = repr(v)
                        m = re.search(r"text='([^']+)'", s)
                        if m:
                            return m.group(1)
                    except Exception:
                        pass
                    return str(v or '')

                # Allowlist for relations (optional)
                import os
                allow_env = os.getenv('HOTMEM_RELIK_ALLOWED_RELS', '')
                allowed = set([s.strip() for s in allow_env.split(',') if s.strip()]) or {
                    'works_at','lives_in','teach_at','born_in','moved_from','went_to',
                    'also_known_as','name','age','has','owns'
                }

                def _plausible(ent: str) -> bool:
                    if not ent or ent == '--nme--':
                        return False
                    if ent.isdigit():
                        return False
                    if len(ent) > 80:
                        return False
                    # must share at least one token with text
                    toks = [w for w in ent.split() if w.isalpha() and len(w) >= 2]
                    if not toks:
                        return False
                    return any(tok in low_text for tok in toks)

                triples: List[Tuple[str, str, str, float]] = []
                for item in candidates or []:
                    if isinstance(item, dict):
                        s = _as_text(item.get('s') or item.get('subject') or item.get('head') or '')
                        r = _as_text(item.get('r') or item.get('relation') or item.get('predicate') or '')
                        d = _as_text(item.get('d') or item.get('object') or item.get('tail') or '')
                        c = float(item.get('score') or item.get('confidence') or 0.85)
                    elif isinstance(item, (list, tuple)) and len(item) >= 3:
                        s, r, d = _as_text(item[0]), _as_text(item[1]), _as_text(item[2])
                        c = 0.85
                    else:
                        # Try object attributes
                        s = _as_text(getattr(item, 'subject', None) or getattr(item, 'head', None))
                        r = _as_text(getattr(item, 'relation', None) or getattr(item, 'predicate', None))
                        d = _as_text(getattr(item, 'object', None) or getattr(item, 'tail', None))
                        c = float(getattr(item, 'score', 0.85) or 0.85)
                    s = s.strip().lower()
                    d = d.strip().lower()
                    r = r.strip()
                    if s and r and d:
                        cr = self._canon_rel(r)
                        # filter by allowlist and plausibility
                        if cr in allowed and _plausible(s) and _plausible(d):
                            triples.append((s, cr, d, c))
                    # Time budget check
                    if (time.perf_counter() - t_start) * 1000 > max_ms:
                        break
                return triples
            except Exception as e:
                logger.debug(f"[ReLiK] native extract failed: {e}")
                return []
        # HF text2text backend
        if self.backend == 'hf' and self.pipe is not None:
            prompt = (
                "Extract factual relations as triples from the text. "
                "Return JSON: {\"triples\":[{\"s\":...,\"r\":...,\"d\":...,\"score\":...}, ...]}\n" \
                f"TEXT: {text}"
            )
            try:
                out = self.pipe(prompt, max_new_tokens=128, do_sample=False)
                if isinstance(out, list) and len(out) > 0:
                    gen = out[0].get("generated_text") or out[0].get("text") or ""
                    parsed = self._parse_json(gen) or self._parse_inline(gen)
                else:
                    parsed = []
            except Exception as e:
                logger.debug(f"[ReLiK] HF generation failed: {e}")
                parsed = []
            triples: List[Tuple[str, str, str, float]] = []
            for s, r, d, c in parsed:
                triples.append((s.strip().lower(), self._canon_rel(r), d.strip().lower(), float(c)))
            return triples
        return []


__all__ = ["HotMemExtractor"]
