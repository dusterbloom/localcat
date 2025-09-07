"""
ONNX-based NER and SRL wrappers (fully local) for HotMem.

Two generic taggers using ONNX Runtime + HF tokenizers:
- OnnxTokenNER: BIO/IOB2 per-token labels → entities (text, label, score)
- OnnxSRLTagger: BIO SRL tags (e.g., B-V, B-ARG0, I-ARGM-TMP) → predicate roles → triples

Both are model-agnostic if the output node is named 'logits' of shape [B, T, C],
and label files map index→string (one per line). Tokenizer is set by name.

Enable via env and provide local model/label paths.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from loguru import logger


def _load_labels(path: str) -> List[str]:
    labels: List[str] = []
    if not path or not os.path.exists(path):
        return labels
    try:
        # Allow either plain-text (one per line) or JSON list
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read().strip()
        if txt.startswith('['):
            labels = json.loads(txt)
        else:
            labels = [line.strip() for line in txt.splitlines() if line.strip()]
    except Exception as e:
        logger.warning(f"[ONNX] Failed to load labels at {path}: {e}")
        labels = []
    return labels


def _softmax(logits):
    import numpy as np
    x = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / (np.sum(exp, axis=-1, keepdims=True) + 1e-9)


@dataclass
class OnnxTokenNER:
    model_path: str
    labels_path: str
    tokenizer_name: str = 'bert-base-cased'
    max_len: int = 256
    provider: Optional[str] = None

    def __post_init__(self):
        try:
            import onnxruntime as ort  # type: ignore
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(f"[ONNX NER] Missing dependency: {e}")

        if not os.path.exists(self.model_path):
            raise RuntimeError(f"[ONNX NER] Model not found: {self.model_path}")
        self.labels = _load_labels(self.labels_path)
        if not self.labels:
            raise RuntimeError("[ONNX NER] Labels not found or empty")

        sess_opts = ort.SessionOptions()
        providers = [self.provider] if self.provider else None
        self.session = ort.InferenceSession(self.model_path, sess_options=sess_opts, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        if not getattr(self.tokenizer, 'is_fast', False):
            logger.warning("[ONNX NER] Fast tokenizer not available; offset mapping disabled")

        # Infer input names
        self.input_names = {i.name for i in self.session.get_inputs()}

    def extract(self, text: str) -> List[Tuple[str, str, float, Tuple[int, int]]]:
        """
        Return entities as (span_text, label, score, (char_start, char_end)).
        """
        if not text:
            return []
        tok = self.tokenizer(
            text,
            return_tensors='np',
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
        )
        inputs = {}
        for name in ['input_ids', 'attention_mask', 'token_type_ids']:
            if name in self.input_names and name in tok:
                inputs[name] = tok[name]

        outputs = self.session.run(None, inputs)
        logits = outputs[0]  # [B, T, C]
        probs = _softmax(logits)[0]
        ids = tok['input_ids'][0]
        offsets = tok.get('offset_mapping')

        ents: List[Tuple[str, str, float, Tuple[int, int]]] = []
        curr_label = None
        curr_tokens = []
        curr_scores = []
        curr_start = None
        curr_end = None

        # Skip special tokens via offset (0,0)
        for i in range(len(ids)):
            start, end = (0, 0)
            if offsets is not None:
                start, end = offsets[0][i]
            if end == 0 and start == 0:
                continue
            label_id = int(probs[i].argmax())
            label = self.labels[label_id]
            score = float(probs[i][label_id])

            if label.startswith('B-'):
                # flush previous
                if curr_label and curr_tokens:
                    span_text = text[curr_start:curr_end]
                    ents.append((span_text, curr_label, sum(curr_scores)/max(1, len(curr_scores)), (curr_start, curr_end)))
                curr_label = label[2:]
                curr_tokens = [i]
                curr_scores = [score]
                curr_start, curr_end = int(start), int(end)
            elif label.startswith('I-') and curr_label == label[2:]:
                curr_tokens.append(i)
                curr_scores.append(score)
                curr_end = int(end)
            else:
                # flush if active
                if curr_label and curr_tokens:
                    span_text = text[curr_start:curr_end]
                    ents.append((span_text, curr_label, sum(curr_scores)/max(1, len(curr_scores)), (curr_start, curr_end)))
                curr_label, curr_tokens, curr_scores, curr_start, curr_end = None, [], [], None, None

        if curr_label and curr_tokens:
            span_text = text[curr_start:curr_end]
            ents.append((span_text, curr_label, sum(curr_scores)/max(1, len(curr_scores)), (curr_start, curr_end)))
        return ents


@dataclass
class OnnxSRLTagger:
    model_path: str
    labels_path: str
    tokenizer_name: str = 'bert-base-cased'
    max_len: int = 256
    provider: Optional[str] = None

    def __post_init__(self):
        try:
            import onnxruntime as ort  # type: ignore
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(f"[ONNX SRL] Missing dependency: {e}")
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"[ONNX SRL] Model not found: {self.model_path}")
        self.labels = _load_labels(self.labels_path)
        if not self.labels:
            raise RuntimeError("[ONNX SRL] Labels not found or empty")
        sess_opts = ort.SessionOptions()
        providers = [self.provider] if self.provider else None
        self.session = ort.InferenceSession(self.model_path, sess_options=sess_opts, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _map_role(self, tag: str) -> Optional[str]:
        t = tag.upper().replace('B-', '').replace('I-', '')
        # Map PropBank roles to universal roles
        if t == 'V':
            return 'V'
        if t in {'ARG0'}:
            return 'agent'
        if t in {'ARG1'}:
            return 'patient'
        if t in {'ARG2'}:  # instrument/attribute
            return 'instrument'
        if t in {'ARG3'}:
            return 'beneficiary'
        if t in {'ARG4'}:
            return 'source'
        if t in {'ARGM-TMP', 'ARGM_TMP', 'ARGM-TIME'}:
            return 'temporal'
        if t in {'ARGM-LOC', 'ARGM_LOC'}:
            return 'location'
        if t in {'ARGM-DIR', 'ARGM_DIR'}:
            return 'location'  # directional/location
        if t in {'ARGM-CAU', 'ARGM_CAU'}:
            return 'cause'
        if t in {'ARGM-BEN', 'ARGM_BEN'}:
            return 'beneficiary'
        if t in {'ARGM-MNR', 'ARGM_MNR'}:
            return None
        return None

    def extract(self, text: str) -> List[Dict[str, str]]:
        """
        Returns a list of predicate-role dicts: {'predicate': '...', 'agent': '...', 'patient': '...', ...}
        """
        if not text:
            return []
        tok = self.tokenizer(
            text,
            return_tensors='np',
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
        )
        inputs = {}
        for name in ['input_ids', 'attention_mask', 'token_type_ids']:
            if name in self.input_names and name in tok:
                inputs[name] = tok[name]
        outputs = self.session.run(None, inputs)
        logits = outputs[0]  # [B, T, C]
        probs = _softmax(logits)[0]
        ids = tok['input_ids'][0]
        offsets = tok.get('offset_mapping')

        # Decode tags
        import numpy as np
        tag_ids = np.argmax(probs, axis=-1)
        tags = [self.labels[int(i)] for i in tag_ids]

        def span_text(start: int, end: int) -> str:
            return text[start:end]

        roles: Dict[str, str] = {}
        pred_span = None

        # Aggregate contiguous BIO spans per role
        i = 0
        while i < len(tags):
            start, end = (0, 0)
            if offsets is not None:
                start, end = offsets[0][i]
            tag = tags[i]
            if end == 0 and start == 0:
                i += 1
                continue
            if tag.startswith('B-'):
                role = self._map_role(tag)
                j = i + 1
                last_end = end
                while j < len(tags):
                    s2, e2 = offsets[0][j] if offsets is not None else (0, 0)
                    if not tags[j].startswith('I-'):
                        break
                    last_end = e2
                    j += 1
                text_span = span_text(int(start), int(last_end))
                if role == 'V' or tag.endswith('-V'):
                    pred_span = text_span
                elif role:
                    roles[role] = text_span
                i = j
            else:
                i += 1

        if pred_span:
            roles = {'predicate': pred_span, **roles}
            return [roles]
        return []


__all__ = [
    'OnnxTokenNER',
    'OnnxSRLTagger',
]

