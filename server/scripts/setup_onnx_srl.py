#!/usr/bin/env python3
"""
Attempt to export an SRL tagger to ONNX (token classification with BIO SRL tags).

Usage:
  source server/.venv/bin/activate
  python scripts/setup_onnx_srl.py --model <hf_model_id> --out data/models/srl_onnx

Notes:
- There isn't yet a canonical HF SRL token-classification model; if your model is not
  directly exportable via Optimum, this script will report a helpful error.
- If you already have an ONNX SRL model, set env in server/.env:
    HOTMEM_USE_ONNX_SRL=true
    HOTMEM_ONNX_SRL_MODEL=/abs/path/to/model.onnx
    HOTMEM_ONNX_SRL_LABELS=/abs/path/to/labels.txt
    HOTMEM_ONNX_SRL_TOKENIZER=<tokenizer_id>
"""

import argparse
from pathlib import Path
import json


def export_srl(model_id: str, out_dir: str):
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer, AutoConfig
    except Exception as e:
        print(f"[SRL ONNX] Missing deps (optimum/transformers): {e}")
        return 2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[SRL ONNX] Exporting {model_id} → {out}")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        ort_model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
        ort_model.save_pretrained(str(out))
        tok.save_pretrained(str(out))
    except Exception as e:
        print(f"[SRL ONNX] Export failed. Your model might not be token-classification compatible: {e}")
        print("Provide an ONNX SRL model with BIO tags and labels.txt if possible.")
        return 3

    cfg = AutoConfig.from_pretrained(model_id)
    id2label = getattr(cfg, 'id2label', None)
    labels = []
    if isinstance(id2label, dict):
        labels = [id2label.get(str(i), id2label.get(i, f"LABEL_{i}")) for i in range(len(id2label))]
    else:
        print("[SRL ONNX] No id2label in config; please create labels.txt manually (B-V, I-V, B-ARG0, ...)")
        labels = ["O", "B-V", "I-V", "B-ARG0", "I-ARG0", "B-ARG1", "I-ARG1", "B-ARGM-TMP", "I-ARGM-TMP"]
    (out / 'labels.txt').write_text("\n".join(labels), encoding='utf-8')
    print(f"[SRL ONNX] Wrote labels.txt with {len(labels)} labels")
    print("Set in server/.env:")
    print(f"  HOTMEM_USE_ONNX_SRL=true")
    print(f"  HOTMEM_ONNX_SRL_MODEL={out}/model.onnx")
    print(f"  HOTMEM_ONNX_SRL_LABELS={out}/labels.txt")
    print(f"  HOTMEM_ONNX_SRL_TOKENIZER={model_id}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='HF model id for SRL (token classification)')
    p.add_argument('--out', default='data/models/srl_onnx', help='Output directory')
    args = p.parse_args()
    code = export_srl(args.model, args.out)
    raise SystemExit(code)


if __name__ == '__main__':
    main()

