#!/usr/bin/env python3
"""
Setup local models for HotMem (privacy-first):
- Downloads spaCy models
- Warms sentence-transformers relation normalizer
- Exports an ONNX NER model (token classification) with labels

Optional args:
  --ner MODEL_ID        HuggingFace model id for NER (default: Davlan/xlm-roberta-base-ner-hrl)
  --out DIR             Output dir (default: data/models/ner_onnx)
  --langs LIST          Comma list of spaCy languages to install (default: en)
  --warm-st MODEL_ID    Sentence-transformers model to warm (default: paraphrase-multilingual-MiniLM-L12-v2)

Examples:
  python scripts/setup_local_models.py
  python scripts/setup_local_models.py --ner dslim/bert-base-NER --langs en,es,de,fr,it
"""

import os
import json
import argparse
from pathlib import Path


def ensure_spacy_models(langs: list[str]):
    import spacy
    from spacy.cli.download import download
    model_map = {
        'en': 'en_core_web_sm',
        'es': 'es_core_news_sm',
        'fr': 'fr_core_news_sm',
        'de': 'de_core_news_sm',
        'it': 'it_core_news_sm',
    }
    for lg in langs:
        name = model_map.get(lg)
        if not name:
            print(f"[spaCy] No small model mapped for '{lg}', skipping")
            continue
        try:
            spacy.load(name)
            print(f"[spaCy] Model already installed: {name}")
        except Exception:
            print(f"[spaCy] Downloading: {name}")
            download(name)


def warm_sentence_transformers(model_id: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"[ST] sentence-transformers not installed: {e}")
        return
    print(f"[ST] Loading: {model_id}")
    m = SentenceTransformer(model_id)
    # quick warmup
    _ = m.encode(["warmup"], normalize_embeddings=True)
    print("[ST] Warmed and cached")


def export_onnx_ner(model_id: str, out_dir: str):
    from transformers import AutoConfig
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[ONNX NER] Exporting {model_id} → {out}")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    ort_model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
    ort_model.save_pretrained(str(out))
    tok.save_pretrained(str(out))

    # Write labels.txt from id2label
    cfg = AutoConfig.from_pretrained(model_id)
    id2label = getattr(cfg, 'id2label', None)
    if not id2label:
        # Try to read from model files
        mapping = (out / 'config.json')
        if mapping.exists():
            cj = json.loads(mapping.read_text())
            id2label = cj.get('id2label')
    labels = []
    if isinstance(id2label, dict):
        # Ensure index order; keys may be str or int
        labels = [id2label.get(str(i), id2label.get(i, f"LABEL_{i}")) for i in range(len(id2label))]
    else:
        # Fallback to common BIO set
        labels = [
            'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'
        ]
    (out / 'labels.txt').write_text("\n".join(labels), encoding='utf-8')
    print(f"[ONNX NER] Wrote labels.txt with {len(labels)} labels")

    print("[ONNX NER] Done. Set in server/.env:")
    print(f"  HOTMEM_USE_ONNX_NER=true")
    print(f"  HOTMEM_ONNX_NER_MODEL={out}/model.onnx")
    print(f"  HOTMEM_ONNX_NER_LABELS={out}/labels.txt")
    print(f"  HOTMEM_ONNX_NER_TOKENIZER={model_id}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ner', default='Davlan/xlm-roberta-base-ner-hrl', help='HF model id for NER')
    p.add_argument('--out', default='data/models/ner_onnx', help='Output dir')
    p.add_argument('--langs', default='en', help='Comma-separated spaCy languages (en,es,fr,de,it)')
    p.add_argument('--warm-st', default='paraphrase-multilingual-MiniLM-L12-v2', help='Sentence-transformers model id to warm')
    args = p.parse_args()

    langs = [s.strip() for s in args.langs.split(',') if s.strip()]
    ensure_spacy_models(langs)
    warm_sentence_transformers(args.warm_st)
    export_onnx_ner(args.ner, args.out)


if __name__ == '__main__':
    main()
