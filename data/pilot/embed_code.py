#!/usr/bin/env python3
"""Q6 — embed the pilot with alternative base models (cache for numpy eval).

Tests whether the embedding BASE matters: a stronger general embedder
(BAAI/bge-small-en-v1.5, CLS-pooled) and a code-pretrained-but-not-embedding-tuned
model (microsoft/codebert-base, mean-pooled). Both vs the MiniLM substrate.

Needs the [embed] extra. Caches committed npz; downstream eval is numpy-only.

Run:  python data/pilot/embed_code.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from relsdlc.scrub import scrub_record_text  # noqa: E402

import json  # noqa: E402

MODELS = [
    ("BAAI/bge-small-en-v1.5", "cls", "code-bge.npz"),
    ("microsoft/codebert-base", "mean", "code-codebert.npz"),
]
MAX_LEN = 256


def _load_jsonl(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def embed_with(model_name, pool, out_name):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    records = _load_jsonl(HERE / "records.jsonl")
    ids = [r["id"] for r in records]
    texts = [scrub_record_text(r) or r["id"] for r in records]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    vecs = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        if pool == "cls":
            e = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        vecs.append(F.normalize(e, p=2, dim=1).numpy())
        print(f"  [{model_name}] {min(i + 64, len(texts))}/{len(texts)}", file=sys.stderr)
    matrix = np.concatenate(vecs, 0).astype(np.float16)
    out = HERE / "embeddings" / out_name
    np.savez_compressed(out, ids=np.array(ids), vectors=matrix, model=model_name)
    print(f"wrote {matrix.shape} ({model_name}) -> {out}")


def main() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs [embed] extra: {exc}", file=sys.stderr); raise SystemExit(2)
    for name, pool, out in MODELS:
        embed_with(name, pool, out)


if __name__ == "__main__":
    main()
