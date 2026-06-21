#!/usr/bin/env python3
"""Embed the pilot records with a small pretrained model and CACHE the vectors.

This is the only step that needs the heavy ``[embed]`` extra (transformers +
torch) and a one-time model download. Everything downstream — the relation head
and the cross-repo ablation — runs on the committed cache with numpy only, so the
result is reproducible in CI without torch or a network.

Model: ``sentence-transformers/all-MiniLM-L6-v2`` (22M params, 384-d, CPU-fast),
mean-pooled + L2-normalized. Text is reference-scrubbed first, matching the
de-referenced benchmark.

Run:  python data/pilot/embed_pilot.py     # needs: pip install -e '.[embed]'
Out:  data/pilot/embeddings/minilm-l6-v2.npz   (ids + float16 vectors)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.scrub import scrub_record_text  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT = HERE / "embeddings" / "minilm-l6-v2.npz"
BATCH = 64
MAX_LEN = 256


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs the embed extra (pip install -e '.[embed]'): {exc}", file=sys.stderr)
        raise SystemExit(2)

    records = _load_jsonl(HERE / "records.jsonl")
    ids = [r["id"] for r in records]
    texts = [scrub_record_text(r) or r["id"] for r in records]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()

    vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        vecs.append(emb.numpy())
        print(f"  embedded {min(i + BATCH, len(texts))}/{len(texts)}", file=sys.stderr)

    matrix = np.concatenate(vecs, axis=0).astype(np.float16)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, ids=np.array(ids), vectors=matrix, model=MODEL)
    print(f"wrote {matrix.shape} embeddings ({MODEL}) -> {OUT}")


if __name__ == "__main__":
    main()
