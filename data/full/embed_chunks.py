#!/usr/bin/env python3
"""Embed FULL-text bodies as CHUNKS at several chunk sizes (for MaxP/FirstP/SumP).

Reuses data/full (the de-truncated bodies) — no re-ingest. For each chunk size, every
record's scrubbed title+body is split into overlapping char windows, each window is
embedded with frozen MiniLM, and the per-chunk vectors are cached as a flat matrix +
per-document offsets. Downstream scoring (run_chunk_ablation.py) is numpy-only.

Needs the [embed] extra. Run:  python data/full/embed_chunks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.chunking import chunk_text  # noqa: E402
from relsdlc.scrub import scrub_record_text  # noqa: E402

import json  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIZES = [256, 512, 1024]
OVERLAP = 0.2
MAX_LEN = 256
EMB = HERE / "embeddings"


def _load_jsonl(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs [embed] extra: {exc}", file=sys.stderr); raise SystemExit(2)

    records = _load_jsonl(HERE / "records.jsonl")
    ids = [r["id"] for r in records]
    texts = [scrub_record_text(r) or r["id"] for r in records]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL); model.eval()
    EMB.mkdir(parents=True, exist_ok=True)

    def embed(batch):
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return F.normalize(e, p=2, dim=1).numpy()

    for size in SIZES:
        all_chunks, offsets = [], [0]
        for t in texts:
            cs = chunk_text(t, size=size, overlap=OVERLAP)
            all_chunks.extend(cs)
            offsets.append(offsets[-1] + len(cs))
        vecs = []
        for i in range(0, len(all_chunks), 128):
            vecs.append(embed(all_chunks[i:i + 128]))
            print(f"  size={size}: {min(i + 128, len(all_chunks))}/{len(all_chunks)}", file=sys.stderr)
        matrix = np.concatenate(vecs, 0).astype(np.float16)
        np.savez_compressed(EMB / f"chunks-s{size}.npz", ids=np.array(ids),
                            offsets=np.array(offsets, dtype=np.int64), vectors=matrix,
                            model=MODEL, size=size)
        print(f"wrote {matrix.shape} chunk vecs (size={size}, {len(ids)} docs) -> "
              f"chunks-s{size}.npz")


if __name__ == "__main__":
    main()
