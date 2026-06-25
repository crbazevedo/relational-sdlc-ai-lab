#!/usr/bin/env python3
"""R20 Stage 2 — embed the dense PR nodes (from Stage 1) into the pilot space.

Same recipe as data/pilot/embed_pilot.py EXACTLY (MiniLM-L6 mean-pooled + L2-norm,
reference-scrubbed title+body, float16, CPU) so the new PR vectors live in the same
space as the committed pilot cache and can be aggregated into test-node features in
Stage 3. Only the new modifying-PR records need embedding (~510). Gitignored cache.

Run:  python data/pilot/graph/embed_dense_pr.py     # needs the [embed] extra
Out:  data/pilot/embeddings/dense-pr.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent              # data/pilot/graph
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.scrub import scrub_record_text  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT = REPO_ROOT / "data" / "pilot" / "embeddings" / "dense-pr.npz"
BATCH, MAX_LEN = 64, 256


def main() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    recs = [json.loads(l) for l in (HERE / "records_dense.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
    prs = [r for r in recs if r["type"] == "pull_request"]
    ids = [r["id"] for r in prs]
    texts = [scrub_record_text(r) or r["id"] for r in prs]
    print(f"embedding {len(ids)} dense PR records", file=sys.stderr, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    vecs = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        vecs.append(emb.numpy())
    matrix = np.concatenate(vecs, 0).astype(np.float16)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, ids=np.array(ids), vectors=matrix, model=MODEL)
    print(f"wrote {matrix.shape} -> {OUT}")


if __name__ == "__main__":
    main()
