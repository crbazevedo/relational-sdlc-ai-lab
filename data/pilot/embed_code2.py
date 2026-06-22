#!/usr/bin/env python3
"""Q6 follow-up — embed the pilot with code-AWARE embedding models.

Q6 (run_code_ablation.py / ablation-code.md) found the axis that matters is
*embedding-tuned* vs not, not *code* vs general: a code MLM (CodeBERT) collapses
for retrieval, while a strong embedding-tuned general model (bge-small) edges past
MiniLM. The open follow-up is a model that is BOTH code-aware AND embedding-tuned.

This script embeds the same scrubbed pilot text with two code-aware bases that load
cleanly under transformers 5.x with standard ``AutoModel``/``AutoTokenizer`` (NO
remote code):

  * ``microsoft/unixcoder-base`` — code-pretrained (RoBERTa arch), mean-pooled. A
    code base that is NOT embedding-tuned — the unixcoder analogue of CodeBERT.
  * ``flax-sentence-embeddings/st-codesearch-distilroberta-base`` — DistilRoBERTa
    contrastively fine-tuned on CodeSearchNet code<->doc pairs: code-aware AND
    embedding-tuned, mean-pooled. This is the actual Q6-follow-up target.

A code-embedding model that needed remote code FAILED under transformers 5.x:
``jinaai/jina-embeddings-v2-base-code`` (its remote modeling_bert.py imports
``find_pruneable_heads_and_indices`` from transformers.pytorch_utils, removed in 5.x)
— see docs/ablation-code2.md. ``Salesforce/codet5p-110m-embedding`` likewise fails
under transformers 5.x remote code (carried over from Q6).

Needs the [embed] extra. Caches committed npz; downstream eval is numpy-only.

Run:  python data/pilot/embed_code2.py
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

# (model_name, pooling, output_cache, trust_remote_code)
MODELS = [
    ("microsoft/unixcoder-base", "mean", "code2-unixcoder.npz", False),
    ("flax-sentence-embeddings/st-codesearch-distilroberta-base", "mean",
     "code2-stcodesearch.npz", False),
]
MAX_LEN = 256
BATCH = 64


def _load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def embed_with(model_name, pool, out_name, trust):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    records = _load_jsonl(HERE / "records.jsonl")
    ids = [r["id"] for r in records]
    texts = [scrub_record_text(r) or r["id"] for r in records]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust)
    model.eval()
    vecs = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        if pool == "cls":
            e = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        vecs.append(F.normalize(e, p=2, dim=1).numpy())
        print(f"  [{model_name}] {min(i + BATCH, len(texts))}/{len(texts)}",
              file=sys.stderr)
    matrix = np.concatenate(vecs, 0).astype(np.float16)
    out = HERE / "embeddings" / out_name
    np.savez_compressed(out, ids=np.array(ids), vectors=matrix, model=model_name)
    print(f"wrote {matrix.shape} ({model_name}) -> {out}")


def main() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs [embed] extra: {exc}", file=sys.stderr)
        raise SystemExit(2)
    for name, pool, out, trust in MODELS:
        embed_with(name, pool, out, trust)


if __name__ == "__main__":
    main()
