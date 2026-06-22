#!/usr/bin/env python3
"""Embed the FULL-TEXT records with a frozen MiniLM and CACHE the vectors.

The de-truncated sibling of ``data/pilot/embed_pilot.py``. Two changes:

1. **Source** — embeds ``data/full`` (full body text, up to 8000 chars) instead
   of the truncated pilot.
2. **Window** — ``max_length=512`` (vs the pilot's 256), so the embedder actually
   sees more of the now-longer body. This is the whole point of R14: chars
   500→~2000 were invisible before (truncated below even the embedder's window).

Same model (``sentence-transformers/all-MiniLM-L6-v2``, 384-d), mean-pooled +
L2-normalized, on reference-scrubbed text — matching the de-referenced benchmark.
Everything downstream (data/full/run_full_ablation.py) runs on the committed
cache with numpy alone, reproducible in CI without torch or a network.

To support a CONTROLLED, paired full-vs-truncated comparison (same records, same
split, truncation the only variable), this also writes a second cache that
re-imposes the pilot regime ON THE SAME FULL RECORDS: bodies truncated to 500
chars, embedded at max_length=256. The cross-snapshot comparison against the
frozen truncated pilot is suggestive; this paired cache is the clean A/B.

Run:  python data/full/embed_full.py     # needs the [embed] extra
Out:  data/full/embeddings/minilm-l6-v2-512.npz       (full text, window 512)
      data/full/embeddings/minilm-l6-v2-trunc500.npz  (same records, 500-char
                                                       bodies, window 256 — the
                                                       paired truncated control)
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
OUT = HERE / "embeddings" / "minilm-l6-v2-512.npz"
OUT_TRUNC = HERE / "embeddings" / "minilm-l6-v2-trunc500.npz"
BATCH = 64
MAX_LEN = 512        # R14: use more of the (now full) body, vs the pilot's 256.
TRUNC_CHARS = 500    # the pilot's body cap, re-imposed for the paired control.
TRUNC_MAX_LEN = 256  # the pilot's window, re-imposed for the paired control.


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]


def _truncate_text(text: str, chars: int) -> str:
    """Mimic the pilot's char-cap on the SAME scrubbed text (paired control)."""
    return text[:chars] if len(text) > chars else text


def _embed(texts, tok, model, torch, max_len: int) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        vecs.append(emb.numpy())
        print(f"  embedded {min(i + BATCH, len(texts))}/{len(texts)} (max_length={max_len})",
              file=sys.stderr)
    return np.concatenate(vecs, axis=0).astype(np.float16)


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
    # Paired control: SAME records, bodies char-capped exactly like the pilot.
    trunc_texts = [_truncate_text(t, TRUNC_CHARS) for t in texts]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    # 1) Full text, wide window (the R14 cache).
    full = _embed(texts, tok, model, torch, MAX_LEN)
    np.savez_compressed(OUT, ids=np.array(ids), vectors=full, model=MODEL,
                        max_length=np.int64(MAX_LEN))
    print(f"wrote {full.shape} embeddings ({MODEL}, max_length={MAX_LEN}) -> {OUT}")

    # 2) Same records, pilot-regime control (500-char bodies, window 256).
    trunc = _embed(trunc_texts, tok, model, torch, TRUNC_MAX_LEN)
    np.savez_compressed(OUT_TRUNC, ids=np.array(ids), vectors=trunc, model=MODEL,
                        max_length=np.int64(TRUNC_MAX_LEN),
                        body_chars=np.int64(TRUNC_CHARS))
    print(f"wrote {trunc.shape} embeddings ({MODEL}, max_length={TRUNC_MAX_LEN}, "
          f"body<={TRUNC_CHARS}c) -> {OUT_TRUNC}")


if __name__ == "__main__":
    main()
