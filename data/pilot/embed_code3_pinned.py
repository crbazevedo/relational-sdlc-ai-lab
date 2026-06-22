#!/usr/bin/env python3
"""Q6 finish — embed the pilot with a true code-EMBEDDING base under transformers<5.

R13B / ablation-code2.md left one stone unturned: the genuine code-*embedding*
models (``jinaai/jina-embeddings-v2-base-code``, ``Salesforce/codet5p-110m-embedding``)
both FAIL under transformers 5.x. Their ``trust_remote_code`` modeling files import
``find_pruneable_heads_and_indices`` from ``transformers.pytorch_utils`` — a symbol
REMOVED in transformers 5.x — so the import dies before any forward pass. The
pilot/scale machine ships transformers 5.12, which is why those models were queued
rather than evaluated.

This script closes the gap by running under a PINNED environment:

    transformers >= 4.40, < 5   (the venv pin — restores find_pruneable_heads_and_indices)
    torch 2.x                   (reused from system site-packages)

Recipe (documented in docs/ablation-code3.md):

    uv venv .venv-r15b --python <python3.14> --system-site-packages
    uv pip install --python .venv-r15b 'transformers>=4.40,<5' einops sentencepiece
    .venv-r15b/bin/python data/pilot/embed_code3_pinned.py

It tries the genuine code-embedding models in order and uses the FIRST that both
loads AND produces a sane geometry (cos(related) > cos(unrelated) on a built-in
sanity probe). It caches the chosen model's pilot vectors to
``embeddings/code3-<name>.npz`` (float16; ids + vectors + model), matching the
embed_code/embed_code2 cache contract so the downstream eval stays numpy-only and
transformers-version-agnostic on the committed npz.

PINNED-TRANSFORMERS REQUIREMENT: this module MUST be run under the transformers<5
venv. It refuses (SystemExit 3) if it detects transformers 5.x, because the target
models are precisely the ones that need a pre-5.x ``pytorch_utils`` surface.

Run:  .venv-r15b/bin/python data/pilot/embed_code3_pinned.py
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

# Genuine code-EMBEDDING candidates, in preference order. Each entry:
#   (model_name, pooling, output_cache, trust_remote_code, returns_embedding)
# returns_embedding=True means the model's forward returns a pooled embedding
# directly (codet5p-110m-embedding) rather than last_hidden_state to pool.
MODELS = [
    ("jinaai/jina-embeddings-v2-base-code", "mean",
     "code3-jina-code.npz", True, False),
    ("Salesforce/codet5p-110m-embedding", "none",
     "code3-codet5p.npz", True, True),
    ("nomic-ai/CodeRankEmbed", "mean",
     "code3-coderankembed.npz", True, False),
]
MAX_LEN = 256
BATCH = 32

# Sanity probe: a code snippet, a paraphrase of what it does, and an unrelated
# string. A usable retrieval base must put the paraphrase nearer than the noise.
PROBE_CODE = "def add(a, b):\n    return a + b  # sum two numbers"
PROBE_RELATED = "a function that returns the sum of two integer arguments"
PROBE_UNRELATED = "the weather forecast predicts heavy rain over the weekend"


def _load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _pool(model_out, enc, pool, returns_embedding):
    import torch.nn.functional as F

    if returns_embedding:
        # codet5p-110m-embedding returns the embedding tensor directly.
        e = model_out if hasattr(model_out, "dim") else model_out[0]
        if e.dim() == 1:
            e = e.unsqueeze(0)
        return F.normalize(e, p=2, dim=1)
    h = model_out.last_hidden_state
    if pool == "cls":
        e = h[:, 0]
    else:
        m = enc["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return F.normalize(e, p=2, dim=1)


def _embed_texts(tok, model, texts, pool, returns_embedding):
    import torch

    vecs = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        vecs.append(_pool(out, enc, pool, returns_embedding).cpu().numpy())
        print(f"    {min(i + BATCH, len(texts))}/{len(texts)}", file=sys.stderr)
    return np.concatenate(vecs, 0)


def _sanity_ok(tok, model, pool, returns_embedding):
    """cos(code, related) must exceed cos(code, unrelated) by a clear margin."""
    v = _embed_texts(tok, model, [PROBE_CODE, PROBE_RELATED, PROBE_UNRELATED],
                     pool, returns_embedding)
    cos_rel = float(v[0] @ v[1])
    cos_unrel = float(v[0] @ v[2])
    print(f"  sanity: cos(related)={cos_rel:.3f}  cos(unrelated)={cos_unrel:.3f}",
          file=sys.stderr)
    return cos_rel > cos_unrel + 0.05, cos_rel, cos_unrel


def try_model(model_name, pool, out_name, trust, returns_embedding):
    """Load + sanity-check + embed the pilot. Returns (out_path, model_name) or None."""
    from transformers import AutoModel, AutoTokenizer

    print(f"[try] {model_name}", file=sys.stderr)
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust)
    except Exception as exc:  # noqa: BLE001 — the whole point is to record the blocker
        print(f"  LOAD FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    model.eval()

    try:
        ok, cos_rel, cos_unrel = _sanity_ok(tok, model, pool, returns_embedding)
    except Exception as exc:  # noqa: BLE001
        print(f"  SANITY FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    if not ok:
        print(f"  SANITY REJECT: cos(related)={cos_rel:.3f} not clearly above "
              f"cos(unrelated)={cos_unrel:.3f}", file=sys.stderr)
        return None

    records = _load_jsonl(HERE / "records.jsonl")
    ids = [r["id"] for r in records]
    texts = [scrub_record_text(r) or r["id"] for r in records]
    print(f"  embedding {len(texts)} pilot records ...", file=sys.stderr)
    matrix = _embed_texts(tok, model, texts, pool, returns_embedding).astype(np.float16)
    out = HERE / "embeddings" / out_name
    np.savez_compressed(out, ids=np.array(ids), vectors=matrix, model=model_name)
    print(f"wrote {matrix.shape} ({model_name}) -> {out}")
    return out, model_name


def main() -> None:
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs transformers (pinned <5): {exc}", file=sys.stderr)
        raise SystemExit(2)
    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        print(f"ERROR: transformers {transformers.__version__} is 5.x. The target "
              "code-embedding models need transformers<5 (their remote code imports "
              "find_pruneable_heads_and_indices, removed in 5.x). Use the pinned "
              "venv: see docs/ablation-code3.md.", file=sys.stderr)
        raise SystemExit(3)
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs torch (system site-packages): {exc}", file=sys.stderr)
        raise SystemExit(2)

    print(f"transformers {transformers.__version__} (pinned <5); "
          f"torch {torch.__version__}", file=sys.stderr)

    for name, pool, out, trust, returns_embedding in MODELS:
        result = try_model(name, pool, out, trust, returns_embedding)
        if result is not None:
            print(f"\nCHOSEN: {result[1]} -> {result[0].name}")
            return
    print("\nNO code-embedding model loaded + passed sanity under transformers<5. "
          "See docs/ablation-code3.md for the recorded blockers.", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
