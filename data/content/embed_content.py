#!/usr/bin/env python3
"""Embed file/test CONTENTS as CHUNKS (256/512/1024) + the PR queries (R16A).

The deep-signal counterpart to data/full/embed_chunks.py. For each chunk size,
every content record's scrubbed file text is split into overlapping char windows,
each window is embedded with frozen MiniLM, and the per-chunk vectors are cached as
a flat matrix + per-document offsets. We ALSO embed the PR QUERIES (their
title+body, scrubbed, whole) so the diff->test ablation has a query vector per PR.

Caches go UNDER data/content/embeddings/ (gitignored — regenerable from the
committed file_contents.jsonl + the live PR records via this script).

Needs the [embed] extra. Run:  python data/content/embed_content.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.chunking import chunk_text  # noqa: E402
from relsdlc.scrub import scrub_record_text, scrub_references  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIZES = [256, 512, 1024]
OVERLAP = 0.2
MAX_LEN = 256
EMB = HERE / "embeddings"

CONTENT = HERE / "file_contents.jsonl"
BENCH = HERE / "benchmark" / "diff_to_affected_test.jsonl"
PILOT_RECORDS = REPO_ROOT / "data" / "pilot" / "records.jsonl"


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def _content_text(rec: dict) -> str:
    """Scrub a content record's file text (content.text), references removed."""
    text = (rec.get("content") or {}).get("text", "") or ""
    return scrub_references(text)


def main() -> None:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs [embed] extra: {exc}", file=sys.stderr)
        raise SystemExit(2)

    # Content (file/test) records: deep file text.
    content_records = _load_jsonl(CONTENT)
    content_ids = [r["id"] for r in content_records]
    content_texts = [_content_text(r) or r["id"] for r in content_records]

    # PR query records (from the pilot) referenced by the benchmark — embed whole.
    queries = _load_jsonl(BENCH)
    query_ids_needed = sorted({q["query_record"] for q in queries})
    pilot_by_id = {r["id"]: r for r in _load_jsonl(PILOT_RECORDS)}
    q_ids: list[str] = []
    q_texts: list[str] = []
    for qid in query_ids_needed:
        rec = pilot_by_id.get(qid)
        if rec is None:
            continue
        q_ids.append(qid)
        q_texts.append(scrub_record_text(rec) or qid)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()
    EMB.mkdir(parents=True, exist_ok=True)

    def embed(batch: list[str]) -> np.ndarray:
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return F.normalize(e, p=2, dim=1).numpy()

    def embed_all(texts: list[str], label: str) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), 128):
            vecs.append(embed(texts[i:i + 128]))
            print(f"  {label}: {min(i + 128, len(texts))}/{len(texts)}", file=sys.stderr)
        if not vecs:
            return np.zeros((0, 384), dtype=np.float16)
        return np.concatenate(vecs, 0).astype(np.float16)

    # Document (content) chunk caches, one per chunk size.
    for size in SIZES:
        all_chunks, offsets = [], [0]
        for t in content_texts:
            cs = chunk_text(t, size=size, overlap=OVERLAP)
            all_chunks.extend(cs)
            offsets.append(offsets[-1] + len(cs))
        matrix = embed_all(all_chunks, f"docs size={size}")
        np.savez_compressed(
            EMB / f"content-chunks-s{size}.npz",
            ids=np.array(content_ids),
            offsets=np.array(offsets, dtype=np.int64),
            vectors=matrix, model=MODEL, size=size,
        )
        print(f"wrote {matrix.shape} chunk vecs (size={size}, {len(content_ids)} docs) "
              f"-> content-chunks-s{size}.npz")

    # PR query whole-doc embeddings.
    qmatrix = embed_all(q_texts, "queries")
    np.savez_compressed(
        EMB / "queries.npz",
        ids=np.array(q_ids), vectors=qmatrix, model=MODEL, max_length=MAX_LEN,
    )
    print(f"wrote {qmatrix.shape} query vecs ({len(q_ids)} PRs) -> queries.npz")


if __name__ == "__main__":
    main()
