"""Chunked retrieval scorers (numpy/stdlib) — MaxP / FirstP / SumP / whole-doc mean.

Motivation: R14 showed mean-pooling a long body DILUTES the signal. Instead of one
vector per document, keep one vector per chunk and score a document by an aggregation
over its chunks. This is the standard long-document retrieval toolkit:

- **FirstP** — score by the first chunk only (≈ the 500-char truncation that won R14).
- **MaxP**   — score by the document's best chunk (the operator's idea).
- **SumP**   — sum of chunk scores (length-biased; included for completeness).
- **whole-doc mean** — cosine to the mean of the chunk vectors (the R14 loser).

References: Dai & Callan (2019) FirstP/MaxP/SumP; ColBERT MaxSim (Khattab & Zaharia
2020); PARADE passage aggregation (Li et al. 2020). This module is the chunk-level
(coarse) analogue — frozen embeddings, no training — for a cheap first probe.
"""

from __future__ import annotations

import re

import numpy as np

_WS = re.compile(r"\s+")


def chunk_text(text: str, size: int = 512, overlap: float = 0.2) -> list[str]:
    """Split text into overlapping char windows of ~``size`` chars.

    ``overlap`` is a fraction of ``size``. Always returns at least one chunk; for
    text shorter than ``size`` the single chunk is the whole text.
    """
    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= size:
        return [text]
    step = max(1, int(size * (1.0 - overlap)))
    chunks = [text[i:i + size] for i in range(0, len(text), step)]
    # Drop a tiny trailing fragment fully covered by the previous window.
    return [c for c in chunks if c.strip()] or [text[:size]]


# --- aggregators: query_vec (d,) unit + chunk_vecs (n, d) unit -> scalar score ---

def maxp(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> float:
    if chunk_vecs.shape[0] == 0:
        return -1.0
    return float(np.max(chunk_vecs @ query_vec))


def firstp(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> float:
    if chunk_vecs.shape[0] == 0:
        return -1.0
    return float(chunk_vecs[0] @ query_vec)


def sump(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> float:
    if chunk_vecs.shape[0] == 0:
        return -1.0
    return float(np.sum(chunk_vecs @ query_vec))


def meanp(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> float:
    if chunk_vecs.shape[0] == 0:
        return -1.0
    return float(np.mean(chunk_vecs @ query_vec))


def whole_doc_mean(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> float:
    if chunk_vecs.shape[0] == 0:
        return -1.0
    m = chunk_vecs.mean(axis=0)
    n = np.linalg.norm(m)
    if n == 0:
        return -1.0
    return float((m / n) @ query_vec)


AGGREGATORS = {
    "firstp": firstp,
    "maxp": maxp,
    "sump": sump,
    "meanp": meanp,
    "whole-doc-mean": whole_doc_mean,
}


def rank(query_vec: np.ndarray, candidates: dict[str, np.ndarray], aggregator) -> list[str]:
    """Rank candidate ids (each mapped to its (n_chunks, d) matrix) by the aggregator."""
    scored = [(cid, aggregator(query_vec, chunks)) for cid, chunks in candidates.items()]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return [cid for cid, _ in scored]
