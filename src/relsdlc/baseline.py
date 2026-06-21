"""A dependency-light text/code embedding + retrieval baseline.

This is the *vanilla* embedding the relational model must beat: a deterministic
hashed bag-of-words with TF weighting and cosine similarity. No torch, no numpy,
no network — it exists so the end-to-end benchmark harness is reproducible from a
clean checkout. Stronger off-the-shelf embedders (e.g. code embedding models) are
introduced later as optional dependencies in the P2/P3 baselines.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence

_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_HASH_DIM = 1 << 14  # 16384 buckets; ample for small fixtures.


def _bucket(token: str, dim: int) -> int:
    # crc32 is deterministic across processes and platforms, unlike the builtin
    # hash() for str (which is salted by PYTHONHASHSEED). Reproducibility matters.
    return zlib.crc32(token.encode("utf-8")) % dim


def tokenize(text: str) -> list[str]:
    """Lowercase word/identifier tokenization with camelCase and snake_case splitting."""
    tokens: list[str] = []
    for raw in _WORD.findall(text or ""):
        parts = _CAMEL.sub(" ", raw).split()
        for part in parts:
            for sub in part.split("_"):
                if sub:
                    tokens.append(sub.lower())
        # Also keep the whole identifier (lowercased) as a token.
        tokens.append(raw.lower())
    return tokens


def embed(text: str, dim: int = _HASH_DIM) -> dict[int, float]:
    """Sparse hashed-TF vector. Keys are hash buckets, values are log-scaled TF."""
    counts = Counter(tokenize(text))
    vec: dict[int, float] = {}
    for tok, c in counts.items():
        vec[_bucket(tok, dim)] = vec.get(_bucket(tok, dim), 0.0) + 1.0 + math.log(c)
    return vec


def cosine(a: Mapping[int, float], b: Mapping[int, float]) -> float:
    if not a or not b:
        return 0.0
    # Iterate the smaller vector for the dot product.
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(val * large.get(idx, 0.0) for idx, val in small.items())
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


def rank(query_text: str, candidates: Mapping[str, str]) -> list[str]:
    """Rank candidate ids by cosine similarity to the query (best first).

    Ties break by descending score then ascending id, so the ranking is fully
    deterministic across runs and platforms.
    """
    q = embed(query_text)
    scored = [(cid, cosine(q, embed(text))) for cid, text in candidates.items()]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return [cid for cid, _ in scored]


def rank_against(query_text: str, candidate_ids: Sequence[str],
                 texts: Mapping[str, str]) -> list[str]:
    """Rank a specific candidate subset (by id) using a shared text lookup."""
    subset = {cid: texts.get(cid, "") for cid in candidate_ids}
    return rank(query_text, subset)
