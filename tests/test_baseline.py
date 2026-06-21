"""Baseline embedder: tokenization, deterministic hashing, cosine ranking."""

from __future__ import annotations

import zlib

from relsdlc.baseline import _bucket, cosine, embed, rank, tokenize


def test_tokenize_splits_identifiers():
    toks = tokenize("normalize_range parseTZ UTC-3")
    assert "normalize" in toks
    assert "range" in toks
    assert "parse" in toks
    assert "tz" in toks  # camelCase split + lowercased
    assert "3" in toks


def test_bucket_is_deterministic_crc32():
    # crc32 is stable across processes/platforms (unlike builtin hash() for str).
    assert _bucket("timezone", 1024) == zlib.crc32(b"timezone") % 1024


def test_embed_is_deterministic():
    assert embed("date filter timezone") == embed("date filter timezone")


def test_cosine_self_is_one():
    v = embed("normalize range timezone")
    assert abs(cosine(v, v) - 1.0) < 1e-9


def test_cosine_disjoint_is_zero():
    assert cosine(embed("alpha beta"), embed("gamma delta")) == 0.0


def test_rank_orders_by_overlap():
    candidates = {
        "match": "timezone normalization for UTC-3 date filter",
        "distractor": "currency report column formatting table",
    }
    ranked = rank("fix timezone UTC-3 date filter normalization", candidates)
    assert ranked[0] == "match"


def test_rank_is_deterministic_on_ties():
    # Two candidates with zero overlap tie at score 0; order must be stable (by id).
    candidates = {"b": "xxxx", "a": "yyyy"}
    assert rank("zzzz", candidates) == ["a", "b"]
