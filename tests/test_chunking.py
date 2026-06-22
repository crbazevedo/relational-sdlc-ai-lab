"""Chunking module unit tests + the chunked-ablation finding (FirstP ≥ MaxP)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from relsdlc.chunking import chunk_text, firstp, maxp, meanp, sump, rank, whole_doc_mean


def test_chunk_short_text_is_one_chunk():
    assert chunk_text("short", size=512) == ["short"]
    assert chunk_text("", size=512) == [""]


def test_chunk_long_text_windows_bounded_and_overlapping():
    text = "x" * 1000
    cs = chunk_text(text, size=256, overlap=0.2)
    assert len(cs) > 1
    assert all(len(c) <= 256 for c in cs)
    # step = 256*0.8 ≈ 204, so windows overlap (more than ceil(1000/256)=4 chunks).
    assert len(cs) >= 5


def test_aggregators_on_toy_vectors():
    q = np.array([1.0, 0.0])
    chunks = np.array([[0.0, 1.0], [1.0, 0.0], [0.7071, 0.7071]])  # cos = 0, 1, 0.707
    assert maxp(q, chunks) == pytest.approx(1.0)
    assert firstp(q, chunks) == pytest.approx(0.0)
    assert sump(q, chunks) == pytest.approx(0.0 + 1.0 + 0.7071, abs=1e-3)
    assert meanp(q, chunks) == pytest.approx((0.0 + 1.0 + 0.7071) / 3, abs=1e-3)
    assert 0.0 <= whole_doc_mean(q, chunks) <= 1.0


def test_rank_orders_by_aggregator():
    q = np.array([1.0, 0.0])
    cands = {
        "good": np.array([[1.0, 0.0]]),           # maxp 1.0
        "bad": np.array([[0.0, 1.0], [0.1, 0.99]]),  # maxp ~0.1
    }
    assert rank(q, cands, maxp)[0] == "good"


# --- the chunked ablation finding (reads committed results) ---

RESULTS = Path(__file__).resolve().parents[1] / "data" / "full" / "chunk-results.json"


@pytest.mark.skipif(not RESULTS.exists(), reason="chunk results not present")
def test_firstp_not_beaten_by_maxp_for_issue_to_pr():
    r = json.loads(RESULTS.read_text())["results"]
    # The documented finding: FirstP is not beaten by MaxP at any chunk size (front-loaded signal).
    for size in (256, 512, 1024):
        fp = r.get(f"s{size}:firstp", {}).get("recall_at_k", {}).get("1")
        mp = r.get(f"s{size}:maxp", {}).get("recall_at_k", {}).get("1")
        if fp is not None and mp is not None:
            assert fp >= mp - 1e-9
    # SumP is length-biased and collapses.
    assert r["s512:sump"]["recall_at_k"]["1"] < r["s512:firstp"]["recall_at_k"]["1"]
