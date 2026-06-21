"""Metric definitions: Recall@K, MRR, hard-negative accuracy."""

from __future__ import annotations

import math

from relsdlc.metrics import (
    RetrievalResult,
    evaluate,
    hard_negative_accuracy,
    mrr,
    recall_at_k,
)


def test_recall_at_k_basic():
    r = RetrievalResult.of(["a", "b", "c", "d"], relevant=["c"])
    assert recall_at_k([r], 1) == 0.0
    assert recall_at_k([r], 3) == 1.0


def test_recall_at_k_multiple_relevant():
    r = RetrievalResult.of(["a", "b", "c", "d"], relevant=["a", "d"])
    assert recall_at_k([r], 2) == 0.5  # only 'a' in top-2 of 2 relevant
    assert recall_at_k([r], 4) == 1.0


def test_mrr():
    r1 = RetrievalResult.of(["a", "b"], relevant=["a"])  # rank 1 -> 1.0
    r2 = RetrievalResult.of(["a", "b"], relevant=["b"])  # rank 2 -> 0.5
    assert math.isclose(mrr([r1, r2]), 0.75)


def test_mrr_no_hit_is_zero():
    r = RetrievalResult.of(["a", "b"], relevant=["z"])
    assert mrr([r]) == 0.0


def test_hard_negative_accuracy():
    win = RetrievalResult.of(["pos", "neg"], relevant=["pos"], hard_negatives=["neg"])
    lose = RetrievalResult.of(["neg", "pos"], relevant=["pos"], hard_negatives=["neg"])
    assert hard_negative_accuracy([win, lose]) == 0.5


def test_hard_negative_accuracy_ignores_queries_without_negatives():
    no_neg = RetrievalResult.of(["a"], relevant=["a"])
    win = RetrievalResult.of(["pos", "neg"], relevant=["pos"], hard_negatives=["neg"])
    # no_neg is excluded -> accuracy is over the single hard-neg query.
    assert hard_negative_accuracy([no_neg, win]) == 1.0


def test_evaluate_bundle_shape():
    r = RetrievalResult.of(["a", "b"], relevant=["a"], hard_negatives=["b"])
    out = evaluate([r], ks=(1, 5))
    assert out["n_queries"] == 1
    assert set(out["recall_at_k"]) == {"1", "5"}
    assert 0.0 <= out["mrr"] <= 1.0
