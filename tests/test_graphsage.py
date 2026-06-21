"""Tests for the training-free graph aggregation probe (Track B).

Numpy-only. Runs on the committed pilot data + cached embeddings; every test
skips cleanly if those artifacts are absent (so a sparse checkout still passes).

We deliberately do NOT assert that graph aggregation BEATS plain cosine — this is
an exploratory probe and the honest finding may be that it does not. We assert the
mechanics: aggregation runs, is deterministic, returns unit vectors, the new
benchmark validates clean with >=1 positive per query, and metrics stay in [0, 1].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from relsdlc.graphsage import (
    augmented_vecs,
    build_typed_adjacency,
    graphsage_aggregate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data" / "pilot"
EMB = PILOT / "embeddings" / "minilm-l6-v2.npz"
EDGES = PILOT / "edges.jsonl"
MODIFIES = PILOT / "graph" / "modifies_edges.jsonl"
DIFF2TEST = PILOT / "benchmark" / "diff_to_affected_test.jsonl"


def _load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_embeddings():
    npz = np.load(EMB, allow_pickle=True)
    ids = [str(i) for i in npz["ids"]]
    vecs = np.asarray(npz["vectors"], dtype=np.float64)
    return {i: vecs[k] for k, i in enumerate(ids)}


def _load_edges():
    fixes = [(e["source"], e["target"]) for e in _load_jsonl(EDGES)
             if e.get("relation") == "fixes"]
    mods = [(e["source"], e["target"]) for e in _load_jsonl(MODIFIES)
            if e.get("relation") == "modifies"]
    return fixes, mods


pilot_present = pytest.mark.skipif(
    not (EMB.exists() and EDGES.exists() and MODIFIES.exists()),
    reason="pilot embeddings/graph not present",
)


# --- synthetic, dependency-free unit tests -----------------------------------

def test_build_typed_adjacency_is_symmetric_and_typed():
    adj = build_typed_adjacency(
        fixes_edges=[("pr:1", "iss:1")],
        modifies_edges=[("pr:1", "file:a"), ("pr:1", "test:t")],
    )
    assert "iss:1" in adj["pr:1"]["fixes"]
    assert "pr:1" in adj["iss:1"]["fixes"]          # reverse direction stored
    assert "file:a" in adj["pr:1"]["modifies"]
    assert "pr:1" in adj["file:a"]["modifies"]


def test_leakage_guard_drops_the_eval_edge():
    adj = build_typed_adjacency(
        fixes_edges=[("pr:1", "iss:1")],
        modifies_edges=[("pr:1", "test:t")],
        exclude_fixes_pairs=[("pr:1", "iss:1")],
        exclude_modifies_pairs=[("pr:1", "test:t")],
    )
    # Both edges removed -> the nodes have no neighbours at all.
    assert adj.get("pr:1", {}) == {} or all(not v for v in adj["pr:1"].values())
    assert "iss:1" not in adj or not adj["iss:1"].get("fixes")


def test_aggregation_returns_unit_vectors_and_is_deterministic():
    rng = np.random.default_rng(0)
    node_vecs = {f"pr:{i}": rng.normal(size=8) for i in range(4)}
    fixes = [("pr:0", "iss:0")]
    mods = [("pr:0", "file:a"), ("pr:1", "file:a"), ("pr:2", "file:b")]
    a1 = augmented_vecs(node_vecs, fixes, mods, alpha=0.5, hops=2)
    a2 = augmented_vecs(node_vecs, fixes, mods, alpha=0.5, hops=2)
    assert set(a1) == set(a2)
    for k in a1:
        assert np.allclose(a1[k], a2[k])          # deterministic
        assert abs(np.linalg.norm(a1[k]) - 1.0) < 1e-9   # unit vectors
    # the file node (no own feature) is materialized purely from structure
    assert "file:a" in a1


def test_node_without_own_feature_gets_a_structural_embedding():
    node_vecs = {"pr:0": np.array([1.0, 0.0, 0.0]), "pr:1": np.array([0.0, 1.0, 0.0])}
    mods = [("pr:0", "file:a"), ("pr:1", "file:a")]
    aug = augmented_vecs(node_vecs, [], mods, alpha=0.5, hops=1)
    assert "file:a" in aug
    assert abs(np.linalg.norm(aug["file:a"]) - 1.0) < 1e-9


# --- pilot-data integration tests --------------------------------------------

@pilot_present
def test_pilot_aggregation_runs_and_is_unit_normed():
    vecs = _load_embeddings()
    fixes, mods = _load_edges()
    aug = augmented_vecs(vecs, fixes, mods, alpha=0.5, hops=2)
    assert len(aug) >= len(vecs)        # includes file/test nodes
    sample = list(aug)[:50]
    for k in sample:
        assert abs(np.linalg.norm(aug[k]) - 1.0) < 1e-6


@pilot_present
def test_diff2test_benchmark_validates_clean_with_positives():
    if not DIFF2TEST.exists():
        pytest.skip("diff_to_affected_test benchmark not present")
    from relsdlc.validate import validate_benchmark

    queries = _load_jsonl(DIFF2TEST)
    assert queries, "benchmark is empty"
    for q in queries:
        assert q["task"] == "diff_to_affected_test"
        assert len(q["relevant"]) >= 1                       # >=1 positive
        cands = set(q["candidates"])
        assert all(r in cands for r in q["relevant"])        # positives in pool
        findings = validate_benchmark(q, "diff_to_affected_test.jsonl")
        errors = [f for f in findings if f.severity == "error"]
        assert not errors, errors


@pilot_present
def test_pilot_metrics_in_unit_interval():
    """Score embedder-cosine vs graph-aug-cosine on diff->test; metrics in [0,1].

    No assertion that graph aggregation wins — exploratory probe, honest reporting.
    """
    if not DIFF2TEST.exists():
        pytest.skip("diff_to_affected_test benchmark not present")
    from relsdlc.synth import Query, SynthDataset
    from relsdlc.tower import _eval

    vecs = _load_embeddings()
    dim = len(next(iter(vecs.values())))
    fixes, mods = _load_edges()
    raw = _load_jsonl(DIFF2TEST)[:40]  # subset keeps the test fast
    queries = [Query(query_id=q["query_id"], query_record=q["query_record"],
                     candidates=q["candidates"], relevant=q["relevant"],
                     hard_negatives=q.get("hard_negatives", []), split="test")
               for q in raw]
    ds = SynthDataset(artifacts=[], fixes=[], queries=queries, params={})

    cand_ids = set()
    for q in queries:
        cand_ids.add(q.query_record)
        cand_ids.update(q.candidates)
    gold_mods = [(q.query_record, r) for q in queries for r in q.relevant]
    aug = augmented_vecs(vecs, fixes, mods, alpha=0.5, hops=2,
                         exclude_modifies_pairs=gold_mods,
                         all_node_ids=set(vecs) | cand_ids)

    zero = np.zeros(dim)
    base = {**vecs}
    aug_full = {**aug}
    for c in cand_ids:
        base.setdefault(c, zero)
        aug_full.setdefault(c, zero)

    for scored in (_eval(ds, base, (1, 5, 10), project=None),
                   _eval(ds, aug_full, (1, 5, 10), project=None)):
        for v in scored["recall_at_k"].values():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= scored["mrr"] <= 1.0
        assert 0.0 <= scored["hard_negative_accuracy"] <= 1.0
