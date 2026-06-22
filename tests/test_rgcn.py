"""Learned R-GCN cache sanity + the honest finding (numpy only)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data" / "pilot"
RGCN = PILOT / "embeddings" / "rgcn-frozen.npz"
FROZEN = PILOT / "embeddings" / "minilm-l6-v2.npz"

pytestmark = pytest.mark.skipif(
    not (RGCN.exists() and FROZEN.exists()), reason="rgcn cache not present")


def _ds():
    spec = importlib.util.spec_from_file_location(
        "run_crossrepo_ablation", PILOT / "run_crossrepo_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_pilot_crossrepo()[0]


def _vecs(p):
    d = np.load(p, allow_pickle=False)
    return {str(i): d["vectors"][k].astype(np.float32) for k, i in enumerate(d["ids"])}


def test_rgcn_cache_is_unit_normish():
    d = np.load(RGCN, allow_pickle=False)
    norms = np.linalg.norm(d["vectors"].astype(np.float32), axis=1)
    assert d["vectors"].shape[1] == 384
    assert float(norms.mean()) > 0.9  # unit-normalized (float16 rounding aside)


def test_free_aggregation_beats_frozen_and_rgcn_metrics_valid():
    # Documents the honest finding: parameter-free aggregation >= frozen;
    # the learned R-GCN is NOT asserted to win (it does not at pilot scale).
    from relsdlc.graphsage import augmented_vecs
    from relsdlc.tower import run_cosine_on_vecs
    import json
    ds = _ds()
    frozen = _vecs(FROZEN)
    modifies = [(e["source"], e["target"]) for e in
                (json.loads(l) for l in (PILOT / "graph" / "modifies_edges.jsonl")
                 .read_text().splitlines() if l.strip())]
    free = run_cosine_on_vecs(ds, augmented_vecs(frozen, ds.fixes, modifies,
                              alpha=0.5, hops=2, exclude_fixes_pairs=ds.fixes))
    rgcn = run_cosine_on_vecs(ds, _vecs(RGCN))
    assert free["recall_at_k"]["1"] >= run_cosine_on_vecs(ds, frozen)["recall_at_k"]["1"]
    for v in rgcn["recall_at_k"].values():
        assert 0.0 <= v <= 1.0
