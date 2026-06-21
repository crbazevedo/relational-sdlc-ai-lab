"""Embeddings cross-repo ablation, on the committed cache (numpy only, no torch).

Assertions are qualitative (orderings with margins) so they hold cross-platform
even if a metric's last decimal shifts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data" / "pilot"
EMB = PILOT / "embeddings" / "minilm-l6-v2.npz"

pytestmark = pytest.mark.skipif(
    not EMB.exists() or not (PILOT / "records.jsonl").exists(),
    reason="embedding cache or pilot snapshot not present",
)


def _crossrepo():
    spec = importlib.util.spec_from_file_location(
        "run_crossrepo_ablation", PILOT / "run_crossrepo_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_vecs():
    data = np.load(EMB, allow_pickle=False)
    ids = [str(i) for i in data["ids"]]
    vectors = data["vectors"].astype(np.float32)
    return {i: vectors[k] for k, i in enumerate(ids)}, vectors.shape[1]


def test_embeddings_cache_shape():
    vecs, dim = _load_vecs()
    assert dim == 384
    assert len(vecs) > 1000


def test_embedder_cosine_beats_idf_cross_repo():
    from relsdlc.model import run_ablation
    from relsdlc.tower import run_cosine_on_vecs
    ds, _ = _crossrepo().load_pilot_crossrepo()
    vecs, _ = _load_vecs()
    idf = run_ablation(ds, seed=0, min_df=3)["systems"]["idf-cosine"]["recall_at_k"]["1"]
    emb = run_cosine_on_vecs(ds, vecs)["recall_at_k"]["1"]
    # The headline of R8: pretrained embeddings generalize cross-repo; IDF does not.
    assert emb > idf + 0.05


def test_from_scratch_head_overfits_and_identity_init_is_safe():
    from relsdlc.tower import (
        run_cosine_on_vecs, run_relation_map_on_vecs, run_tower_on_vecs,
    )
    ds, _ = _crossrepo().load_pilot_crossrepo()
    vecs, dim = _load_vecs()
    cos = run_cosine_on_vecs(ds, vecs)["recall_at_k"]["1"]
    tower = run_tower_on_vecs(ds, vecs, dim, seed=0, d_proj=128, epochs=300,
                             lr=0.5, margin=0.2, weight_decay=1e-3)["recall_at_k"]["1"]
    relmap = run_relation_map_on_vecs(ds, vecs, dim, seed=0, epochs=300, lr=0.1,
                                      margin=0.1, decay=2e-2)["recall_at_k"]["1"]
    # A from-scratch head destroys the frozen geometry; identity-init does not.
    assert tower < cos
    assert relmap >= cos - 0.02
