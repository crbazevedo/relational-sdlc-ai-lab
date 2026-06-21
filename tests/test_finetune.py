"""Track A: the LoRA-tuned embedder beats the frozen one cross-repo.

Runs on the committed embedding caches with numpy only (no torch), so it is
reproducible in CI. Asserts the qualitative win (tuned > frozen), not exact floats.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data" / "pilot"
FROZEN = PILOT / "embeddings" / "minilm-l6-v2.npz"
TUNED = PILOT / "embeddings" / "minilm-lora.npz"

pytestmark = pytest.mark.skipif(
    not (FROZEN.exists() and TUNED.exists() and (PILOT / "records.jsonl").exists()),
    reason="embedding caches or pilot snapshot not present",
)


def _crossrepo():
    spec = importlib.util.spec_from_file_location(
        "run_crossrepo_ablation", PILOT / "run_crossrepo_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(path):
    data = np.load(path, allow_pickle=False)
    ids = [str(i) for i in data["ids"]]
    vectors = data["vectors"].astype(np.float32)
    return {i: vectors[k] for k, i in enumerate(ids)}


def test_lora_tuned_beats_frozen_cross_repo():
    from relsdlc.tower import run_cosine_on_vecs
    ds, _ = _crossrepo().load_pilot_crossrepo()
    frozen = run_cosine_on_vecs(ds, _load(FROZEN))
    tuned = run_cosine_on_vecs(ds, _load(TUNED))
    # The headline of Track A: relation-loss LoRA improves cross-repo retrieval.
    assert tuned["recall_at_k"]["1"] > frozen["recall_at_k"]["1"]
    assert tuned["mrr"] > frozen["mrr"]


def test_tuned_cache_shape_and_provenance():
    data = np.load(TUNED, allow_pickle=False)
    assert data["vectors"].shape[1] == 384
    assert "lora" in str(data["model"]).lower()
