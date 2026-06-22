"""Q6: base-model comparison on committed code-embedding caches (numpy only)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data" / "pilot"
EMB = PILOT / "embeddings"

pytestmark = pytest.mark.skipif(
    not ((EMB / "code-bge.npz").exists() and (EMB / "code-codebert.npz").exists()
         and (EMB / "minilm-l6-v2.npz").exists()),
    reason="code caches not present",
)


def _ds():
    spec = importlib.util.spec_from_file_location(
        "run_crossrepo_ablation", PILOT / "run_crossrepo_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_pilot_crossrepo()[0]


def _vecs(name):
    d = np.load(EMB / name, allow_pickle=False)
    return {str(i): d["vectors"][k].astype(np.float32) for k, i in enumerate(d["ids"])}


def _r1(name):
    from relsdlc.tower import run_cosine_on_vecs
    return run_cosine_on_vecs(_ds(), _vecs(name))["recall_at_k"]["1"]


def test_embedding_tuned_base_is_competitive():
    # A strong embedding-tuned general base ties/beats the MiniLM substrate.
    assert _r1("code-bge.npz") >= _r1("minilm-l6-v2.npz") - 0.02


def test_code_mlm_without_embed_tuning_collapses():
    # CodeBERT (MLM, not embedding-tuned) is far worse — the Q6 finding.
    assert _r1("code-codebert.npz") < 0.35
