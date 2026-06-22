"""Q6 follow-up: code-aware embedding base on committed code2 caches (numpy only).

Honest by construction: it asserts the caches are well-formed and the metrics are
valid retrieval numbers in [0, 1] — it does NOT assert the code base wins. Whether
a code-embedding base beats the general substrate is the finding under test, not a
precondition. Skips entirely if the code2 caches are not committed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data" / "pilot"
EMB = PILOT / "embeddings"

CODE2_CACHES = ["code2-unixcoder.npz", "code2-stcodesearch.npz"]
ANY_CODE2 = any((EMB / c).exists() for c in CODE2_CACHES)

pytestmark = pytest.mark.skipif(
    not (ANY_CODE2 and (EMB / "minilm-l6-v2.npz").exists()),
    reason="code2 caches not present",
)

PRESENT = [c for c in CODE2_CACHES if (EMB / c).exists()]


def _ds():
    spec = importlib.util.spec_from_file_location(
        "run_crossrepo_ablation", PILOT / "run_crossrepo_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_pilot_crossrepo()[0]


def _raw(name):
    return np.load(EMB / name, allow_pickle=False)


def _vecs(name):
    d = _raw(name)
    return {str(i): d["vectors"][k].astype(np.float32) for k, i in enumerate(d["ids"])}


def _metrics(name):
    from relsdlc.tower import run_cosine_on_vecs
    return run_cosine_on_vecs(_ds(), _vecs(name))


@pytest.mark.parametrize("name", PRESENT)
def test_cache_shape_ok(name):
    d = _raw(name)
    assert d["ids"].shape[0] == d["vectors"].shape[0]
    assert d["vectors"].shape[0] > 0
    assert d["vectors"].shape[1] > 0
    assert d["vectors"].dtype == np.float16
    assert str(d["model"]) != ""


@pytest.mark.parametrize("name", PRESENT)
def test_metrics_in_unit_interval(name):
    m = _metrics(name)
    for v in m["recall_at_k"].values():
        assert 0.0 <= v <= 1.0
    assert 0.0 <= m["mrr"] <= 1.0
    assert 0.0 <= m["hard_negative_accuracy"] <= 1.0
    assert m["n_queries"] > 0


@pytest.mark.parametrize("name", PRESENT)
def test_report_vs_substrate(name, capsys):
    # Report only — do NOT assert the code base wins (honest).
    from relsdlc.tower import run_cosine_on_vecs
    ds = _ds()
    base = run_cosine_on_vecs(ds, _vecs("minilm-l6-v2.npz"))["recall_at_k"]["1"]
    code = _metrics(name)["recall_at_k"]["1"]
    with capsys.disabled():
        print(f"\n  {name}: R@1={code:.3f}  (substrate minilm R@1={base:.3f}, "
              f"delta={code - base:+.3f})")
    assert isinstance(code, float)
