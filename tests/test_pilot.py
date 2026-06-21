"""The P1 real-data pilot: it validates, and the real ablation runs honestly.

These assertions are deliberately conservative — they pin what is true on real
data (the pilot validates; IDF is a strong baseline; metrics are well-formed),
NOT that the relation model wins (it does not, and that is the documented result).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data" / "pilot"

pytestmark = pytest.mark.skipif(
    not (PILOT / "records.jsonl").exists(),
    reason="pilot snapshot not present",
)


def _load_run_real_ablation():
    spec = importlib.util.spec_from_file_location(
        "run_real_ablation", PILOT / "run_real_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pilot_validates_clean():
    from relsdlc.validate import validate_path
    report = validate_path(PILOT)
    assert report.ok, [str(f) for f in report.errors]
    assert report.n_records > 100
    assert report.n_benchmarks > 50


def test_real_ablation_runs_and_is_wellformed():
    mod = _load_run_real_ablation()
    from relsdlc.model import run_ablation
    ds = mod.load_pilot()
    report = run_ablation(ds, seed=0, min_df=mod.MIN_DF)
    assert report["n_test_queries"] > 0
    for m in report["systems"].values():
        for v in m["recall_at_k"].values():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0


def test_idf_is_a_strong_baseline_on_real_data():
    mod = _load_run_real_ablation()
    from relsdlc.model import run_ablation
    report = run_ablation(mod.load_pilot(), seed=0, min_df=mod.MIN_DF)
    idf = report["systems"]["idf-cosine"]["recall_at_k"]["1"]
    vanilla = report["systems"]["vanilla-tf-cosine"]["recall_at_k"]["1"]
    # IDF reliably helps real issue->PR retrieval; the diagonal relation model
    # is NOT asserted to win (it ties IDF — see docs/ablation-real.md).
    assert idf >= vanilla


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, PILOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_crossrepo_dataset_splits_by_repo_and_is_wellformed():
    mod = _load_module("run_crossrepo_ablation")
    from relsdlc.model import run_ablation
    ds, meta = mod.load_pilot_crossrepo()
    # Train and test repos are disjoint and both populated.
    assert set(meta["train_repos"]).isdisjoint(meta["test_repos"])
    assert meta["train_repos"] and meta["test_repos"]
    report = run_ablation(ds, seed=0, min_df=mod.MIN_DF)
    assert report["n_test_queries"] > 0
    for m in report["systems"].values():
        for v in m["recall_at_k"].values():
            assert 0.0 <= v <= 1.0
