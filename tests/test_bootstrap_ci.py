"""R17a CI hardening: the LoRA win on the headline split has 95% bootstrap CIs
that exclude zero (both query- and repo-cluster), and the per-repo decomposition
is broad-but-not-uniform (5/8 improve). Reads the committed result.

Pins docs/ablation-bootstrap-ci.md against the committed
data/pilot/bootstrap-ci-results.json. Pure JSON read — no numpy/torch; skips
cleanly if the artifact is absent (sparse checkout still passes).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "data" / "pilot" / "bootstrap-ci-results.json"

pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="bootstrap CI result not present")


def _load():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_point_reproduces_committed_headline():
    # The point estimate IS the committed finetune card (the script asserts 1e-9;
    # this pins the rounded headline so a drift is caught in CI too).
    d = _load()
    assert d["point"]["frozen"]["r1"] == pytest.approx(0.5920, abs=1e-3)
    assert d["point"]["lora"]["r1"] == pytest.approx(0.6552, abs=1e-3)
    assert d["point"]["delta_r1"] == pytest.approx(0.0632, abs=1e-3)


def test_both_bootstrap_cis_exclude_zero():
    # The whole point of the wave: the LoRA delta is positive with 95% intervals
    # that exclude zero under BOTH query resampling and repo-cluster resampling.
    d = _load()
    for which in ("query_bootstrap", "repo_cluster_bootstrap"):
        ci = d["ci95"][which]["delta_r1"]
        assert ci[0] > 0, (which, ci)
        ci_mrr = d["ci95"][which]["delta_mrr"]
        assert ci_mrr[0] > 0, (which, ci_mrr)


def test_repo_cluster_overwhelmingly_positive():
    d = _load()
    assert d["ci95"]["repo_cluster_bootstrap"]["frac_resamples_positive_r1"] > 0.95


def test_win_is_broad_but_not_uniform():
    # 5/8 repos improve, with a couple of small regressions — the honest nuance the
    # single number hid. Guards against an over-strong "uniform win" reading.
    d = _load()
    assert d["repos_improved"] == 5
    assert d["repos_regressed"] == 2
    assert d["per_query_flips"]["rank1_gains"] > d["per_query_flips"]["rank1_losses"]
    assert d["per_query_flips"]["net"] == 11
