"""R17b: real co-change density cracks the diff->test structural ceiling.

Pins docs/ablation-diff2test-density.md against the committed deterministic
recompute (data/pilot/diff2test-density-results.json), which replays from the
committed co-change snapshot with no token. Pure JSON read — no numpy/torch;
skips cleanly if the artifact is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "data" / "pilot" / "diff2test-density-results.json"

pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="diff2test density result not present")


def _load():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_density_lifts_the_reachable_ceiling_far_above_r16e():
    # R16E capped reachable recall at 59.8%; real co-change lifts it well past 0.9.
    d = _load()
    assert d["R16E_baseline"]["reachable_ceiling"] == pytest.approx(0.5982, abs=1e-3)
    assert d["strict_nongold"]["reachable_ceiling"] > 0.90
    assert d["strict_nongold"]["reachable_ceiling"] > d["R16E_baseline"]["reachable_ceiling"]


def test_isolation_collapses_under_real_history():
    # The 46.9% isolation was an ingest artefact: real history isolates almost none.
    d = _load()
    assert d["strict_nongold"]["pair_isolation_rate"] < 0.10


def test_gold_tests_are_heavily_cochanged():
    # The mechanism: gold test files are touched by many commits, not one.
    d = _load()
    assert d["cochange_depth"]["median_commits_per_gold_test"] >= 10
