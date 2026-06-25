"""R23 leakage audit: the diff->test result survives the densification/coverage
controls. Reads the committed audit result.

Pins the audit claims in docs/ablation-diff2test-retrieval.md / BENCHMARK.md against
data/pilot/diff2test-audit-results.json. Pure JSON read; skips if absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "data" / "pilot" / "diff2test-audit-results.json"

pytestmark = pytest.mark.skipif(not RES.exists(), reason="R23 audit result not present")


def _d():
    return json.loads(RES.read_text(encoding="utf-8"))


def test_candidate_coverage_is_balanced():
    # The key threat: gold covered but negatives not. Coverage must be comparable.
    c = _d()["coverage"]
    assert c["negative_coverage_rate_mean"] >= 0.7 * c["gold_coverage_rate"]


def test_fair_r1_beats_random_among_covered():
    # Ranking ONLY among covered candidates (removing the covered-vs-uncovered cue),
    # the signal is real: fair R@1 well above random-among-covered.
    f = _d()["fair_R@1_among_covered"]
    assert f["value"] > 0.40
    assert f["value"] > 3.0 * f["random_among_covered_baseline"]


def test_repos_are_disjoint():
    assert _d()["repo_disjointness"]["forks_or_near_dupes"] is False
