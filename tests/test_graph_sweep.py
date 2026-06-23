"""R16E graph-lift sweep: the issue->PR lift is a robust plateau (1 hop suffices),
diff->test is structure-bound (reads the committed result).

Pins docs/ablation-graph-sweep.md against the committed
data/pilot/graph-sweep-results.json. Pure JSON read — no numpy, no torch; skips
cleanly if the sweep artifact is absent (sparse checkout still passes).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "data" / "pilot" / "graph-sweep-results.json"

pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="graph sweep result not present")

ALPHAS = ("0.0", "0.25", "0.5", "0.75", "1.0")


def _load():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _r1(d, task, feat, hops, a):
    return d["grid"][f"{task}/{feat}/h{hops}/a{a}"]["R@1"]


def test_alpha1_is_embedder_cosine_sanity_anchor():
    # alpha=1 keeps a text node's own vector, so graph-aug MUST equal
    # embedder-cosine on issue->PR — a harness sanity check.
    d = _load()
    for feat in ("frozen", "lora"):
        base = d["baselines"][f"issue_to_fixing_pr/{feat}"]["R@1"]
        for hops in d["hops"]:
            assert _r1(d, "issue_to_fixing_pr", feat, hops, "1.0") == pytest.approx(base, abs=1e-4)


def test_issue2pr_lift_is_a_plateau_not_knife_edge():
    # The lift is positive across the WHOLE non-trivial alpha range [0, 0.75], on
    # both feature sets — it is not a tuned spike that only appears at 0.5.
    d = _load()
    for feat in ("frozen", "lora"):
        base = d["baselines"][f"issue_to_fixing_pr/{feat}"]["R@1"]
        for a in ("0.0", "0.25", "0.5", "0.75"):
            assert _r1(d, "issue_to_fixing_pr", feat, 1, a) > base, (feat, a)


def test_one_hop_suffices():
    # hops=1 == hops=2 everywhere on issue->PR: the 2nd hop changes no ranking, so
    # R11B's headline (0.690) is a pure 1-hop neighbourhood effect.
    d = _load()
    for feat in ("frozen", "lora"):
        for a in ALPHAS:
            assert _r1(d, "issue_to_fixing_pr", feat, 1, a) == _r1(d, "issue_to_fixing_pr", feat, 2, a)


def test_diff2test_is_structure_bound():
    # Flat at every (alpha, hops); the isolation diagnostic explains why — ~47% of
    # gold tests are degree-0 after the leakage guard, an ~60% reachable ceiling
    # that no feature or hop count can exceed.
    d = _load()
    for feat in ("frozen", "lora"):
        for hops in d["hops"]:
            for a in ALPHAS:
                assert _r1(d, "diff_to_affected_test", feat, hops, a) <= 0.02
    iso = d["diff2test_isolation"]
    assert iso["pair_isolation_rate"] > 0.40
    assert iso["query_reachable_ceiling"] < 0.65
