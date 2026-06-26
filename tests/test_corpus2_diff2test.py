"""R24 external validity: the diff->test result replicates on the independent
TS/JS corpus (corpus2).

Pins the replication claims in docs/ablation-corpus2-replication.md / BENCHMARK.md /
the paper against data/corpus2/corpus2-diff2test-results.json. Pure JSON read; skips
if absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "data" / "corpus2" / "corpus2-diff2test-results.json"

pytestmark = pytest.mark.skipif(not RES.exists(), reason="R24 corpus2 result not present")


def _d():
    return json.loads(RES.read_text(encoding="utf-8"))


def test_is_an_independent_corpus():
    # Disjoint, different-language ecosystem (not the Python pilot).
    d = _d()
    assert "TS/JS" in d["corpus"]
    assert len(d["dense_repos"]) == 6
    assert d["n_queries"] > 500


def test_candidate_coverage_is_balanced():
    # Same threat as R23: gold must not be far more covered than negatives.
    s = _d()["structure"]
    assert s["negative_coverage"] >= 0.7 * s["gold_reachable"]


def test_fair_r1_replicates_real_discrimination():
    # The clean cross-corpus signal: fair R@1 well above random-among-covered,
    # and close to the pilot's 5.7x (here ~5.5x). Pool-/reachability-invariant.
    d = _d()
    lift = d["fair_R@1_among_covered"] / d["random_among_covered"]
    assert lift > 3.0           # real discrimination (pilot 5.7x)
    assert lift > 0.7 * d["pilot_reference"]["fair_x_random"]   # near-identical to pilot


def test_structure_beats_text():
    # End-to-end graph-aug retrieval clears the text-free baseline (and 0.25 floor).
    d = _d()
    assert d["graph_aug_asof_R@1"] > 0.25
    assert d["graph_aug_asof_R@1"] > d["embedder_cosine_R@1_textfree_baseline"]
