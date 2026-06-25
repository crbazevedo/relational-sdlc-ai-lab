"""R20: densifying the modifies graph activates diff->test retrieval (R@1
0.009 -> ~0.30). Reads the committed result.

Pins docs/ablation-diff2test-retrieval.md against the committed
data/pilot/diff2test-dense-results.json. Pure JSON read — no torch; skips if absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "data" / "pilot" / "diff2test-dense-results.json"

pytestmark = pytest.mark.skipif(not RES.exists(), reason="R20 result not present")


def _d():
    return json.loads(RES.read_text(encoding="utf-8"))


def test_original_edges_reproduce_the_r16e_floor():
    # The clean anchor: with original edges, the same scorer reproduces R16E's 0.009.
    d = _d()
    assert d["graph_aug_orig_R16E"]["R@1"] == pytest.approx(0.009, abs=2e-3)
    assert d["embedder_cosine"]["R@1"] == pytest.approx(0.009, abs=2e-3)


def test_dense_graph_lifts_retrieval_off_the_floor():
    # The whole point: dense edges move diff->test from chance to a real signal.
    d = _d()
    assert d["graph_aug_dense_R20"]["R@1"] > 0.25
    assert d["graph_aug_dense_R20"]["MRR"] > 0.40
    # ...and the gain is purely the density (same scorer, same guard).
    assert d["graph_aug_dense_R20"]["R@1"] > 20 * d["graph_aug_orig_R16E"]["R@1"]


def test_reachability_rose_with_density():
    d = _d()
    assert d["reachable_orig"] == pytest.approx(0.598, abs=1e-2)
    assert d["reachable_dense"] > 0.85
