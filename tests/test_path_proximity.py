"""R27 static path-proximity baseline: pins the committed result so the
"structure loses to a static change-proximity heuristic too" finding cannot
silently regress.

Pure JSON read; skips if absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "data" / "pilot" / "path-proximity-results.json"

pytestmark = pytest.mark.skipif(not RES.exists(), reason="R27 path-proximity result not present")

STRUCTURE_R1 = 0.429  # co-change structure, release-honest (diff2test-strict-results.json)


def _d():
    return json.loads(RES.read_text(encoding="utf-8"))


def test_path_proximity_beats_structure():
    d = _d()
    assert d["path_proximity"]["R@1"] > STRUCTURE_R1


def test_best_static_dominates_structure():
    d = _d()
    assert d["best_static_proximity+bm25"]["R@1"] > STRUCTURE_R1
    # the combined static system is the strongest static signal
    assert d["best_static_proximity+bm25"]["R@1"] >= d["path_proximity"]["R@1"]


def test_harness_is_the_112_query_pilot_pool():
    d = _d()
    assert d["n_queries"] == 112
    assert d["source_coverage"] > 0.8
