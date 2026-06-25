"""R22: the release-honest diff->test number (pure query + temporal as_of) and its
decomposition. Reads the committed result.

Pins BENCHMARK.md / docs/ablation-diff2test-retrieval.md against the committed
data/pilot/diff2test-strict-results.json. Pure JSON read; skips if absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "data" / "pilot" / "diff2test-strict-results.json"

pytestmark = pytest.mark.skipif(not RES.exists(), reason="R22 strict result not present")


def _d():
    return json.loads(RES.read_text(encoding="utf-8"))


def test_release_honest_number_is_strong():
    # The released diff->test number (pure query + as_of) clears 0.40 R@1 — a real
    # signal on a task text-cosine cannot do (0.009).
    d = _d()
    rel = d["pure_query_as_of_RELEASE"]
    assert rel["R@1"] > 0.40
    assert rel["MRR"] > 0.50


def test_decomposition_directions():
    # Pure query beats the alpha-blend (R21); and the as_of cut LOWERS R@1 vs using
    # all modifiers (future modifiers were leaking) — both directions honest.
    d = _d()
    allm = d["pure_query_all_modifiers"]["R@1"]
    asof = d["pure_query_as_of_RELEASE"]["R@1"]
    assert allm > d["r21_alpha_blend_all_modifiers"]["R@1"]  # query rep helps
    assert allm > asof                                        # as_of removes future leakage
    assert d["pure_query_as_of_RELEASE"]["reachable"] < d["pure_query_all_modifiers"]["reachable"]
