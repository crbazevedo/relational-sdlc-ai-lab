"""R19 multi-seed: the same-repo hardness gain is real (positive on every seed,
above MPS noise), not a single-seed fluke. Reads the committed result.

Pins docs/ablation-hardneg-confirm.md against the committed MPS-trained
hardneg-confirm-results.json. Pure JSON read — no torch; skips if absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "data" / "tier2" / "hardneg-confirm-results.json"

pytestmark = pytest.mark.skipif(not RES.exists(), reason="R19 result not present")


def _d():
    return json.loads(RES.read_text(encoding="utf-8"))


def test_frozen_anchor():
    assert _d()["frozen"]["recall_at_k"]["1"] == pytest.approx(0.515, abs=2e-3)


def test_hardness_positive_on_every_seed():
    # The paired repo-hard - random gap must be > 0 on every seed (not a single-seed
    # fluke) and clear of the ~+/-0.008 MPS run-to-run noise.
    p = _d()["paired_vs_random"]["repo-hard_minus_random"]
    assert p["all_positive"] is True
    assert min(p["per_seed"]) > 0.008
    assert p["mean"] > 0.015


def test_repohard_mean_beats_random_mean():
    agg = _d()["aggregate"]
    assert agg["repo-hard"]["delta_r1_mean"] > agg["random"]["delta_r1_mean"]
    # the gap is tight across seeds (this is the whole point — it is not noise)
    assert _d()["paired_vs_random"]["repo-hard_minus_random"]["std"] < 0.005
