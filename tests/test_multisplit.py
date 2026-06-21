"""Track D: the LoRA win is robust across cross-repo splits.

Reads the committed multi-split results (a torch-trained snapshot) and asserts the
qualitative robustness claim — positive delta on every split — so the headline is
guarded without needing torch in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "data" / "pilot" / "multisplit-results.json"

pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="multi-split results not present")


def _load():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_lora_delta_positive_on_every_split():
    d = _load()
    assert len(d["per_split"]) >= 3
    for row in d["per_split"]:
        assert row["delta_r1"] > 0, f"split {row['split']} R@1 delta not positive"
        assert row["delta_mrr"] > 0, f"split {row['split']} MRR delta not positive"


def test_mean_delta_exceeds_one_std():
    agg = _load()["aggregate"]
    # A robust effect: the mean improvement is larger than its spread.
    assert agg["delta_r1"]["mean"] > agg["delta_r1"]["std"]
    assert agg["delta_mrr"]["mean"] > agg["delta_mrr"]["std"]


def test_tuned_mean_beats_frozen_mean():
    agg = _load()["aggregate"]
    assert agg["tuned_r1"]["mean"] > agg["frozen_r1"]["mean"]
    assert agg["tuned_mrr"]["mean"] > agg["frozen_mrr"]["mean"]
