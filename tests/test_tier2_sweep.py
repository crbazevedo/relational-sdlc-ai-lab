"""R16D LoRA sweep: rank is saturated, harder negatives win (reads committed result).

Pins the findings of docs/ablation-lora-sweep.md against the committed
tier2-sweep-results.json: every tuned config beats frozen, the rank axis is flat
(within noise), and the harder-negatives config (r16-b48) is the best.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "data" / "tier2" / "tier2-sweep-results.json"

pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="tier2 sweep result not present")


def _load():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _by_name(d):
    return {c["name"]: c for c in d["configs"]}


def test_every_tuned_config_beats_frozen():
    d = _load()
    frozen = d["frozen"]["recall_at_k"]["1"]
    tuned = [c for c in d["configs"] if c["name"] != "frozen"]
    assert tuned, "no tuned configs recorded"
    for c in tuned:
        assert c["recall_at_k"]["1"] > frozen, c["name"]
        assert c["delta_r1"] > 0, c["name"]


def test_rank_axis_is_saturated():
    # r8 / r16 / r32 at the same batch sit within a tight noise band — more LoRA
    # capacity does not help (the lever is not parameters).
    b = _by_name(_load())
    ranks = [b[n]["recall_at_k"]["1"] for n in ("r8-b32 (baseline)", "r16-b32", "r32-b32")]
    assert max(ranks) - min(ranks) < 0.02, ranks


def test_harder_negatives_is_best():
    # Larger in-batch pool (48 vs 32) at fixed rank is the one axis that moves.
    b = _by_name(_load())
    best = max((c for c in _load()["configs"] if c["name"] != "frozen"),
              key=lambda c: c["recall_at_k"]["1"])
    assert best["name"].startswith("r16-b48"), best["name"]
    assert b["r16-b48 (harder negs)"]["recall_at_k"]["1"] > b["r16-b32"]["recall_at_k"]["1"]
