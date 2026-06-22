"""LoRA-at-Tier-2: the win holds at dense ~80 repos (reads committed result)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "data" / "tier2" / "tier2-finetune-results.json"

pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="tier2 finetune result not present")


def _load():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_lora_beats_frozen_at_tier2():
    d = _load()
    assert d["tuned"]["recall_at_k"]["1"] > d["frozen"]["recall_at_k"]["1"]
    assert d["tuned"]["mrr"] > d["frozen"]["mrr"]
    assert d["delta_r1"] > 0
