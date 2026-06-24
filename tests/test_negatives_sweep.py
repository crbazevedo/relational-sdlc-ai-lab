"""R18 negatives lever: harder (same-repo) negatives beat more (bigger pool), and
the Tier-2 LoRA delta CI excludes zero. Reads the committed results.

Pins docs/ablation-negatives-sweep.md against the committed MPS-trained
negatives-sweep-results.json + the numpy bootstrap negatives-bootstrap-results.json.
Pure JSON read — no torch; skips cleanly if the artifacts are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

T2 = Path(__file__).resolve().parents[1] / "data" / "tier2"
SWEEP = T2 / "negatives-sweep-results.json"
BOOT = T2 / "negatives-bootstrap-results.json"

pytestmark = pytest.mark.skipif(not SWEEP.exists(), reason="R18 sweep result not present")


def _cells():
    d = json.loads(SWEEP.read_text(encoding="utf-8"))
    return {c["name"]: c for c in d["configs"]}, d


def test_frozen_anchor_reproduces_r16c():
    _, d = _cells()
    assert d["frozen"]["recall_at_k"]["1"] == pytest.approx(0.515, abs=2e-3)


def test_h1_more_random_negatives_is_flat():
    # b32/b48/b96 random pool: ΔR@1 stays in a tight ~0.12-0.13 band (no monotone
    # climb) — quantity is not the lever.
    cells, _ = _cells()
    randoms = [c["delta_r1"] for n, c in cells.items() if c.get("batching") == "random"]
    assert len(randoms) >= 3
    assert max(randoms) - min(randoms) < 0.03  # flat, not a climb


def test_h2_hard_negatives_beat_matched_random():
    # repo-homogeneous (hard) batching at pool 48 beats the matched random pool-48
    # cell — hardness, not quantity, is the lever.
    cells, _ = _cells()
    repo = next(c for n, c in cells.items() if c.get("batching") == "repo")
    b48_random = next(c for n, c in cells.items()
                      if c.get("batching") == "random" and c["batch"] == 48)
    assert repo["delta_r1"] > b48_random["delta_r1"]
    assert repo["delta_r1"] > 0.13  # new program best (> R16D's +0.126)


@pytest.mark.skipif(not BOOT.exists(), reason="R18 bootstrap result not present")
def test_tier2_delta_ci_excludes_zero_and_is_broad():
    d = json.loads(BOOT.read_text(encoding="utf-8"))
    for which in ("query_bootstrap", "repo_cluster_bootstrap"):
        assert d["ci95"][which]["delta_r1"][0] > 0.05, which  # lower bound well above 0
    assert d["repos_improved"] >= 30
    assert d["repos_regressed"] == 0
    assert d["per_query_flips"]["net"] > 100
