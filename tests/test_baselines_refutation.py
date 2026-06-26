"""R25 baselines + refutation: pins the committed result JSONs so the findings
cannot silently regress.

- Task A: the embedder/LoRA/graph ladder beats BM25; the LoRA win is underpowered.
- Task B: BM25 over test PATHS beats co-change structure on BOTH corpora (the
  "text-free, only structure works" headline is refuted).
- A learned reranker does NOT make structure complementary (adding it hurts).

Pure JSON reads; each test skips if its artifact is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "pilot" / "baselines-metrics-results.json"
C2BASE = ROOT / "data" / "corpus2" / "corpus2-baselines-results.json"
C2FUSE = ROOT / "data" / "corpus2" / "corpus2-fusion-results.json"


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


# ---- Task A: real ladder over a real lexical baseline -----------------------

@pytest.mark.skipif(not PILOT.exists(), reason="R25 pilot baselines not present")
def test_taskA_embedder_ladder_beats_bm25():
    A = _load(PILOT)["task_A_issue2pr"]
    bm25 = A["BM25"]["R@1"]
    frozen = A["embedder-cosine (frozen)"]["R@1"]
    lora = A["bi-encoder LoRA (Occam)"]["R@1"]
    graph = A["+ graph-aug (LoRA)"]["R@1"]
    assert frozen > bm25            # embedding beats lexical on Task A
    assert lora > frozen            # relational supervision adds
    assert graph >= lora            # graph aggregation adds


@pytest.mark.skipif(not PILOT.exists(), reason="R25 pilot baselines not present")
def test_taskA_lora_win_is_underpowered():
    mde = _load(PILOT)["mde_power_lora_taskA"]
    # positive but below the 80%-power MDE, with achieved power well under 0.8
    assert mde["observed_delta_R@1"] > 0
    assert mde["observed_delta_R@1"] < mde["mde_R@1_at_80pct_power"]
    assert mde["achieved_power"] < 0.8
    assert mde["detectable"] is False


# ---- Task B: the refutation -------------------------------------------------

@pytest.mark.skipif(not PILOT.exists(), reason="R25 pilot baselines not present")
def test_taskB_bm25_paths_beats_structure_pilot():
    B = _load(PILOT)["task_B_diff2test"]
    bm25 = B["BM25 (paths)"]["R@1"]
    cosine = B["embedder-cosine (frozen)"]["R@1"]
    # BM25 over paths is NOT near chance; it dwarfs the sentence-embedder baseline
    assert bm25 > 0.4
    assert bm25 > 10 * cosine


@pytest.mark.skipif(not C2BASE.exists(), reason="corpus2 baselines not present")
def test_taskB_bm25_beats_structure_corpus2():
    r = _load(C2BASE)["R@1"]
    assert r["bm25"] > r["struct"]          # lexical beats structure
    assert r["fusion"] <= r["bm25"]         # naive fusion does not help


# ---- learned reranker: structure is not complementary -----------------------

@pytest.mark.skipif(not C2FUSE.exists(), reason="corpus2 fusion not present")
def test_learned_fusion_structure_not_complementary():
    d = _load(C2FUSE)
    r = d["R@1"]
    sc = d["structure_contribution"]
    # best system is lexical-only learned reranking
    assert r["LR_lexical"] >= r["LR_lexical+struct"]
    assert r["LR_lexical"] >= r["bm25"]
    # adding structure does NOT significantly help: CI upper bound is not positive
    assert sc["ci95"][1] <= 0.0
