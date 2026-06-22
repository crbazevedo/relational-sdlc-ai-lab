"""Provenance guards (audit response M1): mechanically verify the claims that
underpin the torch-trained results, WITHOUT needing torch.

The LoRA caches are committed snapshots CI cannot re-derive. These tests close the
trust gap by asserting — in CI, numpy/json only — the invariants the win depends on:
the cross-repo split is genuinely disjoint, the documented train-pair count is real,
and no test query draws a candidate from a train repo. If any of these breaks, the
'beats frozen cross-repo' claim is no longer trustworthy and CI fails loudly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data" / "pilot"

pytestmark = pytest.mark.skipif(
    not (PILOT / "records.jsonl").exists(), reason="pilot snapshot not present")


def _crossrepo():
    spec = importlib.util.spec_from_file_location(
        "run_crossrepo_ablation", PILOT / "run_crossrepo_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _repo_of(record_id: str) -> str:
    parts = record_id.split(":")
    return parts[1] if len(parts) > 1 else record_id


def test_train_and_test_repos_are_disjoint():
    ds, meta = _crossrepo().load_pilot_crossrepo()
    train, test = set(meta["train_repos"]), set(meta["test_repos"])
    assert train and test
    assert train.isdisjoint(test)


def test_no_test_query_draws_a_train_repo_candidate():
    ds, meta = _crossrepo().load_pilot_crossrepo()
    train = set(meta["train_repos"])
    by = ds.by_id()
    test_qs = [q for q in ds.queries if q.split == "test"]
    assert test_qs
    for q in test_qs:
        # The query and all its candidates must live in held-out (test) repos.
        assert _repo_of(q.query_record) not in train
        for cid in q.candidates:
            assert _repo_of(cid) not in train


def test_train_pairs_are_all_in_train_repos_and_count_is_documented():
    ds, meta = _crossrepo().load_pilot_crossrepo()
    train = set(meta["train_repos"])
    fix_pr = {iss: pr for pr, iss in ds.fixes}
    train_pairs = [iss for iss in (q.query_record for q in ds.queries if q.split == "train")
                   if fix_pr.get(iss)]
    # Every train pair's issue is in a train repo (no leakage into training).
    for iss in train_pairs:
        assert _repo_of(iss) in train
    # The documented training-pair count (ablation-finetune.md: 182) must hold.
    assert len(train_pairs) == 182
