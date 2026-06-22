"""The real Tier-2 dataset (~150 repos): it validates, is substantially bigger
than the 55-repo scale entry, and the robust finding (IDF beats vanilla on real
issue->fixing-PR retrieval) holds at Tier-2 scale.

These assertions are deliberately conservative — they pin what is true on the
larger real snapshot (it validates; it is substantially larger than scale; it has
more repos than scale; IDF R@1 >= vanilla R@1), NOT that the diagonal relation
model wins (it ties IDF — the documented result). Skipped when the live-fetch
snapshot is absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
TIER2 = REPO_ROOT / "data" / "tier2"
SCALE = REPO_ROOT / "data" / "scale"

pytestmark = pytest.mark.skipif(
    not (TIER2 / "records.jsonl").exists(),
    reason="tier2 snapshot not present",
)


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, TIER2 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Split on "\n" only — a record body can contain Unicode line separators (U+2028,
# NEL, \r, vertical tab, form feed) that str.splitlines() would split on, breaking
# a JSON line mid-string. Mirrors the production readers (see test_jsonl_robustness).
def _n_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").split("\n") if line.strip())


def _n_repos(path: Path) -> int:
    import json
    repos = set()
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rid = json.loads(line)["id"]
            # "gh-t2:owner/repo:type:N" / "gh:owner/repo:type:N" -> "owner/repo"
            parts = rid.split(":")
            if len(parts) > 1:
                repos.add(parts[1])
    return len(repos)


def test_tier2_validates_clean():
    from relsdlc.validate import validate_path
    report = validate_path(TIER2)
    assert report.ok, [str(f) for f in report.errors]
    assert report.n_records > 100
    assert report.n_benchmarks > 50


def test_tier2_is_substantially_larger_than_scale():
    tier2_records = _n_lines(TIER2 / "records.jsonl")
    assert tier2_records > 100
    if (SCALE / "records.jsonl").exists():
        scale_records = _n_lines(SCALE / "records.jsonl")
        # "Substantially more" — at least 1.25x the scale entry's record count.
        assert tier2_records >= int(scale_records * 1.25), (
            f"tier2 records ({tier2_records}) not substantially larger than "
            f"scale ({scale_records})"
        )


def test_tier2_has_more_repos_than_scale():
    tier2_repos = _n_repos(TIER2 / "records.jsonl")
    assert tier2_repos > 55
    if (SCALE / "records.jsonl").exists():
        scale_repos = _n_repos(SCALE / "records.jsonl")
        assert tier2_repos > scale_repos, (
            f"tier2 repos ({tier2_repos}) not more than scale ({scale_repos})"
        )


def test_tier2_records_are_id_disjoint_from_scale():
    import json
    tier2_ids = {json.loads(l)["id"]
                 for l in (TIER2 / "records.jsonl").read_text(encoding="utf-8").split("\n")
                 if l.strip()}
    # The gh-t2: namespace must keep tier2 disjoint from the gh: scale snapshot.
    assert all(rid.startswith("gh-t2:") for rid in tier2_ids)
    if (SCALE / "records.jsonl").exists():
        scale_ids = {json.loads(l)["id"]
                     for l in (SCALE / "records.jsonl").read_text(encoding="utf-8").split("\n")
                     if l.strip()}
        assert tier2_ids.isdisjoint(scale_ids)


def test_tier2_ablation_runs_and_is_wellformed():
    mod = _load_module("run_tier2_ablation")
    from relsdlc.model import run_ablation
    ds, meta = mod.load_tier2_crossrepo()
    assert set(meta["train_repos"]).isdisjoint(meta["test_repos"])
    assert meta["train_repos"] and meta["test_repos"]
    # include_metric=False: at Tier-2 the dense diagonal-metric triplet stacks
    # blow up memory/time; its "ties IDF" finding is already established on
    # pilot+scale, so the Tier-2 baseline is vanilla+IDF only (see docs).
    report = run_ablation(ds, seed=0, min_df=mod.MIN_DF, include_metric=False)
    assert report["n_test_queries"] > 0
    for m in report["systems"].values():
        for v in m["recall_at_k"].values():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0


def test_load_tier2_crossrepo_is_importable():
    # Central's LoRA-at-Tier-2 run imports this function — pin its presence.
    mod = _load_module("run_tier2_ablation")
    assert callable(mod.load_tier2_crossrepo)


def test_idf_beats_vanilla_at_tier2():
    mod = _load_module("run_tier2_ablation")
    from relsdlc.model import run_ablation
    ds, _ = mod.load_tier2_crossrepo()
    report = run_ablation(ds, seed=0, min_df=mod.MIN_DF, include_metric=False)
    idf = report["systems"]["idf-cosine"]["recall_at_k"]["1"]
    vanilla = report["systems"]["vanilla-tf-cosine"]["recall_at_k"]["1"]
    # The robust finding: unsupervised IDF reliably helps real issue->PR
    # retrieval. The diagonal relation model is NOT asserted to win (it ties
    # IDF — see docs/tier2-dataset.md and docs/scale-dataset.md).
    assert idf >= vanilla
