"""The Tier-2-entry SCALE dataset: it validates, is bigger than the pilot, and
the robust finding (IDF beats vanilla on real issue->fixing-PR retrieval) holds.

These assertions are deliberately conservative — they pin what is true on the
larger real snapshot (it validates; it is substantially larger than the pilot;
IDF R@1 >= vanilla R@1), NOT that the diagonal relation model wins (it does not
at this scale either — it ties IDF, the documented result). Skipped when the
live-fetch snapshot is absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCALE = REPO_ROOT / "data" / "scale"
PILOT = REPO_ROOT / "data" / "pilot"

pytestmark = pytest.mark.skipif(
    not (SCALE / "records.jsonl").exists(),
    reason="scale snapshot not present",
)


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCALE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _n_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def test_scale_validates_clean():
    from relsdlc.validate import validate_path
    report = validate_path(SCALE)
    assert report.ok, [str(f) for f in report.errors]
    assert report.n_records > 100
    assert report.n_benchmarks > 50


def test_scale_is_substantially_larger_than_pilot():
    scale_records = _n_lines(SCALE / "records.jsonl")
    assert scale_records > 100
    if (PILOT / "records.jsonl").exists():
        pilot_records = _n_lines(PILOT / "records.jsonl")
        # "Substantially more" — at least 1.25x the pilot's record count.
        assert scale_records >= int(pilot_records * 1.25), (
            f"scale records ({scale_records}) not substantially larger than "
            f"pilot ({pilot_records})"
        )


def test_scale_records_are_id_disjoint_from_pilot():
    import json
    scale_ids = {json.loads(l)["id"]
                 for l in (SCALE / "records.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()}
    if (PILOT / "records.jsonl").exists():
        pilot_ids = {json.loads(l)["id"]
                     for l in (PILOT / "records.jsonl").read_text(encoding="utf-8").splitlines()
                     if l.strip()}
        assert scale_ids.isdisjoint(pilot_ids)


def test_scale_ablation_runs_and_is_wellformed():
    mod = _load_module("run_scale_ablation")
    from relsdlc.model import run_ablation
    ds, meta = mod.load_scale_crossrepo()
    assert set(meta["train_repos"]).isdisjoint(meta["test_repos"])
    assert meta["train_repos"] and meta["test_repos"]
    report = run_ablation(ds, seed=0, min_df=mod.MIN_DF)
    assert report["n_test_queries"] > 0
    for m in report["systems"].values():
        for v in m["recall_at_k"].values():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0


def test_idf_beats_vanilla_at_scale():
    mod = _load_module("run_scale_ablation")
    from relsdlc.model import run_ablation
    ds, _ = mod.load_scale_crossrepo()
    report = run_ablation(ds, seed=0, min_df=mod.MIN_DF)
    idf = report["systems"]["idf-cosine"]["recall_at_k"]["1"]
    vanilla = report["systems"]["vanilla-tf-cosine"]["recall_at_k"]["1"]
    # The robust finding: unsupervised IDF reliably helps real issue->PR
    # retrieval. The diagonal relation model is NOT asserted to win (it ties
    # IDF — see docs/scale-dataset.md and docs/ablation-real.md).
    assert idf >= vanilla
