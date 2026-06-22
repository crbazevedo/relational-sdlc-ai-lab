"""The R14 full-text dataset: it validates, de-truncation is real, and the
cross-repo ablation runs honestly on full body text (numpy only, no torch).

Assertions are deliberately conservative — they pin what is TRUE on the committed
full-text snapshot (it validates; bodies are clearly de-truncated; IDF is a strong
bag-of-tokens baseline; metrics are well-formed). They do NOT assert that full
text wins — the measured result is that it does not (see docs/full-text-dataset.md),
and a test must not encode a hoped-for outcome.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL = REPO_ROOT / "data" / "full"

pytestmark = pytest.mark.skipif(
    not (FULL / "records.jsonl").exists(),
    reason="full-text snapshot not present",
)


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, FULL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _body_lengths() -> list[int]:
    out = []
    for line in (FULL / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        body = (rec.get("content", {}) or {}).get("body", "") or ""
        out.append(len(body))
    return out


def test_full_validates_clean():
    from relsdlc.validate import validate_path
    report = validate_path(FULL)
    assert report.ok, [str(f) for f in report.errors]
    assert report.n_records > 100
    assert report.n_benchmarks > 50


def test_bodies_are_de_truncated():
    """The whole point of R14: bodies are no longer capped at 500 chars."""
    lengths = _body_lengths()
    # Median body length is clearly above the pilot's 500-char cap.
    assert statistics.median(lengths) > 500
    # And many records carry text the pilot would have thrown away.
    assert sum(1 for n in lengths if n > 500) > 0.25 * len(lengths)
    # The cap is honoured (8000, or 4000 fallback if the size budget tripped).
    assert max(lengths) <= 8000


def test_full_ablation_runs_and_is_wellformed():
    mod = _load_module("run_full_ablation")
    from relsdlc.model import run_ablation
    ds, meta = mod.load_full_crossrepo()
    # Train and test repos are disjoint and both populated (cross-repo split).
    assert set(meta["train_repos"]).isdisjoint(meta["test_repos"])
    assert meta["train_repos"] and meta["test_repos"]
    report = run_ablation(ds, seed=0, min_df=mod.MIN_DF)
    assert report["n_test_queries"] > 0
    for m in report["systems"].values():
        for v in m["recall_at_k"].values():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0
        assert 0.0 <= m["hard_negative_accuracy"] <= 1.0


def test_idf_is_a_strong_baseline_on_full_text():
    mod = _load_module("run_full_ablation")
    from relsdlc.model import run_ablation
    ds, _ = mod.load_full_crossrepo()
    report = run_ablation(ds, seed=0, min_df=mod.MIN_DF)
    idf = report["systems"]["idf-cosine"]["recall_at_k"]["1"]
    vanilla = report["systems"]["vanilla-tf-cosine"]["recall_at_k"]["1"]
    # IDF reliably helps issue->PR retrieval, on full text as on truncated.
    assert idf >= vanilla
