"""The R16A content dataset: real file/test bodies that validate, are genuinely
DEEP, and a diff->affected-test benchmark that resolves + scores honestly.

Assertions pin what is TRUE on the committed content snapshot: content records
validate clean, carry ``content.text``, and the median text length is well above
500 chars (the deep-signal premise). The benchmark queries are
``diff_to_affected_test`` and reference real records. The ablation result, when
present, is read and REPORTED — we do NOT assert MaxP wins (a test must not encode
a hoped-for outcome, even when the measured result happens to be favourable).

Numpy/json-only; the embedding caches are gitignored, so the result-reading test
is skipped when content-chunk-results.json is absent.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENT = REPO_ROOT / "data" / "content"
RECORDS = CONTENT / "file_contents.jsonl"
BENCH = CONTENT / "benchmark" / "diff_to_affected_test.jsonl"
RESULTS = CONTENT / "content-chunk-results.json"

pytestmark = pytest.mark.skipif(
    not RECORDS.exists(), reason="content snapshot not present",
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def _text_lengths() -> list[int]:
    return [len((r.get("content", {}) or {}).get("text", "") or "")
            for r in _load_jsonl(RECORDS)]


def test_content_validates_clean():
    """The content records + the benchmark validate against the full dataset.

    Run over data/ as a whole so the benchmark's query_record refs (the pilot PR
    records) and candidate refs (the content test records) all resolve.
    """
    from relsdlc.validate import validate_path
    report = validate_path(REPO_ROOT / "data")
    assert report.ok, [str(f) for f in report.errors]


def test_content_records_have_text_and_provenance():
    records = _load_jsonl(RECORDS)
    assert len(records) > 100
    for r in records:
        assert r["type"] in ("file", "test")
        assert r["id"].startswith("gh-content:")
        text = (r.get("content", {}) or {}).get("text")
        assert isinstance(text, str) and text, f"missing content.text on {r['id']}"
        prov = r["provenance"]
        assert prov["content_hash"].startswith("sha256:")
        assert prov["method"] == "git_history"
        assert prov["source_url"].startswith("https://github.com/")
        assert len(text) <= 16000  # the cap is honoured.


def test_content_is_deep():
    """The R16A premise: file/test bodies are DEEP (median >> 500 chars)."""
    lengths = _text_lengths()
    assert statistics.median(lengths) > 500
    # The signal-is-deep claim is strong here, not marginal.
    assert statistics.median(lengths) > 2000
    # And many records are long enough to need more than one 512-char chunk.
    assert sum(1 for n in lengths if n > 1024) > 0.5 * len(lengths)


def test_benchmark_is_diff_to_test_and_resolves():
    queries = _load_jsonl(BENCH)
    assert len(queries) > 50
    record_ids = {r["id"] for r in _load_jsonl(RECORDS)}
    for q in queries:
        assert q["task"] == "diff_to_affected_test"
        assert q["query_record"].startswith("gh:")  # a pilot PR record.
        assert ":pr:" in q["query_record"]           # the diff side is a PR.
        assert q["relevant"], f"empty relevant on {q['query_id']}"
        for rel in q["relevant"]:
            assert rel in q["candidates"]
            assert rel in record_ids  # candidate tests are fetched content records.
        # Query (PR) and candidate (test) share a repo -> cross-repo split feasible.
        q_repo = q["query_record"].split(":")[1]
        for c in q["candidates"]:
            assert c.split(":")[1] == q_repo


def test_reports_chunk_ablation_result_if_present():
    """Read and REPORT the deep-signal result. We do NOT assert MaxP wins."""
    if not RESULTS.exists():
        pytest.skip("content-chunk-results.json absent (run the ablation)")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    results = data["results"]
    # The result is well-formed: every aggregator/size cell has bounded metrics.
    for key, m in results.items():
        for v in m["recall_at_k"].values():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0
        assert 0.0 <= m["hard_negative_accuracy"] <= 1.0
    # Honest report (printed with -s): MaxP vs FirstP R@1 at each chunk size.
    sizes = sorted({int(k[1:].split(":")[0]) for k in results})
    print("\ndeep-signal MaxP vs FirstP (R@1), diff_to_affected_test, cross-repo:")
    for size in sizes:
        fp = results[f"s{size}:firstp"]["recall_at_k"]["1"]
        mp = results[f"s{size}:maxp"]["recall_at_k"]["1"]
        print(f"  size {size}: FirstP={fp:.3f}  MaxP={mp:.3f}  delta={mp - fp:+.3f}")
    print(f"  verdict: {data.get('verdict', {}).get('summary', 'n/a')}")
