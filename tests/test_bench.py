"""Benchmark harness: end-to-end fixture run, metric ranges, leakage guard."""

from __future__ import annotations

from relsdlc.bench import leakage_violations, record_text, run_fixture_benchmark


def test_record_text_flattens_content():
    rec = {"id": "i1", "type": "issue",
           "content": {"title": "Timezone bug", "body": "UTC-3 wrong"}}
    text = record_text(rec)
    assert "Timezone bug" in text and "UTC-3 wrong" in text


def test_fixture_benchmark_runs(fixtures_dir):
    report = run_fixture_benchmark(fixtures_dir)
    assert set(report["tasks"]) >= {
        "issue_to_fixing_pr", "diff_to_affected_test", "log_to_likely_file"}
    for task, m in report["tasks"].items():
        assert m["n_queries"] >= 1
        for k, v in m["recall_at_k"].items():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0


def test_fixture_has_no_leakage(fixtures_dir):
    report = run_fixture_benchmark(fixtures_dir)
    assert report["leakage"] == []


def test_issue_to_pr_baseline_finds_fix(fixtures_dir):
    report = run_fixture_benchmark(fixtures_dir, task="issue_to_fixing_pr")
    assert report["tasks"]["issue_to_fixing_pr"]["recall_at_k"]["1"] == 1.0


def test_leakage_guard_fires_on_future_candidate():
    records_by_id = {
        "future-pr": {"id": "future-pr", "type": "pull_request",
                      "valid_from": "2025-01-01T00:00:00Z"},
    }
    query = {"query_id": "q", "query_record": "issue", "candidates": ["future-pr"],
             "relevant": ["future-pr"], "as_of": "2024-01-01T00:00:00Z"}
    assert leakage_violations(query, records_by_id) == ["future-pr"]
