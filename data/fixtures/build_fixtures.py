#!/usr/bin/env python3
"""Deterministically generate the synthetic 'datebox' fixture dataset.

This fixture is original synthetic data (CC0-1.0) authored for this project so
the benchmark harness, schema validation, and CI can run from a clean checkout
with no network and no licensing concerns. It mirrors the worked example in the
position paper: a timezone bug (issue #482) fixed by a PR, with related files,
symbols, tests, and a failing CI log.

Run:  python data/fixtures/build_fixtures.py

Outputs (under data/fixtures/):
  records.jsonl, edges.jsonl, benchmark/issue_to_fixing_pr.jsonl,
  benchmark/diff_to_affected_test.jsonl, benchmark/log_to_likely_file.jsonl
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CARDS = REPO_ROOT / "data" / "cards" / "examples"
RETRIEVED_AT = "2024-02-01T00:00:00Z"
TRANSFORM = "python data/fixtures/build_fixtures.py"
LICENSE = "CC0-1.0"
SOURCE = "synthetic://datebox"


def _hash(payload: dict) -> str:
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


def record(rid: str, rtype: str, content: dict, valid_from: str) -> dict:
    return {
        "id": rid,
        "type": rtype,
        "content": content,
        "valid_from": valid_from,
        "provenance": {
            "source_url": SOURCE,
            "retrieved_at": RETRIEVED_AT,
            "license": LICENSE,
            "content_hash": _hash(content),
            "transform": TRANSFORM,
            "method": "synthetic",
            "observed": True,
        },
    }


def edge(source: str, relation: str, target: str, valid_from: str,
         confidence: float = 1.0) -> dict:
    key = {"source": source, "relation": relation, "target": target}
    return {
        "source": source,
        "relation": relation,
        "target": target,
        "confidence": confidence,
        "valid_from": valid_from,
        "provenance": {
            "source_url": SOURCE,
            "retrieved_at": RETRIEVED_AT,
            "license": LICENSE,
            "content_hash": _hash(key),
            "transform": TRANSFORM,
            "method": "synthetic",
            "observed": True,
        },
    }


def build_records() -> list[dict]:
    return [
        # --- files (created 2024-01-01) ---
        record("file:date_filter.py", "file", {
            "path": "src/datebox/date_filter.py",
            "summary": "Date range filtering with timezone normalization.",
        }, "2024-01-01T00:00:00Z"),
        record("file:parse_tz.py", "file", {
            "path": "src/datebox/parse_tz.py",
            "summary": "Parse timezone offsets and apply UTC conversion.",
        }, "2024-01-01T00:00:00Z"),
        record("file:report.py", "file", {
            "path": "src/datebox/report.py",
            "summary": "Render currency report tables and formatting helpers.",
        }, "2024-01-01T00:00:00Z"),
        # --- symbols ---
        record("symbol:normalize_range", "symbol", {
            "name": "normalize_range",
            "path": "src/datebox/date_filter.py",
            "code": "def normalize_range(start, end, tz): return apply timezone offset to start and end dates",
        }, "2024-01-01T00:00:00Z"),
        record("symbol:parse_tz", "symbol", {
            "name": "parse_tz",
            "path": "src/datebox/parse_tz.py",
            "code": "def parse_tz(name): return utc offset for timezone name including UTC-3",
        }, "2024-01-01T00:00:00Z"),
        # --- tests ---
        record("test:test_date_filter.py", "test", {
            "path": "tests/test_date_filter.py",
            "summary": "Tests for date range filtering and timezone normalization under UTC-3.",
        }, "2024-01-01T00:00:00Z"),
        record("test:test_report.py", "test", {
            "path": "tests/test_report.py",
            "summary": "Tests for currency report formatting and table rendering.",
        }, "2024-01-01T00:00:00Z"),
        # --- issues ---
        record("issue:482", "issue", {
            "title": "Date filter returns incorrect results when timezone is UTC-3",
            "body": "The date range filter normalizes timezone incorrectly for UTC-3; "
                    "filtered results are off by one day. Expected correct date filtering "
                    "under UTC-3 timezone normalization.",
        }, "2024-01-05T00:00:00Z"),
        record("issue:101", "issue", {
            "title": "Currency report formatting misaligns columns",
            "body": "The report table renders currency columns with wrong alignment and "
                    "formatting in the summary view.",
        }, "2024-01-04T00:00:00Z"),
        record("issue:255", "issue", {
            "title": "Add pagination to result listing",
            "body": "Listing endpoint should support pagination with page size and offset.",
        }, "2024-01-03T00:00:00Z"),
        # --- diffs / PRs / commit ---
        record("diff:512", "diff", {
            "summary": "Fix timezone normalization in normalize_range; correct UTC-3 offset via parse_tz.",
            "diff": "modified normalize_range to apply parse_tz offset correctly for UTC-3 dates",
        }, "2024-01-08T00:00:00Z"),
        record("pr:512", "pull_request", {
            "title": "Fix UTC-3 timezone normalization in date filter",
            "body": "Corrects normalize_range to use parse_tz so date filtering is correct "
                    "under UTC-3. Adds a regression test.",
        }, "2024-01-08T00:00:00Z"),
        record("pr:140", "pull_request", {
            "title": "Fix currency report column formatting",
            "body": "Adjusts report table rendering so currency columns align in the summary view.",
        }, "2024-01-06T00:00:00Z"),
        record("pr:300", "pull_request", {
            "title": "Add pagination to result listing endpoint",
            "body": "Implements page size and offset pagination for the listing endpoint.",
        }, "2024-01-07T00:00:00Z"),
        # --- CI log (failing, before the fix lands as merged) ---
        record("ci_log:512-fail", "ci_log", {
            "log": "FAILED tests/test_date_filter.py::test_utc_minus_3 - assertion error: "
                   "normalize_range returned wrong day for UTC-3 timezone offset",
        }, "2024-01-09T00:00:00Z"),
    ]


def build_edges() -> list[dict]:
    return [
        edge("pr:512", "fixes", "issue:482", "2024-01-08T00:00:00Z"),
        edge("pr:140", "fixes", "issue:101", "2024-01-06T00:00:00Z"),
        edge("pr:300", "fixes", "issue:255", "2024-01-07T00:00:00Z"),
        edge("pr:512", "modifies", "file:date_filter.py", "2024-01-08T00:00:00Z"),
        edge("pr:512", "modifies", "file:parse_tz.py", "2024-01-08T00:00:00Z"),
        edge("diff:512", "modifies", "symbol:normalize_range", "2024-01-08T00:00:00Z"),
        edge("diff:512", "modifies", "symbol:parse_tz", "2024-01-08T00:00:00Z"),
        edge("test:test_date_filter.py", "covers", "symbol:normalize_range", "2024-01-01T00:00:00Z"),
        edge("test:test_report.py", "covers", "file:report.py", "2024-01-01T00:00:00Z"),
        edge("ci_log:512-fail", "caused_by", "diff:512", "2024-01-09T00:00:00Z"),
    ]


def build_benchmarks() -> dict[str, list[dict]]:
    return {
        "issue_to_fixing_pr": [{
            "query_id": "q-issue-482",
            "task": "issue_to_fixing_pr",
            "query_record": "issue:482",
            "candidates": ["pr:512", "pr:140", "pr:300"],
            "relevant": ["pr:512"],
            "hard_negatives": ["pr:140"],
            "as_of": "2024-02-01T00:00:00Z",
        }],
        "diff_to_affected_test": [{
            "query_id": "q-diff-512",
            "task": "diff_to_affected_test",
            "query_record": "diff:512",
            "candidates": ["test:test_date_filter.py", "test:test_report.py"],
            "relevant": ["test:test_date_filter.py"],
            "hard_negatives": ["test:test_report.py"],
            "as_of": "2024-02-01T00:00:00Z",
        }],
        "log_to_likely_file": [{
            "query_id": "q-log-512",
            "task": "log_to_likely_file",
            "query_record": "ci_log:512-fail",
            "candidates": ["file:date_filter.py", "file:parse_tz.py", "file:report.py"],
            "relevant": ["file:date_filter.py", "file:parse_tz.py"],
            "hard_negatives": ["file:report.py"],
            "as_of": "2024-02-01T00:00:00Z",
        }],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _file_hash(*paths: Path) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    return "sha256:" + h.hexdigest()


def build_source_card(manifest_hash: str) -> dict:
    return {
        "card_type": "source",
        "id": "src:datebox",
        "name": "datebox synthetic fixture",
        "source_url": SOURCE,
        "retrieved_at": RETRIEVED_AT,
        "license": LICENSE,
        "terms_note": "Original synthetic data authored for this project; freely redistributable.",
        "record_types": ["issue", "pull_request", "commit", "diff", "file",
                         "symbol", "test", "ci_log"],
        "transform": TRANSFORM,
        "content_hash": manifest_hash,
        "redistribution": "synthetic_original",
        "notes": "Mirrors the timezone-bug worked example (issue #482) from the position paper.",
    }


def build_dataset_card(records: list[dict], edges: list[dict], manifest_hash: str) -> dict:
    return {
        "card_type": "dataset",
        "id": "ds:datebox-fixture-v0",
        "name": "datebox fixture",
        "version": "v0",
        "created_at": RETRIEVED_AT,
        "sources": ["src:datebox"],
        "record_counts": dict(sorted(Counter(r["type"] for r in records).items())),
        "edge_counts": dict(sorted(Counter(e["relation"] for e in edges).items())),
        "relation_types": sorted({e["relation"] for e in edges}),
        "split_policy": {
            "frozen": True,
            "method": "synthetic-fixed",
            "seed": 0,
            "boundary": "2024-02-01T00:00:00Z",
        },
        "redistribution": "synthetic_original",
        "known_limitations": [
            "Tiny synthetic fixture for smoke-testing the harness; not evidence of model quality.",
        ],
        "notes": f"Built by {TRANSFORM}. Manifest hash {manifest_hash}.",
    }


def build_experiment_card() -> dict:
    """Run the vanilla baseline on the fixture and record the headline task metrics."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from relsdlc.bench import run_fixture_benchmark  # noqa: E402

    report = run_fixture_benchmark(HERE, task="issue_to_fixing_pr")
    m = report["tasks"]["issue_to_fixing_pr"]
    return {
        "card_type": "experiment",
        "id": "exp:baseline-hashing-tfidf-issue2pr-v0",
        "name": "Baseline hashing-TF-IDF — issue_to_fixing_pr (fixture)",
        "created_at": RETRIEVED_AT,
        "hypothesis": "Vanilla text embedding retrieves the fixing PR for an issue; "
                      "establishes the floor a relation-aware model must beat.",
        "task": "issue_to_fixing_pr",
        "dataset_version": "ds:datebox-fixture-v0",
        "code_version": "relsdlc-0.1.0",
        "seed": 0,
        "command": "relsdlc bench --task issue_to_fixing_pr",
        "system": "baseline-hashing-tfidf",
        "runtime_class": "cpu",
        "metrics": {
            "recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
            "mrr": round(m["mrr"], 4),
            "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
            "n_queries": m["n_queries"],
        },
        "baseline_comparison": "none",
        "error_slices": [],
        "leakage_checks": ["as_of=2024-02-01; no candidate/positive valid after query time"],
        "known_limitations": [
            "Single synthetic query; not statistically meaningful. Exploratory only.",
        ],
        "exploratory": True,
    }


def main() -> None:
    records, edges = build_records(), build_edges()
    _write_jsonl(HERE / "records.jsonl", records)
    _write_jsonl(HERE / "edges.jsonl", edges)
    for name, rows in build_benchmarks().items():
        _write_jsonl(HERE / "benchmark" / f"{name}.jsonl", rows)

    manifest_hash = _file_hash(HERE / "records.jsonl", HERE / "edges.jsonl")
    _write_json(CARDS / "datebox.source-card.json", build_source_card(manifest_hash))
    _write_json(CARDS / "datebox-fixture-v0.dataset-card.json",
                build_dataset_card(records, edges, manifest_hash))
    _write_json(CARDS / "baseline-hashing-tfidf-issue2pr-v0.experiment-card.json",
                build_experiment_card())
    print(f"wrote fixture + example cards to {HERE} and {CARDS}")


if __name__ == "__main__":
    main()
