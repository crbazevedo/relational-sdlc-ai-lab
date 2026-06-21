"""Hermetic (offline) tests over the committed graph-enrichment output.

These never touch the network. They assert the committed ``modifies`` edges and
file/test records under ``data/pilot/graph/`` validate clean, that every edge
resolves to a real PR record (source) and a real file record (target), and that
at least one ``test`` record is present so the diff->affected-test task is
feasible. Skips when the committed graph snapshot is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data" / "pilot"
GRAPH = PILOT / "graph"
FILE_RECORDS = GRAPH / "file_records.jsonl"
MODIFIES_EDGES = GRAPH / "modifies_edges.jsonl"

pytestmark = pytest.mark.skipif(
    not FILE_RECORDS.exists(),
    reason="graph snapshot not present (run data/pilot/build_graph.py)",
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def file_records() -> list[dict]:
    return _read_jsonl(FILE_RECORDS)


@pytest.fixture(scope="module")
def modifies_edges() -> list[dict]:
    return _read_jsonl(MODIFIES_EDGES)


@pytest.fixture(scope="module")
def pr_record_ids() -> set[str]:
    return {
        r["id"]
        for r in _read_jsonl(PILOT / "records.jsonl")
        if r.get("type") == "pull_request"
    }


def test_graph_validates_clean():
    # Validate over the pilot tree (so modifies-edge sources resolve to the PR
    # records that live alongside, exactly as `relsdlc validate data` scans):
    # 0 errors (warnings ok). The graph's own file records + modifies edges are
    # included in this scan.
    from relsdlc.validate import validate_path

    report = validate_path(PILOT)
    assert report.ok, [str(f) for f in report.errors]
    assert report.n_records > 0
    assert report.n_edges > 0
    # The graph contributes file/test records + modifies edges to the scan.
    file_ids = {r["id"] for r in _read_jsonl(FILE_RECORDS)}
    assert file_ids, "expected committed file records"


def test_every_modifies_edge_is_a_modifies_relation(modifies_edges):
    assert modifies_edges, "expected at least one modifies edge"
    for e in modifies_edges:
        assert e["relation"] == "modifies", e


def test_every_edge_source_resolves_to_a_pr_record(modifies_edges, pr_record_ids):
    for e in modifies_edges:
        assert e["source"] in pr_record_ids, (
            f"modifies edge source {e['source']!r} is not a PR record"
        )


def test_every_edge_target_resolves_to_a_file_record(modifies_edges, file_records):
    file_ids = {r["id"] for r in file_records}
    for e in modifies_edges:
        assert e["target"] in file_ids, (
            f"modifies edge target {e['target']!r} is not a file record"
        )


def test_at_least_one_test_record_for_diff_to_test_feasibility(file_records):
    types = {r["type"] for r in file_records}
    assert types <= {"file", "test"}, f"unexpected record types: {types}"
    n_test = sum(1 for r in file_records if r["type"] == "test")
    assert n_test >= 1, "expected >= 1 test record so diff->affected-test is feasible"


def test_file_records_have_real_content_hash(file_records):
    for r in file_records:
        ch = r["provenance"]["content_hash"]
        assert ch != "TODO" and ch.startswith("sha256:"), r["id"]
