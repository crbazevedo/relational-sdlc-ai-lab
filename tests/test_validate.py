"""Validation gates: schema, provenance, referential integrity, leakage."""

from __future__ import annotations

from relsdlc.validate import (
    validate_benchmark,
    validate_dataset,
    validate_edge,
    validate_path,
    validate_record,
)

GOOD_PROV = {
    "source_url": "synthetic://x",
    "retrieved_at": "2024-01-01T00:00:00Z",
    "license": "CC0-1.0",
    "content_hash": "sha256:" + "a" * 64,
}


def _codes(findings):
    return {f.code for f in findings}


def test_valid_record_passes():
    rec = {"id": "r1", "type": "file", "content": {"path": "a.py"}, "provenance": GOOD_PROV}
    assert validate_record(rec) == []


def test_record_missing_provenance_fails():
    rec = {"id": "r1", "type": "file"}
    assert "schema" in _codes(validate_record(rec))


def test_record_todo_hash_fails():
    prov = {**GOOD_PROV, "content_hash": "TODO"}
    rec = {"id": "r1", "type": "file", "provenance": prov}
    # 'TODO' fails the schema pattern AND the provenance.todo gate.
    assert "provenance.todo" in _codes(validate_record(rec))


def test_edge_requires_method():
    prov = {k: v for k, v in GOOD_PROV.items()}  # no 'method'
    edge = {"source": "a", "relation": "fixes", "target": "b",
            "confidence": 1.0, "provenance": prov}
    assert "provenance.method" in _codes(validate_edge(edge))


def test_edge_bad_relation_fails_schema():
    prov = {**GOOD_PROV, "method": "git_history"}
    edge = {"source": "a", "relation": "frobnicates", "target": "b",
            "confidence": 1.0, "provenance": prov}
    assert "schema" in _codes(validate_edge(edge))


def test_edge_confidence_out_of_range_fails():
    prov = {**GOOD_PROV, "method": "git_history"}
    edge = {"source": "a", "relation": "fixes", "target": "b",
            "confidence": 1.5, "provenance": prov}
    assert "schema" in _codes(validate_edge(edge))


def test_dangling_edge_endpoint_fails():
    rec = {"id": "a", "type": "file", "provenance": GOOD_PROV}
    edge = {"source": "a", "relation": "fixes", "target": "ghost",
            "confidence": 1.0, "provenance": {**GOOD_PROV, "method": "git_history"}}
    findings = validate_dataset([rec], [edge])
    assert "ref.dangling" in _codes(findings)


def test_duplicate_record_id_fails():
    rec = {"id": "a", "type": "file", "provenance": GOOD_PROV}
    findings = validate_dataset([rec, dict(rec)], [])
    assert "id.duplicate" in _codes(findings)


def test_benchmark_relevant_must_be_in_candidates():
    q = {"query_id": "q1", "task": "issue_to_fixing_pr", "query_record": "i1",
         "candidates": ["a", "b"], "relevant": ["c"]}
    assert "benchmark.relevant" in _codes(validate_benchmark(q))


def test_benchmark_dangling_reference_fails():
    rec = {"id": "i1", "type": "issue", "provenance": GOOD_PROV}
    q = {"query_id": "q1", "task": "issue_to_fixing_pr", "query_record": "i1",
         "candidates": ["missing"], "relevant": ["missing"]}
    findings = validate_dataset([rec], [], benchmarks=[q])
    assert "ref.dangling" in _codes(findings)


def test_temporal_inconsistency_warns():
    rec = {"id": "a", "type": "file", "valid_from": "2024-02-01T00:00:00Z",
           "provenance": GOOD_PROV}
    # edge claims to be valid before its endpoint exists.
    edge = {"source": "a", "relation": "fixes", "target": "a",
            "confidence": 1.0, "valid_from": "2024-01-01T00:00:00Z",
            "provenance": {**GOOD_PROV, "method": "git_history"}}
    findings = validate_dataset([rec], [edge])
    assert "temporal.inconsistent" in _codes(findings)


def test_fixture_dataset_validates_clean(repo_root):
    report = validate_path(repo_root / "data")
    assert report.ok, [str(f) for f in report.errors]
    assert report.n_records > 0 and report.n_edges > 0
    assert report.n_cards >= 3 and report.n_benchmarks >= 3
