"""Hermetic (offline) tests for the GitHub-ingest tooling.

These tests never touch the network. They transform the RECORDED snapshot under
``tests/ingest_fixtures/`` and assert the produced records/edges conform to the
schemas and that the relations are extracted correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from relsdlc import ingest
from relsdlc.validate import validate_dataset, validate_edge, validate_record

FIXTURES = Path(__file__).resolve().parent / "ingest_fixtures"
REPO = "octo-demo/widgets"


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return ingest.load_snapshot(FIXTURES)


@pytest.fixture(scope="module")
def result(snapshot) -> dict:
    return ingest.transform_snapshot(snapshot)


# --- record mapping ----------------------------------------------------------


def test_records_and_edges_are_schema_valid(result):
    for rec in result["records"]:
        assert validate_record(rec) == [], rec["id"]
    for edge in result["edges"]:
        assert validate_edge(edge) == [], (edge["source"], edge["target"])


def test_dataset_referential_integrity_is_clean(result):
    findings = validate_dataset(result["records"], result["edges"])
    errors = [f for f in findings if f.severity == "error"]
    assert errors == [], [str(f) for f in errors]


def test_pr_in_issues_feed_is_not_mapped_as_issue(result):
    # Issue #21 carries a "pull_request" key in the issues feed; it must be skipped.
    ids = {r["id"] for r in result["records"]}
    assert ingest.issue_id(REPO, 21) not in ids


def test_issue_record_shape(result):
    by_id = {r["id"]: r for r in result["records"]}
    issue = by_id[ingest.issue_id(REPO, 12)]
    assert issue["type"] == "issue"
    assert issue["content"]["title"] == "Slider widget ignores step when dragged"
    assert issue["content"]["labels"] == ["bug", "component:slider"]
    assert issue["valid_from"] == "2024-04-02T09:10:00Z"
    prov = issue["provenance"]
    assert prov["license"] == "MIT"
    assert prov["content_hash"].startswith("sha256:")
    assert prov["source_url"] == "https://github.com/octo-demo/widgets/issues/12"


def test_pull_request_and_commit_records_present(result):
    ids = {r["id"] for r in result["records"]}
    assert ingest.pr_id(REPO, 34) in ids
    assert ingest.commit_id(REPO, "a1b2c3d4e5f60718293a4b5c6d7e8f9001122334") in ids


def test_changed_test_file_is_typed_as_test(result):
    by_id = {r["id"]: r for r in result["records"]}
    test_rec = by_id[ingest.file_id(REPO, "tests/slider.test.js")]
    assert test_rec["type"] == "test"
    src_rec = by_id[ingest.file_id(REPO, "src/widgets/slider.js")]
    assert src_rec["type"] == "file"


# --- edge extraction ---------------------------------------------------------


def _edges(result, relation):
    return [e for e in result["edges"] if e["relation"] == relation]


def test_fixes_edges_from_closing_keywords(result):
    fixes = {(e["source"], e["target"]) for e in _edges(result, "fixes")}
    assert (ingest.pr_id(REPO, 34), ingest.issue_id(REPO, 12)) in fixes  # "Fixes #12"
    assert (ingest.pr_id(REPO, 40), ingest.issue_id(REPO, 18)) in fixes  # "Closes: #18"
    assert len(fixes) == 2


def test_fixes_edges_use_human_label_method(result):
    for e in _edges(result, "fixes"):
        assert e["provenance"]["method"] == "human_label"
        assert 0.0 <= e["confidence"] <= 1.0


def test_dangling_issue_reference_is_not_minted(result):
    # PR #50's body mentions #99, which has no issue record -> no edge to it.
    targets = {e["target"] for e in result["edges"]}
    assert ingest.issue_id(REPO, 99) not in targets


def test_modifies_edges_from_changed_files(result):
    modifies = {(e["source"], e["target"]) for e in _edges(result, "modifies")}
    assert (ingest.pr_id(REPO, 34),
            ingest.file_id(REPO, "src/widgets/slider.js")) in modifies
    assert (ingest.pr_id(REPO, 40),
            ingest.file_id(REPO, "src/widgets/tooltip.js")) in modifies
    assert len(modifies) == 5


def test_modifies_edges_use_git_history_method(result):
    for e in _edges(result, "modifies"):
        assert e["provenance"]["method"] == "git_history"


# --- provenance / reproducibility -------------------------------------------


def test_content_hash_is_deterministic(snapshot):
    a = ingest.transform_snapshot(snapshot)
    b = ingest.transform_snapshot(snapshot)
    assert [r["provenance"]["content_hash"] for r in a["records"]] == \
           [r["provenance"]["content_hash"] for r in b["records"]]


def test_content_hash_matches_canonical_content(result):
    for rec in result["records"]:
        assert rec["provenance"]["content_hash"] == ingest._hash(rec["content"])


def test_no_todo_hashes(result):
    for rec in result["records"]:
        assert rec["provenance"]["content_hash"] != "TODO"
    for edge in result["edges"]:
        assert edge["provenance"]["content_hash"] != "TODO"


# --- committed example matches the transform --------------------------------


def test_committed_example_matches_transform(result):
    """data/ingest_example/ must be reproducible from the recorded fixtures."""
    import json

    example = FIXTURES.parents[1] / "data" / "ingest_example"
    if not (example / "records.jsonl").exists():
        pytest.skip("data/ingest_example not present")

    def _load(path):
        return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]

    committed_records = _load(example / "records.jsonl")
    committed_edges = _load(example / "edges.jsonl")
    assert committed_records == result["records"]
    assert committed_edges == result["edges"]


# --- source card -------------------------------------------------------------


def test_source_card_is_metadata_only(snapshot, result):
    card = ingest.build_source_card(snapshot, len(result["records"]))
    assert card["card_type"] == "source"
    assert card["redistribution"] == "metadata_only"
    assert card["license"] == "MIT"
