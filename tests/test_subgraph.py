"""Hermetic (offline) tests for the GraphRAG subgraph packer (numpy-only).

These exercise ``relsdlc.subgraph`` over the committed pilot records, edges,
graph, and node embeddings — no torch, no network, no SLM. They assert:

* ``retrieve_subgraph`` surfaces the true fixing PR in its top-k candidates for
  most sample issues (relation-conditioned retrieval works);
* ``pack_context`` is deterministic, non-empty, and includes the issue and the
  related file paths the graph attaches;
* the subgraph respects the graph — every ``modifies`` path it emits resolves
  back to a real ``modifies`` edge.

Skips cleanly if any committed input (records / edges / graph / embeddings) is
absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data" / "pilot"
RECORDS = PILOT / "records.jsonl"
EDGES = PILOT / "edges.jsonl"
FILE_RECORDS = PILOT / "graph" / "file_records.jsonl"
MODIFIES_EDGES = PILOT / "graph" / "modifies_edges.jsonl"
EMBEDDINGS = PILOT / "embeddings" / "minilm-l6-v2.npz"
BENCHMARK = PILOT / "benchmark" / "issue_to_fixing_pr.jsonl"

_REQUIRED = [RECORDS, EDGES, FILE_RECORDS, MODIFIES_EDGES, EMBEDDINGS, BENCHMARK]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED),
    reason="pilot data / graph / embeddings / benchmark snapshot not present",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


@pytest.fixture(scope="module")
def graph():
    from relsdlc.subgraph import build_graph_view

    records = _read_jsonl(RECORDS)
    file_records = _read_jsonl(FILE_RECORDS)
    fixes = [
        (e["source"], e["target"])
        for e in _read_jsonl(EDGES)
        if e.get("relation") == "fixes"
    ]
    mods = [
        (e["source"], e["target"])
        for e in _read_jsonl(MODIFIES_EDGES)
        if e.get("relation") == "modifies"
    ]
    npz = np.load(EMBEDDINGS, allow_pickle=True)
    ids = [str(i) for i in npz["ids"]]
    vecs = np.asarray(npz["vectors"], dtype=np.float64)
    node_vecs = {i: vecs[k] for k, i in enumerate(ids)}
    return build_graph_view(records, file_records, fixes, mods, node_vecs)


@pytest.fixture(scope="module")
def queries() -> list[dict]:
    # A fixed, deterministic slice of the benchmark (first 40 queries) so the
    # test is fast and stable regardless of future benchmark growth.
    return _read_jsonl(BENCHMARK)[:40]


def test_graph_view_built(graph):
    assert graph.records, "expected records in the graph view"
    assert graph.vecs, "expected node embeddings in the graph view"
    assert graph.pr_to_modified, "expected at least one PR with modifies edges"
    assert graph.issue_to_fixers, "expected fixes adjacency"


def test_retrieve_surfaces_true_fixing_pr_in_top_k(graph, queries):
    """The gold fixing PR should appear in top-k for the majority of sample issues.

    Retrieval is restricted to each query's candidate pool (the benchmark
    contract), so this measures the relation-conditioned ranking, not recall over
    the full corpus.
    """
    from relsdlc.subgraph import retrieve_subgraph

    top_k = 5
    hits = 0
    checked = 0
    for q in queries:
        sg = retrieve_subgraph(
            q["query_record"], graph, top_k=top_k, candidate_pr_ids=q["candidates"]
        )
        top_ids = {p["id"] for p in sg["related_prs"]}
        gold = set(q["relevant"])
        if gold:
            checked += 1
            hits += bool(gold & top_ids)
    assert checked > 0, "no queries with gold relevance to score"
    # Comfortably above chance; observed ~0.9 on the committed snapshot.
    assert hits / checked >= 0.6, f"only {hits}/{checked} gold PRs in top-{top_k}"


def test_retrieve_respects_top_k_and_ordering(graph, queries):
    from relsdlc.subgraph import retrieve_subgraph

    q = queries[0]
    sg = retrieve_subgraph(q["query_record"], graph, top_k=3, candidate_pr_ids=q["candidates"])
    prs = sg["related_prs"]
    assert len(prs) <= 3
    # Scores are non-increasing (cosine-ranked).
    scores = [p["score"] for p in prs]
    assert scores == sorted(scores, reverse=True), scores


def test_retrieve_is_deterministic(graph, queries):
    from relsdlc.subgraph import retrieve_subgraph

    q = queries[0]
    a = retrieve_subgraph(q["query_record"], graph, top_k=5, candidate_pr_ids=q["candidates"])
    b = retrieve_subgraph(q["query_record"], graph, top_k=5, candidate_pr_ids=q["candidates"])
    assert [p["id"] for p in a["related_prs"]] == [p["id"] for p in b["related_prs"]]
    assert a["edges"] == b["edges"]


def test_subgraph_modifies_paths_resolve_to_real_edges(graph, queries):
    """Every file/test path packed for a PR must come from a real modifies edge."""
    from relsdlc.subgraph import retrieve_subgraph

    # PR id -> set of modified short paths (ground truth from the graph view).
    def paths_of(pr_id: str) -> set[str]:
        out = set()
        for tid in graph.pr_to_modified.get(pr_id, []):
            out.add(graph.path_of(tid) or tid)
        return out

    saw_resolved = False
    for q in queries:
        sg = retrieve_subgraph(
            q["query_record"], graph, top_k=5, candidate_pr_ids=q["candidates"]
        )
        for pr in sg["related_prs"]:
            truth = paths_of(pr["id"])
            for path in pr["files"] + pr["tests"]:
                assert path in truth, (
                    f"packed path {path!r} for {pr['id']} not in its modifies edges"
                )
                saw_resolved = True
    assert saw_resolved, "expected at least one PR with resolved modifies paths"


def test_pack_context_nonempty_and_includes_issue_and_paths(graph, queries):
    from relsdlc.subgraph import pack_context, retrieve_subgraph

    # Pick a query whose top-k contains a PR with at least one changed path so the
    # "includes related file paths" assertion is meaningful.
    chosen = None
    for q in queries:
        sg = retrieve_subgraph(
            q["query_record"], graph, top_k=5, candidate_pr_ids=q["candidates"]
        )
        if any(pr["files"] or pr["tests"] for pr in sg["related_prs"]):
            chosen = (q, sg)
            break
    assert chosen is not None, "no sample issue retrieved a PR with changed paths"
    q, sg = chosen

    ctx = pack_context(sg)
    assert ctx.strip(), "packed context is empty"
    assert "# ISSUE" in ctx
    assert "RELATED PULL REQUESTS" in ctx
    # The issue title (truncated/collapsed) should be present.
    title_head = " ".join(graph.title_of(q["query_record"]).split())[:40]
    if title_head:
        assert title_head in ctx
    # At least one packed file/test path must show up in the packed string.
    some_path = next(
        p
        for pr in sg["related_prs"]
        for p in (pr["files"] + pr["tests"])
    )
    assert some_path in ctx


def test_pack_context_is_deterministic(graph, queries):
    from relsdlc.subgraph import pack_context, retrieve_subgraph

    q = queries[0]
    sg = retrieve_subgraph(q["query_record"], graph, top_k=5, candidate_pr_ids=q["candidates"])
    assert pack_context(sg) == pack_context(sg)


def test_pack_context_is_length_bounded(graph, queries):
    """Truncation budgets keep the packed string from blowing up the SLM prompt."""
    from relsdlc.subgraph import pack_context, retrieve_subgraph

    q = queries[0]
    sg = retrieve_subgraph(q["query_record"], graph, top_k=5, candidate_pr_ids=q["candidates"])
    short = pack_context(sg, issue_body_chars=200)
    # Each line is bounded; the issue description line obeys its char budget.
    desc_lines = [ln for ln in short.splitlines() if ln.startswith("description:")]
    if desc_lines:
        # "description: " prefix (13) + budget (200) + 1 ellipsis char tolerance.
        assert len(desc_lines[0]) <= 13 + 200 + 1
