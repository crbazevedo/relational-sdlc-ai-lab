"""Relation-conditioned (GraphRAG) subgraph retrieval + context packing (numpy).

This is the **MVP-2 entry**: the retrieval half of the relational-SLM layer the
[architecture](../../docs/architecture.md) sketches as

    task -> relation-conditioned retrieval -> subgraph -> prompt/context -> response

Given an *issue* node, it packs a compact, deterministic text context for a
downstream small LM by combining **text similarity with graph structure** —
"GraphRAG" rather than flat top-k chunk RAG:

1. **Retrieve.** Rank candidate PRs by cosine of their pretrained node
   embeddings against the issue's embedding (the Track-A representation), keep
   the top ``k``.
2. **Expand along relations.** For each retrieved PR, follow its ``modifies``
   edges to the concrete ``file`` / ``test`` nodes it changed — the structural
   context a flat text retriever cannot see (file/test nodes have no text
   embedding at all). Also pull the issue's own graph neighbours (the PRs the
   ``fixes`` edges already attach to it, when present).
3. **Pack.** Serialise the issue + the related PRs (each with its changed source
   files and tests, truncated) into one structured, length-bounded string ready
   to drop into an SLM prompt.

Everything is numpy-only and a pure function of the inputs: candidate ordering
breaks cosine ties by id, neighbour lists are sorted, and all truncation is by
fixed character / count budgets. The same inputs always pack the same string.

NO torch, NO network, NO transformers — this module is the CI-tested deliverable.
The actual small-LM generation that consumes the packed context is a separate,
best-effort dry-run (``data/pilot/slm_demo.py``); it is *not* imported here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Relation roles we condition retrieval on.
FIXES = "fixes"
MODIFIES = "modifies"


# --------------------------------------------------------------------------- #
# Lightweight in-memory graph view (built once, queried per issue).
# --------------------------------------------------------------------------- #

@dataclass
class GraphView:
    """A queryable, in-memory view of the records + typed edges + node features.

    All fields are plain dicts/sets so retrieval is deterministic and torch-free.
    """

    records: dict = field(default_factory=dict)          # id -> record dict
    vecs: dict = field(default_factory=dict)             # id -> unit np.ndarray
    pr_to_modified: dict = field(default_factory=dict)   # pr_id -> sorted [file/test id]
    issue_to_fixers: dict = field(default_factory=dict)  # issue_id -> sorted [pr id]
    pr_to_fixed: dict = field(default_factory=dict)      # pr_id -> sorted [issue id]

    def is_type(self, node_id: str, type_name: str) -> bool:
        rec = self.records.get(node_id)
        return bool(rec) and rec.get("type") == type_name

    def title_of(self, node_id: str) -> str:
        rec = self.records.get(node_id) or {}
        return str((rec.get("content") or {}).get("title", "")).strip()

    def body_of(self, node_id: str) -> str:
        rec = self.records.get(node_id) or {}
        return str((rec.get("content") or {}).get("body", "") or "")

    def path_of(self, node_id: str) -> str:
        """Short path for a file/test node (or '' for non-file nodes)."""
        rec = self.records.get(node_id) or {}
        return str((rec.get("content") or {}).get("path", "")).strip()


def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _is_test_path(path: str) -> bool:
    p = path.lower()
    return ("test" in p) or p.endswith("_test.py") or p.startswith("tests/")


def build_graph_view(records, file_records, fixes_edges, modifies_edges, node_vecs):
    """Assemble a :class:`GraphView` from the on-disk record/edge/feature pieces.

    Parameters
    ----------
    records : iterable of record dicts (issues + PRs), each with ``id`` / ``type``.
    file_records : iterable of file/test record dicts (``content.path``).
    fixes_edges : iterable of (pr_id, issue_id).
    modifies_edges : iterable of (pr_id, file_or_test_id).
    node_vecs : dict id -> raw embedding vector (issues + PRs; files have none).

    All adjacency lists are stored sorted by id, so every downstream traversal is
    deterministic.
    """
    recs: dict = {}
    for r in records:
        recs[r["id"]] = r
    for r in file_records:
        recs[r["id"]] = r

    vecs = {nid: _unit(v) for nid, v in node_vecs.items()}

    pr_to_modified = defaultdict(set)
    for pr, tgt in modifies_edges:
        pr_to_modified[pr].add(tgt)

    issue_to_fixers = defaultdict(set)
    pr_to_fixed = defaultdict(set)
    for pr, iss in fixes_edges:
        issue_to_fixers[iss].add(pr)
        pr_to_fixed[pr].add(iss)

    return GraphView(
        records=recs,
        vecs=vecs,
        pr_to_modified={pr: sorted(t) for pr, t in pr_to_modified.items()},
        issue_to_fixers={i: sorted(p) for i, p in issue_to_fixers.items()},
        pr_to_fixed={pr: sorted(i) for pr, i in pr_to_fixed.items()},
    )


# --------------------------------------------------------------------------- #
# Relation-conditioned retrieval.
# --------------------------------------------------------------------------- #

def _rank_candidate_prs(graph: GraphView, issue_id: str, candidate_pr_ids, top_k):
    """Cosine-rank ``candidate_pr_ids`` against the issue embedding.

    Ties (equal score) break by candidate id so the order is deterministic. PRs
    with no embedding are dropped (they cannot be scored by text cosine).
    """
    q = graph.vecs.get(issue_id)
    if q is None:
        return []
    scored = []
    for cid in candidate_pr_ids:
        d = graph.vecs.get(cid)
        if d is None:
            continue
        scored.append((cid, float(q @ d)))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored[:top_k]


def _default_candidate_prs(graph: GraphView) -> list:
    """All PR ids that have an embedding, sorted (the default candidate pool)."""
    return sorted(
        nid
        for nid in graph.vecs
        if graph.is_type(nid, "pull_request")
    )


def _split_files_tests(graph: GraphView, target_ids):
    """Partition a PR's ``modifies`` targets into (source_files, tests), by path."""
    files, tests = [], []
    for tid in target_ids:
        path = graph.path_of(tid) or tid
        if graph.is_type(tid, "test") or _is_test_path(path):
            tests.append(path)
        else:
            files.append(path)
    return sorted(set(files)), sorted(set(tests))


def retrieve_subgraph(
    issue_id: str,
    graph: GraphView,
    *,
    top_k: int = 5,
    candidate_pr_ids=None,
    max_paths_per_pr: int = 12,
) -> dict:
    """Relation-conditioned retrieval: pack a subgraph around ``issue_id``.

    Returns a deterministic dict with the issue node, the top-k related PR nodes
    (each carrying its cosine score and the source files / tests it ``modifies``),
    the issue's own ``fixes`` neighbours (if the graph records them), and the
    edges that connect them.

    The candidate pool defaults to every PR that has an embedding; pass
    ``candidate_pr_ids`` (e.g. a benchmark query's candidate list) to restrict it.
    Nothing here is learned — it is cosine + graph traversal only.
    """
    if candidate_pr_ids is None:
        candidate_pr_ids = _default_candidate_prs(graph)

    ranked = _rank_candidate_prs(graph, issue_id, candidate_pr_ids, top_k)

    related_prs = []
    edges = []
    for pr_id, score in ranked:
        targets = graph.pr_to_modified.get(pr_id, [])
        files, tests = _split_files_tests(graph, targets)
        files = files[:max_paths_per_pr]
        tests = tests[:max_paths_per_pr]
        related_prs.append({
            "id": pr_id,
            "score": round(score, 6),
            "title": graph.title_of(pr_id),
            "files": files,
            "tests": tests,
        })
        for path in files + tests:
            edges.append((pr_id, MODIFIES, path))

    # The issue's own fixes-neighbours (structural context that exists in the
    # graph independent of the cosine ranking). Listed but de-duplicated against
    # the ranked PRs so callers can see "the graph already attaches these".
    issue_fixers = graph.issue_to_fixers.get(issue_id, [])
    for pr_id in issue_fixers:
        edges.append((pr_id, FIXES, issue_id))

    return {
        "issue": {
            "id": issue_id,
            "title": graph.title_of(issue_id),
            "body": graph.body_of(issue_id),
        },
        "related_prs": related_prs,
        "issue_fixers": list(issue_fixers),
        "edges": edges,
        "params": {"top_k": top_k, "max_paths_per_pr": max_paths_per_pr},
    }


# --------------------------------------------------------------------------- #
# Context packing (deterministic, length-bounded).
# --------------------------------------------------------------------------- #

def _truncate(text: str, limit: int) -> str:
    """Collapse whitespace and cut to ``limit`` chars with an ellipsis marker."""
    one_line = " ".join((text or "").split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: max(0, limit - 1)].rstrip() + "…"


def pack_context(
    subgraph: dict,
    *,
    issue_body_chars: int = 600,
    pr_title_chars: int = 120,
    max_prs: int | None = None,
) -> str:
    """Serialise a retrieved subgraph into one structured, bounded prompt string.

    The layout is stable (issue header, then each related PR with its changed
    files + tests), so the same subgraph always packs byte-identically. The
    string is plain text — ready to drop straight into an SLM prompt — and never
    contains the raw issue/PR *number* in a way the model could string-match
    against (the records were already de-referenced upstream by ``scrub``).
    """
    issue = subgraph["issue"]
    prs = subgraph["related_prs"]
    if max_prs is not None:
        prs = prs[:max_prs]

    lines = []
    lines.append("# ISSUE")
    title = _truncate(issue.get("title", ""), 200) or "(no title)"
    lines.append(f"title: {title}")
    body = _truncate(issue.get("body", ""), issue_body_chars)
    if body:
        lines.append(f"description: {body}")
    lines.append("")
    lines.append(f"# RELATED PULL REQUESTS (top {len(prs)} by relevance)")
    if not prs:
        lines.append("(none retrieved)")
    for rank, pr in enumerate(prs, start=1):
        pr_title = _truncate(pr.get("title", ""), pr_title_chars) or "(no title)"
        lines.append(f"## PR #{rank}  (relevance={pr.get('score', 0.0):.3f})")
        lines.append(f"summary: {pr_title}")
        files = pr.get("files", [])
        tests = pr.get("tests", [])
        lines.append(
            "changed files: " + (", ".join(files) if files else "(none recorded)")
        )
        lines.append(
            "changed tests: " + (", ".join(tests) if tests else "(none recorded)")
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
