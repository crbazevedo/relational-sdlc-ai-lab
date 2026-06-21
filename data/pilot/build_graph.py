#!/usr/bin/env python3
"""Enrich the committed pilot graph with ``modifies`` edges + file/test records.

This is a LIVE-FETCH, ONE-TIME SNAPSHOT (it touches the network and is not
reproducible — live data moves). CI never runs it; CI validates the committed
output under ``data/pilot/graph/``.

For each DISTINCT fixing-PR id in ``data/pilot/edges.jsonl`` whose PR record
EXISTS in ``data/pilot/records.jsonl``, it fetches the PR's changed files via
``GET /repos/{owner}/{repo}/pulls/{number}/files`` and maps them into:

* ``file`` / ``test`` records (via ``relsdlc.ingest.map_changed_file``), and
* ``modifies`` edges (via ``relsdlc.ingest.extract_modifies_edges``),

so that every ``modifies`` edge ``source`` resolves to an existing PR record and
every ``target`` resolves to a file record we write here. The two streams are
written to ``data/pilot/graph/`` (file-disjoint from the existing pilot
records/edges) so referential integrity holds when ``relsdlc validate data``
scans all of ``data/``.

It is a polite guest: authenticated GitHub REST (~5000 req/hr), a User-Agent,
rate-limit-aware, paced. Body text is not redistributed — only short file paths,
which are kept verbatim. Provenance license is reused from the matching source
card; redistribution is metadata_only.

Run (needs a token):
    GITHUB_TOKEN=$(gh auth token) PYTHONPATH=src python3 data/pilot/build_graph.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.ingest import (  # noqa: E402
    GITHUB_API,
    _now_iso,
    _request,
    _respect_rate_limit,
    extract_modifies_edges,
    map_changed_file,
)

RECORDS = HERE / "records.jsonl"
EDGES = HERE / "edges.jsonl"
SOURCE_CARDS = HERE / "source-cards.jsonl"
OUT_DIR = HERE / "graph"

PER_PAGE = 100
MAX_PAGES = 3             # cap per PR (paths are short; PRs rarely touch >300 files)
SLEEP = 0.35             # polite pacing between requests

# An early date so the file records predate every modifies edge — this keeps the
# temporal-consistency check quiet (edges never precede their endpoints).
FILE_VALID_FROM = "2024-01-01T00:00:00Z"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _license_map() -> dict[str, str]:
    """repo -> SPDX license, reused from the committed source cards."""
    out: dict[str, str] = {}
    if not SOURCE_CARDS.is_file():
        return out
    for card in _read_jsonl(SOURCE_CARDS):
        cid = card.get("id", "")
        prefix = "src:gh:"
        if cid.startswith(prefix):
            out[cid[len(prefix):]] = card.get("license", "unknown")
    return out


def _pr_valid_from() -> dict[str, str]:
    """PR id -> its ``valid_from`` (the PR record's created_at), for edge timing."""
    out: dict[str, str] = {}
    for r in _read_jsonl(RECORDS):
        if r.get("type") == "pull_request" and r.get("valid_from"):
            out[r["id"]] = r["valid_from"]
    return out


def _distinct_fixing_prs(pr_valid_from: dict[str, str]) -> list[str]:
    """Distinct ``modifies``-eligible PR ids: edge sources whose PR record exists."""
    sources: set[str] = set()
    for edge in _read_jsonl(EDGES):
        if edge.get("relation") == "fixes":
            src = edge.get("source")
            if src in pr_valid_from:
                sources.add(src)
    return sorted(sources)


def _parse_pr_id(pid: str) -> tuple[str, int]:
    """``gh:owner/repo:pr:N`` -> ("owner/repo", N)."""
    assert pid.startswith("gh:"), pid
    rest = pid[len("gh:"):]
    repo, _, number = rest.partition(":pr:")
    return repo, int(number)


def _fetch_pr_files(repo: str, number: int, token: str) -> list[dict]:
    """Paginate a PR's changed files (modestly capped). Returns raw file dicts."""
    files: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        url = (
            f"{GITHUB_API}/repos/{repo}/pulls/{number}/files"
            f"?per_page={PER_PAGE}&page={page}"
        )
        body, headers = _request(url, token)
        _respect_rate_limit(headers)
        if not isinstance(body, list) or not body:
            break
        files.extend(body)
        time.sleep(SLEEP)
        if len(body) < PER_PAGE:
            break
    return files


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "ERROR: set GITHUB_TOKEN (e.g. GITHUB_TOKEN=$(gh auth token)).\n"
            "Unauthenticated GitHub REST is rate-limited to ~60 req/hr.",
            file=sys.stderr,
        )
        return 2

    retrieved_at = _now_iso()
    licenses = _license_map()
    pr_valid_from = _pr_valid_from()
    pr_ids = _distinct_fixing_prs(pr_valid_from)
    print(f"querying {len(pr_ids)} fixing-PRs for changed files...", file=sys.stderr)

    file_records: dict[str, dict] = {}
    edges: list[dict] = []
    queried = 0
    skipped = 0
    per_repo: Counter = Counter()

    for pid in pr_ids:
        repo, number = _parse_pr_id(pid)
        license_ = licenses.get(repo, "unknown")
        try:
            raw_files = _fetch_pr_files(repo, number, token)
        except urllib.error.HTTPError as exc:
            # 404 / 410 for deleted or inaccessible PRs: skip gracefully.
            print(f"  skip {pid}: HTTP {exc.code}", file=sys.stderr)
            skipped += 1
            continue
        except urllib.error.URLError as exc:
            print(f"  skip {pid}: {exc.reason}", file=sys.stderr)
            skipped += 1
            continue
        queried += 1

        # File / test records (deduped by id, valid_from pinned early).
        for f in raw_files:
            path = f.get("filename")
            if not path:
                continue
            frec = map_changed_file(repo, path, retrieved_at, license_)
            frec["valid_from"] = FILE_VALID_FROM
            file_records.setdefault(frec["id"], frec)

        # modifies edges: source is the existing PR id; target is a file we write.
        pr_raw = {"number": number, "html_url": f"https://github.com/{repo}/pull/{number}"}
        pr_edges = extract_modifies_edges(pr_raw, raw_files, repo, retrieved_at, license_)
        # Pin edge valid_from to the LATER of the PR's valid_from and the file's
        # early valid_from, so the edge never precedes EITHER endpoint (no
        # temporal.inconsistent warning fires for source PR or target file).
        edge_from = max(pr_valid_from.get(pid, FILE_VALID_FROM), FILE_VALID_FROM)
        for e in pr_edges:
            assert e["source"] == pid, (e["source"], pid)
            e["valid_from"] = edge_from
        edges.extend(pr_edges)
        per_repo[repo] += 1

        if queried % 25 == 0:
            print(f"  ...{queried} PRs queried", file=sys.stderr)

    # Keep only edges whose target file record we actually wrote (referential
    # integrity); every source already corresponds to an existing PR record.
    target_ids = set(file_records)
    edges = [e for e in edges if e["target"] in target_ids]

    sorted_records = [file_records[rid] for rid in sorted(file_records)]
    sorted_edges = sorted(
        edges, key=lambda e: (e["source"], e["relation"], e["target"])
    )

    n_test = sum(1 for r in sorted_records if r["type"] == "test")
    n_file = len(sorted_records) - n_test

    _write_jsonl(OUT_DIR / "file_records.jsonl", sorted_records)
    _write_jsonl(OUT_DIR / "modifies_edges.jsonl", sorted_edges)

    print(
        f"\nGRAPH: queried {queried} PRs ({skipped} skipped), "
        f"{len(sorted_records)} file/test records ({n_file} file, {n_test} test), "
        f"{len(sorted_edges)} modifies edges -> {OUT_DIR}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
