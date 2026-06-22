#!/usr/bin/env python3
"""LIVE fetch real FILE/TEST contents for the deep-signal chunking probe (R16A).

R15 found chunking/MaxP does NOT beat FirstP for issue->fixing-PR because that
signal is *front-loaded* (the issue lede carries it). The hypothesis R16A tests:
chunking pays off where the relevant content is **deep** — long source/test FILE
bodies. The pilot graph stores file/test *paths*, not contents, so we fetch the
real file text here.

For each distinct FILE/TEST path in data/pilot/graph/file_records.jsonl whose repo
is **permissively licensed** (MIT / BSD / Apache / PSF / ISC — checked against the
pilot source cards), this GETs the GitHub contents API for the default branch,
base64-decodes, caps at 16000 chars, and writes a schema-valid ``file``/``test``
record with full provenance to data/content/file_contents.jsonl.

Permissive code contents ARE redistributable, so these source cards declare
``redistribution: snippets_permitted`` (NOT the metadata_only of the pilot cards)
and ``content.text`` carries the (capped) file text. Ids are namespaced
``gh-content:owner/repo:file:path`` so they never collide with the pilot's
path-only file nodes.

This script also writes the **diff->affected-test benchmark** from the same fetch:
for each PR (a "diff") that modifies >= 1 TEST whose content we fetched, it emits a
``diff_to_affected_test`` query (query_record = the pilot PR record; candidates =
same-repo ``gh-content:`` test records; relevant = the modified tests). The split
is feasible cross-repo (query PR repo == candidate test repo).

Polite by construction (reuses the ingest helpers): sets a User-Agent, honours
rate-limit headers, paces between requests. 404 / 410 (deleted / renamed paths,
default-branch drift) are skipped gracefully.

Run (LIVE, requires a token)::

    GITHUB_TOKEN=$(gh auth token) python data/content/build_content.py

Without GITHUB_TOKEN this refuses (unauthenticated REST is ~60 req/hr).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.ingest import (  # noqa: E402
    GITHUB_API,
    _request,
    _respect_rate_limit,
    _is_test_path,
    _now_iso,
)

PILOT = REPO_ROOT / "data" / "pilot"
FILE_RECORDS = PILOT / "graph" / "file_records.jsonl"
MODIFIES_EDGES = PILOT / "graph" / "modifies_edges.jsonl"
PILOT_RECORDS = PILOT / "records.jsonl"
SOURCE_CARDS = PILOT / "source-cards.jsonl"
OUT_RECORDS = HERE / "file_contents.jsonl"
OUT_CARDS = HERE / "source-cards.jsonl"
OUT_BENCH = HERE / "benchmark" / "diff_to_affected_test.jsonl"

# Benchmark construction parameters (deterministic).
BENCH_AS_OF = "2026-06-22T02:00:00Z"   # late: candidates/positives never post-date.
BENCH_SEED = 0
BENCH_HARD_NEG = 5     # same-repo near tests as designated hard negatives.
BENCH_RANDOM = 3       # a few same-repo random fillers (beyond the hard negatives).

# Clearly-permissive SPDX ids whose CODE CONTENTS may be redistributed (capped
# snippets). "unknown" / "NOASSERTION" and copyleft are deliberately excluded.
PERMISSIVE = {
    "MIT", "BSD-3-Clause", "BSD-2-Clause", "Apache-2.0", "ISC",
    "PSF-2.0", "0BSD",
}

CONTENT_CAP = 16000  # max chars of file text kept per record.
TRANSFORM = "python data/content/build_content.py"
SLEEP = 0.4          # polite pacing between contents requests.


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n")
            if ln.strip()]


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content_id(repo: str, path: str) -> str:
    return f"gh-content:{repo}:file:{path}"


def _repo_of_file_id(file_id: str) -> str:
    # "gh:owner/repo:file:path" -> "owner/repo"
    return file_id.split(":")[1]


def _path_of_file_record(rec: dict) -> str:
    return (rec.get("content") or {}).get("path", "")


def _licenses() -> dict[str, str]:
    """repo (owner/name) -> SPDX license, from the pilot source cards."""
    out: dict[str, str] = {}
    for card in _load_jsonl(SOURCE_CARDS):
        cid = card.get("id", "")
        if cid.startswith("src:gh:"):
            out[cid[len("src:gh:"):]] = card.get("license", "unknown")
    return out


def _default_branch(repo: str, token: str | None, cache: dict[str, str]) -> str:
    if repo in cache:
        return cache[repo]
    branch = "main"
    try:
        meta, headers = _request(f"{GITHUB_API}/repos/{repo}", token)
        _respect_rate_limit(headers)
        branch = (meta or {}).get("default_branch") or "main"
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, AttributeError):
        pass
    cache[repo] = branch
    return branch


def fetch_content(repo: str, path: str, ref: str, token: str | None) -> str | None:
    """GET the file blob for ``repo/path`` at ``ref``; base64-decode to text.

    Returns the decoded text, or None when the path is missing (404/410) or not a
    plain blob (directory / submodule / undecodable binary).
    """
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}?ref={ref}"
    try:
        body, headers = _request(url, token)
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410):
            return None
        raise
    _respect_rate_limit(headers)
    if not isinstance(body, dict):
        return None  # a directory listing returns a JSON array.
    if body.get("encoding") != "base64" or "content" not in body:
        return None
    try:
        raw = base64.b64decode(body["content"])
        return raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None  # binary / undecodable.


def build_record(repo: str, path: str, ref: str, text: str, license_: str,
                 retrieved_at: str) -> dict:
    rtype = "test" if _is_test_path(path) else "file"
    capped = text[:CONTENT_CAP]
    content = {"path": path, "text": capped}
    blob_url = f"https://github.com/{repo}/blob/{ref}/{path}"
    return {
        "id": _content_id(repo, path),
        "type": rtype,
        "content": content,
        "valid_from": retrieved_at,
        "provenance": {
            "source_url": blob_url,
            "retrieved_at": retrieved_at,
            "license": license_,
            # Hash the canonical content we keep (path + capped text), so the
            # claim "this is the text we ranked" is replayable.
            "content_hash": _hash_text(json.dumps(
                content, sort_keys=True, separators=(",", ":"))),
            "transform": TRANSFORM,
            "method": "git_history",
            "observed": True,
        },
    }


def build_source_card(repo: str, license_: str, retrieved_at: str,
                      n_records: int) -> dict:
    return {
        "card_type": "source",
        "id": f"src:gh-content:{repo}",
        "name": f"GitHub file/test contents — {repo}",
        "source_url": f"https://github.com/{repo}",
        "retrieved_at": retrieved_at,
        "license": license_,
        "terms_note": (
            f"Permissively-licensed ({license_}) source. File/test CONTENTS are "
            "redistributable as capped snippets (<= 16000 chars) with attribution "
            "and license preserved. Polite contents-API ingest: User-Agent set, "
            "rate-limit headers honoured."
        ),
        "record_types": sorted({"file", "test"}),
        "transform": TRANSFORM,
        "redistribution": "snippets_permitted",
        "notes": (
            f"{n_records} file/test bodies fetched from the default branch via the "
            "GitHub contents API, base64-decoded, capped at 16000 chars. Each record "
            "carries a source_url to the blob and the real sha256 of the kept content."
        ),
    }


def build_benchmark(records: list[dict]) -> int:
    """Write the diff->affected-test benchmark from the fetched content records.

    Ground truth = pilot ``modifies`` edges whose TEST target we fetched. Returns
    the number of queries written.
    """
    rng = random.Random(BENCH_SEED)

    pr_ids = {r["id"] for r in _load_jsonl(PILOT_RECORDS)
              if r.get("type") == "pull_request"}

    content_tests_by_repo: dict[str, list[str]] = {}
    fetched_test_ids: set[str] = set()
    for rec in records:
        if rec.get("type") != "test":
            continue
        repo = _repo_of_file_id(rec["id"])
        content_tests_by_repo.setdefault(repo, []).append(rec["id"])
        fetched_test_ids.add(rec["id"])
    for repo in content_tests_by_repo:
        content_tests_by_repo[repo].sort()

    # Pilot path-only TEST node id -> the matching gh-content test id (if fetched).
    pilot_to_content: dict[str, str] = {}
    for rec in _load_jsonl(FILE_RECORDS):
        if rec.get("type") != "test":
            continue
        repo = _repo_of_file_id(rec["id"])
        path = _path_of_file_record(rec)
        cid = _content_id(repo, path)
        if cid in fetched_test_ids:
            pilot_to_content[rec["id"]] = cid

    pr_modifies: dict[str, set[str]] = {}
    for edge in _load_jsonl(MODIFIES_EDGES):
        if edge.get("relation") != "modifies":
            continue
        cid = pilot_to_content.get(edge.get("target"))
        if cid is None:
            continue
        pr_modifies.setdefault(edge["source"], set()).add(cid)

    queries: list[dict] = []
    for pr in sorted(pr_modifies):
        if pr not in pr_ids:
            continue  # query_record must resolve to a pilot PR record.
        repo = _repo_of_file_id(pr)
        relevant = sorted(pr_modifies[pr])
        pool = content_tests_by_repo.get(repo, [])
        others = [t for t in pool if t not in relevant]
        if not others:
            continue  # need >= 1 non-relevant same-repo test to rank against.

        shuffled = list(others)
        rng.shuffle(shuffled)
        hard_negs = shuffled[:BENCH_HARD_NEG]
        random_fill = shuffled[BENCH_HARD_NEG:BENCH_HARD_NEG + BENCH_RANDOM]
        candidates = sorted(set(relevant) | set(hard_negs) | set(random_fill))
        queries.append({
            "query_id": f"q-{pr}",
            "task": "diff_to_affected_test",
            "query_record": pr,
            "candidates": candidates,
            "relevant": relevant,
            "hard_negatives": sorted(hard_negs),
            "as_of": BENCH_AS_OF,
        })

    queries.sort(key=lambda q: q["query_id"])
    OUT_BENCH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_BENCH.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False, sort_keys=True) + "\n")
    repos = sorted({_repo_of_file_id(q["query_record"]) for q in queries})
    print(f"wrote {len(queries)} diff_to_affected_test queries across {len(repos)} "
          f"repos -> {OUT_BENCH.relative_to(REPO_ROOT)}")
    return len(queries)


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "ERROR: this is a LIVE fetch and needs GITHUB_TOKEN.\n"
            "Run: GITHUB_TOKEN=$(gh auth token) python data/content/build_content.py",
            file=sys.stderr,
        )
        return 2

    licenses = _licenses()
    file_records = _load_jsonl(FILE_RECORDS)

    # Distinct (repo, path) for permissively-licensed repos only.
    wanted: dict[str, str] = {}      # content_id -> path (dedup, deterministic order)
    repo_of: dict[str, str] = {}     # content_id -> repo
    skipped_license: set[str] = set()
    for rec in file_records:
        if rec.get("type") not in ("file", "test"):
            continue
        repo = _repo_of_file_id(rec["id"])
        lic = licenses.get(repo, "unknown")
        if lic not in PERMISSIVE:
            skipped_license.add(repo)
            continue
        path = _path_of_file_record(rec)
        if not path:
            continue
        cid = _content_id(repo, path)
        wanted.setdefault(cid, path)
        repo_of[cid] = repo

    ordered = sorted(wanted)
    print(f"permissive repos: {sorted({repo_of[c] for c in ordered})}", file=sys.stderr)
    if skipped_license:
        print(f"skipped non-permissive repos: {sorted(skipped_license)}", file=sys.stderr)
    print(f"distinct file/test paths to fetch: {len(ordered)}", file=sys.stderr)

    retrieved_at = _now_iso()
    branch_cache: dict[str, str] = {}
    records: list[dict] = []
    counts = {"file": 0, "test": 0}
    n_missing = 0
    n_repos_seen: set[str] = set()

    for i, cid in enumerate(ordered, start=1):
        repo = repo_of[cid]
        path = wanted[cid]
        ref = _default_branch(repo, token, branch_cache)
        try:
            text = fetch_content(repo, path, ref, token)
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            print(f"  WARN {repo}/{path}: {exc}", file=sys.stderr)
            text = None
        if text is None or not text.strip():
            # Skip missing (404/410/binary) AND empty marker files (py.typed,
            # empty __init__.py) — an empty body carries no deep signal.
            n_missing += 1
        else:
            rec = build_record(repo, path, ref, text, licenses[repo], retrieved_at)
            records.append(rec)
            counts[rec["type"]] += 1
            n_repos_seen.add(repo)
        if i % 25 == 0:
            print(f"  {i}/{len(ordered)} fetched "
                  f"({counts['file']} file, {counts['test']} test, "
                  f"{n_missing} skipped)", file=sys.stderr)
        time.sleep(SLEEP)

    records.sort(key=lambda r: r["id"])
    OUT_RECORDS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RECORDS.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    # One source card per repo we actually fetched content for.
    per_repo: dict[str, int] = {}
    for r in records:
        repo = _repo_of_file_id(r["id"])
        per_repo[repo] = per_repo.get(repo, 0) + 1
    cards = [
        build_source_card(repo, licenses[repo], retrieved_at, per_repo.get(repo, 0))
        for repo in sorted(n_repos_seen)
    ]
    with OUT_CARDS.open("w", encoding="utf-8") as fh:
        for c in cards:
            fh.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")

    print(
        f"\nwrote {len(records)} content records "
        f"({counts['file']} file, {counts['test']} test) "
        f"across {len(n_repos_seen)} repos; {n_missing} skipped (404/410/binary) "
        f"-> {OUT_RECORDS.relative_to(REPO_ROOT)}"
    )
    print(f"wrote {len(cards)} source cards -> {OUT_CARDS.relative_to(REPO_ROOT)}")

    n_queries = build_benchmark(records)
    print(f"benchmark: {n_queries} diff_to_affected_test queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
