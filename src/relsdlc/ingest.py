"""Map public GitHub REST data into the record/edge schema with provenance.

This module is the public-ingest tooling for Wave R4. It is **offline by
default**: it reads RECORDED GitHub-API JSON responses and transforms them into
schema-conformant records and edges. It only touches the network when
explicitly asked (``--live`` plus a ``GITHUB_TOKEN``), and even then it is
polite (sets a ``User-Agent``, honours rate-limit headers).

Design constraints (Wave R4):

* Pure standard library — ``urllib``, ``json``, ``hashlib``, ``argparse``.
  No new third-party dependencies; the validation/benchmark paths keep their
  existing dependency surface.
* The DEFAULT code path never opens a socket. Live fetch is opt-in and gated.
* Every produced record and edge carries full ``provenance`` (source_url,
  retrieved_at, license, content_hash = sha256 of canonical content, transform,
  method, observed) so ``relsdlc validate`` passes and any claim can be replayed.

Mapping summary
---------------

* GitHub issue JSON  -> ``issue`` record.
* GitHub PR JSON     -> ``pull_request`` record (and ``file`` records for the
  files it changed, when a files payload is present).
* GitHub commit JSON -> ``commit`` record.
* "Fixes/Closes #N" in a PR body -> ``fixes`` edge (method=human_label).
* A PR's changed files            -> ``modifies`` edge (method=git_history).

Run
---

Offline (default), transform a recorded snapshot into records + edges::

    python -m relsdlc.ingest transform \\
        --fixtures tests/ingest_fixtures \\
        --out data/ingest_example

Live (opt-in, requires a token)::

    GITHUB_TOKEN=ghp_xxx python -m relsdlc.ingest fetch \\
        --repo owner/name --out tests/ingest_fixtures --live
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# --- constants ---------------------------------------------------------------

GITHUB_API = "https://api.github.com"
USER_AGENT = "relational-sdlc-ai-lab-ingest/0 (+https://github.com/crbazevedo/relational-sdlc-ai-lab)"
TRANSFORM = "python -m relsdlc.ingest transform"

# "Fixes #12", "Closes: #34", "resolved #5", "fix gh-7", "closes owner/repo#9".
_CLOSING_KEYWORDS = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)"
_ISSUE_REF = re.compile(
    rf"\b{_CLOSING_KEYWORDS}\b[:\s]*"
    r"(?:[\w.\-]+/[\w.\-]+)?(?:#|gh-)(\d+)",
    re.IGNORECASE,
)


# --- canonical hashing (mirrors data/fixtures/build_fixtures.py) -------------


def _hash(payload: dict) -> str:
    """sha256 of the canonical JSON of ``payload`` (sorted keys, tight separators)."""
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- id helpers --------------------------------------------------------------


def issue_id(repo: str, number: int) -> str:
    return f"gh:{repo}:issue:{number}"


def pr_id(repo: str, number: int) -> str:
    return f"gh:{repo}:pr:{number}"


def commit_id(repo: str, sha: str) -> str:
    return f"gh:{repo}:commit:{sha}"


def file_id(repo: str, path: str) -> str:
    return f"gh:{repo}:file:{path}"


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    base = lowered.rsplit("/", 1)[-1]
    return (
        "/test" in lowered
        or lowered.startswith("test")
        or "/tests/" in lowered
        or base.startswith("test_")
        or base.endswith("_test.py")
        or "_test." in base
        or ".test." in base
        or ".spec." in base
    )


# --- record / edge builders --------------------------------------------------


def _provenance(source_url: str, retrieved_at: str, license_: str,
                content: dict, *, method: str | None = None,
                observed: bool = True) -> dict:
    prov = {
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "license": license_,
        "content_hash": _hash(content),
        "transform": TRANSFORM,
        "observed": observed,
    }
    if method is not None:
        prov["method"] = method
    return prov


def map_issue(raw: dict, repo: str, retrieved_at: str, license_: str) -> dict:
    """GitHub issue JSON -> ``issue`` record. (PRs also appear in the issues API;
    callers should filter those out via the ``pull_request`` key.)"""
    number = raw["number"]
    content = {
        "number": number,
        "title": raw.get("title", ""),
        "body": raw.get("body") or "",
        "state": raw.get("state", ""),
        "labels": sorted(_label_names(raw)),
    }
    return {
        "id": issue_id(repo, number),
        "type": "issue",
        "content": content,
        "valid_from": raw.get("created_at") or retrieved_at,
        "provenance": _provenance(
            raw.get("html_url") or f"{GITHUB_API}/repos/{repo}/issues/{number}",
            retrieved_at, license_, content, method="git_history",
        ),
    }


def map_pull_request(raw: dict, repo: str, retrieved_at: str, license_: str) -> dict:
    """GitHub PR JSON -> ``pull_request`` record."""
    number = raw["number"]
    content = {
        "number": number,
        "title": raw.get("title", ""),
        "body": raw.get("body") or "",
        "state": raw.get("state", ""),
        "merged": bool(raw.get("merged_at")),
        "merge_commit_sha": raw.get("merge_commit_sha"),
    }
    return {
        "id": pr_id(repo, number),
        "type": "pull_request",
        "content": content,
        "valid_from": raw.get("created_at") or retrieved_at,
        "provenance": _provenance(
            raw.get("html_url") or f"{GITHUB_API}/repos/{repo}/pulls/{number}",
            retrieved_at, license_, content, method="git_history",
        ),
    }


def map_commit(raw: dict, repo: str, retrieved_at: str, license_: str) -> dict:
    """GitHub commit JSON -> ``commit`` record."""
    sha = raw["sha"]
    commit = raw.get("commit", {})
    content = {
        "sha": sha,
        "message": commit.get("message", ""),
        "author": (commit.get("author") or {}).get("name", ""),
    }
    return {
        "id": commit_id(repo, sha),
        "type": "commit",
        "content": content,
        "valid_from": (commit.get("author") or {}).get("date") or retrieved_at,
        "provenance": _provenance(
            raw.get("html_url") or f"{GITHUB_API}/repos/{repo}/commits/{sha}",
            retrieved_at, license_, content, method="git_history",
        ),
    }


def map_changed_file(repo: str, path: str, retrieved_at: str, license_: str) -> dict:
    """A path touched by a PR -> a ``file`` (or ``test``) record."""
    rtype = "test" if _is_test_path(path) else "file"
    content = {"path": path}
    return {
        "id": file_id(repo, path),
        "type": rtype,
        "content": content,
        "provenance": _provenance(
            f"https://github.com/{repo}/blob/HEAD/{path}",
            retrieved_at, license_, content, method="git_history",
        ),
    }


def extract_fixes_edges(pr_raw: dict, repo: str, retrieved_at: str,
                        license_: str, known_issue_ids: set[str]) -> list[dict]:
    """``fixes`` edges from "Fixes/Closes #N" in a PR body (method=human_label)."""
    number = pr_raw["number"]
    body = pr_raw.get("body") or ""
    src = pr_id(repo, number)
    valid_from = pr_raw.get("created_at") or retrieved_at
    edges: list[dict] = []
    seen: set[int] = set()
    for match in _ISSUE_REF.finditer(body):
        issue_number = int(match.group(1))
        if issue_number == number or issue_number in seen:
            continue
        seen.add(issue_number)
        target = issue_id(repo, issue_number)
        if target not in known_issue_ids:
            # Don't mint dangling edges; referential integrity would reject them.
            continue
        key = {"source": src, "relation": "fixes", "target": target}
        edges.append({
            "source": src,
            "relation": "fixes",
            "target": target,
            # Human-authored closing keyword: high confidence, directly observed.
            "confidence": 0.95,
            "valid_from": valid_from,
            "provenance": _provenance(
                pr_raw.get("html_url") or f"{GITHUB_API}/repos/{repo}/pulls/{number}",
                retrieved_at, license_, key, method="human_label", observed=True,
            ),
        })
    return edges


def extract_modifies_edges(pr_raw: dict, files: list[dict], repo: str,
                           retrieved_at: str, license_: str) -> list[dict]:
    """``modifies`` edges from a PR's changed files (method=git_history)."""
    number = pr_raw["number"]
    src = pr_id(repo, number)
    valid_from = pr_raw.get("created_at") or retrieved_at
    edges: list[dict] = []
    for f in files:
        path = f.get("filename")
        if not path:
            continue
        target = file_id(repo, path)
        key = {"source": src, "relation": "modifies", "target": target}
        edges.append({
            "source": src,
            "relation": "modifies",
            "target": target,
            # Observed from the diff itself: high confidence.
            "confidence": 1.0,
            "valid_from": valid_from,
            "provenance": _provenance(
                pr_raw.get("html_url") or f"{GITHUB_API}/repos/{repo}/pulls/{number}/files",
                retrieved_at, license_, key, method="git_history", observed=True,
            ),
        })
    return edges


def _label_names(raw: dict) -> list[str]:
    out = []
    for label in raw.get("labels", []) or []:
        if isinstance(label, dict):
            name = label.get("name")
        else:
            name = label
        if name:
            out.append(name)
    return out


# --- transform a recorded snapshot ------------------------------------------


def transform_snapshot(snapshot: dict, retrieved_at: str | None = None) -> dict:
    """Transform a recorded snapshot dict into ``{"records": [...], "edges": [...]}``.

    The snapshot shape (as recorded under ``tests/ingest_fixtures/``)::

        {
          "repo": "owner/name",
          "license": "MIT",
          "retrieved_at": "2024-05-01T00:00:00Z",
          "issues": [<github issue json>, ...],   # may include PRs (they carry "pull_request")
          "pulls":  [<github pr json>, ...],
          "commits": [<github commit json>, ...],
          "pull_files": { "<pr_number>": [<github file json>, ...] }
        }

    Returns records + edges sorted deterministically by id / (source,relation,target).
    """
    repo = snapshot["repo"]
    license_ = snapshot.get("license", "unknown")
    retrieved_at = retrieved_at or snapshot.get("retrieved_at") or _now_iso()
    pull_files = snapshot.get("pull_files", {})

    records: dict[str, dict] = {}
    edges: list[dict] = []

    # Issues (skip entries that are actually PRs in the issues feed).
    issue_ids: set[str] = set()
    for raw in snapshot.get("issues", []):
        if raw.get("pull_request"):
            continue
        rec = map_issue(raw, repo, retrieved_at, license_)
        records[rec["id"]] = rec
        issue_ids.add(rec["id"])

    # Commits.
    for raw in snapshot.get("commits", []):
        rec = map_commit(raw, repo, retrieved_at, license_)
        records[rec["id"]] = rec

    # Pull requests + their changed files + edges.
    for raw in snapshot.get("pulls", []):
        rec = map_pull_request(raw, repo, retrieved_at, license_)
        records[rec["id"]] = rec
        number = raw["number"]
        files = pull_files.get(str(number)) or pull_files.get(number) or []
        for f in files:
            path = f.get("filename")
            if not path:
                continue
            frec = map_changed_file(repo, path, retrieved_at, license_)
            # Don't clobber a richer record already present for this path.
            records.setdefault(frec["id"], frec)
        edges.extend(extract_modifies_edges(raw, files, repo, retrieved_at, license_))
        edges.extend(extract_fixes_edges(raw, repo, retrieved_at, license_, issue_ids))

    sorted_records = [records[rid] for rid in sorted(records)]
    sorted_edges = sorted(edges, key=lambda e: (e["source"], e["relation"], e["target"]))
    return {"records": sorted_records, "edges": sorted_edges}


def build_source_card(snapshot: dict, n_records: int) -> dict:
    repo = snapshot["repo"]
    license_ = snapshot.get("license", "unknown")
    retrieved_at = snapshot.get("retrieved_at") or _now_iso()
    record_types = sorted({
        "issue", "pull_request", "commit", "file", "test",
    })
    card = {
        "card_type": "source",
        "id": f"src:gh:{repo}",
        "name": f"GitHub public metadata — {repo}",
        "source_url": f"https://github.com/{repo}",
        "retrieved_at": retrieved_at,
        "license": license_,
        "terms_note": (
            "GitHub public REST metadata. Authenticated REST ~5000 req/hr; "
            "polite ingest sets a User-Agent and honours rate-limit headers. "
            "Only metadata is redistributed in this public repo."
        ),
        "record_types": record_types,
        "transform": TRANSFORM,
        "redistribution": "metadata_only",
        "notes": (
            "Offline example produced from a recorded API snapshot under "
            "tests/ingest_fixtures/. Records carry provenance back to the public "
            "GitHub URLs; full source text is not redistributed."
        ),
    }
    return card


# --- io ----------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_snapshot(fixtures_dir: Path) -> dict:
    """Load a recorded snapshot from a fixtures directory.

    Accepts either a single ``snapshot.json`` (the whole shape) or the separate
    ``issues.json`` / ``pulls.json`` / ``commits.json`` / ``pull_files.json``
    plus a ``meta.json`` carrying repo/license/retrieved_at.
    """
    single = fixtures_dir / "snapshot.json"
    if single.is_file():
        return json.loads(single.read_text(encoding="utf-8"))

    meta_path = fixtures_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"no snapshot.json or meta.json under {fixtures_dir}; "
            "expected a recorded API snapshot"
        )
    snapshot = json.loads(meta_path.read_text(encoding="utf-8"))

    def _maybe(name: str, key: str) -> None:
        p = fixtures_dir / name
        if p.is_file():
            snapshot[key] = json.loads(p.read_text(encoding="utf-8"))

    _maybe("issues.json", "issues")
    _maybe("pulls.json", "pulls")
    _maybe("commits.json", "commits")
    _maybe("pull_files.json", "pull_files")
    return snapshot


# --- live fetch (opt-in, polite) --------------------------------------------


def _request(url: str, token: str | None) -> tuple[dict | list, dict]:
    """One polite GitHub REST GET. Returns (json, response-headers)."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (https only, opt-in)
        body = json.loads(resp.read().decode("utf-8"))
        headers = {k.lower(): v for k, v in resp.headers.items()}
    return body, headers


def _respect_rate_limit(headers: dict) -> None:
    """Sleep if the remaining rate-limit budget is exhausted."""
    remaining = headers.get("x-ratelimit-remaining")
    reset = headers.get("x-ratelimit-reset")
    try:
        if remaining is not None and int(remaining) <= 0 and reset is not None:
            wait = max(0, int(reset) - int(time.time())) + 1
            print(f"rate limit reached; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    except (TypeError, ValueError):
        pass


def fetch_repo_snapshot(repo: str, token: str | None, *,
                        max_items: int = 30, sleep: float = 0.5) -> dict:
    """LIVE: fetch a small snapshot of a public repo's issues/PRs/commits.

    Polite by construction: sets a User-Agent, honours rate-limit headers, and
    pauses ``sleep`` seconds between page requests. Requires ``--live`` at the
    CLI; without it this function is never called.
    """
    snapshot: dict = {
        "repo": repo,
        "license": "unknown",
        "retrieved_at": _now_iso(),
        "issues": [],
        "pulls": [],
        "commits": [],
        "pull_files": {},
    }

    # License (best effort).
    try:
        lic, headers = _request(f"{GITHUB_API}/repos/{repo}/license", token)
        _respect_rate_limit(headers)
        spdx = (lic.get("license") or {}).get("spdx_id")
        if spdx and spdx != "NOASSERTION":
            snapshot["license"] = spdx
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError):
        pass

    issues, headers = _request(
        f"{GITHUB_API}/repos/{repo}/issues?state=all&per_page={max_items}", token)
    _respect_rate_limit(headers)
    snapshot["issues"] = [i for i in issues if not i.get("pull_request")]
    time.sleep(sleep)

    pulls, headers = _request(
        f"{GITHUB_API}/repos/{repo}/pulls?state=all&per_page={max_items}", token)
    _respect_rate_limit(headers)
    snapshot["pulls"] = pulls
    time.sleep(sleep)

    for pr in pulls:
        number = pr["number"]
        files, headers = _request(
            f"{GITHUB_API}/repos/{repo}/pulls/{number}/files?per_page={max_items}", token)
        _respect_rate_limit(headers)
        snapshot["pull_files"][str(number)] = files
        time.sleep(sleep)

    commits, headers = _request(
        f"{GITHUB_API}/repos/{repo}/commits?per_page={max_items}", token)
    _respect_rate_limit(headers)
    snapshot["commits"] = commits
    return snapshot


# --- CLI ---------------------------------------------------------------------


def _cmd_transform(args: argparse.Namespace) -> int:
    fixtures = Path(args.fixtures)
    snapshot = load_snapshot(fixtures)
    result = transform_snapshot(snapshot, retrieved_at=args.retrieved_at)
    out = Path(args.out)
    _write_jsonl(out / "records.jsonl", result["records"])
    _write_jsonl(out / "edges.jsonl", result["edges"])
    if args.source_card:
        _write_json(out / "source-card.json",
                    build_source_card(snapshot, len(result["records"])))
    print(
        f"transformed {snapshot.get('repo')}: "
        f"{len(result['records'])} records, {len(result['edges'])} edges -> {out}"
    )
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    if not args.live:
        print(
            "ERROR: live fetch requires --live (offline by default).\n"
            "Use `python -m relsdlc.ingest transform` to map a recorded snapshot,\n"
            "or pass --live with GITHUB_TOKEN set to fetch from the network.",
            file=sys.stderr,
        )
        return 2
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "WARNING: --live without GITHUB_TOKEN; unauthenticated GitHub REST is "
            "rate-limited to ~60 req/hr. Set GITHUB_TOKEN for ~5000 req/hr.",
            file=sys.stderr,
        )
    snapshot = fetch_repo_snapshot(args.repo, token, max_items=args.max_items)
    out = Path(args.out)
    _write_json(out / "snapshot.json", snapshot)
    print(
        f"fetched {args.repo}: {len(snapshot['issues'])} issues, "
        f"{len(snapshot['pulls'])} pulls, {len(snapshot['commits'])} commits -> {out}",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relsdlc.ingest",
        description="Map public GitHub REST data into the record/edge schema "
                    "with provenance. Offline by default; live fetch is opt-in.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_t = sub.add_parser(
        "transform",
        help="OFFLINE: transform a recorded API snapshot into records.jsonl + edges.jsonl",
    )
    p_t.add_argument("--fixtures", default="tests/ingest_fixtures",
                     help="directory with a recorded snapshot (default: tests/ingest_fixtures)")
    p_t.add_argument("--out", default="data/ingest_example",
                     help="output directory (default: data/ingest_example)")
    p_t.add_argument("--retrieved-at", default=None,
                     help="override the retrieved_at timestamp (default: snapshot's)")
    p_t.add_argument("--source-card", action="store_true",
                     help="also write a source-card.json for the repo")
    p_t.set_defaults(func=_cmd_transform)

    p_f = sub.add_parser(
        "fetch",
        help="LIVE (opt-in): fetch a small public-repo snapshot to a fixtures dir",
    )
    p_f.add_argument("--repo", required=True, help="owner/name of a public repo")
    p_f.add_argument("--out", default="tests/ingest_fixtures",
                     help="output directory for the recorded snapshot")
    p_f.add_argument("--live", action="store_true",
                     help="actually touch the network (required for fetch)")
    p_f.add_argument("--max-items", type=int, default=30,
                     help="page size / cap per resource (default: 30)")
    p_f.set_defaults(func=_cmd_fetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
