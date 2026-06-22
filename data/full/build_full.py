#!/usr/bin/env python3
"""Build the FULL-TEXT GitHub dataset (Wave R14) + the issue->fixing-PR benchmark.

This is the de-truncated sibling of ``data/pilot/build_pilot.py``. Same 20 repos,
same schema / provenance / relation mining / cross-repo-capable benchmark — the
ONLY substantive change is the body-text policy:

* pilot  : bodies truncated to 500 chars, ``redistribution = metadata_only``.
* full   : bodies kept up to 8000 chars, ``redistribution = full_text``.

Why: the pilot truncates at 500 chars — BELOW the embedder's window (max_length
256 tokens ~= 1000+ chars) and BELOW the bag-of-tokens reach (which tokenizes the
whole string). So every prior number was measured on text the model could only
partly see. This dataset re-ingests the full descriptions so the lift from the
extra text can be MEASURED (data/full/run_full_ablation.py).

Redistribution posture: ``full_text``. The redistributed content is public
issue/PR prose from permissively-licensed (MIT/BSD/Apache/PSF) repos, kept for
research use, with provenance back to the source URLs. This follows the
GH-Archive precedent (public GitHub event/issue text is routinely redistributed
for research). The 8000-char cap captures ~all real human-written descriptions
while bounding pathological machine-pasted logs.

Records keep a DISTINCT id namespace (``gh-full:owner/repo:...``) so the frozen
full dataset and the frozen pilot dataset can coexist under ``data/`` and both
pass ``relsdlc validate`` (which dedups record ids across the whole tree). The
``owner/repo`` segment is still the second ``:``-field, so the cross-repo split
logic (data/full/run_full_ablation.py) parses it unchanged.

Run (needs a token):
    GITHUB_TOKEN=$(gh auth token) python data/full/build_full.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CARDS = REPO_ROOT / "data" / "cards" / "examples"
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.baseline import tokenize  # noqa: E402
from relsdlc.ingest import (  # noqa: E402
    GITHUB_API, _now_iso, _request, _respect_rate_limit, transform_snapshot,
)

# The SAME 20 repos as data/pilot/build_pilot.py — the comparison is full-vs-
# truncated on the same corpus, not a different corpus.
REPOS = [
    "pytest-dev/pytest", "fastapi/fastapi", "pydantic/pydantic", "psf/requests",
    "pallets/flask", "pallets/click", "pallets/jinja", "encode/httpx",
    "encode/starlette", "encode/uvicorn", "psf/black", "Textualize/rich",
    "fastapi/typer", "python-attrs/attrs", "tox-dev/tox", "python-poetry/poetry",
    "astral-sh/ruff", "pallets/werkzeug", "python-pillow/Pillow", "scrapy/scrapy",
]

# Distinct id namespace so data/full and data/pilot coexist without colliding.
ID_PREFIX = "gh-full"

BODY_CAP = 8000           # chars of body kept (full_text; bounds pathological logs)
FALLBACK_CAP = 4000       # used if records.jsonl would exceed the size budget
SIZE_BUDGET_BYTES = 20 * 1024 * 1024  # ~20 MB target for records.jsonl
PAGES = 2                 # pages per resource (matches the pilot)
PER_PAGE = 100
HARD_NEGATIVES = 8
RANDOM_NEGATIVES = 4
TRAIN_FRAC = 0.6


def _paginate(path: str, token: str, pages: int = PAGES) -> list:
    out: list = []
    for page in range(1, pages + 1):
        sep = "&" if "?" in path else "?"
        url = f"{GITHUB_API}{path}{sep}per_page={PER_PAGE}&page={page}"
        body, headers = _request(url, token)
        _respect_rate_limit(headers)
        if not isinstance(body, list) or not body:
            break
        out.extend(body)
        time.sleep(0.4)  # polite pacing between pages
        if len(body) < PER_PAGE:
            break
    return out


def _cap_bodies(snapshot: dict, cap: int) -> None:
    """De-truncation: keep bodies up to ``cap`` chars (vs the pilot's 500)."""
    for key in ("issues", "pulls"):
        for raw in snapshot.get(key, []):
            b = raw.get("body") or ""
            if len(b) > cap:
                raw["body"] = b[:cap]


def fetch_repo(repo: str, token: str, retrieved_at: str, cap: int) -> dict:
    snapshot = {"repo": repo, "license": "unknown", "retrieved_at": retrieved_at,
                "issues": [], "pulls": [], "commits": [], "pull_files": {}}
    try:
        lic, headers = _request(f"{GITHUB_API}/repos/{repo}/license", token)
        _respect_rate_limit(headers)
        spdx = (lic.get("license") or {}).get("spdx_id")
        if spdx and spdx != "NOASSERTION":
            snapshot["license"] = spdx
    except Exception:  # noqa: BLE001
        pass
    issues = _paginate(f"/repos/{repo}/issues?state=closed&sort=updated&direction=desc", token)
    snapshot["issues"] = [i for i in issues if not i.get("pull_request")]
    snapshot["pulls"] = _paginate(
        f"/repos/{repo}/pulls?state=closed&sort=updated&direction=desc", token)
    _cap_bodies(snapshot, cap)
    return snapshot


def _renamespace(record_id: str) -> str:
    """gh:owner/repo:issue:N -> gh-full:owner/repo:issue:N (keep owner/repo as field 1)."""
    if record_id.startswith("gh:"):
        return ID_PREFIX + ":" + record_id[len("gh:"):]
    return record_id


def _renamespace_record(rec: dict) -> dict:
    rec = dict(rec)
    rec["id"] = _renamespace(rec["id"])
    return rec


def _renamespace_edge(edge: dict) -> dict:
    edge = dict(edge)
    edge["source"] = _renamespace(edge["source"])
    edge["target"] = _renamespace(edge["target"])
    return edge


def _text(rec: dict) -> str:
    c = rec.get("content", {})
    return f"{c.get('title', '')} {c.get('body', '')}".strip()


def build_benchmark(records: list[dict], edges: list[dict], rng_seed: int = 0):
    """issue->fixing-PR queries with same-repo hard negatives + a temporal split."""
    import random
    rng = random.Random(rng_seed)

    by_id = {r["id"]: r for r in records}
    prs_by_repo: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r["type"] == "pull_request":
            prs_by_repo[r["id"].split(":pr:")[0]].append(r["id"])

    fixes = [(e["source"], e["target"]) for e in edges if e["relation"] == "fixes"]
    fixes = [(pr, iss) for pr, iss in fixes if pr in by_id and iss in by_id]

    # Temporal split by issue creation time.
    fixes_sorted = sorted(fixes, key=lambda pi: by_id[pi[1]].get("valid_from", ""))
    n_train = int(len(fixes_sorted) * TRAIN_FRAC)
    split_of_issue = {}
    for i, (pr, iss) in enumerate(fixes_sorted):
        split_of_issue[iss] = "train" if i < n_train else "test"

    queries = []
    for pr, iss in fixes_sorted:
        repo = iss.split(":issue:")[0]
        pool = [p for p in prs_by_repo[repo] if p != pr]
        if len(pool) < 5:
            continue
        issue_toks = set(tokenize(_text(by_id[iss])))
        ranked = sorted(
            pool,
            key=lambda p: (-len(issue_toks & set(tokenize(_text(by_id[p])))), p),
        )
        hard = ranked[:HARD_NEGATIVES]
        rest = ranked[HARD_NEGATIVES:]
        rnd = rng.sample(rest, min(RANDOM_NEGATIVES, len(rest))) if rest else []
        candidates = [pr] + hard + rnd
        rng.shuffle(candidates)
        queries.append({
            "query_id": f"q-{iss}",
            "task": "issue_to_fixing_pr",
            "query_record": iss,
            "candidates": candidates,
            "relevant": [pr],
            "hard_negatives": hard,
        })
    return queries, split_of_issue


def _assemble(snapshots: dict, cap: int):
    """Transform snapshots -> renamespaced records + fixes edges + benchmark."""
    all_records: dict[str, dict] = {}
    all_edges: list[dict] = []
    licenses: dict[str, str] = {}
    retrieved_at = next(iter(snapshots.values()))["retrieved_at"]

    for repo in REPOS:
        snap = snapshots[repo]
        # Re-cap (cheap) so a fallback cap takes effect without re-fetching.
        _cap_bodies(snap, cap)
        licenses[repo] = snap["license"]
        result = transform_snapshot(snap, retrieved_at=retrieved_at)
        for rec in result["records"]:
            rec = _renamespace_record(rec)
            all_records[rec["id"]] = rec
        all_edges.extend(_renamespace_edge(e)
                         for e in result["edges"] if e["relation"] == "fixes")

    records = list(all_records.values())
    queries, split_of_issue = build_benchmark(records, all_edges)

    # Stamp record split (issues by time; PRs follow the issue they fix).
    pr_split = {}
    for q in queries:
        pr_split[q["relevant"][0]] = split_of_issue.get(q["query_record"], "train")
    for r in records:
        if r["id"] in split_of_issue:
            r["split"] = split_of_issue[r["id"]]
        elif r["id"] in pr_split:
            r["split"] = pr_split[r["id"]]

    # Keep only records referenced by the benchmark + their fixes edges.
    used_ids = set()
    for q in queries:
        used_ids.add(q["query_record"])
        used_ids.update(q["candidates"])
    records = [r for r in records if r["id"] in used_ids]
    kept_ids = {r["id"] for r in records}
    edges = [e for e in all_edges if e["source"] in kept_ids and e["target"] in kept_ids]
    return records, edges, queries, split_of_issue, licenses, retrieved_at


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: set GITHUB_TOKEN (e.g. GITHUB_TOKEN=$(gh auth token))", file=sys.stderr)
        raise SystemExit(2)
    retrieved_at = _now_iso()

    # Fetch once (the expensive, network part) at the full cap.
    snapshots: dict[str, dict] = {}
    for repo in REPOS:
        snap = fetch_repo(repo, token, retrieved_at, BODY_CAP)
        snapshots[repo] = snap
        print(f"  {repo:<28} {snap['license']:<12} "
              f"issues={len(snap['issues'])} pulls={len(snap['pulls'])}",
              file=sys.stderr)

    cap = BODY_CAP
    records, edges, queries, split_of_issue, licenses, retrieved_at = _assemble(snapshots, cap)

    # Size guard: if records.jsonl would exceed the budget, lower the cap and note it.
    def _records_bytes(recs) -> int:
        return sum(len(json.dumps(r, ensure_ascii=False, sort_keys=True)) + 1 for r in recs)

    if _records_bytes(records) > SIZE_BUDGET_BYTES:
        cap = FALLBACK_CAP
        print(f"NOTE: records exceed {SIZE_BUDGET_BYTES // (1024*1024)} MB at cap "
              f"{BODY_CAP}; lowering body cap to {FALLBACK_CAP}.", file=sys.stderr)
        records, edges, queries, split_of_issue, licenses, retrieved_at = _assemble(snapshots, cap)

    _write_jsonl(HERE / "records.jsonl", sorted(records, key=lambda r: r["id"]))
    _write_jsonl(HERE / "edges.jsonl",
                 sorted(edges, key=lambda e: (e["source"], e["relation"], e["target"])))
    _write_json(HERE / "split.json", {
        "method": "temporal-by-issue-created",
        "train_frac": TRAIN_FRAC,
        "train": sorted(i for i, s in split_of_issue.items() if s == "train"),
        "test": sorted(i for i, s in split_of_issue.items() if s == "test"),
    })

    # AS_OF postdates every candidate so the legitimate (later) fixing PR is visible.
    as_of = retrieved_at
    for q in queries:
        q["as_of"] = as_of
    _write_jsonl(HERE / "benchmark" / "issue_to_fixing_pr.jsonl",
                 sorted(queries, key=lambda q: q["query_id"]))

    _write_source_cards(licenses, retrieved_at, cap)
    _write_dataset_card(records, edges, queries, retrieved_at, cap)

    size_mb = (HERE / "records.jsonl").stat().st_size / (1024 * 1024)
    print(f"\nFULL: {len(records)} records, {len(edges)} fixes edges, "
          f"{len(queries)} queries across {len(REPOS)} repos "
          f"(body cap={cap} chars, records.jsonl={size_mb:.1f} MB; "
          f"train issues={sum(1 for s in split_of_issue.values() if s=='train')}, "
          f"test={sum(1 for s in split_of_issue.values() if s=='test')})", file=sys.stderr)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _terms_note(cap: int) -> str:
    return ("GitHub public REST metadata; closed issues + closed PRs. "
            "Authenticated REST ~5000 req/hr; polite, paced ingest. "
            f"Full body text kept (up to {cap} chars) for research use; "
            "provenance links back to the public source URLs.")


def _write_source_cards(licenses: dict, retrieved_at: str, cap: int) -> None:
    rows = []
    for repo, lic in sorted(licenses.items()):
        rows.append({
            "card_type": "source",
            "id": f"src:gh-full:{repo}",
            "name": f"GitHub public full-text — {repo}",
            "source_url": f"https://github.com/{repo}",
            "retrieved_at": retrieved_at,
            "license": lic,
            "terms_note": _terms_note(cap),
            "record_types": ["issue", "pull_request"],
            "transform": "python data/full/build_full.py",
            "redistribution": "full_text",
            "notes": "Full issue/PR text from a permissively-licensed repo, "
                     "redistributed for research use (GH-Archive precedent for "
                     "public GitHub text). Records carry provenance to source URLs.",
        })
    _write_jsonl(HERE / "source-cards.jsonl", rows)


def _write_dataset_card(records, edges, queries, retrieved_at, cap) -> None:
    card = {
        "card_type": "dataset",
        "id": "ds:gh-full-v0",
        "name": "public GitHub issue->fixing-PR (full text)",
        "version": "v0",
        "created_at": retrieved_at,
        "sources": [f"src:gh-full:{r}" for r in REPOS],
        "record_counts": dict(sorted(Counter(r["type"] for r in records).items())),
        "edge_counts": dict(sorted(Counter(e["relation"] for e in edges).items())),
        "relation_types": sorted({e["relation"] for e in edges}),
        "split_policy": {
            "frozen": True,
            "method": "temporal-by-issue-created",
            "seed": 0,
            "boundary": f"earliest {int(TRAIN_FRAC*100)}% of fixes by issue date = train",
        },
        "redistribution": "full_text",
        "known_limitations": [
            "Closed issues + closed PRs from ~20 permissive Python-ecosystem repos; "
            "fixes edges mined from closing keywords (recall-limited).",
            f"Body text kept up to {cap} chars (full_text), vs the pilot's 500-char "
            "truncation; long machine-pasted logs are still bounded by the cap.",
            "Pilot scale; a one-time live snapshot, not reproducible from CI.",
        ],
        "notes": "Built by data/full/build_full.py (live GitHub REST). The de-truncated "
                 "sibling of ds:gh-pilot-v0 (same 20 repos). Redistribution is full_text "
                 "(public issue/PR prose, research use, GH-Archive precedent). Hard "
                 "negatives are same-repo PRs by title/body token overlap. Ids use the "
                 "gh-full: namespace so the full and pilot frozen versions coexist.",
    }
    _write_json(CARDS / "gh-full-v0.dataset-card.json", card)


if __name__ == "__main__":
    main()
