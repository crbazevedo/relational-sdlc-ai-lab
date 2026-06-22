#!/usr/bin/env python3
"""Build the Tier-2-entry SCALE GitHub dataset + the issue->fixing-PR benchmark.

Wave R12A (Track D scale). This grows the frozen 20-repo pilot
(``data/pilot/``) to a ~55-repo "Tier-2 entry" so the relational result can be
re-confirmed at larger scale, written as a NEW dataset under ``data/scale/``.
The pilot is left untouched (other experiments depend on it).

Like the pilot, this is a LIVE-FETCH, ONE-TIME SNAPSHOT (it touches the network
and is not reproducible — live data moves). CI never runs it; CI validates the
committed snapshot and re-runs the (deterministic) ablation on it.

It is a polite guest: authenticated GitHub REST (~5000 req/hr), a User-Agent,
rate-limit-aware, paced. It fetches only CLOSED issues + CLOSED pulls metadata
(no per-PR file calls), maps them into the record/edge schema via the R4 ingest
tooling, mines ``fixes`` edges from closing keywords, truncates body text
(redistribution = metadata_only), and freezes a temporal train/test split.

The ids stay in the ``gh:owner/repo:...`` namespace. Because ``relsdlc validate
data`` validates the WHOLE data tree at once, any scale record whose id already
exists in the committed pilot snapshot is dropped (and the benchmark is rebuilt
only from the surviving records) so the two datasets never collide on ids.

Run (needs a token):
    GITHUB_TOKEN=$(gh auth token) python data/scale/build_scale.py
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
PILOT = REPO_ROOT / "data" / "pilot"
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.baseline import tokenize  # noqa: E402
from relsdlc.ingest import (  # noqa: E402
    GITHUB_API, _now_iso, _request, _respect_rate_limit, transform_snapshot,
)

# The pilot's 20 repos PLUS ~35 more permissive (MIT/BSD/Apache), test-heavy,
# active repos — mostly the wider Python ecosystem plus a few JS/Go/Rust with
# permissive licenses. Repos that error (404 / moved / rate-limited) are SKIPPED
# at fetch time, so this list is a candidate set, not a hard requirement.
PILOT_REPOS = [
    "pytest-dev/pytest", "fastapi/fastapi", "pydantic/pydantic", "psf/requests",
    "pallets/flask", "pallets/click", "pallets/jinja", "encode/httpx",
    "encode/starlette", "encode/uvicorn", "psf/black", "Textualize/rich",
    "fastapi/typer", "python-attrs/attrs", "tox-dev/tox", "python-poetry/poetry",
    "astral-sh/ruff", "pallets/werkzeug", "python-pillow/Pillow", "scrapy/scrapy",
]

EXTRA_REPOS = [
    # Python data / scientific ecosystem (permissive).
    "sqlalchemy/sqlalchemy", "numpy/numpy", "pandas-dev/pandas",
    "scikit-learn/scikit-learn", "aio-libs/aiohttp", "python/mypy",
    "HypothesisWorks/hypothesis", "pydantic/pydantic-core",
    "ewels/rich-click", "yaml/pyyaml", "urllib3/urllib3", "certifi/python-certifi",
    "pypa/packaging", "pypa/virtualenv", "pypa/pip", "pypa/setuptools",
    "pypa/wheel", "pyca/cryptography", "lxml/lxml",
    "matplotlib/matplotlib", "mwaskom/seaborn", "networkx/networkx",
    "sympy/sympy", "dask/dask", "sdispater/pendulum", "arrow-py/arrow",
    "Delgan/loguru", "jd/tenacity", "encode/httpcore", "agronholm/anyio",
    "python-trio/trio", "samuelcolvin/dirty-equals", "Textualize/textual",
    "more-itertools/more-itertools", "theskumar/python-dotenv",
    "encode/databases", "pallets/markupsafe", "python-jsonschema/jsonschema",
    "pytest-dev/pytest-cov", "tiangolo/sqlmodel", "marshmallow-code/marshmallow",
    "pytest-dev/pluggy", "wimglenn/johnnydep",
    # A few permissive non-Python ecosystems.
    "expressjs/express", "lodash/lodash", "chalk/chalk", "sindresorhus/got",
    "gin-gonic/gin", "spf13/cobra", "serde-rs/serde", "clap-rs/clap",
]

REPOS = PILOT_REPOS + EXTRA_REPOS

# Cap the number of repos that actually land in the committed snapshot so
# records.jsonl stays well under the ~8 MB budget. The pilot kept ~2087 records
# (~2.1 MB) from 20 repos; ~55 repos at the same density would be ~5-6 MB. We
# still cap defensively in main() by size.
MAX_REPOS = 55
MAX_RECORDS_BYTES = 7_500_000  # ~7.5 MB ceiling for the committed records.jsonl

BODY_TRUNC = 500          # chars of body kept (metadata_only, embedding-sufficient)
PAGES = 2                 # pages per resource
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
        time.sleep(0.4)
        if len(body) < PER_PAGE:
            break
    return out


def _truncate_bodies(snapshot: dict) -> None:
    for key in ("issues", "pulls"):
        for raw in snapshot.get(key, []):
            b = raw.get("body") or ""
            if len(b) > BODY_TRUNC:
                raw["body"] = b[:BODY_TRUNC]


def fetch_repo(repo: str, token: str, retrieved_at: str) -> dict:
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
    _truncate_bodies(snapshot)
    return snapshot


def _text(rec: dict) -> str:
    c = rec.get("content", {})
    return f"{c.get('title', '')} {c.get('body', '')}".strip()


def _pilot_ids() -> set[str]:
    """Ids already committed in the pilot snapshot — kept disjoint from scale.

    ``relsdlc validate data`` validates the whole tree at once, so a scale id
    that collides with a pilot id would be a duplicate-id error. We keep the
    ``gh:owner/repo:...`` namespace and simply drop any colliding id here.
    """
    path = PILOT / "records.jsonl"
    if not path.is_file():
        return set()
    return {json.loads(l)["id"]
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


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
    # Keep only fixes whose issue + PR records both exist.
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


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: set GITHUB_TOKEN (e.g. GITHUB_TOKEN=$(gh auth token))", file=sys.stderr)
        raise SystemExit(2)
    retrieved_at = _now_iso()
    pilot_ids = _pilot_ids()

    all_records: dict[str, dict] = {}
    all_edges: list[dict] = []
    licenses: dict[str, str] = {}
    per_repo_fixes: Counter = Counter()
    ingested_repos: list[str] = []
    skipped: list[str] = []

    for repo in REPOS:
        if len(ingested_repos) >= MAX_REPOS:
            break
        try:
            snap = fetch_repo(repo, token, retrieved_at)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{repo} ({type(exc).__name__})")
            print(f"  SKIP {repo}: {type(exc).__name__}", file=sys.stderr)
            continue
        if not snap["issues"] and not snap["pulls"]:
            skipped.append(f"{repo} (empty)")
            print(f"  SKIP {repo}: empty", file=sys.stderr)
            continue
        licenses[repo] = snap["license"]
        ingested_repos.append(repo)
        result = transform_snapshot(snap, retrieved_at=retrieved_at)
        for rec in result["records"]:
            # Keep scale disjoint from the committed pilot snapshot.
            if rec["id"] in pilot_ids:
                continue
            all_records[rec["id"]] = rec
        n_fixes = sum(1 for e in result["edges"] if e["relation"] == "fixes")
        per_repo_fixes[repo] = n_fixes
        all_edges.extend(e for e in result["edges"] if e["relation"] == "fixes")
        print(f"  {repo:<32} {snap['license']:<12} "
              f"issues={len(snap['issues'])} pulls={len(snap['pulls'])} fixes={n_fixes}",
              file=sys.stderr)

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

    # Keep only records referenced by the benchmark + their fixes edges (lean snapshot).
    used_ids = set()
    for q in queries:
        used_ids.add(q["query_record"])
        used_ids.update(q["candidates"])
    records = [r for r in records if r["id"] in used_ids]
    kept_ids = {r["id"] for r in records}
    edges = [e for e in all_edges
             if e["source"] in kept_ids and e["target"] in kept_ids]

    records = sorted(records, key=lambda r: r["id"])

    # Defensive size cap: if the lean snapshot still exceeds the byte budget,
    # drop whole repos (lowest fixes-yield first) until it fits, then rebuild
    # the benchmark on the survivors so referential integrity stays intact.
    def _records_bytes(recs) -> int:
        return sum(len(json.dumps(r, ensure_ascii=False, sort_keys=True)) + 1 for r in recs)

    if _records_bytes(records) > MAX_RECORDS_BYTES:
        repo_rank = [r for r, _ in per_repo_fixes.most_common()]
        repo_rank.reverse()  # weakest-yield repos dropped first
        keep_repos = set(ingested_repos)
        while repo_rank and _records_bytes(
            [r for r in records if r["id"].split(":issue:")[0].split(":pr:")[0] in keep_repos]
        ) > MAX_RECORDS_BYTES:
            keep_repos.discard(repo_rank.pop(0))
        # Re-derive everything from the surviving repos.
        def _repo_of(rid: str) -> str:
            return rid.split(":issue:")[0].split(":pr:")[0]
        survivor_recs = [r for r in all_records.values() if _repo_of(r["id"]) in keep_repos]
        survivor_edges = [e for e in all_edges
                          if _repo_of(e["source"]) in keep_repos
                          and _repo_of(e["target"]) in keep_repos]
        queries, split_of_issue = build_benchmark(survivor_recs, survivor_edges)
        pr_split = {}
        for q in queries:
            pr_split[q["relevant"][0]] = split_of_issue.get(q["query_record"], "train")
        for r in survivor_recs:
            if r["id"] in split_of_issue:
                r["split"] = split_of_issue[r["id"]]
            elif r["id"] in pr_split:
                r["split"] = pr_split[r["id"]]
        used_ids = set()
        for q in queries:
            used_ids.add(q["query_record"])
            used_ids.update(q["candidates"])
        records = sorted((r for r in survivor_recs if r["id"] in used_ids),
                         key=lambda r: r["id"])
        kept_ids = {r["id"] for r in records}
        edges = [e for e in survivor_edges
                 if e["source"] in kept_ids and e["target"] in kept_ids]
        ingested_repos = [r for r in ingested_repos if r in keep_repos]
        licenses = {r: lic for r, lic in licenses.items() if r in keep_repos}

    _write_jsonl(HERE / "records.jsonl", records)
    _write_jsonl(HERE / "edges.jsonl",
                 sorted(edges, key=lambda e: (e["source"], e["relation"], e["target"])))

    # AS_OF must postdate every candidate so the legitimate (later) fixing PR is
    # visible — the temporal guard here is the train/test split, not as_of.
    as_of = retrieved_at
    for q in queries:
        q["as_of"] = as_of
    _write_jsonl(HERE / "benchmark" / "issue_to_fixing_pr.jsonl",
                 sorted(queries, key=lambda q: q["query_id"]))
    _write_json(HERE / "split.json", {
        "method": "temporal-by-issue-created",
        "train_frac": TRAIN_FRAC,
        "train": sorted(i for i, s in split_of_issue.items() if s == "train"),
        "test": sorted(i for i, s in split_of_issue.items() if s == "test"),
    })

    _write_source_cards(licenses, retrieved_at)
    _write_dataset_card(records, edges, queries, ingested_repos, retrieved_at)

    print(f"\nSCALE: {len(records)} records, {len(edges)} fixes edges, "
          f"{len(queries)} queries across {len(ingested_repos)} repos "
          f"(train issues={sum(1 for s in split_of_issue.values() if s=='train')}, "
          f"test={sum(1 for s in split_of_issue.values() if s=='test')})", file=sys.stderr)
    if skipped:
        print(f"skipped {len(skipped)} repos: {', '.join(skipped)}", file=sys.stderr)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_source_cards(licenses: dict, retrieved_at: str) -> None:
    rows = []
    for repo, lic in sorted(licenses.items()):
        rows.append({
            "card_type": "source",
            "id": f"src:gh:{repo}",
            "name": f"GitHub public metadata — {repo}",
            "source_url": f"https://github.com/{repo}",
            "retrieved_at": retrieved_at,
            "license": lic,
            "terms_note": "GitHub public REST metadata; closed issues + closed PRs. "
                          "Authenticated REST ~5000 req/hr; polite, paced ingest. "
                          "Body text truncated; only metadata is redistributed.",
            "record_types": ["issue", "pull_request"],
            "transform": "python data/scale/build_scale.py",
            "redistribution": "metadata_only",
        })
    _write_jsonl(HERE / "source-cards.jsonl", rows)


def _write_dataset_card(records, edges, queries, repos, retrieved_at) -> None:
    card = {
        "card_type": "dataset",
        "id": "ds:gh-scale2-v0",
        "name": "public GitHub issue->fixing-PR Tier-2-entry scale dataset",
        "version": "v0",
        "created_at": retrieved_at,
        "sources": [f"src:gh:{r}" for r in sorted(repos)],
        "record_counts": dict(sorted(Counter(r["type"] for r in records).items())),
        "edge_counts": dict(sorted(Counter(e["relation"] for e in edges).items())),
        "relation_types": sorted({e["relation"] for e in edges}),
        "split_policy": {
            "frozen": True,
            "method": "temporal-by-issue-created",
            "seed": 0,
            "boundary": f"earliest {int(TRAIN_FRAC*100)}% of fixes by issue date = train",
        },
        "redistribution": "metadata_only",
        "known_limitations": [
            f"Closed issues + closed PRs from {len(repos)} permissive "
            "(MIT/BSD/Apache) repos — the wider Python ecosystem plus a few "
            "JS/Go/Rust; fixes edges mined from closing keywords (recall-limited).",
            "Body text truncated to 500 chars; metadata only, not full source.",
            "Tier-2-entry scale; a one-time live snapshot, not reproducible from CI. "
            "Ids that collide with the frozen pilot snapshot are dropped so the two "
            "datasets stay id-disjoint.",
            "Re-confirms the bag-of-tokens baselines (vanilla / IDF / diagonal) at "
            "scale; embeddings / LoRA at scale are a torch follow-up.",
        ],
        "notes": "Built by data/scale/build_scale.py (live GitHub REST). "
                 "Grows the 20-repo pilot (ds:gh-pilot-v0) to a ~55-repo Tier-2 "
                 "entry. Hard negatives are same-repo PRs by title/body token overlap.",
    }
    _write_json(CARDS / "gh-scale2-v0.dataset-card.json", card)


if __name__ == "__main__":
    main()
