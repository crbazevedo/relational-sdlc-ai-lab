#!/usr/bin/env python3
"""Build the real **Tier-2 dataset** (~120-150 permissive repos) + the
issue->fixing-PR benchmark.

Wave R16B (Track D scale). This grows the 55-repo ``data/scale/`` "Tier-2
entry" to a real **Tier-2 entry of ~120-150 permissive repos**, written as a
NEW dataset under ``data/tier2/`` so the relational result can be re-confirmed
at a larger scale. The pilot and scale snapshots are left untouched (other
experiments depend on them).

Like the pilot/scale, this is a LIVE-FETCH, ONE-TIME SNAPSHOT (it touches the
network and is not reproducible — live data moves). CI never runs it; CI
validates the committed snapshot and re-runs the (deterministic) ablation.

It is a polite guest: authenticated GitHub REST (~5000 req/hr), a User-Agent,
rate-limit-aware, paced. It fetches only CLOSED issues + CLOSED pulls metadata
(no per-PR file calls), maps them into the record/edge schema via the R4 ingest
tooling, mines ``fixes`` edges from closing keywords, truncates body text to
**500 chars** (R14's paired control proved 500 chars *outperforms* full text for
this task AND keeps size bounded; redistribution = metadata_only), and freezes a
temporal train/test split.

NAMESPACE. The record ids are minted in the dedicated ``gh-t2:owner/repo:...``
namespace (NOT ``gh:...``), so they never collide with the pilot / scale / full
snapshots on the global ``relsdlc validate data`` whole-tree scan. The
``owner/repo`` stays the 2nd ``:``-field so the cross-repo split parser
(``record_id.split(":")[1]``) still recovers the repo.

HARD SIZE BUDGET. ``data/tier2/records.jsonl`` must stay under ~12 MB. If the
lean snapshot would exceed it, whole repos (lowest fixes-yield first) are dropped
until it fits and the benchmark is rebuilt on the survivors.

Run (needs a token):
    GITHUB_TOKEN=$(gh auth token) python data/tier2/build_tier2.py
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

# The id namespace for THIS dataset. Keeping the ``owner/repo`` as the 2nd
# ``:``-field means ``record_id.split(":")[1]`` still recovers the repo (the
# cross-repo split parser in run_tier2_ablation.load_tier2_crossrepo), while the
# ``gh-t2`` prefix keeps every id disjoint from pilot/scale/full on the global
# ``validate data`` scan.
NS = "gh-t2"

# The 55 repos from data/scale/ (the pilot's 20 + ~35) PLUS ~70-95 more across
# Python / JS / TS / Go / Rust — all permissive (MIT/BSD/Apache/ISC/PSF), test-
# heavy, active. Repos that error (404 / moved / rate-limited) OR aren't
# permissive are SKIPPED at fetch time, so this is a candidate set, not a hard
# requirement.

# --- the 55 from data/scale/ -------------------------------------------------
SCALE_REPOS = [
    # pilot's 20
    "pytest-dev/pytest", "fastapi/fastapi", "pydantic/pydantic", "psf/requests",
    "pallets/flask", "pallets/click", "pallets/jinja", "encode/httpx",
    "encode/starlette", "encode/uvicorn", "psf/black", "Textualize/rich",
    "fastapi/typer", "python-attrs/attrs", "tox-dev/tox", "python-poetry/poetry",
    "astral-sh/ruff", "pallets/werkzeug", "python-pillow/Pillow", "scrapy/scrapy",
    # scale's +35
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
    # scale's permissive non-Python
    "expressjs/express", "lodash/lodash", "chalk/chalk", "sindresorhus/got",
    "gin-gonic/gin", "spf13/cobra", "serde-rs/serde", "clap-rs/clap",
]

# --- ~95 NEW permissive, test-heavy, active repos for Tier-2 -----------------
NEW_REPOS = [
    # More Python (permissive)
    "tornadoweb/tornado", "benoitc/gunicorn", "celery/celery", "celery/kombu",
    "celery/billiard", "redis/redis-py", "boto/boto3", "boto/botocore",
    "psycopg/psycopg2", "psycopg/psycopg", "MagicStack/asyncpg",
    "coleifer/peewee", "fabric/fabric", "paramiko/paramiko", "gevent/gevent",
    "joblib/joblib", "tqdm/tqdm", "click-contrib/click-completion",
    "pyca/bcrypt", "pyca/pynacl", "scikit-image/scikit-image",
    "scipy/scipy", "statsmodels/statsmodels", "pyca/pyopenssl",
    "Pylons/pyramid", "Pylons/waitress", "Pylons/webob",
    "python-pillow/Pillow-SIMD", "MongoEngine/mongoengine",
    "kennethreitz/records", "miguelgrinberg/Flask-SocketIO",
    "pytest-dev/pytest-asyncio", "pytest-dev/pytest-xdist",
    "pytest-dev/pytest-mock", "Suor/funcy", "pytoolz/toolz",
    "wtforms/wtforms", "marshmallow-code/apispec", "marshmallow-code/webargs",
    "graphql-python/graphene", "strawberry-graphql/strawberry",
    "encode/django-rest-framework", "tiangolo/typer-cli",
    "facebookresearch/hydra", "omry/omegaconf", "python-rope/rope",
    "PyCQA/flake8", "PyCQA/isort", "PyCQA/pylint", "PyCQA/bandit",
    "PyCQA/pyflakes", "PyCQA/pycodestyle", "asottile/pyupgrade",
    "pre-commit/pre-commit", "jaraco/keyring", "platformdirs/platformdirs",
    "pytest-dev/execnet", "agronholm/apscheduler",
    # JS / TS (permissive)
    "axios/axios", "expressjs/morgan", "expressjs/body-parser",
    "sindresorhus/ky", "sindresorhus/p-map", "sindresorhus/execa",
    "remy/nodemon", "motdotla/dotenv", "validatorjs/validator.js",
    "moment/moment", "date-fns/date-fns", "iamkun/dayjs",
    "jquense/yup", "colinhacks/zod", "winstonjs/winston",
    "websockets/ws", "socketio/socket.io", "fastify/fastify",
    "koajs/koa", "hapijs/hapi", "nodejs/undici", "node-fetch/node-fetch",
    "jsdom/jsdom", "cheeriojs/cheerio", "Unitech/pm2",
    "TanStack/query", "reduxjs/redux", "pmndrs/zustand",
    # Go (permissive)
    "gorilla/mux", "go-chi/chi", "labstack/echo", "gofiber/fiber",
    "sirupsen/logrus", "uber-go/zap", "stretchr/testify",
    "google/uuid", "spf13/viper", "spf13/afero", "urfave/cli",
    "go-yaml/yaml", "json-iterator/go", "valyala/fasthttp",
    "jmoiron/sqlx", "go-gorm/gorm", "patrickmn/go-cache",
    # Rust (permissive)
    "tokio-rs/tokio", "tokio-rs/bytes", "tokio-rs/tracing",
    "serde-rs/json", "rust-lang/regex", "rayon-rs/rayon",
    "BurntSushi/ripgrep", "sharkdp/fd", "sharkdp/bat",
    "rust-random/rand", "hyperium/hyper", "seanmonstar/reqwest",
    "dtolnay/anyhow", "dtolnay/thiserror", "clap-rs/clap_derive",
]

REPOS = SCALE_REPOS + NEW_REPOS

# Target ~120-150 repos in the committed snapshot. We over-list candidates so
# that skips (non-permissive / error / empty) still leave us in range, and cap
# defensively in main() so we don't blow the size budget.
MAX_REPOS = 80   # density over breadth (operator): fewer repos, deeper per-repo coverage
MAX_RECORDS_BYTES = 18_000_000  # ~18 MB ceiling (denser per-repo coverage -> more records)

# Permissive SPDX ids we accept (MIT/BSD/Apache/ISC/PSF families). GitHub's
# /license endpoint returns an SPDX id; "unknown" (NOASSERTION or no LICENSE
# file auto-classified) is accepted ONLY for the curated scale repos that are
# well-known permissive projects whose LICENSE GitHub does not map to one SPDX
# id (numpy's BSD-variant, cryptography's Apache-OR-BSD, matplotlib's PSF-style,
# etc.). Any NEW repo that resolves to a non-permissive or unknown license is
# skipped.
PERMISSIVE_SPDX = {
    "MIT", "MIT-0", "BSD-2-Clause", "BSD-3-Clause", "BSD-3-Clause-Clear",
    "Apache-2.0", "ISC", "PSF-2.0", "0BSD", "Unlicense", "Zlib",
    "BSD-2-Clause-Patent",
}
# Curated allow-list of scale repos whose GitHub /license is "unknown" but which
# are well-known permissive projects (kept for continuity with data/scale/).
KNOWN_PERMISSIVE_UNKNOWN = {
    "numpy/numpy", "pyca/cryptography", "matplotlib/matplotlib", "python/mypy",
    "sympy/sympy", "python-trio/trio", "pypa/packaging", "certifi/python-certifi",
    "networkx/networkx", "HypothesisWorks/hypothesis", "python-pillow/Pillow",
    "scipy/scipy", "scikit-image/scikit-image",
}

BODY_TRUNC = 500          # chars of body kept (R14 control: 500 > full for this task)
PAGES = 5                 # density: up to ~500 closed issues + ~500 closed PRs per repo
PER_PAGE = 100
HARD_NEGATIVES = 8
RANDOM_NEGATIVES = 4
TRAIN_FRAC = 0.6


def _ns_id(rid: str) -> str:
    """Re-namespace a ``gh:owner/repo:type:N`` id into ``gh-t2:owner/repo:type:N``."""
    if rid.startswith("gh:"):
        return NS + rid[2:]
    return rid


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


def _repo_license(repo: str, token: str) -> str:
    try:
        lic, headers = _request(f"{GITHUB_API}/repos/{repo}/license", token)
        _respect_rate_limit(headers)
        spdx = (lic.get("license") or {}).get("spdx_id")
        if spdx and spdx != "NOASSERTION":
            return spdx
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _is_permissive(repo: str, license_: str) -> bool:
    if license_ in PERMISSIVE_SPDX:
        return True
    if license_ == "unknown" and repo in KNOWN_PERMISSIVE_UNKNOWN:
        return True
    return False


def fetch_repo(repo: str, token: str, retrieved_at: str) -> dict:
    snapshot = {"repo": repo, "license": "unknown", "retrieved_at": retrieved_at,
                "issues": [], "pulls": [], "commits": [], "pull_files": {}}
    snapshot["license"] = _repo_license(repo, token)
    issues = _paginate(f"/repos/{repo}/issues?state=closed&sort=updated&direction=desc", token)
    snapshot["issues"] = [i for i in issues if not i.get("pull_request")]
    snapshot["pulls"] = _paginate(
        f"/repos/{repo}/pulls?state=closed&sort=updated&direction=desc", token)
    _truncate_bodies(snapshot)
    return snapshot


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


def _repo_of(rid: str) -> str:
    return rid.split(":issue:")[0].split(":pr:")[0]


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: set GITHUB_TOKEN (e.g. GITHUB_TOKEN=$(gh auth token))", file=sys.stderr)
        raise SystemExit(2)
    retrieved_at = _now_iso()

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
        if not _is_permissive(repo, snap["license"]):
            skipped.append(f"{repo} (license={snap['license']})")
            print(f"  SKIP {repo}: non-permissive ({snap['license']})", file=sys.stderr)
            continue
        if not snap["issues"] and not snap["pulls"]:
            skipped.append(f"{repo} (empty)")
            print(f"  SKIP {repo}: empty", file=sys.stderr)
            continue
        licenses[repo] = snap["license"]
        ingested_repos.append(repo)
        result = transform_snapshot(snap, retrieved_at=retrieved_at)
        for rec in result["records"]:
            rec["id"] = _ns_id(rec["id"])
            all_records[rec["id"]] = rec
        n_fixes = 0
        for e in result["edges"]:
            if e["relation"] != "fixes":
                continue
            e["source"] = _ns_id(e["source"])
            e["target"] = _ns_id(e["target"])
            all_edges.append(e)
            n_fixes += 1
        per_repo_fixes[repo] = n_fixes
        print(f"  {repo:<36} {snap['license']:<14} "
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

        def _ns_repo(rid: str) -> str:
            # "gh-t2:owner/repo:issue:N" -> "owner/repo"
            return rid.split(":", 1)[1].split(":issue:")[0].split(":pr:")[0]

        while repo_rank and _records_bytes(
            [r for r in records if _ns_repo(r["id"]) in keep_repos]
        ) > MAX_RECORDS_BYTES:
            keep_repos.discard(repo_rank.pop(0))
        # Re-derive everything from the surviving repos.
        survivor_recs = [r for r in all_records.values() if _ns_repo(r["id"]) in keep_repos]
        survivor_edges = [e for e in all_edges
                          if _ns_repo(e["source"]) in keep_repos
                          and _ns_repo(e["target"]) in keep_repos]
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
        print(f"  size-cap: dropped to {len(ingested_repos)} repos to fit "
              f"{MAX_RECORDS_BYTES} bytes", file=sys.stderr)

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

    print(f"\nTIER-2: {len(records)} records, {len(edges)} fixes edges, "
          f"{len(queries)} queries across {len(ingested_repos)} repos "
          f"(train issues={sum(1 for s in split_of_issue.values() if s=='train')}, "
          f"test={sum(1 for s in split_of_issue.values() if s=='test')})", file=sys.stderr)
    print(f"records.jsonl bytes: {_records_bytes(records):,}", file=sys.stderr)
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
            "id": f"src:{NS}:{repo}",
            "name": f"GitHub public metadata — {repo}",
            "source_url": f"https://github.com/{repo}",
            "retrieved_at": retrieved_at,
            "license": lic,
            "terms_note": "GitHub public REST metadata; closed issues + closed PRs. "
                          "Authenticated REST ~5000 req/hr; polite, paced ingest. "
                          "Body text truncated to 500 chars; only metadata is redistributed.",
            "record_types": ["issue", "pull_request"],
            "transform": "python data/tier2/build_tier2.py",
            "redistribution": "metadata_only",
        })
    _write_jsonl(HERE / "source-cards.jsonl", rows)


def _write_dataset_card(records, edges, queries, repos, retrieved_at) -> None:
    card = {
        "card_type": "dataset",
        "id": "ds:gh-tier2-v0",
        "name": "public GitHub issue->fixing-PR Tier-2 scale dataset (~150 repos)",
        "version": "v0",
        "created_at": retrieved_at,
        "sources": [f"src:{NS}:{r}" for r in sorted(repos)],
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
            "(MIT/BSD/Apache/ISC/PSF) repos across Python/JS/TS/Go/Rust; fixes "
            "edges mined from closing keywords (recall-limited).",
            "Body text truncated to 500 chars; metadata only, not full source. "
            "R14's paired control showed 500-char truncation outperforms full text "
            "for this task while keeping size bounded.",
            "Tier-2 scale; a one-time live snapshot, not reproducible from CI. Ids "
            "are minted in the gh-t2: namespace so they stay disjoint from the "
            "pilot/scale/full snapshots on the global validate-data scan.",
            "Re-confirms the bag-of-tokens baselines (vanilla / IDF / diagonal) at "
            "Tier-2 scale; the LoRA-at-Tier-2 run is a torch follow-up owned by central.",
        ],
        "notes": "Built by data/tier2/build_tier2.py (live GitHub REST). Grows the "
                 "55-repo data/scale/ entry (ds:gh-scale2-v0) to a real ~150-repo "
                 "Tier-2 entry. Hard negatives are same-repo PRs by title/body token "
                 "overlap.",
    }
    _write_json(CARDS / "gh-tier2-v0.dataset-card.json", card)


if __name__ == "__main__":
    main()
