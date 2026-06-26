#!/usr/bin/env python3
"""R24a — ingest a SECOND corpus (TypeScript/JavaScript) for external validity.

The diff->test result and the regime characterization are on ONE corpus (Python-
ecosystem pilot). This ingests an independent, DIFFERENT-language corpus so we can
test whether (R24b) the co-change structure that makes diff->test work replicates,
and (R24c) the retrieval result replicates. Genuinely a refutation test.

Per repo it fetches recent merged PRs + their changed files and emits PR/test/file
records + `modifies` edges, reusing the ingest mappers verbatim (schema-/provenance-
clean). Live (needs GITHUB_TOKEN); per-repo checkpointed + resumable; polite. Chunk
with RELSDLC_MAX_REPOS to stay under the background-job time limit.

Run: GITHUB_TOKEN=$GH_PAT RELSDLC_MAX_REPOS=6 python data/corpus2/build_corpus2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent              # data/corpus2
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.ingest import (  # noqa: E402
    GITHUB_API, _request, _respect_rate_limit, _now_iso,
    map_pull_request, map_changed_file, extract_modifies_edges, _is_test_path,
)

# Active TS/JS repos with real test suites — deliberately disjoint ecosystem from the
# Python pilot (vite/vue/svelte/astro/express/prettier/axios/zod/date-fns/trpc/query/hono).
REPOS = [
    "vuejs/core", "vitejs/vite", "sveltejs/svelte", "withastro/astro",
    "expressjs/express", "prettier/prettier", "axios/axios", "colinhacks/zod",
    "date-fns/date-fns", "trpc/trpc", "TanStack/query", "honojs/hono",
]
PAGES = int(os.environ.get("RELSDLC_PAGES", "2"))       # merged PRs/repo ~ PAGES*100
START_PAGE = int(os.environ.get("RELSDLC_START_PAGE", "1"))  # resume at a later page to ADD density
MAX_REPOS = int(os.environ.get("RELSDLC_MAX_REPOS", "99"))
PACE = 0.12
RECS = HERE / "records.jsonl"
EDGES = HERE / "modifies_edges.jsonl"
PROG = HERE / "build-progress.json"


def _jl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()] if p.exists() else []


def _repo_license(repo, token):
    try:
        lic, hdr = _request(f"{GITHUB_API}/repos/{repo}/license", token)
        _respect_rate_limit(hdr)
        spdx = (lic.get("license") or {}).get("spdx_id")
        return spdx if spdx and spdx != "NOASSERTION" else "unknown"
    except Exception:
        return "unknown"


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("set GITHUB_TOKEN", file=sys.stderr); sys.exit(2)
    done = set(json.loads(PROG.read_text()).get("repos", [])) if PROG.exists() else set()
    recs = {r["id"]: r for r in _jl(RECS)}
    edges = {(e["source"], e["target"]): e for e in _jl(EDGES)}
    print(f"corpus2: {len(REPOS)} repos, {len(done)} done; PAGES={PAGES} MAX_REPOS={MAX_REPOS}",
          file=sys.stderr, flush=True)
    ran = 0
    for repo in REPOS:
        if repo in done:
            continue
        if ran >= MAX_REPOS:
            print(f"MAX_REPOS reached — exiting (resumable)", flush=True); break
        print(f"=== {repo} ===", file=sys.stderr, flush=True)
        lic = _repo_license(repo, token); ts = _now_iso(); ne = nr = 0
        for page in range(START_PAGE, PAGES + 1):
            try:
                pulls, hdr = _request(f"{GITHUB_API}/repos/{repo}/pulls?state=closed&per_page=100&page={page}", token)
            except urllib.error.HTTPError as e:
                print(f"  page {page} HTTP {e.code}", file=sys.stderr); break
            _respect_rate_limit(hdr); time.sleep(PACE)
            if not isinstance(pulls, list) or not pulls:
                break
            for pr in pulls:
                if not pr.get("merged_at"):
                    continue
                num = pr["number"]
                try:
                    files, hdr = _request(f"{GITHUB_API}/repos/{repo}/pulls/{num}/files?per_page=100", token)
                except Exception:
                    continue
                _respect_rate_limit(hdr); time.sleep(PACE)
                if not isinstance(files, list):
                    continue
                test_files = [f for f in files if f.get("filename") and _is_test_path(f["filename"])]
                if not test_files:
                    continue
                prrec = map_pull_request(pr, repo, ts, lic)
                if prrec["id"] not in recs:
                    recs[prrec["id"]] = prrec; nr += 1
                for f in test_files:
                    fr = map_changed_file(repo, f["filename"], ts, lic)
                    recs.setdefault(fr["id"], fr)
                for e in extract_modifies_edges(pr, test_files, repo, ts, lic):
                    k = (e["source"], e["target"])
                    if k not in edges:
                        edges[k] = e; ne += 1
        with EDGES.open("w", encoding="utf-8") as fh:
            for e in sorted(edges.values(), key=lambda e: (e["source"], e["target"])):
                fh.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
        with RECS.open("w", encoding="utf-8") as fh:
            for r in sorted(recs.values(), key=lambda r: r["id"]):
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        done.add(repo); ran += 1
        PROG.write_text(json.dumps({"repos": sorted(done), "pages": PAGES}, indent=2) + "\n")
        print(f"  +{ne} edges, +{nr} PR records (totals: {len(edges)} edges, {len(recs)} records)",
              file=sys.stderr, flush=True)
    n_pr = sum(1 for r in recs.values() if r["type"] == "pull_request")
    n_test = sum(1 for r in recs.values() if r["type"] == "test")
    print(f"corpus2 so far: {len(edges)} modifies edges, {n_pr} PRs, {n_test} test nodes, "
          f"{len(done)}/{len(REPOS)} repos", flush=True)


if __name__ == "__main__":
    main()
