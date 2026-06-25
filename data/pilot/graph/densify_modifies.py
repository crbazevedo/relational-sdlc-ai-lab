#!/usr/bin/env python3
"""R20 Stage 1 — densify the pilot diff->test `modifies` graph (live, one-time).

R16E: diff->affected-test is structure-bound — after the leakage guard removes the
gold (PR,test) edge, 47% of gold test nodes are isolated (only the gold PR modifies
them in the pilot graph) -> R@1 0.009. R17b proved that was an *ingest-depth*
artefact (real co-change median 35 commits/test -> 96.4% reachable ceiling). This
stage converts that structural unlock into an actual retrieval graph: for each of
the 8 TEST-split repos it fetches more merged PRs + their changed files and emits
extra (PR, test) `modifies` edges + the PR/test records (with text) so the candidate
test nodes get real aggregated features for R20 Stage 3.

Only the 8 test-split repos need densifying (the diff->test eval candidates live
there). Reuses the ingest mappers verbatim, so the output is schema-/provenance-clean
and `relsdlc validate` passes. Live (needs GITHUB_TOKEN); per-repo checkpointed +
resumable; polite (User-Agent, rate-limit-aware, paced).

Run: GITHUB_TOKEN=$GH_PAT python data/pilot/graph/densify_modifies.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent              # data/pilot/graph
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data" / "pilot"))

from relsdlc.ingest import (  # noqa: E402
    GITHUB_API, _request, _respect_rate_limit, _now_iso,
    map_pull_request, map_changed_file, extract_modifies_edges, _is_test_path,
)
from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402

PAGES = int(os.environ.get("RELSDLC_PAGES", "2"))    # merged PRs/repo ~ PAGES*100
PACE = 0.12
EDGES_OUT = HERE / "modifies_edges_dense.jsonl"
RECS_OUT = HERE / "records_dense.jsonl"              # new PR + test/file records
PROG = HERE / "densify-progress.json"


def _load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def _repo_license(repo: str, token: str) -> str:
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
        print("set GITHUB_TOKEN (live fetch)", file=sys.stderr)
        sys.exit(2)
    _, meta = load_pilot_crossrepo()
    repos = meta["test_repos"]

    done = set(json.loads(PROG.read_text()).get("repos", [])) if PROG.exists() else set()
    edges = {(e["source"], e["target"]): e for e in _load_jsonl(EDGES_OUT)}
    recs = {r["id"]: r for r in _load_jsonl(RECS_OUT)}
    orig_edges = {(e["source"], e["target"]) for e in _load_jsonl(HERE / "modifies_edges.jsonl")}

    print(f"densify {len(repos)} test repos (PAGES={PAGES}); {len(done)} already done",
          file=sys.stderr, flush=True)
    for repo in repos:
        if repo in done:
            print(f"skip {repo} (done)", file=sys.stderr)
            continue
        print(f"=== {repo} ===", file=sys.stderr, flush=True)
        lic = _repo_license(repo, token)
        ts = _now_iso()
        new_e = new_r = 0
        for page in range(1, PAGES + 1):
            try:
                pulls, hdr = _request(
                    f"{GITHUB_API}/repos/{repo}/pulls?state=closed&per_page=100&page={page}", token)
            except urllib.error.HTTPError as e:
                print(f"  page {page} HTTP {e.code}", file=sys.stderr)
                break
            _respect_rate_limit(hdr)
            time.sleep(PACE)
            if not isinstance(pulls, list) or not pulls:
                break
            for pr in pulls:
                if not pr.get("merged_at"):
                    continue
                num = pr["number"]
                try:
                    files, hdr = _request(
                        f"{GITHUB_API}/repos/{repo}/pulls/{num}/files?per_page=100", token)
                except Exception:
                    continue
                _respect_rate_limit(hdr)
                time.sleep(PACE)
                if not isinstance(files, list):
                    continue
                test_files = [f for f in files if f.get("filename") and _is_test_path(f["filename"])]
                if not test_files:
                    continue
                # PR record (text for embedding) + test-file records + modifies edges
                prrec = map_pull_request(pr, repo, ts, lic)
                if prrec["id"] not in recs:
                    recs[prrec["id"]] = prrec
                    new_r += 1
                for f in test_files:
                    frec = map_changed_file(repo, f["filename"], ts, lic)
                    recs.setdefault(frec["id"], frec)
                for e in extract_modifies_edges(pr, test_files, repo, ts, lic):
                    k = (e["source"], e["target"])
                    if k in orig_edges or k in edges:
                        continue
                    edges[k] = e
                    new_e += 1
        # checkpoint per repo
        with EDGES_OUT.open("w", encoding="utf-8") as fh:
            for e in sorted(edges.values(), key=lambda e: (e["source"], e["target"])):
                fh.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
        with RECS_OUT.open("w", encoding="utf-8") as fh:
            for r in sorted(recs.values(), key=lambda r: r["id"]):
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        done.add(repo)
        PROG.write_text(json.dumps({"repos": sorted(done), "pages": PAGES}, indent=2) + "\n")
        print(f"  +{new_e} edges, +{new_r} PR records (totals: {len(edges)} edges, "
              f"{len(recs)} records)", file=sys.stderr, flush=True)
    print(f"DONE -> {EDGES_OUT.name} ({len(edges)} edges), {RECS_OUT.name} ({len(recs)} records)",
          flush=True)


if __name__ == "__main__":
    main()
