#!/usr/bin/env python3
"""R17b — crack the diff->test structural ceiling by measuring real co-change depth.

R16E found diff->affected-test is structure-bound: after the leakage guard removes
the gold (query-PR, test) edge, **46.9%** of gold test nodes are degree-0 in the
PILOT modifies graph (only the gold PR modifies them), capping reachable recall at
**59.8%**. R16E argued the limiter is co-change *density* — an ingest artefact of
fetching few PRs per repo — not the method. This wave tests that argument directly
against ground truth: for every gold test file on the held-out split, how many
distinct changes in the repo's real history actually touch it?

It is a polite, targeted, ONE-TIME live snapshot (like the Tier-2 build): one
GitHub `commits?path=` call per distinct gold test file (~110 calls), capped at
100 commits/file. It does NOT mutate the frozen pilot graph — it writes a separate
co-change snapshot and a deterministic recompute, leaving every committed artefact
other experiments depend on untouched.

Stage 1 (live, needs GITHUB_TOKEN):  fetch -> data/pilot/graph/diff2test-cochange.json
Stage 2 (deterministic, offline):    recompute -> data/pilot/diff2test-density-results.json

Run:  GITHUB_TOKEN=$GH_PAT PYTHONPATH=src python data/pilot/densify_diff2test.py
      PYTHONPATH=src python data/pilot/densify_diff2test.py --recompute-only
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402
from run_gnn_ablation import load_diff2test_crossrepo, load_graph_edges  # noqa: E402

API = "https://api.github.com"
COCHANGE = HERE / "graph" / "diff2test-cochange.json"
RESULTS = HERE / "diff2test-density-results.json"
PER_PAGE = 100  # one page is plenty to separate "isolated" from "co-changed"


# --------------------------------------------------------------- live fetch
def gh_get(url: str, token: str):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "relational-sdlc-ai-lab/diff2test-cochange")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            remaining = r.headers.get("X-RateLimit-Remaining")
            reset = r.headers.get("X-RateLimit-Reset")
            if remaining is not None and int(remaining) == 0 and reset:
                time.sleep(max(0, int(reset) - int(time.time())) + 1)
            return json.loads(r.read().decode("utf-8")), 200
    except urllib.error.HTTPError as e:
        return None, e.code
    except Exception:
        return None, -1


def parse_node(node_id: str):
    """'gh:owner/repo:file:path/to/x.py' -> ('owner/repo', 'path/to/x.py')."""
    parts = node_id.split(":", 3)
    return parts[1], parts[3]


def gold_test_nodes_and_pr_dates():
    """Distinct gold test nodes on the test split + each test node's gold-PR dates.

    The gold-PR date (the modifies-edge valid_from) lets us exclude the gold PR's
    own change when counting co-change, mirroring the R16E leakage guard.
    """
    _, meta = load_pilot_crossrepo()
    diff_ds = load_diff2test_crossrepo(set(meta["train_repos"]))
    edges_pr_date: dict[tuple[str, str], str] = {}
    fixes_edges, modifies_edges = load_graph_edges()
    # valid_from per (pr, target) from the raw modifies edges
    raw = [json.loads(l) for l in (HERE / "graph" / "modifies_edges.jsonl")
           .read_text(encoding="utf-8").split("\n") if l.strip()]
    for e in raw:
        edges_pr_date[(e["source"], e["target"])] = e.get("valid_from", "")
    nodes: dict[str, set[str]] = {}
    for q in diff_ds.queries:
        if q.split != "test" or not q.relevant:
            continue
        for t in q.relevant:
            d = edges_pr_date.get((q.query_record, t), "")
            nodes.setdefault(t, set()).add(d[:10])  # gold-PR date (yyyy-mm-dd)
    return diff_ds, nodes


def fetch(token: str):
    diff_ds, nodes = gold_test_nodes_and_pr_dates()
    out = {}
    keys = sorted(nodes)
    print(f"fetching co-change history for {len(keys)} distinct gold test files ...")
    for i, node in enumerate(keys, 1):
        repo, path = parse_node(node)
        url = f"{API}/repos/{repo}/commits?path={urllib.parse.quote(path)}&per_page={PER_PAGE}"
        js, status = gh_get(url, token)
        commits = []
        if status == 200 and isinstance(js, list):
            for c in js:
                d = (c.get("commit", {}).get("committer", {}) or {}).get("date", "")
                commits.append({"sha": c["sha"][:12], "date": d[:10]})
        out[node] = {
            "repo": repo, "path": path, "status": status,
            "n_commits": len(commits), "capped": len(commits) >= PER_PAGE,
            "gold_pr_dates": sorted(nodes[node]),
            "commits": commits,
        }
        if i % 20 == 0:
            print(f"  {i}/{len(keys)} (last: {repo} n_commits={len(commits)} status={status})")
        time.sleep(0.12)  # polite pacing
    COCHANGE.write_text(json.dumps({"per_page": PER_PAGE, "nodes": out}, indent=2) + "\n",
                        encoding="utf-8")
    print(f"wrote {COCHANGE}  ({len(out)} files)")
    return out


# --------------------------------------------------- deterministic recompute
def recompute():
    snap = json.loads(COCHANGE.read_text(encoding="utf-8"))["nodes"]
    _, meta = load_pilot_crossrepo()
    diff_ds = load_diff2test_crossrepo(set(meta["train_repos"]))

    def other_commits(node) -> int:
        """Commits touching the file on dates other than the gold PR's (non-gold co-change)."""
        info = snap.get(node)
        if not info:
            return 0
        gold = set(info.get("gold_pr_dates", []))
        return sum(1 for c in info["commits"] if c["date"] not in gold)

    def n_commits(node) -> int:
        info = snap.get(node)
        return info["n_commits"] if info else 0

    # Recompute the R16E diagnostic under real co-change. A gold test is reachable
    # if it is touched by >=1 commit OTHER than the gold PR's change (date-excluded);
    # we also report the looser ">=2 distinct commits" view for robustness.
    pair_total = pair_iso_strict = pair_iso_loose = 0
    q_total = q_iso_strict = 0
    depths = []
    for q in diff_ds.queries:
        if q.split != "test" or not q.relevant:
            continue
        q_total += 1
        rel_iso_strict = 0
        for t in q.relevant:
            pair_total += 1
            depths.append(n_commits(t))
            if other_commits(t) < 1:
                pair_iso_strict += 1
                rel_iso_strict += 1
            if n_commits(t) < 2:
                pair_iso_loose += 1
        if rel_iso_strict == len(q.relevant):
            q_iso_strict += 1

    depths.sort()
    median = depths[len(depths) // 2] if depths else 0
    res = {
        "method": "live commits?path co-change; gold-PR date excluded (strict) ",
        "per_page_cap": snap and json.loads(COCHANGE.read_text())["per_page"],
        "pair_total": pair_total,
        "R16E_baseline": {"pair_isolation_rate": 0.4688, "reachable_ceiling": 0.5982},
        "strict_nongold": {
            "pair_isolated": pair_iso_strict,
            "pair_isolation_rate": round(pair_iso_strict / pair_total, 4),
            "query_all_isolated": q_iso_strict,
            "reachable_ceiling": round(1 - q_iso_strict / q_total, 4),
        },
        "loose_ge2_commits": {
            "pair_isolated": pair_iso_loose,
            "pair_isolation_rate": round(pair_iso_loose / pair_total, 4),
        },
        "cochange_depth": {
            "median_commits_per_gold_test": median,
            "max": max(depths) if depths else 0,
            "files_touched_by_1_commit": sum(1 for d in depths if d <= 1),
            "files_capped_at_100": sum(1 for v in snap.values() if v.get("capped")),
            "files_missing_path_404": sum(1 for v in snap.values() if v["status"] != 200 or v["n_commits"] == 0),
        },
    }
    RESULTS.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    b = res["R16E_baseline"]; s = res["strict_nongold"]
    print("R17b — diff->test co-change density (real history) vs R16E pilot graph")
    print(f"  gold (PR,test) pairs on test split: {pair_total}")
    print(f"  R16E (pilot ingest)  : isolation {b['pair_isolation_rate']:.1%}  "
          f"reachable ceiling {b['reachable_ceiling']:.1%}")
    print(f"  real co-change (strict, gold-date excluded):")
    print(f"      isolation {s['pair_isolation_rate']:.1%}  "
          f"reachable ceiling {s['reachable_ceiling']:.1%}  "
          f"(ceiling {s['reachable_ceiling']-b['reachable_ceiling']:+.1%})")
    print(f"  loose (>=2 commits)  : isolation {res['loose_ge2_commits']['pair_isolation_rate']:.1%}")
    print(f"  co-change depth: median {median} commits/gold-test, "
          f"max {res['cochange_depth']['max']}, "
          f"{res['cochange_depth']['files_touched_by_1_commit']} touched by <=1, "
          f"{res['cochange_depth']['files_capped_at_100']} capped at 100")
    print(f"  wrote {RESULTS}")


def main():
    if "--recompute-only" in sys.argv:
        recompute()
        return
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("set GITHUB_TOKEN (the fetch is live); or pass --recompute-only", file=sys.stderr)
        sys.exit(2)
    fetch(token)
    recompute()


if __name__ == "__main__":
    main()
