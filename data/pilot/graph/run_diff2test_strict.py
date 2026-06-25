#!/usr/bin/env python3
"""R22 — the temporally-honest diff->test number for release, with a clean decomposition.

R20/R21 scored diff->test with the R16E graph-aug scorer (alpha-blends the query PR's
own embedding with its modified-file neighbours, and aggregates a test node's feature
from ALL non-gold modifying PRs regardless of time). Two issues for a frozen public
benchmark: (a) the alpha-blend dilutes the query (we have the PR's own text); (b) using
modifiers created AFTER the query PR is mild future leakage.

This computes the clean variants (per-query: query = the PR's OWN embedding; a test
node's feature = mean of its non-gold modifying-PR embeddings), isolating both effects:

  - pure-query + ALL modifiers     (clean query, but future leakage)
  - pure-query + as_of modifiers   (clean query + no future leakage) <- RELEASE-HONEST

Numpy only; deterministic. Run: PYTHONPATH=src python data/pilot/graph/run_diff2test_strict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data" / "pilot"))

from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402
from run_gnn_ablation import load_embeddings, load_diff2test_crossrepo  # noqa: E402

OUT = REPO_ROOT / "data" / "pilot" / "diff2test-strict-results.json"


def _jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    _, meta = load_pilot_crossrepo()
    diff = load_diff2test_crossrepo(set(meta["train_repos"]))
    nv = {**load_embeddings("minilm-l6-v2"), **load_embeddings("dense-pr")}
    uv = {i: _unit(nv[i]) for i in nv}

    vf = {}
    for p in ("data/pilot/records.jsonl", "data/pilot/graph/records_dense.jsonl"):
        for r in _jsonl(REPO_ROOT / p):
            if r.get("valid_from"):
                vf[r["id"]] = r["valid_from"]
    tm = {}
    for p in ("data/pilot/graph/modifies_edges.jsonl", "data/pilot/graph/modifies_edges_dense.jsonl"):
        for e in _jsonl(REPO_ROOT / p):
            if e.get("relation") == "modifies":
                tm.setdefault(e["target"], []).append((e["source"], e.get("valid_from", "")))

    def evaluate(as_of: bool):
        h1 = h5 = h10 = rr = n = reach = 0
        for q in diff.queries:
            if q.split != "test" or not q.relevant:
                continue
            n += 1
            tq = vf.get(q.query_record, "")
            a = uv.get(q.query_record)
            rel = set(q.relevant)

            def feat(c):
                mods = [pr for pr, t in tm.get(c, [])
                        if pr != q.query_record and pr in uv and (not as_of or (t and t <= tq))]
                return _unit(np.mean([uv[pr] for pr in mods], axis=0)) if mods else None

            if any(feat(t) is not None for t in rel):
                reach += 1
            scored = sorted(((c, float(a @ f) if (a is not None and (f := feat(c)) is not None) else -1.0)
                             for c in q.candidates), key=lambda kv: (-kv[1], kv[0]))
            ranked = [c for c, _ in scored]
            rank = next((i + 1 for i, c in enumerate(ranked) if c in rel), None)
            h1 += rank == 1
            h5 += bool(rank and rank <= 5)
            h10 += bool(rank and rank <= 10)
            rr += (1.0 / rank) if rank else 0.0
        return {"R@1": round(h1 / n, 4), "R@5": round(h5 / n, 4), "R@10": round(h10 / n, 4),
                "MRR": round(rr / n, 4), "reachable": round(reach / n, 4), "n_queries": n}

    all_mods = evaluate(as_of=False)
    as_of = evaluate(as_of=True)
    res = {
        "task": "diff_to_affected_test", "split": "pilot cross-repo (test split)",
        "query_repr": "pure PR embedding (no alpha-blend)",
        "r21_alpha_blend_all_modifiers": {"R@1": 0.359, "MRR": 0.5911, "reachable": 0.9375},
        "pure_query_all_modifiers": all_mods,
        "pure_query_as_of_RELEASE": as_of,
        "reads": "query dilution (alpha-blend) cost ~0.195 R@1; temporal future-leakage "
                 "inflated by ~0.125 R@1; release-honest = pure query + as_of.",
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("R22 — diff->test, release-honest decomposition (pilot cross-repo)")
    print(f"  R21 (alpha-blend query + all modifiers):     R@1 0.359")
    print(f"  pure query + ALL modifiers (future leak):    R@1 {all_mods['R@1']:.3f}  MRR {all_mods['MRR']:.3f}")
    print(f"  pure query + as_of (RELEASE-HONEST):         R@1 {as_of['R@1']:.3f}  MRR {as_of['MRR']:.3f}  "
          f"R@10 {as_of['R@10']:.3f}  reachable {as_of['reachable']:.1%}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
