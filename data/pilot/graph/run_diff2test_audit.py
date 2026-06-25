#!/usr/bin/env python3
"""R23 — leakage/densification audit of the diff->test result (credibility gate).

The ML-venue panel flagged the diff->test number (R@1 ~0.43) as gated by a
test-side densification audit. The sharpest threat: CANDIDATE-COVERAGE PARITY. If
the densification gave the GOLD test node a feature (>=1 as_of non-gold modifier)
but left most NEGATIVE candidates unrankable (no modifier -> zero/-1 score, ranked
last), then high R@1 is an artefact of "gold has a feature, negatives don't", NOT
real discrimination among rankable candidates.

This audit (numpy, deterministic; reuses the dense graph + embeddings + as_of cut):
  1. gold vs negative candidate COVERAGE rates (is gold more covered than negatives?);
  2. FAIR R@1 — rank ONLY among COVERED candidates (removes the covered-vs-uncovered
     trivial signal), vs random-among-covered (1/pool). If fair R@1 >> random, the
     signal is real; if fair R@1 ~ random, it was a coverage artefact;
  3. overall R@1 (uncovered ranked last) vs fair R@1 — the gap is the coverage-artefact
     share;
  4. train/test repo disjointness (fork / near-duplicate base-name check).

Run: PYTHONPATH=src python data/pilot/graph/run_diff2test_audit.py
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

OUT = REPO_ROOT / "data" / "pilot" / "diff2test-audit-results.json"


def _jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def _u(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    _, meta = load_pilot_crossrepo()
    diff = load_diff2test_crossrepo(set(meta["train_repos"]))
    nv = {**load_embeddings("minilm-l6-v2"), **load_embeddings("dense-pr")}
    uv = {i: _u(nv[i]) for i in nv}

    vf = {}
    for p in ("data/pilot/records.jsonl", "data/pilot/graph/records_dense.jsonl"):
        for r in _jl(REPO_ROOT / p):
            if r.get("valid_from"):
                vf[r["id"]] = r["valid_from"]
    tm = {}
    for p in ("data/pilot/graph/modifies_edges.jsonl", "data/pilot/graph/modifies_edges_dense.jsonl"):
        for e in _jl(REPO_ROOT / p):
            if e.get("relation") == "modifies":
                tm.setdefault(e["target"], []).append((e["source"], e.get("valid_from", "")))

    def feat(c, tq, qpr):
        mods = [pr for pr, t in tm.get(c, []) if pr != qpr and pr in uv and t and t <= tq]
        return _u(np.mean([uv[pr] for pr in mods], axis=0)) if mods else None

    n = 0
    gold_cov = 0
    neg_cov_frac = []
    # overall R@1 (uncovered -> -1, ranked last) and fair R@1 (only covered candidates)
    overall_h1 = 0
    fair_h1 = 0
    fair_n = 0           # queries where gold is covered (fair pool is defined)
    fair_pool_sizes = []
    rand_among_cov = []
    for q in diff.queries:
        if q.split != "test" or not q.relevant:
            continue
        n += 1
        tq = vf.get(q.query_record, "")
        a = uv.get(q.query_record)
        rel = set(q.relevant)
        feats = {c: feat(c, tq, q.query_record) for c in q.candidates}
        covered = {c for c, f in feats.items() if f is not None}
        gold_is_cov = any(c in covered for c in rel)
        gold_cov += gold_is_cov
        negs = [c for c in q.candidates if c not in rel]
        if negs:
            neg_cov_frac.append(sum(c in covered for c in negs) / len(negs))
        # overall: uncovered get -1
        sc = sorted(((c, float(a @ feats[c]) if feats[c] is not None else -1.0) for c in q.candidates),
                    key=lambda kv: (-kv[1], kv[0]))
        overall_h1 += sc[0][0] in rel
        # fair: only among covered candidates (requires gold covered)
        if gold_is_cov:
            fair_n += 1
            covlist = [c for c in q.candidates if c in covered]
            fair_pool_sizes.append(len(covlist))
            rand_among_cov.append(1.0 / len(covlist))
            scf = sorted(((c, float(a @ feats[c])) for c in covlist), key=lambda kv: (-kv[1], kv[0]))
            fair_h1 += scf[0][0] in rel

    # repo disjointness / fork check
    def base(r):
        return r.split("/")[-1].lower()
    train_bases = {base(r) for r in meta["train_repos"]}
    overlap = sorted(base(r) for r in meta["test_repos"] if base(r) in train_bases)

    res = {
        "n_queries": n,
        "coverage": {
            "gold_coverage_rate": round(gold_cov / n, 4),
            "negative_coverage_rate_mean": round(float(np.mean(neg_cov_frac)), 4),
            "parity_note": "if gold >> negative coverage, R@1 risks being a 'gold has a feature' artefact",
        },
        "overall_R@1_uncovered_last": round(overall_h1 / n, 4),
        "fair_R@1_among_covered": {
            "value": round(fair_h1 / fair_n, 4) if fair_n else None,
            "n_queries_gold_covered": fair_n,
            "mean_fair_pool": round(float(np.mean(fair_pool_sizes)), 2) if fair_pool_sizes else None,
            "random_among_covered_baseline": round(float(np.mean(rand_among_cov)), 4) if rand_among_cov else None,
        },
        "repo_disjointness": {
            "train_repos": meta["train_repos"], "test_repos": meta["test_repos"],
            "shared_base_names": overlap, "forks_or_near_dupes": bool(overlap),
        },
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    cov, fair = res["coverage"], res["fair_R@1_among_covered"]
    print("R23 — diff->test leakage / densification audit")
    print(f"  candidate coverage:  gold {cov['gold_coverage_rate']:.1%}  vs  negatives {cov['negative_coverage_rate_mean']:.1%}  "
          f"(parity {'OK' if cov['negative_coverage_rate_mean'] >= cov['gold_coverage_rate']*0.7 else 'SKEWED toward gold'})")
    print(f"  overall R@1 (uncovered ranked last):   {res['overall_R@1_uncovered_last']:.3f}")
    print(f"  FAIR R@1 (only among covered cands):   {fair['value']:.3f}  "
          f"(n={fair['n_queries_gold_covered']}, mean pool {fair['mean_fair_pool']}, "
          f"random-among-covered {fair['random_among_covered_baseline']:.3f})")
    lift = fair['value'] / fair['random_among_covered_baseline'] if fair['random_among_covered_baseline'] else 0
    print(f"  => fair R@1 is {lift:.1f}x random-among-covered  "
          f"({'REAL discrimination' if lift > 1.8 else 'WEAK — coverage artefact risk'})")
    print(f"  repo disjointness: shared base names = {overlap or 'none'} "
          f"({'CLEAN' if not overlap else 'CHECK'})")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
