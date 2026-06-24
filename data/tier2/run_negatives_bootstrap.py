#!/usr/bin/env python3
"""R18 — bootstrap CIs on the best Tier-2 LoRA cell (closes R17a's open Tier-2 CI).

Reads the frozen + best-LoRA embeddings written by run_negatives_sweep.py and,
exactly as R17a did for the pilot, computes per-query metrics on the 1,171 held-out
Tier-2 queries, then a paired **query bootstrap** and a **repo-cluster bootstrap**
(the cross-repo-honest CI) on ΔR@1 / ΔMRR, plus a per-repo decomposition. Numpy
only; no torch; deterministic (seed 0, B=10000).

Run:  PYTHONPATH=src python data/tier2/run_negatives_bootstrap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from run_tier2_ablation import load_tier2_crossrepo, _repo_of  # noqa: E402

EMB = HERE / "embeddings"
OUT = HERE / "negatives-bootstrap-results.json"
B, SEED = 10000, 0


def _load_vecs(name):
    d = np.load(EMB / name, allow_pickle=False)
    ids, V = d["ids"], np.asarray(d["vectors"], dtype=np.float32)
    return {str(i): V[k] for k, i in enumerate(ids)}


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def per_query(ds, vecs):
    cache = {}

    def u(cid):
        if cid not in cache:
            cache[cid] = _unit(vecs[cid]) if cid in vecs else None
        return cache[cid]

    rows = []
    for q in ds.queries:
        if q.split != "test":
            continue
        a = u(q.query_record)
        if a is None:
            continue
        scored = []
        for cid in q.candidates:
            cv = u(cid)
            scored.append((cid, float(a @ cv) if cv is not None else -1.0))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        ranked = [c for c, _ in scored]
        rel = set(q.relevant)
        rank = next((i + 1 for i, c in enumerate(ranked) if c in rel), None)
        hns = None
        if q.hard_negatives and rel:
            hn = set(q.hard_negatives)
            bp = next((i for i, c in enumerate(ranked) if c in rel), None)
            bn = next((i for i, c in enumerate(ranked) if c in hn), None)
            hns = 0 if bp is None else (1 if bn is None else int(bp < bn))
        rows.append({"repo": _repo_of(q.query_record),
                     "hit1": 1.0 if rank == 1 else 0.0,
                     "rr": (1.0 / rank) if rank else 0.0, "hns": hns})
    return rows


def _ci(s):
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    lora_name = "minilm-lora-best.npz" if (EMB / "minilm-lora-best.npz").exists() else "minilm-lora.npz"
    for n in ("minilm-l6-v2.npz", lora_name):
        if not (EMB / n).exists():
            print(f"ERROR missing {EMB / n}; run run_negatives_sweep.py first", file=sys.stderr)
            raise SystemExit(2)
    ds, meta = load_tier2_crossrepo()
    fr = per_query(ds, _load_vecs("minilm-l6-v2.npz"))
    lo = per_query(ds, _load_vecs(lora_name))
    n = len(fr)
    assert n == len(lo)

    fr_h1 = np.array([r["hit1"] for r in fr]); lo_h1 = np.array([r["hit1"] for r in lo])
    fr_rr = np.array([r["rr"] for r in fr]); lo_rr = np.array([r["rr"] for r in lo])
    d_r1 = float(lo_h1.mean() - fr_h1.mean()); d_mrr = float(lo_rr.mean() - fr_rr.mean())
    rng = np.random.default_rng(SEED)

    bq_r1 = np.empty(B); bq_mrr = np.empty(B)
    for i in range(B):
        s = rng.integers(0, n, n)
        bq_r1[i] = lo_h1[s].mean() - fr_h1[s].mean()
        bq_mrr[i] = lo_rr[s].mean() - fr_rr[s].mean()

    repos = sorted({r["repo"] for r in fr})
    by_repo = {rp: np.array([j for j, r in enumerate(fr) if r["repo"] == rp]) for rp in repos}
    by_repo = {rp: idx for rp, idx in by_repo.items() if len(idx)}
    keys = list(by_repo)
    br_r1 = np.empty(B); br_mrr = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([by_repo[keys[k]] for k in pick])
        br_r1[i] = lo_h1[idx].mean() - fr_h1[idx].mean()
        br_mrr[i] = lo_rr[idx].mean() - fr_rr[idx].mean()

    per_repo = []
    for rp in keys:
        idx = by_repo[rp]
        per_repo.append({"repo": rp, "n": int(len(idx)),
                         "frozen_r1": round(float(fr_h1[idx].mean()), 4),
                         "lora_r1": round(float(lo_h1[idx].mean()), 4),
                         "delta_r1": round(float(lo_h1[idx].mean() - fr_h1[idx].mean()), 4)})
    per_repo.sort(key=lambda d: d["delta_r1"], reverse=True)
    gains = int(np.sum((fr_h1 == 0) & (lo_h1 == 1)))
    losses = int(np.sum((fr_h1 == 1) & (lo_h1 == 0)))

    out = {
        "task": "issue_to_fixing_pr", "split": "dense Tier-2 cross-repo (32 held-out repos)",
        "lora_cache": lora_name, "n_queries": n, "n_test_repos": len(keys),
        "bootstrap": {"B": B, "seed": SEED},
        "point": {"frozen_r1": float(fr_h1.mean()), "lora_r1": float(lo_h1.mean()),
                  "delta_r1": d_r1, "delta_mrr": d_mrr},
        "ci95": {
            "query_bootstrap": {"delta_r1": _ci(bq_r1), "delta_mrr": _ci(bq_mrr),
                                "p_one_sided_le0_r1": float(np.mean(bq_r1 <= 0))},
            "repo_cluster_bootstrap": {"delta_r1": _ci(br_r1), "delta_mrr": _ci(br_mrr),
                                       "frac_resamples_positive_r1": float(np.mean(br_r1 > 0))},
        },
        "per_query_flips": {"rank1_gains": gains, "rank1_losses": losses, "net": gains - losses},
        "repos_improved": sum(1 for r in per_repo if r["delta_r1"] > 0),
        "repos_regressed": sum(1 for r in per_repo if r["delta_r1"] < 0),
        "per_repo": per_repo,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    qb = out["ci95"]["query_bootstrap"]; rb = out["ci95"]["repo_cluster_bootstrap"]
    print(f"R18 bootstrap — best Tier-2 LoRA cell ({lora_name}), n={n}, {len(keys)} repos")
    print(f"  frozen R@1 {fr_h1.mean():.4f}  LoRA R@1 {lo_h1.mean():.4f}  ΔR@1 {d_r1:+.4f}  ΔMRR {d_mrr:+.4f}")
    print(f"  ΔR@1 query CI  [{qb['delta_r1'][0]:+.4f},{qb['delta_r1'][1]:+.4f}]  p[Δ≤0]={qb['p_one_sided_le0_r1']:.4f}")
    print(f"  ΔR@1 repo  CI  [{rb['delta_r1'][0]:+.4f},{rb['delta_r1'][1]:+.4f}]  pos={rb['frac_resamples_positive_r1']*100:.1f}%")
    print(f"  rank-1 flips +{gains}/-{losses} (net {gains-losses:+d}); "
          f"repos {out['repos_improved']} up / {out['repos_regressed']} down")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
