#!/usr/bin/env python3
"""R17a — harden the headline: confidence intervals + per-repo decomposition of
the LoRA win on the de-referenced, cross-repo issue->fixing-PR split.

R11A already gives the *cross-split* spread of the LoRA delta (mean ΔR@1
+0.061±0.021 over 5 held-out-repo partitions). This wave adds the two things that
single point estimate lacks on the *headline* default split (174 held-out queries,
frozen R@1 0.592 -> LoRA 0.655):

1. **Within-split confidence.** A paired bootstrap over the 174 queries (does the
   delta survive query resampling?) AND a **repo-cluster bootstrap** over the 8
   held-out repos (the honest CI for a cross-repo claim — queries inside one repo
   are correlated, so the repo is the resampling unit).
2. **Per-repo decomposition.** Where does the win come from — is it broad, or one
   or two repos? Reported per held-out repo, with a paired McNemar/sign test on
   the per-query rank-1 flips (gains vs. losses).

It reuses the loaders, the cross-repo split, and the exact ranking of
``run_gnn_ablation`` / ``run_crossrepo_ablation`` VERBATIM, and asserts the
aggregate reproduces the committed ``finetune-results.json`` before doing any
inference — so the CIs annotate a number the audit already trusts.

Run:  PYTHONPATH=src python data/pilot/run_bootstrap_ci.py
Numpy only; no network, no torch. Deterministic (seed 0).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from run_crossrepo_ablation import load_pilot_crossrepo, _repo_of  # noqa: E402
from run_gnn_ablation import load_embeddings, _with_zero_fallback, _candidate_ids  # noqa: E402

B = 10000
SEED = 0
FINETUNE = HERE / "finetune-results.json"
OUT = HERE / "bootstrap-ci-results.json"


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def per_query_metrics(dataset, base_vecs, dim):
    """Replicate run_gnn_ablation.score_embedder_cosine PER QUERY (test split).

    Returns a list of dicts in dataset query order (test queries only):
    {query_id, repo, hit1, hit5, hit10, rr, hns} where hns is 1/0/None.
    Ranking is identical to tower._eval: unit-cosine, tie-broken by candidate id.
    """
    vecs = _with_zero_fallback(base_vecs, _candidate_ids(dataset), dim)
    unit = {cid: _unit(vecs[cid]) for cid in _candidate_ids(dataset)}
    rows = []
    for q in dataset.queries:
        if q.split != "test":
            continue
        a = unit[q.query_record]
        scored = [(cid, float(a @ unit[cid])) for cid in q.candidates]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        ranked = [cid for cid, _ in scored]
        rel = set(q.relevant)
        # first relevant rank (1-based)
        rr_rank = next((i + 1 for i, cid in enumerate(ranked) if cid in rel), None)
        # hard-negative success: relevant outranks every hard negative
        hns = None
        if q.hard_negatives and rel:
            hn = set(q.hard_negatives)
            best_pos = next((i for i, c in enumerate(ranked) if c in rel), None)
            best_neg = next((i for i, c in enumerate(ranked) if c in hn), None)
            if best_pos is None:
                hns = 0
            elif best_neg is None:
                hns = 1
            else:
                hns = 1 if best_pos < best_neg else 0
        rows.append({
            "query_id": q.query_id,
            "repo": _repo_of(q.query_record),
            "hit1": 1.0 if rr_rank == 1 else 0.0,
            "hit5": 1.0 if (rr_rank is not None and rr_rank <= 5) else 0.0,
            "hit10": 1.0 if (rr_rank is not None and rr_rank <= 10) else 0.0,
            "rr": (1.0 / rr_rank) if rr_rank else 0.0,
            "hns": hns,
        })
    return rows


def _agg(rows, key):
    return float(np.mean([r[key] for r in rows]))


def _hns_acc(rows):
    vals = [r["hns"] for r in rows if r["hns"] is not None]
    return float(np.mean(vals)) if vals else 0.0


def _ci(samples, lo=2.5, hi=97.5):
    return [float(np.percentile(samples, lo)), float(np.percentile(samples, hi))]


def _binom_two_sided_p(b, c):
    """Exact two-sided sign test on b gains vs c losses (H0: p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def main() -> None:
    issue_ds, meta = load_pilot_crossrepo()
    frozen = load_embeddings("minilm-l6-v2")
    lora = load_embeddings("minilm-lora")
    dim = len(next(iter(frozen.values())))

    fr = per_query_metrics(issue_ds, frozen, dim)
    lo = per_query_metrics(issue_ds, lora, dim)
    assert [r["query_id"] for r in fr] == [r["query_id"] for r in lo]
    n = len(fr)

    # ---- 1. reproduce the committed aggregate (audit gate) --------------
    committed = json.loads(FINETUNE.read_text())["systems"]
    point = {
        "n_queries": n,
        "frozen": {"r1": _agg(fr, "hit1"), "r5": _agg(fr, "hit5"),
                   "r10": _agg(fr, "hit10"), "mrr": _agg(fr, "rr"),
                   "hns": _hns_acc(fr)},
        "lora": {"r1": _agg(lo, "hit1"), "r5": _agg(lo, "hit5"),
                 "r10": _agg(lo, "hit10"), "mrr": _agg(lo, "rr"),
                 "hns": _hns_acc(lo)},
    }
    checks = [
        (point["frozen"]["r1"], committed["embedder-cosine-frozen"]["recall_at_k"]["1"]),
        (point["frozen"]["mrr"], committed["embedder-cosine-frozen"]["mrr"]),
        (point["lora"]["r1"], committed["embedder-cosine-lora"]["recall_at_k"]["1"]),
        (point["lora"]["mrr"], committed["embedder-cosine-lora"]["mrr"]),
        (point["lora"]["hns"], committed["embedder-cosine-lora"]["hard_negative_accuracy"]),
    ]
    for got, exp in checks:
        assert abs(got - exp) < 1e-9, f"reproduction mismatch: {got} vs committed {exp}"

    d_r1 = point["lora"]["r1"] - point["frozen"]["r1"]
    d_mrr = point["lora"]["mrr"] - point["frozen"]["mrr"]

    fr_h1 = np.array([r["hit1"] for r in fr])
    lo_h1 = np.array([r["hit1"] for r in lo])
    fr_rr = np.array([r["rr"] for r in fr])
    lo_rr = np.array([r["rr"] for r in lo])
    rng = np.random.default_rng(SEED)

    # ---- 2a. query-level paired bootstrap -------------------------------
    bq_r1 = np.empty(B)
    bq_mrr = np.empty(B)
    for i in range(B):
        s = rng.integers(0, n, n)
        bq_r1[i] = lo_h1[s].mean() - fr_h1[s].mean()
        bq_mrr[i] = lo_rr[s].mean() - fr_rr[s].mean()

    # ---- 2b. repo-cluster bootstrap (the cross-repo-honest CI) ----------
    repos = meta["test_repos"]
    by_repo = {rp: np.array([j for j, r in enumerate(fr) if r["repo"] == rp]) for rp in repos}
    by_repo = {rp: idx for rp, idx in by_repo.items() if len(idx)}
    repo_keys = list(by_repo)
    br_r1 = np.empty(B)
    br_mrr = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, len(repo_keys), len(repo_keys))
        idx = np.concatenate([by_repo[repo_keys[k]] for k in pick])
        br_r1[i] = lo_h1[idx].mean() - fr_h1[idx].mean()
        br_mrr[i] = lo_rr[idx].mean() - fr_rr[idx].mean()

    # ---- 3. per-repo decomposition + McNemar ----------------------------
    per_repo = []
    for rp in repos:
        idx = by_repo.get(rp)
        if idx is None:
            continue
        f1 = fr_h1[idx].mean(); l1 = lo_h1[idx].mean()
        per_repo.append({
            "repo": rp, "n": int(len(idx)),
            "frozen_r1": round(float(f1), 4), "lora_r1": round(float(l1), 4),
            "delta_r1": round(float(l1 - f1), 4),
            "frozen_mrr": round(float(fr_rr[idx].mean()), 4),
            "lora_mrr": round(float(lo_rr[idx].mean()), 4),
        })
    per_repo.sort(key=lambda d: d["delta_r1"], reverse=True)

    gains = int(np.sum((fr_h1 == 0) & (lo_h1 == 1)))
    losses = int(np.sum((fr_h1 == 1) & (lo_h1 == 0)))
    mcnemar_p = _binom_two_sided_p(gains, losses)
    repos_improved = sum(1 for r in per_repo if r["delta_r1"] > 0)
    repos_regressed = sum(1 for r in per_repo if r["delta_r1"] < 0)

    out = {
        "task": "issue_to_fixing_pr",
        "split": "default cross-repo (8 held-out repos)",
        "n_queries": n, "n_test_repos": len(repo_keys),
        "bootstrap": {"B": B, "seed": SEED},
        "point": {"delta_r1": d_r1, "delta_mrr": d_mrr, **point},
        "ci95": {
            "query_bootstrap": {
                "delta_r1": _ci(bq_r1), "delta_mrr": _ci(bq_mrr),
                "p_one_sided_le0_r1": float(np.mean(bq_r1 <= 0)),
                "p_one_sided_le0_mrr": float(np.mean(bq_mrr <= 0)),
            },
            "repo_cluster_bootstrap": {
                "delta_r1": _ci(br_r1), "delta_mrr": _ci(br_mrr),
                "p_one_sided_le0_r1": float(np.mean(br_r1 <= 0)),
                "frac_resamples_positive_r1": float(np.mean(br_r1 > 0)),
            },
        },
        "per_query_flips": {
            "rank1_gains": gains, "rank1_losses": losses, "net": gains - losses,
            "mcnemar_sign_test_p": mcnemar_p,
        },
        "per_repo": per_repo,
        "repos_improved": repos_improved, "repos_regressed": repos_regressed,
        "cross_split_reference_R11A": {
            "note": "mean over 5 held-out-repo partitions (multisplit-results.json)",
            "delta_r1_mean": 0.0614, "delta_r1_std": 0.0211,
            "delta_mrr_mean": 0.0521, "delta_mrr_std": 0.0103,
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # ---- print ----------------------------------------------------------
    print(f"R17a — LoRA win CIs on the headline split (n={n} queries, "
          f"{len(repo_keys)} held-out repos, B={B})")
    print(f"  point: frozen R@1 {point['frozen']['r1']:.4f}  LoRA R@1 {point['lora']['r1']:.4f}  "
          f"ΔR@1 {d_r1:+.4f}  |  ΔMRR {d_mrr:+.4f}  (reproduces committed cards ✓)")
    qb = out["ci95"]["query_bootstrap"]; rb = out["ci95"]["repo_cluster_bootstrap"]
    print(f"  ΔR@1 95% CI  query-bootstrap : [{qb['delta_r1'][0]:+.4f}, {qb['delta_r1'][1]:+.4f}]  "
          f"(one-sided p[Δ≤0]={qb['p_one_sided_le0_r1']:.4f})")
    print(f"  ΔR@1 95% CI  repo-cluster    : [{rb['delta_r1'][0]:+.4f}, {rb['delta_r1'][1]:+.4f}]  "
          f"(resamples positive: {rb['frac_resamples_positive_r1']*100:.1f}%)")
    print(f"  rank-1 flips : +{gains} gained / -{losses} lost (net {gains-losses:+d}), "
          f"sign-test p={mcnemar_p:.2e}")
    print(f"  per-repo     : {repos_improved}/{len(repo_keys)} improved, {repos_regressed} regressed")
    print(f"  {'repo':<24}{'n':>4}{'frozenR@1':>11}{'loraR@1':>9}{'ΔR@1':>8}")
    for r in per_repo:
        print(f"  {r['repo']:<24}{r['n']:>4}{r['frozen_r1']:>11.3f}{r['lora_r1']:>9.3f}{r['delta_r1']:>+8.3f}")


if __name__ == "__main__":
    main()
