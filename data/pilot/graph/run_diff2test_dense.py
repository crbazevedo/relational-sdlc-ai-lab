#!/usr/bin/env python3
"""R20 Stage 3 — diff->affected-test retrieval on the DENSIFIED modifies graph.

R16E: diff->test graph-aggregation is flat at R@1 0.009 because 47% of gold test
nodes are isolated after the leakage guard. R17b: that ceiling (59.8% reachable) is
an ingest artefact; real co-change lifts it to 96.4%. Stages 1-2 built a denser
modifies graph (982 extra (PR,test) edges over the 8 test repos) + embedded the new
PR nodes. This stage runs the SAME R16E scorer + leakage guard with original vs
original+dense edges, isolating the densification effect. Numpy only; deterministic.

Run: PYTHONPATH=src python data/pilot/graph/run_diff2test_dense.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent              # data/pilot/graph
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data" / "pilot"))

from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402
from run_gnn_ablation import (  # noqa: E402
    load_embeddings, load_diff2test_crossrepo, load_graph_edges,
    gold_modifies_pairs, score_embedder_cosine,
)
from run_graph_sweep import score_graph_aug  # noqa: E402

ALPHA, HOPS = 0.5, 1
OUT = HERE.parents[0] / "diff2test-dense-results.json"


def _dense_modifies():
    rows = [json.loads(l) for l in (HERE / "modifies_edges_dense.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
    return [(e["source"], e["target"]) for e in rows if e.get("relation") == "modifies"]


def _reachable(diff_ds, modifies, gold_pairs):
    """Fraction of test-split queries with >=1 relevant test reachable (has a
    non-gold modifier) after the leakage guard — the structural ceiling."""
    test_to_prs = {}
    for pr, tgt in modifies:
        test_to_prs.setdefault(tgt, set()).add(pr)
    gold = set(gold_pairs)
    q_total = q_reach = 0
    for q in diff_ds.queries:
        if q.split != "test" or not q.relevant:
            continue
        q_total += 1
        if any(test_to_prs.get(t, set()) - {q.query_record} for t in q.relevant):
            q_reach += 1
    return round(q_reach / q_total, 4) if q_total else None


def main():
    issue_ds, meta = load_pilot_crossrepo()
    diff_ds = load_diff2test_crossrepo(set(meta["train_repos"]))

    frozen = load_embeddings("minilm-l6-v2")
    dense_pr = load_embeddings("dense-pr")
    node_vecs = {**frozen, **dense_pr}              # add the new modifying-PR vectors
    dim = len(next(iter(frozen.values())))

    fixes_edges, modifies_orig = load_graph_edges()
    modifies_dense = _dense_modifies()
    modifies_union = modifies_orig + modifies_dense
    diff_gold = gold_modifies_pairs(diff_ds)

    def cell(m):
        r = m["recall_at_k"]
        return {"R@1": round(r["1"], 4), "R@5": round(r["5"], 4), "R@10": round(r["10"], 4),
                "MRR": round(m["mrr"], 4), "n_queries": m["n_queries"]}

    emb = cell(score_embedder_cosine(diff_ds, node_vecs, dim))
    aug_orig = cell(score_graph_aug(diff_ds, node_vecs, dim, fixes_edges, modifies_orig,
                                    alpha=ALPHA, hops=HOPS, exclude_modifies_pairs=diff_gold))
    aug_dense = cell(score_graph_aug(diff_ds, node_vecs, dim, fixes_edges, modifies_union,
                                     alpha=ALPHA, hops=HOPS, exclude_modifies_pairs=diff_gold))

    out = {
        "task": "diff_to_affected_test", "split": "pilot cross-repo (test split)",
        "alpha": ALPHA, "hops": HOPS,
        "n_dense_edges": len(modifies_dense), "n_dense_pr_nodes": len(dense_pr),
        "reachable_orig": _reachable(diff_ds, modifies_orig, diff_gold),
        "reachable_dense": _reachable(diff_ds, modifies_union, diff_gold),
        "embedder_cosine": emb,
        "graph_aug_orig_R16E": aug_orig,
        "graph_aug_dense_R20": aug_dense,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("R20 — diff->test retrieval on the densified modifies graph (pilot cross-repo)")
    print(f"  dense graph: +{len(modifies_dense)} (PR,test) edges, +{len(dense_pr)} PR nodes "
          f"over the 8 test repos")
    print(f"  reachable (query has a non-gold modifier after the guard): "
          f"{out['reachable_orig']:.1%} (orig) -> {out['reachable_dense']:.1%} (dense)")
    print(f"  {'system':<34}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}")
    print(f"  {'embedder-cosine (no graph)':<34}{emb['R@1']:>8.3f}{emb['R@5']:>8.3f}{emb['R@10']:>8.3f}{emb['MRR']:>8.3f}")
    print(f"  {'graph-aug, orig edges (R16E)':<34}{aug_orig['R@1']:>8.3f}{aug_orig['R@5']:>8.3f}{aug_orig['R@10']:>8.3f}{aug_orig['MRR']:>8.3f}")
    print(f"  {'graph-aug, +dense edges (R20)':<34}{aug_dense['R@1']:>8.3f}{aug_dense['R@5']:>8.3f}{aug_dense['R@10']:>8.3f}{aug_dense['MRR']:>8.3f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
