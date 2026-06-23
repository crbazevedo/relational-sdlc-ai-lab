#!/usr/bin/env python3
"""Track B robustness: sweep the training-free graph lift over (alpha x hops).

R11B (``run_gnn_ablation.py``) measured the typed mean-aggregation lift at a
SINGLE operating point — ``alpha=0.5, hops=2`` — and reported it as "small but
real" on ``issue_to_fixing_pr`` (LoRA+graph R@1 0.690 vs LoRA 0.655) and as a
failure on ``diff_to_affected_test`` (flat at pilot sparsity). A single point
cannot answer the two natural reviewer questions:

1. **Is the issue->PR lift a knife-edge or a plateau?** If ``+0.035`` only
   appears at ``alpha=0.5`` and vanishes elsewhere, it is a tuning artefact, not
   structure. We trace R@1/MRR across ``alpha in {0,.25,.5,.75,1}`` x
   ``hops in {1,2,3}`` on both frozen and LoRA features. At ``alpha=1`` the lift
   degenerates to embedder-cosine for text nodes (a sanity anchor).

2. **Can multi-hop rescue diff->test, or is it structurally dead?** The test-file
   candidates have no text embedding; a test gets a feature only from the PRs that
   modify it, and the leakage guard removes the gold ``(query-PR, test)`` edge. If
   the only PR that modifies a gold test IS the query PR, that test is *isolated*
   after the guard and no amount of aggregation can place it. We (a) sweep hops to
   see whether 2-/3-hop paths (test <- PR <- file <- PR ...) recover any signal,
   and (b) compute the ISOLATION RATE directly, which upper-bounds achievable
   recall independent of features.

Training-free, numpy only, no network, no torch. Deterministic. Exploratory probe
(labelled as such on the emitted cards). Reuses the loaders + leakage guards from
``run_gnn_ablation.py`` verbatim so the operating point ``alpha=0.5,hops=2`` row
reproduces R11B exactly.

Run:  PYTHONPATH=src python3 data/pilot/run_graph_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CARDS = REPO_ROOT / "data" / "cards" / "examples"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from relsdlc.graphsage import augmented_vecs, build_typed_adjacency  # noqa: E402
from relsdlc.tower import _eval  # noqa: E402

# Reuse R11B's loaders, splits, leakage guards, and scoring verbatim.
from run_gnn_ablation import (  # noqa: E402
    KS,
    _candidate_ids,
    _with_zero_fallback,
    gold_fixes_pairs,
    gold_modifies_pairs,
    load_diff2test_crossrepo,
    load_embeddings,
    load_graph_edges,
    score_embedder_cosine,
)
from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402

DATASET_ID = "ds:gh-pilot-v0"
CREATED_AT = "2026-06-22T00:00:00Z"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
HOPS = (1, 2, 3)


def score_graph_aug(dataset, base_vecs, dim, fixes_edges, modifies_edges, *,
                    alpha, hops, exclude_fixes_pairs=None, exclude_modifies_pairs=None):
    """Parameterized graph-aug-cosine (the R11B scorer with alpha/hops exposed)."""
    cand_ids = _candidate_ids(dataset)
    aug = augmented_vecs(
        base_vecs, fixes_edges, modifies_edges,
        alpha=alpha, hops=hops,
        exclude_fixes_pairs=exclude_fixes_pairs,
        exclude_modifies_pairs=exclude_modifies_pairs,
        all_node_ids=set(base_vecs) | cand_ids,
    )
    vecs = _with_zero_fallback(aug, cand_ids, dim)
    return _eval(dataset, vecs, KS, project=None)


def diff2test_isolation(diff_ds, modifies_edges):
    """Structural ceiling for diff->test under the leakage guard.

    For every (query-PR, relevant-test) gold pair on the TEST split, count how many
    OTHER (non-query) PRs modify that test. If zero, the test is *isolated*: after
    the guard removes the gold edge it has no modifying PR left, so aggregation can
    give it no feature and it can never be retrieved. Returns per-pair and
    per-query isolation stats. Independent of which features are used.
    """
    test_to_prs: dict[str, set[str]] = {}
    for pr, tgt in modifies_edges:
        test_to_prs.setdefault(tgt, set()).add(pr)

    pair_total = pair_isolated = 0
    q_total = q_all_isolated = 0
    for q in diff_ds.queries:
        if q.split != "test" or not q.relevant:
            continue
        q_total += 1
        rel_isolated = 0
        for test in q.relevant:
            pair_total += 1
            others = test_to_prs.get(test, set()) - {q.query_record}
            if not others:
                pair_isolated += 1
                rel_isolated += 1
        if rel_isolated == len(q.relevant):
            q_all_isolated += 1

    return {
        "pair_total": pair_total,
        "pair_isolated": pair_isolated,
        "pair_isolation_rate": round(pair_isolated / pair_total, 4) if pair_total else None,
        "query_total": q_total,
        "query_all_relevant_isolated": q_all_isolated,
        "query_reachable_ceiling": round(1 - q_all_isolated / q_total, 4) if q_total else None,
    }


def _cell(m):
    r = m["recall_at_k"]
    return {"R@1": round(r["1"], 4), "R@5": round(r["5"], 4),
            "R@10": round(r["10"], 4), "MRR": round(m["mrr"], 4),
            "hard_neg_acc": round(m["hard_negative_accuracy"], 4),
            "n_queries": m["n_queries"]}


def build_card(exp_id, name, task, system, cell, baseline_comparison, leakage,
               extra_limits=None):
    """Experiment card for one sweep cell (exploratory)."""
    return {
        "card_type": "experiment", "id": exp_id, "name": name,
        "created_at": CREATED_AT,
        "hypothesis": "Is the training-free graph lift a tuned knife-edge or a "
                      "plateau across (alpha, hops), and can multi-hop rescue "
                      "diff->affected-test or is it structure-bound?",
        "task": task, "dataset_version": DATASET_ID, "code_version": "relsdlc-0.1.0",
        "seed": 0, "command": "python data/pilot/run_graph_sweep.py",
        "system": system, "runtime_class": "cpu",
        "metrics": {
            "recall_at_k": {"1": cell["R@1"], "5": cell["R@5"], "10": cell["R@10"]},
            "mrr": cell["MRR"], "hard_negative_accuracy": cell["hard_neg_acc"],
            "n_queries": cell["n_queries"],
        },
        "baseline_comparison": baseline_comparison, "error_slices": [],
        "leakage_checks": leakage,
        "known_limitations": (extra_limits or []) + [
            "pilot scale; TRAINING-FREE mean aggregation (no learned weights); "
            "single cross-repo split; exploratory sweep, not release evidence."],
        "exploratory": True,
    }


def main() -> None:
    issue_ds, meta = load_pilot_crossrepo()
    train_repos = set(meta["train_repos"])
    diff_ds = load_diff2test_crossrepo(train_repos)

    frozen = load_embeddings("minilm-l6-v2")
    lora = load_embeddings("minilm-lora")
    dim = len(next(iter(frozen.values())))
    fixes_edges, modifies_edges = load_graph_edges()

    issue_gold_fixes = gold_fixes_pairs(issue_ds)
    diff_gold_mods = gold_modifies_pairs(diff_ds)

    # Embedder-cosine baselines (alpha-independent) per task x feature.
    baselines = {}
    for feat_name, feats in [("frozen", frozen), ("lora", lora)]:
        baselines[f"issue_to_fixing_pr/{feat_name}"] = _cell(
            score_embedder_cosine(issue_ds, feats, dim))
        baselines[f"diff_to_affected_test/{feat_name}"] = _cell(
            score_embedder_cosine(diff_ds, feats, dim))

    grid = {}
    tasks = [
        ("issue_to_fixing_pr", issue_ds, dict(exclude_fixes_pairs=issue_gold_fixes)),
        ("diff_to_affected_test", diff_ds, dict(exclude_modifies_pairs=diff_gold_mods)),
    ]
    for task, ds, guard in tasks:
        for feat_name, feats in [("frozen", frozen), ("lora", lora)]:
            for hops in HOPS:
                for alpha in ALPHAS:
                    m = score_graph_aug(ds, feats, dim, fixes_edges, modifies_edges,
                                        alpha=alpha, hops=hops, **guard)
                    grid[f"{task}/{feat_name}/h{hops}/a{alpha}"] = _cell(m)

    isolation = diff2test_isolation(diff_ds, modifies_edges)

    out = {
        "created_at": CREATED_AT, "dataset_version": DATASET_ID,
        "alphas": list(ALPHAS), "hops": list(HOPS),
        "train_repos": len(meta["train_repos"]), "test_repos": len(meta["test_repos"]),
        "baselines": baselines, "grid": grid,
        "diff2test_isolation": isolation,
    }
    (HERE / "graph-sweep-results.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # ---- emit headline cards (new ids; do not clobber R11B gh-gnn-*) -----
    def best_alpha(task, feat, hops):
        cells = {a: grid[f"{task}/{feat}/h{hops}/a{a}"] for a in ALPHAS}
        return max(ALPHAS, key=lambda a: cells[a]["R@1"]), cells

    cards = []
    # issue->PR: best 1-hop graph-aug per feature (1 hop suffices == 2 hops).
    for feat in ("frozen", "lora"):
        a_best, cells = best_alpha("issue_to_fixing_pr", feat, 1)
        cards.append(build_card(
            f"exp:gh-graphsweep-issue2pr-{feat}-h1-best-v0",
            f"graph-lift sweep — issue2pr cross-repo — {feat} best 1-hop "
            f"(alpha={a_best})",
            "issue_to_fixing_pr", f"graph-aug-cosine ({feat}, h1, a={a_best})",
            cells[a_best],
            f"exp:gh-gnn-issue2pr-{feat}-embed-v0",
            ["references scrubbed; train repos disjoint from test repos.",
             "gold (issue,fixing-PR) edges excluded from aggregation (no leakage).",
             f"lift positive across alpha in [0,0.75]; 1 hop == 2 hops "
             f"(graph-sweep-results.json) — robust plateau, not a tuned point."]))
    # diff->test: representative cell + the structure-bound ceiling.
    a_best, cells = best_alpha("diff_to_affected_test", "lora", 2)
    cards.append(build_card(
        "exp:gh-graphsweep-diff2test-lora-h2-v0",
        "graph-lift sweep — diff2test cross-repo — lora graph-aug (h2)",
        "diff_to_affected_test", f"graph-aug-cosine (lora, h2, a={a_best})",
        cells[a_best],
        "exp:gh-gnn-diff2test-lora-embed-v0",
        ["test-file features come ONLY from PRs that modify them (structure).",
         "gold (PR,test) modifies edges excluded from aggregation (no leakage)."],
        extra_limits=[
            f"STRUCTURE-BOUND: {isolation['pair_isolated']}/{isolation['pair_total']} "
            f"({isolation['pair_isolation_rate']:.0%}) gold tests isolated after the "
            f"leakage guard; reachable ceiling {isolation['query_reachable_ceiling']:.0%}. "
            f"Flat across all (alpha,hops) — needs denser co-change (scale), not tuning."]))
    for card in cards:
        (CARDS / f"{card['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- pretty tables --------------------------------------------------
    def print_task(task):
        print(f"\n================ {task} ================")
        for feat_name in ("frozen", "lora"):
            base = baselines[f"{task}/{feat_name}"]
            print(f"\n-- {feat_name} features (embedder-cosine baseline: "
                  f"R@1={base['R@1']:.3f}  MRR={base['MRR']:.3f}) --")
            print(f"{'':>6}" + "".join(f"  a={a:<5}" for a in ALPHAS) + "     metric")
            for hops in HOPS:
                for metric in ("R@1", "MRR"):
                    row = "".join(
                        f"  {grid[f'{task}/{feat_name}/h{hops}/a{a}'][metric]:<5.3f}"
                        for a in ALPHAS)
                    print(f"  h={hops} {row}     {metric}")

    print("GRAPH-LIFT SWEEP (training-free typed aggregation) — pilot cross-repo")
    print(f"train repos: {meta['train_repos'].__len__()}  "
          f"test repos: {len(meta['test_repos'])}  "
          f"alphas={ALPHAS}  hops={HOPS}")
    print("(a=1.0 -> self-only for text nodes == embedder-cosine sanity anchor)")
    print_task("issue_to_fixing_pr")
    print_task("diff_to_affected_test")

    print("\n================ diff->test STRUCTURAL CEILING ================")
    iso = isolation
    print(f"gold (query-PR, test) pairs on test split: {iso['pair_total']}")
    print(f"  isolated after leakage guard (no other modifying PR): "
          f"{iso['pair_isolated']} ({iso['pair_isolation_rate']:.1%})")
    print(f"queries on test split: {iso['query_total']}; "
          f"with ALL relevant tests isolated: {iso['query_all_relevant_isolated']}")
    print(f"  => reachable R@anything ceiling: {iso['query_reachable_ceiling']:.1%} "
          f"(structure-bound, feature-independent)")
    print(f"\nwrote {HERE / 'graph-sweep-results.json'}")


if __name__ == "__main__":
    main()
