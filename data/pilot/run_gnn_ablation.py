#!/usr/bin/env python3
"""Track B graph probe: does GRAPH STRUCTURE add signal beyond pairwise text cosine?

A *training-free* probe. It scores the de-referenced, cross-repo split on BOTH
tasks with two systems each, on frozen MiniLM features and on the LoRA-tuned
features:

- ``embedder-cosine``  — pairwise cosine on the pretrained node vectors (the
  Track-A/embed control). This is what the relational contribution must beat.
- ``graph-aug-cosine`` — the same cosine, but on node vectors first augmented by
  ``relsdlc.graphsage`` (typed mean aggregation over the ``fixes`` + ``modifies``
  graph, leakage-guarded so the eval edge is never aggregated through).

Why two tasks:
- ``issue_to_fixing_pr`` — both endpoints are text records the embedder can see, so
  graph aggregation competes head-to-head with text cosine.
- ``diff_to_affected_test`` — the candidates are TEST-FILE nodes that have NO text
  embedding at all. Pairwise text cosine literally cannot represent them (they
  fall back to a zero vector → random); a graph aggregation gives each test node a
  feature built from the PRs that modify it. This is the cleanest test of "a
  relation cosine can't capture".

This is an exploratory first probe, NOT the final word: a training-free aggregation
either shows there is structural signal to exploit (motivating a *learned* GNN /
R-GCN in torch) or shows there is not at pilot scale. Either result is honest and
the cards/docs are labeled exploratory.

Run:  python data/pilot/run_gnn_ablation.py
Numpy only; no network, no torch. Deterministic.
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

from relsdlc.graphsage import augmented_vecs  # noqa: E402
from relsdlc.synth import Query, SynthDataset  # noqa: E402
from relsdlc.tower import _eval  # noqa: E402

# Reuse the de-referenced cross-repo split helper.
sys.path.insert(0, str(HERE))
from run_crossrepo_ablation import load_pilot_crossrepo, _repo_of  # noqa: E402

DATASET_ID = "ds:gh-pilot-v0"
CREATED_AT = "2026-06-21T00:00:00Z"
KS = (1, 5, 10)
ALPHA = 0.5
HOPS = 2


# --- feature + graph loading -------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_embeddings(name: str) -> dict[str, np.ndarray]:
    """Load a committed .npz (keys ids, vectors) into a dict id -> float64 vector."""
    npz = np.load(REPO_ROOT / "data" / "pilot" / "embeddings" / f"{name}.npz",
                  allow_pickle=True)
    ids = [str(i) for i in npz["ids"]]
    vecs = np.asarray(npz["vectors"], dtype=np.float64)
    return {i: vecs[k] for k, i in enumerate(ids)}


def load_graph_edges():
    """Return (fixes_edges, modifies_edges) as (pr, issue) and (pr, file/test) pairs."""
    fixes = _load_jsonl(REPO_ROOT / "data" / "pilot" / "edges.jsonl")
    mods = _load_jsonl(REPO_ROOT / "data" / "pilot" / "graph" / "modifies_edges.jsonl")
    fixes_edges = [(e["source"], e["target"]) for e in fixes if e.get("relation") == "fixes"]
    modifies_edges = [(e["source"], e["target"]) for e in mods if e.get("relation") == "modifies"]
    return fixes_edges, modifies_edges


# --- diff_to_affected_test dataset (cross-repo split) ------------------------

def load_diff2test_crossrepo(train_repos: set[str]) -> SynthDataset:
    """Build a SynthDataset of diff->test queries with the SAME cross-repo split.

    ``fixes`` is left empty (no training: cosine is parameter-free). Each query's
    split is 'test' iff the query PR is in a held-out repo, matching the issue->PR
    split exactly.
    """
    raw = _load_jsonl(REPO_ROOT / "data" / "pilot" / "benchmark" / "diff_to_affected_test.jsonl")
    queries = []
    for q in raw:
        split = "train" if _repo_of(q["query_record"]) in train_repos else "test"
        queries.append(Query(
            query_id=q["query_id"], query_record=q["query_record"],
            candidates=q["candidates"], relevant=q["relevant"],
            hard_negatives=q.get("hard_negatives", []), split=split))
    return SynthDataset(artifacts=[], fixes=[], queries=queries,
                        params={"task": "diff_to_affected_test", "split": "cross-repo"})


# --- scoring helpers ---------------------------------------------------------

def _with_zero_fallback(vecs: dict, ids, dim: int) -> dict:
    """Copy ``vecs`` and add a zero vector for any ``id`` it is missing.

    A zero vector scores cosine 0 against everything — the honest behaviour of a
    text embedder for a node it has no text for (file/test nodes). Keeps ``_eval``
    from KeyError-ing while making "the embedder can't see this node" explicit.
    """
    out = dict(vecs)
    zero = np.zeros(dim, dtype=np.float64)
    for cid in ids:
        if cid not in out:
            out[cid] = zero
    return out


def _candidate_ids(dataset: SynthDataset) -> set[str]:
    ids = set()
    for q in dataset.queries:
        ids.add(q.query_record)
        ids.update(q.candidates)
    return ids


def score_embedder_cosine(dataset, base_vecs, dim):
    vecs = _with_zero_fallback(base_vecs, _candidate_ids(dataset), dim)
    return _eval(dataset, vecs, KS, project=None)


def score_graph_aug(dataset, base_vecs, dim, fixes_edges, modifies_edges,
                    *, exclude_fixes_pairs=None, exclude_modifies_pairs=None):
    cand_ids = _candidate_ids(dataset)
    aug = augmented_vecs(
        base_vecs, fixes_edges, modifies_edges,
        alpha=ALPHA, hops=HOPS,
        exclude_fixes_pairs=exclude_fixes_pairs,
        exclude_modifies_pairs=exclude_modifies_pairs,
        all_node_ids=set(base_vecs) | cand_ids,
    )
    vecs = _with_zero_fallback(aug, cand_ids, dim)
    return _eval(dataset, vecs, KS, project=None)


def gold_fixes_pairs(dataset) -> list[tuple[str, str]]:
    """(pr, issue) gold pairs for the issue->PR task — the leakage edge."""
    return [(q.relevant[0], q.query_record) for q in dataset.queries if q.relevant]


def gold_modifies_pairs(dataset) -> list[tuple[str, str]]:
    """(pr, test) gold pairs for the diff->test task — the leakage edges."""
    pairs = []
    for q in dataset.queries:
        for rel in q.relevant:
            pairs.append((q.query_record, rel))
    return pairs


# --- cards + results ---------------------------------------------------------

def build_card(exp_id, name, task, system, m, compare, leakage):
    return {
        "card_type": "experiment", "id": exp_id, "name": name,
        "created_at": CREATED_AT,
        "hypothesis": "On the de-referenced cross-repo split, does training-free "
                      "typed graph aggregation add retrieval signal beyond pairwise "
                      "text cosine on pretrained node features?",
        "task": task, "dataset_version": DATASET_ID, "code_version": "relsdlc-0.1.0",
        "seed": 0, "command": "python data/pilot/run_gnn_ablation.py",
        "system": system, "runtime_class": "cpu",
        "metrics": {
            "recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
            "mrr": round(m["mrr"], 4),
            "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
            "n_queries": m["n_queries"],
        },
        "baseline_comparison": compare, "error_slices": [],
        "leakage_checks": leakage,
        "known_limitations": ["pilot scale; TRAINING-FREE mean aggregation (no "
                              "learned GNN weights); frozen/LoRA node features; "
                              "exploratory probe, not the final word — a learned "
                              "R-GCN is the follow-up if structure shows signal."],
        "exploratory": True,
    }


def _row(name, m):
    r = m["recall_at_k"]
    return (f"{name:<28}{r['1']:>8.3f}{r['5']:>8.3f}{r['10']:>8.3f}"
            f"{m['mrr']:>8.3f}{m['hard_negative_accuracy']:>12.3f}")


def main() -> None:
    issue_ds, meta = load_pilot_crossrepo()
    train_repos = set(meta["train_repos"])
    diff_ds = load_diff2test_crossrepo(train_repos)

    frozen = load_embeddings("minilm-l6-v2")
    lora = load_embeddings("minilm-lora")
    dim = len(next(iter(frozen.values())))

    fixes_edges, modifies_edges = load_graph_edges()

    # Leakage guards: never aggregate a node through the gold eval edge.
    issue_gold_fixes = gold_fixes_pairs(issue_ds)
    diff_gold_mods = gold_modifies_pairs(diff_ds)

    results = {}
    cards = []

    # ---- Task 1: issue_to_fixing_pr -------------------------------------
    for feat_name, feats in [("frozen", frozen), ("lora", lora)]:
        emb = score_embedder_cosine(issue_ds, feats, dim)
        aug = score_graph_aug(
            issue_ds, feats, dim, fixes_edges, modifies_edges,
            exclude_fixes_pairs=issue_gold_fixes)
        results[f"issue_to_fixing_pr/{feat_name}/embedder-cosine"] = emb
        results[f"issue_to_fixing_pr/{feat_name}/graph-aug-cosine"] = aug
        cards.append(build_card(
            f"exp:gh-gnn-issue2pr-{feat_name}-embed-v0",
            f"GNN probe — issue2pr cross-repo — {feat_name} embedder-cosine",
            "issue_to_fixing_pr", f"embedder-cosine ({feat_name})", emb,
            "none" if feat_name == "frozen" else
            "exp:gh-gnn-issue2pr-frozen-embed-v0",
            ["references scrubbed; train repos disjoint from test repos."]))
        cards.append(build_card(
            f"exp:gh-gnn-issue2pr-{feat_name}-graphaug-v0",
            f"GNN probe — issue2pr cross-repo — {feat_name} graph-aug-cosine",
            "issue_to_fixing_pr", f"graph-aug-cosine ({feat_name})", aug,
            f"exp:gh-gnn-issue2pr-{feat_name}-embed-v0",
            ["references scrubbed; train repos disjoint from test repos.",
             "gold (issue,fixing-PR) edges excluded from aggregation (no leakage)."]))

    # ---- Task 2: diff_to_affected_test ---------------------------------
    for feat_name, feats in [("frozen", frozen), ("lora", lora)]:
        emb = score_embedder_cosine(diff_ds, feats, dim)
        aug = score_graph_aug(
            diff_ds, feats, dim, fixes_edges, modifies_edges,
            exclude_modifies_pairs=diff_gold_mods)
        results[f"diff_to_affected_test/{feat_name}/embedder-cosine"] = emb
        results[f"diff_to_affected_test/{feat_name}/graph-aug-cosine"] = aug
        cards.append(build_card(
            f"exp:gh-gnn-diff2test-{feat_name}-embed-v0",
            f"GNN probe — diff2test cross-repo — {feat_name} embedder-cosine",
            "diff_to_affected_test", f"embedder-cosine ({feat_name})", emb,
            "none" if feat_name == "frozen" else
            "exp:gh-gnn-diff2test-frozen-embed-v0",
            ["test-file nodes have no text embedding (zero vector) — cosine cannot "
             "represent them; this is the honest baseline."]))
        cards.append(build_card(
            f"exp:gh-gnn-diff2test-{feat_name}-graphaug-v0",
            f"GNN probe — diff2test cross-repo — {feat_name} graph-aug-cosine",
            "diff_to_affected_test", f"graph-aug-cosine ({feat_name})", aug,
            f"exp:gh-gnn-diff2test-{feat_name}-embed-v0",
            ["test-file features come ONLY from the PRs that modify them (structure).",
             "gold (PR,test) modifies edges excluded from aggregation (no leakage)."]))

    for card in cards:
        (CARDS / f"{card['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    n_issue_test = sum(1 for q in issue_ds.queries if q.split == "test")
    n_diff_test = sum(1 for q in diff_ds.queries if q.split == "test")
    (REPO_ROOT / "data" / "pilot" / "gnn-results.json").write_text(
        json.dumps({
            "results": results, "meta": meta,
            "alpha": ALPHA, "hops": HOPS,
            "n_issue_test_queries": n_issue_test,
            "n_diff_test_queries": n_diff_test,
        }, indent=2) + "\n", encoding="utf-8")

    # ---- print the table -----------------------------------------------
    print("GRAPH-STRUCTURE PROBE (training-free aggregation) — cross-repo split")
    print(f"train repos: {len(meta['train_repos'])}  test repos: {len(meta['test_repos'])}  "
          f"(alpha={ALPHA}, hops={HOPS})")
    hdr = f"{'task / features / system':<28}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'HardNegAcc':>12}"
    for task, n_test in [("issue_to_fixing_pr", n_issue_test),
                         ("diff_to_affected_test", n_diff_test)]:
        print()
        print(f"== {task}  (test queries: {n_test}) ==")
        print(hdr)
        for feat_name in ("frozen", "lora"):
            for system in ("embedder-cosine", "graph-aug-cosine"):
                key = f"{task}/{feat_name}/{system}"
                print(_row(f"{feat_name}/{system}", results[key]))


if __name__ == "__main__":
    main()
