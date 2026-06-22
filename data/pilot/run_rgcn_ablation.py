#!/usr/bin/env python3
"""Learned R-GCN vs training-free aggregation vs frozen cosine (issue→PR, cross-repo).

Numpy only, on committed caches. Run train_rgcn.py first to produce the learned
embeddings.

Run:  python data/pilot/run_rgcn_ablation.py
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

from relsdlc.graphsage import augmented_vecs  # noqa: E402
from relsdlc.tower import run_cosine_on_vecs  # noqa: E402
from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402

FROZEN = HERE / "embeddings" / "minilm-l6-v2.npz"
RGCN = HERE / "embeddings" / "rgcn-frozen.npz"


def _load(p):
    d = np.load(p, allow_pickle=False)
    ids, V = d["ids"], np.asarray(d["vectors"], dtype=np.float32)  # decompress once (NpzFile lazy member)
    return {str(i): V[k] for k, i in enumerate(ids)}


def _jsonl(p): return [json.loads(l) for l in p.read_text().split("\n") if l.strip()]


def main() -> None:
    ds, meta = load_pilot_crossrepo()
    frozen = _load(FROZEN)
    modifies = [(e["source"], e["target"]) for e in _jsonl(HERE / "graph" / "modifies_edges.jsonl")]
    free = augmented_vecs(frozen, ds.fixes, modifies, alpha=0.5, hops=2,
                          exclude_fixes_pairs=ds.fixes)
    systems = {
        "frozen-cosine": run_cosine_on_vecs(ds, frozen),
        "free-aggregation (R11B)": run_cosine_on_vecs(ds, free),
        "learned-rgcn (1-hop)": run_cosine_on_vecs(ds, _load(RGCN)),
    }
    print("Learned R-GCN vs free aggregation (de-referenced cross-repo, issue_to_fixing_pr)")
    print(f"{'system':<26}{'R@1':>8}{'R@5':>8}{'MRR':>8}")
    cards = []
    for name, m in systems.items():
        print(f"{name:<26}{m['recall_at_k']['1']:>8.3f}{m['recall_at_k']['5']:>8.3f}{m['mrr']:>8.3f}")
        if "learned" in name:
            cards.append({
                "card_type": "experiment", "id": "exp:gh-rgcn-1hop-v0",
                "name": "GitHub pilot cross-repo — learned 1-hop R-GCN",
                "created_at": "2026-06-21T00:00:00Z",
                "hypothesis": "Does a learned relational GNN beat training-free graph aggregation on issue→PR?",
                "task": "issue_to_fixing_pr", "dataset_version": "ds:gh-pilot-v0",
                "code_version": "relsdlc-0.1.0", "seed": 0,
                "command": "python data/pilot/train_rgcn.py && python data/pilot/run_rgcn_ablation.py",
                "system": "learned-rgcn-1hop", "runtime_class": "cpu",
                "metrics": {"recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
                            "mrr": round(m["mrr"], 4),
                            "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
                            "n_queries": m["n_queries"]},
                "baseline_comparison": "exp:gh-gnn-issue2pr-frozen-graph-v0", "error_slices": [],
                "leakage_checks": ["fixes edge is supervision only, never message-passing; cross-repo split."],
                "known_limitations": ["1-hop, frozen features, pilot scale. Exploratory."],
                "exploratory": True,
            })
    for c in cards:
        (CARDS / f"{c['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "rgcn-results.json").write_text(json.dumps({"systems": systems}, indent=2) + "\n")


if __name__ == "__main__":
    main()
