#!/usr/bin/env python3
"""The decisive experiment: do pretrained embeddings + our relation head beat IDF
cross-repo, and does our head add value over the embedder alone?

Runs five systems on the SAME de-referenced, cross-repo split as
run_crossrepo_ablation.py, so every number is comparable:

  1. vanilla-tf-cosine     bag-of-tokens cosine
  2. idf-cosine            bag-of-tokens + IDF      <- the bar from R7
  3. relation-metric       bag-of-tokens diagonal metric
  4. embedder-cosine       frozen MiniLM, raw cosine   <- the control
  5. embedder+relation     frozen MiniLM + our two-tower relation head  <- our contribution

Numpy only: it consumes the committed embedding cache (data/pilot/embeddings/),
so it is reproducible without torch or a download. Generate the cache first with
`python data/pilot/embed_pilot.py` (needs the [embed] extra).

Run:  python data/pilot/run_embed_ablation.py
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

from relsdlc.model import run_ablation  # noqa: E402
from relsdlc.tower import (  # noqa: E402
    run_cosine_on_vecs, run_relation_map_on_vecs, run_tower_on_vecs,
)
from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402

EMB = HERE / "embeddings" / "minilm-l6-v2.npz"
MIN_DF = 3
CREATED_AT = "2026-06-21T00:00:00Z"
DATASET_ID = "ds:gh-pilot-v0"


def load_embeddings() -> tuple[dict, int, str]:
    data = np.load(EMB, allow_pickle=False)
    ids = [str(i) for i in data["ids"]]
    vectors = data["vectors"].astype(np.float32)
    model = str(data["model"]) if "model" in data else "unknown"
    return {i: vectors[k] for k, i in enumerate(ids)}, vectors.shape[1], model


def build_cards(systems: dict, model: str) -> list[dict]:
    out = []
    for sys_name, exp_id, compare, desc in [
        ("embedder-cosine", "exp:gh-embed-cosine-v0", "exp:gh-xrepo-idf-v0",
         f"Frozen {model}, raw cosine (no training)."),
        ("embedder+tower", "exp:gh-embed-tower-v0", "exp:gh-embed-cosine-v0",
         f"Frozen {model} + a FROM-SCRATCH two-tower head (cautionary: overfits)."),
        ("embedder+relation-map", "exp:gh-embed-relmap-v0", "exp:gh-embed-cosine-v0",
         f"Frozen {model} + an identity-initialized relation operator."),
    ]:
        m = systems[sys_name]
        out.append({
            "card_type": "experiment", "id": exp_id,
            "name": f"GitHub pilot cross-repo de-referenced — {sys_name}",
            "created_at": CREATED_AT, "hypothesis": desc,
            "task": "issue_to_fixing_pr", "dataset_version": DATASET_ID,
            "code_version": "relsdlc-0.1.0", "seed": 0,
            "command": "python data/pilot/run_embed_ablation.py",
            "system": sys_name, "runtime_class": "cpu",
            "metrics": {
                "recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
                "mrr": round(m["mrr"], 4),
                "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
                "n_queries": m["n_queries"],
            },
            "baseline_comparison": compare, "error_slices": [],
            "leakage_checks": ["references scrubbed; train repos disjoint from test repos; "
                               "embeddings are frozen (no fit on test repos)."],
            "known_limitations": [
                "Pilot scale; frozen general-text embedder (not code-specific); "
                "relation head trained on ~180 cross-repo pairs. Exploratory.",
            ],
            "exploratory": True,
        })
    return out


def main() -> None:
    if not EMB.exists():
        print(f"ERROR: missing {EMB}. Run: python data/pilot/embed_pilot.py "
              "(needs pip install -e '.[embed]')", file=sys.stderr)
        raise SystemExit(2)

    ds, meta = load_pilot_crossrepo()
    vecs, dim, model = load_embeddings()

    base = run_ablation(ds, seed=0, min_df=MIN_DF)["systems"]
    systems = {
        "vanilla-tf-cosine": base["vanilla-tf-cosine"],
        "idf-cosine": base["idf-cosine"],
        "relation-metric": base["relation-metric"],
        "embedder-cosine": run_cosine_on_vecs(ds, vecs),
        "embedder+tower": run_tower_on_vecs(ds, vecs, dim, seed=0, d_proj=128,
                                            epochs=600, lr=0.5, margin=0.2,
                                            weight_decay=1e-3),
        "embedder+relation-map": run_relation_map_on_vecs(ds, vecs, dim, seed=0,
                                                          epochs=300, lr=0.1,
                                                          margin=0.1, decay=2e-2),
    }

    for card in build_cards(systems, model):
        (CARDS / f"{card['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "embed-results.json").write_text(
        json.dumps({"model": model, "dim": dim, "systems": systems, "meta": meta},
                   indent=2) + "\n", encoding="utf-8")

    print(f"DE-REFERENCED CROSS-REPO + EMBEDDINGS — issue_to_fixing_pr  (model: {model})")
    print(f"train repos: {len(meta['train_repos'])}  test repos: {len(meta['test_repos'])}")
    print(f"{'system':<22}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'HardNegAcc':>12}")
    for name in ["vanilla-tf-cosine", "idf-cosine", "relation-metric",
                 "embedder-cosine", "embedder+tower", "embedder+relation-map"]:
        m = systems[name]
        r = m["recall_at_k"]
        print(f"{name:<22}{r['1']:>8.3f}{r['5']:>8.3f}{r['10']:>8.3f}"
              f"{m['mrr']:>8.3f}{m['hard_negative_accuracy']:>12.3f}")


if __name__ == "__main__":
    main()
