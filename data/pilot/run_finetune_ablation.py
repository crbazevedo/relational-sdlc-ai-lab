#!/usr/bin/env python3
"""Track A result: does LoRA fine-tuning with the relation loss beat the FROZEN
embedder cross-repo? Numpy only, on the committed caches (no torch).

Compares, on the same de-referenced cross-repo split:
  - idf-cosine                 (bag-of-tokens bar)
  - embedder-cosine (frozen)   (the control from R8: R@1 0.59)
  - embedder-cosine (LoRA)     (the fine-tuned representation)   <- the question
  - LoRA + identity-init operator (does a head help on tuned vectors?)

Run:  python data/pilot/run_finetune_ablation.py
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
from relsdlc.tower import run_cosine_on_vecs, run_relation_map_on_vecs  # noqa: E402
from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402

FROZEN = HERE / "embeddings" / "minilm-l6-v2.npz"
TUNED = HERE / "embeddings" / "minilm-lora.npz"
CREATED_AT = "2026-06-21T00:00:00Z"
DATASET_ID = "ds:gh-pilot-v0"


def _load(path: Path):
    data = np.load(path, allow_pickle=False)
    ids = [str(i) for i in data["ids"]]
    vectors = data["vectors"].astype(np.float32)
    model = str(data["model"]) if "model" in data else "unknown"
    return {i: vectors[k] for k, i in enumerate(ids)}, vectors.shape[1], model


def build_cards(systems: dict, tuned_model: str) -> list[dict]:
    out = []
    for sys_name, exp_id, compare, desc in [
        ("embedder-cosine-lora", "exp:gh-finetune-cosine-v0", "exp:gh-embed-cosine-v0",
         f"LoRA-tuned {tuned_model}, cosine — relation loss reshaping the representation."),
        ("lora+relation-map", "exp:gh-finetune-relmap-v0", "exp:gh-finetune-cosine-v0",
         "LoRA-tuned embedder + identity-init relation operator."),
    ]:
        m = systems[sys_name]
        out.append({
            "card_type": "experiment", "id": exp_id,
            "name": f"GitHub pilot cross-repo de-referenced — {sys_name}",
            "created_at": CREATED_AT, "hypothesis": desc,
            "task": "issue_to_fixing_pr", "dataset_version": DATASET_ID,
            "code_version": "relsdlc-0.1.0", "seed": 0,
            "command": "python data/pilot/finetune_embed.py && "
                       "python data/pilot/run_finetune_ablation.py",
            "system": sys_name, "runtime_class": "cpu",
            "metrics": {
                "recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
                "mrr": round(m["mrr"], 4),
                "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
                "n_queries": m["n_queries"],
            },
            "baseline_comparison": compare, "error_slices": [],
            "leakage_checks": ["references scrubbed; LoRA trained on TRAIN repos only; "
                               "eval on held-out test repos."],
            "known_limitations": [
                "Pilot scale (~182 train pairs); general-text base; LoRA r=8. Exploratory.",
            ],
            "exploratory": True,
        })
    return out


def main() -> None:
    for p in (FROZEN, TUNED):
        if not p.exists():
            print(f"ERROR: missing {p}. Run embed_pilot.py / finetune_embed.py "
                  "(needs the [embed] extra).", file=sys.stderr)
            raise SystemExit(2)

    ds, meta = load_pilot_crossrepo()
    frozen, _, _ = _load(FROZEN)
    tuned, dim, tuned_model = _load(TUNED)

    base = run_ablation(ds, seed=0, min_df=3)["systems"]
    systems = {
        "idf-cosine": base["idf-cosine"],
        "embedder-cosine-frozen": run_cosine_on_vecs(ds, frozen),
        "embedder-cosine-lora": run_cosine_on_vecs(ds, tuned),
        "lora+relation-map": run_relation_map_on_vecs(ds, tuned, dim, seed=0,
                                                      epochs=300, lr=0.1, margin=0.1,
                                                      decay=2e-2),
    }

    for card in build_cards(systems, tuned_model):
        (CARDS / f"{card['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "finetune-results.json").write_text(
        json.dumps({"tuned_model": tuned_model, "systems": systems, "meta": meta},
                   indent=2) + "\n", encoding="utf-8")

    print(f"TRACK A — LoRA fine-tune vs frozen (de-referenced cross-repo, issue_to_fixing_pr)")
    print(f"train repos: {len(meta['train_repos'])}  test repos: {len(meta['test_repos'])}")
    print(f"{'system':<26}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'HardNegAcc':>12}")
    for name in ["idf-cosine", "embedder-cosine-frozen", "embedder-cosine-lora",
                 "lora+relation-map"]:
        m = systems[name]
        r = m["recall_at_k"]
        print(f"{name:<26}{r['1']:>8.3f}{r['5']:>8.3f}{r['10']:>8.3f}"
              f"{m['mrr']:>8.3f}{m['hard_negative_accuracy']:>12.3f}")


if __name__ == "__main__":
    main()
