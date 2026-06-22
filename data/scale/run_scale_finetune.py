#!/usr/bin/env python3
"""Does the LoRA win hold at 55-repo scale? Frozen vs LoRA-tuned, cross-repo.

Numpy only, on the committed scale caches (run finetune_scale.py first).

Run:  python data/scale/run_scale_finetune.py
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

from relsdlc.tower import run_cosine_on_vecs  # noqa: E402
from run_scale_ablation import load_scale_crossrepo  # noqa: E402

EMB = HERE / "embeddings"


def _load(name):
    d = np.load(EMB / name, allow_pickle=False)
    ids, V = d["ids"], np.asarray(d["vectors"], dtype=np.float32)  # decompress once (NpzFile lazy member)
    return {str(i): V[k] for k, i in enumerate(ids)}


def main() -> None:
    for n in ("minilm-l6-v2.npz", "minilm-lora.npz"):
        if not (EMB / n).exists():
            print(f"ERROR: missing {EMB / n}; run data/scale/finetune_scale.py", file=sys.stderr)
            raise SystemExit(2)
    ds, meta = load_scale_crossrepo()
    frozen = run_cosine_on_vecs(ds, _load("minilm-l6-v2.npz"))
    tuned = run_cosine_on_vecs(ds, _load("minilm-lora.npz"))
    d_r1 = tuned["recall_at_k"]["1"] - frozen["recall_at_k"]["1"]
    d_mrr = tuned["mrr"] - frozen["mrr"]

    print(f"LoRA-at-scale (55 repos; {len(meta['train_repos'])} train / "
          f"{len(meta['test_repos'])} test repos)")
    print(f"{'system':<22}{'R@1':>8}{'R@5':>8}{'MRR':>8}")
    print(f"{'frozen embedder':<22}{frozen['recall_at_k']['1']:>8.3f}"
          f"{frozen['recall_at_k']['5']:>8.3f}{frozen['mrr']:>8.3f}")
    print(f"{'LoRA-tuned':<22}{tuned['recall_at_k']['1']:>8.3f}"
          f"{tuned['recall_at_k']['5']:>8.3f}{tuned['mrr']:>8.3f}")
    print(f"delta                  R@1 {d_r1:+.3f}  MRR {d_mrr:+.3f}")

    card = {
        "card_type": "experiment", "id": "exp:gh-scale-lora-v0",
        "name": "55-repo scale — LoRA-tuned vs frozen (issue_to_fixing_pr)",
        "created_at": "2026-06-21T00:00:00Z",
        "hypothesis": "Does the pilot LoRA win hold on the ~55-repo dataset cross-repo?",
        "task": "issue_to_fixing_pr", "dataset_version": "ds:gh-scale2-v0",
        "code_version": "relsdlc-0.1.0", "seed": 0,
        "command": "python data/scale/finetune_scale.py && python data/scale/run_scale_finetune.py",
        "system": "minilm-lora-scale", "runtime_class": "cpu",
        "metrics": {"recall_at_k": {k: round(v, 4) for k, v in tuned["recall_at_k"].items()},
                    "mrr": round(tuned["mrr"], 4),
                    "hard_negative_accuracy": round(tuned["hard_negative_accuracy"], 4),
                    "n_queries": tuned.get("n_queries", 0)},
        "baseline_comparison": "exp:gh-scale2-idf-v0", "error_slices": [],
        "leakage_checks": ["LoRA trains on train-repo pairs only; eval held-out repos; refs scrubbed."],
        "known_limitations": [f"Single cross-repo split; ~55 repos; frozen R@1 "
                              f"{frozen['recall_at_k']['1']:.3f}, delta R@1 {d_r1:+.3f}. Exploratory."],
        "exploratory": True,
    }
    (CARDS / "gh-scale-lora-v0.experiment-card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "scale-finetune-results.json").write_text(
        json.dumps({"frozen": frozen, "tuned": tuned,
                    "delta_r1": d_r1, "delta_mrr": d_mrr, "meta": meta}, indent=2) + "\n")


if __name__ == "__main__":
    main()
