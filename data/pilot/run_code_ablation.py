#!/usr/bin/env python3
"""Q6 — does the embedding BASE matter? Frozen MiniLM vs bge-small vs codebert.

Numpy only, on the committed caches, same de-referenced cross-repo split.

Run:  python data/pilot/run_code_ablation.py
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
from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402

BASES = [
    ("minilm-l6-v2.npz", "minilm-l6 (substrate)"),
    ("code-bge.npz", "bge-small-en (stronger general)"),
    ("code-codebert.npz", "codebert (code-pretrained, not embed-tuned)"),
]


def _load(name):
    d = np.load(HERE / "embeddings" / name, allow_pickle=False)
    return {str(i): d["vectors"][k].astype(np.float32) for k, i in enumerate(d["ids"])}


def main() -> None:
    ds, meta = load_pilot_crossrepo()
    systems = {}
    print("Q6 — base-model comparison (de-referenced cross-repo, issue_to_fixing_pr)")
    print(f"{'base':<44}{'R@1':>8}{'R@5':>8}{'MRR':>8}")
    cards = []
    for name, label in BASES:
        path = HERE / "embeddings" / name
        if not path.exists():
            print(f"  (skip {label}: missing {name})", file=sys.stderr); continue
        m = run_cosine_on_vecs(ds, _load(name))
        systems[label] = m
        print(f"{label:<44}{m['recall_at_k']['1']:>8.3f}{m['recall_at_k']['5']:>8.3f}{m['mrr']:>8.3f}")
        sysid = name.replace(".npz", "").replace("-", "")
        cards.append({
            "card_type": "experiment", "id": f"exp:gh-code-{sysid}-v0",
            "name": f"GitHub pilot cross-repo — frozen {label}",
            "created_at": "2026-06-21T00:00:00Z",
            "hypothesis": "Does the embedding base model change frozen cross-repo retrieval (Q6)?",
            "task": "issue_to_fixing_pr", "dataset_version": "ds:gh-pilot-v0",
            "code_version": "relsdlc-0.1.0", "seed": 0,
            "command": "python data/pilot/embed_code.py && python data/pilot/run_code_ablation.py",
            "system": label, "runtime_class": "cpu",
            "metrics": {"recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
                        "mrr": round(m["mrr"], 4),
                        "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
                        "n_queries": m["n_queries"]},
            "baseline_comparison": "exp:gh-embed-cosine-v0", "error_slices": [],
            "leakage_checks": ["frozen embeddings; references scrubbed; cross-repo split."],
            "known_limitations": ["Pilot scale; frozen (no fine-tune). Exploratory."],
            "exploratory": True,
        })
    for c in cards:
        (CARDS / f"{c['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "code-results.json").write_text(
        json.dumps({"systems": systems, "meta": meta}, indent=2) + "\n")


if __name__ == "__main__":
    main()
