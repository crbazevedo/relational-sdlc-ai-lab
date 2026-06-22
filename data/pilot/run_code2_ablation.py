#!/usr/bin/env python3
"""Q6 follow-up — does a code-AWARE *embedding* base beat the general substrate?

Numpy only, on the committed caches, same de-referenced cross-repo split as
run_code_ablation.py. Compares the MiniLM substrate and the bge-small general
embedder (reused from the Q6 caches if present) against two code-aware bases:

  * unixcoder-base — code-pretrained (RoBERTa), NOT embedding-tuned (mean-pooled).
  * st-codesearch-distilroberta-base — code-aware AND embedding-tuned (contrastive
    on CodeSearchNet) — the actual Q6-follow-up target.

Reuses existing caches where present; skips any missing cache (so it still runs in
CI on whatever is committed). Writes experiment cards + code2-results.json.

Run:  python data/pilot/run_code2_ablation.py
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

# (cache_name, label, card_system, card_sysid)
BASES = [
    ("minilm-l6-v2.npz", "minilm-l6 (substrate)",
     "minilm-l6 (substrate)", "minilml6v2"),
    ("code-bge.npz", "bge-small-en (stronger general)",
     "bge-small-en (stronger general)", "codebge"),
    ("code2-unixcoder.npz", "unixcoder (code, not embed-tuned)",
     "unixcoder (code, not embed-tuned)", "unixcoder"),
    ("code2-stcodesearch.npz", "st-codesearch (code + embed-tuned)",
     "st-codesearch (code + embed-tuned)", "stcodesearch"),
]


def _load(name):
    d = np.load(HERE / "embeddings" / name, allow_pickle=False)
    ids, V = d["ids"], np.asarray(d["vectors"], dtype=np.float32)  # decompress once (NpzFile lazy member)
    return {str(i): V[k] for k, i in enumerate(ids)}


def _card(sysid, label, m):
    return {
        "card_type": "experiment", "id": f"exp:gh-code2-{sysid}-v0",
        "name": f"GitHub pilot cross-repo — frozen {label}",
        "created_at": "2026-06-21T00:00:00Z",
        "hypothesis": "Does a code-AWARE embedding base beat the general substrate "
                      "on frozen cross-repo retrieval (Q6 follow-up)?",
        "task": "issue_to_fixing_pr", "dataset_version": "ds:gh-pilot-v0",
        "code_version": "relsdlc-0.1.0", "seed": 0,
        "command": "python data/pilot/embed_code2.py && "
                   "python data/pilot/run_code2_ablation.py",
        "system": label, "runtime_class": "cpu",
        "metrics": {
            "recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
            "mrr": round(m["mrr"], 4),
            "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
            "n_queries": m["n_queries"]},
        "baseline_comparison": "exp:gh-embed-cosine-v0", "error_slices": [],
        "leakage_checks": ["frozen embeddings; references scrubbed; cross-repo split."],
        "known_limitations": ["Pilot scale; frozen (no fine-tune). Exploratory."],
        "exploratory": True,
    }


def main() -> None:
    ds, meta = load_pilot_crossrepo()
    systems = {}
    cards = []
    print("Q6 follow-up — code-aware embedding base "
          "(de-referenced cross-repo, issue_to_fixing_pr)")
    print(f"{'base':<46}{'R@1':>8}{'R@5':>8}{'MRR':>8}")
    for name, label, card_sys, sysid in BASES:
        path = HERE / "embeddings" / name
        if not path.exists():
            print(f"  (skip {label}: missing {name})", file=sys.stderr)
            continue
        m = run_cosine_on_vecs(ds, _load(name))
        systems[label] = m
        print(f"{label:<46}{m['recall_at_k']['1']:>8.3f}"
              f"{m['recall_at_k']['5']:>8.3f}{m['mrr']:>8.3f}")
        cards.append(_card(sysid, card_sys, m))

    for c in cards:
        (CARDS / f"{c['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "code2-results.json").write_text(
        json.dumps({"systems": systems, "meta": meta}, indent=2) + "\n")


if __name__ == "__main__":
    main()
