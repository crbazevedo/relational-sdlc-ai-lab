#!/usr/bin/env python3
"""Q6 finish — does a TRUE code-EMBEDDING base beat the general substrate?

Numpy only, on the committed caches, same de-referenced cross-repo split as the
earlier code ablations. Compares the two frozen general baselines (MiniLM 0.592,
bge 0.598) against the genuine code-embedding base that finally loaded under a
PINNED transformers<5 env (see embed_code3_pinned.py / docs/ablation-code3.md):

  * minilm-l6 (substrate)             — general, embedding-tuned (frozen baseline)
  * bge-small-en (stronger general)   — general, embedding-tuned (frozen baseline)
  * jina-code (code + embedding-tuned) — the genuine code-EMBEDDING base (Q6 target)

Reuses existing caches where present; skips any missing cache (so it still runs in
CI on whatever is committed). Writes experiment cards + code3-results.json. The
eval itself is transformers-version-agnostic — it only reads the committed npz.

Run:  python data/pilot/run_code3_ablation.py
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

# (cache_name, label, sysid). The new code-embedding base is detected from the
# first code3-*.npz present, so this script tracks whichever model embed_code3
# chose without editing here.
BASELINES = [
    ("minilm-l6-v2.npz", "minilm-l6 (substrate)", "minilml6v2"),
    ("code-bge.npz", "bge-small-en (stronger general)", "codebge"),
]
CODE3_GLOB = "code3-*.npz"


def _load(name):
    d = np.load(HERE / "embeddings" / name, allow_pickle=False)
    ids, V = d["ids"], np.asarray(d["vectors"], dtype=np.float32)  # decompress once (NpzFile lazy member)
    return {str(i): V[k] for k, i in enumerate(ids)}


def _code3_caches():
    """Every committed code3-*.npz, each labelled by its embedded model name."""
    out = []
    for path in sorted((HERE / "embeddings").glob(CODE3_GLOB)):
        model = str(np.load(path, allow_pickle=False)["model"])
        short = model.split("/")[-1]
        sysid = "code3" + "".join(c for c in short.lower() if c.isalnum())
        out.append((path.name, f"{short} (code + embed-tuned)", sysid))
    return out


def _card(sysid, label, m):
    return {
        "card_type": "experiment", "id": f"exp:gh-code3-{sysid}-v0",
        "name": f"GitHub pilot cross-repo — frozen {label}",
        "created_at": "2026-06-21T00:00:00Z",
        "hypothesis": "Does a TRUE code-EMBEDDING base (loaded under a pinned "
                      "transformers<5 env) beat the general substrate on frozen "
                      "cross-repo retrieval (Q6 finish)?",
        "task": "issue_to_fixing_pr", "dataset_version": "ds:gh-pilot-v0",
        "code_version": "relsdlc-0.1.0", "seed": 0,
        "command": "uv pip install --python .venv-r15b 'transformers>=4.40,<5' && "
                   ".venv-r15b/bin/python data/pilot/embed_code3_pinned.py && "
                   "python data/pilot/run_code3_ablation.py",
        "system": label, "runtime_class": "cpu",
        "metrics": {
            "recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
            "mrr": round(m["mrr"], 4),
            "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
            "n_queries": m["n_queries"]},
        "baseline_comparison": "exp:gh-xrepo-idf-v0", "error_slices": [],
        "leakage_checks": ["frozen embeddings; references scrubbed; cross-repo split."],
        "known_limitations": [
            "Pilot scale; frozen (no fine-tune). Exploratory. The code-embedding "
            "base required a pinned transformers<5 env to load (remote code)."],
        "exploratory": True,
    }


def main() -> None:
    ds, meta = load_pilot_crossrepo()
    bases = list(BASELINES) + _code3_caches()
    systems = {}
    cards = []
    print("Q6 finish — TRUE code-EMBEDDING base vs general substrate "
          "(de-referenced cross-repo, issue_to_fixing_pr)")
    print(f"{'base':<46}{'R@1':>8}{'R@5':>8}{'MRR':>8}")
    for name, label, sysid in bases:
        path = HERE / "embeddings" / name
        if not path.exists():
            print(f"  (skip {label}: missing {name})", file=sys.stderr)
            continue
        m = run_cosine_on_vecs(ds, _load(name))
        systems[label] = m
        print(f"{label:<46}{m['recall_at_k']['1']:>8.3f}"
              f"{m['recall_at_k']['5']:>8.3f}{m['mrr']:>8.3f}")
        cards.append(_card(sysid, label, m))

    for c in cards:
        (CARDS / f"{c['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "code3-results.json").write_text(
        json.dumps({"systems": systems, "meta": meta}, indent=2) + "\n")


if __name__ == "__main__":
    main()
