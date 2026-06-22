#!/usr/bin/env python3
"""Chunked retrieval ablation: MaxP vs FirstP vs SumP vs whole-doc-mean × chunk size.

Numpy only, on the committed full-text records + the chunk caches (chunks-s*.npz).
Query = the issue's FIRST chunk (R14: the issue lede carries the signal); the
DOCUMENT (PR) aggregation is what varies — isolating "how to represent the long PR."
The pivotal comparison is MaxP vs FirstP: does chunking recover signal beyond the lede?

Run:  python data/full/run_chunk_ablation.py   (after data/full/embed_chunks.py)
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

from relsdlc.chunking import AGGREGATORS, rank  # noqa: E402
from relsdlc.metrics import RetrievalResult, evaluate  # noqa: E402
from run_full_ablation import load_full_crossrepo  # noqa: E402

SIZES = [256, 512, 1024]
EMB = HERE / "embeddings"
ORDER = ["whole-doc-mean", "firstp", "sump", "meanp", "maxp"]


def _load_chunks(size):
    d = np.load(EMB / f"chunks-s{size}.npz", allow_pickle=False)
    ids = [str(i) for i in d["ids"]]
    off = d["offsets"]
    vecs = d["vectors"].astype(np.float32)
    return {i: vecs[off[k]:off[k + 1]] for k, i in enumerate(ids)}


def main() -> None:
    missing = [s for s in SIZES if not (EMB / f"chunks-s{s}.npz").exists()]
    if missing:
        print(f"ERROR: missing chunk caches {missing}; run data/full/embed_chunks.py",
              file=sys.stderr)
        raise SystemExit(2)
    ds, meta = load_full_crossrepo()
    test_q = [q for q in ds.queries if q.split == "test"]
    print(f"chunked ablation — issue_to_fixing_pr, {len(test_q)} held-out test queries "
          f"({len(meta['train_repos'])} train / {len(meta['test_repos'])} test repos)")
    print(f"{'chunk_size':<12}{'aggregator':<16}{'R@1':>8}{'R@5':>8}{'MRR':>8}")

    results = {}
    for size in SIZES:
        chunks = _load_chunks(size)
        for agg_name in ORDER:
            agg = AGGREGATORS[agg_name]
            rr = []
            for q in test_q:
                qmat = chunks.get(q.query_record)
                if qmat is None or qmat.shape[0] == 0:
                    continue
                qvec = qmat[0]  # issue first chunk
                cand = {c: chunks[c] for c in q.candidates if c in chunks}
                ranked = rank(qvec, cand, agg)
                rr.append(RetrievalResult.of(ranked, q.relevant, q.hard_negatives))
            m = evaluate(rr)
            results[f"s{size}:{agg_name}"] = m
            print(f"{size:<12}{agg_name:<16}{m['recall_at_k']['1']:>8.3f}"
                  f"{m['recall_at_k']['5']:>8.3f}{m['mrr']:>8.3f}")
        print()

    (HERE / "chunk-results.json").write_text(json.dumps({"results": results, "meta": meta},
                                                        indent=2) + "\n")
    # Cards for the headline pair at the best chunk size (by MaxP R@1).
    best = max(SIZES, key=lambda s: results[f"s{s}:maxp"]["recall_at_k"]["1"])
    for agg in ("firstp", "maxp"):
        m = results[f"s{best}:{agg}"]
        c = {
            "card_type": "experiment", "id": f"exp:gh-chunk-{agg}-s{best}-v0",
            "name": f"Chunked retrieval — {agg} @ chunk_size {best} (issue_to_fixing_pr)",
            "created_at": "2026-06-21T00:00:00Z",
            "hypothesis": "Does MaxP over chunks recover signal beyond FirstP (the lede)?",
            "task": "issue_to_fixing_pr", "dataset_version": "ds:gh-full-v0",
            "code_version": "relsdlc-0.1.0", "seed": 0,
            "command": "python data/full/embed_chunks.py && python data/full/run_chunk_ablation.py",
            "system": f"{agg}-chunk{best}", "runtime_class": "cpu",
            "metrics": {"recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
                        "mrr": round(m["mrr"], 4),
                        "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
                        "n_queries": m.get("n_queries", 0)},
            "baseline_comparison": "exp:gh-full-embed-cosine-v0", "error_slices": [],
            "leakage_checks": ["frozen embeddings; cross-repo split; query=issue first chunk."],
            "known_limitations": ["Frozen MiniLM chunk embeddings; query=issue first chunk "
                                  "(not full MaxSim); pilot scale. Exploratory."],
            "exploratory": True,
        }
        (CARDS / f"{c['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
