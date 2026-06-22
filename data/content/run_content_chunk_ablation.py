#!/usr/bin/env python3
"""Deep-signal chunking ablation: MaxP vs FirstP vs SumP vs meanP vs whole-doc-mean.

The R16A question. R15 found chunking/MaxP does NOT beat FirstP for
issue->fixing-PR, because that signal is **front-loaded** (the issue lede carries
it, so the first chunk already wins). The hypothesis here: chunking should pay off
where the relevant content is **deep** — long source/test FILE bodies, where the
matching code may be anywhere in the file, not in the first 512 chars.

Task: diff_to_affected_test, cross-repo. Query = the PR (its scrubbed title+body,
whole MiniLM embedding from queries.npz). DOCUMENT = a candidate test FILE, whose
deep text is represented as chunk vectors; the AGGREGATION over those chunks is
what varies — isolating "how to represent a long file":

  - FirstP         — first chunk only (the head of the file: imports + setup).
  - MaxP           — the file's single best-matching chunk (the operator's bet).
  - SumP           — sum of chunk scores (length-biased).
  - meanP          — mean of chunk scores.
  - whole-doc-mean — cosine to the mean chunk vector (R14's diluted loser).

Numpy-only, on the committed file_contents.jsonl + the (gitignored) caches.
Cross-repo split: query PR repo == candidate test repo, so splitting repos gives
disjoint train/test repos. There is no training here (frozen embeddings), so we
report on the TEST repos only — the held-out generalization slice.

Run:  python data/content/run_content_chunk_ablation.py
      (after data/content/embed_content.py)
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

from relsdlc.chunking import AGGREGATORS, rank  # noqa: E402
from relsdlc.metrics import RetrievalResult, evaluate  # noqa: E402

SIZES = [256, 512, 1024]
EMB = HERE / "embeddings"
BENCH = HERE / "benchmark" / "diff_to_affected_test.jsonl"
ORDER = ["whole-doc-mean", "firstp", "sump", "meanp", "maxp"]
TRAIN_REPO_FRAC = 0.6
CREATED_AT = "2026-06-22T00:00:00Z"
DATASET_ID = "ds:gh-content-v0"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n")
            if ln.strip()]


def _repo_of(rec_id: str) -> str:
    return rec_id.split(":")[1]


def _load_doc_chunks(size: int) -> dict[str, np.ndarray]:
    d = np.load(EMB / f"content-chunks-s{size}.npz", allow_pickle=False)
    ids = [str(i) for i in d["ids"]]
    off = d["offsets"]
    vecs = d["vectors"].astype(np.float32)
    return {i: vecs[off[k]:off[k + 1]] for k, i in enumerate(ids)}


def _load_queries() -> dict[str, np.ndarray]:
    d = np.load(EMB / "queries.npz", allow_pickle=False)
    ids = [str(i) for i in d["ids"]]
    vecs = d["vectors"].astype(np.float32)
    return {i: vecs[k] for k, i in enumerate(ids)}


def _crossrepo_test_split(queries: list[dict]) -> set[str]:
    """The TEST repos (held-out): the complement of the first 60% of repos."""
    repos = sorted({_repo_of(q["query_record"]) for q in queries})
    n_train = int(len(repos) * TRAIN_REPO_FRAC)
    train = set(repos[:n_train])
    return set(repos) - train


def main() -> None:
    missing = [s for s in SIZES if not (EMB / f"content-chunks-s{s}.npz").exists()]
    if missing or not (EMB / "queries.npz").exists():
        print(f"ERROR: missing caches (sizes {missing}, queries.npz present="
              f"{(EMB / 'queries.npz').exists()}); run data/content/embed_content.py",
              file=sys.stderr)
        raise SystemExit(2)

    bench = _load_jsonl(BENCH)
    qvecs = _load_queries()
    test_repos = _crossrepo_test_split(bench)
    train_repos = sorted({_repo_of(q["query_record"]) for q in bench} - test_repos)
    test_q = [q for q in bench if _repo_of(q["query_record"]) in test_repos]

    print(f"deep-signal chunking ablation — diff_to_affected_test, cross-repo")
    print(f"{len(train_repos)} train repos / {len(test_repos)} test repos; "
          f"{len(test_q)} held-out test queries (of {len(bench)} total)")
    print(f"query = PR whole-embedding; document = test-file chunk aggregation\n")
    print(f"{'chunk_size':<12}{'aggregator':<16}{'R@1':>8}{'R@5':>8}{'MRR':>8}"
          f"{'HardNegAcc':>12}")

    results: dict[str, dict] = {}
    for size in SIZES:
        docs = _load_doc_chunks(size)
        for agg_name in ORDER:
            agg = AGGREGATORS[agg_name]
            rr = []
            for q in test_q:
                qvec = qvecs.get(q["query_record"])
                if qvec is None:
                    continue
                cand = {c: docs[c] for c in q["candidates"] if c in docs}
                if not cand:
                    continue
                ranked = rank(qvec, cand, agg)
                rr.append(RetrievalResult.of(
                    ranked, q["relevant"], q.get("hard_negatives", [])))
            m = evaluate(rr)
            results[f"s{size}:{agg_name}"] = m
            print(f"{size:<12}{agg_name:<16}{m['recall_at_k']['1']:>8.3f}"
                  f"{m['recall_at_k']['5']:>8.3f}{m['mrr']:>8.3f}"
                  f"{m['hard_negative_accuracy']:>12.3f}")
        print()

    # Honest verdict: does MaxP beat FirstP on deep file contents?
    verdict_lines = []
    maxp_wins = 0
    for size in SIZES:
        fp = results[f"s{size}:firstp"]["recall_at_k"]["1"]
        mp = results[f"s{size}:maxp"]["recall_at_k"]["1"]
        delta = round(mp - fp, 4)
        won = mp > fp
        maxp_wins += int(won)
        verdict_lines.append(
            f"  size {size}: MaxP R@1 {mp:.3f} vs FirstP R@1 {fp:.3f}  "
            f"(delta {delta:+.3f}) -> {'MaxP wins' if won else 'FirstP >= MaxP'}")
    overall = ("MaxP beats FirstP at all chunk sizes" if maxp_wins == len(SIZES)
               else f"MaxP beats FirstP at {maxp_wins}/{len(SIZES)} chunk sizes"
               if maxp_wins else "FirstP >= MaxP at every chunk size")
    print("VERDICT — does chunking (MaxP) finally help where signal is DEEP?")
    for ln in verdict_lines:
        print(ln)
    print(f"  => {overall}")

    meta = {
        "train_repos": train_repos,
        "test_repos": sorted(test_repos),
        "n_test_queries": len(test_q),
        "task": "diff_to_affected_test",
    }
    (HERE / "content-chunk-results.json").write_text(
        json.dumps({
            "results": results, "meta": meta,
            "verdict": {"maxp_beats_firstp_at_n_sizes": maxp_wins,
                        "n_sizes": len(SIZES), "summary": overall},
        }, indent=2) + "\n", encoding="utf-8")

    # Cards for the headline pair at the best chunk size (by MaxP R@1).
    best = max(SIZES, key=lambda s: results[f"s{s}:maxp"]["recall_at_k"]["1"])
    for agg in ("firstp", "maxp"):
        m = results[f"s{best}:{agg}"]
        c = {
            "card_type": "experiment", "id": f"exp:gh-content-{agg}-s{best}-v0",
            "name": f"Deep-signal chunked retrieval — {agg} @ chunk_size {best} "
                    "(diff_to_affected_test)",
            "created_at": CREATED_AT,
            "hypothesis": "Does MaxP over chunks beat FirstP (the file head) when the "
                          "relevant signal is DEEP in long test-file contents?",
            "task": "diff_to_affected_test", "dataset_version": DATASET_ID,
            "code_version": "relsdlc-0.1.0", "seed": 0,
            "command": "python data/content/embed_content.py && "
                       "python data/content/run_content_chunk_ablation.py",
            "system": f"{agg}-content-chunk{best}", "runtime_class": "cpu",
            "metrics": {
                "recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
                "mrr": round(m["mrr"], 4),
                "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
                "n_queries": m.get("n_queries", 0),
            },
            "baseline_comparison": (f"exp:gh-content-firstp-s{best}-v0"
                                    if agg == "maxp" else "none"),
            "error_slices": [],
            "leakage_checks": [
                "references scrubbed from file text; train repos disjoint from test "
                "repos (cross-repo); frozen MiniLM embeddings (no fit on test repos); "
                "query = PR whole-embedding, document = test-file chunk aggregation.",
            ],
            "known_limitations": [
                "Frozen general-text MiniLM chunk embeddings (not code-specific); file "
                "text capped at 16000 chars; coarse MaxP (not token-level MaxSim); "
                "pilot scale. Exploratory.",
            ],
            "exploratory": True,
        }
        (CARDS / f"{c['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
