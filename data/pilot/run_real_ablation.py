#!/usr/bin/env python3
"""Re-run the vanilla / IDF / relation-metric ablation on the REAL pilot dataset.

This is the honest test of the synthetic result: does relation supervision beat
surface similarity on real public issue->fixing-PR retrieval? Deterministic given
the committed pilot snapshot (no network). Writes experiment cards + a results
JSON; prints the table.

Run:  python data/pilot/run_real_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CARDS = REPO_ROOT / "data" / "cards" / "examples"
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.model import run_ablation  # noqa: E402
from relsdlc.synth import Artifact, Query, SynthDataset  # noqa: E402

MIN_DF = 3
TRAIN_SEED = 0
CREATED_AT = "2026-06-21T00:00:00Z"
DATASET_ID = "ds:gh-pilot-v0"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_pilot() -> SynthDataset:
    records = _load_jsonl(HERE / "records.jsonl")
    queries_raw = _load_jsonl(HERE / "benchmark" / "issue_to_fixing_pr.jsonl")
    split = json.loads((HERE / "split.json").read_text(encoding="utf-8"))
    split_of = {**{i: "train" for i in split["train"]}, **{i: "test" for i in split["test"]}}

    artifacts = []
    for r in records:
        c = r.get("content", {})
        text = f"{c.get('title', '')} {c.get('body', '')}".strip()
        artifacts.append(Artifact(
            id=r["id"], type=r["type"], component=0, tokens=[text],
            split=r.get("split", "train"),
        ))

    fixes, queries = [], []
    for q in queries_raw:
        pos = q["relevant"][0]
        fixes.append((pos, q["query_record"]))
        queries.append(Query(
            query_id=q["query_id"], query_record=q["query_record"],
            candidates=q["candidates"], relevant=q["relevant"],
            hard_negatives=q.get("hard_negatives", []),
            split=split_of.get(q["query_record"], "train"),
        ))
    return SynthDataset(artifacts=artifacts, fixes=fixes, queries=queries,
                        params={"dataset": DATASET_ID, "min_df": MIN_DF})


def build_cards(ablation: dict) -> list[dict]:
    systems = ablation["systems"]
    out = []
    for sys_name, exp_id, compare in [
        ("vanilla-tf-cosine", "exp:gh-pilot-vanilla-v0", "none"),
        ("idf-cosine", "exp:gh-pilot-idf-v0", "exp:gh-pilot-vanilla-v0"),
        ("relation-metric", "exp:gh-pilot-relation-v0", "exp:gh-pilot-idf-v0"),
    ]:
        m = systems[sys_name]
        out.append({
            "card_type": "experiment",
            "id": exp_id,
            "name": f"GitHub pilot — {sys_name} (issue_to_fixing_pr)",
            "created_at": CREATED_AT,
            "hypothesis": "On real public issue->fixing-PR retrieval, does this "
                          "weighting beat plain cosine?",
            "task": "issue_to_fixing_pr",
            "dataset_version": DATASET_ID,
            "code_version": "relsdlc-0.1.0",
            "seed": TRAIN_SEED,
            "command": "python data/pilot/run_real_ablation.py",
            "system": sys_name,
            "runtime_class": "cpu",
            "metrics": {
                "recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
                "mrr": round(m["mrr"], 4),
                "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
                "n_queries": m["n_queries"],
            },
            "baseline_comparison": compare,
            "error_slices": [],
            "leakage_checks": ["temporal split by issue creation date; relation "
                               "metric trains only on train-split fixes."],
            "known_limitations": [
                "Pilot scale (~20 repos); fixes mined from closing keywords; "
                "body text truncated. Exploratory.",
            ],
            "exploratory": True,
        })
    return out


def main() -> None:
    ds = load_pilot()
    ablation = run_ablation(ds, seed=TRAIN_SEED, min_df=MIN_DF)
    for card in build_cards(ablation):
        out = CARDS / f"{card['id'].split(':', 1)[1]}.experiment-card.json"
        out.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "ablation-results.json").write_text(
        json.dumps(ablation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"REAL pilot ablation — issue_to_fixing_pr "
          f"(train q: {ablation['n_train_queries']}, test q: {ablation['n_test_queries']}, "
          f"vocab: {ablation['vocab_size']})")
    print(f"{'system':<20}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'HardNegAcc':>12}")
    for name, m in ablation["systems"].items():
        r = m["recall_at_k"]
        print(f"{name:<20}{r['1']:>8.3f}{r['5']:>8.3f}{r['10']:>8.3f}"
              f"{m['mrr']:>8.3f}{m['hard_negative_accuracy']:>12.3f}")
    lw = ablation["learned_weights"]
    print(f"learned weights — impl(mean)={lw['mean_impl_weight']:.3f} "
          f"topic(mean)={lw['mean_topic_weight']:.3f} (n/a for real tokens)")


if __name__ == "__main__":
    main()
