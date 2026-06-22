#!/usr/bin/env python3
"""Re-confirm the bag-of-tokens baselines on the Tier-2-entry SCALE dataset.

Wave R12A (Track D scale). This re-runs the vanilla / IDF / diagonal-relation
ablation on the larger ``data/scale/`` snapshot, to check whether the pilot's
robust finding — **unsupervised IDF weighting reliably beats plain cosine on
real issue->fixing-PR retrieval** — survives at ~55-repo scale.

numpy-only (NO torch). Deterministic given the committed snapshot (no network).

Two systems beyond vanilla, scored on the SAME unit vectors so the comparison is
fair (see src/relsdlc/model.py):

- ``vanilla-tf-cosine`` — plain cosine on bag-of-token vectors (the floor).
- ``idf-cosine`` — unsupervised corpus IDF weighting (no relation labels).
- ``relation-metric`` — a diagonal metric supervised by the ``fixes`` relation.

Like the pilot's cross-repo ablation it also (1) SCRUBS explicit cross-references
(#N, gh-N, URLs, SHAs) from the text so retrieval can't string-match the issue
number, and (2) splits train/test by REPOSITORY so a win must generalize to
unseen repos. Embeddings / LoRA at scale are a torch follow-up (see
docs/ablation-scale.md, docs/ablation-finetune.md); this wave re-confirms the
bag-of-tokens baselines and grows the data.

Run:  python data/scale/run_scale_ablation.py
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
from relsdlc.scrub import scrub_record_text  # noqa: E402
from relsdlc.synth import Artifact, Query, SynthDataset  # noqa: E402

MIN_DF = 3
TRAIN_REPO_FRAC = 0.6
CREATED_AT = "2026-06-21T00:00:00Z"
DATASET_ID = "ds:gh-scale2-v0"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _repo_of(record_id: str) -> str:
    # "gh:owner/repo:issue:N" -> "owner/repo"
    parts = record_id.split(":")
    return parts[1] if len(parts) > 1 else record_id


def load_scale_crossrepo() -> tuple[SynthDataset, dict]:
    records = _load_jsonl(HERE / "records.jsonl")
    queries_raw = _load_jsonl(HERE / "benchmark" / "issue_to_fixing_pr.jsonl")

    repos = sorted({_repo_of(r["id"]) for r in records})
    n_train = int(len(repos) * TRAIN_REPO_FRAC)
    train_repos = set(repos[:n_train])

    def split_of(record_id: str) -> str:
        return "train" if _repo_of(record_id) in train_repos else "test"

    artifacts = [
        Artifact(id=r["id"], type=r["type"], component=0,
                 tokens=[scrub_record_text(r)], split=split_of(r["id"]))
        for r in records
    ]
    fixes, queries = [], []
    for q in queries_raw:
        pos = q["relevant"][0]
        fixes.append((pos, q["query_record"]))
        queries.append(Query(
            query_id=q["query_id"], query_record=q["query_record"],
            candidates=q["candidates"], relevant=q["relevant"],
            hard_negatives=q.get("hard_negatives", []),
            split=split_of(q["query_record"])))
    ds = SynthDataset(artifacts=artifacts, fixes=fixes, queries=queries,
                      params={"dataset": DATASET_ID, "min_df": MIN_DF,
                              "split": "cross-repo", "dereferenced": True})
    meta = {"train_repos": sorted(train_repos),
            "test_repos": sorted(set(repos) - train_repos)}
    return ds, meta


def build_cards(systems: dict) -> list[dict]:
    out = []
    for sys_name, exp_id, compare in [
        ("vanilla-tf-cosine", "exp:gh-scale2-vanilla-v0", "none"),
        ("idf-cosine", "exp:gh-scale2-idf-v0", "exp:gh-scale2-vanilla-v0"),
        ("relation-metric", "exp:gh-scale2-relation-v0", "exp:gh-scale2-idf-v0"),
    ]:
        m = systems[sys_name]
        out.append({
            "card_type": "experiment", "id": exp_id,
            "name": f"GitHub Tier-2-entry scale, cross-repo de-referenced — {sys_name}",
            "created_at": CREATED_AT,
            "hypothesis": "At ~55-repo scale, with references removed and repos "
                          "held out, does the pilot's robust finding hold — does "
                          "this weighting still beat plain cosine on issue->"
                          "fixing-PR retrieval that generalizes across repos?",
            "task": "issue_to_fixing_pr", "dataset_version": DATASET_ID,
            "code_version": "relsdlc-0.1.0", "seed": 0,
            "command": "python data/scale/run_scale_ablation.py",
            "system": sys_name, "runtime_class": "cpu",
            "metrics": {
                "recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
                "mrr": round(m["mrr"], 4),
                "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
                "n_queries": m["n_queries"],
            },
            "baseline_comparison": compare, "error_slices": [],
            "leakage_checks": ["references scrubbed from text; train repos disjoint "
                               "from test repos (cross-repo generalization)."],
            "known_limitations": [
                "Tier-2-entry scale (~55 repos); bag-of-tokens features; fixes "
                "mined from closing keywords; body text truncated. Exploratory. "
                "Embeddings / LoRA at scale are a torch follow-up.",
            ],
            "exploratory": True,
        })
    return out


def main() -> None:
    ds, meta = load_scale_crossrepo()
    base = run_ablation(ds, seed=0, min_df=MIN_DF)
    systems = dict(base["systems"])

    for card in build_cards(systems):
        (CARDS / f"{card['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "scale-results.json").write_text(
        json.dumps({"systems": systems, "meta": meta,
                    "n_test_queries": base["n_test_queries"],
                    "n_train_queries": base["n_train_queries"],
                    "vocab_size": base["vocab_size"]}, indent=2) + "\n", encoding="utf-8")

    print("SCALE (Tier-2-entry) de-referenced cross-repo ablation — issue_to_fixing_pr")
    print(f"train repos: {len(meta['train_repos'])}  test repos: {len(meta['test_repos'])}  "
          f"(train q: {base['n_train_queries']}, test q: {base['n_test_queries']}, "
          f"vocab: {base['vocab_size']})")
    print(f"{'system':<20}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'HardNegAcc':>12}")
    for name in ["vanilla-tf-cosine", "idf-cosine", "relation-metric"]:
        m = systems[name]
        r = m["recall_at_k"]
        print(f"{name:<20}{r['1']:>8.3f}{r['5']:>8.3f}{r['10']:>8.3f}"
              f"{m['mrr']:>8.3f}{m['hard_negative_accuracy']:>12.3f}")


if __name__ == "__main__":
    main()
