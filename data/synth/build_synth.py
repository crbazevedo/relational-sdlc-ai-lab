#!/usr/bin/env python3
"""Generate the frozen synthetic relation benchmark + the ablation experiment cards.

The DATA (records/edges/benchmark/split + source/dataset cards) is produced from a
seeded PCG64 stream only, so it is byte-identical across platforms and safe to
drift-check in CI. The EXPERIMENT CARDS contain metrics from numpy training, whose
last-decimal values can vary across BLAS/platforms; they are written only in full
mode and are NOT regenerated in CI. A test asserts the qualitative ablation result
with tolerances instead.

Run:
  python data/synth/build_synth.py            # data + cards (local authoring)
  python data/synth/build_synth.py --data-only # deterministic data only (CI)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CARDS = REPO_ROOT / "data" / "cards" / "examples"
sys.path.insert(0, str(REPO_ROOT / "src"))

from relsdlc.synth import generate  # noqa: E402

SEED = 7
ABLATION_SEED = 0
CREATED_AT = "2024-06-01T00:00:00Z"
AS_OF = "2025-01-01T00:00:00Z"
RECORD_VALID_FROM = "2024-01-01T00:00:00Z"
EDGE_VALID_FROM = "2024-06-01T00:00:00Z"
LICENSE = "CC0-1.0"
SOURCE = "synthetic://relation-bench"
TRANSFORM = "python data/synth/build_synth.py"
DATASET_ID = "ds:synth-relation-v0"


def _hash(payload) -> str:
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _prov(extra_hash_payload, method="synthetic") -> dict:
    return {
        "source_url": SOURCE,
        "retrieved_at": CREATED_AT,
        "license": LICENSE,
        "content_hash": _hash(extra_hash_payload),
        "transform": TRANSFORM,
        "method": method,
        "observed": True,
    }


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_records(ds) -> list[dict]:
    out = []
    for a in ds.artifacts:
        content = {"title": f"{a.type} {a.id}", "body": a.text}
        out.append({
            "id": a.id,
            "type": a.type,
            "content": content,
            "valid_from": RECORD_VALID_FROM,
            "split": a.split,
            "attributes": {"component": a.component},
            "provenance": _prov(content),
        })
    return out


def build_edges(ds) -> list[dict]:
    out = []
    for pr_id, issue_id in ds.fixes:
        key = {"source": pr_id, "relation": "fixes", "target": issue_id}
        out.append({
            "source": pr_id,
            "relation": "fixes",
            "target": issue_id,
            "confidence": 1.0,
            "valid_from": EDGE_VALID_FROM,
            "provenance": _prov(key),
        })
    return out


def build_benchmark(ds) -> list[dict]:
    # The committed eval set is the TEST-split queries.
    out = []
    for q in ds.queries:
        if q.split != "test":
            continue
        out.append({
            "query_id": q.query_id,
            "task": "issue_to_fixing_pr",
            "query_record": q.query_record,
            "candidates": q.candidates,
            "relevant": q.relevant,
            "hard_negatives": q.hard_negatives,
            "as_of": AS_OF,
        })
    return out


def build_source_card(manifest_hash: str) -> dict:
    return {
        "card_type": "source",
        "id": "src:synth-relation",
        "name": "synthetic relation benchmark generator",
        "source_url": SOURCE,
        "retrieved_at": CREATED_AT,
        "license": LICENSE,
        "terms_note": "Original synthetic data; freely redistributable.",
        "record_types": ["issue", "pull_request"],
        "transform": TRANSFORM,
        "content_hash": manifest_hash,
        "redistribution": "synthetic_original",
        "notes": "Latent-component design: rare impl tokens carry the fix link; "
                 "common topic tokens are misleading surface noise.",
    }


def build_dataset_card(records, edges, manifest_hash, params) -> dict:
    return {
        "card_type": "dataset",
        "id": DATASET_ID,
        "name": "synthetic relation benchmark",
        "version": "v0",
        "created_at": CREATED_AT,
        "sources": ["src:synth-relation"],
        "record_counts": dict(sorted(Counter(r["type"] for r in records).items())),
        "edge_counts": dict(sorted(Counter(e["relation"] for e in edges).items())),
        "relation_types": sorted({e["relation"] for e in edges}),
        "split_policy": {
            "frozen": True,
            "method": "synthetic-structural-by-record",
            "seed": params["seed"],
            "boundary": "record.split field (train/test)",
        },
        "redistribution": "synthetic_original",
        "known_limitations": [
            "Synthetic. Demonstrates the mechanism (relation supervision beats "
            "vanilla/IDF surface similarity); NOT a real-world result.",
            "Single relation (fixes), single task (issue_to_fixing_pr).",
        ],
        "notes": f"Built by {TRANSFORM} seed={params['seed']}. Manifest {manifest_hash}.",
    }


def build_experiment_cards(ablation: dict) -> list[dict]:
    systems = ablation["systems"]
    cards = []
    for sys_name, exp_id, hypo, compare in [
        ("vanilla-tf-cosine", "exp:synth-vanilla-v0",
         "Plain cosine retrieves the fixing PR for an issue.", "none"),
        ("idf-cosine", "exp:synth-idf-v0",
         "Unsupervised IDF reweighting beats plain cosine.", "exp:synth-vanilla-v0"),
        ("relation-metric", "exp:synth-relation-v0",
         "Relation-supervised token weighting beats vanilla AND unsupervised IDF.",
         "exp:synth-idf-v0"),
    ]:
        m = systems[sys_name]
        cards.append({
            "card_type": "experiment",
            "id": exp_id,
            "name": f"Synthetic ablation — {sys_name} (issue_to_fixing_pr)",
            "created_at": CREATED_AT,
            "hypothesis": hypo,
            "task": "issue_to_fixing_pr",
            "dataset_version": DATASET_ID,
            "code_version": "relsdlc-0.1.0",
            "seed": ABLATION_SEED,
            "command": "relsdlc ablation",
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
            "leakage_checks": [
                "train/test split by record; relation metric trains only on "
                "train-split fixes; eval on held-out test issues.",
            ],
            "known_limitations": [
                "Synthetic dataset; exploratory. Real public-data validation is the "
                "P1->P2 follow-up.",
            ],
            "exploratory": True,
        })
    return cards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-only", action="store_true",
                    help="write only the deterministic data + source/dataset cards (CI)")
    args = ap.parse_args()

    ds = generate(seed=SEED)
    records, edges, benchmark = build_records(ds), build_edges(ds), build_benchmark(ds)

    _write_jsonl(HERE / "records.jsonl", records)
    _write_jsonl(HERE / "edges.jsonl", edges)
    _write_jsonl(HERE / "benchmark" / "issue_to_fixing_pr.jsonl", benchmark)
    _write_json(HERE / "split.json", {
        "train": sorted(a.id for a in ds.artifacts if a.split == "train"),
        "test": sorted(a.id for a in ds.artifacts if a.split == "test"),
        "params": ds.params,
    })

    manifest_hash = "sha256:" + hashlib.sha256(
        (HERE / "records.jsonl").read_bytes() + (HERE / "edges.jsonl").read_bytes()
    ).hexdigest()
    _write_json(CARDS / "synth-relation.source-card.json", build_source_card(manifest_hash))
    _write_json(CARDS / "synth-relation-v0.dataset-card.json",
                build_dataset_card(records, edges, manifest_hash, ds.params))

    if args.data_only:
        print(f"wrote synth DATA to {HERE} (data-only)")
        return

    from relsdlc.model import run_ablation  # noqa: E402
    ablation = run_ablation(ds, seed=ABLATION_SEED)
    for card in build_experiment_cards(ablation):
        _write_json(CARDS / f"{card['id'].split(':', 1)[1]}.experiment-card.json", card)
    _write_json(HERE / "ablation-results.json", ablation)
    print(f"wrote synth DATA + ablation cards to {HERE} and {CARDS}")


if __name__ == "__main__":
    main()
