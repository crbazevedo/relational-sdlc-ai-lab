#!/usr/bin/env python3
"""FULL-TEXT cross-repo issue->fixing-PR ablation + the full-vs-truncated lift.

Same de-referenced, cross-repo protocol as data/pilot/run_crossrepo_ablation.py
(references scrubbed; train repos disjoint from test repos), but on the FULL-TEXT
dataset (data/full, bodies up to 8000 chars) and with the 512-window embedding
cache. This is the measurement R14 exists to make: how much did the pilot's
500-char truncation cost us?

Four systems, numpy-only (consumes the committed embedding cache, no torch):

  1. vanilla-tf-cosine     bag-of-tokens cosine
  2. idf-cosine            bag-of-tokens + IDF        <- the bag-of-tokens bar
  3. relation-metric       bag-of-tokens diagonal metric
  4. embedder-cosine(512)  frozen MiniLM @ max_length=512, raw cosine

It then reports the lift two ways:

* PAIRED CONTROL (the clean A/B) — the SAME records and split, re-run with bodies
  truncated to 500 chars and the embedder at max_length=256 (the pilot regime).
  Truncation is the only variable, so this lift is causal.
* CROSS-SNAPSHOT reference — the four systems on the frozen truncated pilot (read
  from data/pilot/embed-results.json). Suggestive, but it conflates truncation
  with snapshot drift (a different fetch, different issue/PR set).

Writes: data/cards/examples/gh-full-*.experiment-card.json + data/full/full-results.json.

Run:  python data/full/run_full_ablation.py
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

from relsdlc.model import run_ablation  # noqa: E402
from relsdlc.scrub import scrub_record_text  # noqa: E402
from relsdlc.synth import Artifact, Query, SynthDataset  # noqa: E402
from relsdlc.tower import run_cosine_on_vecs  # noqa: E402

EMB = HERE / "embeddings" / "minilm-l6-v2-512.npz"
EMB_TRUNC = HERE / "embeddings" / "minilm-l6-v2-trunc500.npz"
PILOT_TRUNC_RESULTS = REPO_ROOT / "data" / "pilot" / "embed-results.json"
TRUNC_CHARS = 500  # the pilot's char cap, re-imposed on the SAME records.
MIN_DF = 3
TRAIN_REPO_FRAC = 0.6
CREATED_AT = "2026-06-21T00:00:00Z"
DATASET_ID = "ds:gh-full-v0"

# The four comparable systems, in print order.
SYSTEMS = ["vanilla-tf-cosine", "idf-cosine", "relation-metric", "embedder-cosine"]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]


def _repo_of(record_id: str) -> str:
    # "gh-full:owner/repo:issue:N" -> "owner/repo"
    parts = record_id.split(":")
    return parts[1] if len(parts) > 1 else record_id


def load_full_crossrepo(trunc_chars: int | None = None) -> tuple[SynthDataset, dict]:
    """Cross-repo, de-referenced dataset over data/full (adapted from the pilot).

    ``trunc_chars`` re-imposes the pilot's char-cap on the SAME records/split for
    the paired control (truncation as the only variable).
    """
    records = _load_jsonl(HERE / "records.jsonl")
    queries_raw = _load_jsonl(HERE / "benchmark" / "issue_to_fixing_pr.jsonl")

    repos = sorted({_repo_of(r["id"]) for r in records})
    n_train = int(len(repos) * TRAIN_REPO_FRAC)
    train_repos = set(repos[:n_train])

    def split_of(record_id: str) -> str:
        return "train" if _repo_of(record_id) in train_repos else "test"

    def text_of(r: dict) -> str:
        t = scrub_record_text(r)
        if trunc_chars is not None and len(t) > trunc_chars:
            t = t[:trunc_chars]
        return t

    artifacts = [
        Artifact(id=r["id"], type=r["type"], component=0,
                 tokens=[text_of(r)], split=split_of(r["id"]))
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
                              "split": "cross-repo", "dereferenced": True,
                              "body_cap": 8000})
    meta = {"train_repos": sorted(train_repos),
            "test_repos": sorted(set(repos) - train_repos)}
    return ds, meta


def load_embeddings(path: Path = EMB) -> tuple[dict, int, str, int]:
    data = np.load(path, allow_pickle=False)
    ids = [str(i) for i in data["ids"]]
    vectors = data["vectors"].astype(np.float32)
    model = str(data["model"]) if "model" in data else "unknown"
    max_len = int(data["max_length"]) if "max_length" in data else 512
    return {i: vectors[k] for k, i in enumerate(ids)}, vectors.shape[1], model, max_len


def load_truncated_baseline() -> dict:
    """The same four systems on the TRUNCATED pilot (the number to beat)."""
    if not PILOT_TRUNC_RESULTS.exists():
        return {}
    d = json.loads(PILOT_TRUNC_RESULTS.read_text(encoding="utf-8"))
    out = {}
    for name in SYSTEMS:
        m = d.get("systems", {}).get(name)
        if m:
            out[name] = m
    return out


def build_cards(systems: dict, model: str, max_len: int) -> list[dict]:
    specs = [
        ("vanilla-tf-cosine", "exp:gh-full-vanilla-v0", "none",
         "On full body text, does bag-of-tokens cosine recover issue->fixing-PR "
         "across held-out repos?"),
        ("idf-cosine", "exp:gh-full-idf-v0", "exp:gh-full-vanilla-v0",
         "Does IDF over full body text beat vanilla, and does the extra (de-truncated) "
         "text move the bag-of-tokens bar?"),
        ("relation-metric", "exp:gh-full-relation-v0", "exp:gh-full-idf-v0",
         "Does the relation-supervised diagonal metric beat IDF on full text?"),
        ("embedder-cosine", "exp:gh-full-embed-cosine-v0", "exp:gh-full-idf-v0",
         f"Frozen {model} @ max_length={max_len} on full body text, raw cosine "
         "(no training) — how much does using more of the body lift cross-repo R@1?"),
    ]
    out = []
    for sys_name, exp_id, compare, hyp in specs:
        m = systems[sys_name]
        out.append({
            "card_type": "experiment", "id": exp_id,
            "name": f"GitHub full-text cross-repo de-referenced — {sys_name}",
            "created_at": CREATED_AT, "hypothesis": hyp,
            "task": "issue_to_fixing_pr", "dataset_version": DATASET_ID,
            "code_version": "relsdlc-0.1.0", "seed": 0,
            "command": "python data/full/run_full_ablation.py",
            "system": sys_name, "runtime_class": "cpu",
            "metrics": {
                "recall_at_k": {k: round(v, 4) for k, v in m["recall_at_k"].items()},
                "mrr": round(m["mrr"], 4),
                "hard_negative_accuracy": round(m["hard_negative_accuracy"], 4),
                "n_queries": m["n_queries"],
            },
            "baseline_comparison": compare, "error_slices": [],
            "leakage_checks": ["references scrubbed from text; train repos disjoint "
                               "from test repos (cross-repo generalization); "
                               "embeddings frozen (no fit on test repos)."],
            "known_limitations": [
                "Pilot scale; full body text (up to 8000 chars) but frozen general-text "
                "embedder (not code-specific). Exploratory.",
            ],
            "exploratory": True,
        })
    return out


def _fmt_row(name: str, m: dict) -> str:
    r = m["recall_at_k"]
    return (f"{name:<22}{r['1']:>8.3f}{r['5']:>8.3f}{r['10']:>8.3f}"
            f"{m['mrr']:>8.3f}{m['hard_negative_accuracy']:>12.3f}")


def _systems_for(ds, vecs) -> dict:
    """The four comparable systems on one (dataset, embedding-cache) pair."""
    base = run_ablation(ds, seed=0, min_df=MIN_DF)["systems"]
    return {
        "vanilla-tf-cosine": base["vanilla-tf-cosine"],
        "idf-cosine": base["idf-cosine"],
        "relation-metric": base["relation-metric"],
        "embedder-cosine": run_cosine_on_vecs(ds, vecs),
    }


def _lift(full: dict, ref: dict) -> dict:
    """Per-system (full − ref) deltas on R@k and MRR, where ref has the system."""
    out = {}
    for name in SYSTEMS:
        if name in ref:
            out[name] = {
                k: round(full[name]["recall_at_k"][k] - ref[name]["recall_at_k"][k], 4)
                for k in full[name]["recall_at_k"]
            }
            out[name]["mrr"] = round(full[name]["mrr"] - ref[name]["mrr"], 4)
    return out


def main() -> None:
    if not EMB.exists():
        print(f"ERROR: missing {EMB}. Run: python data/full/embed_full.py "
              "(needs pip install -e '.[embed]')", file=sys.stderr)
        raise SystemExit(2)

    # Full text (8000 chars) + wide window (512).
    ds, meta = load_full_crossrepo()
    vecs, dim, model, max_len = load_embeddings(EMB)
    full = _systems_for(ds, vecs)
    n_test = run_ablation(ds, seed=0, min_df=MIN_DF)["n_test_queries"]

    # Paired control: SAME records/split, pilot regime (500-char bodies, window 256).
    paired = {}
    paired_max_len = None
    if EMB_TRUNC.exists():
        ds_t, _ = load_full_crossrepo(trunc_chars=TRUNC_CHARS)
        vecs_t, _, _, paired_max_len = load_embeddings(EMB_TRUNC)
        paired = _systems_for(ds_t, vecs_t)

    # Cross-snapshot reference: the four systems on the frozen truncated pilot.
    xsnap = load_truncated_baseline()

    for card in build_cards(full, model, max_len):
        (CARDS / f"{card['id'].split(':', 1)[1]}.experiment-card.json").write_text(
            json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lift_paired = _lift(full, paired) if paired else {}
    lift_xsnap = _lift(full, xsnap) if xsnap else {}

    (HERE / "full-results.json").write_text(
        json.dumps({
            "model": model, "dim": dim, "max_length": max_len, "body_cap": 8000,
            "meta": meta, "n_test_queries": n_test,
            "systems": full,
            "paired_truncated_control": {
                "note": "SAME records + split as `systems`; bodies char-capped to "
                        f"{TRUNC_CHARS} and embedder at max_length={paired_max_len}. "
                        "Truncation is the only variable, so this lift is causal.",
                "max_length": paired_max_len, "body_cap": TRUNC_CHARS,
                "systems": paired,
            },
            "cross_snapshot_truncated_pilot": {
                "note": "The four systems on the FROZEN truncated pilot "
                        "(data/pilot/embed-results.json). Suggestive but conflates "
                        "truncation with snapshot drift (different fetch + record set).",
                "systems": xsnap,
            },
            "lift_paired_full_minus_truncated": lift_paired,
            "lift_crosssnapshot_full_minus_truncatedpilot": lift_xsnap,
        }, indent=2) + "\n", encoding="utf-8")

    print(f"FULL-TEXT DE-REFERENCED CROSS-REPO — issue_to_fixing_pr  "
          f"(model: {model}, max_length={max_len}, body cap=8000)")
    print(f"train repos: {len(meta['train_repos'])}  test repos: {len(meta['test_repos'])}  "
          f"(n_test queries: {n_test})")
    print()
    print(f"{'FULL TEXT (8000 / 512)':<22}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'HardNegAcc':>12}")
    for name in SYSTEMS:
        print(_fmt_row(name, full[name]))

    if paired:
        print()
        print("PAIRED CONTROL — same records/split, truncation the only variable")
        print(f"{f'TRUNC (500 / {paired_max_len}) same recs':<22}"
              f"{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'HardNegAcc':>12}")
        for name in SYSTEMS:
            print(_fmt_row(name, paired[name]))
        print()
        print(f"{'LIFT (paired)':<22}{'dR@1':>8}{'dR@5':>8}{'dR@10':>8}{'dMRR':>8}")
        for name in SYSTEMS:
            lk = lift_paired[name]
            print(f"{name:<22}{lk['1']:>+8.3f}{lk['5']:>+8.3f}{lk['10']:>+8.3f}{lk['mrr']:>+8.3f}")

    if xsnap:
        print()
        print("CROSS-SNAPSHOT — vs the frozen truncated pilot (conflates snapshot drift)")
        print(f"{'TRUNC PILOT (500 / 256)':<22}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'HardNegAcc':>12}")
        for name in SYSTEMS:
            if name in xsnap:
                print(_fmt_row(name, xsnap[name]))
        print()
        print(f"{'LIFT (cross-snapshot)':<22}{'dR@1':>8}{'dR@5':>8}{'dR@10':>8}{'dMRR':>8}")
        for name in SYSTEMS:
            if name in lift_xsnap:
                lk = lift_xsnap[name]
                print(f"{name:<22}{lk['1']:>+8.3f}{lk['5']:>+8.3f}{lk['10']:>+8.3f}{lk['mrr']:>+8.3f}")


if __name__ == "__main__":
    main()
