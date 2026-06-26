#!/usr/bin/env python3
"""R27 — a static path-proximity test-selection baseline for diff->affected-test.

The SE reviewer's question: does a classic *static* test-selection heuristic — pick the
test whose path is closest to the files the PR changed — also beat co-change structure?
(Ekstazi/STARTS are JVM dynamic RTS and do not run on our Python/TS corpora; the
language-agnostic, deployable static analogue is changed-source-path -> test-path
proximity, which uses only the diff available at query time, no history.)

For each diff->test query PR we take its changed SOURCE files (the non-test files it
modifies, from the pilot `modifies` graph) and rank candidate tests by core-token overlap
between each candidate's path and the changed source paths (basename/module match, e.g.
`pydantic/_internal/_generics.py` -> `tests/test_generics.py`). Scored on the SAME 112-query
pilot harness as embedder-cosine (0.009), BM25-over-paths (0.536), and co-change structure
(0.429, release-honest), so the four are directly comparable.

Run: PYTHONPATH=src python data/pilot/run_path_proximity.py
Numpy-free, deterministic; reuses the committed harness + graph.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from relsdlc.ingest import _is_test_path  # noqa: E402
from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402
from run_gnn_ablation import load_diff2test_crossrepo  # noqa: E402
from run_baselines_metrics import BM25, build_text_provider, _path_of  # noqa: E402

OUT = HERE / "path-proximity-results.json"
STOP = {"test", "tests", "py", "src", "lib", "init", "main", "index", "spec", "mod",
        "the", "a", "of", "to"}


def _core(path: str) -> set[str]:
    """Discriminative path tokens: split on separators + camelCase, drop stopwords."""
    out = set()
    for part in re.split(r"[/._\-]", path):
        part = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
        for w in re.findall(r"[a-z0-9]+", part.lower()):
            if w not in STOP and len(w) > 1:
                out.add(w)
    return out


def _suite(rows):
    n = len(rows)
    return {"n": n,
            "R@1": round(sum(r[0] == 1 for r in rows) / n, 4),
            "R@5": round(sum(r[0] is not None and r[0] <= 5 for r in rows) / n, 4),
            "R@10": round(sum(r[0] is not None and r[0] <= 10 for r in rows) / n, 4),
            "MRR": round(sum((1.0 / r[0]) if r[0] else 0.0 for r in rows) / n, 4)}


def _rank_rows(queries, score_fn):
    rows = []
    for q in queries:
        rel = set(q.relevant)
        scored = sorted(((c, score_fn(q, c)) for c in q.candidates),
                        key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))
        rank = next((i + 1 for i, (c, _) in enumerate(scored) if c in rel), None)
        rows.append((rank,))
    return rows


def main():
    issue_ds, meta = load_pilot_crossrepo()
    diff_ds = load_diff2test_crossrepo(set(meta["train_repos"]))
    queries = [q for q in diff_ds.queries if q.split == "test"]

    records = [json.loads(l) for l in (HERE / "records.jsonl").read_text().split("\n") if l.strip()]
    toks = build_text_provider(records)

    # PR -> changed SOURCE files (non-test) from the pilot modifies graph
    pr_src = {}
    for e in (json.loads(l) for l in (HERE / "graph" / "modifies_edges.jsonl").read_text().split("\n") if l.strip()):
        if e.get("relation") == "modifies":
            p = e["target"].split(":")[-1]
            if not _is_test_path(p):
                pr_src.setdefault(e["source"], []).append(p)
    src_core = {pr: [_core(p) for p in ps] for pr, ps in pr_src.items()}

    covered = sum(1 for q in queries if src_core.get(q.query_record))

    # --- path-proximity: candidate test path vs changed source paths -------------
    def prox(q, cand):
        sets = src_core.get(q.query_record, [])
        if not sets:
            return (0, 0.0)
        ct = _core(_path_of(cand))
        if not ct:
            return (0, 0.0)
        best_n, best_j = 0, 0.0
        for s in sets:
            inter = len(ct & s)
            if inter:
                j = inter / len(ct | s)
                if (inter, j) > (best_n, best_j):
                    best_n, best_j = inter, j
        return (best_n, best_j)

    # --- BM25 over paths (anchor: should reproduce 0.536) ------------------------
    cand_ids = {c for q in queries for c in q.candidates}
    bm = BM25({c: toks(c) for c in cand_ids})

    def bm25(q, cand):
        return (bm.score(toks(q.query_record), cand), 0.0)

    # --- best static system: path-proximity, BM25 as tie-break ------------------
    def static_combo(q, cand):
        p = prox(q, cand)
        return (p[0], p[1] + 0.001 * bm.score(toks(q.query_record), cand))

    res = {
        "harness": "pilot diff->affected-test, test split (same 112-query pool as the baselines)",
        "n_queries": len(queries),
        "source_coverage": round(covered / len(queries), 4),
        "path_proximity": _suite(_rank_rows(queries, prox)),
        "bm25_over_paths_anchor": _suite(_rank_rows(queries, bm25)),
        "best_static_proximity+bm25": _suite(_rank_rows(queries, static_combo)),
        "reference": {"embedder_cosine_R@1": 0.009,
                      "co_change_structure_R@1_release": 0.429,
                      "co_change_structure_R@1_no_asof": 0.554},
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    pp, bm_a, cb = res["path_proximity"], res["bm25_over_paths_anchor"], res["best_static_proximity+bm25"]
    print("R27 — static path-proximity test selection vs the diff->test field")
    print(f"  harness: {len(queries)} queries; {res['source_coverage']:.0%} have >=1 changed source file")
    print(f"  {'system':<34}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR':>7}")
    print(f"  {'embedder-cosine (cited)':<34}{0.009:>7.3f}{'—':>7}{'—':>7}{'—':>7}")
    print(f"  {'co-change structure +as_of (cited)':<34}{0.429:>7.3f}{'—':>7}{'—':>7}{0.574:>7.3f}")
    print(f"  {'BM25 over paths (anchor)':<34}{bm_a['R@1']:>7.3f}{bm_a['R@5']:>7.3f}{bm_a['R@10']:>7.3f}{bm_a['MRR']:>7.3f}")
    print(f"  {'PATH-PROXIMITY (static, this wave)':<34}{pp['R@1']:>7.3f}{pp['R@5']:>7.3f}{pp['R@10']:>7.3f}{pp['MRR']:>7.3f}")
    print(f"  {'best static (proximity+BM25)':<34}{cb['R@1']:>7.3f}{cb['R@5']:>7.3f}{cb['R@10']:>7.3f}{cb['MRR']:>7.3f}")
    verdict = ("structure LOSES to a static path heuristic too" if pp["R@1"] >= 0.429
               else "structure beats raw path-proximity but loses to BM25/best-static"
               if cb["R@1"] >= 0.429 else "structure competitive with static heuristics")
    print(f"  => {verdict}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
