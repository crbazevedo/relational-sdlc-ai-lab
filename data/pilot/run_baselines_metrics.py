#!/usr/bin/env python3
"""R25 — baselines + full metric suite + CIs + MDE + Task-B text-free construct check.

Closes the paper's [PENDING] empirical items in one numpy-only, deterministic pass:

  - §5 BM25 baseline (both tasks) and the "Occam" single-bi-encoder baseline (= the
    LoRA embedder, here labeled as such);
  - §4 Task-B text-free construct check: a strong lexical baseline (BM25 over the test
    PATH tokens, and a path/identifier-overlap heuristic) floors near chance on the
    text-free test candidates, while typed co-change structure wins — establishing the
    "text-free" premise empirically rather than by assertion;
  - §3 the full metric suite (R@1/5/10, MRR, nDCG@10, Hits@10) with bootstrap CIs
    (query-level AND repo-cluster — the honest unit for a cross-repo claim) and the
    candidate-pool sizes;
  - §3 the minimum-detectable-effect / power analysis for the LoRA Task-A win.

It reuses the EXACT de-referenced cross-repo loaders + ranking of run_gnn_ablation /
run_bootstrap_ci, and REPRODUCES the committed anchors (Task-A frozen 0.592, LoRA 0.655,
+graph 0.690; Task-B embedder-cosine 0.009) before reporting anything new — so the new
baselines/CIs annotate numbers the audit already trusts.

Run:  PYTHONPATH=src python data/pilot/run_baselines_metrics.py
Numpy only; no network, no torch. Deterministic (seed 0).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from relsdlc.scrub import scrub_record_text  # noqa: E402
from run_crossrepo_ablation import load_pilot_crossrepo, _repo_of  # noqa: E402
from run_gnn_ablation import (  # noqa: E402
    load_diff2test_crossrepo, load_embeddings, load_graph_edges,
    _candidate_ids, _with_zero_fallback, gold_fixes_pairs, gold_modifies_pairs,
)
from relsdlc.graphsage import augmented_vecs  # noqa: E402

B = 10000
SEED = 0
KS = (1, 5, 10)
ALPHA, HOPS = 0.5, 2
OUT = HERE / "baselines-metrics-results.json"
_TOKEN = re.compile(r"[a-z0-9]+")


# --- text + tokenization -----------------------------------------------------

def _tok(text: str) -> list[str]:
    # lowercase; split snake/camel/path separators into word tokens
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    return _TOKEN.findall(s.lower())


def _path_of(node_id: str) -> str:
    # "gh:owner/repo:file:tests/test_request.py" -> "tests/test_request.py"
    return node_id.split(":")[-1]


def build_text_provider(records: list[dict]):
    """id -> token list. Records (issue/PR) use scrubbed content; nodes absent from
    records (test-file nodes) fall back to their PATH tokens — the only text a lexical
    model could ever see for a text-free candidate."""
    rtext = {r["id"]: _tok(scrub_record_text(r) or "") for r in records}

    def toks(node_id: str) -> list[str]:
        if node_id in rtext:
            return rtext[node_id]
        return _tok(_path_of(node_id))      # test-file node: path tokens only
    return toks


# --- Okapi BM25 (hand-rolled, k1=1.5 b=0.75) ---------------------------------

class BM25:
    def __init__(self, docs: dict[str, list[str]], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.len = {i: len(t) for i, t in docs.items()}
        self.avgdl = (sum(self.len.values()) / len(self.len)) if self.len else 0.0
        self.tf = {i: _counter(t) for i, t in docs.items()}
        df: dict[str, int] = {}
        for t in docs.values():
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def score(self, query: list[str], doc_id: str) -> float:
        tf, dl = self.tf[doc_id], self.len[doc_id]
        s = 0.0
        for w in query:
            if w not in tf:
                continue
            f = tf[w]
            s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0)))
        return s


def _counter(toks):
    c: dict[str, int] = {}
    for w in toks:
        c[w] = c.get(w, 0) + 1
    return c


# --- per-query ranking -> rows (the full metric primitives) ------------------

def _rows(dataset, rank_fn):
    """rank_fn(q) -> ranked candidate-id list (best first). Returns one row per TEST
    query: rank of first relevant + hit@{1,5,10}, rr, ndcg@10, hits@10, hns, repo."""
    rows = []
    for q in dataset.queries:
        if q.split != "test":
            continue
        ranked = rank_fn(q)
        rel = set(q.relevant)
        r = next((i + 1 for i, c in enumerate(ranked) if c in rel), None)
        # nDCG@10 (general; IDCG for |rel| ideal positions)
        dcg = sum(1.0 / math.log2(i + 2) for i, c in enumerate(ranked[:10]) if c in rel)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), 10)))
        hns = None
        if q.hard_negatives and rel:
            hn = set(q.hard_negatives)
            bp = next((i for i, c in enumerate(ranked) if c in rel), None)
            bn = next((i for i, c in enumerate(ranked) if c in hn), None)
            hns = 0 if bp is None else (1 if bn is None else int(bp < bn))
        rows.append({
            "query_id": q.query_id, "repo": _repo_of(q.query_record),
            "hit1": float(r == 1), "hit5": float(r is not None and r <= 5),
            "hit10": float(r is not None and r <= 10),
            "rr": (1.0 / r) if r else 0.0,
            "ndcg10": (dcg / idcg) if idcg else 0.0,
            "hits10": float(r is not None and r <= 10), "hns": hns,
            "pool": len(q.candidates),
        })
    return rows


def _cosine_ranker(dataset, base_vecs, dim):
    vecs = _with_zero_fallback(base_vecs, _candidate_ids(dataset), dim)
    unit = {cid: _unit(vecs[cid]) for cid in _candidate_ids(dataset)}

    def rank(q):
        a = unit[q.query_record]
        sc = sorted(((c, float(a @ unit[c])) for c in q.candidates), key=lambda kv: (-kv[1], kv[0]))
        return [c for c, _ in sc]
    return rank


def _bm25_ranker(dataset, toks):
    docs = {c: toks(c) for c in _candidate_ids(dataset)}
    bm = BM25(docs)

    def rank(q):
        qt = toks(q.query_record)
        sc = sorted(((c, bm.score(qt, c)) for c in q.candidates), key=lambda kv: (-kv[1], kv[0]))
        return [c for c, _ in sc]
    return rank


def _overlap_ranker(dataset, toks):
    """Lexical path/identifier overlap: |query tokens ∩ candidate tokens| (IDF-free).
    The naive 'does the diff text mention the test path' heuristic a skeptic proposes."""
    def rank(q):
        qs = set(toks(q.query_record))
        sc = sorted(((c, len(qs & set(toks(c)))) for c in q.candidates), key=lambda kv: (-kv[1], kv[0]))
        return [c for c, _ in sc]
    return rank


def _graphaug_ranker(dataset, base_vecs, dim, fixes_e, mods_e, *, excl_fixes=None, excl_mods=None):
    cand = _candidate_ids(dataset)
    aug = augmented_vecs(base_vecs, fixes_e, mods_e, alpha=ALPHA, hops=HOPS,
                         exclude_fixes_pairs=excl_fixes, exclude_modifies_pairs=excl_mods,
                         all_node_ids=set(base_vecs) | cand)
    return _cosine_ranker(dataset, _with_zero_fallback(aug, cand, dim), dim)


# --- aggregation, CIs, MDE ---------------------------------------------------

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _agg(rows, k):
    return float(np.mean([r[k] for r in rows]))


def _hns(rows):
    v = [r["hns"] for r in rows if r["hns"] is not None]
    return float(np.mean(v)) if v else None


def _suite(rows):
    return {"n": len(rows), "pool_mean": round(float(np.mean([r["pool"] for r in rows])), 1),
            "R@1": round(_agg(rows, "hit1"), 4), "R@5": round(_agg(rows, "hit5"), 4),
            "R@10": round(_agg(rows, "hit10"), 4), "MRR": round(_agg(rows, "rr"), 4),
            "nDCG@10": round(_agg(rows, "ndcg10"), 4), "Hits@10": round(_agg(rows, "hits10"), 4),
            "HardNegAcc": (round(_hns(rows), 4) if _hns(rows) is not None else None)}


def _ci_query(rows, key, rng):
    arr = np.array([r[key] for r in rows]); n = len(arr)
    bs = [arr[rng.integers(0, n, n)].mean() for _ in range(B)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def _ci_repo(rows, key, rng):
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["repo"], []).append(r[key])
    repos = list(by); means = {k: np.mean(v) for k, v in by.items()}
    counts = {k: len(v) for k, v in by.items()}
    bs = []
    for _ in range(B):
        pick = [repos[i] for i in rng.integers(0, len(repos), len(repos))]
        num = sum(means[p] * counts[p] for p in pick); den = sum(counts[p] for p in pick)
        bs.append(num / den)
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _mde(frozen_rows, lora_rows):
    """Paired hit@1 MDE/power (two-sided alpha=0.05, target power 0.8)."""
    assert [r["query_id"] for r in frozen_rows] == [r["query_id"] for r in lora_rows]
    d = np.array([l["hit1"] - f["hit1"] for f, l in zip(frozen_rows, lora_rows)])
    n = len(d); sd = float(np.std(d, ddof=1)); delta = float(d.mean())
    za, zb = 1.959963985, 0.841621234
    mde = (za + zb) * sd / math.sqrt(n) if sd > 0 else 0.0
    power = _phi(abs(delta) * math.sqrt(n) / sd - za) if sd > 0 else 1.0
    return {"n": n, "observed_delta_R@1": round(delta, 4), "sd_paired": round(sd, 4),
            "mde_R@1_at_80pct_power": round(mde, 4), "achieved_power": round(power, 3),
            "gains": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "detectable": bool(abs(delta) >= mde)}


# --- main --------------------------------------------------------------------

def main():
    records = [json.loads(l) for l in (HERE / "records.jsonl").read_text().split("\n") if l.strip()]
    toks = build_text_provider(records)
    issue_ds, meta = load_pilot_crossrepo()
    diff_ds = load_diff2test_crossrepo(set(meta["train_repos"]))
    frozen = load_embeddings("minilm-l6-v2"); lora = load_embeddings("minilm-lora")
    dim = len(next(iter(frozen.values())))
    fixes_e, mods_e = load_graph_edges()
    rng = np.random.default_rng(SEED)

    # ---- Task A: issue -> fixing PR (text-rich) -------------------------
    A = {}
    A["BM25"] = _rows(issue_ds, _bm25_ranker(issue_ds, toks))
    A["embedder-cosine (frozen)"] = _rows(issue_ds, _cosine_ranker(issue_ds, frozen, dim))
    A["bi-encoder LoRA (Occam)"] = _rows(issue_ds, _cosine_ranker(issue_ds, lora, dim))
    A["+ graph-aug (LoRA)"] = _rows(issue_ds, _graphaug_ranker(
        issue_ds, lora, dim, fixes_e, mods_e, excl_fixes=gold_fixes_pairs(issue_ds)))

    # ---- Task B: diff -> affected test (text-free) ----------------------
    Bt = {}
    Bt["BM25 (paths)"] = _rows(diff_ds, _bm25_ranker(diff_ds, toks))
    Bt["path-overlap"] = _rows(diff_ds, _overlap_ranker(diff_ds, toks))
    Bt["embedder-cosine (frozen)"] = _rows(diff_ds, _cosine_ranker(diff_ds, frozen, dim))
    Bt["graph-aug (structure)"] = _rows(diff_ds, _graphaug_ranker(
        diff_ds, frozen, dim, fixes_e, mods_e, excl_mods=gold_modifies_pairs(diff_ds)))

    # ---- anchor reproduction (audit gate) -------------------------------
    anchors = {
        "A/frozen/R@1": (_agg(A["embedder-cosine (frozen)"], "hit1"), 0.592),
        "A/LoRA/R@1": (_agg(A["bi-encoder LoRA (Occam)"], "hit1"), 0.655),
        "A/+graph/R@1": (_agg(A["+ graph-aug (LoRA)"], "hit1"), 0.690),
        "B/cosine/R@1": (_agg(Bt["embedder-cosine (frozen)"], "hit1"), 0.009),
    }
    bad = {k: (round(got, 4), exp) for k, (got, exp) in anchors.items() if abs(got - exp) > 0.02}
    assert not bad, f"anchor mismatch (harness drift): {bad}"

    # ---- CIs on the headline R@1 + MDE ----------------------------------
    ci = {
        "A/bi-encoder LoRA/R@1/query": _ci_query(A["bi-encoder LoRA (Occam)"], "hit1", rng),
        "A/bi-encoder LoRA/R@1/repo": _ci_repo(A["bi-encoder LoRA (Occam)"], "hit1", rng),
        "A/BM25/R@1/query": _ci_query(A["BM25"], "hit1", rng),
        "B/graph-aug/R@1/query": _ci_query(Bt["graph-aug (structure)"], "hit1", rng),
        "B/graph-aug/R@1/repo": _ci_repo(Bt["graph-aug (structure)"], "hit1", rng),
        "B/BM25/R@1/query": _ci_query(Bt["BM25 (paths)"], "hit1", rng),
    }
    mde = _mde(A["embedder-cosine (frozen)"], A["bi-encoder LoRA (Occam)"])

    res = {
        "meta": {"n_test_repos": len(meta["test_repos"]), "seed": SEED, "bootstrap": B},
        "task_A_issue2pr": {name: _suite(rows) for name, rows in A.items()},
        "task_B_diff2test": {name: _suite(rows) for name, rows in Bt.items()},
        "confidence_intervals_R@1": ci,
        "mde_power_lora_taskA": mde,
        "construct_check_note": (
            "Task B candidates are text-free test-file nodes (only a path). BM25 over path "
            "tokens and path-overlap both floor near chance; only co-change structure ranks."),
        "released_dense_asof_note": "Task-B release-honest dense+as_of number is R@1 0.429 (BENCHMARK.md / diff2test-strict-results.json).",
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    def table(title, d):
        print(f"\n== {title} ==")
        print(f"{'system':<28}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR':>7}{'nDCG':>7}{'Hits':>7}{'pool':>6}")
        for name, s in d.items():
            print(f"{name:<28}{s['R@1']:>7.3f}{s['R@5']:>7.3f}{s['R@10']:>7.3f}"
                  f"{s['MRR']:>7.3f}{s['nDCG@10']:>7.3f}{s['Hits@10']:>7.3f}{s['pool_mean']:>6.0f}")

    print("R25 — baselines + full metric suite (de-referenced cross-repo split)")
    print(f"anchors reproduced ✓  (test repos {len(meta['test_repos'])}, bootstrap B={B})")
    table(f"Task A — issue→fixing-PR (n={res['task_A_issue2pr']['BM25']['n']})", res["task_A_issue2pr"])
    table(f"Task B — diff→affected-test, TEXT-FREE (n={res['task_B_diff2test']['BM25 (paths)']['n']})", res["task_B_diff2test"])
    print("\nR@1 95% CIs:")
    for k, v in ci.items():
        print(f"  {k:<34} [{v[0]:.3f}, {v[1]:.3f}]")
    print(f"\nMDE/power (LoRA Task-A, paired hit@1, n={mde['n']}):")
    print(f"  observed ΔR@1 {mde['observed_delta_R@1']:+.3f} ({mde['gains']} gains / {mde['losses']} losses); "
          f"sd {mde['sd_paired']:.3f}")
    print(f"  MDE@80%power {mde['mde_R@1_at_80pct_power']:.3f}; achieved power {mde['achieved_power']:.2f}; "
          f"detectable={mde['detectable']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
