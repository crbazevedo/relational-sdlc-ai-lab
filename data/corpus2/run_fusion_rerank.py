#!/usr/bin/env python3
"""R25c — can a LEARNED reranker salvage co-change structure as a COMPLEMENTARY signal?

R25/R25b refuted "structure beats text": BM25 over test PATH tokens beats co-change
structure on both corpora, and naive equal-weight fusion HURTS (corpus2 0.497 < BM25
0.609). The honest salvage question is narrower: combined by a *learned* ranker, does
structure add anything OVER lexical features alone?

The decisive control is two learned rankers on the SAME cross-repo (leave-one-repo-out)
protocol:
  - LR(lexical)        : features = [bm25, path-overlap]
  - LR(lexical+struct) : features = [bm25, path-overlap, struct-cosine, struct-support,
                         covered]
If LR(lexical+struct) > LR(lexical) by a bootstrap-CI margin, structure is genuinely
complementary (the gain cannot come from better lexical weighting, since the lexical
features are identical). If they tie, structure adds nothing and we say so.

Also reports the FALLBACK regime: among queries where BM25's top-1 is wrong, how many
does adding structure recover? Reuses the corpus2 same-repo-negative pool (SEED=0,
as_of, gold edge removed). numpy only; deterministic.

Run: PYTHONPATH=src python data/corpus2/run_fusion_rerank.py
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
from relsdlc.scrub import scrub_record_text  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_REPOS = {"vuejs/core", "vitejs/vite", "sveltejs/svelte", "withastro/astro",
               "expressjs/express", "prettier/prettier"}
N_NEG, SEED, B = 15, 0, 10000
OUT = HERE / "corpus2-fusion-results.json"
_TOKEN = re.compile(r"[a-z0-9]+")


def _jl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def _repo(i):
    return i.split(":")[1]


def _tok(t):
    return _TOKEN.findall(re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(t)).lower())


def _path(i):
    return i.split(":")[-1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.len = {i: len(t) for i, t in docs.items()}
        self.avgdl = (sum(self.len.values()) / len(self.len)) or 1.0
        self.tf = {i: _ctr(t) for i, t in docs.items()}
        df = {}
        for t in docs.values():
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def score(self, q, did):
        tf, dl = self.tf[did], self.len[did]
        return sum(self.idf.get(w, 0.0) * tf[w] * (self.k1 + 1) /
                   (tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                   for w in q if w in tf)


def _ctr(toks):
    c = {}
    for w in toks:
        c[w] = c.get(w, 0) + 1
    return c


def fit_lr(X, y, l2=1.0, iters=1500, lr=0.3):
    """Class-balanced L2 logistic regression (numpy GD). Returns weight vec incl. bias."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    pos, neg = max(y.sum(), 1), max(len(y) - y.sum(), 1)
    sw = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        reg = l2 * np.r_[w[:-1], 0.0] / len(y)
        w -= lr * (Xb.T @ (sw * (p - y)) / len(y) + reg)
    return w


def _score(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def main():
    recs = _jl(HERE / "records.jsonl")
    edges = [e for e in _jl(HERE / "modifies_edges.jsonl") if e.get("relation") == "modifies"]
    prs = [r for r in recs if r["type"] == "pull_request"]
    vf = {r["id"]: r.get("valid_from", "") for r in recs}
    pr_tok = {r["id"]: _tok(scrub_record_text(r) or r["id"]) for r in prs}

    test_mods, repo_tests = {}, {}
    for e in edges:
        test_mods.setdefault(e["target"], []).append((e["source"], e.get("valid_from", "")))
        repo_tests.setdefault(_repo(e["target"]), set()).add(e["target"])
    q_rel = {}
    for e in edges:
        if _repo(e["source"]) in DENSE_REPOS:
            q_rel.setdefault(e["source"], set()).add(e["target"])

    # global BM25 over all test paths (stable idf)
    all_tests = {t for ts in repo_tests.values() for t in ts}
    bm = BM25({t: _tok(_path(t)) for t in all_tests})

    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    ids = [r["id"] for r in prs]
    texts = [scrub_record_text(r) or r["id"] for r in prs]
    vecs = {}
    for i in range(0, len(ids), 64):
        enc = tok(texts[i:i+64], padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), p=2, dim=1).numpy()
        for k, j in enumerate(ids[i:i+64]):
            vecs[j] = e[k]
    print(f"embedded {len(vecs)} PRs", file=sys.stderr, flush=True)

    rng = np.random.default_rng(SEED)

    def struct(test, tq, qpr):
        mods = [pr for pr, t in test_mods.get(test, []) if pr != qpr and pr in vecs and t and t <= tq]
        if not mods:
            return None, 0
        v = np.mean([vecs[pr] for pr in mods], axis=0)
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v), len(mods)

    # build per-(query,candidate) feature rows
    # features: bm25, overlap, struct_cos, struct_support(log), covered
    rows = []   # dict per candidate
    qmeta = []  # per query: (qid, repo, cand_index_range)
    for qi, (qpr, rels) in enumerate(sorted(q_rel.items())):
        repo = _repo(qpr)
        cat = list(repo_tests.get(repo, set()) - rels)
        if len(cat) < 3:
            continue
        tq = vf.get(qpr, "")
        a = vecs.get(qpr)
        qt = pr_tok.get(qpr, [])
        qset = set(qt)
        negs = list(rng.choice(cat, size=min(N_NEG, len(cat)), replace=False))
        cands = list(rels) + negs
        start = len(rows)
        for c in cands:
            sv, sup = struct(c, tq, qpr)
            scos = float(a @ sv) if (a is not None and sv is not None) else 0.0
            rows.append({
                "qid": qpr, "repo": repo, "cand": c, "y": 1 if c in rels else 0,
                "f": [bm.score(qt, c), len(qset & set(_tok(_path(c)))),
                      scos, math.log1p(sup), 1.0 if sup > 0 else 0.0],
            })
        qmeta.append((qpr, repo, (start, len(rows))))

    F = np.array([r["f"] for r in rows], dtype=float)
    Y = np.array([r["y"] for r in rows], dtype=float)
    repos_of_row = np.array([r["repo"] for r in rows])
    LEX = [0, 1]                 # bm25, overlap
    LEXSTRUCT = [0, 1, 2, 3, 4]  # + struct_cos, support, covered

    def loro_scores(cols):
        """Leave-one-repo-out CV; return per-row predicted score (np array)."""
        pred = np.zeros(len(rows))
        for test_repo in sorted(set(repos_of_row)):
            tr = repos_of_row != test_repo
            te = repos_of_row == test_repo
            mu, sd = F[tr][:, cols].mean(0), F[tr][:, cols].std(0)
            sd = np.where(sd > 0, sd, 1.0)
            Xtr = (F[tr][:, cols] - mu) / sd
            Xte = (F[te][:, cols] - mu) / sd
            w = fit_lr(Xtr, Y[tr])
            pred[te] = _score(Xte, w)
        return pred

    pred_lex = loro_scores(LEX)
    pred_ls = loro_scores(LEXSTRUCT)

    # per-query top-1 hit for each system
    def hits(scorer):
        h = []
        for (_, _, (s, e)) in qmeta:
            seg = rows[s:e]
            sc = scorer(s, e, seg)
            top = max(range(len(seg)), key=lambda k: (sc[k], seg[k]["cand"]))
            h.append(seg[top]["y"])
        return np.array(h, dtype=float)

    h_bm25 = hits(lambda s, e, seg: [r["f"][0] for r in seg])
    h_struct = hits(lambda s, e, seg: [r["f"][2] if r["f"][4] > 0 else -1 for r in seg])
    h_lex = hits(lambda s, e, seg: pred_lex[s:e])
    h_ls = hits(lambda s, e, seg: pred_ls[s:e])

    def r1(h):
        return float(h.mean())

    def ci_diff(ha, hb):
        d = ha - hb
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
        return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    # fallback regime: where BM25 top-1 is wrong, does +struct recover?
    bm25_wrong = h_bm25 == 0
    recov_ls = float(h_ls[bm25_wrong].mean()) if bm25_wrong.any() else 0.0
    recov_struct = float(h_struct[bm25_wrong].mean()) if bm25_wrong.any() else 0.0

    res = {
        "corpus": "corpus2 (TS/JS), same-repo negs, leave-one-repo-out CV", "n_queries": len(h_bm25),
        "R@1": {"bm25": round(r1(h_bm25), 4), "struct": round(r1(h_struct), 4),
                "LR_lexical": round(r1(h_lex), 4), "LR_lexical+struct": round(r1(h_ls), 4)},
        "structure_contribution": {
            "delta_R@1_lexstruct_minus_lex": round(r1(h_ls) - r1(h_lex), 4),
            "ci95": ci_diff(h_ls, h_lex),
            "delta_R@1_lexstruct_minus_bm25": round(r1(h_ls) - r1(h_bm25), 4),
            "ci95_vs_bm25": ci_diff(h_ls, h_bm25)},
        "fallback_regime": {
            "n_bm25_wrong": int(bm25_wrong.sum()),
            "recovery_LR_lexstruct": round(recov_ls, 4),
            "recovery_struct_alone": round(recov_struct, 4)},
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    R = res["R@1"]; S = res["structure_contribution"]
    print("R25c — learned reranker: does structure add OVER lexical? (corpus2, LORO CV)")
    print(f"  n={res['n_queries']} queries")
    print(f"  BM25 alone          R@1 {R['bm25']:.3f}")
    print(f"  struct alone        R@1 {R['struct']:.3f}")
    print(f"  LR(lexical)         R@1 {R['LR_lexical']:.3f}")
    print(f"  LR(lexical+struct)  R@1 {R['LR_lexical+struct']:.3f}")
    print(f"  >> structure contribution (lexstruct - lex): {S['delta_R@1_lexstruct_minus_lex']:+.3f}  "
          f"CI95 {S['ci95']}  -> {'COMPLEMENTARY' if S['ci95'][0] > 0 else 'NOT significant'}")
    print(f"  >> vs BM25 alone: {S['delta_R@1_lexstruct_minus_bm25']:+.3f}  CI95 {S['ci95_vs_bm25']}")
    fb = res["fallback_regime"]
    print(f"  fallback (BM25 top-1 wrong, n={fb['n_bm25_wrong']}): "
          f"LR+struct recovers {fb['recovery_LR_lexstruct']:.3f}, struct-alone {fb['recovery_struct_alone']:.3f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
