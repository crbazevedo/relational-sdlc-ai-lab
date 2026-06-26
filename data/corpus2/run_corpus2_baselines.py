#!/usr/bin/env python3
"""R25b — is co-change structure COMPLEMENTARY to a lexical path baseline on Task B?

R25 surfaced a refutation: on the pilot, BM25 over test PATH tokens reaches R@1 0.536
on the 'text-free' diff->test task — it does NOT floor near chance, because a PR's text
often lexically names the test it touches. So 'text-free -> only structure works' is
false as stated. This script asks the honest follow-up on the corpus2 (TS/JS) same-repo-
negative pool used by run_corpus2_diff2test.py (SEED=0, N_NEG=15, as_of):

  - BM25 (PR text -> test path tokens)         [the strong lexical baseline]
  - graph-aug + as_of (co-change structure)    [the released structure system]
  - late-fusion (z-normalised BM25 + structure)
  - COMPLEMENTARITY: queries structure gets right that BM25 misses, and vice versa.

If fusion > max(BM25, structure) and each catches queries the other misses, the honest
Task-B contribution is 'co-change structure is a complementary signal to lexical paths',
not 'structure beats text'. Reuses the corpus2 pool construction verbatim.

Run: PYTHONPATH=src python data/corpus2/run_corpus2_baselines.py
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
N_NEG, SEED = 15, 0
OUT = HERE / "corpus2-baselines-results.json"
_TOKEN = re.compile(r"[a-z0-9]+")


def _jl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def _repo(i):
    return i.split(":")[1]


def _tok(text):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    return _TOKEN.findall(s.lower())


def _path(node_id):
    return node_id.split(":")[-1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b, self.docs = k1, b, docs
        self.len = {i: len(t) for i, t in docs.items()}
        self.avgdl = (sum(self.len.values()) / len(self.len)) if self.len else 1.0
        self.tf = {i: _ctr(t) for i, t in docs.items()}
        df = {}
        for t in docs.values():
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def score(self, q, did):
        tf, dl = self.tf[did], self.len[did]
        s = 0.0
        for w in q:
            if w in tf:
                f = tf[w]
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0)))
        return s


def _ctr(toks):
    c = {}
    for w in toks:
        c[w] = c.get(w, 0) + 1
    return c


def _z(d):
    v = np.array(list(d.values()), dtype=float)
    mu, sd = v.mean(), v.std()
    return {k: ((x - mu) / sd if sd > 0 else 0.0) for k, x in d.items()}


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

    # embed PRs (CPU MiniLM) — structure features
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
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        e = torch.nn.functional.normalize(e, p=2, dim=1).numpy()
        for k, j in enumerate(ids[i:i+64]):
            vecs[j] = e[k]
    print(f"embedded {len(vecs)} PRs", file=sys.stderr, flush=True)

    rng = np.random.default_rng(SEED)

    def feat(test, tq, qpr):
        mods = [pr for pr, t in test_mods.get(test, []) if pr != qpr and pr in vecs and t and t <= tq]
        if not mods:
            return None
        v = np.mean([vecs[pr] for pr in mods], axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    n = 0
    hit = {"bm25": 0, "struct": 0, "fusion": 0}
    only_struct = only_bm25 = both = neither = 0
    for qpr, rels in q_rel.items():
        repo = _repo(qpr)
        cat = list(repo_tests.get(repo, set()) - rels)
        if len(cat) < 3:
            continue
        n += 1
        tq = vf.get(qpr, "")
        a = vecs.get(qpr)
        negs = list(rng.choice(cat, size=min(N_NEG, len(cat)), replace=False))
        cands = list(rels) + negs
        qt = pr_tok.get(qpr, [])
        docs = {c: _tok(_path(c)) for c in cands}
        bm = BM25(docs)
        bm_sc = {c: bm.score(qt, c) for c in cands}
        st_sc = {}
        for c in cands:
            f = feat(c, tq, qpr)
            st_sc[c] = float(a @ f) if (a is not None and f is not None) else -1.0
        fz = _z(bm_sc)
        sz = _z({c: (st_sc[c] if st_sc[c] > -1 else min(st_sc.values())) for c in cands})
        fu_sc = {c: fz[c] + sz[c] for c in cands}

        def top(score):
            return sorted(cands, key=lambda c: (-score[c], c))[0]
        bm_ok = top(bm_sc) in rels
        st_ok = top(st_sc) in rels
        fu_ok = top(fu_sc) in rels
        hit["bm25"] += bm_ok; hit["struct"] += st_ok; hit["fusion"] += fu_ok
        if st_ok and not bm_ok: only_struct += 1
        elif bm_ok and not st_ok: only_bm25 += 1
        elif bm_ok and st_ok: both += 1
        else: neither += 1

    res = {
        "corpus": "corpus2 (TS/JS), same-repo negatives", "n_queries": n,
        "R@1": {k: round(v / n, 4) for k, v in hit.items()},
        "complementarity": {"struct_only": only_struct, "bm25_only": only_bm25,
                            "both": both, "neither": neither},
        "pilot_note": "pilot Task B: BM25(paths) R@1 0.536 vs embedder-cosine 0.009",
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    r = res["R@1"]; c = res["complementarity"]
    print("R25b — lexical vs structure vs fusion on corpus2 (same-repo negatives)")
    print(f"  n={n} queries")
    print(f"  BM25(paths) R@1   {r['bm25']:.3f}")
    print(f"  structure   R@1   {r['struct']:.3f}")
    print(f"  late-fusion R@1   {r['fusion']:.3f}")
    print(f"  complementarity: struct-only {c['struct_only']}, bm25-only {c['bm25_only']}, "
          f"both {c['both']}, neither {c['neither']}")
    gain = r['fusion'] - max(r['bm25'], r['struct'])
    print(f"  fusion vs best single: {gain:+.3f}  -> "
          f"{'COMPLEMENTARY (structure adds over lexical)' if gain > 0.02 else 'NOT clearly complementary'}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
