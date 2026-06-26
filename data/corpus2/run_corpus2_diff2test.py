#!/usr/bin/env python3
"""R24b/c — diff->test on the SECOND corpus (TS/JS): does it replicate the pilot?

External-validity test. Builds a diff->affected-test benchmark inline from the
corpus2 modifies graph (gold (PR,test) pairs; candidate pool = relevant tests +
sampled same-repo test hard negatives; query = the PR), then runs — with the SAME
release-honest method as the pilot (pure-PR-embedding query + temporal as_of cut,
gold edge removed) — the structure diagnostic, the retrieval, the text-free baseline,
and the leakage audit. Compares to the pilot (diff->test R@1 0.43, fair 0.50=5.7x).

Eval on the densified repos only (where co-change is dense enough). MiniLM PR
embeddings on CPU (same recipe as the pilot cache). Run:
  PYTHONPATH=src python data/corpus2/run_corpus2_diff2test.py
"""
from __future__ import annotations

import json
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
OUT = HERE / "corpus2-diff2test-results.json"


def _jl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def _repo(i):
    return i.split(":")[1]


def main():
    recs = _jl(HERE / "records.jsonl")
    edges = [e for e in _jl(HERE / "modifies_edges.jsonl") if e.get("relation") == "modifies"]
    by = {r["id"]: r for r in recs}
    prs = [r for r in recs if r["type"] == "pull_request"]
    vf = {r["id"]: r.get("valid_from", "") for r in recs}

    # modifies graph: test -> [(pr, valid_from)]; and per-repo test catalogue
    test_mods, repo_tests = {}, {}
    for e in edges:
        test_mods.setdefault(e["target"], []).append((e["source"], e.get("valid_from", "")))
        repo_tests.setdefault(_repo(e["target"]), set()).add(e["target"])

    # gold queries: PR -> tests it modifies, in a DENSE repo
    q_rel = {}
    for e in edges:
        if _repo(e["source"]) in DENSE_REPOS:
            q_rel.setdefault(e["source"], set()).add(e["target"])

    # embed all PR text (CPU, scrubbed title+body) — same recipe as the pilot cache
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
        e = (h*m).sum(1)/m.sum(1).clamp(min=1e-9)
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
        return v/n if n > 0 else v

    n = struct_reach = h1 = mrr = emb_h1 = 0
    fair_h1 = fair_n = 0
    fair_pools, rand_cov = [], []
    gold_cov = neg_cov = neg_tot = 0
    pool_sizes = []
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
        pool_sizes.append(len(cands))
        feats = {c: feat(c, tq, qpr) for c in cands}
        covered = {c for c, f in feats.items() if f is not None}
        gold_is_cov = any(c in covered for c in rels)
        struct_reach += gold_is_cov; gold_cov += gold_is_cov
        neg_cov += sum(c in covered for c in negs); neg_tot += len(negs)
        # release-honest retrieval (pure-PR query + as_of); uncovered -> -1
        sc = sorted(((c, float(a@feats[c]) if (a is not None and feats[c] is not None) else -1.0) for c in cands),
                    key=lambda kv: (-kv[1], kv[0]))
        ranked = [c for c, _ in sc]
        rank = next((i+1 for i, c in enumerate(ranked) if c in rels), None)
        h1 += rank == 1; mrr += (1.0/rank) if rank else 0.0
        # text-free baseline: test embedding is zero -> score 0 -> random among cands
        emb_h1 += (sorted(cands)[0] in rels)  # deterministic tie -> ~random
        # fair audit
        if gold_is_cov:
            fair_n += 1
            cov = [c for c in cands if c in covered]
            fair_pools.append(len(cov)); rand_cov.append(1.0/len(cov))
            scf = sorted(((c, float(a@feats[c])) for c in cov), key=lambda kv: (-kv[1], kv[0]))
            fair_h1 += scf[0][0] in rels

    res = {
        "corpus": "corpus2 (TS/JS, densified repos)", "dense_repos": sorted(DENSE_REPOS),
        "n_queries": n, "mean_pool": round(float(np.mean(pool_sizes)), 1),
        "structure": {"gold_reachable": round(struct_reach/n, 4),
                      "negative_coverage": round(neg_cov/neg_tot, 4) if neg_tot else None},
        "embedder_cosine_R@1_textfree_baseline": round(emb_h1/n, 4),
        "graph_aug_asof_R@1": round(h1/n, 4), "graph_aug_asof_MRR": round(mrr/n, 4),
        "fair_R@1_among_covered": round(fair_h1/fair_n, 4) if fair_n else None,
        "random_among_covered": round(float(np.mean(rand_cov)), 4) if rand_cov else None,
        "pilot_reference": {"diff2test_R@1": 0.429, "fair_R@1": 0.500, "fair_x_random": 5.7, "textfree_baseline": 0.009},
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    s = res
    fx = s["fair_R@1_among_covered"]/s["random_among_covered"] if s["random_among_covered"] else 0
    print("R24c — diff->test on corpus2 (TS/JS), vs pilot")
    print(f"  n={n} queries, mean pool {s['mean_pool']}, dense repos {len(DENSE_REPOS)}")
    print(f"  structure: gold reachable {s['structure']['gold_reachable']:.1%} | neg coverage {s['structure']['negative_coverage']:.1%}")
    print(f"  text-free baseline (embedder-cosine) R@1: {s['embedder_cosine_R@1_textfree_baseline']:.3f}  (pilot 0.009)")
    print(f"  graph-aug + as_of R@1: {s['graph_aug_asof_R@1']:.3f}  MRR {s['graph_aug_asof_MRR']:.3f}  (pilot R@1 0.429)")
    print(f"  FAIR R@1 (among covered): {fx:.1f}x random  (pilot 5.7x)  -> "
          f"{'REPLICATES' if fx > 1.8 and s['graph_aug_asof_R@1'] > 0.25 else 'WEAKER/CHECK'}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
