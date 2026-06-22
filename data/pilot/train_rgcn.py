#!/usr/bin/env python3
"""Learned relational GNN (1-hop R-GCN) vs the training-free graph aggregation.

R11B's free mean-aggregation gave a small issue→PR lift. This learns the
aggregation: a PR's embedding becomes W_self·x_PR + W_f2p·mean(x of files it
modifies); an issue's is W_self·x_issue. Initialized at W_self=I, W_f2p=0 — so it
starts EXACTLY at frozen cosine and can only improve — and trained with InfoNCE on
TRAIN-repo `fixes` pairs (the `fixes` edge is supervision, never a message-passing
edge, so there is no leakage). Inductive: built on pretrained node features, so it
applies to held-out repos.

Needs the [embed] extra (torch). Caches the learned node embeddings; eval is
numpy-only on the cache.

Run:  python data/pilot/train_rgcn.py
Out:  data/pilot/embeddings/rgcn-frozen.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from run_crossrepo_ablation import load_pilot_crossrepo  # noqa: E402

BASE = HERE / "embeddings" / "minilm-l6-v2.npz"
OUT = HERE / "embeddings" / "rgcn-frozen.npz"
EPOCHS = 200
LR = 5e-3
TEMP = 0.05
SEED = 0


def _load_jsonl(p): return [json.loads(l) for l in p.read_text().split("\n") if l.strip()]


def main() -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: needs [embed] extra: {exc}", file=sys.stderr); raise SystemExit(2)
    torch.manual_seed(SEED)

    d = np.load(BASE, allow_pickle=False)
    ids = [str(i) for i in d["ids"]]
    idx = {i: k for k, i in enumerate(ids)}
    X = torch.tensor(d["vectors"].astype(np.float32))
    dim = X.shape[1]

    # PR -> list of modified-file feature indices (the message-passing graph).
    files_of_pr: dict[str, list[int]] = {}
    for e in _load_jsonl(HERE / "graph" / "modifies_edges.jsonl"):
        if e["source"] in idx and e["target"] in idx:
            files_of_pr.setdefault(e["source"], []).append(idx[e["target"]])

    ds, meta = load_pilot_crossrepo()
    fix_pr = {iss: pr for pr, iss in ds.fixes}
    by = ds.by_id()
    train_pairs = [(idx[iss], idx[fix_pr[iss]]) for iss in (q.query_record for q in ds.queries
                   if q.split == "train") if iss in idx and fix_pr.get(iss) in idx]

    # Precompute mean file-feature per PR (0 if none).
    mean_files = torch.zeros((len(ids), dim))
    for pr, fids in files_of_pr.items():
        mean_files[idx[pr]] = X[fids].mean(0)

    W_self = torch.eye(dim, requires_grad=True)
    W_f2p = torch.zeros((dim, dim), requires_grad=True)
    opt = torch.optim.Adam([W_self, W_f2p], lr=LR)

    def encode_all():
        h = X @ W_self.T + mean_files @ W_f2p.T
        return torch.nn.functional.normalize(h, p=2, dim=1)

    rng = np.random.default_rng(SEED)
    pairs = np.array(train_pairs)
    for ep in range(EPOCHS):
        order = rng.permutation(len(pairs))
        H = encode_all()
        a = H[pairs[order, 0]]
        b = H[pairs[order, 1]]
        sims = (a @ b.T) / TEMP
        lab = torch.arange(a.shape[0])
        loss = (torch.nn.functional.cross_entropy(sims, lab)
                + torch.nn.functional.cross_entropy(sims.T, lab)) / 2
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep + 1}/{EPOCHS} loss={float(loss):.4f}", file=sys.stderr)

    with torch.no_grad():
        H = encode_all().numpy().astype(np.float16)
    np.savez_compressed(OUT, ids=np.array(ids), vectors=H, model="rgcn-1hop-frozen")
    print(f"wrote learned R-GCN embeddings {H.shape} -> {OUT}")


if __name__ == "__main__":
    main()
