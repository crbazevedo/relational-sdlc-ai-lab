"""A two-tower projection relation embedder (numpy).

Unlike the diagonal metric (which only reweights shared tokens), this learns an
asymmetric relation operator: separate low-rank projections for the query side
(issue) and the document side (PR/code), trained so that a query lands near the
artifact it is related to. It can align *different* tokens across the two sides
(issue symptom words ↔ change words) — the cross-token structure the diagonal
metric cannot represent and that real issue↔fix matching needs.

    a = Wq · u_issue      b = Wd · u_doc      score = <a, b>

Trained with a margin triplet loss on (issue, true-doc, negative) over the TRAIN
split only; evaluated by projected cosine on held-out queries. Deterministic
given the seed.
"""

from __future__ import annotations

import numpy as np

from .baseline import tokenize
from .metrics import RetrievalResult, evaluate
from .model import Vocab, _vectors
from .synth import SynthDataset


def _train_triplets(dataset: SynthDataset, vecs):
    fix_doc_of_query = {iss: pr for pr, iss in dataset.fixes}
    Ui, Up, Un = [], [], []
    for q in dataset.queries:
        if q.split != "train":
            continue
        pos = fix_doc_of_query.get(q.query_record)
        if pos is None or pos not in vecs:
            continue
        for cid in q.candidates:
            if cid == pos or cid not in vecs:
                continue
            Ui.append(vecs[q.query_record])
            Up.append(vecs[pos])
            Un.append(vecs[cid])
    if not Ui:
        return None
    return np.stack(Ui), np.stack(Up), np.stack(Un)


def train_tower(dataset, vecs, dim, d_proj=64, epochs=400, lr=0.5,
                margin=0.2, weight_decay=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    V = dim
    scale = 1.0 / np.sqrt(V)
    Wq = rng.normal(0, scale, size=(d_proj, V))
    Wd = rng.normal(0, scale, size=(d_proj, V))

    trip = _train_triplets(dataset, vecs)
    if trip is None:
        return Wq, Wd
    Ui, Up, Un = trip
    T = Ui.shape[0]

    for _ in range(epochs):
        Ai = Ui @ Wq.T            # (T, d)
        Bp = Up @ Wd.T
        Bn = Un @ Wd.T
        sp = np.sum(Ai * Bp, axis=1)
        sn = np.sum(Ai * Bn, axis=1)
        viol = (margin - sp + sn) > 0
        if viol.any():
            Aiv, Bpv, Bnv = Ai[viol], Bp[viol], Bn[viol]
            Uiv, Upv, Unv = Ui[viol], Up[viol], Un[viol]
            gWq = (Bnv - Bpv).T @ Uiv / T
            gWd = Aiv.T @ (Unv - Upv) / T
        else:
            gWq = np.zeros_like(Wq)
            gWd = np.zeros_like(Wd)
        Wq -= lr * (gWq + weight_decay * Wq)
        Wd -= lr * (gWd + weight_decay * Wd)
    return Wq, Wd


def _proj_unit(mat, vec):
    p = mat @ vec
    n = np.linalg.norm(p)
    return p / n if n > 0 else p


def _unit(vec):
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def _eval(dataset, vecs, ks, project=None):
    """Rank TEST-split candidates. ``project=(Wq, Wd)`` applies the learned towers;
    ``project=None`` is raw cosine on the input vectors (embedder-alone control)."""
    def q_embed(vec):
        return _proj_unit(project[0], vec) if project else _unit(vec)

    def d_embed(vec):
        return _proj_unit(project[1], vec) if project else _unit(vec)

    qmap, dmap, results = {}, {}, []
    for q in dataset.queries:
        if q.split != "test":
            continue
        if q.query_record not in qmap:
            qmap[q.query_record] = q_embed(vecs[q.query_record])
        a = qmap[q.query_record]
        scored = []
        for cid in q.candidates:
            if cid not in dmap:
                dmap[cid] = d_embed(vecs[cid])
            scored.append((cid, float(a @ dmap[cid])))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        results.append(RetrievalResult.of([cid for cid, _ in scored],
                                          q.relevant, q.hard_negatives))
    return evaluate(results, ks=ks)


def run_tower(dataset: SynthDataset, ks=(1, 5, 10), min_df=1, **train_kwargs) -> dict:
    """Bag-of-tokens variant: build a token vocab, then train + score the tower."""
    vocab = Vocab.build([tokenize(a.text) for a in dataset.artifacts], min_df=min_df)
    vecs = _vectors(dataset, vocab)
    Wq, Wd = train_tower(dataset, vecs, vocab.size, **train_kwargs)
    return _eval(dataset, vecs, ks, project=(Wq, Wd))


def run_tower_on_vecs(dataset, vecs: dict, dim: int, ks=(1, 5, 10), **train_kwargs) -> dict:
    """Dense-feature variant: train the relation head on precomputed vectors."""
    Wq, Wd = train_tower(dataset, vecs, dim, **train_kwargs)
    return _eval(dataset, vecs, ks, project=(Wq, Wd))


def run_cosine_on_vecs(dataset, vecs: dict, ks=(1, 5, 10)) -> dict:
    """Embedder-alone control: raw cosine on the precomputed vectors, no training."""
    return _eval(dataset, vecs, ks, project=None)


def train_relation_map(dataset, vecs, dim, epochs=200, lr=0.2, margin=0.1,
                       decay=1e-2, seed=0):
    """A relation operator M (h_p ~= M h_q) INITIALIZED AT THE IDENTITY.

    At init, score = (M u_i)·u_p = cosine, so it starts exactly at the
    embedder-alone control and can only refine it. ``decay`` pulls M back toward
    the identity, so it cannot stray far from the pretrained geometry — the fix
    for the from-scratch tower that overfit and destroyed the embeddings.
    """
    M = np.eye(dim)
    trip = _train_triplets(dataset, vecs)
    if trip is None:
        return M
    Ui, Up, Un = trip
    T = Ui.shape[0]
    eye = np.eye(dim)
    for _ in range(epochs):
        Ai = Ui @ M.T                      # rows = M u_i
        sp = np.sum(Ai * Up, axis=1)
        sn = np.sum(Ai * Un, axis=1)
        viol = (margin - sp + sn) > 0
        if viol.any():
            grad = (Un[viol] - Up[viol]).T @ Ui[viol] / T
        else:
            grad = np.zeros_like(M)
        M -= lr * (grad + decay * (M - eye))
    return M


def run_relation_map_on_vecs(dataset, vecs: dict, dim: int, ks=(1, 5, 10),
                             **train_kwargs) -> dict:
    """Identity-initialized relation operator on frozen embeddings."""
    M = train_relation_map(dataset, vecs, dim, **train_kwargs)
    return _eval(dataset, vecs, ks, project=(M, np.eye(dim)))
