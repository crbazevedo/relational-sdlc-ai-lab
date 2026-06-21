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


def train_tower(dataset, vocab, vecs, d_proj=64, epochs=400, lr=0.5,
                margin=0.2, weight_decay=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    V = vocab.size
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


def run_tower(dataset: SynthDataset, ks=(1, 5, 10), min_df=1, **train_kwargs) -> dict:
    """Train the two-tower projection and score it on TEST queries."""
    vocab = Vocab.build([tokenize(a.text) for a in dataset.artifacts], min_df=min_df)
    vecs = _vectors(dataset, vocab)
    Wq, Wd = train_tower(dataset, vocab, vecs, **train_kwargs)

    qproj = {}
    dproj = {}
    results = []
    for q in dataset.queries:
        if q.split != "test":
            continue
        if q.query_record not in qproj:
            qproj[q.query_record] = _proj_unit(Wq, vecs[q.query_record])
        a = qproj[q.query_record]
        scored = []
        for cid in q.candidates:
            if cid not in dproj:
                dproj[cid] = _proj_unit(Wd, vecs[cid])
            scored.append((cid, float(a @ dproj[cid])))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        ranked = [cid for cid, _ in scored]
        results.append(RetrievalResult.of(ranked, q.relevant, q.hard_negatives))
    return evaluate(results, ks=ks)
