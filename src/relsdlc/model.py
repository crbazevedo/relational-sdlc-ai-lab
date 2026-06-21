"""Relation-aware retrieval models for the ablation (numpy).

Two systems, scored on the *same* unit vectors so the comparison is fair:

- ``vanilla`` — cosine similarity on bag-of-token vectors. The off-the-shelf
  text-similarity floor.
- ``relation-metric`` — a relation-supervised diagonal metric. It learns a
  non-negative weight per token, trained on the ``fixes`` relation so that tokens
  predictive of the fix (component/impl tokens) are up-weighted and ambiguous
  surface tokens (topics) are down-weighted. With all weights = 1 it is exactly
  vanilla cosine, so any gain comes purely from the relation supervision.

Everything is deterministic given the dataset seed.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .baseline import tokenize
from .metrics import RetrievalResult, evaluate
from .synth import SynthDataset


@dataclass
class Vocab:
    token2idx: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.token2idx)

    @classmethod
    def build(cls, token_lists) -> "Vocab":
        toks = sorted({t for tl in token_lists for t in tl})
        return cls({t: i for i, t in enumerate(toks)})

    def vectorize(self, tokens) -> np.ndarray:
        v = np.zeros(self.size, dtype=np.float64)
        for t in tokens:
            idx = self.token2idx.get(t)
            if idx is not None:
                v[idx] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


def _vectors(dataset: SynthDataset, vocab: Vocab) -> dict[str, np.ndarray]:
    return {a.id: vocab.vectorize(tokenize(a.text)) for a in dataset.artifacts}


def idf_weights(dataset: SynthDataset, vocab: Vocab) -> np.ndarray:
    """Unsupervised inverse-document-frequency weights over the vocabulary.

    The middle tier of the ablation: it down-weights common tokens using corpus
    statistics only — no relation labels. If the relation-supervised metric beats
    this, the gain is more than a frequency effect.
    """
    n_docs = len(dataset.artifacts)
    df: Counter = Counter()
    for a in dataset.artifacts:
        for t in set(tokenize(a.text)):
            if t in vocab.token2idx:
                df[t] += 1
    w = np.ones(vocab.size, dtype=np.float64)
    for t, i in vocab.token2idx.items():
        w[i] = math.log((1 + n_docs) / (1 + df[t])) + 1.0
    return w


def _rank(query_vec, candidate_ids, vecs, weight=None) -> list[str]:
    if weight is None:
        scores = {cid: float(query_vec @ vecs[cid]) for cid in candidate_ids}
    else:
        wq = weight * query_vec
        scores = {cid: float(wq @ vecs[cid]) for cid in candidate_ids}
    # Deterministic: by descending score then ascending id.
    return sorted(candidate_ids, key=lambda cid: (-scores[cid], cid))


def train_relation_metric(dataset: SynthDataset, vocab: Vocab,
                          vecs: dict[str, np.ndarray],
                          epochs: int = 300, lr: float = 0.5,
                          margin: float = 0.15, l2: float = 1e-4,
                          seed: int = 0) -> np.ndarray:
    """Learn non-negative token weights w = theta**2 via a margin triplet loss.

    Triplets come only from TRAIN queries: (issue, true_pr, negative_candidate).
    Returns the weight vector w (length vocab.size); w == 1 reproduces vanilla.
    """
    rng = np.random.default_rng(seed)
    fix_pr_of_issue = {iss: pr for pr, iss in dataset.fixes}
    train_q = [q for q in dataset.queries if q.split == "train"]

    anchors, positives, negatives = [], [], []
    for q in train_q:
        pos = fix_pr_of_issue[q.query_record]
        for cid in q.candidates:
            if cid == pos:
                continue
            anchors.append(q.query_record)
            positives.append(pos)
            negatives.append(cid)
    if not anchors:
        return np.ones(vocab.size)

    Ui = np.stack([vecs[a] for a in anchors])
    Up = np.stack([vecs[p] for p in positives])
    Un = np.stack([vecs[n] for n in negatives])
    T = Ui.shape[0]

    theta = np.ones(vocab.size, dtype=np.float64)
    for _ in range(epochs):
        w = theta ** 2
        sp = (w * Ui * Up).sum(axis=1)
        sn = (w * Ui * Un).sum(axis=1)
        viol = (margin - sp + sn) > 0
        if viol.any():
            # d/dtheta of (sn - sp) = 2*theta * Ui*(Un - Up)
            grad = 2 * theta * (Ui[viol] * (Un[viol] - Up[viol])).sum(axis=0) / T
        else:
            grad = np.zeros_like(theta)
        grad += l2 * theta
        theta -= lr * grad
        theta = np.clip(theta, 0.0, None)
    return theta ** 2


def run_ablation(dataset: SynthDataset, ks=(1, 5, 10), seed: int = 0) -> dict:
    """Score three tiers on TEST queries: vanilla, unsupervised IDF, relation metric.

    All three use the same unit vectors and candidate pools, so differences come
    only from the token weighting:
      - vanilla         : no weighting (plain cosine).
      - idf-cosine      : unsupervised corpus IDF weighting.
      - relation-metric : weights learned from the ``fixes`` relation.
    """
    vocab = Vocab.build([tokenize(a.text) for a in dataset.artifacts])
    vecs = _vectors(dataset, vocab)
    idf = idf_weights(dataset, vocab)
    rel = train_relation_metric(dataset, vocab, vecs, seed=seed)

    test_q = [q for q in dataset.queries if q.split == "test"]

    def eval_system(weight) -> dict:
        results = []
        for q in test_q:
            ranked = _rank(vecs[q.query_record], q.candidates, vecs, weight=weight)
            results.append(RetrievalResult.of(ranked, q.relevant, q.hard_negatives))
        return evaluate(results, ks=ks)

    # mean learned weight by token family — shows what the supervision recovered.
    impl_w = [rel[i] for t, i in vocab.token2idx.items() if t.startswith("impl")]
    topic_w = [rel[i] for t, i in vocab.token2idx.items() if t.startswith("topic")]

    return {
        "systems": {
            "vanilla-tf-cosine": eval_system(None),
            "idf-cosine": eval_system(idf),
            "relation-metric": eval_system(rel),
        },
        "n_train_queries": sum(1 for q in dataset.queries if q.split == "train"),
        "n_test_queries": len(test_q),
        "vocab_size": vocab.size,
        "learned_weights": {
            "mean_impl_weight": float(np.mean(impl_w)) if impl_w else 0.0,
            "mean_topic_weight": float(np.mean(topic_w)) if topic_w else 0.0,
        },
        "params": dataset.params,
    }


def learned_weight_summary(dataset: SynthDataset, seed: int = 0) -> dict:
    """Mean learned weight for impl vs topic tokens — shows what the metric learned."""
    vocab = Vocab.build([tokenize(a.text) for a in dataset.artifacts])
    vecs = _vectors(dataset, vocab)
    weight = train_relation_metric(dataset, vocab, vecs, seed=seed)
    impl = [weight[i] for t, i in vocab.token2idx.items() if t.startswith("impl")]
    topic = [weight[i] for t, i in vocab.token2idx.items() if t.startswith("topic")]
    return {
        "mean_impl_weight": float(np.mean(impl)) if impl else 0.0,
        "mean_topic_weight": float(np.mean(topic)) if topic else 0.0,
    }
