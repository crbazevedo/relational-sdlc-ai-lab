"""Retrieval metrics for relational SDLC benchmarks (pure standard library).

A benchmark query produces a ranked list of candidate record ids. Each query
carries the set of relevant (positive) ids and, optionally, a set of designated
hard-negative ids. These functions turn those into Recall@K, MRR, and
hard-negative accuracy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalResult:
    """One evaluated query.

    ``ranked`` is the candidate id list, best first.
    ``relevant`` is the set of correct ids.
    ``hard_negatives`` are plausible-but-wrong ids used to stress the model.
    """

    ranked: Sequence[str]
    relevant: frozenset[str]
    hard_negatives: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def of(cls, ranked: Iterable[str], relevant: Iterable[str],
           hard_negatives: Iterable[str] = ()) -> "RetrievalResult":
        return cls(list(ranked), frozenset(relevant), frozenset(hard_negatives))


def recall_at_k_single(result: RetrievalResult, k: int) -> float:
    if not result.relevant:
        return 0.0
    top = set(result.ranked[:k])
    hits = len(top & result.relevant)
    return hits / len(result.relevant)


def reciprocal_rank_single(result: RetrievalResult) -> float:
    for rank, cid in enumerate(result.ranked, start=1):
        if cid in result.relevant:
            return 1.0 / rank
    return 0.0


def hard_negative_success_single(result: RetrievalResult) -> bool | None:
    """True if a relevant id outranks every hard negative.

    Returns None when the query has no hard negatives (excluded from the mean).
    """
    if not result.hard_negatives or not result.relevant:
        return None
    best_pos = _best_rank(result.ranked, result.relevant)
    best_neg = _best_rank(result.ranked, result.hard_negatives)
    if best_pos is None:
        return False
    if best_neg is None:
        return True
    return best_pos < best_neg


def _best_rank(ranked: Sequence[str], ids: frozenset[str]) -> int | None:
    for rank, cid in enumerate(ranked):
        if cid in ids:
            return rank
    return None


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def recall_at_k(results: Iterable[RetrievalResult], k: int) -> float:
    return _mean(recall_at_k_single(r, k) for r in results)


def mrr(results: Iterable[RetrievalResult]) -> float:
    return _mean(reciprocal_rank_single(r) for r in results)


def hard_negative_accuracy(results: Iterable[RetrievalResult]) -> float:
    outcomes = [s for r in results if (s := hard_negative_success_single(r)) is not None]
    return _mean(1.0 if s else 0.0 for s in outcomes)


def evaluate(results: Sequence[RetrievalResult], ks: Sequence[int] = (1, 5, 10)) -> dict:
    """Aggregate the standard metric bundle for a benchmark task."""
    results = list(results)
    return {
        "n_queries": len(results),
        "recall_at_k": {str(k): recall_at_k(results, k) for k in ks},
        "mrr": mrr(results),
        "hard_negative_accuracy": hard_negative_accuracy(results),
    }
