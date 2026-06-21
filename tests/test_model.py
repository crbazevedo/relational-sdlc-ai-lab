"""The ablation result: relation supervision beats IDF beats vanilla.

These assertions are tolerant (orderings + margins), not exact numbers, so they
hold across platforms even if the last decimal of a metric shifts.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from relsdlc.model import Vocab, idf_weights, run_ablation  # noqa: E402
from relsdlc.synth import generate  # noqa: E402
from relsdlc.baseline import tokenize  # noqa: E402


def _r1(report, system):
    return report["systems"][system]["recall_at_k"]["1"]


def test_relation_metric_beats_idf_beats_vanilla():
    ds = generate(seed=7)
    report = run_ablation(ds, seed=0)
    vanilla = _r1(report, "vanilla-tf-cosine")
    idf = _r1(report, "idf-cosine")
    relation = _r1(report, "relation-metric")
    # The thesis: relation supervision wins, and by a clear margin over both.
    assert relation > idf > vanilla
    assert relation - vanilla > 0.3
    assert relation - idf > 0.15


def test_relation_metric_wins_on_mrr_and_hardneg():
    report = run_ablation(generate(seed=7), seed=0)
    rel = report["systems"]["relation-metric"]
    van = report["systems"]["vanilla-tf-cosine"]
    assert rel["mrr"] > van["mrr"]
    assert rel["hard_negative_accuracy"] > van["hard_negative_accuracy"]


def test_supervision_upweights_impl_downweights_topics():
    report = run_ablation(generate(seed=7), seed=0)
    lw = report["learned_weights"]
    # Predictive (rare) impl tokens end up weighted well above ambiguous topics.
    assert lw["mean_impl_weight"] > lw["mean_topic_weight"]


def test_ablation_is_deterministic():
    r1 = run_ablation(generate(seed=7), seed=0)
    r2 = run_ablation(generate(seed=7), seed=0)
    assert _r1(r1, "relation-metric") == _r1(r2, "relation-metric")


def test_metrics_in_unit_range():
    report = run_ablation(generate(seed=7), seed=0)
    for m in report["systems"].values():
        for v in m["recall_at_k"].values():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0
        assert 0.0 <= m["hard_negative_accuracy"] <= 1.0


def test_idf_weights_downweight_common_tokens():
    ds = generate(seed=7)
    vocab = Vocab.build([tokenize(a.text) for a in ds.artifacts])
    idf = idf_weights(ds, vocab)
    impl = [idf[i] for t, i in vocab.token2idx.items() if t.startswith("impl")]
    topic = [idf[i] for t, i in vocab.token2idx.items() if t.startswith("topic")]
    # Rare impl tokens get higher IDF than common topic tokens.
    assert np.mean(impl) > np.mean(topic)
