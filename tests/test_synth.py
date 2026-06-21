"""Synthetic generator: determinism, structure, and the hard-negative property."""

from __future__ import annotations

from relsdlc.synth import generate


def test_generate_is_deterministic():
    a = generate(seed=7)
    b = generate(seed=7)
    assert [x.id for x in a.artifacts] == [x.id for x in b.artifacts]
    assert [x.tokens for x in a.artifacts] == [x.tokens for x in b.artifacts]
    assert a.fixes == b.fixes
    assert [q.candidates for q in a.queries] == [q.candidates for q in b.queries]


def test_every_issue_has_one_fixing_pr_in_candidates():
    ds = generate(seed=7)
    fix_pr = {iss: pr for pr, iss in ds.fixes}
    for q in ds.queries:
        true_pr = fix_pr[q.query_record]
        assert q.relevant == [true_pr]
        assert true_pr in q.candidates
        for neg in q.hard_negatives:
            assert neg in q.candidates
            assert neg != true_pr


def test_true_pr_and_issue_share_component_but_negatives_do_not():
    ds = generate(seed=7)
    by_id = ds.by_id()
    fix_pr = {iss: pr for pr, iss in ds.fixes}
    for q in ds.queries:
        issue = by_id[q.query_record]
        true_pr = by_id[fix_pr[q.query_record]]
        assert true_pr.component == issue.component
        for neg in q.hard_negatives:
            assert by_id[neg].component != issue.component


def test_both_splits_present_and_components_covered_in_train():
    ds = generate(seed=7)
    splits = {a.split for a in ds.artifacts}
    assert splits == {"train", "test"}
    # Every component must appear in train so its impl tokens are learnable.
    train_components = {a.component for a in ds.artifacts if a.split == "train"}
    all_components = {a.component for a in ds.artifacts}
    assert train_components == all_components
