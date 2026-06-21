"""The two-tower projection: validated on cross-token structure.

The load-bearing claim: the tower recovers a relation where the two sides share
NO tokens (so vanilla / IDF / the diagonal metric are at chance), proving it
captures cross-token structure they cannot.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from relsdlc.model import run_ablation  # noqa: E402
from relsdlc.synth import generate, generate_crosstoken  # noqa: E402
from relsdlc.tower import run_tower  # noqa: E402


def test_tower_solves_cross_token_where_diagonal_cannot():
    ds = generate_crosstoken(seed=11)
    base = run_ablation(ds, seed=0)
    tower = run_tower(ds, seed=0, d_proj=64, epochs=500)
    diag = base["systems"]["relation-metric"]["recall_at_k"]["1"]
    van = base["systems"]["vanilla-tf-cosine"]["recall_at_k"]["1"]
    # Surface + diagonal methods are near chance on disjoint vocab; the tower wins.
    assert van < 0.2
    assert diag < 0.2
    assert tower["recall_at_k"]["1"] > 0.6
    assert tower["recall_at_k"]["1"] > diag + 0.4


def test_tower_beats_vanilla_on_per_token_too():
    # On the per-token benchmark the tower still beats plain cosine (learns signal).
    ds = generate(seed=7)
    base = run_ablation(ds, seed=0)
    tower = run_tower(ds, seed=0)
    assert tower["recall_at_k"]["1"] > base["systems"]["vanilla-tf-cosine"]["recall_at_k"]["1"]


def test_tower_is_deterministic():
    ds = generate_crosstoken(seed=11)
    a = run_tower(ds, seed=0, d_proj=32, epochs=100)
    b = run_tower(ds, seed=0, d_proj=32, epochs=100)
    assert a["recall_at_k"]["1"] == b["recall_at_k"]["1"]


def test_crosstoken_generator_disjoint_vocab():
    ds = generate_crosstoken(seed=11)
    by_id = ds.by_id()
    fix_pr = {iss: pr for pr, iss in ds.fixes}
    for q in ds.queries:
        issue_toks = {t for t in by_id[q.query_record].tokens}
        pr_toks = {t for t in by_id[fix_pr[q.query_record]].tokens}
        shared_non_topic = {t for t in (issue_toks & pr_toks) if not t.startswith("topic")}
        assert not shared_non_topic  # only topic noise may overlap
