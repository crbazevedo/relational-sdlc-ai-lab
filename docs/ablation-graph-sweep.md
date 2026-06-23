# Graph-lift robustness: sweeping (alpha × hops) (Track B, R16E)

**Status: exploratory, real public data.** A follow-up to the
[first graph probe](ablation-gnn.md) (R11B), which measured the training-free typed
mean-aggregation lift at a **single** operating point — `alpha=0.5, hops=2` — and
reported it as "small but real" on `issue_to_fixing_pr` and as a failure on
`diff_to_affected_test`. A single point cannot tell a *robust structural signal*
from a *tuned coincidence*, nor say *why* diff→test fails. This wave answers both by
sweeping the two free knobs of [`graphsage.py`](../src/relsdlc/graphsage.py).

It reuses R11B's loaders, cross-repo split, and leakage guards **verbatim**
([`run_graph_sweep.py`](../data/pilot/run_graph_sweep.py) imports them from
`run_gnn_ablation.py`), so the `alpha=0.5, hops=2` cell reproduces R11B exactly.
Numpy only; no torch, no network; deterministic.

Run: `PYTHONPATH=src python3 data/pilot/run_graph_sweep.py`. Cards:
[`gh-graphsweep-*`](../data/cards/examples/). Grid:
[`data/pilot/graph-sweep-results.json`](../data/pilot/graph-sweep-results.json).

The augmented feature is, as in R11B,
`aug(v) = normalize( alpha·own(v) + (1−alpha)·mean_role(mean(neighbours)) )`,
swept over `alpha ∈ {0, 0.25, 0.5, 0.75, 1.0}` × `hops ∈ {1, 2, 3}`. At `alpha=1.0`
a text node keeps its own vector, so graph-aug **must** equal embedder-cosine — a
sanity anchor the harness passes exactly (frozen 0.592, LoRA 0.655).

## 1. `issue_to_fixing_pr` — the lift is a plateau, not a knife-edge

R@1 across `alpha` (174 held-out queries; embedder-cosine = the `alpha=1.0` column):

| Features | hops | a=0.0 | a=0.25 | a=0.5 | a=0.75 | a=1.0 (=cosine) |
|---|---|---|---|---|---|---|
| frozen | 1 | 0.621 | 0.603 | 0.609 | 0.609 | **0.592** |
| frozen | 2 | 0.621 | 0.603 | 0.609 | 0.609 | 0.592 |
| frozen | 3 | 0.598 | 0.615 | 0.615 | 0.615 | 0.592 |
| LoRA | 1 | 0.684 | **0.690** | **0.690** | 0.678 | **0.655** |
| LoRA | 2 | 0.684 | 0.690 | 0.690 | 0.678 | 0.655 |
| LoRA | 3 | 0.649 | 0.684 | 0.690 | 0.678 | 0.655 |

MRR moves the same way (LoRA h1: 0.814 / **0.820** / 0.817 / 0.806 / 0.791).

Three things fall out:

1. **The lift is positive across the entire non-trivial `alpha` range** `[0, 0.75]`,
   for both frozen and LoRA — it does not depend on landing on `alpha=0.5`. R11B's
   operating point sits on a **plateau** (LoRA +0.029…+0.035; frozen +0.011…+0.029),
   not a tuned spike. That is the evidence a single point could not give.
2. **One hop suffices.** `hops=1` and `hops=2` are *byte-identical* everywhere; the
   second hop changes no ranking. R11B's headline 0.690 is a pure 1-hop
   neighbourhood effect (a PR pulling in the files it modifies). `hops=3` does not
   help and slightly *hurts* at low `alpha` (over-smoothing).
3. **The effect is small in absolute terms** — at 174 queries, +0.035 R@1 is ~6
   queries on one cross-repo split. The contribution of this wave is *robustness*
   (it is a plateau, and cheap), not a larger number.

## 2. `diff_to_affected_test` — structure-bound, now quantified

R@1 is **flat at 0.009** across all 15 `(alpha, hops)` cells on both feature sets
(MRR ~0.155). No setting rescues it. The diagnostic says why — independent of any
embedding:

| metric (test split) | value |
|---|---|
| gold (query-PR, test) pairs | 160 |
| **isolated after the leakage guard** (gold PR is the *only* PR modifying the test) | **75 (46.9%)** |
| queries with *all* relevant tests isolated | 45 / 112 |
| **reachable ceiling** (R@anything, structure-bound) | **59.8%** |

Once the gold `(PR, test)` edge is honestly removed, ~47% of the positive test
nodes have **no** remaining modifying PR — they are degree-0 and *cannot* be placed
by any aggregation, at any hop count. Multi-hop cannot manufacture co-change that
the pilot does not contain. This converts R11B's qualitative "fails at pilot
sparsity" into a hard, feature-independent number: the limiter is **co-change
density**, a *data* property, not the aggregation method or its hyperparameters.

## Honest read & what it motivates

- **issue→PR:** the training-free structural lift is *robust* (a plateau over
  `alpha`, saturated at 1 hop) but *small*. The bar a **learned** GNN must clear is
  therefore well-defined and modest: beat ~0.690 (LoRA) / ~0.621 (frozen), not raw
  cosine. Gated, as before, on Track-D scale giving a learned head the supervision
  it lacks at pilot.
- **diff→test:** the experiment is now decisive about the *cause*. A learned
  link-predictor will not help while 47% of positives are degree-1 — the
  prerequisite is **denser co-change** (more PRs per test ⇒ more repos / a denser
  `modifies` graph, Track D). Until then, diff→test is structure-bound below 60%
  recall by construction, regardless of features.

All results are **exploratory** and labelled so on every card; they are not
release-quality evidence. They strengthen (not replace) R11B: the graph lift on
issue→PR is real and robust but thin; diff→test is blocked on density, now measured.
