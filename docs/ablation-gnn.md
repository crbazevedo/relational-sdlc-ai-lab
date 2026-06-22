# Graph structure vs. pairwise text cosine (Track B, first probe)

**Status: exploratory, real public data.** This is the first probe of Track B's
question — *does graph structure add signal beyond pairwise text cosine?* — on the
multi-relation pilot graph (`fixes` + `modifies`, with `file`/`test` nodes from the
[graph enrichment](graph-enrichment.md)).

It is deliberately the **cheapest possible probe**: a *training-free* GraphSAGE-style
mean aggregation ([`src/relsdlc/graphsage.py`](../src/relsdlc/graphsage.py)) over the
typed graph, scored by the same cosine used everywhere else. There are no learned
weights. A node's augmented feature is

    aug(v) = normalize( alpha * own(v) + (1 - alpha) * mean over edge-roles of mean(neighbours) )

with `alpha = 0.5`, two hops, typed by edge role (a PR aggregates the files it
modifies + the issue it fixes; a file/test node — which has **no text embedding** —
is defined purely by the PRs that modify it). It runs on the cached frozen MiniLM
features and the LoRA-tuned features, on the same de-referenced cross-repo split as
the rest of the lab (10 train repos, 8 held out). Numpy only; CI needs no torch.

**Leakage guard.** The aggregation never reads the edge it is being evaluated on:
for `issue_to_fixing_pr` the gold `(issue, fixing-PR)` `fixes` edges are excluded;
for `diff_to_affected_test` the gold `(PR, test)` `modifies` edges are excluded. A
control run that *keeps* the gold `modifies` edge confirms the guard matters — it
trivially hits R@1 0.868, MRR 1.00 because the test node then literally averages in
the query PR's own vector. With the answer edge removed, that shortcut is gone.

Run: `python data/pilot/run_gnn_ablation.py`. Cards:
[`data/cards/examples/gh-gnn-*.experiment-card.json`](../data/cards/examples/).

## Results

### `issue_to_fixing_pr` (174 held-out queries)

| Features | System | R@1 | R@5 | R@10 | MRR | Hard-neg |
|---|---|---|---|---|---|---|
| frozen | embedder-cosine | 0.592 | 0.920 | 0.989 | 0.728 | 0.592 |
| frozen | **graph-aug-cosine** | **0.609** | **0.954** | 0.989 | **0.754** | **0.615** |
| LoRA | embedder-cosine | 0.655 | 0.994 | 0.994 | 0.791 | 0.661 |
| LoRA | **graph-aug-cosine** | **0.690** | 0.994 | 0.994 | **0.817** | **0.690** |

### `diff_to_affected_test` (112 held-out queries)

| Features | System | R@1 | R@5 | R@10 | MRR | Hard-neg |
|---|---|---|---|---|---|---|
| frozen | embedder-cosine | 0.009 | 0.175 | 0.705 | 0.155 | 0.089 |
| frozen | graph-aug-cosine | 0.009 | 0.175 | 0.705 | 0.155 | 0.089 |
| LoRA | embedder-cosine | 0.009 | 0.175 | 0.705 | 0.155 | 0.089 |
| LoRA | graph-aug-cosine | 0.009 | 0.184 | 0.705 | 0.156 | 0.089 |

## Honest read

**Mixed, and net: training-free aggregation is not the answer.**

1. **`issue_to_fixing_pr`: a small but consistent lift.** One hop of typed
   aggregation that pulls in the *files a PR modifies* nudges R@1 up by ~+0.02–0.04
   and MRR by ~+0.03 over `embedder-cosine`, on both frozen and LoRA features
   (frozen 0.592→0.609, LoRA 0.655→0.690). This is real — the gold `fixes` edge is
   excluded, so the gain comes from the `modifies` structure, not the answer. It
   says there *is* a little structural signal a text embedder misses (which PR
   touches which files is informative about which issue it fixes). But the effect is
   modest and a free mean-aggregation, not a learned operator, captured it — so the
   bar a learned GNN must clear is "beat this small lift", not "beat raw cosine".

2. **`diff_to_affected_test`: training-free aggregation essentially fails.** With
   the gold `(PR, test)` edge honestly removed, **75 of 160 relevant test files
   (47%) become isolated** — they are modified by *only* their query PR in the
   pilot, so once the answer edge is gone they have no other structural neighbour
   and fall back to a zero vector (unrankable). Plain text cosine can't see test
   nodes at all (no text → R@1 ≈ 0.009, i.e. random). Mean aggregation gives the
   non-isolated half a feature, but averaged over all queries it barely moves the
   needle (R@5 0.175→0.184 on LoRA only). A free neighbour-average cannot
   generalise from "PRs that co-modify this test" to "PRs semantically like the
   query PR".
   *(Note: unlike the same-repo `issue→PR` pools, the `diff→test` candidate pools
   mix test files across repos — an inconsistency that does not affect the verdict,
   since the task is documented as failing at pilot scale, but worth tightening when
   the task is revisited at scale.)*

## What this motivates

This was a *first probe, not the final word.* A training-free aggregation only tells
us whether the structure is so blunt that even averaging helps — and the verdict is:

- On `issue_to_fixing_pr` there is a thin structural signal worth a **learned**
  inductive GNN (R-GCN / GraphSAGE with trained weights, torch) that can weight
  edge roles and hops instead of a fixed `alpha`/mean — the natural Track B
  follow-up, gated on clearing the small free-aggregation lift above.
- On `diff_to_affected_test` the limiter is **data sparsity, not method**: half the
  positives are degree-1 after the leakage guard. A learned link-predictor that maps
  *query-PR text* to *test-file* via a trained relation operator (rather than
  averaging the test's PR neighbours) is the right shape, but it needs either richer
  co-change structure or more repos (Track D scale) before the signal exists.

Per the Track B gate ([research-roadmap](research-roadmap.md)): the free probe shows
a *thin* signal on issue→PR and *no* training-free signal on diff→test at pilot
scale. That is enough to justify the **learned** GNN/KG-embedding experiment, with
clear-eyed expectations — and a reminder that graph structure is not yet shown to be
load-bearing for diff→test until the graph (or the repo count) is denser.

All results above are **exploratory** and labelled so on every experiment card; they
are not release-quality evidence.
