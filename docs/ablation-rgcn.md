# Learned R-GCN vs training-free aggregation (issue→PR, cross-repo)

**Status: exploratory, real public data — honest negative.** R11B's parameter-free
graph aggregation gave a small issue→PR lift. Does *learning* the aggregation (a
1-hop R-GCN) beat it? At pilot scale: **no.**

## Method

A 1-hop relational layer over the typed graph: a PR's embedding becomes
`W_self·x_PR + W_f2p·mean(x of the files it modifies)`; an issue's is `W_self·x_issue`.
Initialized at `W_self = I`, `W_f2p = 0` — so it starts *exactly* at frozen cosine —
and trained with InfoNCE on **train-repo** `fixes` pairs. The `fixes` edge is
supervision only, never a message-passing edge, so there is no leakage. Inductive
(built on pretrained features → applies to held-out repos).
[`data/pilot/train_rgcn.py`](../data/pilot/train_rgcn.py).

## Result (8 held-out repos)

| System | R@1 | R@5 | MRR |
|---|---|---|---|
| frozen-cosine | 0.592 | 0.920 | 0.728 |
| **free-aggregation (R11B, parameter-free)** | **0.609** | 0.954 | 0.754 |
| learned R-GCN (1-hop) | 0.575 | 0.937 | 0.730 |

Card: [`gh-rgcn-1hop-v0`](../data/cards/examples/gh-rgcn-1hop-v0.experiment-card.json).

## Finding

The learned R-GCN **underperforms the parameter-free aggregation** (and is a hair
below frozen). The InfoNCE loss barely moved from its identity init, and the small
learned changes generalized slightly *worse* than a fixed mean. This is the same
pattern seen throughout the lab: **at pilot scale, learned parameters bolted on top
of frozen features overfit** — learning helps only when it reshapes the *base*
representation (the LoRA win), not when added as a separate head/layer over frozen
vectors.

## Implication

- Free aggregation (R@1 0.609) remains the best graph method here; structure helps
  a little and a parameter-free combiner captures it.
- A learned GNN needs **more supervision (Track-D repo scale)** and/or a richer,
  better-regularized architecture (multi-hop R-GCN with per-relation transforms,
  edge dropout) before its extra capacity pays off. Re-run this at scale.
- `diff→test` still needs denser co-change (most relevant tests are isolated at
  pilot scale — see [ablation-gnn.md](ablation-gnn.md)).
