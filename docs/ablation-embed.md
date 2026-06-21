# Pretrained embeddings + the relation operator (cross-repo)

**Status: exploratory, real public data.** This closes the loop opened by
[ablation-crossrepo.md](ablation-crossrepo.md), which concluded that bag-of-tokens
can't generalize across repos and a pretrained embedder is needed. It is now
tested, on the same de-referenced, cross-repo split.

## Setup

A small frozen embedder — `sentence-transformers/all-MiniLM-L6-v2` (22M params,
384-d, CPU) — embeds the reference-scrubbed issue/PR text. The vectors are cached
once ([`data/pilot/embed_pilot.py`](../data/pilot/embed_pilot.py), needs the
`[embed]` extra) and committed, so every system below runs on the cache with numpy
alone — reproducible in CI without torch or a download.

## Result (same split: 10 train repos, 8 held-out test repos)

| System | Recall@1 | Recall@5 | Recall@10 | MRR | Hard-neg |
|---|---|---|---|---|---|
| vanilla-tf-cosine | 0.391 | 0.724 | 0.931 | 0.544 | 0.397 |
| idf-cosine (the bag-of-tokens bar) | 0.460 | 0.828 | 0.977 | 0.624 | 0.460 |
| relation-metric (diagonal) | 0.454 | 0.799 | 0.960 | 0.618 | 0.471 |
| **embedder-cosine** | **0.592** | 0.920 | 0.989 | 0.728 | 0.592 |
| embedder + from-scratch tower | 0.190 | 0.644 | 0.920 | 0.381 | 0.218 |
| **embedder + identity-init operator** | 0.592 | **0.931** | 0.989 | **0.737** | 0.592 |

Cards: [`data/cards/examples/gh-embed-*.experiment-card.json`](../data/cards/examples/).

## Three findings

1. **Pretrained embeddings are the cross-repo win.** Frozen MiniLM cosine lifts
   R@1 from IDF's 0.46 to **0.59** (R@5 0.92, MRR 0.73) — with zero training. This
   confirms the [cross-repo conclusion](ablation-crossrepo.md): meaning transfers
   across repos where tokens do not. The substrate question is settled.

2. **A from-scratch relation head on frozen embeddings is actively harmful.**
   Training a random two-tower projection on ~180 cross-repo pairs collapses R@1 to
   **0.19** — it overfits the train repos and destroys the embedder's native
   geometry. Bolting a learned operator on top, naively, is worse than nothing.

3. **An identity-initialized operator is safe but adds ~nothing.** Starting the
   operator at the identity (so it begins exactly at cosine) and regularizing back
   toward it keeps R@1 at 0.59 with a marginal R@5/MRR bump (0.92→0.93, 0.728→0.737).
   It cannot hurt, but at this scale it cannot meaningfully help either: the frozen
   embedder's geometry already has little headroom for a small-data refinement.

## What this means for "the relational contribution as the core"

On *frozen* embeddings at pilot scale, a bolt-on relation operator has no room to
add value. So the relational contribution cannot be a post-hoc head — it must live
in one of two places:

- **Inside the representation** — fine-tune the embedder end-to-end with the
  contrastive/relation loss (LoRA on the small model). The relation signal then
  *reshapes* the embedding rather than projecting a frozen one. This is the next
  experiment, and findings 2–3 are exactly why it is the right one.
- **In graph structure the cosine can't see** — link prediction over the typed
  SDLC graph (issues↔PRs↔files↔commits↔tests) with a relational GNN / KG
  embedding. Cosine uses only pairwise text; a GNN can exploit multi-hop relational
  structure. This needs the richer graph (the file/commit/test edges deferred from
  P1) and an inductive setup for cross-repo transfer.

The evidence has narrowed the search precisely: **embeddings for the substrate;
the relational contribution via fine-tuning and/or graph link-prediction, not a
frozen-embedder head.**
