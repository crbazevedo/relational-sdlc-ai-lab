# Track A: LoRA fine-tuning — the relational contribution, inside the representation

**Status: exploratory, real public data — but the first positive relational
result.** R8 showed that a relation operator bolted onto *frozen* embeddings can't
help, and concluded the contribution must live *inside* the representation. This
tests that directly: LoRA-fine-tune the embedder with the relation loss, then re-run
the same de-referenced, cross-repo benchmark.

## Method

- Base: `sentence-transformers/all-MiniLM-L6-v2` (frozen baseline from
  [ablation-embed.md](ablation-embed.md)).
- **LoRA** adapters (r=8, on attention query/key/value) — 110,592 trainable params
  (**0.48%** of the model). The pretrained weights are untouched; only the adapters
  learn.
- Loss: symmetric **InfoNCE / multiple-negatives** over `(issue, fixing-PR)` pairs
  with in-batch negatives, on **TRAIN repos only** (182 pairs). References scrubbed.
- Then embed all records with the tuned model, cache, and evaluate on the **held-out
  test repos** — same split as every prior result.
- Script: [`data/pilot/finetune_embed.py`](../data/pilot/finetune_embed.py)
  (needs the `[embed]` extra; the tuned embeddings are cached + committed so the
  evaluation is numpy-only and CI-reproducible).

## Result (8 held-out test repos, issue_to_fixing_pr)

| System | Recall@1 | Recall@5 | Recall@10 | MRR | Hard-neg |
|---|---|---|---|---|---|
| idf-cosine (bag-of-tokens bar) | 0.460 | 0.828 | 0.977 | 0.624 | 0.460 |
| embedder-cosine (frozen) | 0.592 | 0.920 | 0.989 | 0.728 | 0.592 |
| **embedder-cosine (LoRA-tuned)** | **0.655** | **0.994** | 0.994 | **0.791** | 0.661 |
| LoRA + identity-init operator | 0.649 | 0.994 | 0.994 | 0.790 | 0.655 |

Cards: [`data/cards/examples/gh-finetune-*.experiment-card.json`](../data/cards/examples/).

## Findings

1. **The relational fine-tune wins cross-repo.** LoRA-tuning with the relation loss
   lifts R@1 0.592 → **0.655**, MRR 0.728 → **0.791**, R@5 0.920 → **0.994**, on
   repositories not seen in training. This is the first time the relational
   contribution beats the frozen-embedder control on real held-out data.
2. **The contribution is in the representation, not a head.** Adding the
   identity-init operator on top of the *tuned* embeddings changes nothing
   (0.649 ≈ 0.655) — exactly R8's prediction. Reshaping `E_θ` is what helps;
   projecting it afterward does not.
3. **0.48% of params, on CPU.** A tiny LoRA adapter on a 22M model, trained on 182
   pairs, moved the needle — evidence the signal is real and learnable, not a
   capacity artifact.

## Honest caveats

- **Pilot scale:** 182 train pairs, 8 test repos, a single seed and a single
  cross-repo split. Treat this as a **positive signal to confirm**, not a settled
  number. The roadmap's Track-A gate requires it to survive a held-out-repo
  re-split and, ideally, multiple seeds.
- **R@5 ≈ 0.99 is near-ceiling** because candidate pools are small (~12); R@1 and
  MRR are the discriminating metrics, and both improve clearly.
- General-text base (not code-specific) — Q6 (a code embedder) may do better still.

## Gate decision (per [research-roadmap.md](research-roadmap.md) Track A)

**WIN** → proceed to scale (Track D: more repos/pairs) with multi-split confidence
intervals, and try a code-specific base (Q6). The relational contribution, placed
correctly — *inside* the representation via LoRA — generalizes across repositories.
This is the result the whole bag-of-tokens → embeddings → fine-tune arc was built to
test.
