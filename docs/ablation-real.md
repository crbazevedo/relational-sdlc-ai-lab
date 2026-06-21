# Real-data ablation: the synthetic win does NOT transfer (yet)

**Status: exploratory, real public data (pilot scale).** This is the honest test
of the [synthetic ablation](ablation.md) on real GitHub data. The headline: on
real issue→fixing-PR retrieval, surface similarity is already strong, unsupervised
IDF is the best simple baseline, and the diagonal relation-metric **does not beat
it**. A negative result, reported plainly.

## Dataset

The [P1 pilot](../data/pilot/) — a one-time live snapshot of **20 permissive
Python-ecosystem repos** (pytest, fastapi, pydantic, flask, click, jinja, black,
ruff, poetry, scrapy, …). Closed issues + closed PRs; `fixes` edges mined from
closing keywords ("Fixes #N"); body text truncated to 500 chars
(redistribution = metadata_only). Frozen **temporal** split (earliest 60% of
fixes by issue date = train). Dataset card:
[`data/cards/examples/gh-pilot-v0.dataset-card.json`](../data/cards/examples/gh-pilot-v0.dataset-card.json).

- 2,087 records · 356 `fixes` edges · 356 queries (212 train / 144 test).
- Hard negatives: same-repo PRs ranked by title/body token overlap with the issue.

## Result

Reproduce with `python data/pilot/run_real_ablation.py` (deterministic; vocab
min_df=3, vocab size 3541):

| System | Recall@1 | Recall@5 | Recall@10 | MRR | Hard-neg accuracy |
|---|---|---|---|---|---|
| vanilla-tf-cosine | 0.347 | 0.715 | 0.944 | 0.522 | 0.354 |
| **idf-cosine** | **0.438** | **0.812** | 0.958 | **0.611** | 0.438 |
| relation-metric | 0.417 | 0.785 | 0.965 | 0.595 | 0.438 |

Per-system experiment cards:
[`data/cards/examples/gh-pilot-*.experiment-card.json`](../data/cards/examples/).

## What this shows (honestly)

1. **Real issue→fix retrieval is surface-rich.** Vanilla cosine already gets
   R@5 ≈ 0.72 — the opposite of the synthetic benchmark, because real PRs quote
   and restate the issues they fix.
2. **IDF is the strong simple baseline.** Down-weighting common tokens by corpus
   frequency lifts R@1 by ~9 points and R@5 by ~10. If you do one thing, do IDF.
3. **The diagonal relation-metric does NOT beat IDF** (R@1 0.42 vs 0.44; it ties
   on hard-negative accuracy). Its learned weights barely move from 1.0: on real
   text, the `fixes` signal is not a per-token reweighting beyond frequency, so a
   diagonal metric has nothing extra to learn.

## Why the synthetic win didn't transfer

The synthetic benchmark was solvable by *per-token* reweighting (rare impl tokens
carried the link). Real issue↔fix association is **cross-token and semantic**: the
issue describes a symptom, the PR describes a change, and matching them needs a
model that can associate *different* tokens, not just up-weight shared ones. A
diagonal metric cannot represent that; at best it recovers IDF.

## What it motivates (P3)

- A **cross-token relation operator** — bilinear (`hᵤᵀ W hᵥ`) or a low-rank
  projection bi-encoder — that can learn issue-token ↔ PR-token associations.
- **Real text/code embeddings** (an off-the-shelf code embedder) fine-tuned on
  the `fixes` relation, instead of bag-of-tokens.
- **Relations where surface text is weaker** — diff→affected-test, log→file —
  where structure (co-change, coverage) should carry more than restated text.

The deliverable of P1 is not a win; it is a **frozen public benchmark + honest
baselines** that make the real research question measurable. The simple bar to
beat is now explicit: **IDF at R@1 0.44 / MRR 0.61**.
