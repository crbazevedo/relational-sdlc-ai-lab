# De-referenced, cross-repo ablation: why bag-of-tokens can't generalize

**Status: exploratory, real public data.** This is the non-degenerate testbed:
explicit references removed, and repositories held out. The finding is sharp and
negative — and it points directly at what the next model must be.

## The two fixes that make the testbed honest

1. **Reference removal.** An explicit link (`Fixes #123`) is a valid *label* but a
   degenerate *test* — a regex recovers it. [`relsdlc.scrub`](../src/relsdlc/scrub.py)
   strips `#N`, `gh-N`, `owner/repo#N`, issue/PR URLs, and commit SHAs from both
   sides, so a system must match on *semantics*, not the issue number.
2. **Cross-repo split.** Train repos are disjoint from test repos (10 train / 8
   test). A win now requires a relation that **generalizes to unseen
   repositories**, not repo-specific surface memorization.

## Result (174 held-out-repo queries, vocab 3,479)

| System | Recall@1 | Recall@5 | Recall@10 | MRR | Hard-neg acc |
|---|---|---|---|---|---|
| vanilla-tf-cosine | 0.391 | 0.724 | 0.931 | 0.544 | 0.397 |
| **idf-cosine** | **0.460** | **0.828** | 0.977 | **0.624** | 0.460 |
| relation-metric (diagonal) | 0.454 | 0.799 | 0.960 | 0.618 | 0.471 |
| relation-tower (projection) | 0.241 | 0.713 | 0.931 | 0.445 | 0.362 |

Cards: [`data/cards/examples/gh-xrepo-*.experiment-card.json`](../data/cards/examples/).

## Is the tower broken? No — it's validated, and it still loses

The two-tower projection is the right model for *cross-token* structure, and a
controlled check proves it works. On a **cross-token synthetic** benchmark where
the issue and its fix share **no** tokens (`relsdlc.synth.generate_crosstoken`):

| | vanilla | IDF | diagonal | **tower** |
|---|---|---|---|---|
| Recall@1 | 0.010 | 0.010 | 0.010 | **0.832** |

Vanilla, IDF, and the diagonal metric all sit at chance (they can only reweight
*shared* tokens; there are none). The tower recovers the cross-token mapping. The
implementation is correct.

So why does the same model score **0.241 (below vanilla)** on real cross-repo? In
the synthetic case the concepts appear in both train and test. Across **held-out
repos**, the discriminative vocabulary (API names, module names, repo idioms) is
*new* — its projection columns were never trained — so the tower applies random
projections to exactly the tokens that matter, and underperforms an untrained
cosine. The shared tokens it *can* project are the ambiguous common words.

## The conclusion

1. **Real issue→fix is surface-rich.** Vanilla cosine already gets R@5 ≈ 0.72,
   even de-referenced — the signal was always shared symptom/domain vocabulary,
   not the issue number. Reference removal is necessary for honesty but does not
   reveal hidden semantic difficulty.
2. **IDF is the robust baseline** (R@1 0.46) precisely *because* it does not train
   — corpus frequency transfers across repos for free.
3. **From-scratch token projections cannot generalize cross-repo.** Tokens don't
   transfer between repositories; only *meaning* does.

Therefore the next model must sit on features that already generalize across
repos: **pretrained text/code embeddings** (a code embedder), with the relation
operator (the tower / a bilinear head) learned *on top of* them. That is the
concrete P3, and it is exactly why the original plan starts from an off-the-shelf
embedder rather than bag-of-words. The bag-of-tokens stack has now told us, with
evidence, the precise reason it is not enough.
