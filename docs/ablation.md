# Ablation: relation-supervised retrieval vs vanilla text similarity

**Status: exploratory, synthetic.** This is a controlled demonstration of the
lab's central mechanism, not a real-world result. Real public-data validation is
the P1→P2 follow-up (see [roadmap.md](roadmap.md)).

## Hypothesis

> When the operational relation between artifacts (here, `fixes`: which PR fixes
> an issue) is carried by *rare, predictive* tokens while the *surface* is
> dominated by *common, ambiguous* tokens, vanilla text-similarity retrieval is
> misled, and a model trained on the relation recovers the true link.

## The benchmark

[`src/relsdlc/synth.py`](../src/relsdlc/synth.py) procedurally generates a
codebase of latent **components**. Two token families:

- **impl tokens** — component-specific and rare (appear in one component only).
  They identify the component, hence the fix.
- **topic tokens** — global "symptom" words, common and shared across components.
  Ambiguous.

An **issue** is topic-heavy with a faint sprinkle of its component's impl tokens.
Its **fixing PR** is impl-heavy. So the only reliable surface link between an
issue and its true fix is a rare impl token; meanwhile **hard-negative** PRs from
other components share *topic* words with the issue and look more similar on the
surface. The frozen dataset and the eval queries live under
[`data/synth/`](../data/synth/); the split is by record (`train`/`test`), and
every component appears in train so its impl tokens are learnable.

## Systems (all scored on the same vectors and candidate pools)

| System | Weighting | Supervision |
|---|---|---|
| `vanilla-tf-cosine` | none (plain cosine) | — |
| `idf-cosine` | unsupervised corpus IDF | none |
| `relation-metric` | learned per-token weights | the `fixes` relation (train split only) |

`relation-metric` ([`src/relsdlc/model.py`](../src/relsdlc/model.py)) learns a
non-negative weight per token via a margin triplet loss over train `fixes` pairs.
With all weights = 1 it is exactly vanilla cosine, so any gain comes purely from
the relation supervision.

## Result

Reproduce with `relsdlc ablation` (seed 7 dataset, seed 0 training):

| System | Recall@1 | Recall@5 | MRR | Hard-neg accuracy |
|---|---|---|---|---|
| vanilla-tf-cosine | 0.106 | 0.309 | 0.276 | 0.106 |
| idf-cosine | 0.383 | 0.532 | 0.501 | 0.383 |
| **relation-metric** | **0.819** | **0.894** | **0.864** | **0.819** |

(94 held-out test queries.) Per-system experiment cards:
[`data/cards/examples/synth-*.experiment-card.json`](../data/cards/examples/).

The learned weights tell the story: mean impl-token weight ≈ **1.36**, mean
topic-token weight ≈ **0.28**. The relation supervision recovered that rare impl
tokens predict the fix and common topic tokens do not — without ever being told.

## Interpretation

- **Vanilla fails** (R@1 ≈ 0.11): plain cosine is dominated by the common topic
  tokens, so topic-sharing distractors outrank the true fix.
- **IDF helps but is not enough** (R@1 ≈ 0.38): down-weighting common tokens by
  corpus frequency partially recovers the signal — the gain is not *just* an IDF
  effect, which is why the comparison matters.
- **Relation supervision wins decisively** (R@1 ≈ 0.82): learning the weighting
  from the `fixes` relation more than doubles IDF and is ~8× vanilla on Recall@1.

## What this does NOT show

- It is **synthetic**. It demonstrates the mechanism under a known structure; it
  does not measure performance on real repositories.
- It uses a **single relation** (`fixes`) and a **single task**
  (`issue_to_fixing_pr`).
- The `relation-metric` is a diagonal token reweighting — a deliberately simple,
  interpretable model, not a neural bi-encoder. The real-data work will need
  cross-token (bilinear/projection) relation operators where the right weighting
  is not a simple IDF.

The honest claim is narrow and falsifiable: *relation supervision recovers a link
that surface similarity cannot, on a controlled benchmark.*

**Follow-up (done):** the same ablation on a real public GitHub pilot is in
[ablation-real.md](ablation-real.md). Spoiler — it does **not** transfer: real
issue→fix retrieval is surface-rich, IDF is the strong baseline, and the diagonal
relation-metric ties it rather than beating it. That negative result is the point:
it shows the synthetic mechanism needs a cross-token model to matter on real data.
