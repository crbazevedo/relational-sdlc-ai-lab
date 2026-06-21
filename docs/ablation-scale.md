# Track D: is the LoRA win robust? (multi-split confidence)

**Status: exploratory, real public data.** R10A showed LoRA fine-tuning with the
relation loss beats the frozen embedder on one cross-repo split. The obvious
worry: is that one split lucky? This re-runs the whole train→eval over **5
different held-out-repo partitions** and reports the spread.

## Method

For each of 5 partitions (seeded `100..104`), repositories are split 60/40 into
train/test, MiniLM is LoRA-fine-tuned (r=8, InfoNCE) on that partition's
**train-repo** `fixes` pairs, and frozen-cosine vs tuned-cosine are scored on the
partition's **held-out test repos**. One variable changes (which repos are held
out); everything else is fixed.
[`data/pilot/run_multisplit.py`](../data/pilot/run_multisplit.py); raw numbers in
`data/pilot/multisplit-results.json`.

## Result

| Split | held-out repos | frozen R@1 | LoRA R@1 | Δ R@1 | Δ MRR |
|---|---|---|---|---|---|
| 0 | 8 | 0.566 | 0.643 | **+0.077** | +0.066 |
| 1 | 8 | 0.513 | 0.592 | **+0.079** | +0.057 |
| 2 | 8 | 0.495 | 0.542 | **+0.047** | +0.045 |
| 3 | 8 | 0.481 | 0.558 | **+0.078** | +0.056 |
| 4 | 8 | 0.413 | 0.440 | **+0.027** | +0.037 |
| **mean ± std** | | **0.494 ± 0.050** | **0.555 ± 0.067** | **+0.061 ± 0.021** | **+0.052 ± 0.010** |

## Read

- **The win is robust.** The LoRA delta is **positive on all 5 splits** for both
  R@1 and MRR. The mean ΔR@1 (+0.061) is ~3× its std (0.021); ΔMRR (+0.052) is ~5×
  its std (0.010). This is not a single-split artifact.
- **Absolute difficulty varies by which repos are held out** (frozen R@1 ranges
  0.41–0.57) — expected, since some repositories are harder to generalize to. The
  *relational improvement* survives that variation.
- **Still pilot scale.** 18 repos, ~180–210 train pairs per split, one seed per
  partition, a general-text base. The direction and consistency are trustworthy;
  the magnitude will move with scale and a code-specific base (Q6).

## Gate decision (Track D)

**Confirmed.** The relational fine-tune generalizes across held-out repositories
with a consistent positive margin. Next on this track: more repositories/pairs
(Tier-2) to tighten the estimate, multiple seeds per partition, and a code-aware
base. Compounding note: [the graph track](ablation-gnn.md) found that mean
aggregation over the typed graph adds a further small lift *on top of* the LoRA
representation (LoRA+graph R@1 0.69 on the canonical split) — representation and
structure stack.
