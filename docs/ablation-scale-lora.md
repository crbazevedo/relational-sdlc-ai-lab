# LoRA-at-scale: the win holds (and grows) on 55 repos

**Status: exploratory, real public data.** R11A showed the LoRA fine-tune win is
robust across 5 cross-repo splits of the 20-repo pilot. This re-runs the same
recipe on the **55-repo scale dataset** (R12A) — same protocol, more repositories —
to check the win isn't a small-dataset effect.

## Method

Identical to the pilot fine-tune: LoRA (r=8, attention q/k/v) on MiniLM, symmetric
InfoNCE on **train-repo** `fixes` pairs only, evaluated on **held-out test repos**,
references scrubbed. Frozen and tuned embeddings are scored on the same queries and
candidate pools. [`data/scale/finetune_scale.py`](../data/scale/finetune_scale.py),
[`data/scale/run_scale_finetune.py`](../data/scale/run_scale_finetune.py).

## Result (55 repos; 20 train / 14 held-out test repos)

| System | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| frozen embedder-cosine | 0.489 | 0.902 | 0.657 |
| **LoRA-tuned embedder-cosine** | **0.569** | **0.951** | **0.729** |
| **delta** | **+0.080** | **+0.049** | **+0.072** |

Card: [`gh-scale-lora-v0`](../data/cards/examples/gh-scale-lora-v0.experiment-card.json).

## Read

- **The win holds at scale, and the margin is if anything larger** — ΔR@1 +0.080
  here vs +0.063 on the pilot single split and +0.061±0.021 across the 5 pilot
  splits. More training repositories give the relation loss more to learn from.
- **Absolute numbers are lower than the pilot** (frozen 0.489 vs 0.592) because the
  scale split holds out 14 repositories spanning a wider domain — a harder, more
  honest generalization test. The *relational improvement* survives it.
- Single split here (the 5-split confidence study is the pilot's
  [ablation-scale.md](ablation-scale.md)); a multi-split at scale is the natural
  next confidence check.

## Takeaway

The core result — **LoRA fine-tuning with the relation loss generalizes across
held-out repositories** — is not a pilot artifact. It strengthens with scale,
which is the signal that justifies pushing to Tier-2 (200–500 repos) and a
code-embedding base.
