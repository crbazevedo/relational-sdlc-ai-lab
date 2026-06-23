# LoRA sweep on dense Tier-2: is the +0.114 win a floor or near-tuned? (R16D)

**Status: exploratory, real public data.** R16C shipped a LoRA-at-Tier-2 result —
on the dense 78-repo set, a LoRA fine-tune of MiniLM-L6 beats the frozen embedder
cross-repo by **ΔR@1 +0.114** ([tier2-dataset.md](tier2-dataset.md)). The shipped
recipe used rank `r=8`, in-batch InfoNCE pool 32. This wave asks the obvious
follow-up before reaching for a GPU: **was that recipe under-tuned, or is +0.114
near the CPU ceiling for this model?** Two axes, same fixed frozen baseline, same
held-out test repos with references scrubbed.

[`data/tier2/sweep_tier2_lora.py`](../data/tier2/sweep_tier2_lora.py);
raw numbers in `data/tier2/tier2-sweep-results.json`. MiniLM-L6 q/k/v, symmetric
InfoNCE on TRAIN-repo fixes pairs only, 12 epochs, eval = raw cosine on the 32
held-out test repos (1,171 queries). CPU-only training; numpy eval.

## Result

| config | R@1 | R@5 | R@10 | MRR | HardNegAcc | ΔR@1 |
|---|---|---|---|---|---|---|
| frozen MiniLM-L6 | 0.515 | 0.829 | 0.969 | 0.655 | 0.523 | — |
| r8-b32 (shipped) | 0.629 | 0.921 | 0.985 | 0.757 | 0.635 | +0.114 |
| r16-b32 | 0.626 | 0.921 | 0.986 | 0.756 | 0.631 | +0.111 |
| r32-b32 | 0.632 | 0.921 | 0.985 | 0.760 | 0.638 | +0.117 |
| **r16-b48** (harder negs) | **0.641** | 0.921 | 0.986 | **0.763** | **0.646** | **+0.126** |

The `r8-b32` row reproduces the shipped R16C result **exactly** (0.629 / +0.114),
so the sweep harness is the same recipe, deterministic, and the cells are directly
comparable.

## Read

- **Rank is saturated.** r8 → r16 → r32 at the same batch land at **0.629 / 0.626 /
  0.632** — a ±0.003 R@1 band, i.e. noise. Doubling and quadrupling the LoRA adapter
  capacity does **not** help; `r=8` was already well-chosen for this task and scale.
  More parameters are not the lever.
- **Harder negatives are the lever — modestly.** Holding rank fixed and growing the
  in-batch InfoNCE pool 32 → 48 lifts R@1 **0.626 → 0.641** (+0.015 from the
  negatives alone) and beats the shipped recipe by **+0.012 R@1 / +0.011 hard-neg**.
  A larger pool gives each anchor more — and on average harder — negatives to
  contrast against, which is exactly where this task's signal is. **`r16-b48` is the
  new best recipe in the program (ΔR@1 +0.126).**
- **So +0.114 was near-tuned, not the floor — and the remaining headroom points at
  GPU.** The one axis that moved is the negative pool, and on a 16 GB CPU box the
  pool is capped (~48 before memory pressure). Pushing it further — large-batch
  contrastive, explicit mined hard negatives at scale — is a **GPU-memory** lever,
  not a CPU one. This sweep both (a) confirms the shipped result is honest and (b)
  sharpens the case that the next gain wants a GPU.

## How to regenerate

```bash
caffeinate -ims env PYTHONUNBUFFERED=1 .venv-embed/bin/python \
  data/tier2/sweep_tier2_lora.py > data/tier2/sweep.log 2>&1
```

Reuses the committed-recipe frozen cache if present; embeddings are computed
in-memory per config (no npz round-trip); each config is independently guarded so
one failure (e.g. OOM) does not abort the rest. ~37 min/config on CPU.

## Artifacts
- `data/tier2/sweep_tier2_lora.py` — the sweep (rank × in-batch-pool).
- `data/tier2/tier2-sweep-results.json` — raw metrics for every config.
- `tests/test_tier2_sweep.py` — reads the result, pins that every config beats
  frozen and that the harder-negatives config is best.
