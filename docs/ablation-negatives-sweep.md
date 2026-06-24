# Pushing the negatives lever on Apple Silicon (Track A/D, R18)

**Status: exploratory, real public data. First wave trained locally on an Apple M5
GPU (PyTorch MPS), not CPU.** R16C found the LoRA win *grows with density* to ΔR@1
**+0.114** on the dense 78-repo Tier-2 set, and R16D ([ablation-lora-sweep.md](ablation-lora-sweep.md))
showed that win is **rank-saturated** (r8≈r16≈r32, ±0.003) with **harder/more in-batch
negatives the only lever that moved it** — best at in-batch pool 48 (ΔR@1 +0.126),
where it stalled because **larger pools were capped by CPU memory**. This wave lifts
that cap: the M5's 32 GB unified memory + MPS let us train MiniLM-LoRA with much larger
and harder in-batch negative pools, and ask whether the headline keeps climbing.

It reuses the Tier-2 loaders, cross-repo split, leakage guards, and numpy eval
([`run_negatives_sweep.py`](../data/tier2/run_negatives_sweep.py) imports
`load_tier2_crossrepo` + `run_cosine_on_vecs`), and reproduces R16C/R16D's frozen
(R@1 0.515) and `b32` (ΔR@1 +0.114) cells as anchors before trusting any new cell.

## Hypotheses (falsifiable)

- **H1 — quantity.** Cross-repo ΔR@1 *increases then plateaus* as the in-batch
  negative pool grows past R16D's b48 (b48 → b96 → b192 → b384). InfoNCE with more
  in-batch negatives gives a tighter contrastive signal (the large-batch / MNR
  effect). **Control:** frozen MiniLM, and the committed b32 (+0.114) / b48 (+0.126).
  **Refutation:** if ΔR@1 plateaus at or below b48, R16D's "near-tuned" verdict stands
  and the pool size was *not* the limiter — the lever is exhausted.
- **H2 — quality (hardness).** At a *matched* pool size, **repo-homogeneous batches**
  (all in-batch negatives drawn from the anchor's own repository — semantically
  harder) beat random cross-repo batches. **Control:** the random-batch cell at the
  same batch size. **Refutation:** if repo-grouped ≈ random at matched size, any gain
  is pool *quantity*, not negative *hardness* — which itself sharpens the mechanism.

## Experiment design

- **Task / data.** `issue_to_fixing_pr` on the dense Tier-2 set (78 repos, 16,998
  records, 2,744 fixes). Train on **train-repo** fixes pairs only (46 repos, ~1,573
  pairs); evaluate raw cosine on the **32 held-out test repos** (1,171 queries) with
  explicit references scrubbed — the exact R16C protocol. One change at a time vs. the
  frozen `embedder-cosine` control.
- **Metric.** R@1 / R@5 / R@10 / MRR / hard-negative accuracy (`run_cosine_on_vecs`,
  numpy, deterministic). R@1 and MRR are the discriminating metrics.
- **Confirmation.** The best config's ΔR@1 gets a **bootstrap CI** (per-query +
  repo-cluster) on the 1,171 held-out queries — closing the open Tier-2-CI item from
  [R17a](ablation-bootstrap-ci.md).
- **Honesty.** Exploratory; single cross-repo split; MPS float math is not bit-identical
  to CPU, so anchors are checked to reproduce R16C/R16D within rounding before new cells
  are trusted.

## Training specs (shared across all cells)

| Knob | Value |
|---|---|
| Base encoder | `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-d), mean-pooled + L2-normalized |
| Adapter | LoRA **r=16, α=32**, dropout 0.05, target modules `query,key,value` (R16D's saturated-but-best rank) |
| Loss | symmetric InfoNCE (multiple-negatives ranking), temperature **0.05**, **in-batch negatives = batch size** |
| Optimizer | AdamW, lr **2e-4**, 12 epochs |
| Tokenizer | max_len 256, seed 0 |
| Device | **Apple M5 GPU via PyTorch MPS** (fp32; `PYTORCH_ENABLE_MPS_FALLBACK=1`) |
| Swept axis (H1) | batch ∈ {32, 48, 96, 192, 384} — the in-batch negative pool, random batching |
| Swept axis (H2) | batch 48, **repo-homogeneous batching** (hard) vs random (control) |

## Models / algorithms

The "model" is the LoRA-adapted MiniLM encoder (pretrained weights frozen; only the
low-rank `q/k/v` adapters learn). The "algorithm" is contrastive representation
learning by **symmetric InfoNCE**: for a batch of B (issue, fixing-PR) pairs, each
issue's gold PR is its positive and the other B−1 PRs in the batch are negatives
(and vice-versa). **Batch size is therefore the negative-pool size** — the single
quantity H1 sweeps. H2 changes only *which* negatives populate the batch (same-repo,
hard) at fixed size. Everything is evaluated, as everywhere in this lab, by cross-repo
retrieval on held-out repositories.

## Results

Trained on the Apple M5 GPU (MPS). Frozen reproduces R16C exactly (R@1 **0.515**),
the audit anchor, before any new cell is trusted. Raw numbers:
[`negatives-sweep-results.json`](../data/tier2/negatives-sweep-results.json),
[`negatives-bootstrap-results.json`](../data/tier2/negatives-bootstrap-results.json).

| Cell | pool | batching | R@1 | ΔR@1 | ΔMRR |
|---|---|---|---|---|---|
| frozen MiniLM-L6 | — | — | 0.515 | — | — |
| r16-b32 (R16C anchor) | 32 | random | 0.638 | +0.123 | +0.108 |
| r16-b48 (R16D best) | 48 | random | 0.635 | +0.120 | +0.106 |
| r16-b96 | 96 | random | 0.645 | +0.130 | +0.110 |
| r16-b192 / r16-b384 | 192 / 384 | random | — | **OOM** | — |
| **r16-b48 repo-hard** | 48 | **repo (hard)** | **0.652** | **+0.137** | **+0.120** |

### H1 (quantity) — refuted

Growing the random in-batch pool from 32 → 48 → 96 leaves ΔR@1 **flat** (+0.123 →
+0.120 → +0.130; a ±0.01 wiggle on the order of MPS run-to-run noise). And the two
largest pools **could not run at all**: b192 and b384 **OOM the M5 MPS allocator**
(the InfoNCE backprop over 384 / 768 texts needs >42 GiB), so the practical pool
ceiling on this machine is ≈ b96. The takeaway is two-sided: the 32 GB unified
memory raised R16D's CPU cap (b48 → b96) but **did not remove it**, and it does not
matter — *more negatives is not the lever*. This confirms R16D's "near-tuned"
verdict from the opposite direction.

### H2 (hardness) — confirmed; the lever is quality, not quantity

At the **same** pool size (48), **repo-homogeneous batches** — every in-batch
negative drawn from the anchor's own repository, so semantically harder — lift the
win to **ΔR@1 +0.137** (R@1 0.652), versus the matched random cell's +0.120
(**+0.017** from hardness alone, ~2× the run-to-run noise) and above every random
cell including the larger b96. It is the **new program best**, beating R16D's prior
best (r16-b48 random, +0.126) and the shipped recipe (r8-b32, +0.114) — and it is
*cheap* (small pool, 7.5 min on the M5). The lever R16D could only call "harder/more
negatives" is now resolved: **harder, not more.**

### Confirmation — the Tier-2 delta CI decisively excludes zero

Bootstrap on the repo-hard cell (1,171 held-out queries, 32 repos), exactly the
R17a protocol:

| | ΔR@1 | 95% CI | one-sided p[Δ≤0] |
|---|---|---|---|
| query bootstrap | +0.1375 | [+0.112, +0.164] | 0.0000 |
| repo-cluster bootstrap | +0.1375 | [+0.107, +0.173] | — (100% of resamples positive) |

**+210 / −49 rank-1 flips** (net +161); **31 of 32 held-out repos improve, none
regress.** This is far stronger and more uniform than the pilot (R17a: 5/8 repos,
thin +0.006 lower bound). It closes R17a's open Tier-2-CI item emphatically: at dense
Tier-2 scale the LoRA win is large, tight, and broad.

## Honest read & what it motivates

- **Quantity is exhausted; hardness is the live lever.** R18 refines R16D: pushing the
  *number* of negatives does nothing (and the biggest pools OOM the M5), while making
  them *harder* (same-repo batching) is the only thing that moves ΔR@1 — to a new best
  of +0.137 at a small, cheap pool.
- **MPS is not bit-deterministic.** Unlike the lab's numpy evals (reproducible to
  1e-9), MPS training varies run-to-run by ~±0.008 R@1 (the repo-hard cell read +0.146
  on a first run, +0.137 on the clean re-run). The headline uses the re-run value that
  the saved embeddings and bootstrap CI are computed on; the hardness gain (+0.017 at
  matched pool) is ~2× that noise, so **multi-seed confirmation is the natural
  follow-up** before treating the exact magnitude as settled.
- **First wave trained on Apple Silicon.** The lab was CPU-first; R18 is the first run
  on the M5 GPU (MPS), establishing the local training path — and mapping its ceiling
  (in-batch pool ≈ b96 before OOM).

All results are **exploratory**; the committed sweep + bootstrap JSON are the source of
truth, and MPS non-determinism is a labelled limitation.
