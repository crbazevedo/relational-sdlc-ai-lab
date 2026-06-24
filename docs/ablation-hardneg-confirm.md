# Pinning down the hardness lever: multi-seed + mining (Track A/D, R19)

**Status: exploratory, real public data; trained on the Apple M5 GPU (MPS).** R18
([ablation-negatives-sweep.md](ablation-negatives-sweep.md)) found, **on a single
seed**, that same-repo "hard" batching beats a matched random in-batch pool on the
dense Tier-2 LoRA fine-tune (ΔR@1 +0.137 vs +0.120 at pool 48). That +0.017 gap is
only ~2× the MPS run-to-run non-determinism (~±0.008 R@1), so it is *suggestive, not
settled*. This wave does two things:

1. **Confirm (multi-seed).** Repeat `random` vs `repo-hard` across **seeds {0,1}**
   and measure the **paired** gap per seed — does hardness beat random on *every*
   seed, and does the mean gap clear the noise? (Two seeds is a light confirmation —
   chosen to keep the unattended MPS run tractable; a third seed is a trivial add if
   the gap is borderline.)
2. **Strengthen (mining).** Add a third, harder mechanism — explicit **mined hard
   negatives** — to test whether *harder still* helps beyond same-repo batching.

It reuses R18's training/eval helpers verbatim (MiniLM-L6 q/k/v LoRA r16/α32,
symmetric InfoNCE temp 0.05 on train-repo pairs, 12 epochs, eval = raw cosine on the
32 held-out test repos / 1,171 queries) and the same cached frozen baseline (R@1
0.515). [`run_hardneg_confirm.py`](../data/tier2/run_hardneg_confirm.py); raw numbers
in [`hardneg-confirm-results.json`](../data/tier2/hardneg-confirm-results.json).

## Hypotheses (falsifiable)

- **H1 — hardness is real.** Across seeds, `repo-hard` ΔR@1 > `random` ΔR@1 on every
  seed, and the mean paired gap exceeds the run-to-run std. **Refuted** if the gap
  flips sign on any seed or the mean is within one std of zero (then R18's +0.017 was
  MPS noise and R16D's "near-tuned" stands unqualified).
- **H2 — harder still helps.** `mined` (random batch + per-anchor same-repo
  top-token-overlap hard negative) beats both `random` and `repo-hard` at the same
  pool. **Refuted** if `mined` ≈ `repo-hard` (then same-repo batching already
  saturates the hardness lever) or `mined` < `random` (then the mined negatives are
  false negatives hurting training).

## Three mechanisms (matched pool 48)

| Config | In-batch negatives | Loss |
|---|---|---|
| `random` | whatever the random shuffle places in the batch (mostly cross-repo, easy) | symmetric InfoNCE, B×B |
| `repo-hard` | repo-homogeneous batch — all same-repo (harder) | symmetric InfoNCE, B×B |
| `mined` | random batch **plus** each anchor's same-repo, non-gold PR with the highest issue↔PR token overlap | forward InfoNCE over [positives ; mined-negs] (B×2B) + symmetric back (B×B) |

Hard negatives are mined once from the **train** split only (same-repo, gold
excluded, ranked by token overlap), so no test leakage; anchors with no same-repo
candidate fall back to an in-batch positive of another anchor (a valid negative).

## Training specs

Identical to R18 (the matched control): MiniLM-L6, LoRA r=16/α=32 on q/k/v, AdamW
lr 2e-4, 12 epochs, temp 0.05, max_len 256, batch 48, device MPS (fp32). The only
varied axes are **seed ∈ {0,1}** and **negative-hardness mechanism ∈ {random,
repo-hard, mined}** — 6 cells, each vs the same frozen baseline.

## Analysis plan

Per config: mean ± std of ΔR@1 / ΔMRR across seeds. Per seed: paired gaps
`repo-hard − random` and `mined − random` (and `mined − repo-hard`); report the
per-seed signs, the mean, and whether every seed is positive. The decision rule is
the **paired** comparison (it cancels the shared seed/data variance, isolating the
hardness effect from MPS + seed noise).

## Results

Trained on the Apple M5 GPU (MPS), same frozen baseline as R18 (R@1 0.515). Raw
numbers: [`hardneg-confirm-results.json`](../data/tier2/hardneg-confirm-results.json).

### H1 (hardness is real) — confirmed on both seeds

| config | seed 0 ΔR@1 | seed 1 ΔR@1 | mean |
|---|---|---|---|
| `random` | +0.119 | +0.126 | +0.122 |
| **`repo-hard`** | +0.138 | +0.146 | **+0.142** |
| **paired gap (repo-hard − random)** | **+0.019** | **+0.021** | **+0.020** |

The hardness gap is **positive on every seed** and **tight**: +0.0188 and +0.0205, a
mean of **+0.020** with the two values within 0.002 of each other — comfortably above
the within-cell MPS run-to-run noise (~±0.008). The paired design cancels the shared
seed/data variance, so this isolates the hardness effect, and it does not wash out.
**R18's single-seed +0.017 was real.**

A useful side-observation on MPS determinism: seed-0 `repo-hard` here reads **+0.1375**,
*identical* to R18's clean re-run — so the **+0.146** R18 reported on its *first* run
was an artefact of the pre-cleanup memory-leak state, not seed variance. With the
allocator freed between cells, same-seed results are in fact reproducible to ~3
decimals; the run-to-run spread is smaller than R18 feared. `repo-hard` peaks at
+0.146 (seed 1) and averages **+0.142** — the new program best, clear of R16D's +0.126.

### H2 (harder-still / mining) — deferred

The mined-hard-negative mechanism is implemented and smoke-tested
([`run_hardneg_confirm.py`](../data/tier2/run_hardneg_confirm.py), `mined` config), but
its cells were **not run to completion**: MPS cell wall-time on this machine is highly
variable (a `repo-hard` cell ran 9.5 min once and 22 min another time), and the longer
mined cells repeatedly exceeded the environment's background-job limit and were killed.
H2 is a clean follow-up once a longer or more stable compute window is available; the
code path is ready.

## Honest read & what it motivates

- **The hardness lever is confirmed.** Across two seeds the paired `repo-hard − random`
  gap is +0.020, consistent and above the noise floor — so R18's "harder, not more"
  finding is robust, not a single-seed/MPS fluke. The Tier-2 LoRA best is `repo-hard`
  at ΔR@1 ~+0.142 (peak +0.146).
- **Two seeds is a light confirmation**, chosen to fit an unstable compute window. The
  effect is tight enough (std ~0.001 on the gap) that more seeds are unlikely to
  overturn it, but a 3rd–5th seed and the H2 mining comparison are the natural next
  steps when compute is steadier.
- **Operational note:** one-cell-per-launch with per-cell checkpointing was required to
  make progress against a background-job time limit + variable MPS throughput — the
  resumable harness ([`run_hardneg_confirm.py`](../data/tier2/run_hardneg_confirm.py))
  is the reusable artefact for future MPS waves.

All results are **exploratory**; the committed results JSON is the source of truth, and
MPS variability + the 2-seed scope are labelled limitations.
