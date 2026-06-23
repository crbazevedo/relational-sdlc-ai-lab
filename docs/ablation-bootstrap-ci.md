# Hardening the headline: CIs + per-repo decomposition of the LoRA win (Track A/D, R17a)

**Status: exploratory, real public data.** The program's headline positive result is
the relational LoRA fine-tune beating the frozen embedder cross-repo
([ablation-finetune.md](ablation-finetune.md)): on the default de-referenced
cross-repo split (8 held-out repos, 174 queries), R@1 `0.592 → 0.655`, MRR
`0.728 → 0.791`. [R11A](ablation-scale.md) already showed the win is not split luck —
positive on **all 5** held-out-repo partitions (mean ΔR@1 +0.061 ± 0.021). What that
single default-split number still lacked was a **within-split confidence interval**
and an answer to *where the win comes from*. This wave supplies both, without any new
training.

It reuses the loaders, the cross-repo split, and the exact ranking of
[`run_gnn_ablation.py`](../data/pilot/run_gnn_ablation.py) /
[`run_crossrepo_ablation.py`](../data/pilot/run_crossrepo_ablation.py) **verbatim**
([`run_bootstrap_ci.py`](../data/pilot/run_bootstrap_ci.py)), and **asserts the
aggregate reproduces the committed [`finetune-results.json`](../data/pilot/finetune-results.json)
to 1e-9 before any inference** — so the intervals annotate a number the
[R13 audit](research-roadmap.md#audit) already trusts byte-for-byte. Numpy only; no
torch, no network; deterministic (seed 0, B = 10,000). Raw numbers:
[`data/pilot/bootstrap-ci-results.json`](../data/pilot/bootstrap-ci-results.json).

## 1. Two bootstraps — both exclude zero

Two resampling units answer two different questions. A **query bootstrap** (resample
the 174 queries) asks *is the delta stable to query sampling?* A **repo-cluster
bootstrap** (resample the 8 held-out repos, take all their queries) is the honest CI
for a *cross-repo* claim — queries inside one repo are correlated, so the repo is the
unit. R11A's ±0.021 is the orthogonal third view (variation over *which* repos are
held out).

| Quantity (default split, n=174, 8 repos) | Point | Query-bootstrap 95% CI | Repo-cluster 95% CI |
|---|---|---|---|
| **ΔR@1** (LoRA − frozen) | **+0.063** | **[+0.006, +0.121]** | **[+0.007, +0.122]** |
| **ΔMRR** | **+0.064** | **[+0.027, +0.102]** | **[+0.024, +0.110]** |

One-sided bootstrap mass at or below zero: **p ≈ 0.017** (query) / **0.016**
(repo-cluster) for ΔR@1; the repo-cluster delta is positive in **98.4 %** of
resamples. Both intervals exclude zero on both metrics. **ΔMRR is the cleaner signal**
— its lower bound (~+0.024) sits well clear of zero, whereas ΔR@1's lower bound
(~+0.006) is thin, consistent with R11A's weakest split (ΔR@1 +0.027).

## 2. Where the win comes from — broad, but not uniform

Per held-out repo (R@1), sorted by delta:

| Repo | n | frozen R@1 | LoRA R@1 | ΔR@1 |
|---|---|---|---|---|
| pydantic/pydantic | 12 | 0.333 | 0.500 | **+0.167** |
| python-pillow/Pillow | 12 | 0.583 | 0.750 | **+0.167** |
| python-poetry/poetry | 31 | 0.516 | 0.645 | **+0.129** |
| tox-dev/tox | 17 | 0.706 | 0.824 | **+0.118** |
| pytest-dev/pytest | 45 | 0.644 | 0.711 | +0.067 |
| psf/requests | 4 | 0.750 | 0.750 | 0.000 |
| scrapy/scrapy | 35 | 0.486 | 0.457 | **−0.029** |
| python-attrs/attrs | 18 | 0.833 | 0.778 | **−0.056** |

**5 of 8 repos improve, 1 is flat, 2 regress slightly.** The win is not carried by a
single repo, but it is not universal either — `scrapy` and `attrs` regress by a
handful of queries each. A paired McNemar/sign test on the per-query rank-1 flips:
**+18 queries gained, −7 lost** (net **+11**), two-sided sign-test **p ≈ 0.043**. So
the fine-tune moves ~25 of 174 queries at rank 1, net clearly positive, and the
significance is real but not overwhelming at this scale.

## Honest read & what it motivates

- **The headline survives a proper CI.** On the default split the LoRA delta is
  positive with 95 % intervals that exclude zero under both query and repo-cluster
  resampling, and MRR more decisively than R@1. Combined with R11A's all-5-splits
  result, the Track-A win is robust to (a) which repos are held out, (b) query
  resampling, and (c) repo-level correlation.
- **It is real but thin at pilot scale**, and **uneven across repos** — exactly the
  profile expected of a +0.06 effect on 174 queries. This is *why* the program's next
  unlock is scale/density: the Tier-2 ΔR@1 +0.114 (R16C) is the same effect with more
  contrast pairs, and its own CI is the natural GPU-scale follow-up (Tier-2 LoRA
  caches are gitignored and need torch to regenerate, so its bootstrap is out of scope
  for this CPU wave).
- **Nothing here is a new claim** — it sharpens an existing one. The point estimate is
  the committed card; this wave adds the error bars and the per-repo story the single
  number could not show.

All results are **exploratory** and pilot-scale; the committed experiment cards remain
the source of truth for the point estimates this wave annotates.
