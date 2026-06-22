# Full-text dataset: de-truncating the bodies and measuring the lift

**Status: exploratory, real public data.** This closes a loose end from the pilot:
issue/PR bodies were truncated to **500 chars** — below the embedder's window
(`max_length=256` ≈ 1000+ chars) and below the bag-of-tokens reach (which
tokenizes the whole string). The worry was that every prior number *understated*
the systems, because the model never saw chars 500→1500+. R14 re-ingests the same
20 repos with full text and **measures** what truncation actually cost.

The honest answer, measured on a clean paired control: **truncation did not cost
us — it slightly helped.** The discriminative signal for issue→fixing-PR lives in
the first ~500 chars (title + opening description); the rest is mostly shared
boilerplate that *hurts* this retrieval task. Details below.

## What changed

| | Pilot (`data/pilot/`) | Full (`data/full/`) |
|---|---|---|
| Body char cap | **500** | **8000** (4000 fallback if >20 MB; not triggered — 5.1 MB) |
| Embedder window | `max_length=256` | `max_length=512` |
| Redistribution posture | `metadata_only` | **`full_text`** |
| Id namespace | `gh:owner/repo:…` | `gh-full:owner/repo:…` |
| Records / fixes / queries | 2087 / 356 / 356 | 2308 / 509 / 509 |
| Median body length | **500** (capped) | **914** (mean 1618, max 8000) |
| Bodies > 500 chars | 0 (all capped) | 1653 / 2308 (72%) |

Everything else is identical to the pilot: the same 20 permissively-licensed
Python-ecosystem repos, the same schema / provenance / `fixes`-edge mining, the
same temporal split policy, the same de-referenced cross-repo benchmark.

- Build: [`data/full/build_full.py`](../data/full/build_full.py) (live GitHub REST).
- Embed: [`data/full/embed_full.py`](../data/full/embed_full.py) (the `[embed]` extra).
- Ablation: [`data/full/run_full_ablation.py`](../data/full/run_full_ablation.py) (numpy only).
- Cards: [`data/cards/examples/gh-full-*.json`](../data/cards/examples/).

### Why `gh-full:` ids

`relsdlc validate` dedups record ids across the *whole* `data/` tree. The full
dataset re-ingests the same repos as the frozen pilot, so plain `gh:owner/repo:…`
ids would collide. The `gh-full:` prefix keeps both frozen versions coexisting and
passing validation, while `owner/repo` stays the second `:`-field so the cross-repo
split parses it unchanged.

## Redistribution posture: `metadata_only` → `full_text`

The pilot only redistributed metadata + truncated text. The full dataset
redistributes the full issue/PR prose, so its source/dataset cards declare
`redistribution: "full_text"`.

**Rationale.** The content is *public* issue/PR text from permissively-licensed
repos (MIT / BSD-3-Clause / Apache-2.0 / PSF — one repo's SPDX id resolved as
`unknown`), kept for **research use**, with provenance back to the source URLs on
every record. This follows the **GH-Archive precedent**: public GitHub event and
issue text is routinely archived and redistributed for research. The 8000-char cap
captures ~all real human-written descriptions while bounding pathological
machine-pasted logs/stack-traces.

## Result (same de-referenced, cross-repo split: 10 train repos, 8 held-out test repos)

Two comparisons. The **paired control** is the clean A/B — the same records and
the same benchmark, re-run with bodies char-capped to 500 and the embedder at
`max_length=256`, so **truncation is the only variable** and the lift is causal.
The **cross-snapshot** row compares against the frozen truncated *pilot*; it is
only suggestive, because it also changes the snapshot (a different fetch with a
different issue/PR set, and a different benchmark).

### Paired control — same 271 test queries, truncation the only variable

| System | full R@1 | trunc R@1 | **ΔR@1** | full R@5 | trunc R@5 | full MRR | trunc MRR |
|---|---|---|---|---|---|---|---|
| vanilla-tf-cosine | 0.402 | 0.550 | **−0.148** | 0.734 | 0.893 | 0.552 | 0.695 |
| idf-cosine | 0.509 | 0.646 | **−0.137** | 0.867 | 0.952 | 0.667 | 0.773 |
| relation-metric | 0.513 | 0.605 | **−0.092** | 0.863 | 0.926 | 0.656 | 0.735 |
| **embedder-cosine (512 vs 256)** | 0.546 | 0.694 | **−0.148** | 0.900 | 0.970 | 0.696 | 0.810 |

**Every system is worse on full text.** The 500-char arm beats the 8000-char arm
by 0.09–0.15 R@1 across the board.

### Cross-snapshot — vs the frozen truncated pilot (conflates snapshot drift)

| System | full R@1 | trunc-pilot R@1 | ΔR@1 |
|---|---|---|---|
| vanilla-tf-cosine | 0.402 | 0.391 | +0.011 |
| idf-cosine | 0.509 | 0.460 | +0.050 |
| relation-metric | 0.513 | 0.454 | +0.059 |
| **embedder-cosine** | 0.546 | 0.592 | −0.046 |

The bag-of-tokens systems *appear* to gain here, but that gain is mostly snapshot
drift (different records, an easier query set): the paired control, which holds the
records fixed, shows the opposite. Trust the paired control.

(The truncated-pilot embedder cosine — the documented `R@1 0.592` from
[ablation-embed.md](ablation-embed.md) — is itself on the *easier* pilot snapshot;
the full-text embedder on the harder full snapshot lands at 0.546.)

## The honest verdict: truncation did not cost us

The premise of this wave — "the current results likely UNDERSTATE everything" — is
**refuted** by the paired control. De-truncating to 8000 chars and widening the
window to 512 made retrieval **worse**, not better, for every system.

Why, mechanistically:

1. **The signal is front-loaded.** For issue→fixing-PR, the discriminating content
   is the title and the first paragraph — what the artifact *is about*. That fits
   comfortably in 500 chars.
2. **The tail is shared boilerplate.** Beyond ~500 chars, bodies fill with
   issue/PR-template scaffolding, "Steps to reproduce", environment dumps, stack
   traces, and checklists. Those tokens are *shared* across many same-repo
   candidates, so they raise similarity to the **hard negatives** (which are
   selected as same-repo PRs with high token overlap) more than to the true PR.
3. **Mean-pooling dilutes.** The frozen embedder mean-pools over tokens; pooling
   over more boilerplate dilutes the discriminative head of the text. A wider
   window let *more* noise in, which is why the embedder dropped the most (−0.148).

So this is the "small lift / the window caps it" outcome the wave anticipated —
except sharper: it is a **negative** lift. The cap was not throwing away signal; it
was, accidentally, a cheap and effective denoiser.

### What this means going forward

- **Keep a tight body window for this task.** 500 chars (or the title + first
  paragraph) is a strong, cheap default for issue→fixing-PR retrieval. The pilot's
  truncation was a feature, not a bug.
- **The full text is not useless — it is just not useful *as undifferentiated
  pooled tokens*.** Extracting structure from the long tail (code blocks, error
  signatures, changed-symbol mentions) and weighting it, rather than averaging it
  in, is the way to make the extra text pay. That is a representation problem, and
  it aligns with [ablation-embed.md](ablation-embed.md)'s conclusion: the
  relational contribution belongs *inside the representation* (fine-tuning, or
  graph structure), not in pooling more raw text.

## Reproduce

```
# 1) (live, one-time) re-ingest the 20 repos with full bodies
GITHUB_TOKEN=$(gh auth token) python data/full/build_full.py

# 2) (needs the [embed] extra) cache the 512- and paired-256 embeddings
python data/full/embed_full.py

# 3) numpy-only: the cross-repo ablation + the full-vs-truncated lift
python data/full/run_full_ablation.py     # writes data/full/full-results.json
```

The ablation and the tests ([`tests/test_full.py`](../tests/test_full.py)) run on
the committed snapshot + embedding cache with numpy alone — no torch, no network.
