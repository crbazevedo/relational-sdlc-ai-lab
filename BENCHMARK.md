# Relational SDLC Retrieval Benchmark — leaderboard

A **frozen, de-referenced, cross-repo** benchmark for relational retrieval over public
GitHub SDLC records. The deliverable of this lab is not a single model but *this
benchmark plus honest baselines* — the thing that makes the relational thesis
falsifiable. Tasks, splits, and metrics are defined in
[docs/benchmark-definition.md](docs/benchmark-definition.md); every number below is
backed by a committed [experiment card](data/cards/examples/) and reproduces from the
committed caches with **numpy only** (no GPU, no network).

> **Status: exploratory, pilot/Tier-2 scale.** These are signals to confirm at larger
> scale, not release-grade SOTA claims. Numbers are honest (wins *and* negatives), and
> the experiment cards are authoritative where this table and they disagree.

## Method invariants (what makes a number trustworthy)

- **Links are labels, not features** — explicit references (`#N`, URLs, SHAs) are
  scrubbed from inputs; a relation a regex can recover is never a test.
- **Cross-repo splits** — train repos are disjoint from test repos; a win must
  generalize to unseen repositories.
- **The bar is `embedder-cosine`** — the relational contribution is the *delta* over a
  frozen pretrained embedder on the same split.
- **Leakage guards** — the gold edge is removed before any graph aggregation; for
  diff→test the released number additionally enforces a temporal `as_of` cut (a test's
  evidence may only come from changes *before* the query).
- **R@1 and MRR are the discriminating metrics** (small candidate pools make R@5/R@10
  near-ceiling).

---

## Task 1 — `issue_to_fixing_pr`

Given an issue, retrieve the PR(s) that fixed it.

### Pilot (20 repos, de-referenced cross-repo, 174 held-out queries)

| System | R@1 | R@5 | MRR | Card |
|---|---|---|---|---|
| vanilla token cosine | 0.391 | 0.724 | 0.544 | `gh-xrepo-vanilla-v0` |
| unsupervised IDF cosine | 0.460 | 0.828 | 0.624 | `gh-xrepo-idf-v0` |
| **embedder-cosine** (frozen MiniLM-L6) | **0.592** | 0.920 | 0.728 | `gh-embed-cosine` |
| **+ relational LoRA** (relation InfoNCE) | **0.655** | 0.994 | 0.791 | `gh-finetune-cosine` |
| + LoRA + parameter-free graph lift | **0.690** | — | 0.820 | `gh-graphsweep-issue2pr-lora-h1-best` |

*Robustness:* the LoRA win is positive on all 5 held-out-repo splits (ΔR@1 +0.061±0.021),
with within-split 95% CIs that exclude zero (`bootstrap-ci-results.json`).

### Dense Tier-2 (78 repos, ~35 queries/repo, 1,171 held-out queries)

| System | R@1 | MRR | Card |
|---|---|---|---|
| unsupervised IDF cosine | 0.389 | 0.542 | `gh-tier2-idf-v0` |
| **embedder-cosine** (frozen) | **0.515** | 0.655 | `gh-tier2-vanilla-v0` |
| + relational LoRA | **0.629** | 0.757 | `gh-tier2-lora-v0` |
| **+ LoRA + same-repo hard negatives** | **0.66** | 0.78 | `gh-tier2-hardneg-multiseed-v0` |

*The LoRA win grows with density* (ΔR@1 +0.07 pilot → +0.114 Tier-2), and **negative
hardness, not quantity, is the lever** — confirmed multi-seed (paired gap +0.020).

---

## Task 2 — `diff_to_affected_test`

Given a PR/diff, retrieve the test file(s) it affects. Candidates are test-file nodes
with **no text embedding** — a test's score comes purely from *co-change structure*
(the PR's similarity to the other PRs that historically modified that test), with the
gold edge removed and a temporal `as_of` cut applied.

| System | R@1 | R@5 | MRR | Card |
|---|---|---|---|---|
| embedder-cosine (no graph) | 0.009 | 0.175 | 0.155 | — |
| graph-aggregation, sparse pilot graph | 0.009 | 0.175 | 0.155 | `gh-graphsweep-diff2test-lora-h2` |
| **graph-aggregation, densified graph + `as_of`** | **0.429** | 0.759 | 0.574 | `gh-diff2test-dense-v0` |

A task text-cosine *cannot* do (0.009) reaches **R@1 0.429** once the `modifies` graph
is dense enough — the relational thesis in its purest form. The blocker was always
co-change **density**, not the method: pilot sparsity isolated 47% of gold tests; a
denser graph (live-ingested) raises reachability to ~86% under the honest `as_of` cut.

*Honest decomposition (`diff2test-strict-results.json`):* on the dense graph, using the
PR's own embedding as the query (vs. an α-blend) is worth +0.195 R@1; enforcing the
temporal `as_of` cut costs +0.125 (future modifiers were leaking). The released number
applies both — clean query, no future leakage.

---

## Reproduce

```bash
pip install -e ".[dev]"            # numpy + jsonschema + pytest
relsdlc validate data              # schema + provenance + referential integrity + leakage gates (0 errors)
PYTHONPATH=src python data/pilot/run_crossrepo_ablation.py     # Task 1 bag-of-tokens baselines
PYTHONPATH=src python data/pilot/graph/run_diff2test_strict.py # Task 2 release-honest number
pytest -q                          # pins every committed result
```

Embedding/training steps (LoRA, dense-graph PR vectors) need the `[embed]` extra
(torch); the resulting caches are committed where small, so the numpy evaluation paths
above replay every number without a GPU.

## Honest limitations

- Exploratory, pilot/Tier-2 scale; single cross-repo split per task (Task 1 also has a
  5-split CI). Not release-grade SOTA.
- Tier-2 LoRA numbers were trained on an Apple M5 GPU (MPS), which is **not
  bit-deterministic** (~±0.008 R@1 run-to-run); the hardness gain is confirmed
  multi-seed.
- The two other defined tasks (`log_to_likely_file`, `pr_to_missing_test`) are specified
  in the benchmark definition but not yet populated with frozen splits.
- `diff_to_affected_test` density is still below the 96.4% commit-level reachable
  ceiling — more PRs / commit-level edges would push R@1 higher.
