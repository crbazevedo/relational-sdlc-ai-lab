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
- **The bar is a strong lexical baseline (BM25), not just `embedder-cosine`** — the
  relational contribution is the *delta* over BM25 **and** a frozen pretrained embedder
  on the same split. (R25 lesson: for the text-free diff→test task, `embedder-cosine` is
  *not* a sufficient bar — a BM25 over the candidate **paths** beats co-change structure;
  see Task 2.)
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
| **BM25** (lexical baseline) | **0.500** | 0.874 | 0.658 | `baselines-metrics-results.json` |
| **embedder-cosine** (frozen MiniLM-L6) | **0.592** | 0.920 | 0.728 | `gh-embed-cosine` |
| **+ relational LoRA** (relation InfoNCE) | **0.655** | 0.994 | 0.791 | `gh-finetune-cosine` |
| + LoRA + parameter-free graph lift | **0.690** | — | 0.820 | `gh-graphsweep-issue2pr-lora-h1-best` |

*Robustness:* the LoRA win is positive on all 5 held-out-repo splits (ΔR@1 +0.061±0.021),
with within-split 95% CIs that exclude zero (`bootstrap-ci-results.json`). *Power (R25):*
the headline-split LoRA delta (+0.063) is honest but **underpowered** — it sits below the
80%-power minimum detectable effect (0.080), achieved power ≈0.60 (`baselines-metrics-results.json`).
The full metric suite (R@1/5/10, MRR, nDCG, Hits) with query- and repo-cluster CIs is in
the same file.

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
with **no text *embedding*** — but they are **not text-free**: each carries a **path**
(`tests/test_root_model.py`), and a PR's prose shares vocabulary with the path of the
test it touches.

> **⚠️ R25 correction — a path-lexical baseline beats co-change structure here; the
> "text-free, only structure works" framing is refuted.** We previously reported
> `embedder-cosine 0.009` as the baseline, but that is near-zero only because a *sentence
> embedder* discards path tokens. The honest baseline is a **BM25 over the candidate
> paths**, and it wins on both corpora. Full analysis:
> [ablation-baselines-and-refutation.md](docs/ablation-baselines-and-refutation.md).

| System | Pilot R@1 (112 q) | corpus2 R@1 (905 q, same-repo negs) | Source |
|---|---|---|---|
| sentence embedder-cosine | 0.009 | 0.134 | `baselines-metrics-results.json` |
| co-change structure (graph-agg + `as_of`) | 0.429 (dense) / 0.009 (sparse) | 0.348 | `diff2test-strict-results.json` |
| **BM25 over test paths** | **0.536** | **0.609** | `corpus2-baselines-results.json` |
| **learned lexical reranker** (BM25 + path-overlap, LORO CV) | — | **0.707** | `corpus2-fusion-results.json` |

**BM25-over-paths beats co-change structure on both corpora**, including against corpus2's
harder same-repo negatives. A *learned* reranker over lexical features is the best system
(R@1 0.707); adding structure features to it **significantly hurts** (ΔR@1 −0.038, 95% CI
[−0.066, −0.009]), and naive fusion also hurts. So co-change structure is **neither
superior to nor complementary with** lexical retrieval on this task.

*What the structure work did establish (still true, just not the headline):* the co-change
graph reaches R@1 0.429 dense+`as_of` (`diff2test-strict-results.json`); that number is not
a coverage artefact (fair R@1 0.500 = 5.7× random, balanced gold/negative coverage,
disjoint train/test repos — `diff2test-audit-results.json`); and the structure signal
replicates on the independent TS/JS corpus (0.348, fair 5.5× random —
`corpus2-diff2test-results.json`). These are real properties of co-change geometry — a
simpler lexical baseline just solves the task better, so the structure is not *needed*.

*Honest residual:* on the ~⅓ of changes where BM25's top-1 is wrong, co-change structure
independently recovers ~31% of the affected tests — a fallback signal, not a system that
beats lexical (gating on it lowers overall R@1).

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
