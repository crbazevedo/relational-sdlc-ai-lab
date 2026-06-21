# Benchmark Definition

The first jewel is a measurable claim, not an autonomous agent:

> Relation-trained embeddings retrieve the right files / tests / PRs better than
> off-the-shelf text/code embeddings on frozen public SDLC relation tasks.

This document fixes the tasks, the candidate construction, the metrics, and the
anti-leakage rules so any run is replayable and comparable. Each query is a
[`benchmark-query`](../schemas/benchmark-query.schema.json) object; each result
is recorded in an [`experiment-card`](../schemas/experiment-card.schema.json).

## Tasks

| Task id | Query record | Candidate pool | Relevant (positive) | Relation used to label |
|---|---|---|---|---|
| `issue_to_fixing_pr` | an `issue` | `pull_request` records | the PR(s) that fixed it | `fixes` |
| `diff_to_affected_test` | a `diff` | `test` records | tests exercising modified symbols | `modifies` ∘ `covers` |
| `log_to_likely_file` | a failing `ci_log` | `file` records | files implicated in the failure | `caused_by` ∘ `modifies` |
| `pr_to_missing_test` | a `pull_request` | `test` records | tests that should have existed | review / coverage gap |

A query carries an explicit **candidate pool** so retrieval is scored against a
fixed set, and an optional **`as_of`** timestamp that turns on the leakage check.

## Metrics

For each query the system returns a ranked candidate list. Aggregated over the
frozen evaluation set:

- **Recall@K** — fraction of relevant items found in the top *K* (K ∈ {1, 5, 10}).
- **MRR** — mean reciprocal rank of the first relevant item.
- **Hard-negative accuracy** — fraction of queries where a relevant item outranks
  every designated hard negative (the wrong file in the same package, the wrong
  test in the same suite, a plausible-but-unrelated PR).

Definitions live in [`src/relsdlc/metrics.py`](../src/relsdlc/metrics.py) and are
pinned by `tests/test_metrics.py`.

## Hard negatives

Hard negatives are the point of the benchmark — generic similarity already finds
easy positives. Each query should include negatives that are *near* the positive:
same package, same suite, same author, overlapping vocabulary. Without them,
Recall@K flatters every model.

## Frozen splits and temporal validity

- Splits are **frozen** and recorded in the [dataset card](../schemas/dataset-card.schema.json).
- The default split method for real data is **temporal-by-commit-date**: train on
  the past, evaluate on the future. Random splits leak.
- Every record/edge carries `valid_from`. A query's `as_of` time excludes any
  candidate or positive that only becomes valid later — `relsdlc bench` reports
  such cases as **leakage** and exits non-zero.

## Baselines before training

Per the research lifecycle, no relation-aware model is claimed to "win" until a
baseline is measured on the same frozen split:

1. `baseline-hashing-tfidf` — the dependency-light floor in
   [`src/relsdlc/baseline.py`](../src/relsdlc/baseline.py) (no network, runs in CI).
2. An off-the-shelf text/code embedding model (P2, optional dependency).
3. The relation-aware bi-encoder + relation-head reranker (P3) — the system that
   must beat both.

## Reproducing the fixture benchmark

```bash
python data/fixtures/build_fixtures.py     # regenerate the synthetic fixture + cards
relsdlc validate data                      # schema + provenance + leakage gates
relsdlc bench                              # Recall@K / MRR / hard-neg per task
```

The committed [example experiment card](../data/cards/examples/baseline-hashing-tfidf-issue2pr-v0.experiment-card.json)
is the baseline run on the synthetic fixture. It is labeled **exploratory**: a
single synthetic query is a smoke test of the harness, not evidence of model
quality. Release-quality claims require the P1 public dataset and a frozen split.
