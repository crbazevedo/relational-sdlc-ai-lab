# Journey 03 — Run the benchmark

This journey runs the retrieval benchmark, reads the metrics, explains the
leakage guard, and records a result as an experiment card. The full benchmark
contract is in [docs/benchmark-definition.md](../benchmark-definition.md); this
is the runnable walkthrough.

## The four tasks

The benchmark fixes four SDLC retrieval tasks. Each query carries an explicit
**candidate pool** so retrieval is scored against a fixed set:

| Task id | Query record | Candidate pool | Relevant (positive) | Relation used to label |
|---|---|---|---|---|
| `issue_to_fixing_pr` | an `issue` | `pull_request` records | the PR(s) that fixed it | `fixes` |
| `diff_to_affected_test` | a `diff` | `test` records | tests exercising modified symbols | `modifies` ∘ `covers` |
| `log_to_likely_file` | a failing `ci_log` | `file` records | files implicated in the failure | `caused_by` ∘ `modifies` |
| `pr_to_missing_test` | a `pull_request` | `test` records | tests that should have existed | review / coverage gap |

The synthetic `datebox` fixture ships runnable queries for the first three;
`pr_to_missing_test` is part of the benchmark definition and is exercised once a
dataset provides the queries.

## 1. Run all tasks

```bash
relsdlc bench
```

Expected output:

```text
diff_to_affected_test: n=1 R@1=1.000 R@5=1.000 R@10=1.000 MRR=1.000 HardNegAcc=1.000
issue_to_fixing_pr: n=1 R@1=1.000 R@5=1.000 R@10=1.000 MRR=1.000 HardNegAcc=1.000
log_to_likely_file: n=1 R@1=0.500 R@5=1.000 R@10=1.000 MRR=1.000 HardNegAcc=1.000
```

To restrict to one task:

```bash
relsdlc bench --task issue_to_fixing_pr
```

To emit a structured report (use this when recording a result):

```bash
relsdlc bench --task issue_to_fixing_pr --json
```

```json
{
  "tasks": {
    "issue_to_fixing_pr": {
      "n_queries": 1,
      "recall_at_k": {
        "1": 1.0,
        "5": 1.0,
        "10": 1.0
      },
      "mrr": 1.0,
      "hard_negative_accuracy": 1.0
    }
  },
  "leakage": []
}
```

## 2. Read the metrics

For each query the system returns a ranked candidate list; the metrics aggregate
over the frozen evaluation set:

- **Recall@K** — fraction of relevant items found in the top *K* (`K ∈ {1, 5,
  10}`). `R@1 = 1.000` means the top-ranked candidate is relevant.
- **MRR** — mean reciprocal rank of the first relevant item. Rank 1 → `1.000`,
  rank 2 → `0.500`, and so on.
- **Hard-negative accuracy** — fraction of queries where a relevant item outranks
  *every* designated hard negative.

**Hard negatives are the point.** Generic similarity already finds easy
positives; a model only earns credit when it beats the near-miss — the wrong file
in the same package, the wrong test in the same suite, a plausible-but-unrelated
PR. Without hard negatives, Recall@K flatters every model. In the fixture,
`issue_to_fixing_pr`'s hard negative is `pr:140` ("Fix currency report column
formatting") — same "Fix …" framing, wrong subsystem.

The current `log_to_likely_file` line (`R@1 = 0.500`) is the honest one: the
vanilla baseline finds both implicated files within the top 5 but does not rank
both first. That residual is the headroom a relation-aware model targets
([Journey 04](04-toward-relation-aware-models.md)).

## 3. The leakage guard

Every record and edge carries `valid_from`. A query may carry an `as_of`
timestamp; when it does, `relsdlc bench` excludes any candidate or positive that
only becomes valid **after** the query time and reports it as **leakage**.

- If `report["leakage"]` is non-empty, `relsdlc bench` prints the violations to
  stderr and **exits non-zero**, so leakage cannot silently inflate scores.
- In the fixture, every query uses `as_of = 2024-02-01T00:00:00Z`, which is after
  all record `valid_from` times, so `leakage` is `[]`.

This is why splits must be **frozen and temporal-by-commit-date** for real data:
random splits leak future information into training/evaluation. The split policy
is recorded in the dataset card (see
[Journey 02](02-build-a-dataset.md#5-write-a-dataset-card)).

## 4. Baselines before training

No relation-aware model is claimed to "win" until a baseline is measured on the
**same frozen split**. The order is:

1. `baseline-hashing-tfidf` — the dependency-light floor in
   [`src/relsdlc/baseline.py`](../../src/relsdlc/baseline.py) (no network, runs
   in CI). This is what `relsdlc bench` runs by default.
2. An off-the-shelf text/code embedding model (optional dependency).
3. The relation-aware bi-encoder + relation-head reranker — the system that must
   beat both.

## 5. Record the result as an experiment card

Every benchmark claim becomes an **experiment card**. Start from
[`data/cards/templates/experiment-card.template.json`](../../data/cards/templates/experiment-card.template.json)
and fill in the run. The committed baseline example is
[`data/cards/examples/baseline-hashing-tfidf-issue2pr-v0.experiment-card.json`](../../data/cards/examples/baseline-hashing-tfidf-issue2pr-v0.experiment-card.json):

```json
{
  "card_type": "experiment",
  "id": "exp:baseline-hashing-tfidf-issue2pr-v0",
  "name": "Baseline hashing-TF-IDF — issue_to_fixing_pr (fixture)",
  "created_at": "2024-02-01T00:00:00Z",
  "hypothesis": "Vanilla text embedding retrieves the fixing PR for an issue; establishes the floor a relation-aware model must beat.",
  "task": "issue_to_fixing_pr",
  "dataset_version": "ds:datebox-fixture-v0",
  "code_version": "relsdlc-0.1.0",
  "seed": 0,
  "command": "relsdlc bench --task issue_to_fixing_pr",
  "system": "baseline-hashing-tfidf",
  "runtime_class": "cpu",
  "metrics": {
    "recall_at_k": {"1": 1.0, "5": 1.0, "10": 1.0},
    "mrr": 1.0,
    "hard_negative_accuracy": 1.0,
    "n_queries": 1
  },
  "baseline_comparison": "none",
  "error_slices": [],
  "leakage_checks": ["as_of=2024-02-01; no candidate/positive valid after query time"],
  "known_limitations": ["Single synthetic query; not statistically meaningful. Exploratory only."],
  "exploratory": true
}
```

Notice it is labeled **`exploratory: true`**. A single synthetic query is a smoke
test of the harness, not evidence of model quality. Release-quality claims
require the public dataset (roadmap P1) and a frozen split. When you record a
relation-aware run, set `baseline_comparison` to the baseline card's `id` so the
comparison is explicit and replayable.

## 6. Keep the gates green

```bash
relsdlc validate data    # 0 errors, 0 warnings
relsdlc bench            # numbers above; exits non-zero on leakage
pytest -q                # metrics / baseline / validation regressions
```

When all three reproduce the numbers above, the benchmark is intact and your
experiment card is a replayable record.
