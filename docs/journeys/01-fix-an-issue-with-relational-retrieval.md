# Journey 01 — Fix an issue with relational retrieval

This walks the position paper's worked example end-to-end, using only commands
that exist today. The scenario is the synthetic `datebox` timezone bug,
**issue #482**:

> *Date filter returns incorrect results when timezone is UTC-3.*

We want to answer the retrieval question an SDLC agent actually faces: **given
this issue, which pull request fixes it?** — and then watch the same relational
graph answer the neighboring questions (which file is implicated by the failing
log, which test exercises the diff).

> The vanilla embedding in this repo is the *floor* a relation-aware model must
> beat. On this tiny fixture the floor already scores well; the interesting gap
> shows up on the harder `log_to_likely_file` task (see the end of this journey)
> and, later, on a real public dataset.

## 0. Setup

From the repository root, on a clean checkout:

```bash
pip install -e ".[dev]"
```

## 1. Build the synthetic fixture

The `datebox` fixture is generated, not hand-maintained, so it is reproducible
from a clean checkout with no network:

```bash
python data/fixtures/build_fixtures.py
```

This (re)writes the records, relation edges, benchmark queries, and example
cards under `data/`. The fixture mirrors the issue #482 worked example: 15
records (issues, PRs, a diff, files, symbols, tests, a CI log), 10 relation
edges (`fixes`, `modifies`, `covers`, `caused_by`), and three benchmark queries.

## 2. Validate the data (the first boring gate)

```bash
relsdlc validate data
```

Expected output:

```text
validated 15 records, 10 edges, 3 cards, 3 benchmark queries: 0 error(s), 0 warning(s)
```

`validate` checks schema validity, provenance completeness, referential
integrity (every edge endpoint and every benchmark candidate resolves to a real
record), and the temporal-leakage guard. Exit code `0` means the data is sound
enough to benchmark.

## 3. Run the retrieval benchmark

```bash
relsdlc bench
```

Expected output:

```text
diff_to_affected_test: n=1 R@1=1.000 R@5=1.000 R@10=1.000 MRR=1.000 HardNegAcc=1.000
issue_to_fixing_pr: n=1 R@1=1.000 R@5=1.000 R@10=1.000 MRR=1.000 HardNegAcc=1.000
log_to_likely_file: n=1 R@1=0.500 R@5=1.000 R@10=1.000 MRR=1.000 HardNegAcc=1.000
```

The line that answers our question is **`issue_to_fixing_pr`**.

## 4. Read the retrieved candidates

The `issue_to_fixing_pr` query lives in
[`data/fixtures/benchmark/issue_to_fixing_pr.jsonl`](../../data/fixtures/benchmark/issue_to_fixing_pr.jsonl):

```json
{
  "query_id": "q-issue-482",
  "task": "issue_to_fixing_pr",
  "query_record": "issue:482",
  "candidates": ["pr:512", "pr:140", "pr:300"],
  "relevant": ["pr:512"],
  "hard_negatives": ["pr:140"],
  "as_of": "2024-02-01T00:00:00Z"
}
```

So the benchmark hands the system three candidate PRs and asks it to rank them:

| Candidate | What it is | Role |
|---|---|---|
| `pr:512` | "Fix UTC-3 timezone normalization in date filter" | **relevant** (the true fix) |
| `pr:140` | "Fix currency report column formatting" | **hard negative** — also a "Fix …" PR, overlapping vocabulary, wrong subsystem |
| `pr:300` | "Add pagination to result listing endpoint" | easy negative |

The retrieval system embeds the issue text and each candidate's text with the
dependency-light baseline embedder (hashed bag-of-words + cosine, in
[`src/relsdlc/baseline.py`](../../src/relsdlc/baseline.py)), then ranks by
similarity. To see the per-query numbers as structured data:

```bash
relsdlc bench --task issue_to_fixing_pr --json
```

Expected output:

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

## 5. Interpret the result

- **`R@1 = 1.000`** — the true fixing PR (`pr:512`) is ranked first. The system
  picks the right PR, not the plausible-but-wrong "Fix currency report
  formatting" PR.
- **`MRR = 1.000`** — the first relevant item is at rank 1 (reciprocal rank 1/1).
- **`HardNegAcc = 1.000`** — the relevant PR outranks the designated hard
  negative (`pr:140`). This is the metric that matters: generic similarity finds
  easy positives, so a model only earns credit when it beats the *near-miss*.
- **`leakage: []`** — with `as_of = 2024-02-01`, no candidate or positive became
  valid after the query time, so the temporal guard found nothing to exclude.

In plain terms: starting from the issue, the relational lab retrieved the correct
fixing PR and correctly preferred it over a same-vocabulary distractor. That is
the unit of work — *issue → likely fixing PR* — that an SDLC agent would use to
jump straight to the relevant change instead of re-reading the whole repo.

## 6. Where the floor cracks

Look again at the `log_to_likely_file` line: **`R@1 = 0.500`**. That query
(`q-log-512`, in
[`data/fixtures/benchmark/log_to_likely_file.jsonl`](../../data/fixtures/benchmark/log_to_likely_file.jsonl))
starts from the failing CI log and asks for the *two* implicated files
(`file:date_filter.py` and `file:parse_tz.py`), with `file:report.py` as the
hard negative. The vanilla embedder finds both relevant files within the top 5
(`R@5 = 1.000`) but only ranks one of them first (`R@1 = 0.500`).

This is exactly the gap a **relation-aware** model is meant to close: the failing
log is connected to the fix by `caused_by` and `modifies` edges
(`ci_log:512-fail → caused_by → diff:512 → modifies → {symbol:normalize_range,
symbol:parse_tz}`), and following those relations — not just text similarity —
should surface both files at the top. Journey 04 sketches that recipe.

## 7. Re-run the gates anytime

```bash
relsdlc validate data    # 0 errors, 0 warnings
relsdlc bench            # numbers above; exits non-zero only on leakage
pytest -q                # schema / validation / metrics / baseline regressions
```

If all three are green and reproduce the numbers above, the worked example is
intact. See [Journey 03](03-run-the-benchmark.md) for the full benchmark and how
to record a result as an experiment card.
