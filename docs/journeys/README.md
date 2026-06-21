# User Journeys

End-to-end, copy-pasteable recipes for working in this lab. Each journey uses
**only commands that exist today** and shows the real output you should see on a
clean checkout.

Run every command from the repository root, after installing the package:

```bash
pip install -e ".[dev]"
```

| # | Journey | What you walk through |
|---|---|---|
| 01 | [Fix an issue with relational retrieval](01-fix-an-issue-with-relational-retrieval.md) | The `datebox` timezone-bug worked example (issue #482) end-to-end: build the fixture, validate, run the benchmark, read the retrieved candidates and interpret the metrics. |
| 02 | [Build a dataset](02-build-a-dataset.md) | Intake → write a source card → add records and relation edges with provenance → `relsdlc validate data`. |
| 03 | [Run the benchmark](03-run-the-benchmark.md) | The four benchmark tasks, `relsdlc bench`, reading Recall@K / MRR / hard-negative accuracy, the leakage guard, and recording an experiment card. |
| 04 | [Toward relation-aware models](04-toward-relation-aware-models.md) | The conceptual recipe from the vanilla baseline to a relation-aware model, and the baseline-vs-relation-aware ablation on a frozen split with hard negatives. |

## The boring-gates philosophy

The whole point of this lab is that **claims are mechanically checkable**. A
result you cannot replay is not a result. Three gates must stay green and
reproducible on every change:

- **`relsdlc validate data`** — schema validity, provenance completeness,
  referential integrity (every edge endpoint and benchmark candidate resolves to
  a real record), and the temporal-leakage guard. Exit code is non-zero on any
  error, so it doubles as a CI gate.
- **`relsdlc bench`** — runs the retrieval benchmark over the synthetic fixture
  and prints Recall@K / MRR / hard-negative accuracy per task. It exits non-zero
  if the leakage guard fires.
- **`pytest -q`** — regression tests for the schemas, validation, metrics, and
  baseline embedder.

"Boring" is the goal. These checks should be fast, deterministic, and run from a
clean checkout with no network. The synthetic `datebox` fixture (CC0, original)
exists precisely so the entire pipeline runs offline. When a gate is boring, the
interesting work — relation-aware models — can be trusted, because the floor it
must beat is measured the same way every time.

A few invariants that keep the gates honest:

1. **Every record and edge carries provenance** — source URL, retrieval time,
   license, and a content hash. No provenance, no merge.
2. **Every claim is an experiment card** — a single synthetic query is labeled
   `exploratory`; release-quality claims need a real public dataset and a frozen
   split.
3. **Splits are frozen and temporal.** A query's `as_of` time excludes any
   candidate that only becomes valid later; the benchmark reports such cases as
   leakage and exits non-zero.
4. **Reproducibility lives in Git.** Cards, schemas, small fixtures, and rebuild
   scripts are committed; local planning files, run state, caches, large raw
   data, and credentials stay out of public history (see
   [docs/operating-boundary.md](../operating-boundary.md)).
