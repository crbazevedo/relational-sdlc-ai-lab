# Cracking the diff→test ceiling: real co-change density (Track B/C, R17b)

**Status: exploratory, real public data (one-time live snapshot).** R16E found
`diff_to_affected_test` is **structure-bound** on the pilot: once the leakage guard
removes the gold `(query-PR, test)` edge, **46.9 %** of gold test nodes are degree-0
(only the gold PR modifies them in the pilot graph), capping reachable recall at
**59.8 %** — flat across every `(alpha, hops)` cell. R16E argued the limiter is
co-change *density*, an artefact of ingesting few PRs per repo, **not** the
aggregation method. This wave tests that argument against ground truth.

## What it measures

For every distinct gold test file on the held-out split (110 files across 8 repos),
it asks GitHub how many distinct commits in the repo's real history touch that exact
path — one `commits?path=` call per file (capped at 100), a polite ~110-call
one-time snapshot. A gold test is **reachable** if it is touched by ≥ 1 change *other
than the gold PR's* (we exclude commits on the gold PR's date, mirroring the leakage
guard); we also report the looser "≥ 2 distinct commits" view. It does **not** mutate
the frozen pilot graph — it writes a separate snapshot
([`data/pilot/graph/diff2test-cochange.json`](../data/pilot/graph/diff2test-cochange.json))
and a deterministic recompute
([`data/pilot/diff2test-density-results.json`](../data/pilot/diff2test-density-results.json)),
so the offline replay (and CI) needs no token.
[`densify_diff2test.py`](../data/pilot/densify_diff2test.py).

## Result — the ceiling was an ingest artefact

| Co-change source | gold tests isolated | reachable ceiling |
|---|---|---|
| R16E pilot modifies graph | 46.9 % (75/160) | **59.8 %** |
| **Real commit history (non-gold, gold-date excluded)** | **4.4 %** | **96.4 %** |
| Real history, looser "≥ 2 commits" | 7.5 % | — |

The gold test files are, in reality, **heavily co-changed**: **median 35 commits per
file** touch each gold test (max 100; 25 files hit the 100-commit page cap). Only
**12 of 110** are genuinely touched by ≤ 1 change. So the pilot's 46.9 % isolation
was almost entirely because the snapshot ingested only a handful of PRs per repo —
**exactly the density artefact R16E named.** With real history the structural ceiling
rises from **59.8 % to 96.4 %** (**+36.6 points**): a sufficiently dense `modifies`
graph removes the structural blocker for ~96 % of held-out queries.

## Honest read & scope

- **The R16E diagnosis is confirmed, decisively.** diff→test is *not* method-bound;
  it was data-bound, and the data limit is an ingest-depth artefact, not an
  intrinsic sparsity of co-change. A denser ingest lifts the reachable ceiling to the
  mid-90s.
- **This is a ceiling, not a retrieval score.** It measures *structural reachability*
  (can aggregation place the test at all?), not R@1. Turning the headroom into actual
  retrieval needs the denser graph **plus embeddings for the new PR nodes** — and
  re-embedding is torch-gated, so the end-to-end diff→test re-eval is a GPU/torch
  follow-up (Track D scale). What this wave establishes is that the follow-up is now
  *worth running*: the ceiling no longer blocks it.
- **Granularity caveat.** Co-change is measured at commit level (commits touching the
  path), as a faithful proxy for a denser PR-level `modifies` graph; the gold PR's own
  change is date-excluded. The ~4 % residual (12 rarely-touched or renamed/moved
  files) is real sparsity that density does not fix.
- **One-time live snapshot.** Like the Tier-2 build, the fetch touches the live API
  and is not reproducible from scratch; the committed co-change snapshot makes the
  recompute deterministic and CI-replayable.

This converts R16E's quantified blocker into a quantified *unlock*: the path to a
working diff→test is **denser co-change at scale**, now measured to be sufficient in
principle (59.8 % → 96.4 % reachable). All results are **exploratory**; the committed
snapshot and recompute are the source of truth.
