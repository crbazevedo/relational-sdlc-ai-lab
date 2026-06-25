# Activating the second task: diff→test retrieval on a densified graph (Track B/C, R20)

**Status: exploratory, real public data; one-time live densification + M5 embedding.**
The program's relational thesis has, until now, been demonstrated on essentially
**one** task — `issue_to_fixing_pr`. The second task, `diff_to_affected_test`, has
been stuck at the floor since R11B: graph-aggregation R@1 **0.009**, because (R16E)
47% of gold test nodes are isolated after the leakage guard, and (R17b) that 59.8%
reachable ceiling is an **ingest-depth artefact** — real co-change touches each gold
test a median of 35 times (96.4% reachable). This wave converts that structural
unlock into an **actual retrieval result**, moving the thesis from one task to two.

## Method — three stages

1. **Densify the `modifies` graph (live).** For each of the 8 held-out test repos,
   fetch more merged PRs + their changed files and add (PR, test) `modifies` edges +
   the PR/test records (reusing the ingest mappers, so the output is schema- and
   provenance-clean). [`densify_modifies.py`](../data/pilot/graph/densify_modifies.py)
   added **982** (PR, test) edges and **510** modifying-PR nodes over the 8 test repos.
2. **Embed the new PR nodes (M5).** Same recipe as the pilot cache (MiniLM-L6
   mean-pooled + L2-norm, reference-scrubbed) so the new vectors share the pilot space.
   [`embed_dense_pr.py`](../data/pilot/graph/embed_dense_pr.py).
3. **Retrieve (numpy).** The **same R16E scorer + leakage guard**, run with original
   vs. original+dense edges so the only thing that changes is the graph density.
   [`run_diff2test_dense.py`](../data/pilot/graph/run_diff2test_dense.py); raw numbers
   in [`diff2test-dense-results.json`](../data/pilot/diff2test-dense-results.json).

The query is the (de-referenced) PR; candidates are test-file nodes, which have **no
text embedding** — each test node's feature is the mean of the embeddings of the PRs
that modify it, with the gold (query-PR, test) edge removed. So this is purely a
*structural* signal: a test is retrieved if the query PR resembles the *other* PRs
that touch it.

## Result — the second task works

| system (pilot cross-repo test split, 112 queries, α=0.5, 1 hop) | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| embedder-cosine (no graph; test nodes are zero vectors) | 0.009 | 0.175 | 0.705 | 0.155 |
| graph-aug, original edges (**reproduces R16E**) | 0.009 | 0.175 | 0.705 | 0.155 |
| **graph-aug, +dense edges (R20)** | **0.305** | **0.657** | **0.792** | **0.508** |

Reachability (a query has ≥1 relevant test with a non-gold modifier after the guard)
rises **59.8% → 89.3%** with the dense edges. R@1 goes **0.009 → 0.305 (≈34×)** and
MRR **0.155 → 0.508**. The R16E floor reproduces *exactly* (0.009), so the entire gain
is attributable to the densified graph — confirming R17b's prediction that the blocker
was **data density, not the method**.

## Honest read & what it motivates

- **diff→test is no longer dead.** A second SDLC retrieval task now shows a real,
  honest relational signal (R@1 0.305, MRR 0.508) where it was at chance — directly
  validating the program's "denser co-change unblocks it" diagnosis end to end
  (R16E quantified the blocker → R17b measured the ceiling → R20 hits a real number).
- **It is bounded by density, and there is headroom.** 89.3% reachable (vs the 96.4%
  commit-level ceiling) — the PR-level densification (recent merged PRs touching test
  files) captures less than full commit history, so more PRs per repo (or commit-level
  edges) would push R@1 higher. R@1 0.305 is a floor for this approach, not a ceiling.
- **The signal is structural, not lexical.** Test nodes carry no text; the result comes
  entirely from co-change geometry (a PR resembling the other PRs that touch a test).
  That is the relational thesis in its purest form on a task text-cosine *cannot* do
  (embedder-cosine = 0.009).
- **Scope:** pilot scale, single cross-repo split, one (α, hops) point; the dense graph
  is a one-time live snapshot (committed for offline replay). Exploratory.

This is the wave that makes the thesis a *method* (two tasks) rather than a single
result. All numbers are exploratory; the committed dense graph + results JSON are the
source of truth.
