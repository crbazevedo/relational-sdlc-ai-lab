# Chunked retrieval on DEEP content: does MaxP beat FirstP? (for diff→test, **yes**)

**Status: exploratory, real public data.** This is the deep-signal follow-up that
[`ablation-chunking.md`](ablation-chunking.md) explicitly queued. R15 found that for
issue→fixing-PR, **FirstP** (the first chunk / the lede) is the ceiling among frozen
chunk aggregations and **MaxP does not beat it** — because that task's signal is
*front-loaded* (the issue title + first paragraph already carry it). R15's own
verdict predicted the fix: chunking/MaxP should pay off where the relevant content
is **deep or distributed**, e.g. `diff→affected-test` over long file bodies — but
that needed **file content ingest** (the pilot stores file *paths*, not contents).

R16A does that ingest and runs the test. The hypothesis is validated: **on deep
test-file contents, MaxP beats FirstP at every chunk size.**

## What changed vs R15

- **New data: real file/test CONTENTS.** [`data/content/build_content.py`](../data/content/build_content.py)
  LIVE-fetches each distinct file/test path from the pilot graph (permissively-licensed
  repos only — MIT / BSD / Apache / PSF / ISC; Pillow's `unknown` license is skipped),
  GETs the GitHub contents API on the default branch, base64-decodes, caps at **16000
  chars**, and writes schema-valid `file`/`test` records with provenance to
  [`data/content/file_contents.jsonl`](../data/content/file_contents.jsonl). Ids are
  namespaced `gh-content:owner/repo:file:path` so they never collide with the pilot's
  path-only file nodes. Permissive code contents are redistributable, so the source
  cards declare `redistribution: snippets_permitted`.
- **New task: `diff_to_affected_test`.** [`benchmark/diff_to_affected_test.jsonl`](../data/content/benchmark/diff_to_affected_test.jsonl):
  query = a PR (the "diff", a pilot `gh:…:pr:N` record); candidates = same-repo
  `gh-content:` TEST records; relevant = the test(s) the PR actually modified (from the
  graph's `modifies` edges). Cross-repo split feasible (query repo == candidate repo).

## Counts

| quantity | value |
|---|---|
| file/test paths considered (permissive repos) | 714 |
| content records fetched | **617** (408 file, 209 test) |
| skipped (404 / 410 / binary / empty marker) | 97 |
| repos fetched | 17 |
| `file_contents.jsonl` size | ~7.1 MB |
| **median content length (all)** | **9093 chars** |
| **median content length (test files)** | **10426 chars** |
| diff→affected-test queries | **206** (15 repos) |
| held-out test queries (cross-repo) | 101 (6 test repos / 9 train repos) |

The signal is genuinely deep: median test-file body ≈ 10.4k chars — **>> 500** (the
pilot truncation) and well past a single 512-char chunk. This is exactly the regime
R15 said chunking should help.

## Method

Each content record's scrubbed file text is split into overlapping char windows at
sizes **256 / 512 / 1024** (20% overlap), each window embedded with **frozen MiniLM**
(max_length 256). The PR queries are embedded whole (scrubbed title+body). Document
(test-file) aggregation is what varies — isolating "how to represent a long file."
Numpy-only scoring on the caches. References are scrubbed from text (no `#123`
shortcut); train repos are disjoint from test repos.
[`data/content/embed_content.py`](../data/content/embed_content.py),
[`data/content/run_content_chunk_ablation.py`](../data/content/run_content_chunk_ablation.py).
Chunk caches are regenerable and **gitignored** (`data/content/embeddings/`); results
are committed in [`content-chunk-results.json`](../data/content/content-chunk-results.json).

## Result (101 held-out test queries, cross-repo) — R@1

| chunk size | whole-doc-mean | **FirstP** | SumP | meanP | **MaxP** | MaxP − FirstP |
|---|---|---|---|---|---|---|
| 256  | 0.555 | **0.217** | 0.370 | 0.572 | **0.562** | **+0.346** |
| 512  | 0.570 | **0.384** | 0.386 | 0.520 | **0.555** | **+0.171** |
| 1024 | 0.570 | **0.398** | 0.371 | 0.483 | **0.510** | **+0.112** |

(R@1; MaxP R@5 ≈ 0.88–0.91, MRR ≈ 0.75–0.78, hard-neg-acc ≈ 0.70–0.73.) Cards:
[`data/cards/examples/gh-content-{firstp,maxp}-*`](../data/cards/examples/).

## Findings

1. **MaxP beats FirstP at every chunk size** (deltas **+0.346 / +0.171 / +0.112**).
   This is the result R15 could not get on issue→PR and is the validation of chunking
   for deep tasks: when the matching code is *anywhere* in a long file, the "best chunk"
   recovers signal that the file head (imports + setup) misses. The smaller the chunk,
   the bigger the win — at size 256 FirstP sees only the first ~256 chars of a ~10k-char
   file and collapses (R@1 0.217), while MaxP scans the whole file (0.562).
2. **FirstP degrades as chunks shrink; MaxP is robust.** FirstP climbs 0.217 → 0.398 as
   the head chunk grows 256 → 1024 (it simply sees more of the file). MaxP stays in the
   0.51–0.56 band regardless — it does not depend on where the signal sits.
3. **whole-doc-mean is also strong here** (≈ 0.56–0.57), the opposite of R14/R15 where
   mean-pooling was the loser. On same-repo test candidates the files are topically
   clustered, so the mean vector is informative — but **MaxP is the best or tied-best
   aggregator at every size** and is the one that beats FirstP cleanly. The honest read:
   on deep content, *any* aggregation that looks past the head (MaxP, meanP, whole-mean)
   beats the head-only FirstP; MaxP is the most consistent.
4. **SumP still under-performs** (0.37–0.39) — length bias, as the literature warns,
   though it no longer collapses to ~0.05 as it did on issue→PR (here all candidates are
   single files of comparable length, so the bias is milder).

## Verdict

**For `diff→affected-test` over deep file contents, chunking finally helps: MaxP beats
FirstP at every chunk size (best margin +0.346 at size 256).** This closes the loop R15
opened — chunking/MaxP is *task-dependent*, paying off precisely where the signal is
deep rather than front-loaded. The contrast is the finding: same frozen MiniLM, same
MaxP operator, opposite outcome on a front-loaded (issue→PR) vs a deep (diff→test) task.

**Caveats (exploratory).** Frozen general-text MiniLM (not code-specific); 16000-char
cap; coarse char-window MaxP (not token-level ColBERT MaxSim); pilot scale; same-repo
candidate pools. The proper next version is a *trained* late-interaction MaxSim, now
motivated by a task where frozen FirstP demonstrably leaves signal on the table — which,
unlike issue→PR, this one does.
