# External validity: diff→test replicates on a second (TS/JS) corpus (R24)

**Status: exploratory, real public data (one-time live snapshot).** Every result in
this program was on **one** corpus — a Python-ecosystem pilot. The reviewer panel named
this the single biggest weakness: "regime characterization on *one* corpus (external
validity untested)." This wave builds an **independent second corpus in a different
language ecosystem** (TypeScript/JavaScript) and re-runs the diff→affected-test result
under the identical release-honest protocol. It is a genuine refutation test.

## Corpus

12 active TS/JS repositories — `vuejs/core`, `vitejs/vite`, `sveltejs/svelte`,
`withastro/astro`, `expressjs/express`, `prettier/prettier`, `axios/axios`,
`colinhacks/zod`, `date-fns/date-fns`, `trpc/trpc`, `TanStack/query`, `honojs/hono` —
deliberately disjoint from the Python pilot. Ingested with the same mappers
([build_corpus2.py](../data/corpus2/build_corpus2.py), schema-/provenance-clean):
**4,962 `modifies` edges, 1,272 PR nodes, 2,898 test nodes.** `_is_test_path` correctly
captures TS/JS conventions (`.spec.ts`, `.test.ts`, `__tests__/`). Evaluation is on the
6 densified repos (the others ingested at shallow depth only).

## Two findings

### 1. The co-change *density-response* is not pilot-specific

At shallow ingest (~200 merged PRs/repo) corpus2 is **sparse** — median 1 modifier per
test node — exactly the structure the pilot had before densification (R16E). Densifying
(more PRs) raises co-change coverage, just as in the pilot. So the "sparse-at-shallow-
ingest, dense-with-history" structure that R17b identified is a property of real SDLC
graphs, not of one corpus.

### 2. The diff→test result replicates

Same release-honest method as the pilot (pure-PR-embedding query + temporal `as_of`
cut, gold edge removed), MiniLM PR embeddings, 905 queries across 6 repos, mean
candidate pool 19:

| Metric | Pilot (Python) | **corpus2 (TS/JS)** |
|---|---|---|
| text-free baseline (embedder-cosine) R@1 | 0.009 | ~0.134 (≈ random for the pool) |
| **graph-aug + `as_of` R@1** | **0.429** | **0.351** |
| graph-aug + `as_of` MRR | 0.574 | 0.455 |
| **fair R@1 (among covered) ÷ random** | **5.7×** | **5.5×** |
| coverage parity (gold vs negative) | 85.7% / 81.9% | 54.1% / 57.8% |

**Structure beats text again, and the leakage-robust signal is near-identical**: the
fair R@1 (ranking only among rankable candidates) is **5.5× the random-among-covered
baseline**, against the pilot's 5.7× — the cleanest cross-corpus comparison, since it is
invariant to pool size and reachability. Candidate coverage is balanced (gold 54.1% vs
negative 57.8%), so the corpus2 result is not a coverage artefact either.

## Honest read

- **The headline generalizes across ecosystems.** Co-change geometry carries the
  diff→test signal in TS/JS as in Python, with the same ~5.5–5.7× discrimination over
  chance and balanced coverage. This converts "one corpus" into a two-ecosystem
  replication — the single most important external-validity gain.
- **Absolute numbers differ, and we say why.** corpus2 R@1 (0.351) is below the pilot's
  (0.429): corpus2 was densified less (lower as_of reachability, 54% vs 86%), its
  candidate pool is larger (19 vs ~13) and uses *random* same-repo negatives rather than
  curated hard negatives, and the text-free baseline is higher (a larger pool with
  sometimes-multiple relevant tests lifts chance to ~0.13). The pool-/reachability-
  invariant ratio is what replicates cleanly.
- **Sparsity caveat carries over.** Median modifiers/test is 1 even after densification
  (a skewed distribution: many rarely-touched tests, frequently-modified gold tests);
  deeper ingest would raise reachability and R@1, as in the pilot.
- **Scope.** Still a one-time live snapshot per corpus, structural (no LoRA), and the
  issue→PR task was not replicated here (diff→test is the differentiated result). Two
  corpora is a replication, not a population.

This is the refutation test the regime characterization needed, and it passed. The
committed corpus2 graph + results JSON
([corpus2-diff2test-results.json](../data/corpus2/corpus2-diff2test-results.json)) are
the source of truth.
