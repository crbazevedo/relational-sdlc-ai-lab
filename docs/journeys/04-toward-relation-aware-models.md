# Journey 04 — Toward relation-aware models

The first three journeys run entirely on what exists today: a validated dataset,
a vanilla baseline embedder, and a benchmark. This journey is the **conceptual
recipe** for the next step — going from that vanilla floor to a relation-aware
model, and proving any gain with an honest ablation.

> **Status: planned / upcoming.** The relation-aware model and its ablation
> harness are the in-progress next wave (roadmap [P2](../roadmap.md#p2-relational-embedding-benchmark)).
> This journey describes the recipe and the contract it must satisfy. It does not
> link to model code, because that code is not on `main` yet — when it lands, it
> will plug into the same `relsdlc bench` harness and the same experiment-card
> discipline used in [Journey 03](03-run-the-benchmark.md).

## Why a relation-aware model

The benchmark already shows where the vanilla floor cracks. On the fixture,
`log_to_likely_file` scores `R@1 = 0.500`: text similarity finds both implicated
files within the top 5 but does not rank both first. The signal it ignores is the
**relational structure** — the failing log is connected to the fix through typed
edges:

```text
ci_log:512-fail --caused_by--> diff:512 --modifies--> {symbol:normalize_range, symbol:parse_tz}
```

A model that can follow `caused_by ∘ modifies` should surface both files at the
top, where pure text similarity does not. That is the thesis of the lab: learn
and operate over *verifiable relations*, not only over similar text.

## The conceptual recipe

Per the [architecture notes](../architecture.md), the relational embedding layer
earns trust first. The progression:

1. **Vanilla baseline (today).** `baseline-hashing-tfidf` — hashed bag-of-words
   + cosine ([`src/relsdlc/baseline.py`](../../src/relsdlc/baseline.py)). It is
   deterministic, network-free, and runs in CI. This is the floor, and it is
   already wired into `relsdlc bench`.

2. **Off-the-shelf embedding baseline (planned).** A general text/code embedding
   model, added as an *optional* dependency so the core gates stay
   dependency-light. This is the "is the gain just better embeddings?" control.

3. **Relation-aware bi-encoder (planned).** An encoder `E(record) -> h ∈ R^d`
   trained so that records linked by a positive relation embed near each other.
   Per the architecture, training combines:
   - supervised contrastive loss over related/unrelated record pairs;
   - relation classification — predicting the relation type between two records;
   - graph smoothness for positive relations;
   - **hard negatives** drawn from the same package / suite / author;
   - **temporal splits** so the model never sees the future.

4. **Relation-head reranker (planned).** A scoring head
   `score_r(u, v) = f_r(h_u, h_v)` that reranks candidates by a specific relation
   (e.g. score `fixes(issue, pr)` or `modifies(log, file)`), turning the embedding
   geometry into a relation-conditioned ranking.

Each rung plugs into the existing harness the same way the baseline does: it
takes a query text + a candidate pool and returns a ranked list. The benchmark
([`src/relsdlc/bench.py`](../../src/relsdlc/bench.py)) is written around a
pluggable ranker for exactly this reason, so a new model is scored by the same
Recall@K / MRR / hard-negative metrics and the same leakage guard.

## The ablation: baseline vs relation-aware, same frozen split

A gain only counts if it is measured fairly. The ablation contract:

- **Same frozen split.** Both systems run on the identical evaluation set, frozen
  and recorded in the dataset card. For real data the split is
  temporal-by-commit-date (train on the past, evaluate on the future).
- **Same candidate pools.** Each query already carries an explicit `candidates`
  list, so both systems rank the *same* fixed set.
- **Same hard negatives.** The near-miss negatives (wrong file in the same
  package, wrong test in the same suite, plausible-but-unrelated PR) are part of
  the query. **Hard-negative accuracy is the headline ablation metric** —
  generic similarity finds easy positives, so the relation-aware model has to
  prove itself on the near-misses.
- **Same leakage guard.** Every run honors each query's `as_of` time; a run that
  trips the guard exits non-zero and the result is void.
- **Explicit comparison in the card.** Each run is an experiment card; the
  relation-aware card sets `baseline_comparison` to the baseline card's `id` so
  the delta is a recorded, replayable claim — not a screenshot.

Concretely, the ablation answers: *on the frozen split, does the relation-aware
model raise Recall@K / MRR / hard-negative accuracy over the vanilla baseline,
especially on the tasks where relation structure matters (`log_to_likely_file`,
`diff_to_affected_test`)?* If it does not beat the baseline on hard negatives,
there is no relational gain to claim — and the lab records that honestly rather
than burying it.

## The discipline this must satisfy

When the relation-aware model lands, it inherits the same boring gates:

- **No claim without a baseline** on the same frozen split
  ([research lifecycle §3](../research-lifecycle.md#3-baselines)).
- **No claim without an experiment card** with metrics, leakage checks, and known
  limitations; exploratory runs stay labeled `exploratory`.
- **No release without** a dataset card, a model card, an experiment card,
  rerun instructions, and public source citations
  ([research lifecycle §6](../research-lifecycle.md#6-release)).
- **The gates stay green:** `relsdlc validate data`, `relsdlc bench`, and
  `pytest -q` remain fast, deterministic, and reproducible from a clean checkout.

The relation-aware model is the interesting part. It becomes trustworthy only
because the floor it must beat is measured the same way every time.
