# Baselines, a refuted headline, and a power analysis (R25)

**Status: exploratory, committed-cache experiments (numpy-only, deterministic).** This
wave ran the baselines the paper had marked `[PENDING]`. One of them refuted a headline.
We report it in full because catching it is the benchmark working as intended.

Scripts: [run_baselines_metrics.py](../data/pilot/run_baselines_metrics.py) (pilot, both
tasks), [run_corpus2_baselines.py](../data/corpus2/run_corpus2_baselines.py) (corpus2
lexical-vs-structure-vs-fusion), [run_fusion_rerank.py](../data/corpus2/run_fusion_rerank.py)
(learned reranker, leave-one-repo-out CV). All reuse the de-referenced cross-repo split and
reproduce the committed anchors before reporting anything new.

## 1. Task A — issue→fixing-PR: a real ladder over a real baseline

Full metric suite on the 174 held-out cross-repo queries (pool 13), with a hand-rolled
Okapi BM25 as the lexical baseline:

| System | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| BM25 | 0.500 | 0.874 | 0.971 | 0.658 | 0.732 |
| embedder-cosine (frozen MiniLM) | 0.592 | 0.919 | 0.989 | 0.728 | 0.791 |
| bi-encoder LoRA (the "Occam" baseline) | 0.655 | 0.994 | 0.994 | 0.791 | 0.842 |
| + typed graph aggregation | 0.690 | 0.994 | 0.994 | 0.817 | 0.862 |

The text-embedding ladder genuinely beats BM25, and relational supervision (LoRA) +
co-change aggregation add on top. This result stands.

### 1a. …but the LoRA win is underpowered (MDE / power analysis)

Paired hit@1 over the 174 queries, frozen → LoRA: **ΔR@1 +0.063** (18 gains, 7 losses;
paired sd 0.375). The minimum detectable effect at 80% power is **0.080**, and the
**achieved power for the observed effect is only 0.60**. The 95% CI excludes zero
(matching R17a's [+0.006, +0.121]), so the effect is *positive*, but the study could not
reliably detect effects below ≈0.08, and a replication has only ~60% odds of re-detecting
this one. The honest verb is "modest and underpowered," not "establishes."

R@1 95% bootstrap CIs (B=10⁴): bi-encoder LoRA query-level [0.581, 0.724], **repo-cluster
[0.556, 0.748]**; BM25 [0.425, 0.575].

## 2. Task B — diff→affected-test: the headline does not survive a path baseline

The paper claimed test-file candidates are "text-free," so only co-change structure can
rank them, and that a strong text baseline "floors near chance." **Both halves are wrong.**
A test-file node *does* carry text — its **path** (`tests/test_root_model.py`) — and a
PR's prose shares vocabulary with the path of the test it touches. A BM25 over path tokens
exploits exactly this:

| System | Pilot R@1 (112 q) | corpus2 R@1 (905 q, same-repo hard negs) |
|---|---|---|
| **BM25 over test paths** | **0.536** | **0.609 – 0.685** |
| path/identifier overlap | 0.464 | — |
| co-change structure | 0.009 (sparse) / 0.429 (dense+as_of) | 0.348 |
| sentence-embedder cosine | 0.009 | 0.134 |

BM25-over-paths **beats co-change structure on both corpora**, including against the harder
same-repo negatives of corpus2. The `embedder-cosine 0.009` we had been calling "the
baseline" was never the right one — it is near-zero only because a *sentence embedder*
discards path tokens, not because the task lacks text signal.

**Not leakage.** The PR `content` is title + body prose; it does not contain the changed-file
list (verified: no `tests/`/`.py` tokens in pilot PR text). The signal is the legitimate
domain-vocabulary overlap a deployed system would also have. (TS/JS conventional-commit
titles like `test(Devtools): …` make corpus2 even more lexically solvable.)

## 3. Can a learned reranker salvage structure as complementary? No.

The decisive control: two learned logistic-regression rankers under the *same*
leave-one-repo-out cross-repo protocol on corpus2 (same-repo negatives), differing only in
whether structure features are present.

| System | R@1 |
|---|---|
| structure alone | 0.348 |
| BM25 alone | 0.685 |
| LR(lexical) = bm25 + path-overlap | **0.707** |
| LR(lexical + structure) | 0.670 |

Adding structure to the learned lexical ranker **significantly hurts**: ΔR@1 **−0.038, 95%
CI [−0.066, −0.009]** (excludes zero). Naive equal-weight fusion also hurt (0.497 < 0.609).
The structure features do not transfer across repositories; the model is better off without
them. The single honest residual is descriptive: on the 285 queries where BM25's top-1 is
wrong, structure independently recovers ~⅓ (0.31–0.33) — but this is **not exploitable**
into a better system, because gating on structure degrades the majority of queries BM25
already gets right.

**Conclusion.** The best diff→test system is a learned lexical reranker over path tokens
(R@1 0.707). Co-change structure is neither superior to, nor complementary with, lexical
retrieval here. The "co-change geometry beats text" thesis does not hold for this task.

## 4. A static path-proximity heuristic also beats structure (R27)

The sharpest SE-reviewer objection is that BM25-over-prose is not the field's baseline for
test selection. The field's baseline is *static change-based selection* — pick the tests
nearest the changed files. Ekstazi/STARTS are JVM dynamic RTS and do not run on our
Python/TS corpora, so the language-agnostic, deployable analogue is **changed-source-path →
test-path proximity**: for each query PR, take the non-test files it modified (the diff,
available at query time, no history) and rank candidate tests by core-token overlap with
those source paths (e.g. `pydantic/_internal/_generics.py` → `tests/test_generics.py`).
Scored on the same 112-query pilot harness ([run_path_proximity.py](../data/pilot/run_path_proximity.py)):

| System | R@1 | R@5 | MRR |
|---|---|---|---|
| sentence embedder-cosine | 0.009 | — | — |
| co-change structure (graph-agg + `as_of`) | 0.429 | 0.759 | 0.574 |
| BM25 over paths | ~0.52–0.54 | 0.79 | 0.65 |
| **path-proximity (static, no history)** | **0.580** | 0.750 | 0.681 |
| **best static (path-proximity + BM25)** | **0.679** | 0.893 | 0.787 |

A history-free static heuristic (0.580) beats co-change structure (0.429), and the best
deployable static system — path-proximity combined with PR-prose BM25 — reaches 0.679. The
heuristic is honest, not circular: it uses only the diff a selector has at query time, and it
genuinely fails when the affected test is not named after the changed file (e.g. a change to
`json_schema.py` whose gold test is `test_root_model.py`). So the negative result holds
against the field's own baseline class, not only against text similarity.

## 5. What this means

- **Task A is the positive result** (relational LoRA + aggregation beat BM25 and the frozen
  embedder), tempered by an honest power caveat.
- **Task B is a negative result / cautionary tale**: paths are text, and both a path-lexical
  ranker and a static change-proximity heuristic beat co-change structure on SE test retrieval.
- The program's durable contribution is the benchmark and protocol — de-referencing,
  cross-repo splits, leakage + coverage audits, cross-corpus replication, and the baseline
  rigor that exposed its own headline.

The actionable takeaway for practitioners: on diff→test, try static path-proximity to the
changed files (and PR-prose BM25) before reaching for graph or embedding methods. Source of
truth: the result JSONs committed alongside these scripts.
