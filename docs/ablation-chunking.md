# Chunked retrieval: does MaxP beat FirstP? (for issue→PR, no)

**Status: exploratory, real public data.** R14 showed mean-pooling a long body
dilutes the signal. The proposed fix: chunk the body, keep per-chunk vectors, and
score a document by its **best** chunk (MaxP) rather than the average. This tests
that, with a chunk-size ablation, against the right baseline — **FirstP**.

## Grounding (we did not reinvent)

- **FirstP / MaxP / SumP** — Dai & Callan (2019, *Deeper Text Understanding for IR
  with BERT*); Yilmaz et al. (2019, *Birch*). Score a long doc by its first / best /
  summed passage. MaxP is the operator's idea, with named baselines.
- **ColBERT / ColBERTv2** (Khattab & Zaharia, 2020/2022) — late interaction, the
  **MaxSim** operator. This is the chunk-level (frozen, untrained) analogue.
- **PARADE** (Li et al., 2020) — passage representation aggregation.
- **DPR** (Karpukhin et al., 2020) — passage-level dense retrieval.

## Method

Reuses `data/full` (the de-truncated bodies) — no re-ingest. Each record's scrubbed
title+body is split into overlapping char windows at sizes **256 / 512 / 1024**
(20% overlap), each window embedded with **frozen MiniLM**. The query is the issue's
**first chunk** (R14: the issue lede carries the signal); the **document (PR)
aggregation** is what varies. [`src/relsdlc/chunking.py`](../src/relsdlc/chunking.py),
[`data/full/embed_chunks.py`](../data/full/embed_chunks.py),
[`data/full/run_chunk_ablation.py`](../data/full/run_chunk_ablation.py). (Chunk caches
are regenerable and not committed for size; results are in `chunk-results.json`.)

## Result (271 held-out test queries, cross-repo)

| chunk size | whole-doc-mean | **FirstP** | SumP | meanP | MaxP |
|---|---|---|---|---|---|
| 256 | 0.649 | **0.686** | 0.048 | 0.683 | 0.653 |
| 512 | 0.590 | **0.701** | 0.052 | 0.627 | 0.668 |
| 1024 | 0.616 | **0.653** | 0.052 | 0.649 | 0.624 |

(R@1; FirstP R@5 ≈ 0.97–0.99, MRR ≈ 0.81.) Cards:
[`data/cards/examples/gh-chunk-*`](../data/cards/examples/).

## Findings

1. **FirstP wins at every chunk size; MaxP does not beat it** (0.701 vs 0.668 at the
   best size). For issue→fixing-PR the discriminative signal is front-loaded in the
   title + first paragraph, so "best chunk" recovers nothing beyond "first chunk."
2. **MaxP is slightly *worse* than FirstP** — taking the max over many chunks lets a
   same-repo hard-negative PR win on a single coincidentally-similar chunk (a known
   MaxP failure mode without trained late interaction).
3. **SumP collapses** (R@1 ≈ 0.05) — it is length-biased, rewarding PRs with more
   chunks regardless of relevance. Exactly the caveat the literature gives.
4. **Best frozen number to date: FirstP @ 512 = R@1 0.701** — i.e. "embed the title
   + first ~paragraph." The lede *is* the document for this task.

## Why this was still worth running (and what it motivates)

The result confirms the two predicted refinements:
- **Task-dependence:** chunking/MaxP should pay off where the relevant content is
  *deep or distributed* — long multi-hunk diffs, logs, `diff→affected-test`,
  `log→file` — not the front-loaded issue→PR. Testing that needs **file/log content
  ingest** (we currently store file *paths*, not contents) — the natural next data step.
- **Trained late interaction:** this probe used *frozen* embeddings + untrained MaxP.
  A ColBERT-style *trained* MaxSim (the model learns chunk-level matching) is the
  proper version if chunking is pursued — but it should be motivated by a task where
  even frozen FirstP leaves signal on the table, which issue→PR does not.

**Verdict:** for issue→fixing-PR, FirstP (the lede) is the ceiling among frozen
chunk aggregations; MaxP does not help and SumP hurts. Chunking is queued for the
deep-signal tasks, where the literature says it should matter.
