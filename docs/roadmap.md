# Roadmap

## P0: Public Foundation

Build the public project foundation: README, contribution rules, source
hygiene, architecture notes, roadmap, and research outline.

**Status: substrate landed (Wave R1).** The foundation is now executable, not
just documented:

- JSON schemas for records, relation edges, benchmark queries, and source /
  dataset / experiment cards (`schemas/`);
- the `relsdlc` library + CLI — schema validation, provenance and
  referential-integrity gates, a temporal-leakage guard, Recall@K / MRR /
  hard-negative metrics, and a dependency-light baseline embedder (`src/`);
- a synthetic, reproducible `datebox` fixture + example cards (`data/`);
- the benchmark definition (`docs/benchmark-definition.md`);
- CI that keeps the gates boring: reproducible-fixture check, `relsdlc validate`,
  benchmark smoke, and tests (`.github/workflows/ci.yml`).

## P1: Curated Public GitHub Dataset

Build a small dataset before a large scrape.

**Status: pilot landed.** A one-time live snapshot of 20 permissive
Python-ecosystem repos — 2,087 records, 356 `fixes` edges, 356 issue→PR queries,
frozen temporal split, per-repo source cards + dataset card, redistribution
metadata_only. Built by [`data/pilot/build_pilot.py`](../data/pilot/build_pilot.py).

Target:

- 20 public repos;
- 5k-20k examples;
- issue/PR/commit/diff/test relations;
- frozen train/dev/test splits;
- source records and dataset card.

## P2: Relational Embedding Benchmark

**Status: mechanism demonstrated on synthetic data; real-data baselines + cross-repo
analysis established.**
[ablation.md](ablation.md): on a controlled benchmark, relation-supervised token
weighting beats unsupervised IDF beats vanilla cosine (Recall@1 0.82 vs 0.38 vs
0.11). [ablation-real.md](ablation-real.md): on the real P1 pilot the win does
**not** transfer — IDF is the strong baseline (R@1 0.44) and the diagonal
relation-metric ties it. [ablation-crossrepo.md](ablation-crossrepo.md): with
references removed and repos held out, a two-tower projection wins on cross-token
synthetic (0.83) but loses to vanilla cross-repo (0.24) — bag-of-tokens can't
transfer across repos. Evidence-backed next step: a relation operator on top of
**pretrained code/text embeddings** (the real P3).

Compare:

- vanilla text/code embedding;
- relation-aware bi-encoder;
- relation-head reranker;
- graph reranker.

Metrics:

- Recall@K;
- MRR;
- hard-negative accuracy;
- test-selection precision/recall;
- calibration of relation scores.

## P3: GraphRAG and Dry-Run Agent Policy

Package retrieved subgraphs for SLM consumption and evaluate dry-run tool/test
selection before agent execution.

## P4: Public Release

Publish:

- position paper;
- dataset card;
- model card;
- experiment cards;
- source-review release checklist.
