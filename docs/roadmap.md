# Roadmap

## P0: Public Foundation

Build the public project foundation: README, contribution rules, source
hygiene, architecture notes, roadmap, and research outline.

## P1: Curated Public GitHub Dataset

Build a small dataset before a large scrape.

Target:

- 20 public repos;
- 5k-20k examples;
- issue/PR/commit/diff/test relations;
- frozen train/dev/test splits;
- source records and dataset card.

## P2: Relational Embedding Benchmark

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
