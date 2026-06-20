# Relational SDLC AI Lab

Public research lab for relational-geometric learning over software development
lifecycle records.

The project explores a simple thesis:

> Software engineering agents should learn and operate over verifiable
> relations among records, not only over semantically similar text chunks.

The first target is not a full autonomous coding agent. The first target is a
measurable relational retrieval layer for SDLC tasks:

- issue -> likely fixing PRs;
- diff -> affected tests;
- failing log -> likely files/functions;
- PR -> risk and missing-test candidates;
- task state -> tools or agents that should run next.

Contribution and source-record rules are in [CONTRIBUTING.md](CONTRIBUTING.md).
The public operating boundary is in [docs/operating-boundary.md](docs/operating-boundary.md).
The research lifecycle is in [docs/research-lifecycle.md](docs/research-lifecycle.md).

This public repository is intentionally limited to research notes, source
rules, dataset documentation, model designs, evaluation designs, and
reproducible experiment results. Local planning files, run state, and caches
should stay outside this repository.

## Initial North Stars

- **G1: Public research hygiene.** Public sources and reproducible records.
- **G2: Falsifiable retrieval gains.** Every claim needs a benchmark.
- **G3: Dataset provenance.** Every record has source, license, timestamp,
  and transformation lineage.
- **G4: Relation-aware models.** Embeddings and rerankers should learn explicit
  software lifecycle relations, not only text similarity.
- **G5: Agentic readiness.** Every workflow should eventually expose state,
  preconditions, actions, tools, and outcomes.

## First 90 Days

1. Build a 20-repo public dataset with issues, PRs, commits, diffs, tests, and
   relation edges.
2. Establish vanilla embedding baselines for issue/file/test retrieval.
3. Train a relation-aware embedding model with contrastive and relation-head
   objectives.
4. Evaluate Recall@K, MRR, hard-negative accuracy, and test-selection quality.
5. Publish dataset cards, model cards, experiment cards, and reproducibility
   notes.
6. Keep local lifecycle state outside public history.

## Public Repo Contents

```text
docs/   architecture, roadmap, and research notes
data/   dataset cards, small public fixtures, and redistribution notes
```
