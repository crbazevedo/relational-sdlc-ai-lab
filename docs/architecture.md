# Architecture

The system has two coupled layers.

## 1. Relational Embedding Layer

Records:

- issues;
- PRs;
- commits;
- diffs;
- files;
- symbols;
- tests;
- CI logs;
- tools;
- agent run logs.

Relations:

- `fixes`;
- `modifies`;
- `covers`;
- `fails_on`;
- `caused_by`;
- `requires_tool`;
- `safe_to_parallelize`;
- `suitable_agent`.

Model:

```text
E(record) -> h in R^d
score_r(u, v) = f_r(h_u, h_v)
```

Training:

- supervised contrastive loss;
- relation classification;
- graph smoothness for positive relations;
- hard negatives;
- temporal splits.

## 2. Relational SLM Layer

The SLM consumes relation-packed context:

```text
task -> relation-conditioned retrieval -> subgraph -> prompt/context -> response/action
```

Initial SLM use is dry-run only:

- plan generation;
- test selection explanation;
- risk summary;
- tool recommendation.

Autonomous editing comes later, after retrieval and policy heads are measurable.

## Operating Rule

The embedding layer earns trust first. The SLM acts only after retrieval,
provenance, and evaluation gates are boring.
