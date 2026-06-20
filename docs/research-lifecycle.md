# Research Lifecycle

The project follows a public-data ML lifecycle. The goal is to make every claim
replayable and every dataset/model output reviewable.

## 1. Research Intake

Before collecting data or training a model, record:

- hypothesis;
- target task;
- allowed source classes;
- excluded source classes;
- baseline;
- target metrics;
- expected public outputs;
- known risks.

## 2. Dataset Curation

For each source, record:

- source URL or API query;
- retrieval timestamp;
- license or terms note;
- transform command;
- content hash;
- split assignment;
- redistribution status.

Keep raw large data outside Git. Commit only source cards, dataset cards, small
fixtures, and reproducible scripts.

## 3. Baselines

Every modeling claim needs at least one baseline:

- off-the-shelf embedding or model;
- frozen train/dev/test split;
- hard-negative or adversarial slice;
- metric and direction of improvement.

## 4. Training

Every training run needs:

- data version;
- code version;
- seed policy;
- command;
- hardware or runtime class;
- config;
- output location.

## 5. Evaluation

Evaluation reports should include:

- Recall@K and MRR for retrieval tasks;
- hard-negative accuracy;
- calibration where relation scores are exposed;
- error slices;
- leakage checks;
- comparison against the baseline.

## 6. Release

A result is release-ready only when it has:

- dataset card;
- model card when applicable;
- experiment card;
- known limitations;
- rerun instructions;
- public source citations.

Exploratory work is allowed, but it must be labeled exploratory and should not
be presented as a release-quality result.
