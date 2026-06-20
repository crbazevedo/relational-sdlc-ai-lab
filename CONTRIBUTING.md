# Contributing

This repository is public research infrastructure. Contributions should be
reproducible, source-aware, and easy to audit.

## Source Rules

- Use public repositories, public papers, public documentation, synthetic data,
  or original code written for this project.
- Record source URL, retrieval timestamp, license notes, transform command, and
  content hash for dataset records.
- Do not commit credentials, tokens, local machine paths, or large generated
  datasets.
- Keep large data in object storage and commit only dataset cards, source notes,
  small fixtures, or reproducible scripts.
- Keep local planning files, dispatch state, run logs, caches, checkpoints, and
  vector stores out of public history unless explicitly released.

## Research Rules

- Prefer frozen benchmarks over moving targets.
- Report seeds, commands, metrics, data version, and model version.
- Add a regression fixture when a workflow bug is fixed.
- Label exploratory results clearly; do not present them as release-quality
  evidence.

## Review Checklist

Before merging, confirm:

1. Validation passes.
2. Dataset records include provenance.
3. Claims cite a benchmark, experiment card, or public source.
4. Generated files and caches are excluded.
5. Local planning and run state are excluded.
