# Operating Boundary

This repository is the public research surface. It should contain only material
that can be shared, cited, rerun, and reviewed without relying on local
coordination state.

## Commit Here

- research notes;
- architecture notes;
- source cards;
- dataset cards;
- model cards;
- experiment cards;
- benchmark definitions;
- small public fixtures;
- reproducible scripts needed to rebuild a result.

## Keep Outside This Repo

- local planning files;
- dispatch state;
- run logs;
- credentials and tokens;
- machine-local paths;
- large raw datasets;
- model checkpoints unless explicitly released;
- vector stores and caches;
- restricted source material.

## Adoption Rule

The project may be operated with external lifecycle tooling once that tooling is
ready for use. The tooling state stays outside this repository. Only public
research outputs and reproducibility records belong here.

## Before Publishing

Check every change for:

1. source records for data;
2. frozen benchmark or explicit exploratory label;
3. no restricted source material;
4. no local planning or run state;
5. no credentials, tokens, or local paths;
6. claims tied to a metric, experiment card, or public source.
